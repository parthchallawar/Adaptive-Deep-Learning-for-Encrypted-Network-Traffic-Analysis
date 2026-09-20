"""Evaluation metrics (spec 004, plan T6).

Every metric here is a pure function of saved arrays -- never a model -- so a
policy or threshold can be re-scored from `.npy` logits without re-running
anything (spec 004's "cheap threshold sweeps" requirement). The canonical
classification input is ``logits[N, K, C]`` (one row of `C`-way logits per
prefix length `k` in `1..K_MAX`) plus ``labels[N]`` and ``ppi_len[N]``.

Deliberately pure NumPy/pandas, no scikit-learn: this module is used from
phase 1 (spec 004's "Depends on: 001, 003", both phase-1 specs), and
`scikit-learn` is a phase-2 ("train") extra in `pyproject.toml` -- the
project's own dependency split says phase 1 installs without it. AUROC/AUPR
are the standard rank-based formulas, cross-checked in tests against the
well-known small hand-computed cases.

Every K-based function shares one rule for what "evaluating at K" means on a
flow shorter than K (spec 004's edge case): the *effective* K is
``min(k, ppi_len)``, computed once in :func:`_index_at_k` and used
everywhere else, so a flow's own K95/short-flow bucketing can never silently
disagree between two metrics.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

ClassesArg = Sequence[int] | np.ndarray | None
"""Accepted by every classification metric's optional ``classes`` argument:
either the label universe to score against, or ``None`` to infer it from
whatever true labels are present (spec 004: absent classes are excluded)."""

# --- shared K-indexing rule --------------------------------------------------


def _index_at_k(ppi_len: np.ndarray, k: int | np.ndarray, k_max: int) -> np.ndarray:
    """0-based index into ``logits[:, :, :]``'s K axis for nominal prefix
    length ``k`` (scalar or per-flow array), clamped to each flow's own
    ``ppi_len`` and to ``[1, k_max]``."""
    ppi_len = np.asarray(ppi_len)
    k_arr = np.full(ppi_len.shape, k, dtype=np.int64) if np.isscalar(k) else np.asarray(k)
    effective = np.clip(np.minimum(k_arr, ppi_len), 1, k_max)
    return effective - 1


def logits_at_k(logits: np.ndarray, ppi_len: np.ndarray, k: int | np.ndarray) -> np.ndarray:
    """``logits[N, K_MAX, C]`` -> ``[N, C]``, one row per flow at its own
    effective K."""
    idx = _index_at_k(ppi_len, k, logits.shape[1])
    return logits[np.arange(logits.shape[0]), idx, :]


def predictions_at_k(logits: np.ndarray, ppi_len: np.ndarray, k: int | np.ndarray) -> np.ndarray:
    """Predicted class per flow (argmax) at its own effective K."""
    return logits_at_k(logits, ppi_len, k).argmax(axis=-1)


def short_flow_mask(ppi_len: np.ndarray, k: int) -> np.ndarray:
    """True where a flow is shorter than the nominal ``k`` -- spec 004's
    "short-flow" bucket, reported separately rather than folded silently
    into the headline accuracy."""
    return np.asarray(ppi_len) < k


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


# --- classification (known classes) -----------------------------------------


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(preds) == np.asarray(labels)))


def acc_at_k(logits: np.ndarray, labels: np.ndarray, ppi_len: np.ndarray, k: int) -> float:
    return accuracy(predictions_at_k(logits, ppi_len, k), labels)


def short_flow_accuracy(
    logits: np.ndarray, labels: np.ndarray, ppi_len: np.ndarray, k: int
) -> float | None:
    """Accuracy restricted to flows with ``ppi_len < k`` (spec 004's
    short-flow rule). ``None`` when no flow in the batch is that short."""
    mask = short_flow_mask(ppi_len, k)
    if not mask.any():
        return None
    preds = predictions_at_k(logits, ppi_len, k)
    return accuracy(preds[mask], labels[mask])


def _sorted_classes(labels: np.ndarray, classes: ClassesArg) -> np.ndarray:
    if classes is not None:
        return np.asarray(sorted(classes))
    return np.unique(labels)


def _class_index(values: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """``values`` -> index into sorted ``classes``, or -1 if not present."""
    idx = np.searchsorted(classes, values)
    idx = np.clip(idx, 0, len(classes) - 1)
    hit = classes[idx] == values
    return np.where(hit, idx, -1)


def confusion(
    preds: np.ndarray, labels: np.ndarray, classes: ClassesArg = None
) -> np.ndarray:
    """``[len(classes), len(classes)]`` int64 matrix, ``cm[true, pred]``.
    Predictions or labels outside ``classes`` are dropped (spec 004: classes
    absent from a test period are excluded from that period's averages)."""
    classes_arr = _sorted_classes(labels, classes)
    y_idx = _class_index(np.asarray(labels), classes_arr)
    p_idx = _class_index(np.asarray(preds), classes_arr)
    keep = (y_idx >= 0) & (p_idx >= 0)
    cm = np.zeros((len(classes_arr), len(classes_arr)), dtype=np.int64)
    np.add.at(cm, (y_idx[keep], p_idx[keep]), 1)
    return cm


def per_class_f1(
    preds: np.ndarray, labels: np.ndarray, classes: ClassesArg = None
) -> dict[int, float]:
    """F1 per class, for classes with at least one true instance only
    (spec 004: absent classes are excluded from macro averages, not
    scored as zero)."""
    classes_arr = _sorted_classes(labels, classes)
    cm = confusion(preds, labels, classes_arr)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        int(c): float(v) for c, v, s in zip(classes_arr, f1, support, strict=True) if s > 0
    }


def macro_f1(
    preds: np.ndarray, labels: np.ndarray, classes: ClassesArg = None
) -> float:
    scores = per_class_f1(preds, labels, classes)
    return float(np.mean(list(scores.values()))) if scores else 0.0


def balanced_accuracy(
    preds: np.ndarray, labels: np.ndarray, classes: ClassesArg = None
) -> float:
    classes_arr = _sorted_classes(labels, classes)
    cm = confusion(preds, labels, classes_arr)
    support = cm.sum(axis=1)
    recall = np.divide(
        np.diag(cm).astype(np.float64),
        support,
        out=np.zeros(len(classes_arr)),
        where=support > 0,
    )
    present = support > 0
    return float(recall[present].mean()) if present.any() else 0.0


# --- earliness ----------------------------------------------------------------


def auc_k(
    logits: np.ndarray, labels: np.ndarray, ppi_len: np.ndarray, k_max: int | None = None
) -> float:
    """Mean of ``acc@k`` over ``k`` in ``1..k_max`` -- area under the
    accuracy-vs-K curve, normalised by the K range (spec 004)."""
    k_max = k_max or logits.shape[1]
    accs = [acc_at_k(logits, labels, ppi_len, k) for k in range(1, k_max + 1)]
    return float(np.mean(accs))


def k95(
    logits: np.ndarray, labels: np.ndarray, ppi_len: np.ndarray, k_max: int | None = None
) -> int:
    """Smallest K where ``acc@K >= 0.95 * acc@k_max``."""
    k_max = k_max or logits.shape[1]
    accs = np.array([acc_at_k(logits, labels, ppi_len, k) for k in range(1, k_max + 1)])
    target = 0.95 * accs[-1]
    hits = np.flatnonzero(accs >= target)
    return int(hits[0] + 1) if len(hits) else k_max


def harmonic_earliness(accuracy_value: float, mean_k_value: float, k_max: int = 30) -> float:
    """Harmonic mean of accuracy and earliness (``1 - mean_k / k_max``), the
    single-number summary spec 004 borrows from the ECTS literature."""
    earliness = 1.0 - mean_k_value / k_max
    denom = accuracy_value + earliness
    if denom <= 0:
        return 0.0
    return float(2 * accuracy_value * earliness / denom)


# --- policy (adaptive per-flow K) ---------------------------------------------


def mean_k(k: np.ndarray) -> float:
    return float(np.mean(k))


def p95_k(k: np.ndarray) -> float:
    return float(np.percentile(k, 95))


def coverage(committed: np.ndarray) -> float:
    """Fraction of flows the policy committed to before being forced to a
    default decision at ``k_max`` (spec 004)."""
    return float(np.mean(committed))


def committed_accuracy(preds: np.ndarray, labels: np.ndarray, committed: np.ndarray) -> float:
    committed = np.asarray(committed, dtype=bool)
    if not committed.any():
        return 0.0
    return accuracy(preds[committed], labels[committed])


def pareto_front(
    mean_ks: Sequence[float], accuracies: Sequence[float]
) -> list[tuple[float, float]]:
    """The Pareto-optimal ``(mean_k, accuracy)`` points from sweeping a
    policy's threshold: lower ``mean_k`` and higher accuracy both improve a
    point, so a point is kept only if no other point dominates it."""
    points = sorted(zip(mean_ks, accuracies, strict=True))
    front: list[tuple[float, float]] = []
    best_acc = -np.inf
    for mk, acc in points:
        if acc > best_acc:
            front.append((float(mk), float(acc)))
            best_acc = acc
    return front


# --- open-set / anomaly ---------------------------------------------------------


def auroc(scores: np.ndarray, is_positive: np.ndarray) -> float:
    """Rank-based AUROC (equivalent to the Mann-Whitney U statistic), ties
    handled via average ranking. ``is_positive`` marks the "unknown" class as
    positive (spec 004)."""
    scores = np.asarray(scores, dtype=np.float64)
    is_positive = np.asarray(is_positive, dtype=bool)
    n_pos, n_neg = int(is_positive.sum()), int((~is_positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    sum_ranks_pos = ranks[is_positive].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def aupr(scores: np.ndarray, is_positive: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve), the
    standard step-function formula: sum of precision weighted by the
    recall gained at each positive, in descending score order."""
    scores = np.asarray(scores, dtype=np.float64)
    is_positive = np.asarray(is_positive, dtype=bool)
    n_pos = int(is_positive.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    y = is_positive[order]
    tp_cum = np.cumsum(y)
    fp_cum = np.cumsum(~y)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def fpr_at_tpr(scores: np.ndarray, is_positive: np.ndarray, tpr_target: float = 0.95) -> float:
    """False-positive rate at the operating point where true-positive rate
    first reaches ``tpr_target`` (spec 004: "FPR at 95% TPR")."""
    scores = np.asarray(scores, dtype=np.float64)
    is_positive = np.asarray(is_positive, dtype=bool)
    n_pos, n_neg = int(is_positive.sum()), int((~is_positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    y = is_positive[order]
    tpr = np.cumsum(y) / n_pos
    fpr = np.cumsum(~y) / n_neg
    hits = np.flatnonzero(tpr >= tpr_target)
    return float(fpr[hits[0]]) if len(hits) else 1.0


def open_set_f1(
    preds: np.ndarray, labels: np.ndarray, classes: ClassesArg = None
) -> float:
    """Macro-F1 over known classes plus the unknown class (spec 004),
    reusing :func:`macro_f1` directly: ``preds``/``labels`` already carry
    the shard schema's ``-1`` "unknown" value (`tensors.ARRAY_SPEC`) as
    just one more class, so no separate open-set logic is needed."""
    return macro_f1(preds, labels, classes)


def rejection_earliness(
    k: np.ndarray, is_unknown_true: np.ndarray, is_rejected: np.ndarray
) -> float:
    """Mean K at which *correctly* rejected unknown flows were rejected
    (spec 004). ``nan`` if no unknown flow was ever correctly rejected."""
    is_unknown_true = np.asarray(is_unknown_true, dtype=bool)
    is_rejected = np.asarray(is_rejected, dtype=bool)
    mask = is_unknown_true & is_rejected
    if not mask.any():
        return float("nan")
    return mean_k(np.asarray(k)[mask])


# --- calibration ---------------------------------------------------------------


def _bin_index(confidence: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: the population-weighted mean absolute gap
    between each bin's average confidence and its actual accuracy."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    bin_idx = _bin_index(confidence, n_bins)
    n = len(confidence)
    total = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        total += (mask.sum() / n) * abs(confidence[mask].mean() - correct[mask].mean())
    return float(total)


def reliability_curve(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(bin_centers, bin_accuracy, bin_confidence)``, each length
    ``n_bins``; accuracy/confidence are ``nan`` for an empty bin."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    bin_idx = _bin_index(confidence, n_bins)
    bin_acc = np.full(n_bins, np.nan)
    bin_conf = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            bin_acc[b] = correct[mask].mean()
            bin_conf[b] = confidence[mask].mean()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, bin_acc, bin_conf
