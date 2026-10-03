"""Adaptive per-flow-K policy baselines: P-ECHO and P-CAPE (spec 005, plan T7).

Pure functions of a :class:`~adl_etc.evaluation.dense_logits.DenseLogits`, so
a policy or its calibration can be re-run against saved logits with no model
in the loop (spec 004's "cheap threshold sweeps" rule). Both sweep K over
``dense.evaluated_k`` (sorted) rather than assuming ``1..30``, so they work
identically on a causal model's full range or a per-K model's 13-value grid.

- **P-ECHO** (:func:`p_echo`): commit at the first K where the committed
  class's own softmax probability clears one global ``tau``. Sweeping ``tau``
  traces the accuracy-vs-earliness Pareto front (:func:`echo_pareto_front`).
- **P-CAPE** (:func:`p_cape`): CAPE-Net-style. Per-class thresholds
  ``tau_c`` (fit by :func:`fit_cape_thresholds`) replace ECHO's one global
  ``tau`` -- committing to class *c* needs *c*'s own confidence bar, which is
  higher for classes that are easy to mistake something else for. A single
  energy-based gate at a fixed K (spec 005's K=10) separately flags a flow as
  unknown. Spec 005 calls this "Learn-Then-Test approximated by per-class
  validation quantiles" and does not specify the two mechanisms' priority;
  this module's choice (documented on :func:`p_cape`) is that an early,
  already-committed flow is never revisited by the later energy check.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from adl_etc.evaluation.dense_logits import DenseLogits
from adl_etc.evaluation.metrics import committed_accuracy, mean_k, pareto_front, softmax


@dataclass(frozen=True)
class PolicyResult:
    """One adaptive policy's decision per flow: ``k`` is where it committed
    (or the swept range's last K, if it never did); ``committed`` says which.
    A flow that never commits is still given a ``prediction`` (from the last
    K's argmax), matching spec 004's "forced to a default decision at k_max"
    rule, so callers never have to special-case an all-``False`` mask."""

    k: np.ndarray  # [N] int64
    prediction: np.ndarray  # [N] int64 (-1 = declared unknown, P-CAPE only)
    committed: np.ndarray  # [N] bool

    def __post_init__(self) -> None:
        n = len(self.k)
        for name, arr in (("prediction", self.prediction), ("committed", self.committed)):
            if len(arr) != n:
                raise ValueError(f"{name} must have length {n}, got {len(arr)}")


def _sorted_ks(dense: DenseLogits, ks: Sequence[int] | None) -> list[int]:
    return sorted(ks) if ks is not None else sorted(dense.evaluated_k)


# --- P-ECHO -------------------------------------------------------------------------


def p_echo(
    dense: DenseLogits, ppi_len: np.ndarray, tau: float, *, ks: Sequence[int] | None = None
) -> PolicyResult:
    """Commit at the first K (from ``ks``, default every K ``dense`` holds)
    where the argmax class's softmax probability is ``>= tau``.

    ``tau=0`` commits every flow at the first K in the range (softmax's max
    probability is always > 0). ``tau=1`` never commits before the range's
    last K (softmax's max probability is always < 1 for finite logits), so
    every flow falls through to the forced default decision there -- both
    spec 005's own policy tests.
    """
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}")
    ks_sorted = _sorted_ks(dense, ks)
    n = len(dense)
    committed_k = np.full(n, ks_sorted[-1], dtype=np.int64)
    preds = np.zeros(n, dtype=np.int64)
    committed = np.zeros(n, dtype=bool)
    done = np.zeros(n, dtype=bool)

    for k in ks_sorted:
        probs = softmax(dense.at(k, ppi_len))
        conf = probs.max(axis=-1)
        hit = ~done & (conf >= tau)
        committed_k[hit] = k
        preds[hit] = probs[hit].argmax(axis=-1)
        committed[hit] = True
        done |= hit

    if not done.all():
        last_probs = softmax(dense.at(ks_sorted[-1], ppi_len))
        preds[~done] = last_probs[~done].argmax(axis=-1)
    return PolicyResult(k=committed_k, prediction=preds, committed=committed)


def echo_pareto_front(
    dense: DenseLogits,
    labels: np.ndarray,
    ppi_len: np.ndarray,
    taus: Sequence[float],
    *,
    ks: Sequence[int] | None = None,
) -> list[tuple[float, float]]:
    """Sweeps ``taus`` through :func:`p_echo` and returns the Pareto-optimal
    ``(mean_k, committed_accuracy)`` points (spec 005: "tau sweep gives the
    Pareto front"). Accuracy is over committed flows only, matching
    :func:`~adl_etc.evaluation.metrics.committed_accuracy`."""
    mean_ks, accs = [], []
    for tau in sorted(taus):
        result = p_echo(dense, ppi_len, tau, ks=ks)
        mean_ks.append(mean_k(result.k))
        accs.append(committed_accuracy(result.prediction, labels, result.committed))
    return pareto_front(mean_ks, accs)


# --- P-CAPE -------------------------------------------------------------------------


def energy_score(logits: np.ndarray) -> np.ndarray:
    """The free-energy OOD score (Liu et al. 2020, temperature 1):
    ``-logsumexp(logits, axis=-1)``. A flow whose logits are all small and
    close together (no class stands out -- the open-set signature) has a
    *higher* (less negative) energy than one where a single class dominates,
    so ``energy >= theta`` is the anomaly-gate direction P-CAPE uses."""
    m = logits.max(axis=-1, keepdims=True)
    lse = m[..., 0] + np.log(np.exp(logits - m).sum(axis=-1))
    return -lse


def _threshold_for_precision(conf: np.ndarray, correct: np.ndarray, precision: float) -> float:
    """Smallest confidence threshold ``t`` such that ``mean(correct[conf >= t])
    >= precision`` -- the per-class validation quantile spec 005 calls the
    Learn-Then-Test approximation. ``1.0`` (never commits) when there is no
    evidence, or when no threshold reaches the target precision at all."""
    if len(conf) == 0:
        return 1.0
    order = np.argsort(-conf)
    conf_sorted = conf[order]
    cum_correct = np.cumsum(correct[order].astype(np.float64))
    precision_at = cum_correct / np.arange(1, len(conf) + 1)
    hits = np.flatnonzero(precision_at >= precision)
    if len(hits) == 0:
        return 1.0
    # The largest qualifying prefix is the most flows admitted at the target
    # precision, i.e. the lowest (least restrictive) threshold that qualifies.
    return float(conf_sorted[hits.max()])


def fit_cape_thresholds(
    dense_val: DenseLogits,
    labels_val: np.ndarray,
    ppi_len_val: np.ndarray,
    n_classes: int,
    *,
    k: int | None = None,
    precision: float = 0.95,
    min_support: int = 20,
) -> np.ndarray:
    """Per-class thresholds ``tau_c`` (shape ``[n_classes]``), calibrated once
    from a validation :class:`DenseLogits` at its most-informed K (default
    ``max(dense_val.evaluated_k)``): for each class ``c``, the smallest
    confidence threshold such that flows the model predicts as ``c`` with at
    least that confidence are truly ``c`` at least ``precision`` of the time.

    A class predicted for fewer than ``min_support`` validation flows falls
    back to one global threshold, fit the same way over every val flow
    regardless of predicted class (spec 005, CAPE-Net's own "min support
    20" rule) -- a class this project's random search or a hostile split
    never sees in validation still gets a usable, if conservative, bar.
    """
    k = k if k is not None else max(dense_val.evaluated_k)
    probs = softmax(dense_val.at(k, ppi_len_val))
    preds = probs.argmax(axis=-1)
    conf = probs.max(axis=-1)
    correct = preds == labels_val

    global_tau = _threshold_for_precision(conf, correct, precision)
    tau_c = np.full(n_classes, global_tau, dtype=np.float64)
    for c in range(n_classes):
        mask = preds == c
        if int(mask.sum()) < min_support:
            continue
        tau_c[c] = _threshold_for_precision(conf[mask], correct[mask], precision)
    return tau_c


def p_cape(
    dense: DenseLogits,
    ppi_len: np.ndarray,
    tau_c: np.ndarray,
    theta: float,
    *,
    energy_k: int = 10,
    ks: Sequence[int] | None = None,
) -> PolicyResult:
    """CAPE-Net-style adaptive policy: commit to the predicted class at the
    first K where that class's own ``tau_c`` is cleared, and separately
    reject a flow as unknown (``prediction = -1``) if its energy score at
    ``energy_k`` (spec 005's K=10) is ``>= theta``.

    **Priority, a scope decision spec 005 leaves open:** the energy gate is
    checked only for a flow not already committed by ``energy_k`` -- an
    early, confident class commitment is never revoked by a later anomaly
    signal. ``energy_k`` must be one of ``ks`` (default ``dense.evaluated_k``)
    or the gate can never fire.
    """
    ks_sorted = _sorted_ks(dense, ks)
    if energy_k not in ks_sorted:
        raise ValueError(f"energy_k={energy_k} must be one of the evaluated K's {ks_sorted}")
    n = len(dense)
    committed_k = np.full(n, ks_sorted[-1], dtype=np.int64)
    preds = np.zeros(n, dtype=np.int64)
    committed = np.zeros(n, dtype=bool)
    done = np.zeros(n, dtype=bool)

    for k in ks_sorted:
        logits_k = dense.at(k, ppi_len)
        probs = softmax(logits_k)
        pred_k = probs.argmax(axis=-1)
        conf_k = probs.max(axis=-1)
        active = ~done
        hit = active & (conf_k >= tau_c[pred_k])
        committed_k[hit] = k
        preds[hit] = pred_k[hit]
        committed[hit] = True
        done |= hit

        if k == energy_k:
            still_active = ~done
            reject = still_active & (energy_score(logits_k) >= theta)
            committed_k[reject] = k
            preds[reject] = -1
            committed[reject] = True
            done |= reject

    if not done.all():
        last_probs = softmax(dense.at(ks_sorted[-1], ppi_len))
        preds[~done] = last_probs[~done].argmax(axis=-1)
    return PolicyResult(k=committed_k, prediction=preds, committed=committed)
