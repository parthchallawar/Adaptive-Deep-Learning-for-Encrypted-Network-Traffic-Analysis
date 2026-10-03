"""Per-K calibration of the PAT's outputs (spec 008; plan phase 3, T7).

Pure NumPy, like the rest of ``evaluation/``: no torch, no scikit-learn (the
isotonic fit is pool-adjacent-violators, written here and tested against
scikit-learn's ``IsotonicRegression`` where that is installed).

Fitted on the validation period, in this order:

1. **Logit adjustment** (Menon et al., 2021). Training samples class ``c`` at
   ``f_c = min(cap, mean_count / n_c)`` times its natural rate (the shared
   loop's ``balanced_weights``), so the model learns a prior tilted towards rare
   classes. Adding ``-log f_c`` to class ``c``'s logit undoes exactly that tilt
   (any constant offset cancels in the softmax). Temperature scaling cannot do
   this: it sharpens or flattens scores, it never moves mass between classes.
2. **Temperature per K**: ``T_K = argmin NLL(softmax(adjusted_K / T_K))``.
3. **Commit-safety per K**: isotonic regression of "argmax was right" on
   ``sigmoid(safe_K)``, so ``p_safe(K) ≈ P(correct | commit at K)``; a Platt
   (logistic) fit where a K has fewer than ``min_isotonic`` samples.

**Who is calibrated at K.** The flows that have *reached* packet K
(``ppi_len >= K``) and belong to a known class: the population a stopping
decision at K actually faces. A flow that ended earlier has no "after K packets"
state, and an unknown-class flow has no correct answer to be calibrated against.

Arrays follow the PAT's output layout: per-position ``[N, K_MAX, ...]`` with
index ``k - 1`` = after k packets (causal, ``effective`` indexing).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adl_etc.utils.atomic import atomic_write_json

FORMAT = "adl-calibration-v1"


class CalibrationError(ValueError):
    """A calibration that cannot be fitted, or is used with the wrong checkpoint."""


# --- logit adjustment ----------------------------------------------------------------------


def prior_adjustment(class_counts: np.ndarray, cap: float | None) -> np.ndarray:
    """``-log f_c`` per class, where ``f_c`` is the balanced sampler's
    oversampling factor (``datasets.balanced_weights``); 0 for a class with no
    training flows (nothing to undo) and everywhere when ``cap`` is ``None``
    (training sampled naturally)."""
    counts = np.asarray(class_counts, dtype=np.float64)
    out = np.zeros_like(counts)
    if cap is None:
        return out
    present = counts > 0
    if not present.any():
        return out
    mean_count = counts.sum() / present.sum()
    factor = np.minimum(cap, mean_count / counts[present])
    out[present] = -np.log(factor)
    return out


# --- temperature ---------------------------------------------------------------------------


def _log_softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    logp = _log_softmax(np.asarray(logits, dtype=np.float64) / temperature)
    return float(-logp[np.arange(len(labels)), labels].mean())


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, *, low: float = 0.05, high: float = 20.0
) -> float:
    """``argmin_T NLL(softmax(logits / T))`` by golden-section search over
    ``log T`` (NLL is unimodal in T for fixed logits). ``T = 1`` is returned
    unchanged when it is already at least as good, so the fit never makes NLL
    worse."""
    logits = np.asarray(logits, dtype=np.float64)
    a, b = math.log(low), math.log(high)
    ratio = (math.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = nll(logits, labels, math.exp(c)), nll(logits, labels, math.exp(d))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = nll(logits, labels, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = nll(logits, labels, math.exp(d))
    best = math.exp((a + b) / 2)
    return best if nll(logits, labels, best) < nll(logits, labels, 1.0) else 1.0


# --- isotonic and Platt --------------------------------------------------------------------


def isotonic_fit(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Non-decreasing least-squares fit of ``y`` on ``x`` by pool-adjacent-
    violators. Returns ``(thresholds, values)``: the distinct ``x`` (sorted) and
    the fitted value at each; :func:`isotonic_predict` interpolates between them
    and clips outside, which is scikit-learn's ``IsotonicRegression`` default."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=np.float64)
    if x.size == 0:
        raise CalibrationError("cannot fit an isotonic map to no samples")
    order = np.argsort(x, kind="mergesort")
    x, y, w = x[order], y[order], w[order]
    ux, start = np.unique(x, return_index=True)
    wsum = np.add.reduceat(w, start)
    ymean = np.add.reduceat(w * y, start) / wsum

    # Pool adjacent violators over the distinct-x points.
    values: list[float] = []
    sizes: list[float] = []
    counts: list[int] = []
    for v, s in zip(ymean.tolist(), wsum.tolist(), strict=True):
        values.append(v)
        sizes.append(s)
        counts.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            v2, s2, n2 = values.pop(), sizes.pop(), counts.pop()
            v1, s1, n1 = values.pop(), sizes.pop(), counts.pop()
            values.append((v1 * s1 + v2 * s2) / (s1 + s2))
            sizes.append(s1 + s2)
            counts.append(n1 + n2)
    fitted = np.repeat(np.array(values), np.array(counts))
    return ux, fitted


def isotonic_predict(thresholds: np.ndarray, values: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(x, dtype=np.float64), thresholds, values)


def platt_fit(scores: np.ndarray, targets: np.ndarray, *, iters: int = 50) -> tuple[float, float]:
    """``(a, b)`` with ``P(target) = sigmoid(a · score + b)`` by Newton's method on
    the logistic log-loss (a tiny ridge keeps a separable sample finite)."""
    s = np.asarray(scores, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    a, b, ridge = 1.0, 0.0, 1e-6
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a * s + b)))
        g = np.array([((p - t) * s).sum() + ridge * a, (p - t).sum() + ridge * b])
        r = p * (1 - p)
        h = np.array([[(r * s * s).sum() + ridge, (r * s).sum()],
                      [(r * s).sum(), r.sum() + ridge]])  # fmt: skip
        step = np.linalg.solve(h, g)
        a, b = a - float(step[0]), b - float(step[1])
        if float(np.abs(step).max()) < 1e-10:
            break
    return a, b


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# --- the fitted calibration ----------------------------------------------------------------


@dataclass
class SafeMap:
    """One K's map from ``sigmoid(safe)`` to ``P(correct)``."""

    kind: str  # "isotonic" | "platt" | "constant"
    thresholds: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    a: float = 1.0
    b: float = 0.0

    def __call__(self, safe_logits: np.ndarray) -> np.ndarray:
        if self.kind == "isotonic":
            return isotonic_predict(
                np.array(self.thresholds), np.array(self.values), _sigmoid(safe_logits)
            )
        if self.kind == "platt":
            return _sigmoid(self.a * _sigmoid(safe_logits) + self.b)
        return np.full(np.shape(safe_logits), self.values[0] if self.values else 0.5)


@dataclass
class Calibration:
    checkpoint_hash: str
    adjustment: list[float]  # per class, added to every position's logits
    temperatures: list[float]  # per K, index k - 1
    safe_maps: list[SafeMap]  # per K
    n_fit: list[int]  # samples behind each K's fit
    ece_val: dict[str, float] = field(default_factory=dict)  # fit diagnostics only

    # -- applying -------------------------------------------------------------

    def adjusted(self, logits: np.ndarray) -> np.ndarray:
        """Logit adjustment only (the natural-prior logits), ``[..., C]``."""
        return np.asarray(logits, dtype=np.float64) + np.asarray(self.adjustment)

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        """Calibrated class probabilities for per-position logits ``[N, K, C]``."""
        adj = self.adjusted(logits)
        temps = np.asarray(self.temperatures)[None, : adj.shape[1], None]
        return np.exp(_log_softmax(adj / temps))

    def safe_probability(self, safe_logits: np.ndarray) -> np.ndarray:
        """Calibrated ``P(correct | commit now)`` for ``safe_logits`` ``[N, K]``."""
        safe_logits = np.asarray(safe_logits, dtype=np.float64)
        return np.stack(
            [self.safe_maps[k](safe_logits[:, k]) for k in range(safe_logits.shape[1])], axis=1
        )

    def check(self, checkpoint_hash: str) -> None:
        if checkpoint_hash != self.checkpoint_hash:
            raise CalibrationError(
                f"calibration was fitted for checkpoint {self.checkpoint_hash[:12]}, "
                f"not {checkpoint_hash[:12]}"
            )

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        atomic_write_json(
            path,
            {
                "format": FORMAT,
                "checkpoint_hash": self.checkpoint_hash,
                "adjustment": self.adjustment,
                "temperatures": self.temperatures,
                "safe_maps": [m.__dict__ for m in self.safe_maps],
                "n_fit": self.n_fit,
                "ece_val": self.ece_val,
            },
        )

    @classmethod
    def load(cls, path: str | Path, *, checkpoint_hash: str | None = None) -> Calibration:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("format") != FORMAT:
            raise CalibrationError(f"{path}: not a {FORMAT} file")
        cal = cls(
            checkpoint_hash=raw["checkpoint_hash"],
            adjustment=list(raw["adjustment"]),
            temperatures=list(raw["temperatures"]),
            safe_maps=[SafeMap(**m) for m in raw["safe_maps"]],
            n_fit=list(raw["n_fit"]),
            ece_val=dict(raw.get("ece_val", {})),
        )
        if checkpoint_hash is not None:
            cal.check(checkpoint_hash)
        return cal


def reached(ppi_len: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """The flows calibrated at K: known class (label >= 0) and ``ppi_len >= k``."""
    return (np.asarray(labels) >= 0) & (np.asarray(ppi_len) >= k)


def fit_calibration(
    cls_logits: np.ndarray,
    safe_logits: np.ndarray,
    labels: np.ndarray,
    ppi_len: np.ndarray,
    *,
    class_counts: np.ndarray,
    cap: float | None,
    checkpoint_hash: str,
    min_isotonic: int = 200,
    ece_ks: tuple[int, ...] = (3, 8, 30),
) -> Calibration:
    """Fit spec 008's calibration on validation arrays (model-index ``labels``,
    ``-1`` for unknown-class flows, which are skipped)."""
    from adl_etc.evaluation.metrics import ece

    n, k_max, c = np.shape(cls_logits)
    labels = np.asarray(labels)
    adjustment = prior_adjustment(class_counts, cap)
    if adjustment.shape != (c,):
        raise CalibrationError(f"class_counts has {adjustment.shape[0]} classes, logits {c}")

    temperatures: list[float] = []
    maps: list[SafeMap] = []
    n_fit: list[int] = []
    ece_val: dict[str, float] = {}
    for k in range(1, k_max + 1):
        rows = reached(ppi_len, labels, k)
        n_fit.append(int(rows.sum()))
        if not rows.any():
            temperatures.append(1.0)
            maps.append(SafeMap(kind="constant", values=[0.5]))
            continue
        z = np.asarray(cls_logits[rows, k - 1], dtype=np.float64) + adjustment
        y = labels[rows]
        temp = fit_temperature(z, y)
        temperatures.append(temp)

        correct = (z.argmax(-1) == y).astype(np.float64)
        s = np.asarray(safe_logits[rows, k - 1], dtype=np.float64)
        if rows.sum() >= min_isotonic:
            th, vals = isotonic_fit(_sigmoid(s), correct)
            maps.append(SafeMap(kind="isotonic", thresholds=th.tolist(), values=vals.tolist()))
        else:
            a, b = platt_fit(_sigmoid(s), correct)
            maps.append(SafeMap(kind="platt", a=a, b=b))

        if k in ece_ks:
            p = np.exp(_log_softmax(z / temp))
            ece_val[f"ece_k{k}"] = ece(p.max(-1), correct)

    return Calibration(
        checkpoint_hash=checkpoint_hash,
        adjustment=adjustment.tolist(),
        temperatures=temperatures,
        safe_maps=maps,
        n_fit=n_fit,
        ece_val=ece_val,
    )


def ece_per_k(
    cal: Calibration,
    cls_logits: np.ndarray,
    labels: np.ndarray,
    ppi_len: np.ndarray,
    ks: tuple[int, ...] = (3, 8, 30),
    safe_logits: np.ndarray | None = None,
) -> dict[str, Any]:
    """ECE of the calibrated top-class probability at each K in ``ks``, over the
    flows that reached K (spec 008's criterion, reported on test_id). With
    ``safe_logits``, also ``safe_ece_k<K>``: how far the calibrated
    ``P(correct | commit at K)`` is from the observed hit rate, which is what the
    budget controller (spec 011) relies on."""
    from adl_etc.evaluation.metrics import ece

    out: dict[str, Any] = {}
    labels = np.asarray(labels)
    for k in ks:
        rows = reached(ppi_len, labels, k)
        if not rows.any():
            out[f"ece_k{k}"] = None
            continue
        z = cal.adjusted(cls_logits[rows, k - 1]) / cal.temperatures[k - 1]
        p = np.exp(_log_softmax(z))
        hit = (p.argmax(-1) == labels[rows]).astype(np.float64)
        out[f"ece_k{k}"] = ece(p.max(-1), hit)
        out[f"n_k{k}"] = int(rows.sum())
        if safe_logits is not None:
            out[f"safe_ece_k{k}"] = ece(cal.safe_maps[k - 1](safe_logits[rows, k - 1]), hit)
    return out
