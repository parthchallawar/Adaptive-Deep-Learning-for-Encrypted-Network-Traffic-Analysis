"""Spec 012's label-free drift monitor (plan phase 5, T2).

Torch-free and NumPy-only, because the service (phase 6) runs the same code online as the
offline replay does. Nothing here needs a label.

A window is a fixed number of **decided flows** (5,000). From it :func:`window_stats` computes six
signals; :class:`Cusum` watches each one, standardised by its mean and standard deviation over
validation windows; :class:`DriftMonitor` turns the alarms into a level.

==================== =========================================== ====================
signal               what it is                                  expected under drift
==================== =========================================== ====================
``mean_k``           mean packets read at the decision           rises
``commit_rate``      share decided by the rule, not forced       falls
``reject_rate``      share rejected as unknown                   rises
``mean_safe_at_commit`` mean ``p_safe`` of the flows that were   falls
                     committed to a class
``mean_pkt_nll``     mean next-packet negative log-likelihood    rises
                     (spec 007's head) over the packets read
``prior_kl``         KL(committed-class histogram || val's)      rises
==================== =========================================== ====================

The budget controller holds ``mean_k`` by design (spec 011), so when it runs ``mean_k`` and
``commit_rate`` are informational and the controller's own ``theta_commit`` becomes a signal: a
threshold that keeps moving is the controller reporting that the traffic changed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from adl_etc.inference.policy import COMMIT, FORCED_COMMIT, FORCED_REJECT, REJECT

SIGNALS = (
    "mean_k",
    "commit_rate",
    "reject_rate",
    "mean_safe_at_commit",
    "mean_pkt_nll",
    "prior_kl",
)
#: With the controller running: ``mean_k`` and ``commit_rate`` are held by design.
CONTROLLER_SIGNALS = (
    "reject_rate",
    "mean_safe_at_commit",
    "mean_pkt_nll",
    "prior_kl",
    "theta_commit",
)
KL_SMOOTHING = 1e-3
STD_FLOOR = 1e-9


class Cusum:
    """Two-sided CUSUM on standardised values. ``update(z)`` returns True when either side
    crosses ``h`` and then resets, so a persistent shift alarms again after it re-accumulates."""

    def __init__(self, h: float = 5.0, k: float = 0.5) -> None:
        if h <= 0 or k < 0:
            raise ValueError(f"need h > 0 and k >= 0, got h={h}, k={k}")
        self.h, self.k = float(h), float(k)
        self.pos = 0.0
        self.neg = 0.0

    def update(self, z: float) -> bool:
        if not np.isfinite(z):
            return False
        self.pos = max(0.0, self.pos + z - self.k)
        self.neg = max(0.0, self.neg - z - self.k)
        if self.pos > self.h or self.neg > self.h:
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self.pos = 0.0
        self.neg = 0.0


def class_histogram(pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Counts of committed class indices (``pred >= 0``), length ``n_classes``."""
    p = np.asarray(pred)
    p = p[(p >= 0) & (p < n_classes)]
    return np.bincount(p.astype(np.int64), minlength=n_classes).astype(np.float64)


def prior_kl(counts: np.ndarray, reference: np.ndarray, smoothing: float = KL_SMOOTHING) -> float:
    """KL(p || q) of two class histograms. Each is turned into frequencies, ``smoothing`` is added
    to every class, and the result is renormalised, so a class absent from one side is finite.
    Zero for identical histograms. ``nan`` if either histogram is empty."""
    c, r = np.asarray(counts, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if c.sum() <= 0 or r.sum() <= 0:
        return float("nan")
    p = c / c.sum() + smoothing
    q = r / r.sum() + smoothing
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def window_stats(
    k: np.ndarray,
    decision: np.ndarray,
    prediction: np.ndarray,
    p_at_decision: np.ndarray,
    surprise_at_decision: np.ndarray | None,
    reference_hist: np.ndarray,
    *,
    theta_commit: float | None = None,
) -> dict[str, float]:
    """The six signals (and ``theta_commit`` when given) of one window of decided flows.

    ``k``, ``decision`` (the policy codes), ``prediction`` (class index, -1 = rejected),
    ``p_at_decision`` (``p_safe`` at the decision packet) and ``surprise_at_decision`` (the mean
    next-packet NLL over the packets read, or None for a model without that head) are all per
    flow. ``reference_hist`` is the committed-class histogram of validation."""
    d = np.asarray(decision)
    rejected = (d == REJECT) | (d == FORCED_REJECT)
    committed = (d == COMMIT) | (d == FORCED_COMMIT)
    out: dict[str, float] = {
        "n": float(len(d)),
        "mean_k": float(np.mean(k)),
        "commit_rate": float(np.mean(d == COMMIT)),
        "reject_rate": float(np.mean(rejected)),
        "mean_safe_at_commit": (
            float(np.mean(np.asarray(p_at_decision, dtype=np.float64)[committed]))
            if committed.any()
            else float("nan")
        ),
        "prior_kl": prior_kl(class_histogram(prediction, len(reference_hist)), reference_hist),
    }
    if surprise_at_decision is not None:
        out["mean_pkt_nll"] = float(np.mean(surprise_at_decision))
    if theta_commit is not None:
        out["theta_commit"] = float(theta_commit)
    return out


@dataclass(frozen=True)
class DriftEvent:
    window: int
    level: str  # "none" | "warning" | "alarm"
    alarms: tuple[str, ...]  # signals whose CUSUM crossed in this window
    z: dict[str, float]


class DriftMonitor:
    """Per-signal CUSUMs plus a combined level: ``warning`` if one signal alarms in the window,
    ``alarm`` if two or more distinct signals alarmed within the last ``span`` windows. The first
    ``warmup`` windows are only counted, never alarmed or accumulated."""

    def __init__(
        self,
        reference: Mapping[str, tuple[float, float]],
        *,
        signals: Sequence[str] | None = None,
        h: float = 5.0,
        k: float = 0.5,
        warmup: int = 5,
        span: int = 3,
    ) -> None:
        self.signals = tuple(signals if signals is not None else reference)
        missing = [s for s in self.signals if s not in reference]
        if missing:
            raise KeyError(f"no reference mean/std for {missing}")
        self.reference = {s: (float(reference[s][0]), float(reference[s][1])) for s in self.signals}
        self.cusum = {s: Cusum(h, k) for s in self.signals}
        self.warmup, self.span = int(warmup), int(span)
        self.window = 0
        self._recent: list[set[str]] = []
        self.events: list[DriftEvent] = []

    def z(self, signal: str, value: float) -> float:
        mean, std = self.reference[signal]
        return (value - mean) / max(std, STD_FLOOR)

    def update(self, stats: Mapping[str, float]) -> DriftEvent:
        w = self.window
        self.window += 1
        z = {s: self.z(s, float(stats[s])) for s in self.signals if s in stats}
        alarms: tuple[str, ...] = ()
        if w >= self.warmup:
            alarms = tuple(s for s, v in z.items() if self.cusum[s].update(v))
        self._recent.append(set(alarms))
        self._recent = self._recent[-self.span :]
        recent = set().union(*self._recent)
        if len(recent) >= 2:
            level = "alarm"
        elif alarms:
            level = "warning"
        else:
            level = "none"
        ev = DriftEvent(w, level, alarms, z)
        self.events.append(ev)
        return ev


def reference_from_windows(
    windows: Sequence[Mapping[str, float]], signals: Sequence[str]
) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation of each signal over validation windows (NaN windows skipped)."""
    ref: dict[str, tuple[float, float]] = {}
    for s in signals:
        vals = np.array([w[s] for w in windows if s in w], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            raise ValueError(f"need at least two validation windows with {s!r}, got {len(vals)}")
        ref[s] = (float(vals.mean()), float(vals.std(ddof=1)))
    return ref


def tune_h(
    reference: Mapping[str, tuple[float, float]],
    val_windows: Sequence[Mapping[str, float]],
    signals: Sequence[str],
    *,
    k: float = 0.5,
    warmup: int = 5,
    h_start: float = 5.0,
    h_step: float = 0.5,
    h_max: float = 30.0,
) -> float:
    """The smallest ``h`` (from ``h_start`` in steps of ``h_step``) at which replaying the
    validation windows gives no alarm on any signal (spec 012: tuned on validation by requiring
    zero alarms). Raises if even ``h_max`` alarms."""
    h = h_start
    while h <= h_max + 1e-9:
        mon = DriftMonitor(reference, signals=signals, h=h, k=k, warmup=warmup)
        if not any(mon.update(w).alarms for w in val_windows):
            return float(h)
        h += h_step
    raise ValueError(f"validation still alarms at h = {h_max}")


def summary(events: Sequence[DriftEvent]) -> dict[str, Any]:
    """The first warning and first alarm window and the signals behind them."""
    out: dict[str, Any] = {"first_warning": None, "first_alarm": None, "n_alarm_windows": 0}
    for ev in events:
        if ev.level != "none" and out["first_warning"] is None:
            out["first_warning"] = {"window": ev.window, "signals": list(ev.alarms)}
        if ev.level == "alarm":
            out["n_alarm_windows"] += 1
            if out["first_alarm"] is None:
                out["first_alarm"] = {"window": ev.window, "signals": list(ev.alarms)}
    return out
