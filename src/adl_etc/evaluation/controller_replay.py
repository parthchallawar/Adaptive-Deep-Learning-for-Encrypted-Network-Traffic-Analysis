"""Replay the budget controller over saved scores, in time order (plan phase 4, T5; spec 011's
H4 evaluation; ``docs/design-phase-4.md`` s.4).

Flows are replayed window by window, ``val -> test_id -> test_drift``. Every flow in a window is
decided with the knobs in force when the window **opened**; the controller then sees the window and
moves the knobs for the next one. The static baseline holds the validation-tuned threshold fixed.

Pure NumPy over :class:`~adl_etc.evaluation.scores.ScoreArrays`; the controller itself is
:class:`adl_etc.inference.controller.BudgetController`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from adl_etc.evaluation import replay as R
from adl_etc.evaluation.scores import K_MAX, ScoreArrays, concat
from adl_etc.inference.controller import BudgetController, logit
from adl_etc.inference.unknown import UnknownModel

THETA_LO, THETA_HI = 0.5, 0.999


# --- the stream -------------------------------------------------------------------------------


def stream_of(
    parts: Sequence[ScoreArrays],
) -> tuple[ScoreArrays, np.ndarray, list[tuple[int, int]]]:
    """Concatenate score sets (e.g. val, test_id, test_drift) and return ``(scores, order,
    spans)``: ``order`` lists every flow in replay order (each part sorted by ``(period,
    flow_index)``, parts kept in the order given) and ``spans[i]`` is the ``[start, stop)`` slice
    of ``order`` that belongs to part ``i``."""
    scores = concat(list(parts))
    order, spans, offset = [], [], 0
    for p in parts:
        within = np.lexsort((p.flow_index, p.period))
        order.append(within + offset)
        spans.append((sum(len(o) for o in order[:-1]), sum(len(o) for o in order)))
        offset += len(p)
    return scores, np.concatenate(order), spans


# --- disturbances (spec 011) --------------------------------------------------------------------


def hard_class_mask(
    val: ScoreArrays, k: int = 5, quantile: float = 0.25, min_flows: int = 20
) -> np.ndarray:
    """Boolean per class index: the bottom ``quantile`` of classes by accuracy at ``k`` on the
    known flows of ``val`` (classes with fewer than ``min_flows`` such flows are ignored)."""
    known = val.labels >= 0
    n_classes = int(max(val.labels.max(), val.pred.max())) + 1
    reached = known & (val.ppi_len >= k)
    correct = (val.pred[:, k - 1] == val.labels) & reached
    n = np.bincount(val.labels[reached], minlength=n_classes)
    c = np.bincount(val.labels[correct], minlength=n_classes)
    acc = np.where(n >= min_flows, c / np.maximum(n, 1), np.nan)
    cut = np.nanquantile(acc, quantile)
    return np.nan_to_num(acc, nan=np.inf) <= cut


def double_hard_classes(
    order: np.ndarray,
    scores: ScoreArrays,
    hard: np.ndarray,
    lo: int,
    hi: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample the stretch ``order[lo:hi]`` with weight 2 on flows of hard classes, so their
    share about doubles (disturbance (a))."""
    out = order.copy()
    seg = order[lo:hi]
    labels = scores.labels[seg]
    w = np.where((labels >= 0) & hard[np.maximum(labels, 0)], 2.0, 1.0)
    out[lo:hi] = rng.choice(seg, size=len(seg), replace=True, p=w / w.sum())
    return out


def unknown_burst(
    order: np.ndarray, scores: ScoreArrays, share: float, lo: int, hi: int, rng: np.random.Generator
) -> np.ndarray:
    """Raise the unknown-class share of ``order[lo:hi]`` to ``share`` by swapping known flows for
    unknown flows drawn (with replacement) from the same stretch (disturbance (b))."""
    out = order.copy()
    seg = order[lo:hi]
    unk = scores.labels[seg] < 0
    s0 = float(unk.mean())
    if s0 >= share or not unk.any():
        return out
    q = (share - s0) / (1.0 - s0)
    swap = (~unk) & (rng.random(len(seg)) < q)
    pool = seg[unk]
    new = seg.copy()
    new[swap] = rng.choice(pool, size=int(swap.sum()), replace=True)
    out[lo:hi] = new
    return out


# --- the replay -------------------------------------------------------------------------------


@dataclass
class ReplayTrace:
    windows: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def column(self, name: str) -> np.ndarray:
        return np.array([w[name] for w in self.windows], dtype=np.float64)


def replay(
    scores: ScoreArrays,
    order: np.ndarray,
    controller: BudgetController,
    *,
    unknown: UnknownModel | None = None,
    u: np.ndarray | None = None,
    window: int = 2000,
    k_min: int = 2,
    k_ood: int = 3,
    check_order: str = "reject_first",
    schedule: Mapping[int, float] | None = None,
    static: bool = False,
) -> ReplayTrace:
    """Run ``controller`` (or, with ``static=True``, just its initial knobs) over ``order``.
    ``schedule`` maps a window index to a new target. Rejection is on when ``unknown`` is given
    (its per-K threshold at the controller's current ``f``); otherwise off."""
    if unknown is not None and u is None:
        u = unknown.score_of(scores)
    uu = u if u is not None else np.zeros((len(scores), K_MAX))
    schedule = dict(schedule or {})
    trace = ReplayTrace()
    theta_cache: dict[float, np.ndarray] = {}
    for w, start in enumerate(range(0, len(order), window)):
        if w in schedule:
            controller.set_target(schedule[w])
        idx = order[start : start + window]
        if unknown is not None:
            f = round(controller.f, 6)
            if f not in theta_cache:
                theta_cache[f] = unknown.theta_reject(f)
            theta_rej = theta_cache[f]
        else:
            theta_rej = R.NO_REJECTION
        res = R.safe_stop(
            scores.p_safe[idx],
            uu[idx],
            scores.pred[idx],
            scores.ppi_len[idx],
            controller.theta,
            theta_rej,
            k_min=k_min,
            k_ood=k_ood,
            order=check_order,
            k_cap=controller.k_cap,
        )
        summary = R.summarize(res, scores.labels[idx])
        trace.windows.append(
            {
                "window": w,
                "start": start,
                "n": len(idx),
                "target": controller.target,
                "theta": controller.theta,
                "k_cap": controller.k_cap,
                "f": controller.f,
                **{k: v for k, v in summary.items() if k != "n"},
            }
        )
        if not static:
            ev = controller.update(res.k, res.rejected)
            trace.events.append({"window": w, "reason": ev.reason, "error": ev.error})
    return trace


def calibrate_theta(
    scores: ScoreArrays,
    order: np.ndarray,
    target: float,
    *,
    u: np.ndarray | None = None,
    theta_reject: np.ndarray = R.NO_REJECTION,
    k_min: int = 2,
    k_ood: int = 3,
    check_order: str = "reject_first",
    tol: float = 0.01,
    iters: int = 40,
) -> tuple[float, bool, float]:
    """The ``theta_commit`` whose mean K on this stream equals ``target`` (mean K is
    non-decreasing in theta, so bisect on ``logit(theta)``). Returns ``(theta, reachable,
    achieved_mean_k)``; an unreachable target is clamped to the nearer rail."""
    uu = u if u is not None else np.zeros((len(scores), K_MAX))
    sub = order

    def mean_k(theta: float) -> float:
        r = R.safe_stop(
            scores.p_safe[sub],
            uu[sub],
            scores.pred[sub],
            scores.ppi_len[sub],
            theta,
            theta_reject,
            k_min=k_min,
            k_ood=k_ood,
            order=check_order,
        )
        return float(r.k.mean())

    lo, hi = logit(THETA_LO), logit(THETA_HI)
    k_lo, k_hi = mean_k(THETA_LO), mean_k(THETA_HI)
    if target <= k_lo:
        return THETA_LO, abs(k_lo - target) <= tol, k_lo
    if target >= k_hi:
        return THETA_HI, abs(k_hi - target) <= tol, k_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        theta = 1.0 / (1.0 + np.exp(-mid))
        k = mean_k(theta)
        if abs(k - target) <= tol:
            return float(theta), True, k
        lo, hi = (mid, hi) if k < target else (lo, mid)
    theta = float(1.0 / (1.0 + np.exp(-0.5 * (lo + hi))))
    return theta, True, mean_k(theta)


# --- metrics (spec 011) -----------------------------------------------------------------------


def tracking_summary(
    trace: ReplayTrace, *, warmup: int = 5, tol: float = 0.05, settle_run: int = 3
) -> dict[str, Any]:
    """Tracking error after a ``warmup``, the share of windows within ``tol`` of the target, the
    settling time after every target step, and accuracy over the same windows."""
    mean_k = trace.column("mean_k")
    target = trace.column("target")
    n = np.array([w["n"] for w in trace.windows])
    full = n >= 0.5 * n.max()  # ignore a ragged last window
    signed = (mean_k - target) / target
    err = np.abs(signed)
    usable = full.copy()
    usable[:warmup] = False
    out: dict[str, Any] = {
        "windows": int(len(err)),
        "mean_abs_error": float(err[usable].mean()) if usable.any() else float("nan"),
        "max_abs_error": float(err[usable].max()) if usable.any() else float("nan"),
        "within_tol": float((err[usable] <= tol).mean()) if usable.any() else float("nan"),
    }
    if "known_accuracy" in trace.windows[0]:
        acc = trace.column("known_accuracy")
        w = n * np.isfinite(acc) * usable
        out["accuracy"] = float((np.nan_to_num(acc) * w).sum() / max(w.sum(), 1))
    steps = [i for i in range(1, len(target)) if target[i] != target[i - 1]]
    settle = []
    smooth = []
    for s in steps:
        after = err[s:]
        run = [
            i
            for i in range(len(after) - settle_run + 1)
            if (after[i : i + settle_run] <= tol).all()
        ]
        settle.append(int(run[0]) if run else -1)  # -1: never settled
        # A window's own mean-K is noisy (about 5% with 2000 flows), so "3 windows in a row each
        # within 5%" is often missed by chance even when the controller sits on the target. Added
        # on 2026-11-18, after seeing exactly that on the first real run: the same question asked
        # of the mean signed error over the next ``settle_run`` windows, which averages the noise
        # out and still catches a controller that is slow or biased. Reported beside the strict one.
        sm = [
            i
            for i in range(len(after) - settle_run + 1)
            if abs(float(signed[s + i : s + i + settle_run].mean())) <= tol
        ]
        smooth.append(int(sm[0]) if sm else -1)
    out["settling_windows"] = settle
    out["settling_windows_smoothed"] = smooth
    out["oscillation_events"] = int(sum("oscillation" in e["reason"] for e in trace.events))
    out["infeasible_events"] = int(sum(e["reason"].startswith("infeasible") for e in trace.events))
    return out
