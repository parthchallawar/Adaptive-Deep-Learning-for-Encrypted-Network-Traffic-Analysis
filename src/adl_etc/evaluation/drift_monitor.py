"""Replay the drift monitor over saved scores and check it against the measured decay (plan
phase 5, T2; spec 012's offline validation).

Flows are replayed in windows of 5,000 decided flows, ``val`` then ``test_id`` then
``test_drift`` (each sorted by period and flow index). Two modes, because the budget controller
holds ``mean_k`` by design:

* **static**: ``theta_commit`` fixed at the value that gives mean K = 6 on validation; all six
  signals are watched.
* **controller**: the budget controller (target mean K = 6, the gains phase 4 chose) moves
  ``theta_commit`` window by window; ``mean_k`` and ``commit_rate`` become informational and
  ``theta_commit`` itself is watched.

References (mean and standard deviation of each signal) and the CUSUM threshold ``h`` come from
the **validation** windows only. Then the weekly mean of each signal is compared with the week's
measured macro-F1 (from :mod:`adl_etc.evaluation.drift_curves`) by Spearman correlation, and the
first alarm is compared with the first week whose macro-F1 is 3 points below the test weeks'.

NumPy only; the monitor itself is :mod:`adl_etc.inference.drift`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from adl_etc.evaluation import replay as R
from adl_etc.evaluation.scores import K_MAX, ScoreArrays
from adl_etc.inference import drift as DR
from adl_etc.inference.controller import BudgetController
from adl_etc.inference.unknown import UnknownModel

WINDOW = 5000
TARGET_MEAN_K = 6.0
F_REJECT = 0.02
CONTROLLER_GAINS = {"kp": 3.0, "ki": 0.02}  # phase 4's validation-chosen pair
F1_DROP = 0.03


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (average ranks for ties); ``nan`` for fewer than 3 pairs or a
    constant input."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def replay_windows(
    scores: ScoreArrays,
    *,
    um: UnknownModel | None,
    reference_hist: np.ndarray,
    theta: float,
    controller: BudgetController | None = None,
    window: int = WINDOW,
    f: float = F_REJECT,
    split: str = "",
) -> list[dict[str, Any]]:
    """Decide every window of ``scores`` (time-ordered) and return its signals. With a
    ``controller`` the threshold in force when a window opens is the controller's, and the
    controller then sees the window. A ragged last window (under half a window) is dropped."""
    order = np.lexsort((scores.flow_index, scores.period))
    u = um.score_of(scores) if um is not None else np.zeros((len(scores), K_MAX))
    theta_rej = um.theta_reject(f) if um is not None else R.NO_REJECTION
    surprise = scores.surprise if um is not None else None
    names = scores.meta.get("periods", [])
    out: list[dict[str, Any]] = []
    for start in range(0, len(order), window):
        idx = order[start : start + window]
        if len(idx) < window // 2:
            break
        th = controller.theta if controller is not None else theta
        res = R.safe_stop(
            scores.p_safe[idx],
            u[idx],
            scores.pred[idx],
            scores.ppi_len[idx],
            th,
            theta_rej,
            k_cap=controller.k_cap if controller is not None else K_MAX,
        )
        rows = np.arange(len(idx))
        stats: dict[str, Any] = dict(
            DR.window_stats(
                res.k,
                res.decision,
                res.prediction,
                scores.p_safe[idx][rows, res.k - 1],
                surprise[idx][rows, res.k - 1] if surprise is not None else None,
                reference_hist,
                theta_commit=th if controller is not None else None,
            )
        )
        period = int(np.bincount(scores.period[idx]).argmax())
        stats.update(
            split=split,
            period=names[period] if period < len(names) else str(period),
            week=f"{split}:{names[period] if period < len(names) else period}",
        )
        out.append(stats)
        if controller is not None:
            controller.update(res.k, res.rejected)
    return out


def _signals(mode: str, um: UnknownModel | None) -> tuple[str, ...]:
    base: tuple[str, ...] = DR.SIGNALS if mode == "static" else DR.CONTROLLER_SIGNALS
    if um is None:  # no unknown score, no next-packet head
        base = tuple(s for s in base if s not in ("reject_rate", "mean_pkt_nll"))
    return tuple(base)


def _weekly(windows: Sequence[Mapping[str, Any]], signals: Sequence[str]) -> dict[str, dict]:
    """Weekly mean of every signal over the windows whose dominant period is that week."""
    weeks: dict[str, list[Mapping[str, Any]]] = {}
    for w in windows:
        weeks.setdefault(str(w["week"]), []).append(w)
    out: dict[str, dict] = {}
    for key, ws in weeks.items():
        out[key] = {"n_windows": len(ws)}
        for s in signals:
            vals = [float(w[s]) for w in ws if s in w and np.isfinite(w[s])]
            out[key][s] = float(np.mean(vals)) if vals else float("nan")
    return out


def run_mode(
    mode: str,
    splits: Mapping[str, ScoreArrays],
    um: UnknownModel | None,
    theta: float,
    reference_hist: np.ndarray,
    curves: Mapping[str, Any],
    *,
    window: int = WINDOW,
    target: float = TARGET_MEAN_K,
    gains: Mapping[str, float] = CONTROLLER_GAINS,
) -> dict[str, Any]:
    """One mode end to end: replay val for references and ``h``, replay the test weeks through
    the monitor, then validate against ``curves`` (``drift_curves.weekly_curves`` output)."""
    signals = _signals(mode, um)
    ctrl = (
        BudgetController(target, theta_init=theta, kp=gains["kp"], ki=gains["ki"])
        if mode == "controller"
        else None
    )
    common: dict[str, Any] = {"um": um, "reference_hist": reference_hist, "theta": theta}
    val_w = replay_windows(splits["val"], controller=ctrl, window=window, split="val", **common)
    ref = DR.reference_from_windows(val_w, signals)
    h = DR.tune_h(ref, val_w, signals)
    monitor = DR.DriftMonitor(ref, signals=signals, h=h)
    test_w: list[dict[str, Any]] = []
    for name in ("test_id", "test_drift"):
        test_w += replay_windows(splits[name], controller=ctrl, window=window, split=name, **common)
    for i, w in enumerate(test_w):
        ev = monitor.update(w)
        w.update(level=ev.level, alarms=list(ev.alarms), window=i)

    weekly = _weekly(test_w, signals)
    keyed = {f"{w['split']}:{w['period']}": w for w in curves["weeks"]}
    pairs = [(wk, keyed[wk]) for wk in weekly if wk in keyed]
    corr: dict[str, dict[str, float]] = {}
    for s in signals:
        xs = [weekly[wk][s] for wk, _ in pairs]
        corr[s] = {
            "spearman_vs_policy_macro_f1": spearman(xs, [c["policy_macro_f1"] for _, c in pairs]),
            "spearman_vs_policy_accuracy": spearman(xs, [c["policy_accuracy"] for _, c in pairs]),
        }

    base = [c["policy_macro_f1"] for _, c in pairs if c["split"] == "test_id"]
    floor = float(np.mean(base)) - F1_DROP if base else float("nan")
    drop = next(
        (c for _, c in pairs if c["split"] == "test_drift" and c["policy_macro_f1"] <= floor), None
    )
    week_index = {wk: i for i, (wk, _) in enumerate(pairs)}

    def first_week(levels: tuple[str, ...]) -> dict[str, Any] | None:
        for w in test_w:
            if w["level"] in levels:
                return {
                    "week": w["week"],
                    "week_index": week_index.get(w["week"]),
                    "window": w["window"],
                    "signals": w["alarms"],
                }
        return None

    warn, alarm = first_week(("warning", "alarm")), first_week(("alarm",))
    drop_i = None if drop is None else week_index.get(f"{drop['split']}:{drop['period']}")
    lead: dict[str, Any] = {
        "macro_f1_floor": floor,
        "first_week_below_floor": None if drop is None else f"{drop['split']}:{drop['period']}",
        "first_warning": warn,
        "first_alarm": alarm,
    }
    for label, hit in (("warning", warn), ("alarm", alarm)):
        if hit is not None and drop_i is not None and hit["week_index"] is not None:
            lead[f"lead_weeks_{label}"] = drop_i - hit["week_index"]  # positive = alarm first
    per_signal = _per_signal_first(test_w, signals)
    return {
        "mode": mode,
        "signals": list(signals),
        "h": h,
        "reference": {s: {"mean": m, "std": sd} for s, (m, sd) in ref.items()},
        "val_alarms": 0,
        "windows": test_w,
        "weekly": weekly,
        "correlation": corr,
        "lead": lead,
        "summary": DR.summary(monitor.events),
        "per_signal_first_alarm": per_signal,
        "per_signal_lead_weeks": {
            s: (
                None
                if hit is None or drop_i is None
                else drop_i - week_index.get(hit["week"], drop_i)
            )
            for s, hit in per_signal.items()
        },
    }


def _per_signal_first(windows: Sequence[Mapping[str, Any]], signals: Sequence[str]) -> dict:
    out: dict[str, Any] = {}
    for s in signals:
        w = next((w for w in windows if s in w["alarms"]), None)
        out[s] = None if w is None else {"week": w["week"], "window": w["window"]}
    return out


def monitor_report(
    splits: Mapping[str, ScoreArrays],
    um: UnknownModel | None,
    curves: Mapping[str, Any],
    *,
    window: int = WINDOW,
) -> dict[str, Any]:
    """Both modes for one run. ``splits`` needs ``val``, ``test_id`` and ``test_drift``;
    ``curves`` is the run's :func:`drift_curves.weekly_curves` result."""
    from adl_etc.evaluation.drift_curves import calibrate_commit_threshold

    theta = calibrate_commit_threshold(splits["val"], TARGET_MEAN_K)
    # the reference class histogram: what validation commits to at this threshold
    order = np.lexsort((splits["val"].flow_index, splits["val"].period))
    u = um.score_of(splits["val"]) if um is not None else np.zeros((len(splits["val"]), K_MAX))
    res = R.safe_stop(
        splits["val"].p_safe[order],
        u[order],
        splits["val"].pred[order],
        splits["val"].ppi_len[order],
        theta,
        um.theta_reject(F_REJECT) if um is not None else R.NO_REJECTION,
    )
    n_classes = int(max(int(s.pred.max()) for s in splits.values()) + 1)
    ref_hist = DR.class_histogram(res.prediction, n_classes)
    return {
        "theta_commit": theta,
        "window": window,
        "static": run_mode("static", splits, um, theta, ref_hist, curves, window=window),
        "controller": run_mode("controller", splits, um, theta, ref_hist, curves, window=window),
    }
