"""Weekly decay curves from saved score files (plan phase 5, T1; spec 004's drift protocol).

For every period (a week) of ``test_id`` and ``test_drift``: accuracy and macro-F1 on **known**
flows at fixed K = 5, 8 and 30, the same two numbers under the stopping policy's own decisions,
the mean K that policy needs, and (for a model with an unknown score) the unknown AUROC at K = 8.
The policy is Safe-Stop with rejection off and ``theta_commit`` calibrated **once on validation**
to a target mean K, then held fixed, so a change in the weekly mean K is the traffic changing.

Pure NumPy over :class:`~adl_etc.evaluation.scores.ScoreArrays`; no model, no torch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from adl_etc.evaluation import controller_replay as CR
from adl_etc.evaluation import metrics as M
from adl_etc.evaluation import replay as R
from adl_etc.evaluation.scores import K_MAX, ScoreArrays
from adl_etc.inference.unknown import UnknownModel

FIXED_KS = (5, 8, 30)
TARGET_MEAN_K = 6.0
AUROC_K = 8
#: Every week is cut to at most this many flows (a fixed random draw). Macro-F1 over ~150 classes is
#: biased downward on a small sample, and the drift weeks hold about a third of a test week's flows;
#: without the cap the macro-F1 "decay" would partly be a sample-size effect.
WEEK_CAP = 18_000


def calibrate_commit_threshold(val: ScoreArrays, target: float = TARGET_MEAN_K) -> float:
    """``theta_commit`` that gives mean K = ``target`` on the known flows of validation."""
    known = R.closed_world(val)
    order = np.arange(len(known))
    theta, _, _ = CR.calibrate_theta(known, order, target)
    return float(theta)


def week_row(
    s: ScoreArrays,
    *,
    theta_commit: float,
    um: UnknownModel | None = None,
    ks: Sequence[int] = FIXED_KS,
) -> dict[str, Any]:
    """One period's numbers. ``s`` holds the flows of a single period."""
    known = s.labels >= 0
    kn = s.take(known)
    row: dict[str, Any] = {"n": int(len(s)), "n_known": int(known.sum())}
    if len(kn) == 0:
        return row
    for k in ks:
        pred_k = kn.pred_raw[np.arange(len(kn)), np.minimum(k, kn.last_k) - 1]
        row[f"acc_k{k}"] = float((pred_k == kn.labels).mean())
        row[f"macro_f1_k{k}"] = M.macro_f1(pred_k, kn.labels)
    res = R.safe_stop(
        kn.p_safe,
        np.zeros((len(kn), K_MAX)),
        kn.pred,
        kn.ppi_len,
        theta_commit,
        R.NO_REJECTION,
    )
    row["policy_accuracy"] = float((res.prediction == kn.labels).mean())
    row["policy_macro_f1"] = M.macro_f1(res.prediction, kn.labels)
    row["policy_mean_k"] = float(res.k.mean())
    if um is not None and known.any() and (~known).any():
        u = um.score_of(s)[:, AUROC_K - 1]
        row[f"unknown_auroc_k{AUROC_K}"] = M.auroc(u, ~known)
        row["unknown_share"] = float((~known).mean())
    return row


def weekly_curves(
    splits: Mapping[str, ScoreArrays],
    val: ScoreArrays,
    um: UnknownModel | None = None,
    *,
    ks: Sequence[int] = FIXED_KS,
    target_mean_k: float = TARGET_MEAN_K,
    week_cap: int | None = WEEK_CAP,
    seed: int = 0,
) -> dict[str, Any]:
    """``splits`` maps a split name (e.g. ``test_id``, ``test_drift``) to its scores. Returns
    ``{"theta_commit", "weeks": [{"split", "period", "week_index", ...}, ...]}`` with weeks in
    time order (the dict's order, then period index). Each week is cut to ``week_cap`` flows
    (deterministic given ``seed``) so every week is the same size; ``None`` keeps them all."""
    theta = calibrate_commit_threshold(val, target_mean_k)
    weeks: list[dict[str, Any]] = []
    for name, s in splits.items():
        names = s.meta.get("periods", [])
        for p in sorted(int(x) for x in np.unique(s.period)):
            idx = np.flatnonzero(s.period == p)
            if week_cap is not None and len(idx) > week_cap:
                idx = np.sort(np.random.default_rng([seed, p]).choice(idx, week_cap, replace=False))
            row = week_row(s.take(idx), theta_commit=theta, um=um, ks=ks)
            weeks.append(
                {
                    "split": name,
                    "period": names[p] if p < len(names) else str(p),
                    "week_index": len(weeks),
                    **row,
                }
            )
    return {
        "theta_commit": theta,
        "target_mean_k": target_mean_k,
        "week_cap": week_cap,
        "weeks": weeks,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    return float(np.mean(vals)) if vals else float("nan")


def decay_summary(curves: Mapping[str, Any], ks: Sequence[int] = FIXED_KS) -> dict[str, Any]:
    """How much the metrics fall from the in-distribution weeks to the drift weeks: the mean over
    ``test_id`` weeks, the mean over the last four drift weeks, their difference, and the
    least-squares slope per week over the drift weeks. The last test is whether early prefixes
    decay faster than late ones (``early_minus_late``: the K=5 drop minus the K=30 drop)."""
    weeks = curves["weeks"]
    base = [w for w in weeks if w["split"] == "test_id"]
    drift = [w for w in weeks if w["split"] == "test_drift"]
    tail = drift[-4:]
    out: dict[str, Any] = {}
    keys = [f"acc_k{k}" for k in ks] + [f"macro_f1_k{k}" for k in ks]
    keys += ["policy_accuracy", "policy_macro_f1", "policy_mean_k", f"unknown_auroc_k{AUROC_K}"]
    for key in keys:
        if not any(key in w for w in drift):
            continue
        y = np.array([w[key] for w in drift if key in w], dtype=float)
        x = np.arange(len(y), dtype=float)
        slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else float("nan")
        b, t = _mean(base, key), _mean(tail, key)
        out[key] = {"test_id": b, "last_four_drift": t, "change": t - b, "slope_per_week": slope}
    lo, hi = min(ks), max(ks)
    if f"acc_k{lo}" in out and f"acc_k{hi}" in out:
        out["early_minus_late"] = out[f"acc_k{lo}"]["change"] - out[f"acc_k{hi}"]["change"]
        out["early_minus_late_note"] = (
            f"change in acc_k{lo} minus change in acc_k{hi}; negative means the early prefix "
            "loses more"
        )
    return out
