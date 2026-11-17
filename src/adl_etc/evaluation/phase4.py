"""Phase 4's evaluation protocol over saved scores: H3 (unknown score), H2 (stopping policy), H1's
third criterion (packets-to-decision) and H4 (budget controller). Plan phase 4, T2/T3/T5/T6.

Every function is a pure function of :class:`~adl_etc.evaluation.scores.ScoreArrays` and a fitted
:class:`~adl_etc.inference.unknown.UnknownModel`: no model, no torch. The protocol rules are built
in rather than left to the caller:

- anything fitted or chosen (P-CAPE's per-class thresholds, P-RL's value tables, the controller's
  gains, the starting threshold) uses **validation** scores, and for the unknown model and the
  baselines' fits only the **known** validation flows (the held-out classes are the same in val
  and test, so unknown-class validation flows must never steer a choice);
- operating points are matched on validation and then *reported* on test, so a "same false-positive
  rate" comparison never tunes on the test labels;
- every sweep is labelled as a sweep, not a result: only the numbers a function returns for test
  or drift scores are quotable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from adl_etc.evaluation import controller_replay as CR
from adl_etc.evaluation import metrics as M
from adl_etc.evaluation import replay as R
from adl_etc.evaluation import rl_stop
from adl_etc.evaluation.scores import ScoreArrays
from adl_etc.inference.controller import BudgetController
from adl_etc.inference.unknown import UnknownModel, energy_gate_threshold, rejection_k

REPORT_KS = (1, 2, 3, 5, 8, 12, 20, 30)
BUDGETS = (4.0, 6.0, 8.0)
SIGNALS = ("energy_raw", "energy", "mahal", "surprise", "u")


# --- H3: the unknown score ----------------------------------------------------------------------


def signal_columns(s: ScoreArrays, um: UnknownModel) -> dict[str, np.ndarray]:
    """The score columns an AUROC is taken over, ``[N, 30]`` each. Higher = more unknown."""
    z = um.components(s.energy, s.mahal, s.surprise)
    return {
        "energy_raw": s.energy_raw,
        "energy": s.energy,
        "mahal": s.mahal,
        "surprise": s.surprise,
        "u": um.score(s.energy, s.mahal, s.surprise),
        "z_energy": z[..., 0],
    }


def unknown_auroc(
    s: ScoreArrays, um: UnknownModel, ks: Sequence[int] = REPORT_KS
) -> dict[str, dict[int, float]]:
    """AUROC (unknown = positive) at each K, over the flows that reached it."""
    cols = signal_columns(s, um)
    unk = s.labels < 0
    out: dict[str, dict[int, float]] = {name: {} for name in SIGNALS}
    for k in ks:
        reached = s.ppi_len >= k
        if not (unk[reached].any() and (~unk[reached]).any()):
            continue
        for name in SIGNALS:
            out[name][k] = M.auroc(cols[name][reached, k - 1], unk[reached])
    return out


def rejection_stats(
    s: ScoreArrays, um: UnknownModel, f: float, *, k_ood: int = 3
) -> dict[str, float]:
    """Flow-level rejection by the per-K threshold at false-positive target ``f``: the share of
    known flows rejected at *any* K (it compounds over the K positions a flow passes through, so it
    exceeds ``f``), the share of unknown flows rejected, and the mean K of those rejections."""
    u = um.score_of(s)
    rk = rejection_k(u, um.theta_reject(f), s.last_k, k_ood=k_ood)
    known, unk = s.labels >= 0, s.labels < 0
    rejected_unknown = rk[unk] > 0
    return {
        "f": float(f),
        "known_reject_rate": float((rk[known] > 0).mean()),
        "unknown_reject_rate": float(rejected_unknown.mean()) if unk.any() else float("nan"),
        "mean_rejection_k": float(rk[unk][rejected_unknown].mean())
        if rejected_unknown.any()
        else float("nan"),
    }


def energy_gate_stats(
    val: ScoreArrays, s: ScoreArrays, *, k: int = 10, f: float = 0.02
) -> dict[str, float]:
    """The fixed-K energy gate (P-CAPE's, spec 010's baseline): reject at K = ``k`` any flow whose
    raw energy there is above the quantile of known validation flows. Flow-level rates over all
    flows, so a flow shorter than ``k`` is simply never gated."""
    thr = energy_gate_threshold(val.energy_raw, val.labels, val.ppi_len, k, f)
    hit = (s.energy_raw[:, k - 1] >= thr) & (s.ppi_len >= k)
    known, unk = s.labels >= 0, s.labels < 0
    return {
        "f": float(f),
        "k": float(k),
        "known_reject_rate": float(hit[known].mean()),
        "unknown_reject_rate": float(hit[unk].mean()) if unk.any() else float("nan"),
        "mean_rejection_k": float(k) if hit[unk].any() else float("nan"),
    }


def match_f(val: ScoreArrays, um: UnknownModel, target_rate: float, *, k_ood: int = 3) -> float:
    """The ``f`` whose flow-level known-flow rejection rate on **validation** equals
    ``target_rate`` (it is monotone in ``f``, so bisect)."""
    lo, hi = 1e-4, 0.5
    for _ in range(30):
        mid = float(np.sqrt(lo * hi))
        rate = rejection_stats(val, um, mid, k_ood=k_ood)["known_reject_rate"]
        lo, hi = (mid, hi) if rate < target_rate else (lo, mid)
    return float(np.sqrt(lo * hi))


def h3_report(
    val: ScoreArrays,
    test: ScoreArrays,
    um: UnknownModel,
    *,
    fprs: Sequence[float] = (0.02, 0.05),
    gate_k: int = 10,
    k_ood: int = 3,
) -> dict[str, Any]:
    """AUROC(K) for the combined score and each part, and the earliness comparison with the
    fixed K = 10 energy gate **at the same flow-level known-flow rejection rate**, matched on
    validation."""
    out: dict[str, Any] = {"auroc": unknown_auroc(test, um), "earliness": []}
    for fpr in fprs:
        gate_val = energy_gate_stats(val, val, k=gate_k, f=fpr)
        f = match_f(val, um, gate_val["known_reject_rate"], k_ood=k_ood)
        ours = rejection_stats(test, um, f, k_ood=k_ood)
        gate = energy_gate_stats(val, test, k=gate_k, f=fpr)
        out["earliness"].append(
            {
                "gate": gate,
                "ours": ours,
                "matched_on_val_known_reject_rate": gate_val["known_reject_rate"],
                # spec 010: mean rejection K at least 30% lower than the fixed gate's
                "k_reduction": 1.0 - ours["mean_rejection_k"] / gate["mean_rejection_k"],
            }
        )
    return out


# --- H2: the stopping policy ---------------------------------------------------------------------


def cape_thresholds(
    val_known: ScoreArrays, n_classes: int, precision: float, *, min_support: int = 20
) -> np.ndarray:
    """Per-class commit thresholds at ``precision`` from the known validation flows' last-prefix
    confidence, as ``policies.fit_cape_thresholds`` does but on the saved calibrated columns."""
    from adl_etc.evaluation.policies import _threshold_for_precision

    rows = np.arange(len(val_known))
    last = val_known.last_k - 1
    conf = val_known.conf[rows, last].astype(np.float64)
    pred = val_known.pred[rows, last]
    correct = pred == val_known.labels
    tau = np.full(n_classes, _threshold_for_precision(conf, correct, precision))
    for c in range(n_classes):
        m = pred == c
        if int(m.sum()) >= min_support:
            tau[c] = _threshold_for_precision(conf[m], correct[m], precision)
    return tau


def policy_fronts(
    val: ScoreArrays, test: ScoreArrays, n_classes: int, *, n: int = 200
) -> dict[str, dict[str, np.ndarray]]:
    """Closed-world sweeps (known flows only, rejection off) of every policy on **test**, with every
    fit taken on **known validation** flows. Returns ``{policy: {knob, mean_k, accuracy}}``."""
    val_k, test_k = R.closed_world(val), R.closed_world(test)
    out: dict[str, dict[str, np.ndarray]] = {}

    def pack(rows: list[dict[str, float]]) -> dict[str, np.ndarray]:
        return {
            "knob": np.array([r["theta"] for r in rows]),
            "mean_k": np.array([r["mean_k"] for r in rows]),
            "accuracy": np.array([r["known_accuracy"] for r in rows]),
        }

    out["safe_stop"] = pack(R.sweep_theta_commit(test_k, R.logit_grid(n)))
    taus = np.concatenate([np.linspace(0.02, 0.5, n // 5), R.logit_grid(n - n // 5)])
    out["p_echo"] = pack(R.sweep_echo(test_k, taus))
    rows = []
    for p in np.linspace(0.5, 0.999, max(n // 3, 20)):
        tau_c = cape_thresholds(val_k, n_classes, float(p))
        res = R.cape(test_k.conf, test_k.pred, test_k.energy_raw, test_k.ppi_len, tau_c, np.inf)
        rows.append({"theta": float(p), **R.summarize(res, test_k.labels)})
    out["p_cape"] = pack(rows)
    out["p_rl"] = pack(rl_stop.sweep_rl(val_k, test_k, np.logspace(-3, 1.5, max(n // 4, 20))))
    return out


def front_summary(
    front: dict[str, np.ndarray], budgets: Sequence[float] = BUDGETS
) -> dict[str, Any]:
    mk, acc = front["mean_k"], front["accuracy"]
    return {
        "acc_at_budget": {float(b): M.acc_at_budget(mk, acc, b) for b in budgets},
        "pareto_area": M.pareto_area(mk, acc),
        "smallest_mean_k": float(mk.min()),
        "front": M.pareto_front(mk.tolist(), acc.tolist()),
    }


def mean_k_at_accuracy(mean_ks: Sequence[float], accs: Sequence[float], target: float) -> float:
    """The fewest packets, on average, at which a policy sweep reaches ``target`` accuracy
    (``inf`` if it never does). Spec 007's third H1 criterion compares this between models."""
    mk, acc = np.asarray(mean_ks, dtype=np.float64), np.asarray(accs, dtype=np.float64)
    ok = acc >= target
    return float(mk[ok].min()) if ok.any() else float("inf")


def open_world(
    val: ScoreArrays,
    test: ScoreArrays,
    um: UnknownModel,
    *,
    budget: float = 6.0,
    f: float = 0.02,
) -> dict[str, dict[str, float]]:
    """Safe-Stop with rejection on all test flows, at the commit threshold whose **validation**
    mean K (closed world, rejection off) equals ``budget``, for both check orders."""
    order = np.arange(len(val))
    theta, reachable, achieved = CR.calibrate_theta(val, order, budget)
    u = um.score_of(test)
    out: dict[str, dict[str, float]] = {}
    for check in ("reject_first", "commit_first"):
        res = R.safe_stop(
            test.p_safe,
            u,
            test.pred,
            test.ppi_len,
            theta,
            um.theta_reject(f),
            order=check,
        )
        out[check] = {
            "theta_commit": theta,
            "reachable": float(reachable),
            "val_mean_k": achieved,
            **R.summarize(res, test.labels),
        }
    return out


def h2_report(
    val: ScoreArrays,
    test: ScoreArrays,
    um: UnknownModel,
    n_classes: int,
    *,
    budgets: Sequence[float] = BUDGETS,
) -> dict[str, Any]:
    fronts = policy_fronts(val, test, n_classes)
    return {
        "closed_world": {name: front_summary(fr, budgets) for name, fr in fronts.items()},
        "fronts": {name: {k: v.tolist() for k, v in fr.items()} for name, fr in fronts.items()},
        "open_world": open_world(val, test, um, budget=float(np.median(budgets))),
    }


# --- H4: the budget controller -----------------------------------------------------------------


def tune_gains(
    val: ScoreArrays,
    um: UnknownModel,
    *,
    target: float = 6.0,
    kps: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    kis: Sequence[float] = (0.0, 0.02, 0.05),
    f: float = 0.02,
    window: int = 2000,
) -> dict[str, Any]:
    """Choose the controller gains on the **validation** stream only: the pair with the smallest
    mean tracking error over a step schedule (target -> 0.6 target -> target), among those with
    no oscillation event and a settling time of at most 5 windows. If none qualifies the
    smallest-error pair is returned and ``qualified`` says so."""
    order = np.lexsort((val.flow_index, val.period))
    u = um.score_of(val)
    theta_rej = um.theta_reject(f)
    theta0, _, _ = CR.calibrate_theta(val, order, target, u=u, theta_reject=theta_rej)
    n_win = int(np.ceil(len(order) / window))
    schedule = {n_win // 3: 0.6 * target, 2 * n_win // 3: target}
    rows = []
    for kp in kps:
        for ki in kis:
            ctrl = BudgetController(target, theta_init=theta0, kp=kp, ki=ki, f_init=f)
            trace = CR.replay(val, order, ctrl, unknown=um, u=u, window=window, schedule=schedule)
            summ = CR.tracking_summary(trace)
            settle = summ["settling_windows"]
            ok = (
                summ["oscillation_events"] == 0
                and bool(settle)
                and all(0 <= w <= 5 for w in settle)
            )
            rows.append({"kp": kp, "ki": ki, "qualified": ok, **summ})
    pool = [r for r in rows if r["qualified"]] or rows
    best = min(pool, key=lambda r: r["mean_abs_error"])
    return {"kp": best["kp"], "ki": best["ki"], "qualified": best["qualified"], "grid": rows}


def _slice_trace(trace: CR.ReplayTrace, lo: int, hi: int) -> CR.ReplayTrace:
    keep = [i for i, w in enumerate(trace.windows) if lo <= w["start"] < hi]
    return CR.ReplayTrace(
        [trace.windows[i] for i in keep], [e for e in trace.events if e["window"] in set(keep)]
    )


def h4_report(
    val: ScoreArrays,
    test: ScoreArrays,
    drift: ScoreArrays,
    um: UnknownModel,
    *,
    gains: dict[str, Any] | None = None,
    budgets: Sequence[float] = BUDGETS,
    f: float = 0.02,
    window: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """The controller against a static threshold over ``val -> test_id -> test_drift`` in time
    order, at each budget; then the three disturbances of spec 011. The starting threshold is
    calibrated on validation; gains come from :func:`tune_gains` (validation only) unless given."""
    gains = gains or tune_gains(val, um, f=f, window=window)
    kp, ki = float(gains["kp"]), float(gains["ki"])
    scores, order, spans = CR.stream_of([val, test, drift])
    u = um.score_of(scores)
    theta_rej = um.theta_reject(f)
    names = ("val", "test_id", "test_drift")
    out: dict[str, Any] = {"gains": {"kp": kp, "ki": ki}, "budgets": {}, "disturbances": {}}

    for b in budgets:
        theta0, reachable, achieved = CR.calibrate_theta(
            scores, order[spans[0][0] : spans[0][1]], b, u=u, theta_reject=theta_rej
        )
        row: dict[str, Any] = {"theta_init": theta0, "reachable_on_val": reachable}
        for label, static in (("controller", False), ("static", True)):
            ctrl = BudgetController(b, theta_init=theta0, kp=kp, ki=ki, f_init=f)
            trace = CR.replay(scores, order, ctrl, unknown=um, u=u, window=window, static=static)
            row[label] = {
                n: CR.tracking_summary(_slice_trace(trace, lo, hi), warmup=5 if i == 0 else 0)
                for i, (n, (lo, hi)) in enumerate(zip(names, spans, strict=True))
            }
        out["budgets"][float(b)] = row

    b = float(np.median(budgets))
    theta0, _, _ = CR.calibrate_theta(
        scores, order[spans[0][0] : spans[0][1]], b, u=u, theta_reject=theta_rej
    )
    rng = np.random.default_rng(seed)
    hard = CR.hard_class_mask(val)
    d_lo, d_hi = spans[2]
    third = (d_hi - d_lo) // 3
    t_lo, t_hi = spans[1]
    cases = {
        "hard_classes_doubled": CR.double_hard_classes(
            order, scores, hard, d_lo + third, d_lo + 2 * third, rng
        ),
        "unknown_burst": CR.unknown_burst(
            order,
            scores,
            0.3,
            t_lo + (t_hi - t_lo) // 3,
            t_lo + (t_hi - t_lo) // 3 + 20 * window,
            rng,
        ),
    }
    for name, stream in cases.items():
        res = {}
        for label, static in (("controller", False), ("static", True)):
            ctrl = BudgetController(b, theta_init=theta0, kp=kp, ki=ki, f_init=f)
            trace = CR.replay(scores, stream, ctrl, unknown=um, u=u, window=window, static=static)
            res[label] = CR.tracking_summary(trace, warmup=5)
        out["disturbances"][name] = res

    # (c) step schedule on the in-distribution test span alone
    n_win = int(np.ceil((t_hi - t_lo) / window))
    schedule = {n_win // 3: 5.0, 2 * n_win // 3: 8.0}
    res = {}
    theta_t, _, _ = CR.calibrate_theta(
        scores, order[spans[0][0] : spans[0][1]], 8.0, u=u, theta_reject=theta_rej
    )
    for label, static in (("controller", False), ("static", True)):
        ctrl = BudgetController(8.0, theta_init=theta_t, kp=kp, ki=ki, f_init=f)
        trace = CR.replay(
            scores,
            order[t_lo:t_hi],
            ctrl,
            unknown=um,
            u=u,
            window=window,
            schedule=schedule,
            static=static,
        )
        res[label] = CR.tracking_summary(trace, warmup=5)
    out["disturbances"]["target_step_8_5_8"] = res
    return out


def jsonable(obj: Any) -> Any:
    """Make a report JSON-safe: NumPy scalars and arrays become Python values, every dict key a
    string, and a non-finite float ``None`` (strict JSON has no ``inf`` or ``nan``)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def run_report(
    val: ScoreArrays,
    test: ScoreArrays,
    drift: ScoreArrays | None,
    um: UnknownModel,
    n_classes: int,
    *,
    window: int = 2000,
    skip_h4: bool = False,
) -> dict[str, Any]:
    """H3, H2 and (with ``drift`` scores) H4 for one run, JSON-safe. ``meta`` records which
    checkpoint the scores came from so a table can refuse a mismatched set."""
    report: dict[str, Any] = {
        "meta": {
            "checkpoint_hash": test.meta.get("checkpoint_hash"),
            "run": test.meta.get("run"),
            "n_val": len(val),
            "n_test": len(test),
            "n_drift": None if drift is None else len(drift),
            "heads": test.meta.get("heads"),
            "unknown_weights": list(um.weights),
        },
        "h3": h3_report(val, test, um),
        "h2": h2_report(val, test, um, n_classes),
    }
    if drift is not None and not skip_h4:
        report["h4"] = h4_report(val, test, drift, um, window=window)
    return jsonable(report)


__all__ = [
    "BUDGETS",
    "REPORT_KS",
    "cape_thresholds",
    "energy_gate_stats",
    "front_summary",
    "h2_report",
    "h3_report",
    "h4_report",
    "jsonable",
    "match_f",
    "mean_k_at_accuracy",
    "open_world",
    "policy_fronts",
    "rejection_stats",
    "run_report",
    "signal_columns",
    "tune_gains",
    "unknown_auroc",
]
