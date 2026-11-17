"""Phase 4's evaluation protocol over synthetic streams of known structure."""

from __future__ import annotations

import pytest

from adl_etc.evaluation import phase4 as P4
from tests.evaluation.synth import N_CLASSES, fitted_unknown, make_set


@pytest.fixture(scope="module")
def streams():
    val = make_set(40, 500, seed=1)
    test = make_set(40, 500, seed=2)
    drift = make_set(60, 500, seed=3, drift=1.0)
    return val, test, drift, fitted_unknown(val)


def test_h3_auroc_is_high_where_the_signal_is_and_the_parts_are_reported(streams) -> None:
    val, test, _, um = streams
    rep = P4.h3_report(val, test, um)
    for name in P4.SIGNALS:
        assert name in rep["auroc"]
    assert rep["auroc"]["energy_raw"][5] > 0.85 and rep["auroc"]["u"][5] > 0.85
    assert abs(rep["auroc"]["mahal"][5] - 0.5) < 1e-9  # no projection in this stream: constant


def test_h3_earliness_is_compared_at_the_same_known_flow_rejection_rate(streams) -> None:
    val, test, _, um = streams
    rep = P4.h3_report(val, test, um, fprs=(0.02, 0.05))
    assert len(rep["earliness"]) == 2
    for row in rep["earliness"]:
        gate, ours = row["gate"], row["ours"]
        # matched on validation, so the test rates are close (not identical: test is another sample)
        assert abs(gate["known_reject_rate"] - ours["known_reject_rate"]) < 0.02
        assert gate["mean_rejection_k"] == 10.0
        assert ours["mean_rejection_k"] < 6.0 and row["k_reduction"] > 0.3


def test_match_f_hits_the_target_known_rate_on_validation(streams) -> None:
    val, _, _, um = streams
    for target in (0.03, 0.08):
        f = P4.match_f(val, um, target)
        got = P4.rejection_stats(val, um, f)["known_reject_rate"]
        assert abs(got - target) < 0.005


def test_cape_thresholds_have_one_entry_per_class_and_rise_with_the_precision(streams) -> None:
    val, _, _, _ = streams
    known = val.take(val.labels >= 0)
    lo = P4.cape_thresholds(known, N_CLASSES, 0.6)
    hi = P4.cape_thresholds(known, N_CLASSES, 0.95)
    assert lo.shape == hi.shape == (N_CLASSES,)
    assert (hi >= lo - 1e-12).all()


def test_h2_fronts_summaries_and_monotone_budget(streams) -> None:
    val, test, _, um = streams
    rep = P4.h2_report(val, test, um, N_CLASSES)
    assert set(rep["closed_world"]) == {"safe_stop", "p_echo", "p_cape", "p_rl"}
    for name, summ in rep["closed_world"].items():
        a = [summ["acc_at_budget"][b] for b in (4.0, 6.0, 8.0)]
        assert a == sorted(a), name  # a larger budget never lowers the best accuracy reachable
        assert 0.0 <= summ["pareto_area"] <= 1.0
        assert summ["smallest_mean_k"] >= 1.0
    # on this stream the safety probability is the true correctness probability, so Safe-Stop is
    # at least as good as the others at a mid budget (a sanity check of the sweep, not a result)
    best = max(s["acc_at_budget"][6.0] for s in rep["closed_world"].values())
    assert rep["closed_world"]["safe_stop"]["acc_at_budget"][6.0] >= best - 0.03


def test_open_world_rejection_first_commits_fewer_unknowns(streams) -> None:
    val, test, _, um = streams
    ow = P4.open_world(val, test, um, budget=6.0)
    assert set(ow) == {"reject_first", "commit_first"}
    assert ow["reject_first"]["unknown_committed"] <= ow["commit_first"]["unknown_committed"]
    assert ow["reject_first"]["known_accuracy"] <= ow["commit_first"]["known_accuracy"] + 1e-9


def test_mean_k_at_accuracy_by_hand() -> None:
    assert P4.mean_k_at_accuracy([3.0, 5.0, 8.0], [0.5, 0.7, 0.9], 0.7) == 5.0
    assert P4.mean_k_at_accuracy([3.0, 5.0, 8.0], [0.5, 0.7, 0.9], 0.95) == float("inf")
    assert P4.mean_k_at_accuracy([8.0, 3.0], [0.9, 0.9], 0.8) == 3.0


def test_tune_gains_uses_validation_only_and_returns_a_qualified_pair(streams) -> None:
    val, _, _, um = streams
    g = P4.tune_gains(val, um, window=500)
    assert g["kp"] in (0.5, 1.0, 1.5, 2.0) and g["ki"] in (0.0, 0.02, 0.05)
    assert len(g["grid"]) == 12 and isinstance(g["qualified"], bool)
    best = min(
        (r for r in g["grid"] if r["qualified"]), key=lambda r: r["mean_abs_error"], default=None
    )
    if best is not None:
        assert (best["kp"], best["ki"]) == (g["kp"], g["ki"])


def test_h4_controller_beats_a_static_threshold_on_drifting_traffic(streams) -> None:
    val, test, drift, um = streams
    rep = P4.h4_report(val, test, drift, um, budgets=(5.0,), window=500)
    row = rep["budgets"][5.0]
    ctrl, stat = row["controller"]["test_drift"], row["static"]["test_drift"]
    assert stat["mean_abs_error"] > 0.1  # the fixed threshold drifted off budget
    assert ctrl["mean_abs_error"] < stat["mean_abs_error"] / 2
    assert ctrl["oscillation_events"] == 0
    assert set(rep["disturbances"]) == {
        "hard_classes_doubled",
        "unknown_burst",
        "target_step_8_5_8",
    }
    step = rep["disturbances"]["target_step_8_5_8"]["controller"]
    assert len(step["settling_windows"]) == 2 and all(0 <= w <= 6 for w in step["settling_windows"])
