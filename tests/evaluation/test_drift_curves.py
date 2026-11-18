"""Weekly decay curves (plan phase 5, T1)."""

from __future__ import annotations

import numpy as np

from adl_etc.evaluation import drift_curves as D
from tests.evaluation.synth import fitted_unknown, make_set
from tests.evaluation.test_controller_replay import drifting_scores


def streams(drift: float):
    val = drifting_scores(4, 1000, seed=1, drift=0.0)
    test = drifting_scores(4, 1000, seed=2, drift=0.0)
    late = drifting_scores(10, 1000, seed=3, drift=drift)
    return val, test, late


def test_weeks_are_in_time_order_with_names_and_equal_caps() -> None:
    val, test, late = streams(1.0)
    c = D.weekly_curves({"test_id": test, "test_drift": late}, val, week_cap=600)
    weeks = c["weeks"]
    assert [w["split"] for w in weeks] == ["test_id"] * 4 + ["test_drift"] * 10
    assert [w["week_index"] for w in weeks] == list(range(14))
    assert all(w["n"] == 600 for w in weeks), "every week is cut to the same size"
    # a fixed seed gives the same draw
    c2 = D.weekly_curves({"test_id": test, "test_drift": late}, val, week_cap=600)
    assert c == c2


def test_a_stream_that_gets_harder_shows_falling_fixed_k_accuracy_and_rising_mean_k() -> None:
    # In this synthetic stream the model becomes confident more slowly as time passes, so the
    # fixed-K accuracy falls and a fixed threshold has to read more packets; the stopping rule
    # pays in packets, not in accuracy.
    val, test, late = streams(1.0)
    c = D.weekly_curves({"test_id": test, "test_drift": late}, val, week_cap=None)
    s = D.decay_summary(c)
    assert s["acc_k8"]["change"] < -0.02 and s["acc_k8"]["slope_per_week"] < 0
    assert s["policy_mean_k"]["change"] > 0.5 and s["policy_mean_k"]["slope_per_week"] > 0


def test_a_stationary_stream_shows_no_decay() -> None:
    val, test, late = streams(0.0)
    c = D.weekly_curves({"test_id": test, "test_drift": late}, val, week_cap=None)
    s = D.decay_summary(c)
    assert abs(s["policy_accuracy"]["change"]) < 0.03
    assert abs(s["policy_mean_k"]["change"]) < 0.5


def test_threshold_is_calibrated_on_validation_not_on_the_weeks() -> None:
    val, test, late = streams(1.0)
    c = D.weekly_curves({"test_drift": late}, val, week_cap=None)
    assert abs(c["target_mean_k"] - 6.0) < 1e-9
    # calibrated on val: the val mean K is the target, the drifted weeks' is not
    from adl_etc.evaluation import replay as R

    known = R.closed_world(val)
    r = R.safe_stop(
        known.p_safe,
        np.zeros((len(known), 30)),
        known.pred,
        known.ppi_len,
        c["theta_commit"],
        R.NO_REJECTION,
    )
    assert abs(float(r.k.mean()) - 6.0) < 0.1


def test_unknown_auroc_is_reported_only_with_an_unknown_model() -> None:
    val, test = make_set(30, 500, seed=1), make_set(30, 500, seed=2)
    um = fitted_unknown(val)
    with_um = D.weekly_curves({"test_id": test}, val, um, week_cap=None)
    without = D.weekly_curves({"test_id": test}, val, None, week_cap=None)
    assert "unknown_auroc_k8" in with_um["weeks"][0]
    assert "unknown_auroc_k8" not in without["weeks"][0]


def test_early_minus_late_compares_the_smallest_and_largest_k() -> None:
    val, test, late = streams(1.0)
    c = D.weekly_curves({"test_id": test, "test_drift": late}, val, week_cap=None)
    s = D.decay_summary(c)
    expected = s["acc_k5"]["change"] - s["acc_k30"]["change"]
    assert abs(s["early_minus_late"] - expected) < 1e-12
