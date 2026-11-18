"""Replaying the drift monitor over saved scores (plan phase 5, T2)."""

from __future__ import annotations

import numpy as np

from adl_etc.evaluation import drift_curves as DC
from adl_etc.evaluation import drift_monitor as DM
from tests.evaluation.synth import fitted_unknown, make_set
from tests.evaluation.test_controller_replay import drifting_scores


def test_spearman_by_hand() -> None:
    assert DM.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert DM.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert np.isnan(DM.spearman([1, 1, 1], [1, 2, 3]))  # constant input
    assert np.isnan(DM.spearman([1, 2], [1, 2]))  # too few pairs
    assert abs(DM.spearman([1, 2, 2, 3], [1, 2, 3, 4]) - 0.9486832980505138) < 1e-9  # ties


def streams(drift: float):
    val = drifting_scores(30, 1000, seed=1, drift=0.0)
    test = drifting_scores(8, 1000, seed=2, drift=0.0)
    late = drifting_scores(40, 1000, seed=3, drift=drift)
    return {"val": val, "test_id": test, "test_drift": late}


def test_windows_are_complete_and_tagged_with_their_week() -> None:
    s = streams(0.0)
    ref = np.ones(20)
    w = DM.replay_windows(
        s["test_id"], um=None, reference_hist=ref, theta=0.9, window=2000, split="test_id"
    )
    assert len(w) == 4 and all(x["split"] == "test_id" and x["n"] == 2000 for x in w)
    assert {"mean_k", "commit_rate", "reject_rate", "mean_safe_at_commit", "prior_kl"} <= set(w[0])
    assert "mean_pkt_nll" not in w[0] and "theta_commit" not in w[0]  # no model head, no controller


def test_a_ragged_last_window_is_dropped() -> None:
    s = drifting_scores(1, 2300, seed=1)  # one window of 2300 flows, window 2000: 1 full + 300
    w = DM.replay_windows(s, um=None, reference_hist=np.ones(20), theta=0.9, window=2000)
    assert len(w) == 1


def test_controller_mode_records_its_threshold_and_moves_it() -> None:
    s = streams(1.0)
    from adl_etc.inference.controller import BudgetController

    ctrl = BudgetController(6.0, theta_init=0.9, kp=3.0, ki=0.02)
    w = DM.replay_windows(
        s["test_drift"],
        um=None,
        reference_hist=np.ones(20),
        theta=0.9,
        controller=ctrl,
        window=2000,
    )
    th = [x["theta_commit"] for x in w]
    assert len(set(th)) > 1 and th[0] == 0.9


def test_a_drifting_stream_alarms_and_a_stationary_one_does_not() -> None:
    for drift, expect_alarm in ((1.0, True), (0.0, False)):
        s = streams(drift)
        curves = DC.weekly_curves(
            {k: s[k] for k in ("test_id", "test_drift")}, s["val"], week_cap=None
        )
        rep = DM.monitor_report(s, None, curves, window=2000)
        m = rep["static"]
        assert m["val_alarms"] == 0
        # one chance warning in 48 windows is what a CUSUM at a validation-tuned h allows; the
        # combined "alarm" level (two signals within three windows) must stay quiet when nothing
        # changes, and the stream that gets harder must at least warn
        if expect_alarm:
            assert m["summary"]["first_warning"] is not None, m["summary"]
        else:
            assert m["summary"]["n_alarm_windows"] == 0, m["summary"]
        if expect_alarm:
            assert "mean_k" in m["per_signal_first_alarm"]
            assert m["per_signal_first_alarm"]["mean_k"] is not None


def test_report_has_both_modes_with_the_right_signal_sets() -> None:
    s = streams(1.0)
    curves = DC.weekly_curves({k: s[k] for k in ("test_id", "test_drift")}, s["val"], week_cap=None)
    rep = DM.monitor_report(s, None, curves, window=2000)
    assert "mean_k" in rep["static"]["signals"] and "theta_commit" not in rep["static"]["signals"]
    assert (
        "theta_commit" in rep["controller"]["signals"]
        and "mean_k" not in rep["controller"]["signals"]
    )
    assert set(rep["static"]["correlation"]) == set(rep["static"]["signals"])


def test_unknown_model_adds_the_reject_and_nll_signals() -> None:
    val, test = make_set(40, 500, seed=1), make_set(40, 500, seed=2)
    um = fitted_unknown(val)
    sig = DM._signals("static", um)
    assert "reject_rate" in sig and "mean_pkt_nll" in sig
    assert "reject_rate" not in DM._signals("static", None)
    w = DM.replay_windows(test, um=um, reference_hist=np.ones(12), theta=0.9, window=1000)
    assert "mean_pkt_nll" in w[0] and w[0]["reject_rate"] >= 0.0
