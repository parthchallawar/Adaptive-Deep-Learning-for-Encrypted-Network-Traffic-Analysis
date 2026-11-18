"""Spec 012's drift monitor (plan phase 5, T2): CUSUM, window signals, levels."""

from __future__ import annotations

import numpy as np
import pytest

from adl_etc.inference import drift as DR
from adl_etc.inference.policy import COMMIT, FORCED_COMMIT, FORCED_REJECT, REJECT


def test_cusum_detects_a_two_sigma_step_within_five_windows() -> None:
    rng = np.random.default_rng(0)
    c = DR.Cusum(h=5.0, k=0.5)
    hit = [i for i, x in enumerate(rng.normal(2.0, 1, 20)) if c.update(float(x))]
    assert hit and hit[0] < 5, hit


def test_cusum_is_two_sided() -> None:
    c = DR.Cusum(h=5.0, k=0.5)
    assert any(c.update(-2.5) for _ in range(5))


def test_cusum_is_quiet_on_stationary_noise_at_a_validation_tuned_h() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, 500)
    c = DR.Cusum(h=8.0, k=0.5)  # h = 5 has an average run length of ~465 windows per signal
    assert not any(c.update(float(x)) for x in noise)


def test_cusum_resets_after_an_alarm_and_ignores_nan() -> None:
    c = DR.Cusum(h=2.0, k=0.0)
    assert not c.update(float("nan"))
    assert c.update(3.0) and c.pos == 0.0 and c.neg == 0.0
    with pytest.raises(ValueError):
        DR.Cusum(h=0.0)


def test_prior_kl_is_zero_for_identical_histograms_and_positive_otherwise() -> None:
    h = np.array([10.0, 20.0, 30.0, 0.0])
    assert DR.prior_kl(h, h) == pytest.approx(0.0, abs=1e-12)
    assert DR.prior_kl(h * 7, h) == pytest.approx(0.0, abs=1e-12)  # only the shape matters
    other = np.array([30.0, 20.0, 10.0, 5.0])
    assert DR.prior_kl(other, h) > 0.05
    assert np.isfinite(DR.prior_kl(np.array([0.0, 0.0, 0.0, 9.0]), h))  # absent class is finite
    assert np.isnan(DR.prior_kl(np.zeros(4), h))


def test_prior_kl_by_hand_with_smoothing() -> None:
    p = np.array([1.0, 0.0])
    q = np.array([0.5, 0.5])
    ps = np.array([1.0 + 1e-3, 1e-3])
    ps /= ps.sum()
    qs = np.array([0.501, 0.501])
    qs /= qs.sum()
    assert DR.prior_kl(p, q) == pytest.approx(float(np.sum(ps * np.log(ps / qs))))


def test_window_stats_by_hand() -> None:
    k = np.array([2, 4, 6, 8])
    decision = np.array([COMMIT, REJECT, FORCED_COMMIT, FORCED_REJECT])
    prediction = np.array([0, -1, 1, -1])
    p = np.array([0.9, 0.2, 0.5, 0.1])
    nll = np.array([1.0, 2.0, 3.0, 4.0])
    ref = np.array([5.0, 5.0])
    s = DR.window_stats(k, decision, prediction, p, nll, ref, theta_commit=0.9)
    assert s["mean_k"] == 5.0
    assert s["commit_rate"] == 0.25  # only the first was decided by the rule as a commit
    assert s["reject_rate"] == 0.5  # REJECT and FORCED_REJECT
    assert s["mean_safe_at_commit"] == pytest.approx(0.7)  # flows 0 and 2 were committed
    assert s["mean_pkt_nll"] == 2.5 and s["theta_commit"] == 0.9
    assert s["prior_kl"] == pytest.approx(0.0, abs=1e-12)  # one flow in each class
    assert "mean_pkt_nll" not in DR.window_stats(k, decision, prediction, p, None, ref)


def test_window_stats_with_no_committed_flow_gives_nan_not_a_crash() -> None:
    d = np.array([REJECT, REJECT])
    s = DR.window_stats(
        np.array([3, 3]), d, np.array([-1, -1]), np.array([0.5, 0.5]), None, np.ones(3)
    )
    assert np.isnan(s["mean_safe_at_commit"]) and np.isnan(s["prior_kl"])


def test_monitor_warm_up_suppresses_the_first_windows() -> None:
    mon = DR.DriftMonitor({"a": (0.0, 1.0)}, h=2.0, k=0.0, warmup=5)
    events = [mon.update({"a": 10.0}) for _ in range(5)]
    assert all(e.level == "none" for e in events)
    assert mon.update({"a": 10.0}).level == "warning"


def test_monitor_levels_one_signal_is_a_warning_two_within_three_windows_an_alarm() -> None:
    ref = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
    mon = DR.DriftMonitor(ref, h=2.0, k=0.0, warmup=0)
    assert mon.update({"a": 0.0, "b": 0.0}).level == "none"
    e = mon.update({"a": 5.0, "b": 0.0})
    assert e.level == "warning" and e.alarms == ("a",)
    mon.update({"a": 0.0, "b": 0.0})
    e = mon.update({"a": 0.0, "b": 5.0})  # b alarms two windows after a: still inside the span
    assert e.level == "alarm" and set(DR.summary(mon.events)["first_alarm"]["signals"]) == {"b"}


def test_monitor_alarm_expires_when_the_second_signal_is_too_late() -> None:
    ref = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
    mon = DR.DriftMonitor(ref, h=2.0, k=0.0, warmup=0)
    mon.update({"a": 5.0, "b": 0.0})
    for _ in range(3):
        mon.update({"a": 0.0, "b": 0.0})
    assert mon.update({"a": 0.0, "b": 5.0}).level == "warning"


def test_monitor_rejects_a_signal_without_reference_and_uses_a_std_floor() -> None:
    with pytest.raises(KeyError):
        DR.DriftMonitor({"a": (0.0, 1.0)}, signals=["a", "b"])
    mon = DR.DriftMonitor({"a": (1.0, 0.0)}, h=1.0, k=0.0, warmup=0)
    assert np.isfinite(mon.z("a", 2.0))


def test_tune_h_raises_h_until_validation_is_quiet() -> None:
    rng = np.random.default_rng(3)
    val = [{"a": float(x)} for x in rng.normal(0.0, 1.0, 120)]
    ref = DR.reference_from_windows(val, ["a"])
    h = DR.tune_h(ref, val, ["a"])
    assert h >= 5.0
    mon = DR.DriftMonitor(ref, h=h)
    assert not any(mon.update(w).alarms for w in val)
    # a deliberately small start proves it searches upwards
    h2 = DR.tune_h(ref, val, ["a"], h_start=0.5, h_step=0.5)
    assert h2 <= h


def test_reference_needs_two_windows() -> None:
    with pytest.raises(ValueError):
        DR.reference_from_windows([{"a": 1.0}], ["a"])


def test_a_gradual_shift_in_a_signal_eventually_alarms() -> None:
    rng = np.random.default_rng(4)
    val = [{"mean_k": float(x)} for x in rng.normal(6.0, 0.1, 60)]
    ref = DR.reference_from_windows(val, ["mean_k"])
    h = DR.tune_h(ref, val, ["mean_k"])
    mon = DR.DriftMonitor(ref, h=h)
    stream = [6.0 + 0.02 * i + rng.normal(0, 0.1) for i in range(120)]
    for x in stream:
        mon.update({"mean_k": float(x)})
    assert DR.summary(mon.events)["first_warning"] is not None
