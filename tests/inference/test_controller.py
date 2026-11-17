"""Spec 011's budget controller (plan phase 4, T5)."""

from __future__ import annotations

import numpy as np
import pytest

from adl_etc.inference.controller import BudgetController, logit, sigmoid

LO, HI = logit(0.5), logit(0.999)


def plant(theta: float, rng: np.random.Generator, n: int = 2000, noise: float = 0.0) -> np.ndarray:
    """A synthetic stream whose true mean K rises smoothly with theta: 2 at 0.5, 30 at 0.999."""
    mean = 2.0 + 28.0 * (logit(theta) - LO) / (HI - LO)
    return np.clip(rng.normal(mean, 0.5, n) * (1 + noise), 1, 30)


def run(ctrl: BudgetController, windows: int, *, steps=None, seed: int = 0):
    rng = np.random.default_rng(seed)
    steps = steps or {}
    errors = []
    for w in range(windows):
        if w in steps:
            ctrl.set_target(steps[w])
        k = plant(ctrl.theta, rng)
        ctrl.update(k)
        errors.append(abs(float(np.mean(k)) - ctrl.target) / ctrl.target)
    return errors


def test_converges_to_the_target_within_ten_windows_on_a_known_plant() -> None:
    for target in (4.0, 6.0, 8.0, 12.0):
        ctrl = BudgetController(target, theta_init=0.95)
        errors = run(ctrl, 25)
        assert max(errors[10:]) < 0.03, (target, [round(e, 3) for e in errors])


def test_a_step_change_of_the_target_settles_quickly_without_oscillating() -> None:
    ctrl = BudgetController(8.0, theta_init=0.9)
    errors = run(ctrl, 60, steps={20: 5.0, 40: 8.0})
    assert max(errors[15:20]) < 0.03
    assert max(errors[25:40]) < 0.05  # settled within 5 windows of the first step
    assert max(errors[45:60]) < 0.05
    assert not any("oscillation" in e.reason for e in ctrl.events)


def test_the_integral_clip_matters_a_big_first_error_must_not_wind_the_integral_up() -> None:
    # spec 011's text (kp 0.5, ki 0.05, i_max 5) misses its own acceptance rule on a known plant
    good = BudgetController(4.0, theta_init=0.95)
    bad = BudgetController(4.0, theta_init=0.95, kp=0.5, ki=0.05, i_max=5.0)
    assert max(run(good, 30)[10:]) < 0.03
    assert max(run(bad, 30)[10:]) > 0.1


def test_an_infeasible_target_saturates_with_one_event_and_a_bounded_integral() -> None:
    ctrl = BudgetController(1.2, theta_init=0.9, i_max=5.0)  # the plant never goes below ~2
    run(ctrl, 40)
    assert ctrl.theta == pytest.approx(0.5)
    assert abs(ctrl.integral) <= 5.0
    assert sum(e.reason.startswith("infeasible") for e in ctrl.events) == 1


def test_the_theta_stays_inside_its_rails_for_any_error() -> None:
    ctrl = BudgetController(6.0)
    for w in range(30):
        ctrl.update(np.full(500, 30.0 if w % 2 else 1.0))
        assert 0.5 <= ctrl.theta <= 0.999


def test_a_small_window_changes_nothing() -> None:
    ctrl = BudgetController(6.0, min_flows=200)
    before = (ctrl.theta, ctrl.integral)
    ev = ctrl.update(np.full(199, 30.0))
    assert ev.reason == "frozen" and (ctrl.theta, ctrl.integral) == before


def test_over_budget_lowers_theta_and_under_budget_raises_it() -> None:
    hi = BudgetController(6.0, theta_init=0.9)
    hi.update(np.full(500, 12.0))
    assert hi.theta < 0.9
    lo = BudgetController(6.0, theta_init=0.9)
    lo.update(np.full(500, 3.0))
    assert lo.theta > 0.9


def test_state_round_trip_gives_the_same_trajectory() -> None:
    a = BudgetController(7.0, theta_init=0.8)
    run(a, 10, seed=1)
    state = a.state_dict()
    b = BudgetController(7.0, theta_init=0.5)  # different start: the state must override it
    b.load_state_dict(state)
    rng_a, rng_b = np.random.default_rng(99), np.random.default_rng(99)
    for _ in range(8):
        a.update(plant(a.theta, rng_a))
        b.update(plant(b.theta, rng_b))
        assert a.theta == pytest.approx(b.theta, abs=1e-12)
    with pytest.raises(ValueError, match="metric"):
        BudgetController(7.0, metric="p95_k").load_state_dict(state)


def test_oscillation_guard_halves_the_gains() -> None:
    ctrl = BudgetController(6.0, kp=0.01, ki=0.0)  # too slow to matter: the errors are forced
    kp0 = ctrl.theta_knob.kp
    for w in range(8):  # an adversarial stream whose error flips sign by more than 5% each window
        ctrl.update(np.full(500, 9.0 if w % 2 == 0 else 3.0))
    assert ctrl.theta_knob.kp < kp0
    assert any("oscillation" in e.reason for e in ctrl.events)


def test_p95_mode_moves_the_cap_not_theta() -> None:
    ctrl = BudgetController(10.0, metric="p95_k", theta_init=0.9)
    for _ in range(10):
        ctrl.update(np.full(500, 25.0))
    assert ctrl.k_cap == 20 and ctrl.theta == 0.9
    for _ in range(40):
        ctrl.update(np.full(500, 4.0))
    assert ctrl.k_cap == 30  # recovers when under target


def test_rejection_cap_lowers_f_when_over_and_raises_it_when_under() -> None:
    over = BudgetController(6.0, reject_cap=0.05, f_init=0.05)
    over.update(np.full(500, 6.0), rejected=np.arange(500) % 5 == 0)  # 20% > 5%
    assert over.f < 0.05
    under = BudgetController(6.0, reject_cap=0.05, f_init=0.05)
    under.update(np.full(500, 6.0), rejected=np.zeros(500, bool))
    assert under.f > 0.05


def test_sigmoid_logit_are_inverse_and_stable() -> None:
    for p in (0.001, 0.3, 0.5, 0.999):
        assert sigmoid(logit(p)) == pytest.approx(p)
    assert sigmoid(-800.0) == 0.0 and sigmoid(800.0) == 1.0
