"""Controller replay over saved scores (plan phase 4, T5)."""

from __future__ import annotations

import numpy as np
import pytest

from adl_etc.evaluation import controller_replay as CR
from adl_etc.evaluation.scores import K_MAX, ScoreArrays, fill_forward
from adl_etc.inference.controller import BudgetController


def drifting_scores(
    n_windows: int = 60,
    window: int = 2000,
    seed: int = 0,
    drift: float = 0.0,
    unknown_share: float = 0.1,
) -> ScoreArrays:
    """A stream whose flows get harder over time: p_safe grows more slowly with K as ``drift``
    rises through the stream, so a fixed threshold reads more packets later on."""
    rng = np.random.default_rng(seed)
    n = n_windows * window
    t = np.repeat(np.linspace(0.0, 1.0, n_windows), window)  # 0 -> 1 over the stream
    ppi_len = rng.integers(8, 45, n)
    labels = np.where(rng.random(n) < unknown_share, -1, rng.integers(0, 20, n))
    speed = rng.uniform(0.15, 0.6, n) * (1.0 - drift * t * 0.6)  # slower learners later
    ks = np.arange(1, K_MAX + 1)[None, :]
    p_safe = 1 - np.exp(-ks * speed[:, None]) + rng.normal(0, 0.02, (n, K_MAX))
    p_safe = np.clip(p_safe, 0, 1)
    pred = np.where(rng.random((n, K_MAX)) < p_safe, labels[:, None], (labels[:, None] + 1) % 20)
    f = lambda a, dt: fill_forward(a, ppi_len).astype(dt)  # noqa: E731
    z = np.zeros((n, K_MAX))
    return ScoreArrays(
        pred=f(pred, np.int16),
        conf=f(p_safe, np.float16),
        margin=f(p_safe, np.float16),
        p_safe=f(p_safe, np.float16),
        energy=f(z, np.float32),
        energy_raw=f(z, np.float32),
        mahal=f(z, np.float32),
        pkt_nll=f(z, np.float32),
        next_entropy=f(z, np.float16),
        labels=labels,
        ppi_len=ppi_len,
        flow_index=np.arange(n),
        period=(np.arange(n) // window).astype(np.int16),
    )


def test_stream_order_is_by_period_then_flow_index_within_each_part() -> None:
    a = drifting_scores(3, 100, seed=1)
    b = drifting_scores(2, 100, seed=2)
    # shuffle part a's rows; the order must still come out time-sorted
    perm = np.random.default_rng(0).permutation(len(a))
    a_shuf = a.take(perm)
    scores, order, spans = CR.stream_of([a_shuf, b])
    assert spans == [(0, 300), (300, 500)]
    first = scores.flow_index[order[:300]]
    assert (np.diff(first) > 0).all()  # ascending within the part (periods ascend with the index)
    assert (order[300:] >= 300).all() and sorted(order.tolist()) == list(range(500))


def test_calibrate_theta_hits_the_target_and_clamps_an_unreachable_one() -> None:
    s = drifting_scores(10)
    order = np.arange(len(s))
    for target in (4.0, 6.0, 9.0):
        theta, ok, got = CR.calibrate_theta(s, order, target)
        assert ok and abs(got - target) <= 0.01
    theta, ok, got = CR.calibrate_theta(s, order, 0.5)  # below what k_min = 2 allows
    assert not ok and theta == pytest.approx(0.5)
    theta, ok, got = CR.calibrate_theta(s, order, 29.9)
    assert not ok and theta == pytest.approx(0.999)


def test_controller_holds_the_budget_while_a_static_threshold_drifts() -> None:
    s = drifting_scores(80, drift=1.0, seed=3)
    order = np.arange(len(s))
    theta0, ok, _ = CR.calibrate_theta(s, order[: 10 * 2000], 6.0)
    assert ok
    ctrl = BudgetController(6.0, theta_init=theta0)
    stat = BudgetController(6.0, theta_init=theta0)
    controlled = CR.replay(s, order, ctrl)
    static = CR.replay(s, order, stat, static=True)
    c = CR.tracking_summary(controlled)
    st = CR.tracking_summary(static)
    assert st["mean_abs_error"] > 0.15  # the fixed threshold really did drift off budget
    assert c["mean_abs_error"] < 0.05
    assert c["oscillation_events"] == 0


def test_step_schedule_is_tracked_and_settling_is_reported() -> None:
    s = drifting_scores(70, seed=4)
    order = np.arange(len(s))
    theta0, _, _ = CR.calibrate_theta(s, order[: 10 * 2000], 8.0)
    # kp = 1.5 is a "tuned" gain for this synthetic stream: settling time falls with kp (the default
    # 1.0 takes 3 and 6 windows here; kp = 2.0 takes 1-2). On real data the gain is chosen on val.
    ctrl = BudgetController(8.0, theta_init=theta0, kp=1.5)
    trace = CR.replay(s, order, ctrl, schedule={25: 5.0, 50: 8.0})
    summ = CR.tracking_summary(trace)
    assert len(summ["settling_windows"]) == 2
    assert all(0 <= w <= 5 for w in summ["settling_windows"]), summ["settling_windows"]
    slow = BudgetController(8.0, theta_init=theta0, kp=0.5)  # too timid: the criterion is missed
    s2 = CR.tracking_summary(CR.replay(s, order, slow, schedule={25: 5.0, 50: 8.0}))
    assert max(s2["settling_windows"]) > 5


def _trace(mean_ks: list[float], targets: list[float]) -> CR.ReplayTrace:
    return CR.ReplayTrace(
        [
            {"window": i, "n": 2000, "mean_k": m, "target": t}
            for i, (m, t) in enumerate(zip(mean_ks, targets, strict=True))
        ],
        [],
    )


def test_smoothed_settling_ignores_noise_that_breaks_the_strict_count() -> None:
    # target 8 -> 5 at window 10. The mean sits on 5 from window 11 with +-6% noise that
    # alternates: no 3 windows in a row are each within 5%, but the 3-window average always is.
    ks = [8.0] * 10 + [8.4] + [5.3, 4.7, 5.3, 4.7, 5.3, 4.7, 5.3, 4.7, 5.3, 4.7]
    tr = _trace(ks, [8.0] * 10 + [5.0] * 11)
    s = CR.tracking_summary(tr, warmup=0)
    assert s["settling_windows"] == [-1]  # strict: never "settled"
    assert s["settling_windows_smoothed"] == [1]


def test_smoothed_settling_still_catches_a_slow_or_biased_controller() -> None:
    slow = [8.0] * 10 + [8.0, 7.5, 7.0, 6.6, 6.2, 5.9, 5.6, 5.4, 5.2, 5.1, 5.0, 5.0]
    s = CR.tracking_summary(_trace(slow, [8.0] * 10 + [5.0] * 12), warmup=0)
    assert s["settling_windows_smoothed"][0] > 5
    biased = [8.0] * 10 + [6.5] * 12  # stuck 30% above the new target
    s = CR.tracking_summary(_trace(biased, [8.0] * 10 + [5.0] * 12), warmup=0)
    assert s["settling_windows_smoothed"] == [-1]


def test_knobs_change_only_between_windows() -> None:
    s = drifting_scores(20, seed=5)
    ctrl = BudgetController(6.0, theta_init=0.9)
    trace = CR.replay(s, np.arange(len(s)), ctrl)
    thetas = trace.column("theta")
    # the theta recorded for window w is the one in force when it opened: window 0 has the initial
    assert thetas[0] == pytest.approx(0.9)
    assert (np.diff(thetas) != 0).any()


def test_unknown_burst_raises_the_share_and_keeps_the_stretch_length() -> None:
    s = drifting_scores(20, seed=6)
    order = np.arange(len(s))
    rng = np.random.default_rng(0)
    lo, hi = 5 * 2000, 15 * 2000
    out = CR.unknown_burst(order, s, 0.3, lo, hi, rng)
    assert (
        len(out) == len(order) and (out[:lo] == order[:lo]).all() and (out[hi:] == order[hi:]).all()
    )
    before = (s.labels[order[lo:hi]] < 0).mean()
    after = (s.labels[out[lo:hi]] < 0).mean()
    assert before < 0.15 and 0.27 < after < 0.33
    # every flow of the stretch came from the stretch itself
    assert set(out[lo:hi]) <= set(order[lo:hi])


def test_hard_class_doubling_moves_the_share_of_hard_classes() -> None:
    s = drifting_scores(20, seed=7)
    hard = np.zeros(20, bool)
    hard[:5] = True
    order = np.arange(len(s))
    rng = np.random.default_rng(1)
    lo, hi = 4 * 2000, 14 * 2000
    out = CR.double_hard_classes(order, s, hard, lo, hi, rng)

    def share(idx):
        lab = s.labels[idx]
        return float(((lab >= 0) & hard[np.maximum(lab, 0)]).mean())

    assert share(out[lo:hi]) == pytest.approx(
        share(order[lo:hi]) * 2 / (1 + share(order[lo:hi])), abs=0.02
    )
    assert (out[:lo] == order[:lo]).all()


def test_hard_class_mask_picks_the_lowest_accuracy_classes() -> None:
    s = drifting_scores(6, seed=8)
    # make classes 0-4 always wrong at K = 5
    pred = s.pred.copy()
    for c in range(5):
        pred[s.labels == c, 4] = (c + 1) % 20
    from dataclasses import replace

    mask = CR.hard_class_mask(replace(s, pred=pred))
    assert mask[:5].all() and mask.sum() <= 6


def test_rejection_runs_when_an_unknown_model_is_given() -> None:
    s = drifting_scores(12, seed=9)
    u = np.where(s.labels[:, None] < 0, 5.0, -5.0) * np.ones((len(s), K_MAX))

    class Stub:
        def score_of(self, x):
            return u

        def theta_reject(self, f):
            return np.zeros(K_MAX)

    ctrl = BudgetController(6.0, theta_init=0.9)
    trace = CR.replay(s, np.arange(len(s)), ctrl, unknown=Stub(), static=True)  # type: ignore[arg-type]
    assert np.nanmean(trace.column("unknown_reject_rate")) == pytest.approx(1.0)
    assert np.nanmean(trace.column("known_reject_rate")) == pytest.approx(0.0)
