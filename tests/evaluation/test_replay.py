"""Stopping-policy replay: Safe-Stop, score-based ECHO/CAPE, P-RL (plan phase 4, T3)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from adl_etc.evaluation import metrics as M
from adl_etc.evaluation import policies as PL
from adl_etc.evaluation import replay as R
from adl_etc.evaluation import rl_stop
from adl_etc.evaluation.dense_logits import from_causal
from adl_etc.evaluation.scores import K_MAX, ScoreArrays, class_scores, fill_forward
from adl_etc.inference import policy as P

NO_REJECT = np.full(K_MAX, np.inf)


def make_cols(n: int = 3000, seed: int = 0, nan: bool = False):
    rng = np.random.default_rng(seed)
    ppi_len = rng.integers(1, 45, n)
    p_safe = rng.random((n, K_MAX))
    u = rng.normal(size=(n, K_MAX))
    pred = rng.integers(0, 150, (n, K_MAX))
    if nan:
        p_safe[rng.random((n, K_MAX)) < 0.02] = np.nan
        u[rng.random((n, K_MAX)) < 0.02] = np.nan
    return p_safe, u, pred, ppi_len, rng.normal(1.0, 0.5, K_MAX)


def per_flow(p_safe, u, pred, ppi_len, theta_commit, theta_reject, **kw):
    """The scalar rule applied flow by flow: the reference the vectorised form must equal."""
    cap = kw.pop("k_cap", K_MAX)
    n = len(ppi_len)
    k, code, cls = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    for i in range(n):
        last = min(cap, int(ppi_len[i]))
        for kk in range(1, last + 1):
            d = P.decide(
                kk,
                p_safe[i, kk - 1],
                u[i, kk - 1],
                pred[i, kk - 1],
                theta_commit=theta_commit,
                theta_reject_k=theta_reject[kk - 1],
                last_k=last,
                **kw,
            )
            if d is not None:
                k[i], code[i], cls[i] = d.k, d.code, d.cls
                break
    return k, code, cls


@pytest.mark.parametrize("order", ["reject_first", "commit_first"])
@pytest.mark.parametrize("theta_commit", [0.5, 0.9, 0.999, 1.5])
@pytest.mark.parametrize("k_cap", [30, 7])
@pytest.mark.parametrize("k_min,k_ood", [(2, 3), (1, 1), (3, 10)])
def test_vectorised_safe_stop_equals_the_per_flow_rule(order, theta_commit, k_cap, k_min, k_ood):
    p_safe, u, pred, ppi_len, thr = make_cols(1200, seed=1, nan=True)
    got = R.safe_stop(
        p_safe,
        u,
        pred,
        ppi_len,
        theta_commit,
        thr,
        order=order,
        k_cap=k_cap,
        k_min=k_min,
        k_ood=k_ood,
    )
    k, code, cls = per_flow(
        p_safe,
        u,
        pred,
        ppi_len,
        theta_commit,
        thr,
        order=order,
        k_cap=k_cap,
        k_min=k_min,
        k_ood=k_ood,
    )
    np.testing.assert_array_equal(got.k, k)
    np.testing.assert_array_equal(got.decision, code)
    np.testing.assert_array_equal(got.prediction, cls)


def test_nan_scores_are_counted_and_force_a_rejection_at_the_end() -> None:
    p = np.full((1, K_MAX), np.nan)
    u = np.full((1, K_MAX), np.nan)
    r = R.safe_stop(p, u, np.zeros((1, K_MAX), int), np.array([6]), 0.9, np.full(K_MAX, 1.0))
    assert r.n_nan > 0
    # u = +inf >= any threshold from k_ood = 3 on, so the flow is rejected there
    assert r.k[0] == 3 and r.rejected[0]


def test_check_order_changes_the_outcome_when_both_fire() -> None:
    p = np.full((1, K_MAX), 0.99)
    u = np.full((1, K_MAX), 5.0)
    thr = np.full(K_MAX, 1.0)
    pred = np.full((1, K_MAX), 7)
    kw = {"k_min": 3, "k_ood": 3}  # both conditions first hold at K = 3
    a = R.safe_stop(p, u, pred, np.array([20]), 0.9, thr, order="reject_first", **kw)
    b = R.safe_stop(p, u, pred, np.array([20]), 0.9, thr, order="commit_first", **kw)
    assert (a.k[0], a.decision[0], a.prediction[0]) == (3, P.REJECT, -1)
    assert (b.k[0], b.decision[0], b.prediction[0]) == (3, P.COMMIT, 7)
    # with the defaults only commit is allowed at K = 2, so both orders commit there
    c = R.safe_stop(p, u, pred, np.array([20]), 0.9, thr, order="reject_first")
    assert (c.k[0], c.decision[0]) == (2, P.COMMIT)


def test_a_one_packet_flow_is_decided_at_k1_and_may_be_force_rejected() -> None:
    p = np.zeros((2, K_MAX))
    u = np.array([np.full(K_MAX, 9.0), np.full(K_MAX, -9.0)])
    r = R.safe_stop(p, u, np.ones((2, K_MAX), int), np.array([1, 1]), 0.9, np.full(K_MAX, 1.0))
    assert r.k.tolist() == [1, 1]
    assert r.decision.tolist() == [
        P.FORCED_REJECT,
        P.FORCED_COMMIT,
    ]  # the forced branch ignores k_ood


def test_raising_theta_commit_never_lowers_mean_k_or_raising_f_the_rejections() -> None:
    p_safe, u, pred, ppi_len, thr = make_cols(4000, seed=2)
    mk = [
        R.safe_stop(p_safe, u, pred, ppi_len, t, NO_REJECT).k.mean()
        for t in np.linspace(0.5, 0.999, 40)
    ]
    assert np.all(np.diff(mk) >= -1e-12)
    rej = [
        R.safe_stop(p_safe, u, pred, ppi_len, 0.9, np.full(K_MAX, th)).rejected.mean()
        for th in np.linspace(3.0, -3.0, 20)
    ]  # lowering the threshold = raising f
    assert np.all(np.diff(rej) >= -1e-12)


def test_summarize_by_hand() -> None:
    r = R.ReplayResult(
        k=np.array([2, 3, 4, 5]),
        decision=np.array([P.COMMIT, P.REJECT, P.COMMIT, P.FORCED_REJECT], dtype=np.int8),
        prediction=np.array([0, -1, 9, -1]),
    )
    labels = np.array([0, 1, -1, -1])  # flows 3,4 unknown
    s = R.summarize(r, labels)
    assert s["mean_k"] == pytest.approx(3.5)
    assert s["known_accuracy"] == pytest.approx(0.5)  # flow 1 right, flow 2 rejected
    assert s["known_reject_rate"] == pytest.approx(0.5)
    assert s["unknown_reject_rate"] == pytest.approx(0.5) and s[
        "unknown_committed"
    ] == pytest.approx(0.5)
    assert s["rejection_k"] == pytest.approx(5.0)
    assert s["forced_rate"] == pytest.approx(0.25)


# --- ECHO and CAPE equal the original policies ----------------------------------------------


def real_like_scores(n: int = 1500, c: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    ppi_len = rng.integers(1, 45, n)
    labels = rng.integers(0, c, n)
    skill = rng.uniform(0.1, 1.0, n)
    logits = rng.normal(0, 1, (n, K_MAX, c))
    for k in range(K_MAX):
        logits[np.arange(n), k, labels] += skill * (k + 1) * 0.35
    logits = (
        fill_forward(
            logits.reshape(n, K_MAX, -1).transpose(0, 2, 1).reshape(n * c, K_MAX),
            np.repeat(ppi_len, c),
        )
        .reshape(n, c, K_MAX)
        .transpose(0, 2, 1)
    )
    return logits.astype(np.float32), labels, ppi_len


def test_score_echo_equals_policies_p_echo_at_the_grid_end() -> None:
    logits, labels, ppi_len = real_like_scores()
    adj, temps = np.zeros(logits.shape[-1]), np.ones(K_MAX)
    cs = class_scores(logits, adj, temps)
    dense = from_causal(logits)
    for tau in (0.0, 0.3, 0.6, 0.9, 0.99, 1.0):
        want = PL.p_echo(dense, ppi_len, tau)
        got = R.echo(cs["conf"], cs["pred"], ppi_len, tau, force_at="grid_end")
        np.testing.assert_array_equal(got.k, want.k)
        np.testing.assert_array_equal(got.prediction, want.prediction)
        np.testing.assert_array_equal(got.committed, want.committed)


def test_score_cape_equals_policies_p_cape_at_the_grid_end() -> None:
    logits, labels, ppi_len = real_like_scores(seed=5)
    c = logits.shape[-1]
    cs = class_scores(logits, np.zeros(c), np.ones(K_MAX))
    dense = from_causal(logits)
    tau_c = np.random.default_rng(1).uniform(0.4, 0.95, c)
    for theta in (-1e9, -3.0, -1.0, 1e9):
        want = PL.p_cape(dense, ppi_len, tau_c, theta, energy_k=10)
        got = R.cape(
            cs["conf"],
            cs["pred"],
            cs["energy_raw"],
            ppi_len,
            tau_c,
            theta,
            energy_k=10,
            force_at="grid_end",
        )
        np.testing.assert_array_equal(got.k, want.k)
        np.testing.assert_array_equal(got.prediction, want.prediction)
        np.testing.assert_array_equal(got.committed, want.committed)


def test_echo_flow_end_reports_the_forced_decision_at_the_flows_own_last_packet() -> None:
    conf = np.full((2, K_MAX), 0.1)
    pred = np.zeros((2, K_MAX), int)
    r = R.echo(conf, pred, np.array([5, 40]), 0.99)
    assert r.k.tolist() == [5, 30]
    g = R.echo(conf, pred, np.array([5, 40]), 0.99, force_at="grid_end")
    assert g.k.tolist() == [30, 30]


# --- metrics --------------------------------------------------------------------------------


def test_acc_at_budget_and_pareto_area_by_hand() -> None:
    mk, acc = [3.0, 5.0, 8.0], [0.5, 0.7, 0.9]
    assert M.acc_at_budget(mk, acc, 2.0) == 0.0  # cannot operate that cheaply
    assert M.acc_at_budget(mk, acc, 5.0) == 0.7
    assert M.acc_at_budget(mk, acc, 100.0) == 0.9
    assert M.pareto_area([2.0], [1.0], lo=2.0, hi=12.0) == pytest.approx(1.0)
    assert M.pareto_area([12.0], [1.0], lo=2.0, hi=12.0, n=11) == pytest.approx(1 / 11)


def test_logit_grid_is_increasing_and_inside_the_bounds() -> None:
    g = R.logit_grid(50)
    assert np.all(np.diff(g) > 0) and g[0] == pytest.approx(0.5) and g[-1] == pytest.approx(0.999)


# --- P-RL -----------------------------------------------------------------------------------


def synthetic_scores(n: int = 30000, seed: int = 0) -> ScoreArrays:
    rng = np.random.default_rng(seed)
    ppi_len = rng.integers(1, 45, n)
    labels = rng.integers(0, 20, n)
    skill = rng.random(n)
    ks = np.arange(1, K_MAX + 1)[None, :]
    p_right = 1 - np.exp(-ks * (0.05 + 0.4 * skill[:, None]))
    conf = np.clip(p_right + rng.normal(0, 0.05, (n, K_MAX)), 0, 1)
    margin = np.clip(conf - 0.2 * rng.random((n, K_MAX)), 0, 1)
    right = rng.random((n, K_MAX)) < p_right
    pred = np.where(right, labels[:, None], (labels[:, None] + 1) % 20)
    f = lambda a, dt: fill_forward(a, ppi_len).astype(dt)  # noqa: E731
    z = np.zeros((n, K_MAX))
    return ScoreArrays(
        pred=f(pred, np.int16),
        conf=f(conf, np.float16),
        margin=f(margin, np.float16),
        p_safe=f(conf, np.float16),
        energy=f(z, np.float32),
        energy_raw=f(z, np.float32),
        mahal=f(z, np.float32),
        pkt_nll=f(z, np.float32),
        next_entropy=f(z, np.float16),
        labels=labels,
        ppi_len=ppi_len,
        flow_index=np.arange(n),
        period=np.zeros(n, np.int16),
    )


def test_p_rl_stops_earlier_as_a_packet_gets_more_expensive() -> None:
    s = synthetic_scores()
    mk, acc = [], []
    for lam in (0.0, 0.2, 0.5, 1.0, 3.0):
        r = rl_stop.apply_rl(rl_stop.fit_rl(s, lam), s)
        assert (r.k >= 1).all() and (r.k <= s.last_k).all()
        mk.append(r.k.mean())
        acc.append(float((r.prediction == s.labels).mean()))
    assert np.all(np.diff(mk) <= 1e-9)  # larger lam never raises mean K
    assert acc[0] > acc[-1]  # and the free-waiting end is the more accurate one


def test_p_rl_ignores_unknown_class_flows_when_fitting() -> None:
    s = synthetic_scores(8000, seed=3)
    noisy = np.where(np.arange(len(s)) % 5 == 0, -1, s.labels)
    only_known = s.take(noisy >= 0)
    a = rl_stop.fit_rl(replace(s, labels=noisy), 0.3)  # unknown-class flows present
    b = rl_stop.fit_rl(only_known, 0.3)  # the same fit without them
    np.testing.assert_array_equal(a.q_commit, b.q_commit)


def test_p_rl_two_step_toy_problem_by_hand() -> None:
    # Every flow is 2 packets long. K=1: conf bin A; K=2: bin B. Commit at 1 is right 50% of
    # the time; at 2 it is right 100%. A packet costs lam / 30. With lam = 0 waiting is free,
    # so Q_cont(1) = 1 > 0.5 and the policy waits. With lam = 3, Q_commit(1) = 0.4 and
    # Q_cont(1) = Q_commit(2) = 1 - 0.2 = 0.8: still wait. Committing at 1 wins only once
    # lam / 30 exceeds the 0.5 gained by waiting, e.g. lam = 30.
    n = 400
    conf = np.full((n, K_MAX), 0.5)
    conf[:, 1:] = 0.95
    margin = conf.copy()
    labels = np.zeros(n, int)
    pred = np.zeros((n, K_MAX), int)
    pred[: n // 2, 0] = 1  # half of the K=1 answers are wrong
    ppi_len = np.full(n, 2)
    z = np.zeros((n, K_MAX))
    s = ScoreArrays(
        pred=pred.astype(np.int16),
        conf=conf.astype(np.float16),
        margin=margin.astype(np.float16),
        p_safe=conf.astype(np.float16),
        energy=z.astype(np.float32),
        energy_raw=z.astype(np.float32),
        mahal=z.astype(np.float32),
        pkt_nll=z.astype(np.float32),
        next_entropy=z.astype(np.float16),
        labels=labels,
        ppi_len=ppi_len,
        flow_index=np.arange(n),
        period=np.zeros(n, np.int16),
    )
    free = rl_stop.apply_rl(rl_stop.fit_rl(s, 0.0), s)
    modest = rl_stop.apply_rl(rl_stop.fit_rl(s, 3.0), s)
    pricey = rl_stop.apply_rl(rl_stop.fit_rl(s, 30.0), s)
    assert (free.k == 2).all() and (modest.k == 2).all() and (pricey.k == 1).all()
