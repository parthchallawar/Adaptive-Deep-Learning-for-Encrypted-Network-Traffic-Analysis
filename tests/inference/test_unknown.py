"""Spec 010's unknown score (plan phase 4, T2)."""

from __future__ import annotations

import numpy as np
import pytest

from adl_etc.evaluation.scores import K_MAX, ScoreArrays, fill_forward
from adl_etc.inference import unknown as U

C, D = 6, 8


def synthetic_projections(n: int, seed: int, with_unknown: bool = True):
    """Flows whose projection is a class mean plus noise; unknown flows sit elsewhere."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 2.0, (C, D))
    labels = rng.integers(0, C, n)
    if with_unknown:
        labels = np.where(rng.random(n) < 0.15, -1, labels)
    ppi_len = rng.integers(1, 40, n)
    base = np.where(labels[:, None] >= 0, means[np.maximum(labels, 0)], rng.normal(0, 4.0, (n, D)))
    proj = base[:, None, :] + rng.normal(0, 0.5, (n, K_MAX, D))
    return means, labels, ppi_len, proj


def fit_bank(seed: int = 0, n: int = 6000, shrinkage: float = 0.1):
    _, labels, ppi_len, proj = synthetic_projections(n, seed, with_unknown=False)
    valid = U.class_validity(labels, ppi_len, C, min_class_flows=50)
    sums = U.GaussianSums(C, valid, dim=D)
    for s in range(0, n, 1000):
        sl = slice(s, s + 1000)
        sums.add(proj[sl], labels[sl], ppi_len[sl])
    return sums.bank(shrinkage), labels, ppi_len, proj, sums


def test_running_sums_reproduce_the_pooled_covariance() -> None:
    bank, labels, ppi_len, proj, _ = fit_bank(shrinkage=0.0)
    k = 4  # K = 5
    m = ppi_len >= k + 1
    x, y = proj[m, k], labels[m]
    pooled = sum(
        (x[y == c] - x[y == c].mean(0)).T @ (x[y == c] - x[y == c].mean(0)) for c in range(C)
    ) / (len(y) - C)
    np.testing.assert_allclose(np.linalg.inv(bank.precision[k]), pooled, rtol=1e-8, atol=1e-10)
    for c in range(C):
        np.testing.assert_allclose(bank.means[k, c], x[y == c].mean(0), atol=1e-10)


def test_shrinkage_pulls_the_covariance_toward_a_scaled_identity() -> None:
    plain = fit_bank(shrinkage=0.0)[0]
    shrunk = fit_bank(shrinkage=0.1)[0]
    for k in (0, 9, 24):
        cov = np.linalg.inv(plain.precision[k])
        want = 0.9 * cov + 0.1 * (np.trace(cov) / D) * np.eye(D)
        np.testing.assert_allclose(np.linalg.inv(shrunk.precision[k]), want, rtol=1e-8, atol=1e-10)
        # and the background Gaussian is shrunk the same way
        bg = np.linalg.inv(plain.bg_precision[k])
        np.testing.assert_allclose(
            np.linalg.inv(shrunk.bg_precision[k]),
            0.9 * bg + 0.1 * (np.trace(bg) / D) * np.eye(D),
            rtol=1e-8,
            atol=1e-10,
        )


def test_mahalanobis_matches_a_per_class_loop_and_is_zero_at_a_class_mean() -> None:
    bank, *_ = fit_bank()
    rng = np.random.default_rng(3)
    x = rng.normal(0, 3, (50, K_MAX, D))
    got = bank.mahal(x)
    for k in (0, 7, 29):
        cv = bank.valid[k]
        for i in range(0, 50, 7):
            d = [(x[i, k] - m) @ bank.precision[k] @ (x[i, k] - m) for m in bank.means[k][cv]]
            xb = x[i, k] - bank.bg_mean[k]
            want = min(d) - xb @ bank.bg_precision[k] @ xb
            assert got[i, k] == pytest.approx(want, rel=1e-9, abs=1e-9)
    # the distance term (before subtracting the background) is 0 at a class mean
    k, c = 5, int(np.flatnonzero(bank.valid[5])[0])
    mu = bank.means[k, c]
    xp = mu @ bank.precision[k]
    d0 = (
        (xp * mu).sum()
        - 2 * xp @ bank.means[k][bank.valid[k]].T
        + ((bank.means[k][bank.valid[k]] @ bank.precision[k]) * bank.means[k][bank.valid[k]]).sum(1)
    )
    assert d0.min() == pytest.approx(0.0, abs=1e-8)


def test_unknown_labels_cannot_reach_the_gaussian_fit() -> None:
    sums = U.GaussianSums(C, np.ones((K_MAX, C), bool), dim=D)
    with pytest.raises(U.UnknownModelError):
        sums.add(np.zeros((2, K_MAX, D)), np.array([0, -1]), np.array([5, 5]))


def test_rare_classes_get_no_gaussian() -> None:
    labels = np.array([0] * 100 + [1] * 10)
    ppi_len = np.full(110, 30)
    valid = U.class_validity(labels, ppi_len, 3, min_class_flows=50)
    assert valid[:, 0].all() and not valid[:, 1].any() and not valid[:, 2].any()


def make_val(n: int, seed: int, bank: U.GaussianBank, *, unknown_seed: int | None = None):
    """A validation ScoreArrays whose energy/mahal/surprise are computed from signals."""
    _, labels, ppi_len, proj = synthetic_projections(n, seed)
    rng = np.random.default_rng(seed + 100)
    energy = rng.normal(-5, 1, (n, K_MAX)) + 1.5 * (labels[:, None] < 0)
    surprise_nll = rng.random((n, K_MAX)) + 0.5 * (labels[:, None] < 0)
    mahal = bank.mahal(proj)
    if unknown_seed is not None:  # replace only the unknown-class flows' content
        r2 = np.random.default_rng(unknown_seed)
        u = labels < 0
        energy[u] = r2.normal(10, 5, (u.sum(), K_MAX))
        mahal[u] = r2.normal(50, 20, (u.sum(), K_MAX))
        surprise_nll[u] = r2.random((u.sum(), K_MAX)) * 9
    f = lambda a, dt: fill_forward(a, ppi_len).astype(dt)  # noqa: E731
    return ScoreArrays(
        pred=f(rng.integers(0, C, (n, K_MAX)), np.int16),
        conf=f(rng.random((n, K_MAX)), np.float16),
        margin=f(rng.random((n, K_MAX)), np.float16),
        p_safe=f(rng.random((n, K_MAX)), np.float16),
        energy=f(energy, np.float32),
        energy_raw=f(energy, np.float32),
        mahal=f(mahal, np.float32),
        pkt_nll=f(surprise_nll, np.float32),
        next_entropy=f(rng.random((n, K_MAX)), np.float16),
        labels=labels,
        ppi_len=ppi_len,
        flow_index=np.arange(n),
        period=np.zeros(n, np.int16),
        meta={},
    )


def test_threshold_gives_the_target_rate_on_the_fitting_flows() -> None:
    bank, *_ = fit_bank()
    val = make_val(8000, 1, bank)
    model = U.UnknownModel.fit(bank, val, checkpoint_hash="h")
    f = 0.02
    theta = model.theta_reject(f)
    u = model.score_of(val)
    for k in (1, 3, 10, 25):
        m = (val.labels >= 0) & (val.ppi_len >= k)
        rate = float((u[m, k - 1] >= theta[k - 1]).mean())
        assert abs(rate - f) < 0.004, (k, rate)


def test_unknown_validation_flows_cannot_change_the_fit() -> None:
    bank, *_ = fit_bank()
    a = U.UnknownModel.fit(bank, make_val(3000, 2, bank), checkpoint_hash="h")
    b = U.UnknownModel.fit(bank, make_val(3000, 2, bank, unknown_seed=9), checkpoint_hash="h")
    np.testing.assert_array_equal(a.z_mean, b.z_mean)
    np.testing.assert_array_equal(a.z_std, b.z_std)
    np.testing.assert_array_equal(a.quantiles, b.quantiles)


def test_unknown_flows_score_higher_and_energy_only_weights_work() -> None:
    bank, *_ = fit_bank()
    val = make_val(6000, 4, bank)
    model = U.UnknownModel.fit(bank, val, checkpoint_hash="h")
    u = model.score_of(val)
    unk = val.labels < 0
    assert u[unk, 9].mean() > u[~unk, 9].mean() + 1.0
    energy_only = U.UnknownModel.fit(bank, val, weights=(1.0, 0.0, 0.0), checkpoint_hash="h")
    z = energy_only.components(val.energy, val.mahal, val.surprise)
    np.testing.assert_allclose(energy_only.score_of(val), z[..., 0])


def test_constant_signal_contributes_zero() -> None:
    # surprise at K = 1 is 0 for every flow, so its std is 0 and its z must be 0, not NaN
    bank, *_ = fit_bank()
    val = make_val(2000, 5, bank)
    model = U.UnknownModel.fit(bank, val, checkpoint_hash="h")
    z = model.components(val.energy, val.mahal, val.surprise)
    assert np.isfinite(z[:, 0, 2]).all() and (z[:, 0, 2] == 0).all()


def test_save_load_round_trip_and_checkpoint_guard(tmp_path) -> None:
    bank, *_ = fit_bank()
    model = U.UnknownModel.fit(bank, make_val(2000, 6, bank), checkpoint_hash="hash-A")
    model.save(tmp_path / "unknown")
    again = U.UnknownModel.load(tmp_path / "unknown", checkpoint_hash="hash-A")
    np.testing.assert_array_equal(again.quantiles, model.quantiles)
    np.testing.assert_array_equal(again.bank.precision, model.bank.precision)
    assert again.weights == model.weights
    with pytest.raises(U.UnknownModelError, match="checkpoint"):
        U.UnknownModel.load(tmp_path / "unknown", checkpoint_hash="hash-B")


def test_rejection_k_by_hand() -> None:
    u = np.zeros((3, K_MAX))
    theta = np.full(K_MAX, 1.0)
    u[0, 4] = 2.0  # crosses at K = 5
    u[1, 1] = 2.0  # crosses at K = 2, before k_ood = 3: ignored
    u[2, 7] = 2.0  # crosses at K = 8, but the flow ends at K = 6
    got = U.rejection_k(u, theta, np.array([30, 30, 6]), k_ood=3)
    assert got.tolist() == [5, 0, 0]


def test_energy_gate_threshold_rejects_the_requested_fraction() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    energy = rng.normal(size=(n, K_MAX))
    labels = np.where(rng.random(n) < 0.2, -1, 0)
    ppi_len = np.full(n, 30)
    thr = U.energy_gate_threshold(energy, labels, ppi_len, 10, 0.05)
    known = labels >= 0
    assert abs((energy[known, 9] >= thr).mean() - 0.05) < 0.005
