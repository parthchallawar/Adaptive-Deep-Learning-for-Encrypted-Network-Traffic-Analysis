"""Per-K calibration (spec 008; plan phase 3, T7), checked against known answers:
the adjustment exactly undoes the balanced sampler's prior, temperature recovers
a known scale, PAV matches scikit-learn, Platt recovers known parameters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adl_etc.evaluation import calibration as CAL
from adl_etc.evaluation.metrics import ece

K = 30


def test_the_adjustment_is_minus_log_of_the_samplers_oversampling() -> None:
    from adl_etc.training.datasets import balanced_weights

    rng = np.random.default_rng(0)
    y = rng.choice(6, size=5000, p=[0.6, 0.25, 0.1, 0.04, 0.009, 0.001])
    counts = np.bincount(y, minlength=7)  # class 6 absent

    adj = CAL.prior_adjustment(counts, cap=10.0)

    per_flow = balanced_weights(y, 7, 10.0)
    factor = np.array([per_flow[y == c][0] for c in range(6)])  # ∝ the oversampling factor
    np.testing.assert_allclose(adj[:6] - (-np.log(factor)), (adj[0] + np.log(factor[0])))
    assert adj[6] == 0.0  # nothing to undo for a class never trained on
    assert (CAL.prior_adjustment(counts, cap=None) == 0).all()


def test_adjusted_logits_give_the_natural_posterior_exactly() -> None:
    """A model trained on balanced samples outputs log p(x|y) + log pi_samp(y) (up
    to a constant); adjusting must give the natural posterior."""
    rng = np.random.default_rng(1)
    counts = np.array([5000, 800, 120, 30])
    log_lik = rng.normal(size=(200, 4))
    adj = CAL.prior_adjustment(counts, cap=10.0)
    log_pi_nat = np.log(counts / counts.sum())
    log_pi_samp = log_pi_nat - adj  # pi_samp ∝ pi_nat * f, and adj = -log f
    balanced_logits = log_lik + log_pi_samp

    def softmax(z):
        e = np.exp(z - z.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    np.testing.assert_allclose(softmax(balanced_logits + adj), softmax(log_lik + log_pi_nat))


def test_temperature_recovers_a_known_scale_and_never_hurts() -> None:
    rng = np.random.default_rng(2)
    true = rng.normal(size=(20000, 5)) * 2
    p = np.exp(true) / np.exp(true).sum(-1, keepdims=True)
    labels = np.array([rng.choice(5, p=row) for row in p])

    t = CAL.fit_temperature(true * 3.0, labels)  # an overconfident model, 3x too sharp

    assert t == pytest.approx(3.0, rel=0.05)
    assert CAL.nll(true * 3.0, labels, t) <= CAL.nll(true * 3.0, labels, 1.0)
    assert CAL.fit_temperature(true, labels) == pytest.approx(1.0, rel=0.05)


def test_isotonic_matches_scikit_learn_including_ties() -> None:
    sklearn = pytest.importorskip("sklearn.isotonic", reason="needs scikit-learn")
    rng = np.random.default_rng(3)
    x = np.round(rng.uniform(size=3000), 2)  # many ties
    y = (rng.uniform(size=3000) < x**2).astype(float)

    th, vals = CAL.isotonic_fit(x, y)
    ref = sklearn.IsotonicRegression(out_of_bounds="clip").fit(x, y)
    grid = np.linspace(-0.2, 1.2, 301)

    np.testing.assert_allclose(CAL.isotonic_predict(th, vals, grid), ref.predict(grid), atol=1e-12)


def test_isotonic_output_is_monotone_and_a_probability() -> None:
    rng = np.random.default_rng(4)
    x = rng.uniform(size=500)
    y = (rng.uniform(size=500) < 0.3 + 0.4 * x).astype(float)
    th, vals = CAL.isotonic_fit(x, y)

    assert (np.diff(vals) >= -1e-12).all() and vals.min() >= 0 and vals.max() <= 1
    with pytest.raises(CAL.CalibrationError):
        CAL.isotonic_fit(np.array([]), np.array([]))


def test_platt_recovers_known_parameters() -> None:
    rng = np.random.default_rng(5)
    s = rng.uniform(size=50000)
    t = (rng.uniform(size=50000) < 1 / (1 + np.exp(-(4 * s - 1.5)))).astype(float)

    a, b = CAL.platt_fit(s, t)

    assert (a, b) == (pytest.approx(4.0, rel=0.05), pytest.approx(-1.5, rel=0.05))


# --- the whole fit ----------------------------------------------------------------------------


def synthetic_eval(n: int = 4000, c: int = 4, seed: int = 6):
    """Causal per-position logits that sharpen with K and are 3x overconfident;
    a safe logit that tracks correctness; lengths 3..30 with few long flows."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, c, n)
    labels[:200] = -1  # unknown-class flows: never calibrated against
    ppi_len = np.minimum(rng.geometric(0.15, n) + 2, K)
    signal = np.linspace(0.2, 2.5, K)[None, :, None]
    noise = rng.normal(size=(n, K, c))
    onehot = np.eye(c)[np.maximum(labels, 0)][:, None, :]
    true = noise + signal * onehot * 2
    logits = 3.0 * true  # overconfident
    correct = logits.argmax(-1) == np.maximum(labels, 0)[:, None]
    safe = np.where(correct, 1.5, -1.5) + rng.normal(size=(n, K))
    return logits.astype(np.float32), safe.astype(np.float32), labels, ppi_len, c


def test_fit_calibration_end_to_end(tmp_path: Path) -> None:
    logits, safe, labels, ppi_len, c = synthetic_eval()
    counts = np.bincount(labels[labels >= 0], minlength=c)

    cal = CAL.fit_calibration(logits, safe, labels, ppi_len, class_counts=counts, cap=10.0,
                              checkpoint_hash="abc123", min_isotonic=200)  # fmt: skip

    assert len(cal.temperatures) == len(cal.safe_maps) == len(cal.n_fit) == K
    reach = [int(((labels >= 0) & (ppi_len >= k)).sum()) for k in range(1, K + 1)]
    assert cal.n_fit == reach  # only flows that reached K, known classes only
    kinds = [m.kind for m in cal.safe_maps]
    assert all(
        kind == ("isotonic" if n >= 200 else "platt")
        for kind, n in zip(kinds, reach, strict=True)
    )
    assert "platt" in kinds and "isotonic" in kinds  # both paths exercised
    # The Bayes posterior here is softmax(2 s_K * true) and the model emits 3 * true, so
    # the right temperature is 3 / (2 s_K): it must be recovered at every well-fed K.
    s = np.linspace(0.2, 2.5, K)
    for k in range(K):
        if reach[k] >= 1000:
            assert cal.temperatures[k] == pytest.approx(3 / (2 * s[k]), rel=0.15), k + 1

    raw = CAL.Calibration("abc123", [0.0] * c, [1.0] * K, cal.safe_maps, cal.n_fit)
    got = CAL.ece_per_k(cal, logits, labels, ppi_len)
    before = CAL.ece_per_k(raw, logits, labels, ppi_len)
    assert got["ece_k3"] < before["ece_k3"] and got["ece_k8"] < before["ece_k8"]
    assert got["n_k3"] == reach[2]

    p_safe = cal.safe_probability(safe)
    assert p_safe.shape == safe.shape and p_safe.min() >= 0 and p_safe.max() <= 1
    probs = cal.probabilities(logits)
    np.testing.assert_allclose(probs.sum(-1), 1.0, rtol=1e-6)

    cal.save(tmp_path / "calibration.json")
    back = CAL.Calibration.load(tmp_path / "calibration.json", checkpoint_hash="abc123")
    np.testing.assert_allclose(back.safe_probability(safe), p_safe)
    assert back.temperatures == cal.temperatures
    with pytest.raises(CAL.CalibrationError, match="fitted for checkpoint"):
        CAL.Calibration.load(tmp_path / "calibration.json", checkpoint_hash="other")


def test_the_calibrated_safe_probability_matches_the_observed_hit_rate() -> None:
    logits, safe, labels, ppi_len, c = synthetic_eval(n=20000, seed=7)
    counts = np.bincount(labels[labels >= 0], minlength=c)
    cal = CAL.fit_calibration(logits, safe, labels, ppi_len, class_counts=counts, cap=None,
                              checkpoint_hash="h")  # fmt: skip

    k = 3
    rows = CAL.reached(ppi_len, labels, k)
    p = cal.safe_probability(safe)[rows, k - 1]
    hit = (cal.adjusted(logits[rows, k - 1]).argmax(-1) == labels[rows]).astype(float)
    assert ece(p, hit) < 0.02  # what the controller needs: P(correct | commit at K)


def test_a_k_no_flow_reaches_gets_a_neutral_map() -> None:
    logits, safe, labels, ppi_len, c = synthetic_eval(n=500)
    ppi_len = np.minimum(ppi_len, 10)  # nobody reaches K = 11..30
    cal = CAL.fit_calibration(logits, safe, labels, ppi_len,
                              class_counts=np.ones(c), cap=None, checkpoint_hash="h")  # fmt: skip

    assert cal.n_fit[10:] == [0] * 20 and cal.temperatures[10:] == [1.0] * 20
    assert {m.kind for m in cal.safe_maps[10:]} == {"constant"}
    assert CAL.ece_per_k(cal, logits, labels, ppi_len, ks=(30,))["ece_k30"] is None
