"""``evaluation.bootstrap``: per-flow predictions and the paired bootstrap (plan
phase 3, T8). NumPy only."""

from __future__ import annotations

import numpy as np
import pytest

from adl_etc.data import ppi as P
from adl_etc.evaluation import metrics as M
from adl_etc.evaluation.bootstrap import Predictions, paired_bootstrap
from adl_etc.evaluation.dense_logits import from_causal, from_per_k
from adl_etc.evaluation.report import UNKNOWN_LABEL

N, C = 2000, 5


def _labels_and_len(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, C, N)
    labels[:200] = UNKNOWN_LABEL
    return labels, rng.integers(1, P.K_MAX + 1, N)


def _causal_preds(labels, ppi_len, p_right: float, seed: int) -> Predictions:
    """A causal model right with probability ``p_right`` at every position."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(N, P.K_MAX, C)).astype(np.float32)
    right = rng.random((N, P.K_MAX)) < p_right
    for i in range(N):
        if labels[i] >= 0:
            logits[i, right[i], labels[i]] += 10.0
    return Predictions.from_dense(from_causal(logits), labels, ppi_len, np.arange(N))


def test_predictions_match_the_metrics_module_for_a_causal_model():
    labels, ppi_len = _labels_and_len()
    rng = np.random.default_rng(1)
    dense = from_causal(rng.normal(size=(N, P.K_MAX, C)).astype(np.float32))
    p = Predictions.from_dense(dense, labels, ppi_len, np.arange(N))
    assert p.ks.tolist() == list(range(1, P.K_MAX + 1))
    for k in (1, 5, 30):
        assert np.array_equal(p.preds[:, k - 1], M.predictions_at_k(dense, ppi_len, k))


def test_predictions_keep_a_per_k_models_own_grid():
    labels, ppi_len = _labels_and_len()
    rng = np.random.default_rng(2)
    dense = from_per_k({k: rng.normal(size=(N, C)).astype(np.float32) for k in (1, 3, 10, 30)})
    p = Predictions.from_dense(dense, labels, ppi_len, np.arange(N))
    assert p.ks.tolist() == [1, 3, 10, 30]
    assert np.array_equal(p.preds[:, 2], M.predictions_at_k(dense, ppi_len, 10))


def test_predictions_round_trip(tmp_path):
    labels, ppi_len = _labels_and_len()
    p = _causal_preds(labels, ppi_len, 0.5, seed=3)
    p.save(tmp_path / "p.npz")
    q = Predictions.load(tmp_path / "p.npz")
    for name in ("flow_index", "labels", "ppi_len", "ks", "preds"):
        assert np.array_equal(getattr(p, name), getattr(q, name)), name
    assert q.preds.dtype == np.int16


def test_bootstrap_observed_values_are_the_plain_accuracies_on_known_flows():
    labels, ppi_len = _labels_and_len()
    a = _causal_preds(labels, ppi_len, 0.4, seed=4)
    b = _causal_preds(labels, ppi_len, 0.6, seed=5)
    out = paired_bootstrap(a, b, n=200)
    known = labels != UNKNOWN_LABEL
    for k in (1, 5, 30):
        acc_a = float(np.mean(a.preds[known, k - 1] == labels[known]))
        acc_b = float(np.mean(b.preds[known, k - 1] == labels[known]))
        row = out["acc_at_k"][str(k)]
        assert row["a"] == pytest.approx(acc_a) and row["b"] == pytest.approx(acc_b)
        assert row["diff"] == pytest.approx(acc_b - acc_a)
    assert out["n_flows"] == int(known.sum())
    # expected gap 0.16 (a "wrong" flow is still right 1 time in C by chance:
    # 0.6 + 0.4/5 vs 0.4 + 0.6/5): the interval excludes zero, b never "not better"
    auc = out["auc_k"]
    assert 0.12 < auc["diff"] < 0.20 and auc["ci_lo"] > 0 and auc["p_not_better"] == 0.0


def test_bootstrap_of_a_model_against_itself_is_exactly_zero():
    labels, ppi_len = _labels_and_len()
    a = _causal_preds(labels, ppi_len, 0.5, seed=6)
    out = paired_bootstrap(a, a, n=100)
    assert out["auc_k"]["diff"] == 0.0
    assert out["auc_k"]["ci_lo"] == 0.0 and out["auc_k"]["ci_hi"] == 0.0


def test_bootstrap_is_deterministic_in_its_seed():
    labels, ppi_len = _labels_and_len()
    a = _causal_preds(labels, ppi_len, 0.5, seed=7)
    b = _causal_preds(labels, ppi_len, 0.52, seed=8)
    assert paired_bootstrap(a, b, n=100, seed=3) == paired_bootstrap(a, b, n=100, seed=3)
    assert paired_bootstrap(a, b, n=100, seed=3) != paired_bootstrap(a, b, n=100, seed=4)


def test_bootstrap_refuses_different_flows_or_missing_ks():
    labels, ppi_len = _labels_and_len()
    a = _causal_preds(labels, ppi_len, 0.5, seed=9)
    shifted = Predictions(a.flow_index + 1, a.labels, a.ppi_len, a.ks, a.preds)
    with pytest.raises(ValueError, match="same flows"):
        paired_bootstrap(a, shifted, n=10)
    with pytest.raises(ValueError, match="not in both"):
        paired_bootstrap(a, a, ks=[1, 31], n=10)
