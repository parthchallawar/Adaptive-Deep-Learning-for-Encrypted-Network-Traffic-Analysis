"""Efficiency measurements (plan phase 5, T3; spec 013)."""

from __future__ import annotations

import numpy as np
import pytest

from adl_etc.evaluation import efficiency as E
from tests.evaluation.synth import N_CLASSES, make_set

torch = pytest.importorskip("torch")

from adl_etc.models.registry import build_model  # noqa: E402

TINY = {
    "pat": {
        "name": "pat",
        "d_model": 32,
        "layers": 2,
        "heads": 2,
        "ffn": 64,
        "dropout": 0.0,
        "side_channel": True,
        "causal": True,
        "proj_dim": 16,
        "safe_hidden": 16,
    },
    "gru": {"name": "gru", "stem": 16, "hidden": 32, "layers": 1, "dropout": 0.0},
    "cnn": {
        "name": "cnn",
        "channels": [8, 12],
        "kernels": [3, 3],
        "head_hidden": 16,
        "dropout": 0.0,
    },
}
FIELDS = (
    "params",
    "flops_full_flow",
    "latency_batch1",
    "latency_batch64",
    "stream_step",
    "stream_total_30_ms",
    "flops_stream_total_30",
    "weights_mb",
)


@pytest.mark.parametrize("name", list(TINY))
def test_every_model_returns_every_field(name: str) -> None:
    model = build_model(TINY[name], N_CLASSES).eval()
    r = E.bench_model(model, warmup=1, iters=3, stream_flows=2)
    for f in FIELDS:
        assert f in r, f
    assert r["params"] == sum(p.numel() for p in model.parameters())
    assert r["flops_full_flow"] > 0 and r["latency_batch1"]["median_ms"] > 0
    assert r["latency_batch1"]["p95_ms"] >= r["latency_batch1"]["median_ms"]
    assert r["stream_step"]["median_ms"] > 0


def test_a_causal_models_stream_flops_match_its_full_pass_within_five_percent() -> None:
    """Spec 013's consistency test, on a PAT with realistic proportions (d=128, 4 layers): the
    sum over packet steps equals the full pass to within 5%. The full pass computes every
    position's attention at once, the stream only each new position's, so the gap is the
    attention's causal half; with 31 positions and d=128 it is small."""
    cfg = {**TINY["pat"], "d_model": 128, "layers": 4, "heads": 4, "ffn": 512, "proj_dim": 128}
    cfg["safe_hidden"] = 64
    model = build_model(cfg, N_CLASSES).eval()
    r = E.bench_model(model, warmup=1, iters=2, stream_flows=1)
    ratio = r["flops_stream_total_30"] / r["flops_full_flow"]
    assert 0.95 <= ratio <= 1.05, ratio


def test_the_non_causal_cnn_pays_a_full_pass_per_packet() -> None:
    model = build_model(TINY["cnn"], N_CLASSES).eval()
    r = E.bench_model(model, warmup=1, iters=3, stream_flows=1)
    assert r["flops_stream_total_30"] == 30 * r["flops_full_flow"]
    assert "non-causal" in r["stream_note"]
    assert r["stream_step"]["median_ms"] == r["latency_batch1"]["median_ms"]


def test_bench_all_interleaves_rounds_and_keeps_the_smaller_median() -> None:
    models = {n: build_model(TINY[n], N_CLASSES).eval() for n in ("gru", "cnn")}
    out = E.bench_all(models, rounds=2, threads=1, warmup=1, iters=3, stream_flows=1)
    for r in out.values():
        meds = r["batch1_median_ms_by_round"]
        assert len(meds) == 2 and r["latency_batch1"]["median_ms"] == min(meds)


def test_time_to_decision_sums_the_ipts_up_to_the_decision_packet() -> None:
    ipt = np.array([[0, 10, 20, 30, 40], [0, 5, 5, 5, 5]], dtype=np.float32)
    ppi_len = np.array([5, 3])
    k = np.array([3, 4])  # the second flow has only 3 packets: capped at 3
    np.testing.assert_allclose(E.time_to_decision_ms(ipt, ppi_len, k), [30.0, 10.0])


def test_policy_cost_has_every_policy_at_a_matched_mean_k() -> None:
    val, test = make_set(60, 600, seed=1), make_set(60, 600, seed=2)
    ipt = np.random.default_rng(0).uniform(1, 50, (len(test), 30)).astype(np.float32)
    out = E.policy_cost(val, test, ipt, N_CLASSES, target_mean_k=6.0)
    assert out["n_flows"] == int((test.labels >= 0).sum())
    for name in ("safe_stop", "p_echo", "p_cape", "p_rl"):
        row = out[name]
        assert set(row["packets"]) == {"mean", "median", "p95"}
        assert row["packets"]["p95"] >= row["packets"]["median"] >= 1
        assert 0 < row["time_share_of_flow"] <= 1.0
        assert 0.0 <= row["accuracy"] <= 1.0
    # fits were done on validation only, so the matched point is the val mean K, near the target
    assert abs(out["safe_stop"]["val_mean_k"] - 6.0) < 0.5
