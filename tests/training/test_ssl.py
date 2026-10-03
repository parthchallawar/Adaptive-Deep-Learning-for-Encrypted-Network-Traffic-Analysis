"""Spec 007's self-supervised pretraining (plan phase 3, T5)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from adl_etc.data import ppi as P
from adl_etc.data.features import Standardizer
from adl_etc.models.pat import PAT
from adl_etc.models.registry import build_model
from adl_etc.training.datasets import ArrayData, FlowBatches, to_torch
from adl_etc.training.labels import LabelSpace
from adl_etc.training.ssl import (
    COLLAPSE_UNIFORMITY,
    PretrainPAT,
    SSLBatches,
    knn_accuracy,
    pfc_loss,
    token_hashes,
    uniformity,
)
from adl_etc.utils.config import load_config
from tests.training.util import make_flows, write_shards

C = 4
SPACE = LabelSpace.create(range(C))
SMALL = {"d_model": 32, "layers": 2, "heads": 4, "ffn": 64}


@pytest.fixture(scope="module")
def std(tmp_path_factory: pytest.TempPathFactory) -> Standardizer:
    shards = write_shards(tmp_path_factory.mktemp("std"), make_flows(400, C, seed=1))
    out = Standardizer.fit(shards)
    shards.close()
    return out


def ssl_batches(data: ArrayData, std: Standardizer, **kw) -> SSLBatches:
    return SSLBatches(data, SPACE, standardizer=std, views=("tokens", "continuous"), **kw)


# --- the model --------------------------------------------------------------------------------


def test_pretrain_pat_is_a_pat_with_the_same_weights(std: Standardizer) -> None:
    torch.manual_seed(0)
    pre = PretrainPAT(C, **SMALL).eval()
    plain = PAT(C, **SMALL).eval()
    assert pre.state_dict().keys() == plain.state_dict().keys()
    plain.load_state_dict(pre.state_dict())

    data = make_flows(16, C, seed=2)
    batch = to_torch(FlowBatches(data, SPACE, standardizer=std,
                                 views=("tokens", "continuous")).batch(np.arange(16)))  # fmt: skip
    torch.testing.assert_close(pre(batch)["cls"], plain(batch)["cls"])


# --- the views ---------------------------------------------------------------------------------


def test_views_are_deterministic_per_batch_and_seed(std: Standardizer) -> None:
    data = make_flows(64, C, seed=3, min_len=3, max_len=30)
    idx = np.arange(10, 42)
    a = ssl_batches(data, std, seed=5).batch(idx)
    b = ssl_batches(data, std, seed=5).batch(idx)
    c = ssl_batches(data, std, seed=6).batch(idx)

    for key in ("tokens_a", "tokens_b", "ppi_len_a", "cont_a"):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["tokens_a"], c["tokens_a"])
    shifted = ssl_batches(data, std, seed=5).batch(idx + 1)
    assert not np.array_equal(a["tokens_a"], shifted["tokens_a"])


def test_the_full_view_stays_aligned_with_the_raw_flow(std: Standardizer) -> None:
    data = make_flows(200, C, seed=4, min_len=1, max_len=30)
    out = ssl_batches(data, std, seed=0).batch(np.arange(200))

    np.testing.assert_array_equal(out["ppi_len_b"], out["ppi_len"])
    np.testing.assert_array_equal(out["mask_b"], out["mask"])
    dir_pos = P.DIR_POS
    np.testing.assert_array_equal(out["tokens_b"][..., dir_pos], out["tokens"][..., dir_pos])


def test_the_prefix_view_is_cropped_shorter_than_its_augmented_flow(std: Standardizer) -> None:
    data = make_flows(300, C, seed=5, min_len=1, max_len=30)
    out = ssl_batches(data, std, seed=1).batch(np.arange(300))
    k, eligible = out["ppi_len_a"], out["eligible_a"]

    assert (k[eligible] >= 3).all() and (k <= out["ppi_len"]).all()
    pad = np.arange(P.K_MAX)[None, :] >= k[:, None]
    assert (out["tokens_a"][pad] == 0).all() and (out["cont_a"][pad] == 0).all()
    assert not eligible[out["ppi_len"] <= 3].any()  # too short to crop below the flow


def test_identical_flows_get_identical_twin_keys(std: Standardizer) -> None:
    data = make_flows(20, C, seed=6)
    data.ppi[7] = data.ppi[3]
    data.ppi_len[7] = data.ppi_len[3]
    keys = ssl_batches(data, std).batch(np.arange(20))["twin"]

    assert keys[7] == keys[3]
    assert len(set(keys.tolist())) == 19


def test_mpm_blanks_about_thirty_percent_of_real_packets_and_keeps_them_attended(
    std: Standardizer,
) -> None:
    data = make_flows(400, C, seed=7, min_len=5, max_len=30)
    out = ssl_batches(data, std, mode="mpm").batch(np.arange(400))
    masked, real = out["mpm_mask"], out["mask"]

    assert not (masked & ~real).any()
    assert masked.sum() / real.sum() == pytest.approx(0.3, abs=0.03)
    assert (out["tokens_m"][masked] == 0).all() and (out["cont_m"][masked] == 0).all()
    np.testing.assert_array_equal(out["mask_m"], real)


# --- the losses and diagnostics --------------------------------------------------------------


def test_pfc_puts_positives_on_the_diagonal_and_is_symmetric() -> None:
    torch.manual_seed(0)
    anchors = F.normalize(torch.randn(8, 16), dim=-1)
    positives = F.normalize(anchors + 0.01 * torch.randn(8, 16), dim=-1)
    twin = torch.arange(8)

    loss, top1 = pfc_loss(anchors, positives, twin, temperature=0.05)

    sim = anchors @ positives.T / 0.05
    target = torch.arange(8)
    manual = 0.5 * (F.cross_entropy(sim, target) + F.cross_entropy(sim.T, target))
    torch.testing.assert_close(loss, manual)
    assert float(top1) == 1.0 and float(loss) < 0.1


def test_twins_are_not_each_others_negatives() -> None:
    torch.manual_seed(1)
    base = F.normalize(torch.randn(4, 16), dim=-1)
    anchors = torch.cat([base, base[:1]])  # flow 4 is flow 0's twin
    positives = anchors.clone()

    masked, _ = pfc_loss(anchors, positives, torch.tensor([0, 1, 2, 3, 0]), temperature=0.05)
    unmasked, _ = pfc_loss(anchors, positives, torch.arange(5), temperature=0.05)

    assert float(unmasked) > math.log(2) * 0.3  # the twins confuse each other...
    assert float(masked) < 0.05  # ...unless they are taken out of each other's negatives


def test_the_objective_reads_each_view_where_spec_007_says() -> None:
    """PFC: anchor = view a's proj at its crop position, positive = view b's proj
    at its last packet; NPP: scored on view b against the *raw* next tokens. A
    hand-built output that is right only under that wiring gives ~zero loss."""
    from adl_etc.data import features as FT
    from adl_etc.training.ssl import SSLObjective

    b, k = 6, P.K_MAX
    rng = np.random.default_rng(0)
    raw = torch.zeros(b, k, 4, dtype=torch.long)  # real token ranges, channel by channel
    raw[..., P.IPT_POS] = torch.as_tensor(rng.integers(1, FT.IPT_VOCAB, (b, k)))
    raw[..., P.DIR_POS] = torch.as_tensor(rng.integers(1, FT.DIR_VOCAB, (b, k)))
    raw[..., P.SIZE_POS] = torch.as_tensor(rng.integers(1, FT.SIZE_VOCAB, (b, k)))
    raw[..., P.PUSH_POS] = torch.as_tensor(rng.integers(1, FT.PUSH_VOCAB, (b, k)))
    length = torch.tensor([5, 9, 30, 12, 7, 20])
    crop = torch.tensor([3, 4, 10, 6, 5, 8])
    raw[torch.arange(k)[None, :] >= length[:, None]] = 0
    vecs = F.normalize(torch.randn(b, 128), dim=-1)

    def proj_at(positions: torch.Tensor) -> torch.Tensor:
        p = F.normalize(torch.randn(b, k, 128), dim=-1)  # noise everywhere else
        p[torch.arange(b), positions - 1] = vecs
        return p

    nxt = torch.cat([raw[:, 1:], torch.zeros_like(raw[:, :1])], dim=1)
    view_b = {
        "proj": proj_at(length),
        "next_size": F.one_hot(nxt[..., P.SIZE_POS], FT.SIZE_VOCAB).float() * 50,
        "next_ipt": F.one_hot(nxt[..., P.IPT_POS], FT.IPT_VOCAB).float() * 50,
        "next_dir": F.one_hot(nxt[..., P.DIR_POS], FT.DIR_VOCAB).float() * 50,
        "next_end": torch.zeros(b, k),
    }
    outputs = {"a": {"proj": proj_at(crop)}, "b": view_b}
    jittered = raw.clone()  # view b's own (jittered) tokens differ from the raw ones
    jittered[..., P.SIZE_POS] = torch.where(
        raw[..., P.SIZE_POS] > 0, raw[..., P.SIZE_POS] % (FT.SIZE_VOCAB - 1) + 1, 0
    )
    batch = {
        "tokens": raw, "tokens_b": jittered,
        "ppi_len": length, "ppi_len_a": crop, "ppi_len_b": length,
        "eligible_a": torch.ones(b, dtype=torch.bool), "twin": torch.arange(b),
    }  # fmt: skip

    _, parts = SSLObjective(temperature=0.02).loss(outputs, batch)

    assert float(parts["pfc"]) < 1e-3 and float(parts["pfc_top1"]) == 1.0
    assert float(parts["npp"]) < math.log(2) + 1e-3  # only the zero-logit `end` term remains


def test_uniformity_flags_a_collapse() -> None:
    torch.manual_seed(2)
    spread = F.normalize(torch.randn(512, 128), dim=-1)
    collapsed = F.normalize(torch.ones(512, 128) + 1e-4 * torch.randn(512, 128), dim=-1)

    assert float(uniformity(spread)) < -3.5
    assert float(uniformity(collapsed)) > COLLAPSE_UNIFORMITY


def test_knn_accuracy_is_high_on_clusters_and_chance_on_noise() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 4, 2000)
    centres = rng.normal(size=(4, 16)) * 5
    clustered = centres[y] + rng.normal(size=(2000, 16))

    assert knn_accuracy(clustered, y) > 0.95
    assert knn_accuracy(rng.normal(size=(2000, 16)), y) < 0.4


def test_token_hashes_are_stable_and_distinct() -> None:
    tokens = np.random.default_rng(4).integers(0, 60, (1000, 30, 4))
    h = token_hashes(tokens)
    assert len(set(h.tolist())) == 1000
    np.testing.assert_array_equal(h, token_hashes(tokens.copy()))


# --- end to end ------------------------------------------------------------------------------


def pretrain_cfg(tmp_path: Path, objective: str, *, epochs: int = 2, causal: bool = True):
    from tests.training.test_run import build_cfg

    model = (f"{{name: pat_pretrain, d_model: 32, layers: 2, heads: 4, ffn: 64, "
             f"causal: {str(causal).lower()}}}")  # fmt: skip
    return load_config(build_cfg(
        tmp_path, epochs=epochs, model=model, extra=f"objective: {objective}",
    ))  # fmt: skip


def test_ssl_pretraining_runs_resumes_exactly_probes_and_feeds_fine_tuning(tmp_path: Path) -> None:
    import json
    import time

    from adl_etc.training import run as R
    from adl_etc.training.finetune import load_pretrained

    (tmp_path / "whole").mkdir()
    (tmp_path / "split").mkdir()
    whole_cfg = pretrain_cfg(tmp_path / "whole", "{name: ssl}", epochs=3)
    split_cfg = pretrain_cfg(tmp_path / "split", "{name: ssl}", epochs=3)

    whole = R.run_training(whole_cfg, seed=0)
    first = R.run_training(split_cfg, seed=0, deadline=time.monotonic() - 1, evaluate_after=False)
    second = R.run_training(split_cfg, seed=0)

    assert first.status == "paused" and second.status == "finished"
    strip = lambda h: [{k: v for k, v in r.items()  # noqa: E731
                        if k not in ("epoch_seconds", "flows_per_sec")} for r in h]  # fmt: skip
    assert strip(second.history) == strip(whole.history)  # augmented views and all
    row = whole.history[-1]
    assert {"train_npp", "train_pfc", "train_pfc_top1", "val_ssl_loss", "val_knn_acc",
            "val_uniform", "val_align", "val_collapsed"} <= row.keys()  # fmt: skip
    assert whole.history[-1]["val_npp"] < whole.history[0]["val_npp"]

    run_dir = R.run_dir_for(whole_cfg, 0)
    probe = json.loads((run_dir / "eval" / "val" / "probe.json").read_text())
    assert {"probe_acc_k3", "probe_acc_k30"} <= probe.keys()
    assert not (run_dir / "eval" / "val" / "report.json").exists()  # no classification report

    model = build_model({"name": "pat", **SMALL}, 3)  # a different class count is fine
    info = load_pretrained(model, run_dir / "best.pt")
    assert info["init_epoch"] >= 1


def test_mpm_pretrains_the_bidirectional_model_and_its_weights_load_causally(
    tmp_path: Path,
) -> None:
    from adl_etc.training import run as R
    from adl_etc.training.finetune import load_pretrained

    cfg = pretrain_cfg(tmp_path, "{name: mpm}", epochs=3, causal=False)
    result = R.run_training(cfg, seed=0)

    assert result.status == "finished"
    assert result.history[-1]["val_mpm"] < result.history[0]["val_mpm"]
    causal = build_model({"name": "pat", **SMALL}, C)
    load_pretrained(causal, R.run_dir_for(cfg, 0) / "best.pt")
    assert causal.causal  # MPM's weights, the causal model's mask


def test_ssl_learns_structure_before_any_label(tmp_path: Path) -> None:
    """Spec 007's small overfit test: on easy synthetic flows the kNN probe on
    `proj`, which never saw a label, beats chance by a wide margin."""
    from adl_etc.training import run as R

    cfg = pretrain_cfg(tmp_path, "{name: ssl}", epochs=8)
    result = R.run_training(cfg, seed=0, evaluate_after=False)

    assert max(h["val_knn_acc"] for h in result.history) > 0.6  # chance is 0.25
    assert result.history[-1]["val_collapsed"] == 0.0
