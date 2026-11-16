"""The Prefix-Aware Transformer (spec 006; plan phase 3, T4).

Causality is the property everything else rests on: an output after packet k
that depends on packet k+1 would make every earliness number in the project a
leak. It is tested on every head, through both the tokens and the side channel,
and the non-causal switch is tested to be real (so the causality test cannot
pass vacuously). Streaming is tested against the full forward pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from adl_etc.data import features as FT
from adl_etc.data import ppi as P
from adl_etc.data.features import Standardizer
from adl_etc.models.baselines.predict import predict_dense
from adl_etc.models.pat import PAT, PATStream
from adl_etc.models.registry import build_model
from adl_etc.training.datasets import ArrayData, FlowBatches, to_torch
from adl_etc.training.labels import LabelSpace
from tests.training.util import make_flows, write_shards

C = 5
SPACE = LabelSpace.create(range(C))
HEADS = ("cls", "safe", "next_size", "next_ipt", "next_dir", "next_end", "proj", "z")


@pytest.fixture(scope="module")
def std(tmp_path_factory: pytest.TempPathFactory) -> Standardizer:
    shards = write_shards(tmp_path_factory.mktemp("std"), make_flows(400, C, seed=1))
    out = Standardizer.fit(shards)
    shards.close()
    return out


@pytest.fixture(scope="module")
def flows() -> ArrayData:
    return make_flows(24, C, seed=7, min_len=3, max_len=30)


def batch_of(data: ArrayData, std: Standardizer, views=("tokens", "continuous")) -> dict:
    fb = FlowBatches(data, SPACE, standardizer=std, views=views)
    return to_torch(fb.batch(np.arange(len(data))))


def small(**kw) -> PAT:
    torch.manual_seed(0)
    return PAT(C, **{"d_model": 32, "layers": 2, "heads": 4, "ffn": 64, **kw}).eval()


# --- shape and size ---------------------------------------------------------------------


def test_every_head_has_the_documented_shape(flows: ArrayData, std: Standardizer) -> None:
    out = small()(batch_of(flows, std))
    n = len(flows)

    assert out["cls"].shape == (n, P.K_MAX, C)
    assert out["safe"].shape == out["next_end"].shape == (n, P.K_MAX)
    assert out["next_size"].shape == (n, P.K_MAX, FT.SIZE_VOCAB)
    assert out["next_ipt"].shape == (n, P.K_MAX, FT.IPT_VOCAB)
    assert out["next_dir"].shape == (n, P.K_MAX, FT.DIR_VOCAB)
    assert out["proj"].shape == (n, P.K_MAX, 128)
    assert out["cls0"].shape == (n, C) and out["z0"].shape == (n, 32)
    norms = out["proj"].norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_the_default_model_fits_the_two_million_parameter_budget() -> None:
    n = sum(p.numel() for p in PAT(150).parameters())
    assert 500_000 < n < 2_000_000, n  # spec 006: about 0.9M at the defaults, cap 2M


# --- causality ------------------------------------------------------------------------------


def perturbed(data: ArrayData, j: int) -> ArrayData:
    """Every flow's packet at 0-based index ``j`` changed in size, timing and push."""
    ppi = data.ppi.copy()
    ppi[:, j, P.SIZE_POS] = np.where(ppi[:, j, P.SIZE_POS] > 700, 40, 1400)
    ppi[:, j, P.IPT_POS] = ppi[:, j, P.IPT_POS] + 900
    ppi[:, j, P.PUSH_POS] = 1 - ppi[:, j, P.PUSH_POS]
    return ArrayData(ppi, data.ppi_len, data.label, data.session_id)


@pytest.mark.parametrize("side_channel", [True, False])
@pytest.mark.parametrize("j", [1, 5, 11])
def test_no_output_at_k_depends_on_a_later_packet(
    flows: ArrayData, std: Standardizer, side_channel: bool, j: int
) -> None:
    model = small(side_channel=side_channel)
    views = model.views
    base = model(batch_of(flows, std, views))
    moved = model(batch_of(perturbed(flows, j), std, views))

    for head in HEADS:
        torch.testing.assert_close(moved[head][:, :j], base[head][:, :j], rtol=0, atol=1e-6)
    torch.testing.assert_close(moved["cls0"], base["cls0"], rtol=0, atol=1e-6)
    # ...and the perturbation is real: outputs from packet j on do move.
    assert not torch.allclose(moved["cls"][:, j], base["cls"][:, j])


def test_the_non_causal_switch_really_lets_earlier_positions_see_later_packets(
    flows: ArrayData, std: Standardizer
) -> None:
    model = small(causal=False)
    base = model(batch_of(flows, std))
    moved = model(batch_of(perturbed(flows, 5), std))

    assert not torch.allclose(moved["cls"][:, :5], base["cls"][:, :5])


@pytest.mark.parametrize("causal", [True, False])
def test_padded_positions_are_invisible_whatever_they_hold(
    flows: ArrayData, std: Standardizer, causal: bool
) -> None:
    """For the causal model the causal mask alone already hides padding from real
    positions; the non-causal one (spec 007's MPM) relies on the key-padding mask,
    so both are checked."""
    model = small(causal=causal)
    batch = batch_of(flows, std)
    base = model(batch)
    garbage = dict(batch)
    pad = ~batch["mask"][..., None]
    garbage["tokens"] = torch.where(pad, torch.ones_like(batch["tokens"]), batch["tokens"])
    garbage["cont"] = torch.where(pad, torch.full_like(batch["cont"], 5.0), batch["cont"])
    moved = model(garbage)

    valid = batch["mask"]
    for head in HEADS:
        torch.testing.assert_close(moved[head][valid], base[head][valid], rtol=0, atol=1e-6)


# --- streaming --------------------------------------------------------------------------------


@pytest.mark.parametrize("side_channel", [True, False])
def test_streaming_equals_the_full_forward_pass_at_every_real_position(
    flows: ArrayData, std: Standardizer, side_channel: bool
) -> None:
    model = small(side_channel=side_channel)
    batch = batch_of(flows, std, model.views)
    full = model(batch)

    stream = PATStream(model)
    first = stream.start(len(flows))
    torch.testing.assert_close(first["cls"], full["cls0"], rtol=1e-5, atol=1e-5)
    for k in range(P.K_MAX):
        cont = batch["cont"][:, k] if side_channel else None
        step = stream.step(batch["tokens"][:, k], cont)
        real = batch["mask"][:, k]  # a finished flow's padding is not a packet
        for head in HEADS:
            torch.testing.assert_close(
                step[head][real], full[head][:, k][real], rtol=1e-5, atol=1e-5, msg=head
            )


def test_streaming_refuses_a_training_or_non_causal_model_and_a_31st_packet() -> None:
    with pytest.raises(ValueError, match="eval"):
        PATStream(small().train())
    with pytest.raises(ValueError, match="causal"):
        PATStream(small(causal=False))
    stream = PATStream(small(side_channel=False))
    with pytest.raises(ValueError, match="start"):
        stream.step(torch.ones(2, 4, dtype=torch.long))
    stream.start(2)
    for _ in range(P.K_MAX):
        stream.step(torch.ones(2, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="at most 30"):
        stream.step(torch.ones(2, 4, dtype=torch.long))


# --- heads, layers, side channel -----------------------------------------------------------


def test_training_safe_never_moves_the_heads_it_reads(flows: ArrayData, std: Standardizer) -> None:
    model = small().train()
    out = model(batch_of(flows, std))
    out["safe"].sum().backward()

    for head in ("head_cls", "head_next_size", "head_next_ipt", "head_next_dir", "head_next_end"):
        grads = [p.grad for p in getattr(model, head).parameters()]
        assert all(g is None or float(g.abs().sum()) == 0.0 for g in grads), head
    assert any(float(p.grad.abs().sum()) > 0 for p in model.head_safe.parameters())
    assert float(model.blocks[0].qkv.weight.grad.abs().sum()) > 0  # it does shape z


def test_layer_ids_cover_every_parameter_in_order() -> None:
    model = small()
    ids = {name: model.layer_id(name) for name, _ in model.named_parameters()}

    assert set(ids.values()) == {0, 1, 2, 3}
    assert ids["emb_size.weight"] == ids["bos"] == ids["side.weight"] == 0
    assert ids["blocks.0.qkv.weight"] == 1 and ids["blocks.1.ff.0.weight"] == 2
    assert ids["head_cls.weight"] == ids["norm_f.weight"] == 3


def test_the_side_channel_is_on_by_default_and_needs_its_view(
    flows: ArrayData, std: Standardizer
) -> None:
    on, off = small(), small(side_channel=False)
    assert on.views == ("tokens", "continuous") and off.views == ("tokens",)
    tokens_only = batch_of(flows, std, ("tokens",))

    assert off(tokens_only)["cls"].shape == (len(flows), P.K_MAX, C)
    with pytest.raises(ValueError, match="'cont' view"):
        on(tokens_only)


def test_the_side_channel_carries_size_detail_the_tokens_lose(std: Standardizer) -> None:
    """F10: two sizes in one bin are one token, but not one side-channel input."""
    a = make_flows(1, C, seed=3, min_len=4, max_len=4)
    b = ArrayData(a.ppi.copy(), a.ppi_len, a.label, a.session_id)
    a.ppi[0, 0, P.SIZE_POS], b.ppi[0, 0, P.SIZE_POS] = 1340, 1500  # same bin [1338, 1500)
    ba, bb = batch_of(a, std), batch_of(b, std)
    assert torch.equal(ba["tokens"], bb["tokens"])

    on, off = small(), small(side_channel=False)
    assert not torch.allclose(on(ba)["cls"][:, 0], on(bb)["cls"][:, 0])
    torch.testing.assert_close(off(ba)["cls"], off(bb)["cls"], rtol=0, atol=0)


# --- integration ----------------------------------------------------------------------------


def test_predict_dense_reads_the_cls_head_as_causal_logits(
    flows: ArrayData, std: Standardizer
) -> None:
    model = small()
    dense = predict_dense(model, flows, SPACE, std, batch_size=10)

    assert dense.indexing == "effective" and dense.logits.shape == (len(flows), P.K_MAX, C)
    np.testing.assert_allclose(
        dense.logits, model(batch_of(flows, std))["cls"].detach().numpy(), atol=1e-5
    )


def test_the_registry_builds_a_pat_and_rejects_typos() -> None:
    model = build_model({"name": "pat", "d_model": 32, "layers": 1, "ffn": 64}, C)
    assert isinstance(model, PAT) and model.n_layers == 1
    with pytest.raises(ValueError, match="bad arguments for model 'pat'"):
        build_model({"name": "pat", "dmodel": 32}, C)
    assert type(build_model({"name": "gru"}, C)).__name__ == "RNNBaseline"


def test_the_shipped_config_builds_the_spec_model() -> None:
    from adl_etc.utils.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[2] / "configs/models/pat.yaml")
    model = build_model(cfg.model, 150)

    assert isinstance(model, PAT)
    assert (model.n_layers, model.side_channel, model.causal) == (4, True, True)
    assert model.blocks[0].ff[0].out_features == 512 and model.head_cls.in_features == 128
