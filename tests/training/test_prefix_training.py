"""Spec 008's prefix training of the PAT (plan phase 3, T6): the losses, the
objective, the causal accuracy curve its early stopping reads, and starting
from a pretrained checkpoint."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from adl_etc.data import features as FT
from adl_etc.data import ppi as P
from adl_etc.data.features import Standardizer
from adl_etc.evaluation.dense_logits import from_causal
from adl_etc.evaluation.metrics import auc_k, causal_accuracy_curve
from adl_etc.models.pat import PAT
from adl_etc.training.checkpoint import CheckpointError, save_checkpoint
from adl_etc.training.datasets import FlowBatches
from adl_etc.training.finetune import load_pretrained
from adl_etc.training.labels import LabelSpace
from adl_etc.training.loop import TrainSettings, fit
from adl_etc.training.losses import multi_prefix_ce
from adl_etc.training.objectives import PrefixObjective, build_objective
from adl_etc.training.pat_losses import (
    end_targets,
    npp_loss,
    prefix_ce,
    ramp_weights,
    safe_bce,
    self_distillation,
)
from adl_etc.utils.seeding import seed_everything
from tests.training.util import make_cfg, make_flows, make_info, write_shards

C = 4
SPACE = LabelSpace.create(range(C))


def lengths_mask(lengths: list[int], k: int = P.K_MAX) -> tuple[torch.Tensor, torch.Tensor]:
    ppi_len = torch.tensor(lengths)
    return ppi_len, torch.arange(k)[None, :] < ppi_len[:, None]


# --- npp ----------------------------------------------------------------------------------


def test_end_targets_mark_the_last_packet_and_ignore_truncation_and_padding() -> None:
    target, keep = end_targets(torch.tensor([3, 30, 1]))

    assert target[0, :4].tolist() == [0, 0, 1, 0] and keep[0, :4].tolist() == [1, 1, 1, 0]
    assert not keep[1, 29] and keep[1, :29].all()  # 30 packets: maybe more followed
    assert keep[2, 0] and target[2, 0] == 1 and not keep[2, 1:].any()


def npp_outputs(tokens: torch.Tensor, *, shift: int, sharp: float = 30.0) -> dict:
    """Heads that predict the token ``shift`` positions ahead with confidence."""
    nxt = torch.roll(tokens, -shift, dims=1)
    out = {}
    for head, ch, vocab in (("next_size", P.SIZE_POS, FT.SIZE_VOCAB),
                            ("next_ipt", P.IPT_POS, FT.IPT_VOCAB),
                            ("next_dir", P.DIR_POS, FT.DIR_VOCAB)):  # fmt: skip
        out[head] = F.one_hot(nxt[..., ch], vocab).float() * sharp
    out["next_end"] = torch.zeros(tokens.shape[:2])
    return out


def test_npp_targets_are_the_next_packet_not_this_one() -> None:
    flows = make_flows(6, C, seed=2, min_len=4, max_len=9)
    ppi_len = torch.from_numpy(flows.ppi_len.astype(np.int64))
    tokens = torch.from_numpy(FT.tokenize(flows.ppi, flows.ppi_len))
    batch = {"tokens": tokens, "ppi_len": ppi_len}

    _, right = npp_loss(npp_outputs(tokens, shift=1), batch)
    _, wrong = npp_loss(npp_outputs(tokens, shift=0), batch)

    for part in ("npp_size", "npp_ipt"):
        assert float(right[part]) < 1e-6 < float(wrong[part])
    assert float(right["npp_end"]) == pytest.approx(math.log(2))  # zero logits


def test_npp_ignores_positions_with_no_next_packet() -> None:
    flows = make_flows(5, C, seed=3, min_len=3, max_len=7)
    tokens = torch.from_numpy(FT.tokenize(flows.ppi, flows.ppi_len))
    batch = {"tokens": tokens, "ppi_len": torch.from_numpy(flows.ppi_len.astype(np.int64))}
    out = npp_outputs(tokens, shift=1)
    base, _ = npp_loss(out, batch)

    garbage = {k: v.clone() for k, v in out.items()}
    for i, n in enumerate(flows.ppi_len):  # from the last real packet on, nothing to predict
        for head in ("next_size", "next_ipt", "next_dir"):
            garbage[head][i, int(n) - 1 :] = torch.randn_like(garbage[head][i, int(n) - 1 :]) * 50
    moved, _ = npp_loss(garbage, batch)

    assert float(moved) == pytest.approx(float(base), abs=1e-6)


# --- prefix losses ---------------------------------------------------------------------------


def test_uniform_prefix_ce_is_the_phase_2_multi_prefix_loss() -> None:
    torch.manual_seed(0)
    logits, y = torch.randn(6, P.K_MAX, C), torch.randint(0, C, (6,))
    _, mask = lengths_mask([1, 4, 30, 9, 2, 17])

    got = prefix_ce(logits, y, mask, ramp=1.0, smoothing=0.1)

    torch.testing.assert_close(got, multi_prefix_ce(logits, y, mask, 0.1))


def test_the_ramp_weights_early_positions_down() -> None:
    w = ramp_weights(P.K_MAX, 8.0)
    assert w[:8].tolist() == pytest.approx([k / 8 for k in range(1, 9)])
    assert (w[8:] == 1).all()
    with pytest.raises(ValueError, match="ramp"):
        ramp_weights(P.K_MAX, 0.5)

    torch.manual_seed(1)
    logits, y = torch.randn(3, P.K_MAX, C), torch.randint(0, C, (3,))
    _, mask = lengths_mask([5, 12, 3])
    ce = F.cross_entropy(logits.reshape(-1, C), y.repeat_interleave(P.K_MAX), reduction="none")
    ce = ce.reshape(3, P.K_MAX)
    wm = w[None, :] * mask
    torch.testing.assert_close(
        prefix_ce(logits, y, mask, ramp=8.0, smoothing=0.0), (ce * wm).sum() / wm.sum()
    )


def test_self_distillation_pulls_early_positions_and_never_the_teacher() -> None:
    torch.manual_seed(2)
    logits = torch.randn(3, P.K_MAX, C, requires_grad=True)
    ppi_len, mask = lengths_mask([6, 2, 30])

    loss = self_distillation(logits, mask, ppi_len, ramp=1.0, temperature=2.0)
    loss.backward()

    grad = logits.grad.abs().sum(-1)  # type: ignore[union-attr]
    for i, n in enumerate([6, 2, 30]):
        assert float(grad[i, n - 1]) == pytest.approx(0.0, abs=1e-7)  # the teacher
        assert (grad[i, : n - 1] > 0).all()  # the students
        assert float(grad[i, n:].sum()) == 0.0  # padding
    same = logits.detach()[:, :1].expand(3, P.K_MAX, C)  # every position agrees
    assert float(self_distillation(same, mask, ppi_len)) == pytest.approx(0.0, abs=1e-7)


def test_safe_bce_balances_each_k_and_never_reaches_the_classifier() -> None:
    cls = torch.zeros(4, P.K_MAX, C, requires_grad=True)
    with torch.no_grad():
        cls[0, 0, 1] = 5.0  # flow 0 right at K=1; flows 1-3 wrong (argmax 0, y = 1)
    y = torch.tensor([1, 1, 1, 1])
    with torch.no_grad():
        cls[1:, 0, 0] = 5.0
    _, mask = lengths_mask([1, 1, 1, 1])
    safe = torch.tensor([2.0, -1.0, 0.0, 1.0])[:, None].expand(4, P.K_MAX).clone()
    safe.requires_grad_(True)

    loss = safe_bce(safe, cls, y, mask)
    loss.backward()

    bce_pos = F.softplus(torch.tensor(-2.0))
    bce_neg = F.softplus(torch.tensor([-1.0, 0.0, 1.0])).mean()
    assert float(loss) == pytest.approx(float(0.5 * (bce_pos + bce_neg)), rel=1e-6)
    assert cls.grad is None or float(cls.grad.abs().sum()) == 0.0


# --- the causal accuracy curve -------------------------------------------------------------


def test_the_curve_from_argmaxes_equals_auc_k_from_the_full_logits() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, P.K_MAX, C)).astype(np.float32)
    labels = rng.integers(0, C, 200)
    ppi_len = rng.integers(1, P.K_MAX + 1, 200)

    curve = causal_accuracy_curve(logits.argmax(-1), labels, ppi_len)

    assert curve.shape == (P.K_MAX,)
    assert float(curve.mean()) == pytest.approx(auc_k(from_causal(logits), labels, ppi_len))


# --- the objective end to end --------------------------------------------------------------


@pytest.fixture
def data(tmp_path: Path) -> tuple[FlowBatches, FlowBatches]:
    train_data, val_data = make_flows(600, C, seed=1), make_flows(160, C, seed=2)
    shards = write_shards(tmp_path / "s", train_data)
    std = Standardizer.fit(shards)
    shards.close()
    views = ("tokens", "continuous")
    return (
        FlowBatches(train_data, SPACE, standardizer=std, views=views, train=True),
        FlowBatches(val_data, SPACE, standardizer=std, views=views),
    )


def small_pat(n_classes: int = C, **kw) -> PAT:
    return PAT(n_classes, **{"d_model": 32, "layers": 2, "heads": 4, "ffn": 64, **kw})


def test_prefix_training_learns_and_early_stops_on_auc_k(data, tmp_path: Path) -> None:
    train, val = data
    info = make_info(tmp_path, make_cfg(tmp_path))
    seed_everything(0)
    objective = PrefixObjective(C, 0.1)

    result = fit(
        small_pat(), train, val, objective=objective, run_info=info, run_dir=tmp_path / "run",
        settings=TrainSettings(epochs=8, batch_size=64, lr=2e-3, patience=99, amp=False,
                               ema_decay=0.99, decay_norm_and_bias=False),
    )  # fmt: skip

    row = result.history[-1]
    assert {"train_cls", "train_sd", "train_safe", "train_npp"} <= row.keys()
    assert {"val_auc_k", "val_acc_k1", "val_acc_k3", "val_acc_k5", "val_acc_k8"} <= row.keys()
    assert result.best_metric == max(h["val_auc_k"] for h in result.history)
    assert result.best_metric > 0.6  # chance is 0.25
    assert row["val_acc_k8"] >= row["val_acc_k1"] - 0.02  # more packets do not hurt

    # Independently: the best checkpoint's inference (EMA) weights, scored through
    # predict_dense + auc_k, give exactly the AUC_K validation recorded for it.
    from adl_etc.models.baselines.predict import predict_dense
    from adl_etc.training.checkpoint import inference_state_dict, load_checkpoint

    model = small_pat()
    model.load_state_dict(inference_state_dict(load_checkpoint(result.best_checkpoint)))
    dense = predict_dense(model, val.data, SPACE, val.standardizer)
    want = auc_k(dense, SPACE.to_model(val.data.label), val.data.ppi_len)
    assert result.best_metric == pytest.approx(want, abs=1e-9)


def test_the_total_is_spec_008s_weighted_sum_and_every_head_learns(data) -> None:
    """``L = L_cls + 0.2 L_sd + 1.0 L_safe + 0.1 L_NPP``, and each term really
    reaches its head: drop one and that head stops getting gradient."""
    train, _ = data
    batch = {k: torch.from_numpy(v) for k, v in train.batch(np.arange(64)).items()}
    model = small_pat().train()
    objective = PrefixObjective(C, 0.1)

    total, parts = objective.loss(model(batch), batch)
    total.backward()

    want = parts["cls"] + 0.2 * parts["sd"] + 1.0 * parts["safe"] + 0.1 * parts["npp"]
    assert float(total) == pytest.approx(float(want), rel=1e-6)
    for head in ("head_cls", "head_safe", "head_next_size", "head_next_end"):
        assert float(getattr(model, head)[-1].weight.grad.abs().sum()
                     if isinstance(getattr(model, head), torch.nn.Sequential)
                     else getattr(model, head).weight.grad.abs().sum()) > 0, head  # fmt: skip


def test_build_objective_makes_a_prefix_objective_and_rejects_typos() -> None:
    settings = TrainSettings(label_smoothing=0.0)
    obj = build_objective({"name": "prefix", "ramp": 1.0}, model=small_pat(),
                          n_classes=C, settings=settings)  # fmt: skip
    assert isinstance(obj, PrefixObjective)
    assert (obj.ramp, obj.label_smoothing, obj.monitor) == (1.0, 0.0, "val_auc_k")
    with pytest.raises(ValueError, match="unknown option"):
        build_objective({"name": "prefix", "rampp": 2}, model=small_pat(), n_classes=C,
                        settings=settings)  # fmt: skip
    with pytest.raises(ValueError, match="unknown option"):  # not settable from here
        build_objective({"name": "prefix", "monitor": "val_loss"}, model=small_pat(),
                        n_classes=C, settings=settings)  # fmt: skip


# --- starting from a pretrained checkpoint --------------------------------------------------


def saved(model: torch.nn.Module, path: Path, **extra) -> Path:
    save_checkpoint(path, {"epoch": 3, "model": model.state_dict(), "config_hash": "abc", **extra})
    return path


def test_pretrained_weights_load_except_the_fresh_heads(tmp_path: Path) -> None:
    torch.manual_seed(0)
    pretrained = small_pat(n_classes=7)  # a different class count is fine: cls is fresh
    torch.manual_seed(1)
    model = small_pat()
    fresh_cls = model.head_cls.weight.clone()

    info = load_pretrained(model, saved(pretrained, tmp_path / "ssl.pt"))

    assert torch.equal(model.blocks[0].qkv.weight, pretrained.blocks[0].qkv.weight)
    assert torch.equal(model.head_next_size.weight, pretrained.head_next_size.weight)
    assert torch.equal(model.head_cls.weight, fresh_cls)
    assert info["init_epoch"] == 3 and info["init_used_ema"] is False


def test_the_ema_weights_are_what_gets_loaded(tmp_path: Path) -> None:
    pretrained, averaged = small_pat(), small_pat()
    path = saved(pretrained, tmp_path / "ssl.pt",
                 ema={"decay": 0.999, "updates": 5, "model": averaged.state_dict()})  # fmt: skip
    model = small_pat()

    info = load_pretrained(model, path)

    assert torch.equal(model.emb_size.weight, averaged.emb_size.weight)
    assert info["init_used_ema"] is True


def test_a_checkpoint_that_does_not_fit_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="does not fit"):
        load_pretrained(small_pat(), saved(small_pat(d_model=48), tmp_path / "wide.pt"))
    with pytest.raises(CheckpointError, match="does not fit"):
        load_pretrained(small_pat(), saved(small_pat(layers=3), tmp_path / "deep.pt"))
    with pytest.raises(CheckpointError, match="does not fit"):
        load_pretrained(small_pat(side_channel=False), saved(small_pat(), tmp_path / "side.pt"))


def test_a_checkpoint_trained_on_different_tokens_is_refused(tmp_path: Path) -> None:
    other = small_pat()
    other.tokenizer_id.fill_(12345)  # same shapes, different bin edges

    with pytest.raises(CheckpointError, match="different tokens"):
        load_pretrained(small_pat(), saved(other, tmp_path / "old.pt"))


# --- fine-tuning augmentation ----------------------------------------------------------------


def test_finetune_augmentation_is_deterministic_and_off_means_off(data) -> None:
    from adl_etc.training.finetune import AugmentedBatches

    train, _ = data
    idx = np.arange(64)

    def aug(seed: int, p: float, s: float) -> dict:
        return AugmentedBatches(train.data, SPACE, standardizer=train.standardizer,
                                views=train.views, train=True, seed=seed, drop_p=p,
                                ipt_s=s).batch(idx)  # fmt: skip

    a, b, c = aug(0, 0.05, 0.1), aug(0, 0.05, 0.1), aug(1, 0.05, 0.1)
    plain = train.batch(idx)
    for key in ("tokens", "cont", "mask", "ppi_len"):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["tokens"], c["tokens"])
    assert not np.array_equal(a["tokens"], plain["tokens"])
    np.testing.assert_array_equal(a["y"], plain["y"])  # labels never change
    assert (a["ppi_len"] <= plain["ppi_len"]).all() and (a["ppi_len"] >= 1).all()
    off = aug(0, 0.0, 0.0)  # the abl_noaug ablation
    for key in ("tokens", "cont", "mask", "ppi_len", "y"):
        np.testing.assert_array_equal(off[key], plain[key])
    with pytest.raises(ValueError, match="drop_p"):
        aug(0, 1.0, 0.1)
