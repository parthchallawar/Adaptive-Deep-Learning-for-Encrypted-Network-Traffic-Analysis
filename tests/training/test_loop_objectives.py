"""The loop's phase-3 extensions (plan phase 3, T3): objectives over a dict of
heads, a configurable monitor, label-free training, EMA with warm-up and
layer-wise LR decay. The phase-2 behaviour they must not disturb is held by
``test_loop.py`` and ``test_loop_regression.py``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from adl_etc.data.features import Standardizer
from adl_etc.training import loop as L
from adl_etc.training.checkpoint import inference_state_dict, load_checkpoint
from adl_etc.training.datasets import FlowBatches
from adl_etc.training.ema import ModelEMA, warmup_decay
from adl_etc.training.labels import LabelSpace
from adl_etc.training.loop import ClassifyObjective, TrainSettings, fit, param_groups
from adl_etc.utils.seeding import seed_everything
from adl_etc.utils.tracking import Tracker, read_run_dir
from tests.training.util import (
    TinyMLP,
    TinyPerStep,
    make_cfg,
    make_flows,
    make_info,
    write_shards,
)

N_CLASSES = 4
SPACE = LabelSpace.create(known=range(N_CLASSES))
TIMING = {"epoch_seconds", "flows_per_sec", "train_seconds", "train_flows_per_sec"}


@pytest.fixture
def data(tmp_path: Path) -> tuple[FlowBatches, FlowBatches]:
    train_data = make_flows(600, N_CLASSES, seed=1)
    val_data = make_flows(160, N_CLASSES, seed=2)
    shards = write_shards(tmp_path / "shards", train_data)
    std = Standardizer.fit(shards)
    shards.close()
    return (
        FlowBatches(train_data, SPACE, standardizer=std, train=True),
        FlowBatches(val_data, SPACE, standardizer=std),
    )


def settings(**kw: Any) -> TrainSettings:
    return TrainSettings(**{**dict(epochs=3, batch_size=64, lr=1e-2, patience=10, amp=False), **kw})


def run(
    tmp_path: Path, data: tuple[FlowBatches, FlowBatches], make_model: Any, **kw: Any
) -> tuple[L.TrainResult, torch.nn.Module]:
    """Seed, *then* build the model (so two runs start from the same weights), then fit."""
    train, val = data
    cfg = make_cfg(tmp_path)
    info = make_info(tmp_path, cfg)
    seed_everything(0)
    model = make_model()
    kw.setdefault("settings", settings())
    kw.setdefault("run_dir", tmp_path / "run")
    return fit(model, train, val, run_info=info, **kw), model


def tiny() -> torch.nn.Module:
    return TinyMLP(N_CLASSES)


class Heads(torch.nn.Module):
    """A per-step model that returns a dict of heads, like the PAT."""

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.body = TinyPerStep(n_classes)
        self.aux = torch.nn.Linear(n_classes, 1)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        logits = self.body(batch)
        return {"cls": logits, "aux": self.aux(logits).squeeze(-1)}


@dataclass
class TwoPartObjective:
    """Classification plus a made-up auxiliary loss, reported as a component."""

    monitor: str = "val_macro_f1"
    higher_is_better: bool = True
    uses_labels: bool = True

    def loss(self, outputs: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, Any]:
        cls = L.multi_prefix_ce(outputs["cls"].float(), batch["y"], batch["mask"], 0.0)
        aux = outputs["aux"].float().pow(2).mean()
        return cls + 0.1 * aux, {"cls": cls, "aux": aux}

    def validate(self, model, data, *, batch_size, device, amp) -> dict[str, float]:  # type: ignore[no-untyped-def]
        return L.evaluate(model, data, batch_size=batch_size, device=device, amp=amp, key="cls")


# --- objectives and outputs -------------------------------------------------------------


def test_a_dict_of_heads_trains_and_loss_components_are_recorded(data, tmp_path: Path) -> None:
    result, _ = run(tmp_path, data, Heads, objective=TwoPartObjective(),
                    settings=settings(epochs=8, lr=2e-2))  # fmt: skip

    row = result.history[-1]
    assert {"train_cls", "train_aux", "train_loss"} <= row.keys()
    assert row["train_loss"] == pytest.approx(row["train_cls"] + 0.1 * row["train_aux"], rel=1e-5)
    assert result.best_metric > 0.8  # it really learned (chance is 0.25)


def test_classify_objective_picks_a_named_head(data, tmp_path: Path) -> None:
    result, _ = run(tmp_path, data, Heads, objective=ClassifyObjective(N_CLASSES, key="cls"),
                    settings=settings(epochs=8, lr=2e-2))  # fmt: skip
    assert result.best_metric > 0.8


def test_a_dict_without_a_key_or_with_a_missing_head_is_refused(data, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="name the logits"):
        run(tmp_path, data, Heads)  # default objective, no key
    with pytest.raises(ValueError, match="no 'logits' head"):
        run(tmp_path, data, Heads, objective=ClassifyObjective(N_CLASSES, key="logits"),
            run_dir=tmp_path / "run2")  # fmt: skip


def test_final_only_trains_on_the_last_real_position_alone() -> None:
    logits = torch.randn(3, 5, N_CLASSES, requires_grad=True)
    batch = {"y": torch.tensor([0, 1, 2]), "ppi_len": torch.tensor([2, 5, 1]),
             "mask": torch.arange(5)[None, :] < torch.tensor([2, 5, 1])[:, None]}  # fmt: skip

    loss = L._loss(logits, batch, 0.0, final_only=True)
    loss.backward()

    last = logits[torch.arange(3), batch["ppi_len"] - 1]
    want = F.cross_entropy(last.detach(), batch["y"])
    assert float(loss.detach()) == pytest.approx(float(want))
    grad_rows = logits.grad.abs().sum(-1) > 0  # type: ignore[union-attr]
    expected = torch.zeros(3, 5, dtype=torch.bool)
    expected[torch.arange(3), batch["ppi_len"] - 1] = True
    assert torch.equal(grad_rows, expected)  # no other position gets any gradient


def test_a_lower_is_better_monitor_keeps_the_minimum(
    data, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores = iter([0.9, 0.5, 0.7, 0.6, 0.8, 0.4])

    @dataclass
    class LossMonitored(ClassifyObjective):
        monitor: str = "val_loss"
        higher_is_better: bool = False

        def validate(self, *a: Any, **k: Any) -> dict[str, float]:
            return {"val_loss": next(scores), "val_macro_f1": 0.0}

    result, _ = run(tmp_path, data, tiny, objective=LossMonitored(N_CLASSES),
                    settings=settings(epochs=6, patience=3))  # fmt: skip

    # 0.5 at epoch 2 is the minimum so far; epochs 3-5 fail to beat it -> stop after 5,
    # before the (lower) sixth score is ever reached.
    assert (result.status, result.best_epoch, result.best_metric) == ("early_stopped", 2, 0.5)
    assert result.epochs_done == 5
    assert load_checkpoint(result.best_checkpoint)["epoch"] == 2


def test_validation_must_return_the_monitored_metric(data, tmp_path: Path) -> None:
    @dataclass
    class Wrong(ClassifyObjective):
        monitor: str = "val_auc_k"

    with pytest.raises(ValueError, match="without the monitored 'val_auc_k'"):
        run(tmp_path, data, tiny, objective=Wrong(N_CLASSES))


def test_a_label_free_objective_samples_uniformly_and_needs_no_known_val_flows(
    data, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    real = L.epoch_indices
    monkeypatch.setattr(L, "epoch_indices", lambda *a: seen.append(a[3]) or real(*a))

    @dataclass
    class SelfSupervised:
        monitor: str = "val_recon"
        higher_is_better: bool = False
        uses_labels: bool = False

        def loss(self, outputs: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, Any]:
            return outputs.float().pow(2).mean(), {}

        def validate(self, *a: Any, **k: Any) -> dict[str, float]:
            return {"val_recon": 1.0}

    train, _ = data
    only_unknown = FlowBatches(
        make_flows(30, 6, seed=3, class_ids=[5]),
        LabelSpace.create(known=range(N_CLASSES), unknown=[5]),
        standardizer=train.standardizer,
    )
    train = FlowBatches(train.data, only_unknown.label_space,
                        standardizer=train.standardizer, train=True)  # fmt: skip

    result, _ = run(
        tmp_path, (train, only_unknown), tiny,
        objective=SelfSupervised(), settings=settings(epochs=1, balanced_cap=10.0),
    )  # fmt: skip

    assert result.status == "finished"
    assert seen == [None]  # balanced_cap is ignored without labels


def test_tracker_logs_the_monitor_and_macro_f1_at_the_chosen_epoch(
    data, tmp_path: Path
) -> None:
    scores = iter([0.2, 0.9, 0.4])

    @dataclass
    class AucMonitored(ClassifyObjective):
        monitor: str = "val_auc_k"

        def validate(self, *a: Any, **k: Any) -> dict[str, float]:
            s = next(scores)
            return {"val_auc_k": s, "val_macro_f1": s / 2}

    cfg = make_cfg(tmp_path)
    tracker = Tracker.start(make_info(tmp_path, cfg), cfg, tmp_path / "track")
    run(tmp_path, data, tiny, objective=AucMonitored(N_CLASSES), tracker=tracker)

    final = {m.key: m.value for m in read_run_dir(tmp_path / "track").metrics
             if m.key.startswith("best_")}  # fmt: skip
    assert final == {"best_val_auc_k": 0.9, "best_epoch": 2.0, "best_val_macro_f1": 0.45}


# --- EMA ----------------------------------------------------------------------------------


def test_warmup_decay_starts_fast_and_settles_at_the_target() -> None:
    assert warmup_decay(0.999, 0) == pytest.approx(0.1)
    assert warmup_decay(0.999, 90) == pytest.approx(91 / 100)
    assert warmup_decay(0.999, 10**6) == 0.999


def test_ema_matches_the_closed_form_average() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = ModelEMA(model, 0.9)
    expected = 0.0
    for t, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        with torch.no_grad():
            model.weight.fill_(value)
        ema.update(model)
        d = min(0.9, (1 + t) / (10 + t))
        expected = d * expected + (1 - d) * value

    assert float(ema.module.weight) == pytest.approx(expected)
    assert ema.updates == 5


def test_short_runs_are_not_dominated_by_the_starting_weights() -> None:
    """F11: 220 updates with a plain 0.999 EMA leave ~80% of the start; warm-up
    must leave almost none of it."""
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = ModelEMA(model, 0.999)
    with torch.no_grad():
        model.weight.fill_(1.0)
    for _ in range(220):
        ema.update(model)

    assert float(ema.module.weight) > 0.95  # plain EMA would give 1 - 0.999**220 ≈ 0.20


def test_ema_validates_on_the_average_and_saves_it_for_inference(
    data, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scored: list[torch.nn.Module] = []
    real = L.evaluate

    def spy(model: torch.nn.Module, *a: Any, **k: Any) -> dict[str, float]:
        scored.append(model)
        return real(model, *a, **k)

    monkeypatch.setattr(L, "evaluate", spy)

    result, model = run(tmp_path, data, tiny, settings=settings(ema_decay=0.99))

    assert all(m is not model for m in scored)  # always the EMA copy, never the raw model
    best = load_checkpoint(result.best_checkpoint)
    assert best["ema"]["updates"] == best["epoch"] * -(-600 // 64)
    weights = inference_state_dict(best)
    assert all(torch.equal(weights[k], best["ema"]["model"][k]) for k in weights)
    assert not all(torch.equal(weights[k], best["model"][k]) for k in weights)


def test_inference_weights_are_the_raw_ones_without_ema() -> None:
    payload = {"model": {"w": torch.ones(1)}}
    assert inference_state_dict(payload) is payload["model"]


def test_a_paused_and_resumed_ema_run_equals_an_uninterrupted_one(data, tmp_path: Path) -> None:
    s = settings(epochs=4, ema_decay=0.99)
    whole, _ = run(tmp_path, data, tiny, settings=s, run_dir=tmp_path / "whole")
    first, _ = run(tmp_path, data, tiny, settings=s, run_dir=tmp_path / "split",
                   pause_check=lambda done, _s: done == 2)  # fmt: skip
    second, _ = run(tmp_path, data, tiny, settings=s, run_dir=tmp_path / "split")

    assert (first.status, second.status, second.epochs_run) == ("paused", "finished", 2)
    strip = lambda h: [{k: v for k, v in r.items() if k not in TIMING} for r in h]  # noqa: E731
    assert strip(second.history) == strip(whole.history)
    a = load_checkpoint(tmp_path / "whole" / "ckpt_epoch4.pt")["ema"]
    b = load_checkpoint(tmp_path / "split" / "ckpt_epoch4.pt")["ema"]
    assert a["updates"] == b["updates"]
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])


# --- parameter groups ------------------------------------------------------------------------


class Layered(torch.nn.Module):
    """Embedding (layer 0), two blocks (1, 2), a head (3), with norms and biases."""

    n_layers = 2

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(4 * 30, 8)
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.LayerNorm(8), torch.nn.Linear(8, 8)) for _ in range(2)]
        )
        self.head = torch.nn.Linear(8, N_CLASSES)

    def layer_id(self, name: str) -> int:
        if name.startswith("embed"):
            return 0
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        return self.n_layers + 1

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        h = self.embed(batch["cont"].flatten(1))
        for block in self.blocks:
            h = h + block(h)
        return self.head(h)


def test_no_groups_unless_asked() -> None:
    assert param_groups(Layered(), settings()) is None


def test_layer_decay_scales_each_layers_peak_lr() -> None:
    model = Layered()
    groups = param_groups(model, settings(layer_decay=0.5, lr=1.0))
    assert groups is not None
    lr_of = {id(p): g["lr"] for g in groups for p in g["params"]}

    assert lr_of[id(model.head.weight)] == 1.0
    assert lr_of[id(model.blocks[1][1].weight)] == 0.5
    assert lr_of[id(model.blocks[0][1].weight)] == 0.25
    assert lr_of[id(model.embed.weight)] == 0.125
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))


def test_norm_weights_and_biases_can_be_exempt_from_weight_decay() -> None:
    model = Layered()
    groups = param_groups(model, settings(decay_norm_and_bias=False, weight_decay=0.05))
    assert groups is not None
    wd_of = {id(p): g["weight_decay"] for g in groups for p in g["params"]}

    for name, p in model.named_parameters():
        assert wd_of[id(p)] == (0.0 if p.ndim <= 1 else 0.05), name


def test_layer_decay_needs_a_model_that_names_its_layers() -> None:
    with pytest.raises(ValueError, match="needs a model with `n_layers`"):
        param_groups(TinyMLP(N_CLASSES), settings(layer_decay=0.85))


def test_onecycle_keeps_each_groups_ratio_through_training(data, tmp_path: Path) -> None:
    run(tmp_path, data, Layered, settings=settings(layer_decay=0.5, epochs=2))
    ckpt = load_checkpoint(tmp_path / "run" / "ckpt_epoch2.pt")
    lrs = sorted(g["lr"] for g in ckpt["optimizer"]["param_groups"])

    np.testing.assert_allclose(np.array(lrs[1:]) / np.array(lrs[:-1]), 2.0, rtol=1e-9)


@pytest.mark.parametrize(
    "kw, message",
    [({"ema_decay": 1.0}, "ema_decay"), ({"ema_decay": 0.0}, "ema_decay"),
     ({"layer_decay": 0.0}, "layer_decay"), ({"layer_decay": 1.5}, "layer_decay")],
)  # fmt: skip
def test_invalid_phase3_settings_are_rejected(kw: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        settings(**kw)
