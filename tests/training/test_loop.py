"""The training loop: learning, determinism, and above all resuming (plan T5).

The properties that make it trustworthy are tested directly: an interrupted and
resumed run equals an uninterrupted one bit for bit, a crash between the
checkpoint write and the state write loses nothing, and a resume under a
different label space, standardizer or config is refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from adl_etc.data.features import Standardizer
from adl_etc.training import loop as L
from adl_etc.training.checkpoint import CheckpointError, load_checkpoint
from adl_etc.training.datasets import ArrayData, FlowBatches
from adl_etc.training.labels import LabelSpace
from adl_etc.training.loop import TrainingError, TrainSettings, evaluate, fit
from adl_etc.utils.seeding import seed_everything
from adl_etc.utils.tracking import Tracker, read_run_dir
from tests.training.util import (
    TinyDropoutMLP,
    TinyMLP,
    TinyPerStep,
    make_cfg,
    make_flows,
    make_info,
    write_shards,
)

N_CLASSES = 4
SPACE = LabelSpace.create(known=range(N_CLASSES))
TIMING = {"epoch_seconds", "flows_per_sec"}


class Env:
    """Train/val data plus a fitted standardizer, built once per test."""

    def __init__(self, tmp_path: Path, *, n_train: int = 800, n_val: int = 200) -> None:
        self.tmp = tmp_path
        self.train_data = make_flows(n_train, N_CLASSES, seed=1)
        self.val_data = make_flows(n_val, N_CLASSES, seed=2)
        shards = write_shards(tmp_path, self.train_data)
        self.std = Standardizer.fit(shards)
        shards.close()
        self.train = FlowBatches(self.train_data, SPACE, standardizer=self.std, train=True)
        self.val = FlowBatches(self.val_data, SPACE, standardizer=self.std)

    def settings(self, **kw: Any) -> TrainSettings:
        base = dict(epochs=4, batch_size=64, lr=1e-2, patience=10, amp=False)
        return TrainSettings(**{**base, **kw})

    def run(
        self,
        run_dir: Path,
        *,
        model_cls: type = TinyMLP,
        seed: int = 0,
        resume: bool = True,
        name: str = "bl-tiny-v-synth-s0",
        **fit_kw: Any,
    ) -> tuple[Any, torch.nn.Module]:
        settings = fit_kw.pop("settings", None) or self.settings()
        cfg = make_cfg(self.tmp)
        info = make_info(self.tmp, cfg, name=name, seed=seed)
        seed_everything(seed)
        model = model_cls(N_CLASSES)
        result = fit(
            model,
            self.train,
            self.val,
            settings=settings,
            run_dir=run_dir,
            run_info=info,
            resume=resume,
            **fit_kw,
        )
        return result, model


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def stable(history: list[dict[str, float]]) -> list[dict[str, float]]:
    """History without the wall-clock fields, which legitimately differ."""
    return [{k: v for k, v in row.items() if k not in TIMING} for row in history]


def params_equal(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    pairs = zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    return all(torch.equal(x, y) for x, y in pairs)


# --- settings ---------------------------------------------------------------------------


def test_defaults_are_the_specs_shared_recipe() -> None:
    s = TrainSettings()

    assert (s.epochs, s.batch_size, s.weight_decay) == (20, 4096, 1e-4)
    assert (s.label_smoothing, s.patience, s.balanced_cap) == (0.1, 4, 10.0)
    assert s.amp and s.grad_clip is None


def test_settings_from_a_config_section_and_unknown_keys_raise(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, "  batch_size: 128\n  lr: 0.002\n")

    s = TrainSettings.from_cfg(cfg.train)

    assert (s.epochs, s.batch_size, s.lr) == (3, 128, 0.002)
    with pytest.raises(ValueError, match=r"unknown train setting.*batchsize"):
        TrainSettings.from_cfg({"batchsize": 1})


@pytest.mark.parametrize(
    "kw, message",
    [
        (dict(epochs=0), ">= 1"),
        (dict(lr=0), "lr"),
        (dict(label_smoothing=1.0), "label_smoothing"),
        (dict(balanced_cap=0.5), "balanced_cap"),
        (dict(samples_per_epoch=0), "samples_per_epoch"),
    ],
)
def test_invalid_settings_are_rejected(kw, message) -> None:
    with pytest.raises(ValueError, match=message):
        TrainSettings(**kw)


# --- learning ---------------------------------------------------------------------------


def test_a_model_learns_and_leaves_the_expected_files(env: Env, tmp_path: Path) -> None:
    rd = tmp_path / "run"

    result, _ = env.run(rd, settings=env.settings(epochs=5, keep_last=2))

    assert result.status == "finished" and result.epochs_done == 5 and result.epochs_run == 5
    assert result.best_metric > 0.9  # 4 clearly separable classes
    assert [h["epoch"] for h in result.history] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]
    assert sorted(p.name for p in rd.iterdir()) == [
        "best.pt",
        "ckpt_epoch4.pt",
        "ckpt_epoch5.pt",
        "state.json",
    ]  # keep_last=2 pruned epochs 1-3
    state = json.loads((rd / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "finished" and state["last_checkpoint"] == "ckpt_epoch5.pt"


@pytest.mark.parametrize("model_cls", [TinyMLP, TinyDropoutMLP])
def test_best_checkpoint_reproduces_the_best_validation_score(
    env: Env, tmp_path: Path, model_cls: type
) -> None:
    """With dropout, this also proves evaluation runs in eval mode: in train mode
    the reloaded model's score would be a different random draw."""
    result, _ = env.run(tmp_path / "run", model_cls=model_cls, settings=env.settings(epochs=4))
    ck = load_checkpoint(result.best_checkpoint)

    fresh = model_cls(N_CLASSES)
    fresh.load_state_dict(ck["model"])
    got = evaluate(fresh, env.val, batch_size=64, device=torch.device("cpu"))

    assert ck["epoch"] == result.best_epoch
    assert got["val_macro_f1"] == pytest.approx(result.best_metric, abs=1e-9)


def test_a_per_step_model_trains_with_the_multi_prefix_loss(env: Env, tmp_path: Path) -> None:
    result, model = env.run(
        tmp_path / "run", model_cls=TinyPerStep, settings=env.settings(epochs=8, lr=2e-2)
    )

    # Chance on 4 classes is 0.25. This test is about the per-step path (multi-prefix
    # loss, evaluation at each flow's last packet) training at all, not about a score.
    assert result.best_metric > 0.8
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"] * 0.8
    batch = {k: torch.from_numpy(v) for k, v in env.val.batch(np.arange(4)).items()}
    assert model(batch).shape == (4, 30, N_CLASSES)  # a logit row after every packet


def test_same_seed_gives_an_identical_run_and_a_different_seed_does_not(
    env: Env, tmp_path: Path
) -> None:
    a, ma = env.run(tmp_path / "a", seed=0)
    b, mb = env.run(tmp_path / "b", seed=0)
    c, _ = env.run(tmp_path / "c", seed=1, name="bl-tiny-v-synth-s1")

    assert stable(a.history) == stable(b.history) and params_equal(ma, mb)
    assert stable(a.history) != stable(c.history)


def test_samples_per_epoch_sets_the_number_of_optimizer_steps(env: Env, tmp_path: Path) -> None:
    result, _ = env.run(
        tmp_path / "run", settings=env.settings(epochs=3, batch_size=32, samples_per_epoch=100)
    )

    steps = load_checkpoint(result.run_dir / "ckpt_epoch3.pt")["scheduler"]["last_epoch"]
    assert steps == 3 * 4  # ceil(100 / 32) = 4 steps per epoch


def test_unbalanced_sampling_and_grad_clipping_and_amp_on_cpu_all_run(
    env: Env, tmp_path: Path
) -> None:
    """balanced_cap=null, grad_clip set, amp=True on CPU (silently off) must not crash."""
    result, _ = env.run(
        tmp_path / "run",
        settings=env.settings(epochs=2, balanced_cap=None, grad_clip=1.0, amp=True),
    )

    assert result.status == "finished" and np.isfinite(result.best_metric)


# --- early stopping and pausing ---------------------------------------------------------------


def test_early_stopping_ends_a_run_that_stops_improving(env: Env, tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    noisy = ArrayData(
        env.val_data.ppi,
        env.val_data.ppi_len,
        rng.integers(0, N_CLASSES, len(env.val_data)),  # labels unrelated to the flows
    )
    env.val = FlowBatches(noisy, SPACE, standardizer=env.std)

    result, _ = env.run(tmp_path / "run", settings=env.settings(epochs=30, patience=2))

    assert result.status == "early_stopped" and result.epochs_done < 30
    assert result.epochs_done - result.best_epoch == 2  # exactly `patience` epochs without a gain
    assert load_checkpoint(result.best_checkpoint)["epoch"] == result.best_epoch


def test_pause_check_is_asked_after_each_checkpoint_and_stops_cleanly(
    env: Env, tmp_path: Path
) -> None:
    asked: list[tuple[int, float]] = []
    rd = tmp_path / "run"

    def pause_after_two(done: int, seconds: float) -> bool:
        asked.append((done, seconds))
        assert (rd / f"ckpt_epoch{done}.pt").is_file()  # already safely on disk
        return done == 2

    result, _ = env.run(rd, pause_check=pause_after_two)

    assert result.status == "paused" and result.epochs_done == 2
    assert [a[0] for a in asked] == [1, 2] and all(a[1] > 0 for a in asked)
    assert json.loads((rd / "state.json").read_text(encoding="utf-8"))["status"] == "paused"


def test_the_final_epoch_is_never_reported_as_paused(env: Env, tmp_path: Path) -> None:
    result, _ = env.run(
        tmp_path / "run", settings=env.settings(epochs=2), pause_check=lambda done, s: True
    )

    assert result.status == "paused" and result.epochs_done == 1  # paused after epoch 1 of 2
    resumed, _ = env.run(
        tmp_path / "run", settings=env.settings(epochs=2), pause_check=lambda d, s: True
    )
    assert resumed.status == "finished" and resumed.epochs_done == 2  # not asked after the last


# --- resuming ---------------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls", [TinyMLP, TinyDropoutMLP])
def test_a_paused_and_resumed_run_equals_an_uninterrupted_one_exactly(
    env: Env, tmp_path: Path, model_cls: type
) -> None:
    """Bit-for-bit: model weights, every metric, and the optimizer/scheduler
    state all match, because each epoch's randomness comes from (seed, epoch).
    The dropout model makes that claim bite: its masks come from torch's global
    RNG, which a resumed process would otherwise restart from the wrong place."""
    whole, m_whole = env.run(tmp_path / "whole", model_cls=model_cls)

    first, _ = env.run(
        tmp_path / "split", model_cls=model_cls, pause_check=lambda done, s: done == 2
    )
    second, m_split = env.run(tmp_path / "split", model_cls=model_cls)

    assert first.status == "paused" and first.epochs_done == 2
    assert second.status == "finished" and second.epochs_done == 4
    assert second.epochs_run == 2  # only the remaining epochs were trained
    assert stable(second.history) == stable(whole.history)
    assert params_equal(m_whole, m_split)
    a = load_checkpoint(tmp_path / "whole" / "ckpt_epoch4.pt")
    b = load_checkpoint(tmp_path / "split" / "ckpt_epoch4.pt")
    assert a["scheduler"] == b["scheduler"]
    opt_a, opt_b = a["optimizer"]["state"], b["optimizer"]["state"]
    assert all(torch.equal(opt_a[k]["exp_avg"], opt_b[k]["exp_avg"]) for k in opt_a)


def test_resuming_a_finished_run_does_nothing(env: Env, tmp_path: Path) -> None:
    rd = tmp_path / "run"
    done, _ = env.run(rd)
    before = {p.name: p.read_bytes() for p in rd.iterdir()}

    again, _ = env.run(rd)

    assert again.status == "finished" and again.epochs_run == 0 and again.epochs_done == 4
    assert stable(again.history) == stable(done.history)
    assert {p.name: p.read_bytes() for p in rd.iterdir()} == before  # not a byte changed


def test_a_crash_between_the_checkpoint_and_state_writes_loses_nothing(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checkpoint is written before state.json. If the process dies in
    between, state still points at the previous, intact checkpoint, and the
    resume redoes the epoch and lands exactly where an uninterrupted run does."""
    whole, m_whole = env.run(tmp_path / "whole")

    real_write = L._write_state
    calls = {"n": 0}

    def flaky(run_dir: Path, state: dict[str, Any]) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("killed between the two writes")
        real_write(run_dir, state)

    monkeypatch.setattr(L, "_write_state", flaky)
    rd = tmp_path / "crashed"
    with pytest.raises(OSError, match="killed between"):
        env.run(rd)
    monkeypatch.undo()

    assert (rd / "ckpt_epoch2.pt").is_file()  # written...
    assert (
        json.loads((rd / "state.json").read_text(encoding="utf-8"))["epochs_done"] == 1
    )  # ...unrecorded

    resumed, m_resumed = env.run(rd)

    assert resumed.status == "finished" and resumed.epochs_done == 4
    assert stable(resumed.history) == stable(whole.history)
    assert params_equal(m_whole, m_resumed)


def test_resume_is_refused_under_a_different_config_label_space_or_standardizer(
    env: Env, tmp_path: Path
) -> None:
    rd = tmp_path / "run"
    env.run(rd, pause_check=lambda done, s: True)

    # different config -> different config_hash
    other_cfg = make_cfg(tmp_path, "  lr: 0.5\n")
    info = make_info(tmp_path, other_cfg)
    with pytest.raises(CheckpointError, match="config"):
        fit(
            TinyMLP(N_CLASSES),
            env.train,
            env.val,
            settings=env.settings(),
            run_dir=rd,
            run_info=info,
        )

    # different label space
    shifted = LabelSpace.create(known=range(N_CLASSES), unknown=[9])
    train2 = FlowBatches(env.train_data, shifted, standardizer=env.std, train=True)
    val2 = FlowBatches(env.val_data, shifted, standardizer=env.std)
    info = make_info(tmp_path, make_cfg(tmp_path))
    with pytest.raises(CheckpointError, match="label space"):
        fit(TinyMLP(N_CLASSES), train2, val2, settings=env.settings(), run_dir=rd, run_info=info)

    # different standardizer
    shards = write_shards(tmp_path, make_flows(300, N_CLASSES, seed=99), name="other")
    std2 = Standardizer.fit(shards)
    shards.close()
    train3 = FlowBatches(env.train_data, SPACE, standardizer=std2, train=True)
    val3 = FlowBatches(env.val_data, SPACE, standardizer=std2)
    with pytest.raises(CheckpointError, match="standardizer"):
        fit(TinyMLP(N_CLASSES), train3, val3, settings=env.settings(), run_dir=rd, run_info=info)


def test_an_existing_run_is_not_clobbered_without_resume(env: Env, tmp_path: Path) -> None:
    rd = tmp_path / "run"
    env.run(rd)

    with pytest.raises(CheckpointError, match="resume=True"):
        env.run(rd, resume=False)


# --- guards ------------------------------------------------------------------------------------


def test_a_non_finite_loss_stops_training_and_records_nothing_for_that_epoch(
    env: Env, tmp_path: Path
) -> None:
    class Diverges(TinyMLP):
        def forward(self, batch: dict[str, Any]) -> torch.Tensor:
            return super().forward(batch) * float("nan")

    rd = tmp_path / "run"
    with pytest.raises(TrainingError, match="non-finite"):
        env.run(rd, model_cls=Diverges)

    assert not (rd / "state.json").exists() and not list(rd.glob("ckpt_*"))


def test_a_model_with_the_wrong_number_of_outputs_is_rejected(env: Env, tmp_path: Path) -> None:
    class WrongWidth(TinyMLP):
        def __init__(self, n_classes: int) -> None:
            super().__init__(n_classes + 3)

    with pytest.raises(ValueError, match="emits 7 logits.*4 known classes"):
        env.run(tmp_path / "run", model_cls=WrongWidth)


def test_fit_validates_its_inputs(env: Env, tmp_path: Path) -> None:
    info = make_info(tmp_path, make_cfg(tmp_path))
    kw = dict(settings=env.settings(), run_dir=tmp_path / "r", run_info=info)

    not_train = FlowBatches(env.train_data, SPACE, standardizer=env.std)  # train=False
    with pytest.raises(ValueError, match="train=True"):
        fit(TinyMLP(N_CLASSES), not_train, env.val, **kw)

    other_space = LabelSpace.create(known=range(N_CLASSES), unknown=[7])
    bad_val = FlowBatches(env.val_data, other_space, standardizer=env.std)
    with pytest.raises(ValueError, match="different label spaces"):
        fit(TinyMLP(N_CLASSES), env.train, bad_val, **kw)

    shards = write_shards(tmp_path, make_flows(200, N_CLASSES, seed=5), name="s2")
    other_std = Standardizer.fit(shards)
    shards.close()
    with pytest.raises(ValueError, match="different standardizers"):
        fit(
            TinyMLP(N_CLASSES),
            env.train,
            FlowBatches(env.val_data, SPACE, standardizer=other_std),
            **kw,
        )

    only_unknown = make_flows(20, 6, seed=3, class_ids=[5])
    space = LabelSpace.create(known=range(N_CLASSES), unknown=[5])
    val_none = FlowBatches(only_unknown, space, standardizer=env.std)
    train_ok = FlowBatches(env.train_data, space, standardizer=env.std, train=True)
    with pytest.raises(ValueError, match="no known-class flows"):
        fit(TinyMLP(N_CLASSES), train_ok, val_none, **kw)


def test_unknown_class_flows_in_validation_are_skipped_not_scored(env: Env, tmp_path: Path) -> None:
    space = LabelSpace.create(known=range(N_CLASSES), unknown=[6, 7])
    mixed = make_flows(300, 8, seed=4, class_ids=[0, 1, 2, 3, 6, 7])
    val = FlowBatches(mixed, space, standardizer=env.std)
    train = FlowBatches(env.train_data, space, standardizer=env.std, train=True)

    info = make_info(tmp_path, make_cfg(tmp_path))
    seed_everything(0)
    result = fit(
        TinyMLP(N_CLASSES), train, val, settings=env.settings(epochs=3),
        run_dir=tmp_path / "run", run_info=info,
    )  # fmt: skip

    assert result.best_metric > 0.9  # scored on the 4 known classes only


# --- tracking ------------------------------------------------------------------------------------


def test_fit_logs_each_epoch_to_a_tracker_but_leaves_its_lifecycle_alone(
    env: Env, tmp_path: Path
) -> None:
    cfg = make_cfg(tmp_path)
    info = make_info(tmp_path, cfg)
    tracker = Tracker.start(info, cfg, tmp_path / "track")

    seed_everything(0)
    result = fit(
        TinyMLP(N_CLASSES), env.train, env.val, settings=env.settings(epochs=3),
        run_dir=tmp_path / "run", run_info=info, tracker=tracker,
    )  # fmt: skip

    record = read_run_dir(tmp_path / "track")
    assert record.status == "RUNNING"  # the caller closes it, not fit()
    per_epoch = [m for m in record.metrics if m.key == "val_macro_f1"]
    assert [m.step for m in per_epoch] == [0, 1, 2]
    assert {m.key for m in record.metrics} >= {
        "train_loss", "val_loss", "val_macro_f1", "val_acc", "lr", "epoch_seconds",
        "flows_per_sec", "best_val_macro_f1", "best_epoch",
    }  # fmt: skip
    assert record.params["checkpoint_dir"] == (tmp_path / "run").as_posix()
    assert record.params["best_checkpoint"].endswith("best.pt")
    assert result.status == "finished"


def test_a_paused_run_does_not_log_final_metrics(env: Env, tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    info = make_info(tmp_path, cfg)
    tracker = Tracker.start(info, cfg, tmp_path / "track")

    seed_everything(0)
    fit(
        TinyMLP(N_CLASSES), env.train, env.val, settings=env.settings(),
        run_dir=tmp_path / "run", run_info=info, tracker=tracker,
        pause_check=lambda done, s: True,
    )  # fmt: skip

    keys = {m.key for m in read_run_dir(tmp_path / "track").metrics}
    assert "best_val_macro_f1" not in keys and "val_macro_f1" in keys


# --- pieces that the end-to-end tests cannot see ------------------------------------------------


def test_final_logits_picks_each_flows_last_real_packet() -> None:
    """The per-step model in these tests gives the same answer at any padded
    position, so this is checked directly with logits that differ everywhere."""
    logits = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    batch = {"ppi_len": torch.tensor([2, 4])}

    got = L._final_logits(logits, batch)

    assert torch.equal(got[0], logits[0, 1])  # 2 packets -> position index 1
    assert torch.equal(got[1], logits[1, 3])  # 4 packets -> position index 3
    flat = torch.zeros(2, 3)
    assert L._final_logits(flat, batch) is flat  # per-flow logits pass straight through


def test_the_loop_really_uses_the_capped_balanced_sampler(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    real = L.epoch_indices

    def spy(n_data: int, n_samples: int, rng: Any, weights: Any = None) -> Any:
        seen.append(weights)
        return real(n_data, n_samples, rng, weights)

    monkeypatch.setattr(L, "epoch_indices", spy)

    env.run(tmp_path / "balanced", settings=env.settings(epochs=1, balanced_cap=10.0))
    env.run(tmp_path / "plain", settings=env.settings(epochs=1, balanced_cap=None))

    from adl_etc.training.datasets import balanced_weights

    np.testing.assert_array_equal(seen[0], balanced_weights(env.train.y, N_CLASSES, 10.0))
    assert seen[1] is None


def scripted_validation(monkeypatch: pytest.MonkeyPatch, scores: list[float]) -> None:
    """Make each epoch's validation macro-F1 exactly the next scripted value."""
    it = iter(scores)

    def fake(*args: Any, **kwargs: Any) -> dict[str, float]:
        return {"val_loss": 1.0, "val_acc": 0.5, "val_macro_f1": next(it)}

    monkeypatch.setattr(L, "evaluate", fake)


def test_a_tie_is_not_an_improvement(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that reaches its best score and holds it must still early-stop, and
    best.pt must stay at the *first* epoch that reached it."""
    scripted_validation(monkeypatch, [0.5, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])

    result, _ = env.run(tmp_path / "run", settings=env.settings(epochs=8, patience=2))

    assert result.status == "early_stopped"
    assert (result.best_epoch, result.epochs_done) == (2, 4)  # ties at 3 and 4 use up patience
    assert load_checkpoint(result.best_checkpoint)["epoch"] == 2


def test_patience_counts_epochs_since_the_last_gain_and_resets_on_a_gain(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted_validation(monkeypatch, [0.5, 0.4, 0.6, 0.5, 0.5, 0.5, 0.5, 0.9])

    result, _ = env.run(tmp_path / "run", settings=env.settings(epochs=8, patience=3))

    # epoch 3 is a gain (resets), then 4, 5, 6 fail to beat 0.6 -> stop after epoch 6.
    assert (result.status, result.best_epoch, result.epochs_done) == ("early_stopped", 3, 6)
