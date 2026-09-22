"""``training.run`` end to end: config -> split -> fit -> checkpoint + report
(plan T8). Trains a real (tiny) GRU on synthetic learnable flows so the whole
entry point is exercised for real, including the post-training evaluation
pass, without needing real D1 data on this machine.
"""

from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")

from adl_etc.data.features import Standardizer  # noqa: E402
from adl_etc.training import run as R  # noqa: E402
from adl_etc.utils.config import load_config  # noqa: E402
from tests.training.util import make_flows, write_shards  # noqa: E402

N_CLASSES = 4


def build_cfg(tmp_path, *, epochs=2, results_root=None):
    """A real, small synthetic split (train/val) plus a training config that
    points at it -- everything ``run_training``/``main`` need."""
    data_root = tmp_path / "data"
    train_data = make_flows(300, N_CLASSES, seed=0)
    val_data = make_flows(100, N_CLASSES, seed=1)
    write_shards(data_root, train_data, name="synthetic", period="train", n_classes=N_CLASSES)
    write_shards(data_root, val_data, name="synthetic", period="val", n_classes=N_CLASSES)

    train_ss = write_shards(
        data_root, train_data, name="std_fit", period="all", n_classes=N_CLASSES
    )
    std = Standardizer.fit(train_ss)
    train_ss.close()
    std_path = tmp_path / "standardizer.json"
    std.save(std_path)

    split_yaml = tmp_path / "split.yaml"
    split_yaml.write_text(
        f"""
dataset: synthetic
temporal: false
splits:
  train: {{periods: [train], standardizer: {std_path}}}
  val: {{periods: [val], standardizer: {std_path}}}
""",
        encoding="utf-8",
    )

    rr = results_root if results_root is not None else tmp_path / "results"
    cfg_yaml = tmp_path / "train.yaml"
    cfg_yaml.write_text(
        f"""
model: {{name: gru, stem: 16, hidden: 24, layers: 1, dropout: 0.0}}
train:
  epochs: {epochs}
  batch_size: 64
  lr: 0.005
  weight_decay: 0.0001
  label_smoothing: 0.1
  patience: 99
  balanced_cap: null
  amp: false
seed: 0
split: {split_yaml.as_posix()}
data_root: {data_root.as_posix()}
standardizer: {std_path.as_posix()}
train_split: train
val_split: val
device: cpu
results_root: {rr.as_posix()}
stage: bl
variant: synth
split_name: synth
""",
        encoding="utf-8",
    )
    return cfg_yaml


def test_run_training_writes_a_checkpoint_label_space_and_report(tmp_path):
    cfg_yaml = build_cfg(tmp_path)
    cfg = load_config(cfg_yaml)

    result = R.run_training(cfg, seed=0)

    run_dir = R.run_dir_for(cfg, 0)
    assert run_dir.name == "bl-gru-synth-synth-s0"
    assert result.status in ("finished", "early_stopped")
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "label_space.json").is_file()
    assert (run_dir / "state.json").is_file()
    assert (run_dir / "eval" / "val" / "report.json").is_file()
    assert R.is_finished(run_dir)


def test_run_training_report_reflects_a_model_that_actually_learned(tmp_path):
    from adl_etc.evaluation.report import Report

    cfg_yaml = build_cfg(tmp_path, epochs=25)
    cfg = load_config(cfg_yaml)
    R.run_training(cfg, seed=0)

    report = Report.load(R.run_dir_for(cfg, 0) / "eval" / "val" / "report.json")
    assert report.n_flows == 100
    assert report.n_classes == N_CLASSES
    # The synthetic classes are easy to separate; a real trained GRU should be
    # well above chance (1/4) by the fully-informed K.
    assert report.acc_at_k[report.ks[-1]] > 0.6


def test_a_paused_run_resumes_rather_than_restarting(tmp_path):
    cfg_yaml = build_cfg(tmp_path, epochs=4)
    cfg = load_config(cfg_yaml)

    # A deadline already in the past forces a pause after the first epoch
    # regardless of how long it took.
    paused = R.run_training(cfg, seed=0, deadline=time.monotonic() - 1.0, evaluate_after=False)
    assert paused.status == "paused"
    assert paused.epochs_done == 1
    assert not R.is_finished(R.run_dir_for(cfg, 0))

    resumed = R.run_training(cfg, seed=0, deadline=None)
    assert resumed.status in ("finished", "early_stopped")
    assert resumed.epochs_run == 3  # continued from epoch 1, not restarted
    assert len(resumed.history) == 4
    assert resumed.history[0]["train_loss"] == paused.history[0]["train_loss"]
    assert R.is_finished(R.run_dir_for(cfg, 0))


def test_a_finished_run_is_not_retrained_on_a_second_call(tmp_path):
    cfg_yaml = build_cfg(tmp_path, epochs=2)
    cfg = load_config(cfg_yaml)
    first = R.run_training(cfg, seed=0)

    second = R.run_training(cfg, seed=0)
    assert second.epochs_run == 0
    assert second.history == first.history


def test_run_dir_for_two_seeds_of_the_same_config_differ_only_in_seed(tmp_path):
    cfg_yaml = build_cfg(tmp_path)
    cfg = load_config(cfg_yaml)
    d0, d1 = R.run_dir_for(cfg, 0), R.run_dir_for(cfg, 1)
    assert d0 != d1
    assert d0.name.replace("s0", "s1") == d1.name


def test_time_guard_pauses_only_when_the_next_epoch_would_breach_the_deadline():
    assert R._time_guard(None) is None

    now = time.monotonic()
    generous = R._time_guard(now + 1000.0)
    assert generous is not None
    assert generous(1, 50.0) is False  # 1000s left, next epoch ~60s (with margin): fine

    tight = R._time_guard(now + 10.0)
    assert tight is not None
    assert tight(1, 50.0) is True  # 10s left, next epoch ~60s: would breach


def test_main_cli_runs_one_config_end_to_end(tmp_path):
    cfg_yaml = build_cfg(tmp_path)
    cfg = load_config(cfg_yaml)

    rc = R.main(["--config", str(cfg_yaml)])
    assert rc == 0
    assert R.is_finished(R.run_dir_for(cfg, 0))


def test_main_cli_seed_override_wins_over_the_configs_own_seed(tmp_path):
    cfg_yaml = build_cfg(tmp_path)
    cfg = load_config(cfg_yaml)

    R.main(["--config", str(cfg_yaml), "--seed", "7"])
    assert R.is_finished(R.run_dir_for(cfg, 7))
    assert not R.is_finished(R.run_dir_for(cfg, 0))


def test_is_finished_is_false_for_a_directory_with_no_state_json(tmp_path):
    assert R.is_finished(tmp_path / "never-run") is False


def test_a_paused_run_writes_no_evaluation_report(tmp_path):
    cfg_yaml = build_cfg(tmp_path, epochs=4)
    cfg = load_config(cfg_yaml)
    R.run_training(cfg, seed=0, deadline=time.monotonic() - 1.0)
    assert not (R.run_dir_for(cfg, 0) / "eval").exists()
