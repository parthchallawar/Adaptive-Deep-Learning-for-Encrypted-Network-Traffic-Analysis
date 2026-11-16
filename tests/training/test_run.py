"""``training.run`` end to end: config -> split -> fit -> checkpoint + report
(plan T8). Trains a real (tiny) GRU on synthetic learnable flows so the whole
entry point is exercised for real, including the post-training evaluation
pass, without needing real D1 data on this machine.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adl_etc.data.features import Standardizer  # noqa: E402
from adl_etc.training import run as R  # noqa: E402
from adl_etc.utils.config import load_config  # noqa: E402
from tests.training.util import make_flows, write_shards  # noqa: E402

N_CLASSES = 4


GRU = "{name: gru, stem: 16, hidden: 24, layers: 1, dropout: 0.0}"


def build_cfg(tmp_path, *, epochs=2, results_root=None, model=GRU, extra="", split_extra=""):
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
{split_extra}
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
model: {model}
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
{extra}
""",
        encoding="utf-8",
    )
    return cfg_yaml


PAT_SMALL = "{name: pat, d_model: 32, layers: 2, heads: 4, ffn: 64, dropout: 0.0}"


def test_b5_trains_the_pat_end_to_end_on_its_last_packet_only(tmp_path):
    """B5: the PAT through the real entry point, with the classify objective's
    ``final_only`` from config; the report comes from its ``cls`` head."""
    from adl_etc.evaluation.report import Report
    from adl_etc.training.checkpoint import load_checkpoint

    cfg = load_config(build_cfg(
        tmp_path, epochs=12, model=PAT_SMALL,
        extra="objective: {name: classify, final_only: true}",
    ))  # fmt: skip

    result = R.run_training(cfg, seed=0)

    run_dir = R.run_dir_for(cfg, 0)
    assert run_dir.name == "bl-pat-synth-synth-s0"
    assert result.best_metric > 0.6  # chance is 0.25
    assert "tokens" not in load_checkpoint(run_dir / "best.pt")["model"]  # weights, not data
    report = Report.load(run_dir / "eval" / "val" / "report.json")
    assert report.acc_at_k[report.ks[-1]] > 0.6


def test_prefix_fine_tuning_starts_from_a_pretrained_checkpoint_and_records_it(tmp_path):
    """Spec 008 end to end: a PAT checkpoint (here B5's best.pt standing in for an
    SSL one), then a prefix-trained run initialised from it. The init is recorded
    in the run's params, and a resume never needs the pretrained file again."""
    import shutil

    from adl_etc.utils.tracking import read_run_dir

    (tmp_path / "pre").mkdir()
    pre_cfg = load_config(build_cfg(
        tmp_path / "pre", epochs=2, model=PAT_SMALL,
        extra="objective: {name: classify, final_only: true}",
    ))  # fmt: skip
    R.run_training(pre_cfg, seed=0, evaluate_after=False)
    pretrained = tmp_path / "pretrained.pt"
    shutil.copy(R.run_dir_for(pre_cfg, 0) / "best.pt", pretrained)

    (tmp_path / "ft").mkdir()
    cfg = load_config(build_cfg(
        tmp_path / "ft", epochs=3, model=PAT_SMALL,
        extra=f"objective: {{name: prefix}}\ninit_from: {pretrained.as_posix()}",
    ))  # fmt: skip
    first = R.run_training(cfg, seed=0, deadline=time.monotonic() - 1.0, evaluate_after=False)
    run_dir = R.run_dir_for(cfg, 0)
    params = read_run_dir(run_dir).params
    assert first.status == "paused"
    assert params["init_from"] == pretrained.as_posix() and params["init_epoch"] >= 1

    pretrained.unlink()  # a resume continues from its own checkpoint
    second = R.run_training(cfg, seed=0)

    assert second.status == "finished" and second.epochs_done == 3
    assert "val_auc_k" in second.history[-1]


def test_an_open_set_split_trains_only_its_known_classes(tmp_path):
    """T1a end to end: class 3 is held out, so it never reaches training, the
    label space has 3 known classes, and validation still sees class-3 flows."""
    from adl_etc.training.labels import LabelSpace
    from adl_etc.utils.tracking import read_run_dir

    cfg = load_config(build_cfg(tmp_path, epochs=2, split_extra="unknown_classes: [3]"))

    result = R.run_training(cfg, seed=0)

    run_dir = R.run_dir_for(cfg, 0)
    space = LabelSpace.load(run_dir / "label_space.json")
    assert (space.known, space.unknown) == ((0, 1, 2), (3,))
    params = read_run_dir(run_dir).params
    train_labels = make_flows(300, N_CLASSES, seed=0).label
    assert params["n_train_flows"] == int((train_labels != 3).sum())
    assert params["n_unknown_classes"] == 1
    assert result.status in ("finished", "early_stopped")


def test_a_label_fraction_trains_on_a_seeded_subset_with_the_same_label_space(tmp_path):
    from adl_etc.training.datasets import ArrayData
    from adl_etc.training.labels import LabelSpace
    from adl_etc.utils.tracking import read_run_dir

    cfg = load_config(build_cfg(tmp_path, epochs=1, extra="label_fraction: 0.1"))
    R.run_training(cfg, seed=0, evaluate_after=False)

    run_dir = R.run_dir_for(cfg, 0)
    assert read_run_dir(run_dir).params["n_train_flows"] == 30  # 10% of 300
    assert LabelSpace.load(run_dir / "label_space.json").n_classes == N_CLASSES

    data = make_flows(300, N_CLASSES, seed=0)
    a, b = R.label_subset(data, 0.1, seed=0), R.label_subset(data, 0.1, seed=0)
    c = R.label_subset(data, 0.1, seed=1)
    assert np.array_equal(a.ppi, b.ppi) and not np.array_equal(a.ppi, c.ppi)
    assert R.label_subset(data, 1.0, seed=0) is data
    assert len(R.label_subset(data, 1e-9, seed=0)) == 1  # never empty
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="label_fraction"):
            R.label_subset(data, bad, seed=0)
    assert isinstance(a, ArrayData)


def test_prefix_training_writes_a_calibration_that_evaluation_applies(tmp_path):
    """T7 end to end: a prefix-trained PAT leaves calibration.json beside best.pt
    (bound to its hash) and safe.npy; evaluation.run applies the calibration and
    writes calibrated ECE per K; a calibration for a different checkpoint is refused."""
    import json
    import shutil

    from adl_etc.evaluation import run as EVAL
    from adl_etc.evaluation.calibration import Calibration, CalibrationError
    from adl_etc.utils.provenance import file_sha256

    cfg_yaml = build_cfg(tmp_path, epochs=3, model=PAT_SMALL, extra="objective: {name: prefix}")
    cfg = load_config(cfg_yaml)
    R.run_training(cfg, seed=0)

    run_dir = R.run_dir_for(cfg, 0)
    cal = Calibration.load(run_dir / "calibration.json",
                           checkpoint_hash=file_sha256(run_dir / "best.pt"))  # fmt: skip
    assert len(cal.temperatures) == 30 and len(cal.adjustment) == N_CLASSES
    assert (run_dir / "eval" / "val" / "safe.npy").is_file()

    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        f"model: {PAT_SMALL}\n"
        f"checkpoint: {(run_dir / 'best.pt').as_posix()}\n"
        f"label_space: {(run_dir / 'label_space.json').as_posix()}\n"
        f"standardizer: {cfg.standardizer}\nsplit: {cfg.split}\ndata_root: {cfg.data_root}\n"
        f"eval_split: val\nout_dir: {(tmp_path / 'evalout').as_posix()}\nseed: 0\n"
        "calibration: null\n",  # default: calibration.json beside the checkpoint
        encoding="utf-8",
    )
    assert EVAL.main(["--config", str(eval_yaml)]) == 0
    report = json.loads((tmp_path / "evalout/eval/val/calibration_report.json").read_text())
    assert {"ece_k3", "ece_k8", "safe_ece_k3", "n_k3", "uncalibrated"} <= report.keys()

    other = tmp_path / "other.pt"  # same weights, different bytes -> different hash
    shutil.copy(run_dir / "best.pt", other)
    with other.open("ab") as f:
        f.write(b"\0")
    with pytest.raises(CalibrationError, match="fitted for checkpoint"):
        EVAL.main(["--config", str(eval_yaml), "--override", f"checkpoint={other.as_posix()}",
                   f"calibration={(run_dir / 'calibration.json').as_posix()}"])  # fmt: skip


def test_a_seed_gives_the_same_run_whatever_ran_before_it_in_the_process(tmp_path):
    """A Kaggle queue trains several seeds in one process. The model's initial
    weights must come from the seed alone, not from the RNG state the previous
    queue entry left behind (found in plan phase 3, T5: run_training built the
    model before seeding)."""
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    cfg_a = load_config(build_cfg(tmp_path / "a", epochs=1))
    cfg_b = load_config(build_cfg(tmp_path / "b", epochs=1))

    first = R.run_training(cfg_a, seed=3, evaluate_after=False)
    torch.manual_seed(12345)  # another queue entry ran in between
    torch.rand(1000)
    second = R.run_training(cfg_b, seed=3, evaluate_after=False)

    assert first.history[0]["train_loss"] == second.history[0]["train_loss"]
    assert first.history[0]["val_macro_f1"] == second.history[0]["val_macro_f1"]


def test_prefix_training_uses_the_configured_finetune_augmentation(tmp_path, monkeypatch):
    from adl_etc.training import finetune

    calls = []
    real = finetune.AugmentedBatches.batch
    monkeypatch.setattr(finetune.AugmentedBatches, "batch",
                        lambda self, idx: calls.append(len(idx)) or real(self, idx))  # fmt: skip
    cfg = load_config(build_cfg(
        tmp_path, epochs=1, model=PAT_SMALL,
        extra="objective: {name: prefix}\nfinetune_augment: {drop_p: 0.05, ipt_s: 0.1}",
    ))  # fmt: skip

    R.run_training(cfg, seed=0, evaluate_after=False)

    assert sum(calls) == 300  # every training flow, once (one epoch); never validation


def test_an_unknown_objective_or_option_is_refused(tmp_path):
    for i, (extra, message) in enumerate([
        ("objective: {name: prefixx}", "unknown objective 'prefixx'"),
        ("objective: {name: classify, final: true}", "unknown option"),
    ]):  # fmt: skip
        (tmp_path / str(i)).mkdir()
        cfg = load_config(build_cfg(tmp_path / str(i), model=PAT_SMALL, extra=extra))
        with pytest.raises(ValueError, match=message):
            R.run_training(cfg, seed=0)


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
