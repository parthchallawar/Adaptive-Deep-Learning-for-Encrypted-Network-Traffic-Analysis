"""``training.run_xgb`` end to end: config -> split -> XGBBaseline.fit ->
saved boosters + report (B1, spec 005, plan T8-follow-up).

XGBoost needs real (if tiny) learnable data on real shards, so this uses the
same synthetic-flow helpers as ``test_run.py``, but with no ``Standardizer``
needed at all -- B1's features come straight from ``prefix_flowstats``.
"""

from __future__ import annotations

import pytest

xgboost = pytest.importorskip("xgboost")

from adl_etc.training import run_xgb as X  # noqa: E402
from adl_etc.training.run import is_finished, run_dir_for  # noqa: E402
from adl_etc.utils.config import load_config  # noqa: E402
from tests.training.util import make_flows, write_shards  # noqa: E402

N_CLASSES = 4


def build_cfg(tmp_path, *, results_root=None):
    data_root = tmp_path / "data"
    train_data = make_flows(300, N_CLASSES, seed=0)
    val_data = make_flows(100, N_CLASSES, seed=1)
    write_shards(data_root, train_data, name="synthetic", period="train", n_classes=N_CLASSES)
    write_shards(data_root, val_data, name="synthetic", period="val", n_classes=N_CLASSES)

    split_yaml = tmp_path / "split.yaml"
    split_yaml.write_text(
        """
dataset: synthetic
temporal: false
splits:
  train: {periods: [train]}
  val: {periods: [val]}
""",
        encoding="utf-8",
    )

    rr = results_root if results_root is not None else tmp_path / "results"
    cfg_yaml = tmp_path / "train.yaml"
    cfg_yaml.write_text(
        f"""
model:
  name: xgb
  n_estimators: 50
  learning_rate: 0.3
  max_depth: 3
  subsample: 0.8
  colsample_bytree: 0.8
  early_stopping_rounds: 10
  class_weight_cap: 10.0
  n_jobs: 1
  max_train_flows: null
seed: 0
split: {split_yaml.as_posix()}
data_root: {data_root.as_posix()}
standardizer: results/standardizer.json
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


def test_run_training_fits_every_k_and_writes_a_real_report(tmp_path):
    cfg = load_config(build_cfg(tmp_path))
    summary = X.run_training(cfg, seed=0)

    run_dir = run_dir_for(cfg, 0)
    assert summary is not None
    assert set(summary) == {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30}  # spec 004's grid
    assert all(0.0 <= v["val_macro_f1"] <= 1.0 for v in summary.values())
    assert (run_dir / "model" / "meta.json").is_file()
    assert (run_dir / "label_space.json").is_file()
    assert (run_dir / "eval" / "val" / "report.json").is_file()
    assert is_finished(run_dir)


def test_run_training_report_reflects_a_model_that_actually_learned(tmp_path):
    from adl_etc.evaluation.report import Report

    cfg = load_config(build_cfg(tmp_path))
    X.run_training(cfg, seed=0)

    report = Report.load(run_dir_for(cfg, 0) / "eval" / "val" / "report.json")
    assert report.n_flows == 100
    assert report.n_classes == N_CLASSES
    # The synthetic classes are easy to separate from PPI size/timing alone.
    assert report.acc_at_k[report.ks[-1]] > 0.7


def test_a_finished_run_is_not_retrained_on_a_second_call(tmp_path):
    cfg = load_config(build_cfg(tmp_path))
    first = X.run_training(cfg, seed=0)
    assert first is not None

    second = X.run_training(cfg, seed=0)
    assert second is None  # skipped: is_finished() already true, nothing reloaded


def test_run_training_writes_the_lowercase_status_is_finished_actually_checks(tmp_path):
    """A regression guard for the exact bug this module hit once already:
    is_finished() reads training.loop's lowercase status strings, not
    Tracker's own (RUNNING/FINISHED/KILLED/PAUSED)."""
    import json

    cfg = load_config(build_cfg(tmp_path))
    X.run_training(cfg, seed=0)

    state = json.loads((run_dir_for(cfg, 0) / X.STATE_JSON).read_text(encoding="utf-8"))
    assert state["status"] == "finished"


def test_main_cli_runs_one_seed_end_to_end(tmp_path):
    cfg_yaml = build_cfg(tmp_path)
    cfg = load_config(cfg_yaml)

    rc = X.main(["--config", str(cfg_yaml)])
    assert rc == 0
    assert is_finished(run_dir_for(cfg, 0))


def test_main_cli_seed_override_wins_over_the_configs_own_seed(tmp_path):
    cfg_yaml = build_cfg(tmp_path)
    cfg = load_config(cfg_yaml)

    X.main(["--config", str(cfg_yaml), "--seed", "7"])
    assert is_finished(run_dir_for(cfg, 7))
    assert not is_finished(run_dir_for(cfg, 0))
