"""Tracker and the run-directory format (spec 014, plan phase 2 T2).

Nothing here imports mlflow: that is the point of the format.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
from omegaconf import DictConfig

from adl_etc.utils.config import config_hash, flatten, load_config
from adl_etc.utils.runinfo import RunInfo
from adl_etc.utils.tracking import (
    ARTIFACTS_DIR,
    FINISHED,
    KILLED,
    METRICS_JSONL,
    PARAMS_JSON,
    PAUSED,
    RUN_JSON,
    RUNNING,
    Tracker,
    TrackingError,
    read_run_dir,
)


def make_cfg(tmp_path: Path, extra: str = "") -> DictConfig:
    path = tmp_path / "cfg.yaml"
    path.write_text("seed: 0\noptim:\n  lr: 0.1\n" + extra, encoding="utf-8")
    return load_config(path)


def make_info(
    tmp_path: Path,
    cfg: DictConfig,
    *,
    name: str = "bl-cnn-v-d4-s0",
    commit: str = "c0ffee",
    dirty: bool = False,
) -> RunInfo:
    root = tmp_path / "code"
    root.mkdir(exist_ok=True)
    (root / "GIT_COMMIT").write_text(commit + "\n", encoding="utf-8")
    (root / "GIT_DIRTY").write_text(("1" if dirty else "0") + "\n", encoding="utf-8")
    return RunInfo.collect(name, 0, cfg, code_root=root)


@pytest.fixture
def cfg(tmp_path: Path) -> DictConfig:
    return make_cfg(tmp_path)


@pytest.fixture
def info(tmp_path: Path, cfg: DictConfig) -> RunInfo:
    return make_info(tmp_path, cfg)


# --- what start() writes -----------------------------------------------------------


def test_start_writes_the_run_record_and_the_mandatory_params(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    rd = tmp_path / "run"

    Tracker.start(info, cfg, rd)

    record = read_run_dir(rd)
    assert record.status == RUNNING and record.info == info and record.resume_count == 0
    assert record.ended_at_ms is None and record.metrics == []
    assert (rd / "config_resolved.yaml").is_file()
    # Spec 014: params always carry the flattened config plus the provenance quartet.
    assert record.params == {
        **flatten(cfg),
        "run_name": info.run_name,
        "seed": 0,
        "git_commit": "c0ffee",
        "git_dirty": False,
        "config_hash": config_hash(cfg),
    }


def test_existing_run_directory_is_refused_without_resume(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    rd = tmp_path / "run"
    Tracker.start(info, cfg, rd).close()

    with pytest.raises(TrackingError, match="resume=True"):
        Tracker.start(info, cfg, rd)


def test_resume_without_an_existing_run_is_refused(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    with pytest.raises(TrackingError, match="no run found"):
        Tracker.start(info, cfg, tmp_path / "nothing", resume=True)


# --- params ---------------------------------------------------------------------------


def test_params_are_write_once(tmp_path: Path, cfg: DictConfig, info: RunInfo) -> None:
    t = Tracker.start(info, cfg, tmp_path / "run")

    t.log_params({"ckpt": "a.pt"})
    t.log_params({"ckpt": "a.pt"})  # same value: fine

    with pytest.raises(TrackingError, match="refusing to change"):
        t.log_params({"ckpt": "b.pt"})
    assert read_run_dir(t.run_dir).params["ckpt"] == "a.pt"


def test_config_keys_and_numpy_scalars_are_accepted_as_params(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    import numpy as np

    t = Tracker.start(info, cfg, tmp_path / "run")
    t.log_params({"n": np.int64(5), "lr": np.float32(0.5), "ids": [1, 2], "d": {"a": None}})

    params = read_run_dir(t.run_dir).params
    assert params["n"] == 5 and params["lr"] == 0.5 and params["ids"] == [1, 2]
    assert params["d"] == {"a": None}


def test_unserialisable_param_is_rejected(tmp_path: Path, cfg: DictConfig, info: RunInfo) -> None:
    t = Tracker.start(info, cfg, tmp_path / "run")

    with pytest.raises(TypeError, match="not JSON-serialisable"):
        t.log_params({"bad": object()})


# --- metrics --------------------------------------------------------------------------


def test_metrics_round_trip_with_steps_in_order(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    t = Tracker.start(info, cfg, tmp_path / "run")

    t.log_metrics({"loss": 0.9, "acc": 0.1}, step=0)
    t.log_metrics({"loss": 0.5}, step=1)
    t.log_metrics({"final_f1": 0.77})  # no step

    points = read_run_dir(t.run_dir).metrics
    assert [(p.key, p.value, p.step) for p in points] == [
        ("loss", 0.9, 0),
        ("acc", 0.1, 0),
        ("loss", 0.5, 1),
        ("final_f1", 0.77, None),
    ]
    assert all(p.ts > 1_600_000_000_000 for p in points)  # epoch milliseconds


def test_nan_and_inf_are_kept_because_a_diverged_loss_is_a_result(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    t = Tracker.start(info, cfg, tmp_path / "run")

    t.log_metrics({"loss": float("nan")}, step=3)
    t.log_metrics({"loss": float("inf")}, step=4)

    a, b = read_run_dir(t.run_dir).metrics
    assert math.isnan(a.value) and math.isinf(b.value)


@pytest.mark.parametrize("bad_step", [-1, 1.5, True, "0"])
def test_bad_step_is_rejected(tmp_path: Path, cfg: DictConfig, info: RunInfo, bad_step) -> None:
    t = Tracker.start(info, cfg, tmp_path / "run")

    with pytest.raises(ValueError, match="step"):
        t.log_metrics({"x": 1.0}, step=bad_step)


def test_a_truncated_final_metrics_line_is_tolerated(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    """A process killed mid-write leaves a partial last line; that must not
    make the whole run unreadable."""
    t = Tracker.start(info, cfg, tmp_path / "run")
    t.log_metrics({"loss": 1.0}, step=0)
    t.log_metrics({"loss": 0.5}, step=1)
    with open(t.run_dir / METRICS_JSONL, "a", encoding="utf-8") as fh:
        fh.write('{"key": "loss", "val')  # cut off

    assert [p.step for p in read_run_dir(t.run_dir).metrics] == [0, 1]


def test_a_corrupt_middle_metrics_line_is_an_error(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    t = Tracker.start(info, cfg, tmp_path / "run")
    t.log_metrics({"loss": 1.0}, step=0)
    t.log_metrics({"loss": 0.5}, step=1)
    path = t.run_dir / METRICS_JSONL
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], "not json", lines[1]]) + "\n", encoding="utf-8")

    with pytest.raises(TrackingError, match="line 2 is corrupt"):
        read_run_dir(t.run_dir)


# --- artifacts -------------------------------------------------------------------------


def test_artifacts_are_copied_under_the_run(tmp_path: Path, cfg: DictConfig, info: RunInfo) -> None:
    t = Tracker.start(info, cfg, tmp_path / "run")
    src = tmp_path / "report.json"
    src.write_text('{"acc": 1}', encoding="utf-8")

    plain = t.log_artifact(src)
    nested = t.log_artifact(src, "eval/d1")

    assert plain == t.run_dir / ARTIFACTS_DIR / "report.json"
    assert nested == t.run_dir / ARTIFACTS_DIR / "eval" / "d1" / "report.json"
    assert plain and plain.read_text(encoding="utf-8") == '{"acc": 1}'


@pytest.mark.parametrize(
    "bad",
    [
        "../escape",
        "a/../../b",
        "/abs",  # rooted but driveless: is_absolute() is False on Windows, yet it escapes
        r"\abs",
        "C:/abs",
        "C:rel",  # drive-relative
        r"..\up",
        r"a\..\..\b",
    ],
)
def test_artifact_subdir_cannot_escape_the_run(
    tmp_path: Path, cfg: DictConfig, info: RunInfo, bad: str
) -> None:
    t = Tracker.start(info, cfg, tmp_path / "run")
    src = tmp_path / "f.txt"
    src.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="inside the run"):
        t.log_artifact(src, bad)


def test_missing_artifact_file_is_an_error(tmp_path: Path, cfg: DictConfig, info: RunInfo) -> None:
    t = Tracker.start(info, cfg, tmp_path / "run")

    with pytest.raises(TrackingError, match="not a file"):
        t.log_artifact(tmp_path / "nope.txt")


# --- lifecycle --------------------------------------------------------------------------


def test_clean_exit_finishes_the_run(tmp_path: Path, cfg: DictConfig, info: RunInfo) -> None:
    with Tracker.start(info, cfg, tmp_path / "run") as t:
        t.log_metrics({"x": 1.0})

    record = read_run_dir(t.run_dir)
    assert record.status == FINISHED and record.ended_at_ms is not None and record.error is None


def test_an_exception_kills_the_run_records_why_and_still_propagates(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    rd = tmp_path / "run"

    with pytest.raises(RuntimeError, match="OOM"), Tracker.start(info, cfg, rd):
        raise RuntimeError("OOM at epoch 3")

    record = read_run_dir(rd)
    assert record.status == KILLED and record.error == "RuntimeError: OOM at epoch 3"


def test_keyboard_interrupt_also_kills_the_run(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    rd = tmp_path / "run"

    with pytest.raises(KeyboardInterrupt), Tracker.start(info, cfg, rd):
        raise KeyboardInterrupt

    assert read_run_dir(rd).status == KILLED


def test_a_status_set_before_exit_is_kept(tmp_path: Path, cfg: DictConfig, info: RunInfo) -> None:
    """T8's time guard sets PAUSED and exits cleanly; exit must not overwrite it."""
    with Tracker.start(info, cfg, tmp_path / "run") as t:
        t.set_status(PAUSED)

    assert read_run_dir(t.run_dir).status == PAUSED


def test_close_is_idempotent_and_logging_after_close_fails(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    t = Tracker.start(info, cfg, tmp_path / "run")
    t.close()
    t.close()

    with pytest.raises(TrackingError, match="closed"):
        t.log_metrics({"x": 1.0})
    with pytest.raises(TrackingError, match="closed"):
        t.log_params({"x": 1})


def test_set_status_rejects_running_and_junk(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    t = Tracker.start(info, cfg, tmp_path / "run")

    for bad in (RUNNING, "DONE", ""):
        with pytest.raises(ValueError, match="FINISHED, KILLED or PAUSED"):
            t.set_status(bad)


# --- disabled ----------------------------------------------------------------------------


def test_disabled_tracker_writes_nothing_and_every_call_is_a_no_op(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    rd = tmp_path / "never"
    src = tmp_path / "f.txt"
    src.write_text("x", encoding="utf-8")

    with Tracker.start(info, cfg, rd, enabled=False) as t:
        t.log_params({"a": 1})
        t.log_metrics({"x": 1.0}, step=0)
        assert t.log_artifact(src) is None
        t.set_status(PAUSED)

    assert not rd.exists()
    with pytest.raises(TrackingError, match="disabled"):
        _ = t.run_dir


def test_disabled_tracker_still_propagates_exceptions(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    with (
        pytest.raises(ValueError, match="boom"),
        Tracker.start(info, cfg, tmp_path / "x", enabled=False),
    ):
        raise ValueError("boom")


# --- resume -------------------------------------------------------------------------------


def test_resume_continues_the_same_run(tmp_path: Path, cfg: DictConfig, info: RunInfo) -> None:
    rd = tmp_path / "run"
    first = Tracker.start(info, cfg, rd)
    first.log_params({"ckpt": "epoch1.pt"})
    first.log_metrics({"loss": 1.0}, step=0)
    first.close(PAUSED)
    before = read_run_dir(rd)

    later = RunInfo.collect(info.run_name, 0, cfg, code_root=tmp_path / "code")  # a new process
    second = Tracker.start(later, cfg, rd, resume=True)
    second.log_metrics({"loss": 0.5}, step=1)
    second.close()

    after = read_run_dir(rd)
    assert after.resume_count == 1 and after.status == FINISHED
    assert after.started_at_ms == before.started_at_ms
    assert after.info == before.info  # the original identity, not the resumer's timestamp
    assert after.params["ckpt"] == "epoch1.pt"
    assert [(p.step, p.value) for p in after.metrics] == [(0, 1.0), (1, 0.5)]


def test_resume_under_a_changed_config_is_refused(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    rd = tmp_path / "run"
    Tracker.start(info, cfg, rd).close(PAUSED)
    changed_cfg = make_cfg(tmp_path, "extra: 1\n")

    with pytest.raises(TrackingError, match="config_hash"):
        Tracker.start(make_info(tmp_path, changed_cfg), changed_cfg, rd, resume=True)


def test_resume_on_different_code_is_refused_with_a_clear_message(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    rd = tmp_path / "run"
    Tracker.start(info, cfg, rd).close(PAUSED)

    with pytest.raises(TrackingError, match="code changed mid-run"):
        Tracker.start(make_info(tmp_path, cfg, commit="deadbeef"), cfg, rd, resume=True)
    with pytest.raises(TrackingError, match="code changed mid-run"):
        Tracker.start(make_info(tmp_path, cfg, dirty=True), cfg, rd, resume=True)


def test_resume_under_a_different_run_name_is_refused(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    rd = tmp_path / "run"
    Tracker.start(info, cfg, rd).close(PAUSED)

    with pytest.raises(TrackingError, match="run name"):
        Tracker.start(make_info(tmp_path, cfg, name="bl-cnn-v-d4-s1"), cfg, rd, resume=True)


# --- reading ------------------------------------------------------------------------------


def test_reading_a_non_run_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(TrackingError, match="not a run directory"):
        read_run_dir(tmp_path)


def test_unknown_schema_version_or_status_is_rejected(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    rd = tmp_path / "run"
    Tracker.start(info, cfg, rd).close()
    doc = json.loads((rd / RUN_JSON).read_text(encoding="utf-8"))

    (rd / RUN_JSON).write_text(json.dumps({**doc, "schema_version": 99}), encoding="utf-8")
    with pytest.raises(TrackingError, match="schema_version"):
        read_run_dir(rd)

    (rd / RUN_JSON).write_text(json.dumps({**doc, "status": "DONE"}), encoding="utf-8")
    with pytest.raises(TrackingError, match="unknown status"):
        read_run_dir(rd)


def test_a_run_directory_is_readable_without_params_or_metrics_files(
    tmp_path: Path, cfg: DictConfig, info: RunInfo
) -> None:
    rd = tmp_path / "run"
    Tracker.start(info, cfg, rd).close()
    (rd / PARAMS_JSON).unlink()
    (rd / METRICS_JSONL).unlink()

    record = read_run_dir(rd)
    assert record.params == {} and record.metrics == []


# --- the property the whole design rests on -------------------------------------------------


def test_tracking_never_imports_mlflow() -> None:
    """A Kaggle kernel may have no mlflow. Run this in a fresh interpreter so an
    earlier test that imported mlflow cannot mask a regression."""
    import subprocess

    code = (
        "import sys\n"
        "import adl_etc.utils.tracking, adl_etc.utils.runinfo, adl_etc.utils.config\n"
        "bad = sorted(m for m in sys.modules if m == 'mlflow' or m.startswith('mlflow.'))\n"
        "print('LEAK:' + ','.join(bad[:5]) if bad else 'CLEAN')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "CLEAN", out.stdout
