"""Run directory -> local MLflow, idempotently (spec 014, plan phase 2 T2).

Every test uses one shared SQLite store (creating it costs a few seconds) and
gives each run its own name, so runs never see each other.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from omegaconf import DictConfig

from adl_etc.utils.config import config_hash, load_config
from adl_etc.utils.mlflow_sync import (
    TAG_METRICS_SYNCED,
    TAG_RUN_KEY,
    TAG_STATUS,
    SyncError,
    SyncResult,
    find_run_dirs,
    local_store,
    run_key,
    sync_run_dir,
    sync_tree,
)
from adl_etc.utils.runinfo import RunInfo
from adl_etc.utils.tracking import (
    FINISHED,
    METRICS_JSONL,
    PARAMS_JSON,
    PAUSED,
    Tracker,
    read_run_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Store:
    uri: str
    artifact_root: str
    client: Any

    def sync(self, run_dir: Path, **kw: Any) -> SyncResult:
        return sync_run_dir(run_dir, tracking_uri=self.uri, artifact_root=self.artifact_root, **kw)

    def runs_named(self, name: str) -> list[Any]:
        exp = self.client.get_experiment_by_name("adl-etc")
        if exp is None:
            return []
        runs = self.client.search_runs(
            [exp.experiment_id], filter_string=f"attributes.run_name = '{name}'"
        )
        return list(runs)


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Store]:
    pytest.importorskip("mlflow", reason="needs the `train` extra: pip install -e '.[train]'")
    from mlflow import MlflowClient

    uri, artifact_root = local_store(tmp_path_factory.mktemp("results"))
    yield Store(uri, artifact_root, MlflowClient(tracking_uri=uri))


@pytest.fixture
def work(tmp_path: Path) -> Path:
    return tmp_path


def make_cfg(work: Path, extra: str = "") -> DictConfig:
    path = work / "cfg.yaml"
    path.write_text("seed: 0\noptim:\n  lr: 0.1\n" + extra, encoding="utf-8")
    return load_config(path)


def make_run(
    work: Path,
    name: str,
    *,
    epochs: int = 3,
    end: str | None = FINISHED,
    extra_cfg: str = "",
    subdir: str = "runs",
) -> Path:
    """A run directory with ``epochs`` epochs of metrics and one artifact.
    ``end=None`` leaves the run un-finalised (a killed session)."""
    root = work / "code"
    root.mkdir(exist_ok=True)
    (root / "GIT_COMMIT").write_text("abc1234\n", encoding="utf-8")
    (root / "GIT_DIRTY").write_text("0\n", encoding="utf-8")
    cfg = make_cfg(work, extra_cfg)
    info = RunInfo.collect(name, 0, cfg, code_root=root)
    rd = work / subdir / name

    t = Tracker.start(info, cfg, rd)
    for e in range(epochs):
        t.log_metrics({"loss": 1.0 / (e + 1), "val_f1": 0.1 * (e + 1)}, step=e)
    report = work / f"report-{name}.json"
    report.write_text(json.dumps({"acc": 0.9}), encoding="utf-8")
    t.log_artifact(report, "eval")
    if end is None:
        return rd  # never closed
    t.close(end)
    return rd


# --- a first import -------------------------------------------------------------------------


def test_sync_creates_a_run_with_everything_in_it(store: Store, work: Path) -> None:
    rd = make_run(work, "sync-a-first-s0", epochs=3)
    record = read_run_dir(rd)

    result = store.sync(rd)

    assert result.created and result.mlflow_status == "FINISHED" and result.metrics_logged == 6
    run = store.client.get_run(result.run_id)
    assert run.info.run_name == "sync-a-first-s0"
    assert run.info.status == "FINISHED"
    assert run.info.start_time == record.started_at_ms
    assert run.info.end_time == record.ended_at_ms
    # provenance tags and params (spec 014: traceable to a run name, config hash and commit)
    assert run.data.tags["git_commit"] == "abc1234" and run.data.tags["git_dirty"] == "false"
    assert run.data.tags["config_hash"] == record.info.config_hash
    assert run.data.tags[TAG_RUN_KEY] == run_key(record)
    assert run.data.params["optim.lr"] == "0.1" and run.data.params["seed"] == "0"
    assert run.data.params["config_hash"] == record.info.config_hash
    # metrics, with their steps and original timestamps
    history = store.client.get_metric_history(result.run_id, "loss")
    assert [(m.step, m.value) for m in history] == [(0, 1.0), (1, 0.5), (2, pytest.approx(1 / 3))]
    assert [m.timestamp for m in history] == [p.ts for p in record.metrics if p.key == "loss"]


def test_artifacts_and_run_metadata_are_uploaded(store: Store, work: Path) -> None:
    rd = make_run(work, "sync-b-artifacts-s0")

    result = store.sync(rd)

    listed = {a.path for a in store.client.list_artifacts(result.run_id)}
    assert {"eval", "run"} <= listed
    assert [a.path for a in store.client.list_artifacts(result.run_id, "eval")] == [
        "eval/report-sync-b-artifacts-s0.json"
    ]
    assert {a.path for a in store.client.list_artifacts(result.run_id, "run")} == {
        "run/run.json",
        "run/params.json",
        "run/config_resolved.yaml",
    }
    assert result.artifacts_logged == 1  # the eval report; run metadata isn't counted


# --- idempotence ------------------------------------------------------------------------------


def test_syncing_twice_leaves_one_run_and_adds_nothing(store: Store, work: Path) -> None:
    rd = make_run(work, "sync-c-twice-s0")

    first = store.sync(rd)
    second = store.sync(rd)

    assert first.run_id == second.run_id and not second.created
    assert second.metrics_logged == 0 and second.artifacts_logged == 0
    assert len(store.runs_named("sync-c-twice-s0")) == 1
    assert len(store.client.get_metric_history(first.run_id, "loss")) == 3  # not doubled


def test_a_resumed_run_extends_the_same_mlflow_run(store: Store, work: Path) -> None:
    """The Kaggle story: PAUSED, synced, resumed on a later push, pulled again."""
    rd = make_run(work, "sync-d-resume-s0", epochs=2, end=PAUSED)
    paused = store.sync(rd)
    run = store.client.get_run(paused.run_id)
    assert paused.mlflow_status == "KILLED" and run.data.tags[TAG_STATUS] == PAUSED
    assert run.data.tags[TAG_METRICS_SYNCED] == "4"

    record = read_run_dir(rd)
    t = Tracker.start(record.info, load_config(rd / "config_resolved.yaml"), rd, resume=True)
    for e in (2, 3):
        t.log_metrics({"loss": 1.0 / (e + 1), "val_f1": 0.1 * (e + 1)}, step=e)
    t.close()
    resumed = store.sync(rd)

    assert resumed.run_id == paused.run_id and not resumed.created
    assert resumed.metrics_logged == 4  # only the new epochs
    assert resumed.mlflow_status == "FINISHED"
    run = store.client.get_run(resumed.run_id)
    assert run.info.status == "FINISHED" and run.data.tags[TAG_STATUS] == FINISHED
    assert run.data.tags["adl.resume_count"] == "1"
    assert [m.step for m in store.client.get_metric_history(resumed.run_id, "loss")] == [0, 1, 2, 3]
    assert len(store.runs_named("sync-d-resume-s0")) == 1


# --- statuses ------------------------------------------------------------------------------------


def test_a_run_that_never_finished_imports_as_killed(store: Store, work: Path) -> None:
    rd = make_run(work, "sync-e-unfinished-s0", end=None)

    result = store.sync(rd)

    assert result.adl_status == "RUNNING" and result.mlflow_status == "KILLED"
    assert store.client.get_run(result.run_id).info.status == "KILLED"


def test_live_sync_leaves_a_running_run_running(store: Store, work: Path) -> None:
    rd = make_run(work, "sync-f-live-s0", end=None)

    result = store.sync(rd, finalize_running=False)

    assert result.mlflow_status == "RUNNING"
    assert store.client.get_run(result.run_id).info.status == "RUNNING"


# --- refusing to corrupt what's there ------------------------------------------------------------


def test_an_older_copy_of_a_run_is_refused(store: Store, work: Path) -> None:
    """Pulling an old kernel output over a newer sync must not silently win."""
    rd = make_run(work, "sync-g-stale-s0", epochs=4)
    store.sync(rd)
    path = rd / METRICS_JSONL
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:4]) + "\n", encoding="utf-8")  # only 2 of 4 epochs

    with pytest.raises(SyncError, match="older copy"):
        store.sync(rd)


def test_a_changed_param_is_refused(store: Store, work: Path) -> None:
    rd = make_run(work, "sync-h-param-s0")
    store.sync(rd)
    params = json.loads((rd / PARAMS_JSON).read_text(encoding="utf-8"))
    params["optim.lr"] = 0.5
    (rd / PARAMS_JSON).write_text(json.dumps(params), encoding="utf-8")

    with pytest.raises(SyncError, match="optim.lr"):
        store.sync(rd)


def test_same_name_with_a_different_config_is_a_separate_run(store: Store, work: Path) -> None:
    """Config drift is reported by the results-table script, not hidden here:
    the two runs stay distinguishable by config_hash."""
    a = make_run(work, "sync-i-drift-s0", subdir="a")
    b = make_run(work, "sync-i-drift-s0", extra_cfg="extra: 1\n", subdir="b")

    ra, rb = store.sync(a), store.sync(b)

    assert ra.run_id != rb.run_id
    hashes = {r.data.tags["config_hash"] for r in store.runs_named("sync-i-drift-s0")}
    assert hashes == {config_hash(make_cfg(work)), config_hash(make_cfg(work, "extra: 1\n"))}


def test_an_unreadable_run_dir_is_a_sync_error(store: Store, work: Path) -> None:
    (work / "junk").mkdir()
    (work / "junk" / "run.json").write_text("{ not json", encoding="utf-8")

    with pytest.raises(SyncError, match="unreadable"):
        store.sync(work / "junk")


# --- values that need care -----------------------------------------------------------------------


def test_nan_and_a_very_long_param_survive(store: Store, work: Path) -> None:
    rd = make_run(work, "sync-j-edge-s0", epochs=1)
    root = work / "code"
    cfg = make_cfg(work)
    info = RunInfo.collect("sync-j-edge-s0", 0, cfg, code_root=root)
    del info  # the run already exists; reopen it to add edge values
    record = read_run_dir(rd)
    t = Tracker.start(record.info, cfg, rd, resume=True)
    t.log_params({"classes": list(range(3000))})  # far beyond MLflow's 6000-char limit
    t.log_metrics({"diverged": float("nan")}, step=0)
    t.close()

    result = store.sync(rd)

    param = store.client.get_run(result.run_id).data.params["classes"]
    assert param.endswith("see params.json]") and "truncated" in param and len(param) < 6000
    full = json.loads((rd / PARAMS_JSON).read_text(encoding="utf-8"))["classes"]
    assert full == list(range(3000))  # the untruncated value is still on disk
    (nan_point,) = store.client.get_metric_history(result.run_id, "diverged")
    assert math.isnan(nan_point.value)


# --- trees ---------------------------------------------------------------------------------------


def test_find_run_dirs_finds_nested_runs_in_order(work: Path) -> None:
    make_run(work, "tree-b-s0", subdir="x/y")
    make_run(work, "tree-a-s0", subdir="x")

    found = find_run_dirs(work)

    assert [p.name for p in found] == ["tree-a-s0", "tree-b-s0"]
    assert find_run_dirs(work / "empty") == []


def test_sync_tree_reports_a_bad_dir_without_stopping_the_rest(store: Store, work: Path) -> None:
    make_run(work, "tree-c-s0", subdir="batch")
    make_run(work, "tree-d-s0", subdir="batch")
    bad = work / "batch" / "broken"
    bad.mkdir()
    (bad / "run.json").write_text("{ nope", encoding="utf-8")

    results = sync_tree(work / "batch", tracking_uri=store.uri, artifact_root=store.artifact_root)

    ok = [r for r in results if isinstance(r, SyncResult)]
    failed = [r for r in results if not isinstance(r, SyncResult)]
    assert len(ok) == 2 and len(failed) == 1
    assert failed[0][0] == bad and isinstance(failed[0][1], SyncError)


# --- the CLI -------------------------------------------------------------------------------------


def test_import_script_end_to_end(work: Path) -> None:
    """Dry run touches nothing; a real run imports; a repeat updates nothing;
    an unreadable dir makes the exit code non-zero. One subprocess flow because
    importing mlflow is slow."""
    pytest.importorskip("mlflow", reason="needs the `train` extra")
    make_run(work, "cli-a-s0", subdir="pulled")
    make_run(work, "cli-b-s0", subdir="pulled", end=PAUSED)
    results = work / "results"

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "mlflow_import.py"),
             "--src", str(work / "pulled"), "--results", str(results), *args],
            capture_output=True, text=True, check=False,
        )  # fmt: skip

    dry = run("--dry-run")
    assert dry.returncode == 0 and "cli-a-s0" in dry.stdout and "PAUSED" in dry.stdout
    assert not (results / "mlflow.db").exists()  # a dry run creates no store

    first = run()
    assert first.returncode == 0, first.stderr
    assert first.stdout.count("created") == 2 and "2 synced, 0 failed" in first.stdout
    assert (results / "mlflow.db").is_file()

    again = run()
    assert again.returncode == 0
    assert again.stdout.count("updated") == 2 and "+0 metrics" in again.stdout

    broken = work / "pulled" / "broken"
    broken.mkdir()
    (broken / "run.json").write_text("{ nope", encoding="utf-8")
    bad = run()
    assert bad.returncode == 1 and "FAILED" in bad.stderr and "2 synced, 1 failed" in bad.stdout
