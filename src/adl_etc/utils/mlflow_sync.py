"""Push run directories into a local MLflow store (spec 014).

:class:`~adl_etc.utils.tracking.Tracker` writes plain run directories so that a
Kaggle kernel needs nothing but Python. This module is the other half: on the
machine that has MLflow, it turns those directories, complete or partial, into
MLflow runs.

**Backend: SQLite, not the file store.** MLflow 3.x puts the filesystem backend
in maintenance mode and raises on it unless ``MLFLOW_ALLOW_FILE_STORE=true``.
SQLite needs no server process either (spec 014's real constraint), so the
store is ``sqlite:///results/mlflow.db`` with artifacts under
``results/mlartifacts``. See :func:`local_store`.

**Idempotent.** A run's identity across machines and re-imports is its
:func:`run_key` (name + config hash + start time, all fixed when the run
starts), stored as the tag ``adl.run_key``. Syncing the same directory again
finds the existing MLflow run and adds only what is new: metrics past the
``adl.metrics_synced`` count, params not yet logged, artifacts not yet
uploaded. A run that was PAUSED on Kaggle, resumed and re-pulled therefore
extends the same MLflow run rather than duplicating it.

**Status.** MLflow has no PAUSED. PAUSED, and a run directory that never
reached a terminal status (killed, or a session that ran out), both import as
``KILLED``; the true state is kept in the ``adl.status`` tag. The results-table
script excludes KILLED runs (spec 014).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adl_etc.utils.provenance import stable_hash
from adl_etc.utils.tracking import (
    ARTIFACTS_DIR,
    CONFIG_YAML,
    FINISHED,
    PARAMS_JSON,
    RUN_JSON,
    RUNNING,
    RunRecord,
    TrackingError,
    read_run_dir,
)

DEFAULT_EXPERIMENT = "adl-etc"

TAG_RUN_KEY = "adl.run_key"
TAG_METRICS_SYNCED = "adl.metrics_synced"
TAG_STATUS = "adl.status"

_MAX_PARAM_CHARS = 5_900  # MLflow's limit is 6000
_PARAM_BATCH = 100
_METRIC_BATCH = 1_000


class SyncError(RuntimeError):
    """A run directory that cannot be reconciled with what MLflow already holds."""


@dataclass(frozen=True)
class SyncResult:
    run_dir: Path
    run_id: str
    created: bool
    metrics_logged: int
    artifacts_logged: int
    mlflow_status: str
    adl_status: str


def local_store(results_dir: str | Path) -> tuple[str, str]:
    """``(tracking_uri, artifact_root_uri)`` for the project's local store."""
    results = Path(results_dir).resolve()
    return f"sqlite:///{(results / 'mlflow.db').as_posix()}", (results / "mlartifacts").as_uri()


def run_key(record: RunRecord) -> str:
    """Identity of a run across machines and re-imports.

    Name + config hash + the millisecond the run first started. The start time
    is written once and preserved when a run is resumed, so a run PAUSED on
    Kaggle and resumed later keeps its key, while a fresh attempt with the same
    name and config (a different directory, a different start) gets its own.
    """
    info = record.info
    return stable_hash(
        {
            "run_name": info.run_name,
            "config_hash": info.config_hash,
            "started_at_ms": record.started_at_ms,
        }
    )


def find_run_dirs(root: str | Path) -> list[Path]:
    """Every directory under ``root`` (including ``root``) that holds a ``run.json``."""
    root = Path(root)
    found = [p.parent for p in root.rglob(RUN_JSON)]
    return sorted(set(found))


def _param_str(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    if len(text) > _MAX_PARAM_CHARS:
        return text[:_MAX_PARAM_CHARS] + f"...[truncated, {len(text)} chars; see {PARAMS_JSON}]"
    return text


def _mlflow_status(record: RunRecord, finalize_running: bool) -> str | None:
    """MLflow status to set, or ``None`` to leave the run RUNNING."""
    if record.status == FINISHED:
        return "FINISHED"
    if record.status == RUNNING and not finalize_running:
        return None
    # KILLED, PAUSED, and a RUNNING directory that will never be finalised.
    return "KILLED"


def _existing_artifacts(client: Any, run_id: str, path: str | None = None) -> set[str]:
    found: set[str] = set()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            found |= _existing_artifacts(client, run_id, item.path)
        else:
            found.add(item.path)
    return found


def _artifact_files(run_dir: Path) -> Iterator[tuple[Path, str | None]]:
    """``(local file, artifact subdir)`` for everything worth uploading."""
    artifacts = run_dir / ARTIFACTS_DIR
    if artifacts.is_dir():
        for f in sorted(p for p in artifacts.rglob("*") if p.is_file()):
            rel = f.parent.relative_to(artifacts).as_posix()
            yield f, (None if rel == "." else rel)


def sync_run_dir(
    run_dir: str | Path,
    *,
    tracking_uri: str,
    artifact_root: str | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
    finalize_running: bool = True,
) -> SyncResult:
    """Create or extend the MLflow run for ``run_dir``. Safe to call repeatedly.

    ``finalize_running=False`` is for syncing a run that is still live: it is
    left RUNNING instead of being marked KILLED.
    """
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param, RunTag

    rd = Path(run_dir)
    try:
        record = read_run_dir(rd)
    except TrackingError as e:
        raise SyncError(str(e)) from e

    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment)
    exp_id = (
        exp.experiment_id
        if exp is not None
        else client.create_experiment(experiment, artifact_location=artifact_root)
    )

    key = run_key(record)
    matches = client.search_runs([exp_id], filter_string=f"tags.`{TAG_RUN_KEY}` = '{key}'")
    if len(matches) > 1:
        raise SyncError(f"{len(matches)} MLflow runs share run key {key[:12]} for {rd}")

    info = record.info
    created = not matches
    if created:
        run = client.create_run(
            exp_id,
            start_time=record.started_at_ms,
            run_name=info.run_name,
            tags={
                TAG_RUN_KEY: key,
                "git_commit": info.git_commit,
                "git_dirty": str(info.git_dirty).lower(),
                "config_hash": info.config_hash,
                "seed": str(info.seed),
                "adl.hardware": info.hardware[:250],
            },
        )
    else:
        run = matches[0]
    run_id = run.info.run_id
    tags = run.data.tags

    # -- params: write-once, so only add what is missing, and never change one.
    have = run.data.params
    new_params: list[Param] = []
    for k, v in record.params.items():
        text = _param_str(v)
        if k in have:
            if have[k] != text:
                raise SyncError(f"param {k!r} is {have[k]!r} in MLflow but {text!r} in {rd}")
        else:
            new_params.append(Param(k, text))
    for i in range(0, len(new_params), _PARAM_BATCH):
        client.log_batch(run_id, params=new_params[i : i + _PARAM_BATCH])

    # -- metrics: append-only, resumed from the count already synced.
    already = int(tags.get(TAG_METRICS_SYNCED, "0"))
    if len(record.metrics) < already:
        raise SyncError(
            f"{rd} has {len(record.metrics)} metrics but MLflow already holds {already} for this "
            "run: this is an older copy of the run directory, refusing to sync it over a newer one"
        )
    fresh = record.metrics[already:]
    for i in range(0, len(fresh), _METRIC_BATCH):
        client.log_batch(
            run_id,
            metrics=[
                Metric(m.key, m.value, m.ts, 0 if m.step is None else m.step)
                for m in fresh[i : i + _METRIC_BATCH]
            ],
        )

    # -- artifacts: new files once; run metadata always refreshed (status changes).
    present = _existing_artifacts(client, run_id)
    uploaded = 0
    for local, subdir in _artifact_files(rd):
        remote = f"{subdir}/{local.name}" if subdir else local.name
        if remote in present:
            continue
        client.log_artifact(run_id, str(local), subdir)
        uploaded += 1
    for name in (RUN_JSON, PARAMS_JSON, CONFIG_YAML):
        if (rd / name).is_file():
            client.log_artifact(run_id, str(rd / name), "run")

    # -- status and bookkeeping tags.
    status = _mlflow_status(record, finalize_running)
    client.log_batch(
        run_id,
        tags=[
            RunTag(TAG_METRICS_SYNCED, str(len(record.metrics))),
            RunTag(TAG_STATUS, record.status),
            RunTag("adl.resume_count", str(record.resume_count)),
        ],
    )
    if status is None:
        client.update_run(run_id, status="RUNNING")
    else:
        client.set_terminated(run_id, status=status, end_time=record.ended_at_ms)

    return SyncResult(
        run_dir=rd,
        run_id=run_id,
        created=created,
        metrics_logged=len(fresh),
        artifacts_logged=uploaded,
        mlflow_status=status or "RUNNING",
        adl_status=record.status,
    )


def sync_tree(
    root: str | Path,
    *,
    tracking_uri: str,
    artifact_root: str | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
    finalize_running: bool = True,
) -> list[SyncResult | tuple[Path, Exception]]:
    """Sync every run directory under ``root``. One bad directory does not stop
    the rest; each failure is returned as ``(run_dir, exception)`` in place of a
    result so the caller can report all of them."""
    out: list[SyncResult | tuple[Path, Exception]] = []
    for rd in find_run_dirs(root):
        try:
            out.append(
                sync_run_dir(
                    rd,
                    tracking_uri=tracking_uri,
                    artifact_root=artifact_root,
                    experiment=experiment,
                    finalize_running=finalize_running,
                )
            )
        except (SyncError, TrackingError) as e:
            out.append((rd, e))
    return out


__all__ = [
    "DEFAULT_EXPERIMENT",
    "SyncError",
    "SyncResult",
    "find_run_dirs",
    "local_store",
    "run_key",
    "sync_run_dir",
    "sync_tree",
]
