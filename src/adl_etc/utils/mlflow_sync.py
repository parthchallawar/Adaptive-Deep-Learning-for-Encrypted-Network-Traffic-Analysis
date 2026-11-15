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


# --- evaluation reports (plan phase 3, T8) ---------------------------------------------------
#
# A run's training metrics come from its run directory; its evaluation reports do not:
# ``eval/val/report.json`` sits beside them, and a robustness pass writes
# ``<out_root>/<run name>/eval/<split>/`` (``report.json``, ``robustness.json``,
# ``predictions.npz``) outside any run directory (kernel ``run_eval_entry``). Each such
# directory is attached to the one finished MLflow run of that name: scalars become
# metrics named ``<split>.<metric>`` (``test_id.acc_k5``, ``test_id.drop_10.auc_k``)
# and the files become artifacts under ``eval/<split>/``. A tag holds a hash of what
# was attached, so attaching the same evaluation again does nothing, and attaching a
# *different* one under the same split is refused instead of silently mixing two.

EVAL_REPORT = "report.json"
EVAL_ROBUSTNESS = "robustness.json"
EVAL_FILES = (EVAL_REPORT, EVAL_ROBUSTNESS, "predictions.npz", "calibration_report.json")


@dataclass(frozen=True)
class AttachResult:
    eval_dir: Path
    run_name: str
    split: str
    run_id: str
    attached: bool  # False: this exact evaluation was already attached
    metrics_logged: int


def find_eval_dirs(root: str | Path) -> list[tuple[str, str, Path]]:
    """``(run name, split, directory)`` for every ``<run name>/eval/<split>/report.json``
    under ``root`` whose ``<run name>`` is one of ours (``parse_run_name``). A
    robustness pass's per-condition reports (``.../perturb/<name>/report.json``) are
    summarised by its ``robustness.json`` and not attached on their own."""
    from adl_etc.utils.runinfo import parse_run_name

    found = []
    for report in Path(root).rglob(EVAL_REPORT):
        split_dir = report.parent
        if split_dir.parent.name != "eval":
            continue
        name = split_dir.parent.parent.name
        try:
            parse_run_name(name)
        except ValueError:
            continue
        found.append((name, split_dir.name, split_dir))
    return sorted(found)


def eval_metrics(split: str, report: dict[str, Any], robustness: dict[str, Any] | None) -> dict:
    """The scalars of one evaluation, as ``{"<split>.<metric>": value}``."""
    out: dict[str, float] = {}
    for k, v in report["acc_at_k"].items():
        out[f"{split}.acc_k{k}"] = float(v)
    for k, v in (report.get("auroc_at_k") or {}).items():
        if v is not None:
            out[f"{split}.auroc_k{k}"] = float(v)
    for name in ("auc_k", "macro_f1", "balanced_accuracy", "k95", "n_flows", "n_unknown"):
        out[f"{split}.{name}"] = float(report[name])
    rel = report.get("reliability") or {}
    if rel.get("ece") is not None:
        out[f"{split}.ece_k{rel['k']}"] = float(rel["ece"])
    for cond, row in ((robustness or {}).get("conditions") or {}).items():
        if cond == "clean":
            continue  # identical to the report's own values above
        out[f"{split}.{cond}.auc_k"] = float(row["auc_k"])
        out[f"{split}.{cond}.macro_f1"] = float(row["macro_f1"])
        for k, v in row["acc_at_k"].items():
            out[f"{split}.{cond}.acc_k{k}"] = float(v)
    return out


def attach_eval(
    eval_dir: str | Path,
    *,
    run_name: str,
    split: str,
    tracking_uri: str,
    experiment: str = DEFAULT_EXPERIMENT,
) -> AttachResult:
    """Attach one evaluation directory to the finished MLflow run named ``run_name``
    (see the section comment above). Safe to repeat."""
    import time

    from mlflow import MlflowClient
    from mlflow.entities import Metric, RunTag

    ed = Path(eval_dir)
    report = json.loads((ed / EVAL_REPORT).read_text(encoding="utf-8"))
    rob_path = ed / EVAL_ROBUSTNESS
    robustness = json.loads(rob_path.read_text(encoding="utf-8")) if rob_path.is_file() else None

    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SyncError(f"no MLflow experiment {experiment!r}; import the run directories first")
    candidates = [
        r
        for r in client.search_runs(
            [exp.experiment_id], filter_string=f"attributes.run_name = '{run_name}'"
        )
        if r.data.tags.get(TAG_STATUS) == FINISHED
    ]
    if not candidates:
        raise SyncError(
            f"{ed}: no finished MLflow run named {run_name!r}; import its run directory first"
        )
    if len(candidates) > 1:
        hashes = sorted(r.data.tags.get("config_hash", "")[:12] for r in candidates)
        raise SyncError(f"{ed}: {len(candidates)} finished MLflow runs named {run_name!r} {hashes}")
    run = candidates[0]
    run_id = run.info.run_id

    digest = stable_hash(
        {
            "report": report.get("hash", ""),
            "robustness": None if robustness is None else robustness.get("report_hashes"),
        }
    )
    tag = f"adl.eval.{split}.hash"
    have = run.data.tags.get(tag)
    if have == digest:
        return AttachResult(ed, run_name, split, run_id, attached=False, metrics_logged=0)
    if have is not None:
        raise SyncError(
            f"{ed}: run {run_name!r} already has a different {split!r} evaluation attached "
            f"({have[:12]} vs {digest[:12]}); refusing to mix two evaluations"
        )

    now = int(time.time() * 1000)
    values = eval_metrics(split, report, robustness)
    metrics = [Metric(k, v, now, 0) for k, v in sorted(values.items())]
    for i in range(0, len(metrics), _METRIC_BATCH):
        client.log_batch(run_id, metrics=metrics[i : i + _METRIC_BATCH])
    for name in EVAL_FILES:
        if (ed / name).is_file():
            client.log_artifact(run_id, str(ed / name), f"eval/{split}")
    client.log_batch(run_id, tags=[RunTag(tag, digest)])
    return AttachResult(ed, run_name, split, run_id, attached=True, metrics_logged=len(metrics))


__all__ = [
    "DEFAULT_EXPERIMENT",
    "AttachResult",
    "SyncError",
    "SyncResult",
    "attach_eval",
    "eval_metrics",
    "find_eval_dirs",
    "find_run_dirs",
    "local_store",
    "run_key",
    "sync_run_dir",
    "sync_tree",
]
