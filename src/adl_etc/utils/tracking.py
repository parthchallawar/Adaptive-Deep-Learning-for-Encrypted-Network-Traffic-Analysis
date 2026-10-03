"""Run tracking: a durable, backend-free record of each run (spec 014).

A run's record is a plain directory, written by :class:`Tracker` and read back
by :func:`read_run_dir`::

    <run_dir>/
      run.json              RunInfo + status + timestamps
      params.json           flat {key: value}, write-once per key
      metrics.jsonl         one {"key","value","step","ts"} per line, append-only
      config_resolved.yaml  the resolved config (spec 014)
      artifacts/            small files (reports, figures); never checkpoints

**Why not log to MLflow directly.** A Kaggle kernel may not have mlflow
installed and may have no internet to install it, and MLflow's own on-disk
formats have been changing (the filesystem backend is in maintenance mode and
raises by default in 3.x). A run must not lose its results to either. So this
module imports nothing from mlflow: it works anywhere Python does, and
:mod:`adl_etc.utils.mlflow_sync` pushes finished or partial run directories
into an MLflow store on the machine that has one.

Single writer per run directory is assumed.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import TracebackType
from typing import Any

from omegaconf import DictConfig

from adl_etc.utils.atomic import atomic_write_json
from adl_etc.utils.config import flatten, save_resolved
from adl_etc.utils.runinfo import RunInfo

SCHEMA_VERSION = 1

RUNNING = "RUNNING"
FINISHED = "FINISHED"
KILLED = "KILLED"
PAUSED = "PAUSED"
STATUSES = (RUNNING, FINISHED, KILLED, PAUSED)

RUN_JSON = "run.json"
PARAMS_JSON = "params.json"
METRICS_JSONL = "metrics.jsonl"
CONFIG_YAML = "config_resolved.yaml"
ARTIFACTS_DIR = "artifacts"

_MAX_ERROR_CHARS = 500


class TrackingError(RuntimeError):
    """A run directory that cannot be written or read consistently."""


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class MetricPoint:
    key: str
    value: float
    step: int | None
    ts: int


@dataclass(frozen=True)
class RunRecord:
    """Everything :class:`Tracker` wrote, read back."""

    info: RunInfo
    status: str
    started_at_ms: int
    ended_at_ms: int | None
    resume_count: int
    error: str | None
    params: dict[str, Any]
    metrics: list[MetricPoint]
    run_dir: Path


def _jsonable(value: Any) -> Any:
    """Params are stored as JSON; refuse what would round-trip lossily."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    # numpy scalars and friends
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    raise TypeError(f"param value {value!r} ({type(value).__name__}) is not JSON-serialisable")


def _check_subdir(subdir: str) -> None:
    """Reject anything that could resolve outside ``artifacts/``.

    Checked under both Windows and POSIX rules regardless of the host: on
    Windows ``Path("/abs").is_absolute()`` is False (no drive), yet joining it
    onto another path discards everything but the drive letter, so a rooted
    subdir would escape the run directory.
    """
    for flavour in (PurePosixPath, PureWindowsPath):
        path = flavour(subdir)
        if path.anchor or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"artifact subdir must be relative and inside the run: {subdir!r}")


class Tracker:
    """Writes one run's record. Use :meth:`start`, ideally as a context manager
    so that an exception marks the run ``KILLED`` instead of leaving it
    ``RUNNING`` forever."""

    def __init__(self, run_dir: Path | None, enabled: bool) -> None:
        self._run_dir = run_dir
        self.enabled = enabled
        self._params: dict[str, Any] = {}
        self._run: dict[str, Any] = {}
        self._closed = False

    # -- construction -------------------------------------------------------

    @classmethod
    def start(
        cls,
        info: RunInfo,
        cfg: DictConfig,
        run_dir: str | Path,
        *,
        resume: bool = False,
        enabled: bool = True,
    ) -> Tracker:
        """Begin (or, with ``resume=True``, continue) a run.

        ``enabled=False`` returns a tracker whose every method is a no-op and
        which writes nothing to disk: the ``--smoke`` run and unit tests use it
        so they never touch a results directory.

        An existing run directory is refused unless ``resume=True``, and a
        resume must present the same ``config_hash``: continuing a run under a
        changed config produces a record that describes neither config.
        """
        if not enabled:
            return cls(None, enabled=False)

        rd = Path(run_dir)
        existing = (rd / RUN_JSON).is_file()
        if existing and not resume:
            raise TrackingError(
                f"{rd} already holds a run; pass resume=True to continue it, "
                "or choose a different run directory"
            )
        if resume and not existing:
            raise TrackingError(f"cannot resume: no run found in {rd}")

        tracker = cls(rd, enabled=True)
        rd.mkdir(parents=True, exist_ok=True)

        if existing:
            record = read_run_dir(rd)
            if record.info.config_hash != info.config_hash:
                raise TrackingError(
                    f"cannot resume {rd}: config_hash {info.config_hash[:12]} differs from "
                    f"the run's {record.info.config_hash[:12]}"
                )
            if (record.info.git_commit, record.info.git_dirty) != (info.git_commit, info.git_dirty):
                raise TrackingError(
                    f"cannot resume {rd}: the code changed mid-run "
                    f"({record.info.git_commit[:12]} dirty={record.info.git_dirty} -> "
                    f"{info.git_commit[:12]} dirty={info.git_dirty}); a run resumed on different "
                    "code has no single provenance, so start a new run instead"
                )
            if record.info.run_name != info.run_name:
                raise TrackingError(
                    f"cannot resume {rd}: run name {info.run_name!r} differs from "
                    f"the run's {record.info.run_name!r}"
                )
            tracker._params = dict(record.params)
            tracker._run = {
                "schema_version": SCHEMA_VERSION,
                "info": record.info.to_dict(),  # the original identity, not the resumer's
                "status": RUNNING,
                "started_at_ms": record.started_at_ms,
                "ended_at_ms": None,
                "resume_count": record.resume_count + 1,
                "error": None,
            }
        else:
            tracker._run = {
                "schema_version": SCHEMA_VERSION,
                "info": info.to_dict(),
                "status": RUNNING,
                "started_at_ms": _now_ms(),
                "ended_at_ms": None,
                "resume_count": 0,
                "error": None,
            }
            save_resolved(cfg, rd / CONFIG_YAML)
            (rd / METRICS_JSONL).touch()

        tracker._flush_run()
        tracker.log_params(
            {
                **flatten(cfg),
                "run_name": info.run_name,
                "seed": info.seed,
                "git_commit": info.git_commit,
                "git_dirty": info.git_dirty,
                "config_hash": info.config_hash,
            }
        )
        return tracker

    # -- properties ---------------------------------------------------------

    @property
    def run_dir(self) -> Path:
        if self._run_dir is None:
            raise TrackingError("tracking is disabled: there is no run directory")
        return self._run_dir

    @property
    def status(self) -> str:
        return str(self._run.get("status", RUNNING)) if self.enabled else RUNNING

    # -- logging ------------------------------------------------------------

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Write-once per key: logging a key again with the same value is a
        no-op, with a different value raises (MLflow's own rule, kept so the
        two never disagree about what a run's parameters were)."""
        if not self.enabled:
            return
        self._require_open()
        clean = {str(k): _jsonable(v) for k, v in params.items()}
        changed = False
        for key, value in clean.items():
            if key in self._params:
                if self._params[key] != value:
                    raise TrackingError(
                        f"param {key!r} already logged as {self._params[key]!r}; "
                        f"refusing to change it to {value!r}"
                    )
                continue
            self._params[key] = value
            changed = True
        if changed:
            atomic_write_json(self.run_dir / PARAMS_JSON, self._params)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        if not self.enabled:
            return
        self._require_open()
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise ValueError(f"step must be a non-negative int or None, got {step!r}")
        ts = _now_ms()
        lines = []
        for key, value in metrics.items():
            number = float(value)
            # NaN/inf are allowed on purpose: a diverged loss is a result worth keeping.
            lines.append(
                json.dumps(
                    {"key": str(key), "value": number, "step": step, "ts": ts}, allow_nan=True
                )
            )
        if not lines:
            return
        with open(self.run_dir / METRICS_JSONL, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
            fh.flush()

    def log_artifact(self, path: str | Path, subdir: str | None = None) -> Path | None:
        """Copy a small file into ``artifacts/`` (or ``artifacts/<subdir>/``).

        Checkpoints are deliberately not logged this way: they are large and
        are recorded as a param holding their path instead (spec 014).
        """
        if not self.enabled:
            return None
        self._require_open()
        src = Path(path)
        if not src.is_file():
            raise TrackingError(f"artifact is not a file: {src}")
        if subdir is not None:
            _check_subdir(subdir)
        dest_dir = self.run_dir / ARTIFACTS_DIR / (subdir or "")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        return dest

    # -- status -------------------------------------------------------------

    def set_status(self, status: str) -> None:
        """Set the terminal status (``FINISHED``, ``KILLED`` or ``PAUSED``).
        ``PAUSED`` is a run that saved its state and exited to be resumed."""
        if not self.enabled:
            return
        if status not in (FINISHED, KILLED, PAUSED):
            raise ValueError(f"status must be FINISHED, KILLED or PAUSED, got {status!r}")
        self._run["status"] = status
        self._run["ended_at_ms"] = _now_ms()
        self._flush_run()

    def close(self, status: str | None = None) -> None:
        """Finish the run. Idempotent. A run still ``RUNNING`` becomes
        ``FINISHED``; one already given a status keeps it."""
        if not self.enabled or self._closed:
            return
        if status is not None:
            self.set_status(status)
        elif self._run["status"] == RUNNING:
            self.set_status(FINISHED)
        self._closed = True

    def __enter__(self) -> Tracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None and self.enabled and not self._closed:
            self._run["error"] = f"{exc_type.__name__}: {exc}"[:_MAX_ERROR_CHARS]
            self.close(KILLED)
        else:
            self.close()
        # returns None: never swallows the exception

    # -- internals ----------------------------------------------------------

    def _require_open(self) -> None:
        if self._closed:
            raise TrackingError("run is closed")

    def _flush_run(self) -> None:
        atomic_write_json(self.run_dir / RUN_JSON, self._run)


def read_run_dir(run_dir: str | Path) -> RunRecord:
    """Read a run directory back. Tolerates a truncated *final* metrics line (a
    process killed mid-write) but not a corrupt one in the middle, which would
    mean something other than a crash damaged the record."""
    rd = Path(run_dir)
    run_path = rd / RUN_JSON
    if not run_path.is_file():
        raise TrackingError(f"not a run directory (no {RUN_JSON}): {rd}")
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        info = RunInfo.from_dict(run["info"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise TrackingError(f"unreadable {run_path}: {e}") from e
    if run.get("schema_version") != SCHEMA_VERSION:
        raise TrackingError(
            f"{run_path}: schema_version {run.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
    status = run.get("status")
    if status not in STATUSES:
        raise TrackingError(f"{run_path}: unknown status {status!r}")

    params_path = rd / PARAMS_JSON
    params: dict[str, Any] = (
        json.loads(params_path.read_text(encoding="utf-8")) if params_path.is_file() else {}
    )

    metrics: list[MetricPoint] = []
    metrics_path = rd / METRICS_JSONL
    if metrics_path.is_file():
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                point = MetricPoint(
                    key=str(row["key"]),
                    value=float(row["value"]),
                    step=None if row["step"] is None else int(row["step"]),
                    ts=int(row["ts"]),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                if i == len(lines) - 1:
                    break  # truncated tail from a killed writer
                raise TrackingError(f"{metrics_path} line {i + 1} is corrupt: {e}") from e
            metrics.append(point)

    return RunRecord(
        info=info,
        status=status,
        started_at_ms=int(run["started_at_ms"]),
        ended_at_ms=None if run.get("ended_at_ms") is None else int(run["ended_at_ms"]),
        resume_count=int(run.get("resume_count", 0)),
        error=run.get("error"),
        params=params,
        metrics=metrics,
        run_dir=rd,
    )
