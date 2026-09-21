"""Run identity: what a run was called, and what code, config and machine made it.

Spec 014's contract is that every reported number traces to a run name, a config
hash and a commit. :class:`RunInfo` is that trace, captured once at run start
and written with every run's artefacts.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from adl_etc.utils.config import config_hash
from adl_etc.utils.provenance import git_commit, git_dirty, now_iso

# A name part may not contain "-": the run name is "-"-joined, and a variant
# such as "ssl-npp" would make it impossible to split back into its parts.
# Spec 014's own example uses underscores ("ssl_npp_pfc") for this reason.
_NAME_PART = re.compile(r"[A-Za-z0-9_.]+")
_SEED_PART = re.compile(r"s(\d+)")

# Libraries whose versions decide whether two runs are comparable. Absent ones
# are simply omitted from the record.
_TRACKED_LIBRARIES = (
    "numpy",
    "pandas",
    "pyarrow",
    "omegaconf",
    "torch",
    "scikit-learn",
    "xgboost",
    "mlflow",
    "dpkt",
)

GIT_COMMIT_SIDECAR = "GIT_COMMIT"
GIT_DIRTY_SIDECAR = "GIT_DIRTY"


def make_run_name(stage: str, model: str, variant: str, split: str, seed: int) -> str:
    """``<stage>-<model>-<variant>-<split>-s<seed>``, e.g.
    ``ft-pat-ssl_npp_pfc-d1m3to6-s0``."""
    for label, part in (("stage", stage), ("model", model), ("variant", variant), ("split", split)):
        if not _NAME_PART.fullmatch(part):
            raise ValueError(
                f"run-name {label} {part!r} must match [A-Za-z0-9_.]+ "
                "(no '-': it separates the parts)"
            )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    return f"{stage}-{model}-{variant}-{split}-s{seed}"


def parse_run_name(name: str) -> tuple[str, str, str, str, int]:
    """Inverse of :func:`make_run_name`. The results-table script groups runs
    by every part except the seed."""
    parts = name.split("-")
    if len(parts) != 5:
        raise ValueError(f"not a run name (expected 5 '-'-separated parts): {name!r}")
    stage, model, variant, split, seed_part = parts
    m = _SEED_PART.fullmatch(seed_part)
    if m is None or not all(_NAME_PART.fullmatch(p) for p in (stage, model, variant, split)):
        raise ValueError(f"not a run name: {name!r}")
    return stage, model, variant, split, int(m.group(1))


def _default_code_root() -> Path:
    # src/adl_etc/utils/runinfo.py -> repository root (or, on Kaggle, the root
    # of the mounted code dataset, which holds src/ beside GIT_COMMIT).
    return Path(__file__).resolve().parents[3]


def _read_sidecar(root: Path, name: str) -> str | None:
    try:
        text = (root / name).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def code_provenance(code_root: str | Path | None = None) -> tuple[str, bool]:
    """``(commit, dirty)`` for the code that is running.

    In a git checkout both come from git. A Kaggle kernel has no ``.git``, so
    ``scripts/kaggle_sync.sh push-code`` writes ``GIT_COMMIT`` and ``GIT_DIRTY``
    beside ``src/`` and they are read from there instead.

    When dirtiness cannot be established either way it is reported as ``True``.
    Unknown provenance must not pass the results-table script's clean-tree check.
    """
    root = Path(code_root) if code_root is not None else _default_code_root()

    commit = git_commit(root)
    if commit == "unknown":
        commit = _read_sidecar(root, GIT_COMMIT_SIDECAR) or "unknown"

    dirty = git_dirty(root)
    if dirty is None:
        flag = _read_sidecar(root, GIT_DIRTY_SIDECAR)
        dirty = True if flag is None else flag not in {"0", "false", "False"}
    return commit, dirty


def library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _TRACKED_LIBRARIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def describe_hardware() -> str:
    """One line: OS, CPU count and, if torch sees one, the GPU."""
    parts = [platform.platform(), f"cpus={os.cpu_count()}"]
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            names = sorted({torch.cuda.get_device_name(i) for i in range(n)})
            parts.append(f"gpu={'+'.join(names)} x{n}")
        else:
            parts.append("gpu=none")
    except ImportError:
        parts.append("gpu=unknown (torch not installed)")
    return "; ".join(parts)


@dataclass(frozen=True)
class RunInfo:
    run_name: str
    seed: int
    git_commit: str
    git_dirty: bool
    config_hash: str
    python: str
    libraries: dict[str, str]
    hardware: str
    created_at: str

    @classmethod
    def collect(
        cls,
        run_name: str,
        seed: int,
        cfg: DictConfig,
        *,
        code_root: str | Path | None = None,
    ) -> RunInfo:
        commit, dirty = code_provenance(code_root)
        return cls(
            run_name=run_name,
            seed=seed,
            git_commit=commit,
            git_dirty=dirty,
            config_hash=config_hash(cfg),
            python=sys.version.split()[0],
            libraries=library_versions(),
            hardware=describe_hardware(),
            created_at=now_iso(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunInfo:
        expected = {f.name for f in fields(cls)}
        got = set(data)
        if got != expected:
            raise ValueError(
                f"RunInfo keys mismatch: missing {sorted(expected - got)}, "
                f"unexpected {sorted(got - expected)}"
            )
        return cls(**{**data, "libraries": dict(data["libraries"])})
