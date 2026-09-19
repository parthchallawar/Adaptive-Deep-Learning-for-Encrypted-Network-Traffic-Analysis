"""Provenance helpers: what commit, what bytes, when.

Every artefact this project produces (shard sets, standardizer stats, reports)
must be able to name the code and the source bytes that made it. These three
functions are the whole of that contract; nothing else in the codebase should
hash or timestamp things a different way.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def git_commit(cwd: str | Path | None = None) -> str:
    """The current commit hash, or ``"unknown"`` outside a git checkout.

    Never raises: a Kaggle kernel working from an uploaded dataset has no
    ``.git`` directory, and provenance recording must not fail because of that.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            commit = out.stdout.strip()
            if commit:
                return commit
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def stable_hash(obj: Any) -> str:
    """SHA-256 of ``obj`` via sorted-key JSON, so the same content always hashes
    the same way regardless of dict insertion order.

    ``obj`` must be JSON-serialisable (dicts, lists, str, int, float, bool,
    None). NumPy scalars/arrays are not accepted directly; convert with
    ``.tolist()`` first.
    """
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, read in chunks so large archives don't need
    to fit in memory at once."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
