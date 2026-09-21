"""Checkpoints: atomic to write, restricted to load, refuse to mix (spec 015).

A checkpoint is written to a temp file and renamed, so a Kaggle session killed
mid-write leaves the previous checkpoint intact rather than a truncated one
(spec 015's edge case). It records the label-space hash and standardizer hash
it was trained with, so loading it against different ones fails at load time
and not as a quietly wrong metric an hour later.

Loading uses ``weights_only=True``: the payload is tensors and plain Python
types only, so a corrupt or hostile file cannot execute code.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch

from adl_etc.utils.atomic import atomic_write_bytes

FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """A checkpoint that is missing, unreadable, or incompatible with this run."""


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write ``payload`` (tensors and plain Python types only)."""
    buf = io.BytesIO()
    torch.save({"format": FORMAT_VERSION, **payload}, buf)
    atomic_write_bytes(path, buf.getvalue())


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:  # torch raises a zoo of types for corrupt/unsafe files
        raise CheckpointError(f"unreadable or unsafe checkpoint {path}: {e}") from e
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_VERSION:
        got = payload.get("format") if isinstance(payload, dict) else type(payload).__name__
        raise CheckpointError(f"{path}: format {got!r}, expected {FORMAT_VERSION}")
    return payload


def check_compatible(
    ckpt: dict[str, Any],
    *,
    label_space_hash: str,
    standardizer_hash: str | None,
    config_hash: str | None = None,
) -> None:
    """Raise if this checkpoint was made under a different label space,
    standardizer or (when given) config than the run trying to use it."""
    checks = {
        "label space": (ckpt.get("label_space_hash"), label_space_hash),
        "standardizer": (ckpt.get("standardizer_hash"), standardizer_hash),
    }
    if config_hash is not None:
        checks["config"] = (ckpt.get("config_hash"), config_hash)
    for what, (theirs, ours) in checks.items():
        if theirs != ours:
            raise CheckpointError(
                f"checkpoint was made with a different {what}: "
                f"{str(theirs)[:12]} (checkpoint) vs {str(ours)[:12]} (this run)"
            )
