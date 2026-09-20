"""Dataset manifest (spec 001): register(), verify(), require().

``data/manifest.json`` records, for every downloaded dataset, where it came
from and the hash of every file, so that a shard set's ``meta.json`` can
always name the exact bytes it was built from, and so a script never silently
trains on a truncated or stale download.

Schema (one entry per dataset name, matching the plan's example)::

    {
      "iscx-vpn-2016": {
        "source_url": "...",
        "retrieved": "2026-09-20T12:00:00Z",
        "files": [{"name": "aim_chat_3a.pcap", "bytes": 123, "sha256": "..."}],
        "extracted_to": "data/raw/iscx-vpn-2016/pcap",
        "exporter_git_commit": "..."
      }
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adl_etc.utils.provenance import file_sha256, git_commit, now_iso

DEFAULT_MANIFEST_PATH = Path("data/manifest.json")


def _load(path: Path) -> dict[str, Any]:
    if path.exists():
        return dict(json.loads(path.read_text(encoding="utf-8")))
    return {}


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def register(
    name: str,
    *,
    source_url: str,
    files: list[dict[str, Any]],
    extracted_to: str | None = None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    **extra: Any,
) -> dict[str, Any]:
    """Record (or replace) one dataset's manifest entry.

    ``files`` is a list of ``{"name", "bytes", "sha256"}`` dicts, one per
    downloaded/extracted file. Replacing rather than merging is deliberate: a
    re-run after a partial or failed download should describe exactly what is
    on disk now, not accumulate entries for files that may no longer exist.
    """
    path = Path(manifest_path)
    data = _load(path)
    entry: dict[str, Any] = {
        "source_url": source_url,
        "retrieved": now_iso(),
        "files": files,
        "extracted_to": extracted_to,
        "exporter_git_commit": git_commit(),
        **extra,
    }
    data[name] = entry
    _save(path, data)
    return entry


def verify(
    name: str,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> bool:
    """Re-hashes every file in ``name``'s manifest entry against
    ``extracted_to``. Returns ``False`` rather than raising for any failure
    (missing entry, missing file, hash mismatch) so callers can decide how to
    react — typically by re-running the downloader or falling back to another
    acquisition path (spec 001's Path A/Path B distinction for D1)."""
    path = Path(manifest_path)
    entry = _load(path).get(name)
    if entry is None or entry.get("extracted_to") is None:
        return False
    base = Path(entry["extracted_to"])
    for f in entry["files"]:
        file_path = base / f["name"]
        if not file_path.exists():
            return False
        if file_sha256(file_path) != f["sha256"]:
            return False
    return True


def require(
    name: str,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    hint: str = "",
) -> dict[str, Any]:
    """Returns ``name``'s manifest entry, or raises ``KeyError`` naming the
    download script that would produce it (``hint``)."""
    path = Path(manifest_path)
    entry = _load(path).get(name)
    if entry is None:
        msg = f"no manifest entry for {name!r} in {path}."
        if hint:
            msg += f" Run `{hint}` first."
        raise KeyError(msg)
    return entry
