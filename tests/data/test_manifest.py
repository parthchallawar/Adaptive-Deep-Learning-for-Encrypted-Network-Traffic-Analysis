"""Dataset manifest: register/verify/require (spec 001, plan T3)."""

from __future__ import annotations

import hashlib

import pytest

from adl_etc.data import manifest as M


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_register_and_require(tmp_path):
    path = tmp_path / "manifest.json"
    entry = M.register(
        "iscx-vpn-2016",
        source_url="https://example.test/vpn",
        files=[{"name": "a.pcap", "bytes": 3, "sha256": "abc"}],
        extracted_to="data/raw/iscx-vpn-2016/pcap",
        manifest_path=path,
    )
    assert entry["source_url"] == "https://example.test/vpn"
    assert entry["exporter_git_commit"] != ""

    got = M.require("iscx-vpn-2016", manifest_path=path)
    assert got == entry


def test_require_missing_entry_names_the_hint(tmp_path):
    path = tmp_path / "manifest.json"
    with pytest.raises(KeyError, match="download_iscx.py"):
        M.require("iscx-vpn-2016", manifest_path=path, hint="scripts/download_iscx.py")


def test_register_replaces_not_merges(tmp_path):
    path = tmp_path / "manifest.json"
    files_v1 = [{"name": "x", "bytes": 1, "sha256": "1"}]
    files_v2 = [{"name": "y", "bytes": 2, "sha256": "2"}]
    M.register("d", source_url="u", files=files_v1, manifest_path=path)
    entry = M.register("d", source_url="u2", files=files_v2, manifest_path=path)
    assert entry["source_url"] == "u2"
    assert [f["name"] for f in entry["files"]] == ["y"]


def test_verify_true_when_files_match(tmp_path):
    data_dir = tmp_path / "pcap"
    data_dir.mkdir()
    content = b"hello world"
    (data_dir / "a.pcap").write_bytes(content)

    manifest_path = tmp_path / "manifest.json"
    M.register(
        "d4",
        source_url="u",
        files=[{"name": "a.pcap", "bytes": len(content), "sha256": _sha256(content)}],
        extracted_to=str(data_dir),
        manifest_path=manifest_path,
    )
    assert M.verify("d4", manifest_path=manifest_path) is True


def test_verify_false_on_missing_entry(tmp_path):
    assert M.verify("nope", manifest_path=tmp_path / "manifest.json") is False


def test_verify_false_on_missing_file(tmp_path):
    data_dir = tmp_path / "pcap"
    data_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    M.register(
        "d4",
        source_url="u",
        files=[{"name": "missing.pcap", "bytes": 1, "sha256": "x"}],
        extracted_to=str(data_dir),
        manifest_path=manifest_path,
    )
    assert M.verify("d4", manifest_path=manifest_path) is False


def test_verify_false_on_truncated_file(tmp_path):
    """A truncated re-download must be caught by hash, not just by size."""
    data_dir = tmp_path / "pcap"
    data_dir.mkdir()
    full = b"x" * 1000
    manifest_path = tmp_path / "manifest.json"
    M.register(
        "d4",
        source_url="u",
        files=[{"name": "a.pcap", "bytes": len(full), "sha256": _sha256(full)}],
        extracted_to=str(data_dir),
        manifest_path=manifest_path,
    )
    (data_dir / "a.pcap").write_bytes(full[:500])  # truncated, same-ish size class
    assert M.verify("d4", manifest_path=manifest_path) is False
