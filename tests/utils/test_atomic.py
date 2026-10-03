"""Atomic writes: a reader sees the old file or the new one, never half."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from adl_etc.utils.atomic import atomic_write_bytes, atomic_write_json, atomic_write_text


def test_writes_and_replaces_content(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b.txt"  # parent directory is created

    atomic_write_text(target, "one")
    assert target.read_text(encoding="utf-8") == "one"

    atomic_write_text(target, "two")
    assert target.read_text(encoding="utf-8") == "two"


def test_leaves_no_temp_files_behind(tmp_path: Path) -> None:
    atomic_write_bytes(tmp_path / "f.bin", b"\x00\x01")

    assert sorted(p.name for p in tmp_path.iterdir()) == ["f.bin"]


def test_a_failed_replace_keeps_the_old_file_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.json"
    atomic_write_text(target, "old")

    def boom(src: str, dst: str) -> None:
        raise OSError("disk went away")

    monkeypatch.setattr(os, "replace", boom)

    with pytest.raises(OSError, match="disk went away"):
        atomic_write_text(target, "new")

    monkeypatch.undo()
    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_unserialisable_json_never_touches_the_file(tmp_path: Path) -> None:
    target = tmp_path / "x.json"
    atomic_write_json(target, {"ok": 1})

    with pytest.raises(TypeError):
        atomic_write_json(target, {"bad": object()})

    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": 1}


def test_json_is_sorted_and_newline_terminated(tmp_path: Path) -> None:
    atomic_write_json(tmp_path / "x.json", {"b": 1, "a": 2})

    text = (tmp_path / "x.json").read_text(encoding="utf-8")
    assert text.endswith("\n") and text.index('"a"') < text.index('"b"')
