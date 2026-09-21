"""Atomic, safe-to-load, compatibility-checked checkpoints (plan phase 2 T5)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from adl_etc.training.checkpoint import (
    CheckpointError,
    check_compatible,
    load_checkpoint,
    save_checkpoint,
)


class NotSafe:
    """A class a hostile pickle could use to run code on load."""

    def __reduce__(self):
        return (os.getcwd, ())


def payload() -> dict:
    return {
        "epoch": 3,
        "model": {"w": torch.arange(6.0).reshape(2, 3)},
        "history": [{"epoch": 1.0, "val_macro_f1": 0.5}],
        "label_space_hash": "L" * 64,
        "standardizer_hash": "S" * 64,
        "config_hash": "C" * 64,
        "scaler": {},
        "note": None,
    }


def test_round_trip_keeps_tensors_and_plain_types(tmp_path: Path) -> None:
    path = tmp_path / "ck.pt"

    save_checkpoint(path, payload())
    got = load_checkpoint(path)

    assert got["epoch"] == 3 and got["note"] is None and got["scaler"] == {}
    assert torch.equal(got["model"]["w"], torch.arange(6.0).reshape(2, 3))
    assert got["history"] == [{"epoch": 1.0, "val_macro_f1": 0.5}]


def test_a_failed_save_leaves_the_previous_checkpoint_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of atomic writes: a killed session must not corrupt the last good one."""
    path = tmp_path / "ck.pt"
    save_checkpoint(path, {**payload(), "epoch": 1})

    def boom(src: str, dst: str) -> None:
        raise OSError("session killed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="session killed"):
        save_checkpoint(path, {**payload(), "epoch": 2})
    monkeypatch.undo()

    assert load_checkpoint(path)["epoch"] == 1
    assert [p.name for p in tmp_path.iterdir()] == ["ck.pt"]


def test_loading_refuses_arbitrary_objects_instead_of_running_them(tmp_path: Path) -> None:
    """``weights_only`` loading: a checkpoint is data, never code."""
    path = tmp_path / "evil.pt"
    torch.save({"format": 1, "x": NotSafe()}, path)

    with pytest.raises(CheckpointError, match="unreadable or unsafe"):
        load_checkpoint(path)


def test_missing_corrupt_and_wrong_format_files_raise_checkpoint_errors(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "nope.pt")

    (tmp_path / "junk.pt").write_bytes(b"this is not a checkpoint")
    with pytest.raises(CheckpointError, match="unreadable"):
        load_checkpoint(tmp_path / "junk.pt")

    torch.save({"format": 99, "epoch": 1}, tmp_path / "future.pt")
    with pytest.raises(CheckpointError, match="format"):
        load_checkpoint(tmp_path / "future.pt")

    torch.save([1, 2, 3], tmp_path / "list.pt")
    with pytest.raises(CheckpointError, match="format"):
        load_checkpoint(tmp_path / "list.pt")


def test_a_truncated_checkpoint_is_an_error_not_a_partial_load(tmp_path: Path) -> None:
    path = tmp_path / "ck.pt"
    save_checkpoint(path, payload())
    path.write_bytes(path.read_bytes()[:40])

    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_compatibility_checks_label_space_standardizer_and_config() -> None:
    ck = payload()
    ok = dict(label_space_hash="L" * 64, standardizer_hash="S" * 64, config_hash="C" * 64)

    check_compatible(ck, **ok)  # no exception

    with pytest.raises(CheckpointError, match="label space"):
        check_compatible(ck, **{**ok, "label_space_hash": "X" * 64})
    with pytest.raises(CheckpointError, match="standardizer"):
        check_compatible(ck, **{**ok, "standardizer_hash": "X" * 64})
    with pytest.raises(CheckpointError, match="config"):
        check_compatible(ck, **{**ok, "config_hash": "X" * 64})


def test_config_is_only_checked_when_asked_and_no_standardizer_matches_none() -> None:
    ck = {**payload(), "standardizer_hash": None}

    check_compatible(ck, label_space_hash="L" * 64, standardizer_hash=None)  # tokens-only model
    with pytest.raises(CheckpointError, match="standardizer"):
        check_compatible(ck, label_space_hash="L" * 64, standardizer_hash="S" * 64)
