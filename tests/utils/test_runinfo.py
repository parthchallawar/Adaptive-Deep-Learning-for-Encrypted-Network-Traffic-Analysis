"""Run names and run identity (spec 014, plan phase 2 T1)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from omegaconf import DictConfig

from adl_etc.utils.config import config_hash, load_config
from adl_etc.utils.runinfo import (
    RunInfo,
    code_provenance,
    make_run_name,
    parse_run_name,
)

GIT_IDENTITY = ["-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", *GIT_IDENTITY, *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


# --- run names ---------------------------------------------------------------------


def test_run_name_matches_the_spec_example() -> None:
    name = make_run_name("ft", "pat", "ssl_npp_pfc", "d1m3to6", 0)

    assert name == "ft-pat-ssl_npp_pfc-d1m3to6-s0"


def test_run_name_round_trips() -> None:
    name = make_run_name("bl", "gru", "lstm.h256", "d1_main", 12)

    assert parse_run_name(name) == ("bl", "gru", "lstm.h256", "d1_main", 12)


@pytest.mark.parametrize("field", ["stage", "model", "variant", "split"])
def test_dash_in_a_part_is_rejected(field: str) -> None:
    """A '-' inside a part would make the name impossible to split back apart."""
    parts = {"stage": "ft", "model": "pat", "variant": "v", "split": "d1", field: "a-b"}

    with pytest.raises(ValueError, match="no '-'"):
        make_run_name(seed=0, **parts)


@pytest.mark.parametrize("bad_part", ["", "has space", "slash/x", "é"])
def test_other_illegal_characters_and_empty_parts_are_rejected(bad_part: str) -> None:
    with pytest.raises(ValueError):
        make_run_name("ft", bad_part, "v", "d1", 0)


@pytest.mark.parametrize("bad_seed", [-1, True, 1.5, "0"])
def test_bad_seed_in_a_run_name_is_rejected(bad_seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        make_run_name("ft", "pat", "v", "d1", bad_seed)


@pytest.mark.parametrize(
    "junk",
    ["", "ft-pat-v-d1", "ft-pat-v-d1-0", "ft-pat-v-d1-sx", "ft-pat-v-d1-s0-extra", "a-b-c-d-S1"],
)
def test_parse_rejects_things_that_are_not_run_names(junk: str) -> None:
    with pytest.raises(ValueError):
        parse_run_name(junk)


# --- code provenance ---------------------------------------------------------------


def test_provenance_from_a_git_checkout(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "f.txt").write_text("x\n", encoding="utf-8")
    git(repo, "add", "f.txt")
    git(repo, "commit", "-q", "-m", "init")

    assert code_provenance(repo) == (git(repo, "rev-parse", "HEAD"), False)

    (repo / "f.txt").write_text("y\n", encoding="utf-8")
    assert code_provenance(repo) == (git(repo, "rev-parse", "HEAD"), True)


def test_kaggle_style_root_reads_the_sidecar_files(tmp_path: Path) -> None:
    """No .git (a mounted code dataset): commit and dirtiness come from the
    files ``kaggle_sync.sh push-code`` writes beside ``src/``."""
    (tmp_path / "GIT_COMMIT").write_text("abc123\n", encoding="utf-8")
    (tmp_path / "GIT_DIRTY").write_text("0\n", encoding="utf-8")

    assert code_provenance(tmp_path) == ("abc123", False)

    (tmp_path / "GIT_DIRTY").write_text("1\n", encoding="utf-8")
    assert code_provenance(tmp_path) == ("abc123", True)


def test_missing_dirty_sidecar_counts_as_dirty(tmp_path: Path) -> None:
    """Unknown provenance must not pass the results table's clean-tree check."""
    (tmp_path / "GIT_COMMIT").write_text("abc123\n", encoding="utf-8")

    assert code_provenance(tmp_path) == ("abc123", True)


def test_no_git_and_no_sidecars_is_unknown_and_dirty(tmp_path: Path) -> None:
    assert code_provenance(tmp_path) == ("unknown", True)


def test_empty_commit_sidecar_is_unknown(tmp_path: Path) -> None:
    (tmp_path / "GIT_COMMIT").write_text("\n", encoding="utf-8")

    assert code_provenance(tmp_path)[0] == "unknown"


# --- RunInfo -----------------------------------------------------------------------


def make_cfg(tmp_path: Path, extra: str = "") -> DictConfig:
    path = tmp_path / "c.yaml"
    path.write_text("seed: 0\noptim:\n  lr: 0.1\n" + extra, encoding="utf-8")
    return load_config(path)


def test_collect_records_the_config_hash_and_environment(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    (tmp_path / "GIT_COMMIT").write_text("deadbeef\n", encoding="utf-8")
    (tmp_path / "GIT_DIRTY").write_text("0\n", encoding="utf-8")

    info = RunInfo.collect("ft-pat-v-d1-s0", 0, cfg, code_root=tmp_path)

    assert info.run_name == "ft-pat-v-d1-s0"
    assert info.config_hash == config_hash(cfg)
    assert (info.git_commit, info.git_dirty) == ("deadbeef", False)
    assert info.libraries["numpy"] and info.libraries["omegaconf"]
    assert info.python.count(".") == 2
    assert info.hardware and info.created_at.endswith("Z")


def test_run_info_round_trips_through_json(tmp_path: Path) -> None:
    info = RunInfo.collect("ft-pat-v-d1-s0", 0, make_cfg(tmp_path), code_root=tmp_path)

    restored = RunInfo.from_dict(json.loads(json.dumps(info.to_dict())))

    assert restored == info


def test_from_dict_rejects_missing_and_unexpected_keys(tmp_path: Path) -> None:
    good = RunInfo.collect("ft-pat-v-d1-s0", 0, make_cfg(tmp_path), code_root=tmp_path).to_dict()

    missing = {k: v for k, v in good.items() if k != "config_hash"}
    with pytest.raises(ValueError, match="missing"):
        RunInfo.from_dict(missing)

    with pytest.raises(ValueError, match="unexpected"):
        RunInfo.from_dict({**good, "surprise": 1})


def test_two_different_configs_give_two_different_run_infos(tmp_path: Path) -> None:
    a = RunInfo.collect("r-m-v-s-s0", 0, make_cfg(tmp_path), code_root=tmp_path)
    b = RunInfo.collect("r-m-v-s-s0", 0, make_cfg(tmp_path, "extra: 1\n"), code_root=tmp_path)

    assert a.config_hash != b.config_hash


def test_run_info_is_immutable(tmp_path: Path) -> None:
    info = RunInfo.collect("r-m-v-s-s0", 0, make_cfg(tmp_path), code_root=tmp_path)

    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError, a dataclasses internal
        info.seed = 9  # type: ignore[misc]
