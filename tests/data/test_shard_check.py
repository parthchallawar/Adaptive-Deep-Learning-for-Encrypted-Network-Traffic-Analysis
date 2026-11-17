"""The shard-completeness check (needed 2026-11-17: five D1 weeks were silently damaged)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from adl_etc.data.shard_check import check_dataset, check_period
from tests.training.util import make_flows, write_shards


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    for i, week in enumerate(("WEEK-1", "WEEK-2")):
        write_shards(tmp_path, make_flows(50, 3, seed=i), name="ds", period=week).close()
    return tmp_path / "ds"


def test_a_healthy_export_passes(dataset: Path) -> None:
    flows, problems = check_dataset(dataset)
    assert problems == [] and flows == 100


def test_a_truncated_array_is_reported_with_its_name(dataset: Path) -> None:
    (dataset / "WEEK-2" / "shard-00000" / "ppi.npy").write_bytes(b"")  # the real failure mode
    flows, problems = check_dataset(dataset)
    assert flows == 50  # only the healthy week counts
    assert [str(p) for p in problems if "ppi.npy" in str(p)]
    assert {p.period for p in problems} == {"WEEK-2"}


def test_a_missing_array_and_a_missing_meta_are_reported(dataset: Path) -> None:
    (dataset / "WEEK-1" / "shard-00000" / "ppi_len.npy").unlink()
    shutil.rmtree(dataset / "WEEK-2" / "shard-00000")
    (dataset / "WEEK-2" / "meta.json").unlink()
    n1, bad1 = check_period(dataset / "WEEK-1")
    n2, bad2 = check_period(dataset / "WEEK-2")
    assert any("ppi_len.npy" in p.where and p.what == "missing" for p in bad1)
    assert bad2 and bad2[0].where == "meta.json"


def test_a_half_written_array_is_caught_by_size_not_just_existence(dataset: Path) -> None:
    path = dataset / "WEEK-1" / "shard-00000" / "flowstats.npy"
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    _, problems = check_period(dataset / "WEEK-1")
    assert any("flowstats.npy" in p.where and "does not match" in p.what for p in problems)


def test_row_count_in_meta_is_checked_against_the_arrays(dataset: Path) -> None:
    import json

    meta_path = dataset / "WEEK-1" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["shard_sizes"] = [49]
    meta["n_flows"] = 49
    meta_path.write_text(json.dumps(meta))
    _, problems = check_period(dataset / "WEEK-1")
    assert problems and np.all(["does not match" in p.what for p in problems if "npy" in p.where])
