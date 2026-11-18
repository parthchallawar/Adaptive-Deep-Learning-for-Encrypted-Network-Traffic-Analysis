"""Is a shard export complete and undamaged? (found necessary 2026-11-17, plan phase 4)

A Kaggle output pull can stop early and still exit 0 (docs/kaggle-workflow.md), and that is
how five of the 42 locally stored D1 weeks (48 to 52) ended up with empty or missing arrays
without anything failing: ``load_split`` only reads labels and timestamps, so the damage showed
up only when a model first read ``ppi.npy``.

:func:`check_period` verifies one period directory against its own ``meta.json``: every array in
:data:`~adl_etc.data.tensors.ARRAY_SPEC` exists in every shard, and each file's size is exactly the
shard's row count times the row size plus a small ``.npy`` header. Sizes only; nothing is loaded.
No torch.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adl_etc.data.tensors import ARRAY_SPEC

#: Bytes an ``.npy`` file may hold beyond its data. numpy pads the header to a multiple of 64
#: bytes (128 for every array here); anything looser would hide a one-row discrepancy, since the
#: smallest row is 1 byte and the largest 240.
HEADER_SIZES = (64, 128, 192, 256)


@dataclass(frozen=True)
class Problem:
    period: str
    where: str
    what: str

    def __str__(self) -> str:
        return f"{self.period}: {self.where}: {self.what}"


def _row_bytes(name: str) -> int:
    dtype, shape = ARRAY_SPEC[name]
    return int(np.prod(shape, dtype=np.int64)) * dtype.itemsize if shape else dtype.itemsize


def check_period(period_dir: Path) -> tuple[int, list[Problem]]:
    """``(n_flows_expected, problems)`` for one period directory."""
    name = period_dir.name
    meta_path = period_dir / "meta.json"
    if not meta_path.is_file():
        return 0, [Problem(name, "meta.json", "missing")]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sizes = [int(s) for s in meta["shard_sizes"]]
        n_flows = int(meta["n_flows"])
    except (ValueError, KeyError, TypeError) as err:
        return 0, [Problem(name, "meta.json", f"unreadable ({err})")]
    problems: list[Problem] = []
    if sum(sizes) != n_flows:
        problems.append(
            Problem(name, "meta.json", f"shard_sizes sum {sum(sizes)} != n_flows {n_flows}")
        )
    shard_dirs = sorted(
        p for p in period_dir.iterdir() if p.is_dir() and p.name.startswith("shard-")
    )
    if len(shard_dirs) != len(sizes):
        problems.append(
            Problem(name, "shards", f"{len(shard_dirs)} directories, meta says {len(sizes)}")
        )
    if not (period_dir / "flows.parquet").is_file():
        problems.append(Problem(name, "flows.parquet", "missing"))
    for shard, rows in zip(shard_dirs, sizes, strict=False):
        for array in ARRAY_SPEC:
            path = shard / f"{array}.npy"
            if not path.is_file():
                problems.append(Problem(name, f"{shard.name}/{array}.npy", "missing"))
                continue
            extra = path.stat().st_size - rows * _row_bytes(array)
            if extra not in HEADER_SIZES:
                problems.append(
                    Problem(
                        name,
                        f"{shard.name}/{array}.npy",
                        f"size {path.stat().st_size} does not match {rows} rows "
                        f"(expected {rows * _row_bytes(array)} + header)",
                    )
                )
    return n_flows, problems


def periods_of(dataset_dir: Path) -> Iterator[Path]:
    return iter(sorted(p for p in dataset_dir.iterdir() if p.is_dir()))


def check_dataset(dataset_dir: Path) -> tuple[int, list[Problem]]:
    """Check every period under ``dataset_dir``: ``(flows in healthy periods, problems)``."""
    healthy_flows, problems = 0, []
    for period in periods_of(dataset_dir):
        n, bad = check_period(period)
        problems.extend(bad)
        if not bad:
            healthy_flows += n
    return healthy_flows, problems
