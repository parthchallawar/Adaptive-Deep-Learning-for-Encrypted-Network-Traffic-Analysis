"""Shard writer and reader (spec 003).

The on-disk format every later module loads: a directory of ``.npy`` arrays per
shard plus a ``meta.json`` describing them, and a ``flows.parquet`` audit
sidecar that is never a model input. A directory of ``.npy`` files rather than
a single ``.npz`` is deliberate: ``.npz`` cannot be memory-mapped, and the
whole point of mmap here is that a 4+ GB shard set does not have to fit twice
in Kaggle's ~29 GB RAM (once as the file, once as the loaded array).

Every array in :data:`ARRAY_SPEC` is written by every exporter (spec 001's
PCAP path and its CESNET CSV path alike), so a model never needs to know which
backend produced the flow it is looking at.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from adl_etc.data import ppi as P
from adl_etc.data.flows import FlowRecord
from adl_etc.utils.provenance import git_commit, now_iso

# --- the array contract -----------------------------------------------------

ARRAY_SPEC: dict[str, tuple[np.dtype, tuple[int, ...]]] = {
    "ppi": (np.dtype(np.int16), (P.K_MAX, P.PPI_CHANNELS)),
    "ppi_len": (np.dtype(np.int8), ()),
    "flowstats": (np.dtype(np.float32), (P.FLOWSTATS_DIM,)),
    "label": (np.dtype(np.int16), ()),
    "category": (np.dtype(np.int8), ()),
    "session_id": (np.dtype(np.int32), ()),
    "ts": (np.dtype(np.int64), ()),
}
"""Name -> (dtype, per-flow shape). The single source of truth for the shard
schema; ``ShardWriter`` and ``ShardSet`` both derive their behaviour from it so
that adding a column means editing one dict, not several call sites."""

SCHEMA_VERSION = 1

_PARQUET_COLUMNS = ("key_hash", "start_ts", "end_reason", "session_id", "shard_index", "row_index")


def _full_shape(name: str, n: int) -> tuple[int, ...]:
    _dtype, shape = ARRAY_SPEC[name]
    return (n, *shape)


def _hash_key(key: object, salt: bytes) -> str:
    """Salted SHA-1 of a flow's 5-tuple key, for de-duplication without storing
    an address. The salt is per-export and discarded with the export; it is not
    a security control, only a way to join rows within one export run."""
    import hashlib

    h = hashlib.sha1(salt)
    h.update(repr(key).encode("utf-8"))
    return h.hexdigest()


class ShardWriter:
    """Writes flows into fixed-size shards of ``.npy`` arrays.

    Buffers are preallocated at ``max_flows`` rows so memory is bounded
    regardless of dataset size; ``close()`` truncates the final partial shard.
    """

    def __init__(
        self,
        root: str | Path,
        dataset: str,
        period: str,
        *,
        label_map: dict[str, int],
        category_map: dict[str, int] | None = None,
        max_flows: int = 500_000,
        allow_unknown: bool = False,
        source_manifest_hash: str | None = None,
        flowstats_source: str = "adl_etc",
        overwrite: bool = False,
    ) -> None:
        self._period_dir = Path(root) / dataset / period
        self._dataset = dataset
        self._period = period
        self._label_map = dict(label_map)
        self._category_map = dict(category_map) if category_map else None
        self._max_flows = max_flows
        self._allow_unknown = allow_unknown
        self._source_manifest_hash = source_manifest_hash
        self._flowstats_source = flowstats_source

        if self._period_dir.exists() and any(self._period_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"{self._period_dir} is not empty; pass overwrite=True to replace it"
                )
            self._clear(self._period_dir)
        self._period_dir.mkdir(parents=True, exist_ok=True)

        self._buffers: dict[str, np.ndarray] = {
            name: np.zeros(_full_shape(name, max_flows), dtype=dtype)
            for name, (dtype, _shape) in ARRAY_SPEC.items()
        }
        self._n = 0
        self._shard_index = 0
        self._shard_sizes: list[int] = []
        self._total = 0
        self._salt = os.urandom(16).hex()
        self._counters: dict[str, int] = {}
        self._parquet_rows: dict[str, list[Any]] = {c: [] for c in _PARQUET_COLUMNS}
        self._closed = False

    @staticmethod
    def _clear(period_dir: Path) -> None:
        for child in period_dir.iterdir():
            if child.is_dir():
                for f in child.iterdir():
                    f.unlink()
                child.rmdir()
            else:
                child.unlink()

    def _count(self, name: str, n: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + n

    # -- writing --------------------------------------------------------

    def add(
        self,
        rec: FlowRecord,
        *,
        label: int,
        category: int = -1,
        session_id: int | None = None,
    ) -> None:
        """Append one flow. Dropped (not raised) if it has no PPI: a flow with
        no payload-carrying packets cannot be classified by definition."""
        if self._closed:
            raise RuntimeError("add() called on a closed ShardWriter")
        if rec.ppi_len == 0:
            self._count("dropped_zero_ppi")
            return
        if label == -1 and not self._allow_unknown:
            raise ValueError(
                "unknown label (-1) passed to add() with allow_unknown=False; "
                "unknown classes must never enter a train shard set (spec 004)"
            )

        idx = self._n
        self._buffers["ppi"][idx] = rec.ppi
        self._buffers["ppi_len"][idx] = rec.ppi_len
        self._buffers["flowstats"][idx] = rec.flowstats()
        self._buffers["label"][idx] = label
        self._buffers["category"][idx] = category
        sid = rec.session_id if session_id is None else session_id
        self._buffers["session_id"][idx] = sid
        ts_ms = round(rec.start_ts * 1000)
        self._buffers["ts"][idx] = ts_ms

        self._parquet_rows["key_hash"].append(_hash_key(rec.key, bytes.fromhex(self._salt)))
        self._parquet_rows["start_ts"].append(ts_ms)
        self._parquet_rows["end_reason"].append(rec.end_reason)
        self._parquet_rows["session_id"].append(sid)
        self._parquet_rows["shard_index"].append(self._shard_index)
        self._parquet_rows["row_index"].append(idx)

        self._n += 1
        self._total += 1
        if self._n >= self._max_flows:
            self._flush_shard()

    def add_batch(self, arrays: dict[str, np.ndarray]) -> None:
        """Vectorised path for exporters that already parsed many flows at
        once (the CESNET CSV exporter, spec 001 Path B). ``arrays`` must carry
        every key in :data:`ARRAY_SPEC`, all the same length.

        There is no per-row 5-tuple in this path (the source columns are
        already stripped of identifiers), so the parquet audit sidecar records
        ``key_hash=None`` and ``end_reason=None`` for these rows.
        """
        if self._closed:
            raise RuntimeError("add_batch() called on a closed ShardWriter")
        missing = [k for k in ARRAY_SPEC if k not in arrays]
        if missing:
            raise KeyError(f"add_batch missing required arrays: {missing}")
        n = len(arrays["ppi_len"])
        for name in ARRAY_SPEC:
            if len(arrays[name]) != n:
                raise ValueError("all arrays passed to add_batch must have the same length")

        zero_ppi = arrays["ppi_len"] == 0
        n_dropped = int(zero_ppi.sum())
        if n_dropped:
            self._count("dropped_zero_ppi", n_dropped)
            keep = ~zero_ppi
            arrays = {name: arrays[name][keep] for name in ARRAY_SPEC}
            n -= n_dropped
        if n == 0:
            return

        if not self._allow_unknown and bool(np.any(arrays["label"] == -1)):
            raise ValueError(
                "unknown label (-1) in add_batch and allow_unknown=False; "
                "unknown classes must never enter a train shard set (spec 004)"
            )

        offset = 0
        while offset < n:
            space = self._max_flows - self._n
            take = min(space, n - offset)
            dst = slice(self._n, self._n + take)
            src = slice(offset, offset + take)
            for name in ARRAY_SPEC:
                self._buffers[name][dst] = arrays[name][src]

            row_indices = range(self._n, self._n + take)
            self._parquet_rows["key_hash"].extend([None] * take)
            self._parquet_rows["start_ts"].extend(int(v) for v in arrays["ts"][src])
            self._parquet_rows["end_reason"].extend([None] * take)
            self._parquet_rows["session_id"].extend(int(v) for v in arrays["session_id"][src])
            self._parquet_rows["shard_index"].extend([self._shard_index] * take)
            self._parquet_rows["row_index"].extend(row_indices)

            self._n += take
            self._total += take
            offset += take
            if self._n >= self._max_flows:
                self._flush_shard()

    def _flush_shard(self) -> None:
        if self._n == 0:
            return
        shard_dir = self._period_dir / f"shard-{self._shard_index:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for name in ARRAY_SPEC:
            np.save(shard_dir / f"{name}.npy", self._buffers[name][: self._n])
        self._shard_sizes.append(self._n)
        self._shard_index += 1
        self._n = 0

    def close(self) -> dict:
        """Flush the last partial shard and write ``meta.json`` and
        ``flows.parquet``. Safe to call more than once."""
        if self._closed:
            return self._read_meta()
        self._flush_shard()
        meta = {
            "schema_version": SCHEMA_VERSION,
            "dataset": self._dataset,
            "period": self._period,
            "n_flows": self._total,
            "n_shards": self._shard_index,
            "shard_sizes": self._shard_sizes,
            "label_map": self._label_map,
            "category_map": self._category_map,
            "ppi_columns": list(P.PPI_COLUMNS),
            "flowstats_columns": list(P.FLOWSTATS_COLUMNS),
            "flowstats_source": self._flowstats_source,
            "k_max": P.K_MAX,
            "exporter_git_commit": git_commit(),
            "source_manifest_hash": self._source_manifest_hash,
            "key_salt": self._salt,
            "created_at": now_iso(),
            "counters": self._counters,
        }
        (self._period_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._write_parquet()
        self._closed = True
        return meta

    def _read_meta(self) -> dict:
        return json.loads((self._period_dir / "meta.json").read_text(encoding="utf-8"))

    def _write_parquet(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "key_hash": pa.array(self._parquet_rows["key_hash"], type=pa.string()),
                "start_ts": pa.array(self._parquet_rows["start_ts"], type=pa.int64()),
                "end_reason": pa.array(self._parquet_rows["end_reason"], type=pa.string()),
                "session_id": pa.array(self._parquet_rows["session_id"], type=pa.int32()),
                "shard_index": pa.array(self._parquet_rows["shard_index"], type=pa.int32()),
                "row_index": pa.array(self._parquet_rows["row_index"], type=pa.int32()),
            }
        )
        pq.write_table(table, self._period_dir / "flows.parquet")

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self._closed:
            self.close()


class ShardSet:
    """Read access to a shard set written by :class:`ShardWriter`.

    Backed by ``np.memmap`` by default, so opening a multi-GB shard set costs
    nothing until a column is actually read.
    """

    def __init__(self, period_dir: Path, meta: dict, mmap: bool) -> None:
        self.period_dir = period_dir
        self.meta = meta
        self._mmap = mmap
        self._shard_dirs = [period_dir / f"shard-{i:05d}" for i in range(meta["n_shards"])]
        sizes = meta["shard_sizes"]
        self._offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self._array_cache: dict[tuple[int, str], np.ndarray] = {}
        self._column_cache: dict[str, np.ndarray] = {}

    @classmethod
    def open(cls, root: str | Path, *, mmap: bool = True) -> ShardSet:
        """``root`` is a period directory (``data/processed/<dataset>/<period>``),
        i.e. exactly what a :class:`ShardWriter` was pointed at."""
        period_dir = Path(root)
        meta = json.loads((period_dir / "meta.json").read_text(encoding="utf-8"))
        return cls(period_dir, meta, mmap)

    def __len__(self) -> int:
        return int(self.meta["n_flows"])

    def _shard_array(self, shard_idx: int, name: str) -> np.ndarray:
        key = (shard_idx, name)
        if key not in self._array_cache:
            path = self._shard_dirs[shard_idx] / f"{name}.npy"
            arr = np.load(path, mmap_mode="r") if self._mmap else np.load(path)
            self._array_cache[key] = arr
        return self._array_cache[key]

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        n = len(self)
        if i < 0:
            i += n
        if not (0 <= i < n):
            raise IndexError(i)
        shard_idx = int(np.searchsorted(self._offsets, i, side="right") - 1)
        row = i - int(self._offsets[shard_idx])
        return {name: self._shard_array(shard_idx, name)[row] for name in ARRAY_SPEC}

    def column(self, name: str) -> np.ndarray:
        """The named column concatenated across every shard. Materialises a
        copy (``np.concatenate`` always copies); cheap for anything that fits
        in RAM, which on Kaggle is the whole shard set at once."""
        if name not in ARRAY_SPEC:
            raise KeyError(name)
        if name not in self._column_cache:
            parts = [self._shard_array(i, name) for i in range(len(self._shard_dirs))]
            if parts:
                self._column_cache[name] = np.concatenate(parts, axis=0)
            else:
                dtype, shape = ARRAY_SPEC[name]
                self._column_cache[name] = np.empty((0, *shape), dtype=dtype)
        return self._column_cache[name]

    def batches(
        self,
        size: int,
        *,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        n = len(self)
        order = np.arange(n)
        if shuffle:
            rng = rng if rng is not None else np.random.default_rng()
            rng.shuffle(order)
        columns = {name: self.column(name) for name in ARRAY_SPEC}
        for start in range(0, n, size):
            idx = order[start : start + size]
            yield {name: arr[idx] for name, arr in columns.items()}

    def close(self) -> None:
        """Release open memmap file handles.

        Not required before process exit, but on Windows an open memmap keeps
        its file locked, which blocks deleting or ``overwrite=True``-ing that
        directory from the same process. Anything that reopens or cleans up a
        period directory it has already read should call this first.
        """
        for arr in self._array_cache.values():
            mm = getattr(arr, "_mmap", None)
            if mm is not None:
                mm.close()
        self._array_cache.clear()
        self._column_cache.clear()

    def __enter__(self) -> ShardSet:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
