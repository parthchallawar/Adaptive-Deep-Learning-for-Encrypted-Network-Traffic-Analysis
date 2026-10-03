"""Persisting one evaluation pass's `DenseLogits` to disk and back (spec 004, plan T7).

``results/<run_name>/eval/<split>/`` holds everything a report needs to be
re-scored later with no model in the loop (spec 004's "cheap threshold
sweeps" rule):

    logits.npy         [N, K_MAX, C] float16 -- DenseLogits.logits
    dense_logits.json  {"indexing": ..., "evaluated_k": [...]}
    labels.npy         [N] int64, model-space label ids (unknown = -1)
    ppi_len.npy        [N] int64
    flow_index.npy     [N] int64, each row's index into the *un-capped*
                        split, so a report can be joined back to
                        flows.parquet

fp16 is deliberate: at C=150 classes it is 9 KB/flow dense, so the 250,000-
flow cap below is about 2.25 GB per run per seed (plan phase 2's own sizing);
softmax and the argmax comparisons every metric here does are unaffected by
fp16's precision loss (spec 004 never claims logit-level fidelity, only which
class wins).

:func:`subsample_for_eval` is that cap (spec 004/plan phase 2, 250,000
flows): a uniform seeded draw, never per-class, so a report's own class
balance is the split's real one, not a resampled one -- the same "sample
generously once, never stratify" rule the D1 exporter uses for the same
reason.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adl_etc.evaluation.dense_logits import DenseLogits
from adl_etc.utils.atomic import atomic_write_bytes, atomic_write_json

DEFAULT_EVAL_CAP = 250_000

LOGITS_FILE = "logits.npy"
DENSE_LOGITS_FILE = "dense_logits.json"
LABELS_FILE = "labels.npy"
PPI_LEN_FILE = "ppi_len.npy"
FLOW_INDEX_FILE = "flow_index.npy"


def subsample_for_eval(n: int, cap: int = DEFAULT_EVAL_CAP, seed: int = 0) -> np.ndarray:
    """Sorted indices of a uniform seeded sample of ``min(n, cap)`` flows out
    of ``n``. Ascending order, not draw order, so two calls with the same
    ``n``/``cap``/``seed`` are trivially comparable and a downstream
    ``flow_index`` column stays in the split's own original order."""
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=cap, replace=False))


def _save_npy(path: Path, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, arr)
    atomic_write_bytes(path, buf.getvalue())


def _load_npy(path: Path) -> np.ndarray:
    with path.open("rb") as fh:
        return np.load(fh)


@dataclass(frozen=True)
class EvalArrays:
    """Everything one evaluation pass needs to be scored, round-tripped
    through disk exactly (fp16 logits aside -- see the module docstring)."""

    dense: DenseLogits
    labels: np.ndarray  # [N] int64, model-space ids (unknown = -1)
    ppi_len: np.ndarray  # [N] int64
    flow_index: np.ndarray  # [N] int64, index into the pre-cap split

    def __post_init__(self) -> None:
        n = len(self.dense)
        for name, arr in (
            ("labels", self.labels),
            ("ppi_len", self.ppi_len),
            ("flow_index", self.flow_index),
        ):
            if arr.shape != (n,):
                raise ValueError(f"{name} must be [{n}], got {arr.shape}")

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_npy(out_dir / LOGITS_FILE, self.dense.logits.astype(np.float16))
        atomic_write_json(
            out_dir / DENSE_LOGITS_FILE,
            {"indexing": self.dense.indexing, "evaluated_k": list(self.dense.evaluated_k)},
        )
        _save_npy(out_dir / LABELS_FILE, self.labels.astype(np.int64))
        _save_npy(out_dir / PPI_LEN_FILE, self.ppi_len.astype(np.int64))
        _save_npy(out_dir / FLOW_INDEX_FILE, self.flow_index.astype(np.int64))

    @classmethod
    def load(cls, out_dir: str | Path) -> EvalArrays:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / DENSE_LOGITS_FILE).read_text(encoding="utf-8"))
        logits = _load_npy(out_dir / LOGITS_FILE).astype(np.float32)
        dense = DenseLogits(logits, tuple(meta["evaluated_k"]), meta["indexing"])
        return cls(
            dense=dense,
            labels=_load_npy(out_dir / LABELS_FILE),
            ppi_len=_load_npy(out_dir / PPI_LEN_FILE),
            flow_index=_load_npy(out_dir / FLOW_INDEX_FILE),
        )
