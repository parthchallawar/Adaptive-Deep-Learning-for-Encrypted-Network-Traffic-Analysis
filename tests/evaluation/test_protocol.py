"""Split loading and leakage assertions (spec 004, plan T6)."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from adl_etc.data import ppi as P
from adl_etc.data.features import Standardizer
from adl_etc.data.tensors import ShardWriter
from adl_etc.evaluation import protocol as PR


def _write_period(
    root,
    dataset: str,
    period: str,
    *,
    n: int,
    labels: list[int] | None = None,
    session_ids: list[int] | None = None,
    ts: list[int] | None = None,
) -> None:
    """A minimal but real shard set: one payload-carrying packet per flow (so
    none are dropped for zero PPI), with the given label/session_id/ts
    columns -- everything protocol.py's assertions actually read."""
    labels = labels if labels is not None else [0] * n
    session_ids = session_ids if session_ids is not None else [0] * n
    ts = ts if ts is not None else list(range(n))
    label_map = {str(v): v for v in sorted({v for v in labels if v != -1})} or {"0": 0}

    with ShardWriter(root, dataset, period, label_map=label_map, allow_unknown=True) as w:
        for i in range(n):
            ppi = P.empty_ppi()
            ppi[0] = (0, P.DIR_FWD, 100, 1)
            w.add_batch(
                {
                    "ppi": ppi[None, ...],
                    "ppi_len": np.array([1], dtype=np.int8),
                    "flowstats": np.zeros((1, P.FLOWSTATS_DIM), dtype=np.float32),
                    "label": np.array([labels[i]], dtype=np.int16),
                    "category": np.array([0], dtype=np.int8),
                    "session_id": np.array([session_ids[i]], dtype=np.int32),
                    "ts": np.array([ts[i]], dtype=np.int64),
                }
            )


def _write_yaml(path, spec: dict) -> None:
    OmegaConf.save(OmegaConf.create(spec), path)


# --- rule 1: no period / (period, session_id) in two splits -------------------


def test_no_session_overlap_passes_on_a_clean_spec():
    spec = {
        "dataset": "d",
        "splits": {
            "train": {"periods": ["p1"], "session_ids": [0, 1]},
            "test": {"periods": ["p1"], "session_ids": [2, 3]},
        },
    }
    PR.assert_no_session_overlap(spec)  # does not raise


def test_no_session_overlap_raises_on_duplicate_period():
    spec = {
        "dataset": "d",
        "splits": {"train": {"periods": ["p1"]}, "test": {"periods": ["p1"]}},
    }
    with pytest.raises(PR.LeakageError, match="p1"):
        PR.assert_no_session_overlap(spec)


def test_no_session_overlap_raises_on_duplicate_session_id_within_one_period():
    spec = {
        "dataset": "d",
        "splits": {
            "train": {"periods": ["p1"], "session_ids": [0, 1, 2]},
            "test": {"periods": ["p1"], "session_ids": [2, 3]},
        },
    }
    with pytest.raises(PR.LeakageError, match="session_id 2"):
        PR.assert_no_session_overlap(spec)


def test_no_session_overlap_allows_the_same_session_id_number_in_different_periods():
    # session_id numbering is only ever locally scoped to one period's own
    # export run (D1/D2's per-week CESNET exports, plan T5) -- the same
    # number 0 in two different periods must never be flagged.
    spec = {
        "dataset": "d",
        "splits": {
            "train": {"periods": ["WEEK-11"], "session_ids": [0, 1]},
            "test": {"periods": ["WEEK-27"], "session_ids": [0, 1]},
        },
    }
    PR.assert_no_session_overlap(spec)  # does not raise


# --- rule 3: consistent standardizer hash --------------------------------------


def _fake_standardizer() -> Standardizer:
    return Standardizer(
        cont_mean=np.zeros(4),
        cont_std=np.ones(4),
        flowstats_mean=np.zeros(P.FLOWSTATS_DIM),
        flowstats_std=np.ones(P.FLOWSTATS_DIM),
    )


def test_standardizer_hash_consistent_passes_when_same_file_is_referenced(tmp_path):
    std = _fake_standardizer()
    std.hash = std._compute_hash()  # noqa: SLF001
    path = tmp_path / "std.json"
    std.save(path)

    spec = {
        "dataset": "d",
        "splits": {
            "train": {"periods": ["p1"], "standardizer": str(path)},
            "val": {"periods": ["p2"], "standardizer": str(path)},
        },
    }
    PR.assert_standardizer_hash_consistent(spec)  # does not raise


def test_standardizer_hash_consistent_raises_when_val_was_refit(tmp_path):
    std_a = _fake_standardizer()
    std_a.hash = std_a._compute_hash()  # noqa: SLF001
    path_a = tmp_path / "train.json"
    std_a.save(path_a)

    std_b = _fake_standardizer()
    std_b.cont_mean = std_b.cont_mean + 1.0  # different stats -> different hash
    std_b.hash = std_b._compute_hash()  # noqa: SLF001
    path_b = tmp_path / "val.json"
    std_b.save(path_b)

    spec = {
        "dataset": "d",
        "splits": {
            "train": {"periods": ["p1"], "standardizer": str(path_a)},
            "val": {"periods": ["p2"], "standardizer": str(path_b)},
        },
    }
    with pytest.raises(PR.LeakageError, match="different hashes"):
        PR.assert_standardizer_hash_consistent(spec)


# --- rule 2: no unknown label in train ------------------------------------------


def test_no_unknown_in_train_raises_on_a_real_shard_set(tmp_path):
    _write_period(tmp_path, "d", "p1", n=3, labels=[0, -1, 0])
    _write_period(tmp_path, "d", "p2", n=2, labels=[0, 0])
    spec = {"dataset": "d", "splits": {"train": {"periods": ["p1"]}, "val": {"periods": ["p2"]}}}
    _write_yaml(tmp_path / "split.yaml", spec)

    with pytest.raises(PR.LeakageError, match="unknown"):
        PR.load_split(tmp_path / "split.yaml", root=tmp_path)


def test_unknown_label_is_fine_outside_train(tmp_path):
    _write_period(tmp_path, "d", "p1", n=2, labels=[0, 0])
    _write_period(tmp_path, "d", "p2", n=2, labels=[0, -1])  # unknown in val: allowed
    spec = {"dataset": "d", "splits": {"train": {"periods": ["p1"]}, "val": {"periods": ["p2"]}}}
    _write_yaml(tmp_path / "split.yaml", spec)

    loaded = PR.load_split(tmp_path / "split.yaml", root=tmp_path)
    loaded.close()


# --- rule 4: train ts strictly precedes every other split -----------------------


def test_temporal_order_raises_when_val_overlaps_train(tmp_path):
    _write_period(tmp_path, "d", "p1", n=2, ts=[100, 200])
    _write_period(tmp_path, "d", "p2", n=2, ts=[150, 300])  # 150 <= max(train ts)=200
    spec = {
        "dataset": "d",
        "temporal": True,
        "splits": {"train": {"periods": ["p1"]}, "val": {"periods": ["p2"]}},
    }
    _write_yaml(tmp_path / "split.yaml", spec)

    with pytest.raises(PR.LeakageError, match="ts"):
        PR.load_split(tmp_path / "split.yaml", root=tmp_path)


def test_temporal_order_passes_when_val_is_strictly_after_train(tmp_path):
    _write_period(tmp_path, "d", "p1", n=2, ts=[100, 200])
    _write_period(tmp_path, "d", "p2", n=2, ts=[201, 300])
    spec = {
        "dataset": "d",
        "temporal": True,
        "splits": {"train": {"periods": ["p1"]}, "val": {"periods": ["p2"]}},
    }
    _write_yaml(tmp_path / "split.yaml", spec)

    loaded = PR.load_split(tmp_path / "split.yaml", root=tmp_path)
    loaded.close()


def test_temporal_order_not_checked_when_not_temporal(tmp_path):
    # D3-style grouped split: overlapping ts is fine when temporal is false.
    _write_period(tmp_path, "d", "p1", n=2, ts=[100, 200])
    spec = {
        "dataset": "d",
        "temporal": False,
        "splits": {
            "train": {"periods": ["p1"], "session_ids": [0]},
            "test": {"periods": ["p1"], "session_ids": [1]},
        },
    }
    _write_yaml(tmp_path / "split.yaml", spec)

    loaded = PR.load_split(tmp_path / "split.yaml", root=tmp_path)
    loaded.close()


# --- session_ids filtering (D3's grouped-CV mechanism) --------------------------


def test_session_ids_filter_restricts_one_shared_period(tmp_path):
    _write_period(
        tmp_path,
        "d",
        "all",
        n=4,
        labels=[0, 0, 0, 0],
        session_ids=[0, 1, 2, 3],
        ts=[10, 20, 30, 40],
    )
    spec = {
        "dataset": "d",
        "temporal": False,
        "splits": {
            "train": {"periods": ["all"], "session_ids": [0, 1]},
            "test": {"periods": ["all"], "session_ids": [2, 3]},
        },
    }
    _write_yaml(tmp_path / "split.yaml", spec)

    loaded = PR.load_split(tmp_path / "split.yaml", root=tmp_path)
    train_ss = loaded.shard_sets["train"][0]
    train_mask = loaded.masks["train"][0]
    assert train_mask is not None
    np.testing.assert_array_equal(train_ss.column("session_id")[train_mask], [0, 1])
    test_ss = loaded.shard_sets["test"][0]
    test_mask = loaded.masks["test"][0]
    np.testing.assert_array_equal(test_ss.column("session_id")[test_mask], [2, 3])
    loaded.close()


# --- real end-to-end: D4's real shard data on disk -----------------------------


def test_load_split_d4_anomaly_against_real_shards():
    loaded = PR.load_split("configs/splits/d4_anomaly.yaml")
    try:
        assert loaded.dataset == "ustc-tfc2016"
        (ss,) = loaded.shard_sets["test"]
        assert len(ss) == 403_394
        category = ss.column("category")
        counts = dict(zip(*np.unique(category, return_counts=True), strict=True))
        assert counts[np.int8(0)] == 282_412  # benign
        assert counts[np.int8(1)] == 120_982  # malware
    finally:
        loaded.close()


# --- failure paths leave nothing open -------------------------------------------


def test_load_split_closes_everything_on_a_later_assertion_failure(tmp_path):
    _write_period(tmp_path, "d", "p1", n=2, ts=[100, 200])
    _write_period(tmp_path, "d", "p2", n=2, ts=[150, 300])  # triggers rule 4
    spec = {
        "dataset": "d",
        "temporal": True,
        "splits": {"train": {"periods": ["p1"]}, "val": {"periods": ["p2"]}},
    }
    _write_yaml(tmp_path / "split.yaml", spec)

    with pytest.raises(PR.LeakageError):
        PR.load_split(tmp_path / "split.yaml", root=tmp_path)
    # No lingering memmap handles should block cleanup on Windows.
    import shutil

    shutil.rmtree(tmp_path / "d")
