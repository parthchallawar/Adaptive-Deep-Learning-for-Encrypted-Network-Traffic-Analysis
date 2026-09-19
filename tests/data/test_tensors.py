"""Shard writer and reader (spec 003, plan T1).

``EXPECTED_PPI``/``build_flows`` come from ``test_flows.py``: the export path
must not alter what the flow builder already produced.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from adl_etc.data import ppi as P
from adl_etc.data.flows import PROTO_TCP, EndReason, FlowRecord
from adl_etc.data.tensors import ARRAY_SPEC, ShardSet, ShardWriter
from tests.data.test_flows import EXPECTED_PPI, build_flows


def make_flow(rng: np.random.Generator, session_id: int = 0, ts: float = 0.0) -> FlowRecord:
    """A synthetic, internally-consistent flow record for shard round-trip tests."""
    ppi_len = int(rng.integers(1, P.K_MAX + 1))
    ppi = P.empty_ppi()
    dirs = rng.choice([P.DIR_FWD, P.DIR_REV], size=ppi_len)
    ppi[:ppi_len, P.DIR_POS] = dirs
    ppi[:ppi_len, P.SIZE_POS] = rng.integers(1, P.SIZE_MAX + 1, size=ppi_len)
    ppi[:ppi_len, P.IPT_POS] = np.concatenate([[0], rng.integers(0, 500, size=ppi_len - 1)])
    ppi[:ppi_len, P.PUSH_POS] = rng.integers(0, 2, size=ppi_len)
    packets = int(np.sum(dirs == P.DIR_FWD)) + 2
    packets_rev = int(np.sum(dirs == P.DIR_REV)) + 2
    return FlowRecord(
        key=((b"\x0a\x00\x00\x05", 51_234), (b"\x5d\xb8\xd8\x22", 443), PROTO_TCP),
        proto=PROTO_TCP,
        start_ts=ts,
        end_ts=ts + 1.0,
        ppi=ppi,
        ppi_len=ppi_len,
        packets=packets,
        packets_rev=packets_rev,
        bytes=int(ppi[:ppi_len, P.SIZE_POS][dirs == P.DIR_FWD].sum()),
        bytes_rev=int(ppi[:ppi_len, P.SIZE_POS][dirs == P.DIR_REV].sum()),
        flags_seen=0x10,
        end_reason=EndReason.FIN,
        session_id=session_id,
    )


def zero_ppi_flow() -> FlowRecord:
    return FlowRecord(
        key=((b"\x0a\x00\x00\x05", 1), (b"\x0a\x00\x00\x06", 2), PROTO_TCP),
        proto=PROTO_TCP,
        start_ts=0.0,
        end_ts=0.0,
        ppi=P.empty_ppi(),
        ppi_len=0,
        packets=0,
        packets_rev=0,
        bytes=0,
        bytes_rev=0,
        flags_seen=0,
        end_reason=EndReason.IDLE,
    )


LABEL_MAP = {"a": 0, "b": 1}


def test_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    flows = [make_flow(rng, ts=float(i)) for i in range(1_000)]

    with ShardWriter(tmp_path, "synthetic", "all", label_map=LABEL_MAP, max_flows=10_000) as w:
        for i, flow in enumerate(flows):
            w.add(flow, label=i % 2, category=0)

    ss = ShardSet.open(tmp_path / "synthetic" / "all")
    assert len(ss) == 1_000
    for name, (dtype, _shape) in ARRAY_SPEC.items():
        col = ss.column(name)
        assert col.dtype == dtype
        assert col.shape[0] == 1_000

    np.testing.assert_array_equal(ss.column("ppi")[7], flows[7].ppi)
    assert int(ss.column("ppi_len")[7]) == flows[7].ppi_len
    np.testing.assert_allclose(ss.column("flowstats")[7], flows[7].flowstats())
    np.testing.assert_array_equal(ss.column("label"), np.arange(1_000) % 2)


def test_sharding_boundaries(tmp_path):
    rng = np.random.default_rng(1)
    flows = [make_flow(rng, ts=float(i)) for i in range(250)]

    with ShardWriter(tmp_path, "synthetic", "shards", label_map=LABEL_MAP, max_flows=100) as w:
        for flow in flows:
            w.add(flow, label=0)

    meta = json.loads((tmp_path / "synthetic" / "shards" / "meta.json").read_text())
    assert meta["shard_sizes"] == [100, 100, 50]
    assert meta["n_shards"] == 3

    ss = ShardSet.open(tmp_path / "synthetic" / "shards")
    ts = ss.column("ts")
    np.testing.assert_array_equal(ts, np.arange(250) * 1000)  # insertion order preserved


def test_mmap_backed(tmp_path):
    rng = np.random.default_rng(2)
    with ShardWriter(tmp_path, "synthetic", "mmap", label_map=LABEL_MAP) as w:
        for i in range(10):
            w.add(make_flow(rng, ts=float(i)), label=0)

    ss = ShardSet.open(tmp_path / "synthetic" / "mmap", mmap=True)
    raw = ss._shard_array(0, "ppi")
    assert isinstance(raw, np.memmap)

    ss_eager = ShardSet.open(tmp_path / "synthetic" / "mmap", mmap=False)
    raw_eager = ss_eager._shard_array(0, "ppi")
    assert not isinstance(raw_eager, np.memmap)


def test_unknown_label_guard(tmp_path):
    rng = np.random.default_rng(3)
    flow = make_flow(rng, ts=0.0)

    with (
        ShardWriter(tmp_path, "synthetic", "guard_reject", label_map=LABEL_MAP) as w,
        pytest.raises(ValueError),
    ):
        w.add(flow, label=-1)

    with ShardWriter(
        tmp_path, "synthetic", "guard_allow", label_map=LABEL_MAP, allow_unknown=True
    ) as w:
        w.add(flow, label=-1)
    ss = ShardSet.open(tmp_path / "synthetic" / "guard_allow")
    assert len(ss) == 1
    assert int(ss.column("label")[0]) == -1


def test_zero_ppi_dropped(tmp_path):
    with ShardWriter(tmp_path, "synthetic", "dropped", label_map=LABEL_MAP) as w:
        w.add(zero_ppi_flow(), label=0)
        meta = w.close()

    assert meta["n_flows"] == 0
    assert meta["counters"]["dropped_zero_ppi"] == 1
    ss = ShardSet.open(tmp_path / "synthetic" / "dropped")
    assert len(ss) == 0


def test_meta_records_provenance(tmp_path):
    with ShardWriter(
        tmp_path,
        "synthetic",
        "meta",
        label_map=LABEL_MAP,
        source_manifest_hash="deadbeef",
    ) as w:
        w.add(make_flow(np.random.default_rng(4), ts=0.0), label=0)
        meta = w.close()

    assert meta["exporter_git_commit"] != ""
    assert meta["source_manifest_hash"] == "deadbeef"
    assert meta["flowstats_source"] == "adl_etc"
    assert len(meta["key_salt"]) == 32  # 16 bytes hex
    assert meta["k_max"] == P.K_MAX
    assert meta["ppi_columns"] == list(P.PPI_COLUMNS)


def test_two_writes_same_data_agree(tmp_path):
    def write_once(name):
        rng = np.random.default_rng(5)
        flows = [make_flow(rng, ts=float(i)) for i in range(50)]
        with ShardWriter(tmp_path, "synthetic", name, label_map=LABEL_MAP) as w:
            for flow in flows:
                w.add(flow, label=0)
            return w.close()

    meta1 = write_once("rep1")
    meta2 = write_once("rep2")
    assert meta1["n_flows"] == meta2["n_flows"]
    assert meta1["label_map"] == meta2["label_map"]


def test_overwrite_guard(tmp_path):
    with ShardWriter(tmp_path, "synthetic", "guarded", label_map=LABEL_MAP) as w:
        w.add(make_flow(np.random.default_rng(6), ts=0.0), label=0)

    with pytest.raises(FileExistsError):
        ShardWriter(tmp_path, "synthetic", "guarded", label_map=LABEL_MAP)

    with ShardWriter(tmp_path, "synthetic", "guarded", label_map=LABEL_MAP, overwrite=True) as w:
        w.add(make_flow(np.random.default_rng(7), ts=0.0), label=1)
    ss = ShardSet.open(tmp_path / "synthetic" / "guarded")
    assert len(ss) == 1
    assert int(ss.column("label")[0]) == 1


def test_add_batch_matches_add(tmp_path):
    rng = np.random.default_rng(8)
    flows = [make_flow(rng, ts=float(i)) for i in range(40)]

    with ShardWriter(tmp_path, "synthetic", "via_add", label_map=LABEL_MAP) as w:
        for flow in flows:
            w.add(flow, label=0)

    arrays = {
        "ppi": np.stack([f.ppi for f in flows]),
        "ppi_len": np.array([f.ppi_len for f in flows], dtype=np.int8),
        "flowstats": np.stack([f.flowstats() for f in flows]),
        "label": np.zeros(len(flows), dtype=np.int16),
        "category": np.full(len(flows), -1, dtype=np.int8),
        "session_id": np.array([f.session_id for f in flows], dtype=np.int32),
        "ts": np.array([round(f.start_ts * 1000) for f in flows], dtype=np.int64),
    }
    with ShardWriter(tmp_path, "synthetic", "via_batch", label_map=LABEL_MAP) as w:
        w.add_batch(arrays)

    a = ShardSet.open(tmp_path / "synthetic" / "via_add")
    b = ShardSet.open(tmp_path / "synthetic" / "via_batch")
    for name in ARRAY_SPEC:
        np.testing.assert_array_equal(a.column(name), b.column(name))


def test_add_batch_spans_shard_boundary(tmp_path):
    n = 25
    arrays = {
        "ppi": np.zeros((n, P.K_MAX, P.PPI_CHANNELS), dtype=np.int16),
        "ppi_len": np.ones(n, dtype=np.int8),
        "flowstats": np.zeros((n, P.FLOWSTATS_DIM), dtype=np.float32),
        "label": np.zeros(n, dtype=np.int16),
        "category": np.zeros(n, dtype=np.int8),
        "session_id": np.arange(n, dtype=np.int32),
        "ts": np.arange(n, dtype=np.int64),
    }
    with ShardWriter(
        tmp_path, "synthetic", "batch_boundary", label_map=LABEL_MAP, max_flows=10
    ) as w:
        w.add_batch(arrays)
        meta = w.close()

    assert meta["shard_sizes"] == [10, 10, 5]
    ss = ShardSet.open(tmp_path / "synthetic" / "batch_boundary")
    np.testing.assert_array_equal(ss.column("session_id"), np.arange(n))


def test_reference_flow_round_trip(reference_pcap, tmp_path):
    """The export path must not alter what FlowBuilder already produced."""
    (flow,) = build_flows(reference_pcap)
    with ShardWriter(tmp_path, "reference", "all", label_map={"tls": 0}) as w:
        w.add(flow, label=0, session_id=1)
    with ShardSet.open(tmp_path / "reference" / "all") as ss:
        got = ss[0]
        expected = np.array(EXPECTED_PPI, dtype=np.int16)
        np.testing.assert_array_equal(got["ppi"][: len(EXPECTED_PPI)], expected)
        assert int(got["ppi_len"]) == len(EXPECTED_PPI)
        assert int(got["session_id"]) == 1
