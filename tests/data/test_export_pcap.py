"""PCAP-to-shard export (spec 002, plan T4)."""

from __future__ import annotations

import csv

import numpy as np
import pyarrow.parquet as pq
import pytest

from adl_etc.data import manifest as M
from adl_etc.data.export_pcap import PcapConfig, export_dataset
from adl_etc.data.tensors import ShardSet
from tests.conftest import CLIENT_IP, REFERENCE_PACKETS, SERVER_IP, write_pcap
from tests.data.test_flows import EXPECTED_PPI


def _write_dataset(tmp_path, dataset: str, files: dict[str, list], labels_rows: list[dict]):
    """Builds a data/raw/<dataset>/{pcap/*.pcap,labels.csv} tree under tmp_path."""
    raw_root = tmp_path / "raw"
    pcap_dir = raw_root / dataset / "pcap"
    pcap_dir.mkdir(parents=True)
    for name, packets in files.items():
        write_pcap(pcap_dir / name, packets)

    labels_path = raw_root / dataset / "labels.csv"
    fieldnames = sorted({k for row in labels_rows for k in row})
    with open(labels_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labels_rows)
    return raw_root


def test_reference_export_matches_golden_ppi(tmp_path):
    raw_root = _write_dataset(
        tmp_path,
        "d",
        {"a.pcap": REFERENCE_PACKETS},
        [{"file_name": "a.pcap", "class_name": "tls", "category": "browsing"}],
    )

    summary = export_dataset(
        dataset="d", raw_root=raw_root, out_root=tmp_path / "processed", log=lambda _: None
    )

    assert summary.n_flows == 1
    assert summary.n_files == 1
    assert summary.n_files_skipped_unlabeled == 0

    ss = ShardSet.open(tmp_path / "processed" / "d" / "all")
    got = ss[0]
    expected = np.array(EXPECTED_PPI, dtype=np.int16)
    np.testing.assert_array_equal(got["ppi"][: len(EXPECTED_PPI)], expected)
    assert int(got["ppi_len"]) == len(EXPECTED_PPI)
    assert int(got["label"]) == 0  # only one class -> label_map = {"tls": 0}


def test_two_files_get_distinct_session_ids_and_no_shared_flows(tmp_path):
    raw_root = _write_dataset(
        tmp_path,
        "d",
        {"a.pcap": REFERENCE_PACKETS, "b.pcap": REFERENCE_PACKETS},  # same flow, different files
        [
            {"file_name": "a.pcap", "class_name": "tls", "category": "browsing"},
            {"file_name": "b.pcap", "class_name": "tls", "category": "browsing"},
        ],
    )

    export_dataset(
        dataset="d", raw_root=raw_root, out_root=tmp_path / "processed", log=lambda _: None
    )

    ss = ShardSet.open(tmp_path / "processed" / "d" / "all")
    assert len(ss) == 2
    session_ids = sorted(int(v) for v in ss.column("session_id"))
    assert session_ids == [0, 1]


def test_flows_parquet_row_count_matches_and_key_hash_not_readable(tmp_path):
    raw_root = _write_dataset(
        tmp_path,
        "d",
        {"a.pcap": REFERENCE_PACKETS},
        [{"file_name": "a.pcap", "class_name": "tls", "category": "browsing"}],
    )

    export_dataset(
        dataset="d", raw_root=raw_root, out_root=tmp_path / "processed", log=lambda _: None
    )

    period_dir = tmp_path / "processed" / "d" / "all"
    ss = ShardSet.open(period_dir)
    table = pq.read_table(period_dir / "flows.parquet")
    assert table.num_rows == len(ss)

    client_ip_str = ".".join(str(b) for b in CLIENT_IP)
    server_ip_str = ".".join(str(b) for b in SERVER_IP)
    for key_hash in table.column("key_hash").to_pylist():
        assert client_ip_str not in key_hash
        assert server_ip_str not in key_hash
        assert len(key_hash) == 40  # sha1 hex digest length


def test_unlabeled_files_are_skipped(tmp_path):
    raw_root = _write_dataset(
        tmp_path,
        "d",
        {"a.pcap": REFERENCE_PACKETS, "b.pcap": REFERENCE_PACKETS},
        [{"file_name": "a.pcap", "class_name": "tls", "category": "browsing"}],  # b.pcap unlabelled
    )

    summary = export_dataset(
        dataset="d", raw_root=raw_root, out_root=tmp_path / "processed", log=lambda _: None
    )

    assert summary.n_files == 1
    assert summary.n_files_skipped_unlabeled == 1


def test_empty_class_name_is_treated_as_unlabeled(tmp_path):
    """The heuristic ISCX labeller can leave class_name empty; that file must
    be skipped entirely, not folded into the -1 unknown-class convention."""
    raw_root = _write_dataset(
        tmp_path,
        "d",
        {"a.pcap": REFERENCE_PACKETS},
        [{"file_name": "a.pcap", "class_name": "", "category": ""}],
    )

    summary = export_dataset(
        dataset="d", raw_root=raw_root, out_root=tmp_path / "processed", log=lambda _: None
    )

    assert summary.n_files == 0
    assert summary.n_files_skipped_unlabeled == 1


def test_files_glob_restricts_selection(tmp_path):
    raw_root = _write_dataset(
        tmp_path,
        "d",
        {"youtube1.pcap": REFERENCE_PACKETS, "netflix1.pcap": REFERENCE_PACKETS},
        [
            {"file_name": "youtube1.pcap", "class_name": "streaming", "category": "streaming"},
            {"file_name": "netflix1.pcap", "class_name": "streaming", "category": "streaming"},
        ],
    )

    summary = export_dataset(
        dataset="d",
        raw_root=raw_root,
        out_root=tmp_path / "processed",
        files_glob="*netflix*",
        log=lambda _: None,
    )

    assert summary.n_files == 1


def test_multiple_classes_get_distinct_labels(tmp_path):
    raw_root = _write_dataset(
        tmp_path,
        "d",
        {"a.pcap": REFERENCE_PACKETS, "b.pcap": REFERENCE_PACKETS},
        [
            {"file_name": "a.pcap", "class_name": "aim", "category": "chat"},
            {"file_name": "b.pcap", "class_name": "youtube", "category": "streaming"},
        ],
    )

    export_dataset(
        dataset="d", raw_root=raw_root, out_root=tmp_path / "processed", log=lambda _: None
    )

    meta = ShardSet.open(tmp_path / "processed" / "d" / "all").meta
    assert meta["label_map"] == {"aim": 0, "youtube": 1}  # sorted, deterministic
    assert meta["category_map"] == {"chat": 0, "streaming": 1}


def test_backend_other_than_dpkt_raises(tmp_path):
    raw_root = _write_dataset(tmp_path, "d", {}, [])
    with pytest.raises(NotImplementedError):
        export_dataset(
            dataset="d",
            raw_root=raw_root,
            out_root=tmp_path / "processed",
            config=PcapConfig(backend="ipfixprobe"),
            log=lambda _: None,
        )


def test_source_manifest_hash_recorded_when_manifest_exists(tmp_path):
    raw_root = _write_dataset(
        tmp_path,
        "d",
        {"a.pcap": REFERENCE_PACKETS},
        [{"file_name": "a.pcap", "class_name": "tls", "category": "browsing"}],
    )
    manifest_path = tmp_path / "manifest.json"
    M.register(
        "d",
        source_url="http://example.test",
        files=[{"name": "a.pcap", "bytes": 1, "sha256": "x"}],
        manifest_path=manifest_path,
    )

    export_dataset(
        dataset="d",
        raw_root=raw_root,
        out_root=tmp_path / "processed",
        manifest_path=manifest_path,
        log=lambda _: None,
    )

    meta = ShardSet.open(tmp_path / "processed" / "d" / "all").meta
    assert meta["source_manifest_hash"] is not None


def test_no_manifest_entry_is_not_fatal(tmp_path):
    raw_root = _write_dataset(
        tmp_path,
        "d",
        {"a.pcap": REFERENCE_PACKETS},
        [{"file_name": "a.pcap", "class_name": "tls", "category": "browsing"}],
    )

    summary = export_dataset(
        dataset="d",
        raw_root=raw_root,
        out_root=tmp_path / "processed",
        manifest_path=tmp_path / "nonexistent-manifest.json",
        log=lambda _: None,
    )

    assert summary.n_flows == 1
    meta = ShardSet.open(tmp_path / "processed" / "d" / "all").meta
    assert meta["source_manifest_hash"] is None


def test_summary_render_reports_throughput(tmp_path):
    raw_root = _write_dataset(
        tmp_path,
        "d",
        {"a.pcap": REFERENCE_PACKETS},
        [{"file_name": "a.pcap", "class_name": "tls", "category": "browsing"}],
    )

    summary = export_dataset(
        dataset="d", raw_root=raw_root, out_root=tmp_path / "processed", log=lambda _: None
    )

    text = summary.render()
    assert "flows: 1 written" in text
    assert "pkt/s" in text
    assert summary.packets_processed == len(REFERENCE_PACKETS)
