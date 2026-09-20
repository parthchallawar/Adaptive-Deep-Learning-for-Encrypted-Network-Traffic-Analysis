"""CESNET raw CSV -> shards (spec 001 Path B, plan T5b).

The 10-row fixture below uses the exact header recorded in
docs/datasets/cesnet-tls-year22.md (plan T5a) from a real file, and row 0's
PPI is EXPECTED_PPI from test_flows.py -- the same hand-computed golden array
the PCAP path is tested against, so this cross-checks that the CSV parser
reproduces the identical channel order and sign convention from a completely
different encoding.
"""

from __future__ import annotations

import csv
import json
import lzma
from pathlib import Path

import numpy as np
import pytest

from adl_etc.data import cesnet_csv as C
from adl_etc.data import ppi as P
from adl_etc.data.flows import FlowRecord
from adl_etc.data.tensors import ShardSet
from tests.data.test_flows import EXPECTED_PPI

IDENTIFIER_POISON = {
    "SRC_IP": "POISON-SRC-10.1.2.3",
    "DST_IP": "POISON-DST-10.9.8.7",
    "DST_ASN": "POISON-ASN-64500",
    "DST_PORT": "POISON-PORT-99999",
    "TLS_SNI": "poison.example.com",
    "TLS_JA3": "POISON-JA3-deadbeef",
}

HEADER = (*IDENTIFIER_POISON, *C.USE_COLUMNS)

# (time_first, duration, app, category, ipt, dir, size, push,
#  packets, packets_rev, bytes, bytes_rev, syn, ack)
_ROWS = [
    # Row 0: EXPECTED_PPI, hand-computed in test_flows.py -- cross-checked, not re-derived here.
    (
        "2022-01-03T10:00:00",
        0.470,
        "app-a",
        "cat-a",
        [p[0] for p in EXPECTED_PPI],
        [p[1] for p in EXPECTED_PPI],
        [p[2] for p in EXPECTED_PPI],
        [p[3] for p in EXPECTED_PPI],
        2,
        4,
        597,
        5880,
        1,
        1,
    ),
    ("2022-01-01T08:00:00", 0.0, "app-b", "cat-b", [0], [1], [100], [1], 1, 0, 100, 0, 1, 0),
    ("2022-01-01T09:00:00", 0.0, "app-a", "cat-a", [0], [1], [200], [0], 1, 0, 200, 0, 1, 0),
    (
        "2022-01-02T12:00:00",
        0.02,
        "app-c",
        "cat-a",
        [0, 20],
        [1, -1],
        [150, 300],
        [1, 0],
        1,
        1,
        150,
        300,
        1,
        1,
    ),
    ("2022-01-03T15:00:00", 0.0, "app-b", "cat-b", [0], [-1], [64], [0], 0, 1, 0, 64, 0, 1),
    ("2022-01-09T23:00:00", 0.0, "app-a", "cat-a", [0], [1], [50], [1], 1, 0, 50, 0, 1, 0),
    ("2022-01-10T00:30:00", 0.0, "app-c", "cat-a", [0], [1], [75], [1], 1, 0, 75, 0, 1, 0),
    ("2022-01-10T05:00:00", 0.0, "app-b", "cat-b", [0], [-1], [90], [0], 0, 1, 0, 90, 0, 1),
    ("2022-06-15T12:00:00", 0.0, "app-c", "cat-a", [0], [1], [500], [1], 1, 0, 500, 0, 1, 0),
    ("2022-12-31T20:00:00", 0.0, "app-a", "cat-a", [0], [1], [300], [1], 1, 0, 300, 0, 1, 0),
]


def _row_dict(row: tuple, *, protocol: int = 6) -> dict[str, object]:
    (
        time_first,
        duration,
        app,
        category,
        ipt,
        direction,
        size,
        push,
        packets,
        packets_rev,
        bytes_,
        bytes_rev,
        syn,
        ack,
    ) = row
    ppi_len = len(ipt)
    d: dict[str, object] = dict(IDENTIFIER_POISON)
    d.update(
        {
            "TIME_FIRST": time_first,
            "DURATION": duration,
            "BYTES": bytes_,
            "BYTES_REV": bytes_rev,
            "PACKETS": packets,
            "PACKETS_REV": packets_rev,
            "PPI_LEN": ppi_len,
            "PROTOCOL": protocol,
            "APP": app,
            "CATEGORY": category,
            "FLAG_SYN": syn,
            "FLAG_SYN_REV": 0,
            "FLAG_FIN": 0,
            "FLAG_FIN_REV": 0,
            "FLAG_RST": 0,
            "FLAG_RST_REV": 0,
            "FLAG_PSH": 0,
            "FLAG_PSH_REV": 0,
            "FLAG_ACK": ack,
            "FLAG_ACK_REV": 0,
            "FLAG_URG": 0,
            "FLAG_URG_REV": 0,
            "PPI": json.dumps([ipt, direction, size, push]),
        }
    )
    return d


def _write_rows(fh, rows: list[tuple]) -> None:
    writer = csv.DictWriter(fh, fieldnames=list(HEADER))
    writer.writeheader()
    for row in rows:
        writer.writerow(_row_dict(row))


def write_fixture_csv(path: Path, rows: list[tuple] = _ROWS, *, compress: bool = False) -> Path:
    if compress:
        with lzma.open(path, "wt", newline="") as fh:
            _write_rows(fh, rows)
    else:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            _write_rows(fh, rows)
    return path


# Real per-day files (spec: one flows-YYYYMMDD.csv.xz per day), grouped here
# purely as fixture-authoring convenience -- export_dataset only ever looks at
# each file's *name* for period assignment, never at row content, which is
# exactly the point of the regression test below.
_DAY_FILES: dict[str, list[tuple]] = {
    "20220101": [_ROWS[1], _ROWS[2]],
    "20220102": [_ROWS[3]],
    "20220103": [_ROWS[0], _ROWS[4]],  # row 0 carries EXPECTED_PPI
    "20220109": [_ROWS[5]],
    "20220110": [_ROWS[6], _ROWS[7]],
    "20220615": [_ROWS[8]],
    "20221231": [_ROWS[9]],
}


def write_day_files(dir_path: Path, day_files: dict[str, list[tuple]] = _DAY_FILES) -> list[Path]:
    return [
        write_fixture_csv(dir_path / f"flows-{date}.csv.xz", rows, compress=True)
        for date, rows in day_files.items()
    ]


# --- PPI parsing --------------------------------------------------------------


def test_parse_ppi_cell_matches_hand_computed_golden_array():
    cell = json.dumps(
        [
            [p[0] for p in EXPECTED_PPI],
            [p[1] for p in EXPECTED_PPI],
            [p[2] for p in EXPECTED_PPI],
            [p[3] for p in EXPECTED_PPI],
        ]
    )
    ppi, ppi_len = C.parse_ppi_cell(cell)
    assert ppi_len == len(EXPECTED_PPI)
    np.testing.assert_array_equal(ppi[:ppi_len], np.array(EXPECTED_PPI, dtype=P.PPI_DTYPE))
    assert not ppi[ppi_len:].any()


def test_parse_ppi_cell_no_padding_in_source_field():
    # A 3-packet cell must parse to exactly ppi_len=3, not assume 30 slots.
    cell = json.dumps([[0, 5, 10], [1, -1, 1], [100, 200, 300], [1, 0, 1]])
    ppi, ppi_len = C.parse_ppi_cell(cell)
    assert ppi_len == 3
    expected = np.array([[0, 1, 100, 1], [5, -1, 200, 0], [10, 1, 300, 1]], dtype=P.PPI_DTYPE)
    np.testing.assert_array_equal(ppi[:3], expected)


def test_parse_ppi_cell_clips_like_the_pcap_path():
    # ipt over IPT_MAX_MS and size over SIZE_MAX both occur in the real mirror
    # (confirmed in the schema doc); the CSV path must clip exactly like flows.py.
    ppi, ppi_len = C.parse_ppi_cell(json.dumps([[300_000], [1], [9000], [1]]))
    assert ppi_len == 1
    assert int(ppi[0, P.IPT_POS]) == P.IPT_MAX_MS
    assert int(ppi[0, P.SIZE_POS]) == P.SIZE_MAX


# --- row -> FlowRecord: flowstats is reused, not reimplemented -----------------


def test_row_to_flow_record_reuses_flowrecord_flowstats():
    row = _row_dict(_ROWS[0])
    rec = C.row_to_flow_record(row, session_id=3)
    assert rec.session_id == 3
    assert rec.packets == 2
    assert rec.packets_rev == 4
    assert rec.bytes == 597
    assert rec.bytes_rev == 5880
    # Same 46-column definition the PCAP path uses, computed from the same inputs.
    direct = FlowRecord(
        key=rec.key,
        proto=rec.proto,
        start_ts=rec.start_ts,
        end_ts=rec.end_ts,
        ppi=rec.ppi,
        ppi_len=rec.ppi_len,
        packets=rec.packets,
        packets_rev=rec.packets_rev,
        bytes=rec.bytes,
        bytes_rev=rec.bytes_rev,
        flags_seen=rec.flags_seen,
        end_reason="whatever-flowstats-ignores-this",
    )
    np.testing.assert_array_equal(rec.flowstats(), direct.flowstats())


def test_row_to_flow_record_flags_seen_combines_forward_and_reverse():
    from adl_etc.data.flows import TH_ACK, TH_SYN

    row = _row_dict(_ROWS[0])  # FLAG_SYN=1, FLAG_ACK=1, all _REV columns 0
    rec = C.row_to_flow_record(row)
    assert rec.flags_seen & TH_SYN
    assert rec.flags_seen & TH_ACK


def test_ppi_len_column_mismatch_raises():
    row = _row_dict(_ROWS[0])
    row["PPI_LEN"] = 3  # real column disagrees with the PPI field's own length (6)
    with pytest.raises(ValueError, match="PPI_LEN"):
        C.row_to_flow_record(row)


# --- week labeling: the real, non-ISO mirror convention -----------------------


@pytest.mark.parametrize(
    "timestamp,expected",
    [
        # ISO (2021, 52) -- folded into WEEK-2022-00, confirmed against the mirror.
        ("2022-01-01T00:00:00", "WEEK-2022-00"),
        ("2022-01-02T23:59:59", "WEEK-2022-00"),
        ("2022-01-03T00:00:00", "WEEK-2022-01"),  # true ISO week 1 starts here
        ("2022-01-09T23:59:59", "WEEK-2022-01"),
        ("2022-01-10T00:00:00", "WEEK-2022-02"),
        # Confirmed against the mirror: no rollover into 2023 this year.
        ("2022-12-26T00:00:00", "WEEK-2022-52"),
        ("2022-12-31T23:59:59", "WEEK-2022-52"),
    ],
)
def test_iso_week_matches_real_mirror_folders(timestamp, expected):
    assert C.iso_week(timestamp) == expected


def test_date_from_filename():
    assert C.date_from_filename(Path("flows-20220101.csv.xz")) == "2022-01-01"


def test_date_from_filename_rejects_wrong_name():
    with pytest.raises(ValueError):
        C.date_from_filename(Path("not-a-flows-file.csv"))


# --- class maps -----------------------------------------------------------------


def test_discover_class_maps_across_multiple_files(tmp_path):
    a = write_fixture_csv(tmp_path / "flows-a.csv", _ROWS[:5])
    b = write_fixture_csv(tmp_path / "flows-b.csv", _ROWS[5:])
    label_map, category_map = C.discover_class_maps([a, b])
    assert label_map == {"app-a": 0, "app-b": 1, "app-c": 2}
    assert category_map == {"cat-a": 0, "cat-b": 1}


# --- end-to-end export -----------------------------------------------------------


def test_export_writes_correct_weeks_and_golden_ppi(tmp_path):
    csv_paths = write_day_files(tmp_path)
    out_root = tmp_path / "processed"
    summary = C.export_dataset(
        csv_paths=csv_paths, dataset="cesnet-tls-year22", out_root=out_root, chunksize=4
    )

    assert summary.weeks == {
        "WEEK-2022-00": 3,  # 20220101 (2 rows) + 20220102 (1 row)
        "WEEK-2022-01": 3,  # 20220103 (2 rows, incl. EXPECTED_PPI) + 20220109 (1 row)
        "WEEK-2022-02": 2,  # 20220110
        "WEEK-2022-24": 1,  # 20220615
        "WEEK-2022-52": 1,  # 20221231
    }
    assert summary.n_flows == 10
    assert summary.dropped_zero_ppi == 0

    week01 = ShardSet.open(out_root / "cesnet-tls-year22" / "WEEK-2022-01")
    try:
        assert len(week01) == 3
        # The 20220103 file is processed first among this week's files, and its
        # first row carries EXPECTED_PPI.
        row0 = week01[0]
        assert int(row0["ppi_len"]) == len(EXPECTED_PPI)
        expected = np.array(EXPECTED_PPI, dtype=P.PPI_DTYPE)
        np.testing.assert_array_equal(row0["ppi"][: len(EXPECTED_PPI)], expected)
        assert week01.meta["label_map"] == {"app-a": 0, "app-b": 1, "app-c": 2}
        assert week01.meta["flowstats_source"] == "adl_etc"
    finally:
        week01.close()


def test_period_is_the_files_own_date_not_each_rows_time_first(tmp_path):
    # Reproduces the real bug found by running this exporter against the
    # actual mirror: flows-20220101.csv.xz's TIME_FIRST column spans two UTC
    # calendar dates (measured: 17,850 of 487,081 real rows land on
    # 2021-12-31 UTC, the first ~hour of CET-local New Year's Day), because
    # TIME_FIRST is UTC but the mirror's own file grouping isn't. Every row
    # in one file must land in that file's own period regardless.
    rows = [
        ("2021-12-31T23:30:00", 0.0, "app-a", "cat-a", [0], [1], [10], [1], 1, 0, 10, 0, 1, 0),
        ("2022-01-01T00:30:00", 0.0, "app-a", "cat-a", [0], [1], [20], [1], 1, 0, 20, 0, 1, 0),
    ]
    csv_path = write_fixture_csv(tmp_path / "flows-20220101.csv.xz", rows, compress=True)
    summary = C.export_dataset(
        csv_paths=[csv_path], dataset="cesnet-tls-year22", out_root=tmp_path / "processed"
    )
    assert summary.weeks == {"WEEK-2022-00": 2}


def test_export_never_writes_identifier_columns(tmp_path):
    csv_paths = write_day_files(tmp_path, {"20220103": _DAY_FILES["20220103"]})
    out_root = tmp_path / "processed"
    C.export_dataset(csv_paths=csv_paths, dataset="cesnet-tls-year22", out_root=out_root)

    for period_dir in (out_root / "cesnet-tls-year22").iterdir():
        meta_text = (period_dir / "meta.json").read_text(encoding="utf-8")
        parquet_bytes = (period_dir / "flows.parquet").read_bytes()
        for poison in IDENTIFIER_POISON.values():
            assert poison not in meta_text
            assert poison.encode("utf-8") not in parquet_bytes


def test_export_handles_a_genuinely_empty_day(tmp_path):
    # Confirmed against the real mirror: some days are zero bytes once
    # decompressed, not a header-only CSV (docs/datasets/cesnet-tls-year22.md).
    # It must still be a *valid* (empty) lzma stream, matching the real file
    # byte-for-byte in shape -- a bare 0-byte file isn't the same failure
    # (that raises EOFError from lzma, not pandas' EmptyDataError) and isn't
    # what the mirror actually serves.
    empty = tmp_path / "flows-20221231.csv.xz"
    with lzma.open(empty, "wb"):
        pass
    non_empty = write_fixture_csv(tmp_path / "flows-20220101.csv.xz", _ROWS[:2], compress=True)

    label_map, category_map = C.discover_class_maps([empty, non_empty])
    assert label_map == {"app-a": 0, "app-b": 1}

    summary = C.export_dataset(
        csv_paths=[empty, non_empty],
        dataset="cesnet-tls-year22",
        out_root=tmp_path / "processed",
        label_map=label_map,
        category_map=category_map,
    )
    assert summary.n_flows == 2
    assert summary.rows_processed == 2
    # The empty file contributes no rows, so its week never appears here...
    assert summary.weeks == {"WEEK-2022-00": 2}
    # ...but its (empty) shard set was still written, not skipped outright.
    empty_meta = json.loads(
        (tmp_path / "processed" / "cesnet-tls-year22" / "WEEK-2022-52" / "meta.json").read_text()
    )
    assert empty_meta["n_flows"] == 0


# --- --verify -----------------------------------------------------------------


def test_verify_day_passes_on_consistent_data(tmp_path):
    csv_path = write_fixture_csv(tmp_path / "flows-20220101.csv.xz", _ROWS[:3], compress=True)
    stats = {"global": {"total-saved": 3}, "apps": {"app-a": 2, "app-b": 1}}
    stats_path = tmp_path / "stats-20220101.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    result = C.verify_day(csv_path, stats_path)
    assert result.ok
    # The DataZoo comparison is explicitly noted as skipped, not silently omitted.
    assert any("skipped" in p for p in result.problems)


def test_verify_day_fails_loudly_on_corrupted_count(tmp_path):
    csv_path = write_fixture_csv(tmp_path / "flows-20220101.csv.xz", _ROWS[:3], compress=True)
    stats = {"global": {"total-saved": 999}, "apps": {"app-a": 2, "app-b": 1}}
    stats_path = tmp_path / "stats-20220101.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    result = C.verify_day(csv_path, stats_path)
    assert not result.ok
    assert any("row count" in p for p in result.problems)


def test_verify_day_flags_app_not_in_stats(tmp_path):
    csv_path = write_fixture_csv(tmp_path / "flows-20220101.csv.xz", _ROWS[:3], compress=True)
    # app-b is missing from stats.apps but present in the CSV.
    stats = {"global": {"total-saved": 3}, "apps": {"app-a": 2}}
    stats_path = tmp_path / "stats-20220101.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    result = C.verify_day(csv_path, stats_path)
    assert not result.ok
    assert any("app-b" in p for p in result.problems)


def test_verify_day_handles_empty_day(tmp_path):
    csv_path = tmp_path / "flows-20221231.csv.xz"
    with lzma.open(csv_path, "wb"):
        pass  # zero bytes once decompressed, matching the real mirror's empty-day file
    stats_path = tmp_path / "stats-20221231.json"
    stats_path.write_text(json.dumps({"global": {"total-saved": 0}, "apps": {}}), encoding="utf-8")

    result = C.verify_day(csv_path, stats_path)
    assert result.ok


def test_stats_path_for_rejects_wrong_name():
    with pytest.raises(ValueError):
        C._stats_path_for(Path("not-a-flows-file.csv.xz"))


# --- manifest registration merges, does not replace ----------------------------


def test_register_manifest_merges_across_calls(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    a = write_fixture_csv(tmp_path / "flows-a.csv", _ROWS[:2])
    b = write_fixture_csv(tmp_path / "flows-b.csv", _ROWS[2:4])

    C.register_manifest([a], manifest_path=manifest_path)
    entry = C.register_manifest([b], manifest_path=manifest_path)

    names = {f["name"] for f in entry["files"]}
    assert names == {"flows-a.csv", "flows-b.csv"}
