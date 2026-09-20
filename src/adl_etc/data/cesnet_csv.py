"""CESNET raw weekly CSV -> shards (spec 001 Path B, plan T5b).

Reads ``flows-YYYYMMDD.csv.xz`` from the ``pranjalkar99/cesnet-22`` Kaggle
mirror directly, without DataZoo. The header, the ``PPI`` column's encoding
and which columns are identifiers were not assumed: they were read out of a
real file and recorded in ``docs/datasets/cesnet-tls-year22.md`` (plan T5a)
before this module was written. Every design choice below cites that record.

Flow-statistics computation is **not reimplemented here**. Each row already
carries everything :meth:`~adl_etc.data.flows.FlowRecord.flowstats` needs
(BYTES, PACKETS, PPI, timestamps, flags), so a :class:`FlowRecord` is built
per row and its own ``flowstats()`` is called — the same 46-column
definition D3/D4 use, computed once, not duplicated for a second source.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adl_etc.data import manifest as M
from adl_etc.data import ppi as P
from adl_etc.data.flows import TH_ACK, TH_FIN, TH_PUSH, TH_RST, TH_SYN, TH_URG, FlowRecord
from adl_etc.data.tensors import ShardWriter
from adl_etc.utils.provenance import file_sha256, stable_hash

# --- schema, as recorded in docs/datasets/cesnet-tls-year22.md -------------

IDENTIFIER_COLUMNS = ("SRC_IP", "DST_IP", "DST_ASN", "DST_PORT", "TLS_SNI", "TLS_JA3")
"""Never read from the CSV at all (spec 001's privacy rule), not merely
dropped after parsing."""

USE_COLUMNS = (
    "TIME_FIRST",
    "DURATION",
    "BYTES",
    "BYTES_REV",
    "PACKETS",
    "PACKETS_REV",
    "PPI_LEN",
    "PROTOCOL",
    "APP",
    "CATEGORY",
    "FLAG_SYN",
    "FLAG_SYN_REV",
    "FLAG_FIN",
    "FLAG_FIN_REV",
    "FLAG_RST",
    "FLAG_RST_REV",
    "FLAG_PSH",
    "FLAG_PSH_REV",
    "FLAG_ACK",
    "FLAG_ACK_REV",
    "FLAG_URG",
    "FLAG_URG_REV",
    "PPI",
)
"""Every column this exporter actually reads. Deliberately excludes every
identifier column and CESNET's own ``PHIST_*`` histograms (we compute our
own from the parsed ``PPI``, per the project's "one definition, one place"
rule — see the schema doc)."""

CLASS_SCAN_COLUMNS = ("APP", "CATEGORY")


def _read_csv_chunks(
    path: Path, *, usecols: Sequence[str], chunksize: int
) -> Iterator[pd.DataFrame]:
    """``pd.read_csv(..., chunksize=...)``, tolerating a genuinely empty day.

    Confirmed against a real file: the mirror ships some zero-flow days
    (``stats-20221231.json`` reports ``total-saved: 0``) as a
    **completely empty** ``.csv.xz`` -- no header row, not even a comma --
    which makes ``pandas.read_csv`` raise ``EmptyDataError`` rather than
    yielding zero rows. Every caller here streams through this helper
    instead of calling ``pd.read_csv`` directly, so a zero-flow day is
    treated as zero flows, not a crash.
    """
    import pandas.errors

    try:
        yield from pd.read_csv(path, usecols=list(usecols), chunksize=chunksize)
    except pandas.errors.EmptyDataError:
        return

_FLAG_PAIRS: tuple[tuple[str, str, int], ...] = (
    ("FLAG_SYN", "FLAG_SYN_REV", TH_SYN),
    ("FLAG_FIN", "FLAG_FIN_REV", TH_FIN),
    ("FLAG_RST", "FLAG_RST_REV", TH_RST),
    ("FLAG_PSH", "FLAG_PSH_REV", TH_PUSH),
    ("FLAG_ACK", "FLAG_ACK_REV", TH_ACK),
    ("FLAG_URG", "FLAG_URG_REV", TH_URG),
)

_DUMMY_KEY = ((b"", 0), (b"", 0), 0)
"""FlowRecord.key exists for the PCAP path's parquet key-hash. CESNET rows
carry no 5-tuple (dropped before this module ever sees the file, per spec
001) and are written through ShardWriter.add_batch, which never reads
.key -- the value here is never used."""


# --- PPI parsing -------------------------------------------------------------


def parse_ppi_cell(cell: str) -> tuple[np.ndarray, int]:
    """One CSV ``PPI`` field -> ``(ppi[K_MAX, 4] int16, ppi_len)``.

    The field is ``"[[ipt_ms...], [dir...], [size...], [push...]]"``, already
    ``ppi.py``'s channel order and sign convention (confirmed in the schema
    doc, not assumed), with no padding of its own -- every inner list has
    exactly ``ppi_len`` elements, never 30. Parsed with :func:`json.loads`,
    not ``ast.literal_eval``: both parse every real cell correctly, but
    ``json.loads`` measured ~17x faster on 50,000 real cells, and this call
    happens once per row across the whole corpus.
    """
    ipt, direction, size, push = json.loads(cell)
    n = len(ipt)
    ppi_len = min(n, P.K_MAX)
    out = P.empty_ppi()
    for k in range(ppi_len):
        out[k] = (P.clip_ipt(ipt[k]), direction[k], P.clip_size(size[k]), push[k])
    return out, ppi_len


def _flags_seen(row: Mapping[str, Any]) -> int:
    bits = 0
    for fwd, rev, bit in _FLAG_PAIRS:
        if row[fwd] or row[rev]:
            bits |= bit
    return bits


def row_to_flow_record(
    row: Mapping[str, Any], *, session_id: int = 0, start_ts: float | None = None
) -> FlowRecord:
    """One parsed CSV row -> a :class:`FlowRecord`, so ``flowstats()`` is
    computed by the one shared definition rather than reimplemented here.
    ``row`` is anything with string-keyed lookup (a dict, a ``pd.Series``);
    plain dicts (e.g. a namedtuple's ``_asdict()``) are cheaper per row than
    wrapping each row in a fresh ``pd.Series``. ``start_ts`` can be passed in
    already parsed to avoid parsing ``TIME_FIRST`` twice when the caller also
    needs it for :func:`iso_week`."""
    ppi_arr, ppi_len = parse_ppi_cell(row["PPI"])
    reported_len = int(row["PPI_LEN"])
    if reported_len != ppi_len and reported_len <= P.K_MAX:
        raise ValueError(
            f"PPI_LEN column ({reported_len}) disagrees with the PPI field's own "
            f"length ({ppi_len}); they should always match for a row not truncated "
            f"at K_MAX, so this points at a real encoding mismatch, not noise"
        )
    ts: float = start_ts if start_ts is not None else pd.Timestamp(row["TIME_FIRST"]).timestamp()
    proto = int(row["PROTOCOL"])
    return FlowRecord(
        key=_DUMMY_KEY,
        proto=proto,
        start_ts=ts,
        end_ts=ts + float(row["DURATION"]),
        ppi=ppi_arr,
        ppi_len=ppi_len,
        packets=int(row["PACKETS"]),
        packets_rev=int(row["PACKETS_REV"]),
        bytes=int(row["BYTES"]),
        bytes_rev=int(row["BYTES_REV"]),
        flags_seen=_flags_seen(row),
        end_reason="n/a",  # discarded by add_batch; CESNET's own reason isn't in our vocabulary
        session_id=session_id,
    )


def iso_week(timestamp: str | pd.Timestamp) -> str:
    """A timestamp -> its period label, matching this mirror's own
    ``WEEK-YYYY-NN`` folder names exactly -- checked against the real
    directory layout, not assumed to be plain ISO calendar week.

    For most of the year this *is* the ISO calendar week
    (``WEEK-<iso_year>-<iso_week:02d>``). One edge doesn't follow that rule:
    January 1-2, whose ISO week technically belongs to the **previous** ISO
    year (``pd.Timestamp("2022-01-01").isocalendar()`` returns ``(2021, 52,
    6)``), are folded into ``WEEK-2022-00`` instead of ``WEEK-2021-52`` --
    confirmed by listing the mirror directly: ``WEEK-2022-00/`` holds
    exactly 2022-01-01 and 2022-01-02, and true ISO week 1 of 2022
    (``WEEK-2022-01/``) starts the following Monday, 2022-01-03.

    The symmetric case at the *end* of the year (a December date whose ISO
    week rolls forward into next year, e.g. 2018-12-31 -> ISO 2019-W01)
    does not occur in 2022 -- also confirmed directly: ``WEEK-2022-52/``
    runs 2022-12-26 to 2022-12-31, and neither ``WEEK-2022-53`` nor
    ``WEEK-2023-00`` exists in this mirror. That case is therefore
    unverified and would need the same kind of check before this function
    is pointed at a different year (e.g. spec 001's optional
    CESNET-QUICEXT-25).
    """
    ts = pd.Timestamp(timestamp)
    iso_year, iso_week_num, _ = ts.isocalendar()
    if iso_year < ts.year:
        return f"WEEK-{ts.year}-00"
    return f"WEEK-{int(iso_year)}-{int(iso_week_num):02d}"


# --- class maps --------------------------------------------------------------


def discover_class_maps(
    csv_paths: Sequence[Path], *, chunksize: int = 200_000
) -> tuple[dict[str, int], dict[str, int]]:
    """Scans ``APP``/``CATEGORY`` only (cheap relative to parsing ``PPI``) to
    build stable ``label_map``/``category_map`` covering every file passed in
    one call. Must be given every file a multi-week export will ever write,
    together, in one call -- a label map built from one week and reused for
    the next would assign the same app two different integers across the
    corpus."""
    apps: set[str] = set()
    categories: set[str] = set()
    for path in csv_paths:
        for chunk in _read_csv_chunks(path, usecols=CLASS_SCAN_COLUMNS, chunksize=chunksize):
            apps.update(chunk["APP"].unique().tolist())
            categories.update(chunk["CATEGORY"].unique().tolist())
    label_map = {name: i for i, name in enumerate(sorted(apps))}
    category_map = {name: i for i, name in enumerate(sorted(categories))}
    return label_map, category_map


# --- manifest registration ----------------------------------------------------

CESNET_KAGGLE_DATASET = "pranjalkar99/cesnet-22"


def register_manifest(
    csv_paths: Sequence[Path],
    *,
    dataset: str = "cesnet-tls-year22",
    manifest_path: str | Path = M.DEFAULT_MANIFEST_PATH,
    kaggle_dataset: str = CESNET_KAGGLE_DATASET,
) -> dict[str, Any]:
    """Registers (or extends) ``dataset``'s manifest entry with every file in
    ``csv_paths``, hashed for real. Merges with whatever files are already
    registered (by name) rather than replacing them, since the corpus is
    pulled one day/week at a time across many separate invocations, unlike
    D3/D4's single-archive downloads."""
    existing: dict[str, dict[str, Any]] = {}
    try:
        entry = M.require(dataset, manifest_path=manifest_path)
        existing = {f["name"]: f for f in entry.get("files", [])}
    except KeyError:
        pass
    for path in csv_paths:
        existing[path.name] = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    files = [existing[name] for name in sorted(existing)]
    return M.register(
        dataset,
        source_url=f"kaggle:{kaggle_dataset}",
        files=files,
        extracted_to=None,
        manifest_path=manifest_path,
    )


# --- export --------------------------------------------------------------------


@dataclass
class ExportSummary:
    dataset: str
    n_files: int
    n_files_skipped_unlabeled: int
    weeks: dict[str, int] = field(default_factory=dict)
    """period -> n_flows written to that week's shard set."""
    dropped_zero_ppi: int = 0
    rows_processed: int = 0
    wall_seconds: float = 0.0

    @property
    def n_flows(self) -> int:
        return sum(self.weeks.values())

    @property
    def rows_per_second(self) -> float:
        return self.rows_processed / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def render(self) -> str:
        header = f"{self.dataset}: {self.n_files} files exported across {len(self.weeks)} week(s)"
        if self.n_files_skipped_unlabeled:
            header += f" ({self.n_files_skipped_unlabeled} skipped, no matching stats file)"
        lines = [
            header,
            f"  flows: {self.n_flows} written, {self.dropped_zero_ppi} dropped (zero PPI)",
            f"  rows: {self.rows_processed} in {self.wall_seconds:.1f}s "
            f"({self.rows_per_second:,.0f} rows/s)",
        ]
        for week in sorted(self.weeks):
            lines.append(f"    {week}: {self.weeks[week]} flows")
        return "\n".join(lines)


def date_from_filename(path: Path) -> str:
    """``flows-20220101.csv.xz`` -> ``"2022-01-01"``.

    This is the mirror's own partition key for the file, and it is **not**
    the same thing as "the calendar date most of this file's ``TIME_FIRST``
    values fall on": confirmed against the real 2022-01-01 file, whose
    ``TIME_FIRST`` column actually spans two UTC calendar dates (487,081
    rows on 2022-01-01, but 17,850 -- the first ~hour of CET-local time --
    landing on 2021-12-31 UTC). ``TIME_FIRST`` is UTC; the mirror's own
    day/week grouping is not. Partitioning by each row's own timestamp
    would therefore put ~3.7% of this one file's flows in a
    ``WEEK-2021-52`` shard set that doesn't correspond to anything in the
    mirror's real layout -- caught by running this exporter against the
    real file, not the fixture, before calling T5b done.
    """
    name = path.name
    if not name.startswith("flows-") or not name.endswith(".csv.xz"):
        raise ValueError(f"{path}: expected a flows-YYYYMMDD.csv.xz file name")
    digits = name[len("flows-") : -len(".csv.xz")]
    return f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}"


def _row_arrays(
    chunk: pd.DataFrame,
    *,
    label_map: dict[str, int],
    category_map: dict[str, int],
    session_id: int,
) -> dict[str, list[Any]]:
    """Parses every row of ``chunk`` into ``ShardWriter.add_batch``-ready
    column lists. Every row in ``chunk`` belongs to the same source file and
    therefore the same period -- see :func:`date_from_filename` for why that
    is keyed off the file, not off each row's own ``TIME_FIRST``."""
    cols: dict[str, list[Any]] = {
        "ppi": [],
        "ppi_len": [],
        "flowstats": [],
        "label": [],
        "category": [],
        "session_id": [],
        "ts": [],
    }
    for row in chunk.itertuples(index=False):
        row_d = row._asdict()
        rec = row_to_flow_record(row_d, session_id=session_id)
        cols["ppi"].append(rec.ppi)
        cols["ppi_len"].append(rec.ppi_len)
        cols["flowstats"].append(rec.flowstats())
        cols["label"].append(label_map[row_d["APP"]])
        cols["category"].append(category_map[row_d["CATEGORY"]])
        cols["session_id"].append(session_id)
        cols["ts"].append(round(rec.start_ts * 1000))
    return cols


def _to_batch(cols: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    dtype, shape = P.PPI_DTYPE, (P.K_MAX, P.PPI_CHANNELS)
    return {
        "ppi": np.asarray(cols["ppi"], dtype=dtype).reshape(-1, *shape),
        "ppi_len": np.asarray(cols["ppi_len"], dtype=np.int8),
        "flowstats": np.asarray(cols["flowstats"], dtype=np.float32),
        "label": np.asarray(cols["label"], dtype=np.int16),
        "category": np.asarray(cols["category"], dtype=np.int8),
        "session_id": np.asarray(cols["session_id"], dtype=np.int32),
        "ts": np.asarray(cols["ts"], dtype=np.int64),
    }


def export_dataset(
    *,
    csv_paths: Sequence[Path],
    dataset: str = "cesnet-tls-year22",
    out_root: str | Path = Path("data/processed"),
    label_map: dict[str, int] | None = None,
    category_map: dict[str, int] | None = None,
    chunksize: int = 100_000,
    max_flows: int = 500_000,
    manifest_path: str | Path = M.DEFAULT_MANIFEST_PATH,
    overwrite: bool = False,
    log: Callable[[str], None] = print,
) -> ExportSummary:
    """Streams every file in ``csv_paths`` and writes shards partitioned by
    the source file's own date (:func:`date_from_filename`), which determines
    the week via :func:`iso_week` -- **not** by each row's own ``TIME_FIRST``.
    A row's ``TIME_FIRST`` is UTC while the mirror's own file grouping isn't,
    so a small fraction of any file's rows have a ``TIME_FIRST`` that falls on
    the *previous* UTC calendar date; partitioning by row would scatter a
    handful of flows per file into a shard set that doesn't correspond to
    anything in the mirror's real layout (measured: 17,850 of 487,081 rows,
    3.7%, in the one real file this was checked against). Every row from one
    file always lands in that one file's period.

    ``label_map``/``category_map`` must cover every class in every file
    passed here; when omitted they are discovered with one cheap pre-pass
    (:func:`discover_class_maps`) over exactly the files given, so callers
    processing the corpus in batches must pass every file for a given map
    together in one call, or pass an explicit map built once up front.
    """
    csv_paths = list(csv_paths)
    if label_map is None or category_map is None:
        label_map, category_map = discover_class_maps(csv_paths)

    source_manifest_hash = None
    try:
        entry = M.require(dataset, manifest_path=manifest_path)
        have = {f["name"] for f in entry.get("files", [])}
        if all(p.name in have for p in csv_paths):
            source_manifest_hash = stable_hash({"files": entry["files"]})
    except KeyError:
        pass  # no manifest entry yet is not fatal, same as export_pcap.py

    writers: dict[str, ShardWriter] = {}
    week_flow_counts: dict[str, int] = {}
    dropped_zero_ppi = 0
    rows_processed = 0

    t0 = time.perf_counter()
    try:
        for session_id, path in enumerate(csv_paths):
            week = iso_week(date_from_filename(path))
            writer = writers.get(week)
            if writer is None:
                writer = ShardWriter(
                    out_root,
                    dataset,
                    week,
                    label_map=label_map,
                    category_map=category_map,
                    max_flows=max_flows,
                    source_manifest_hash=source_manifest_hash,
                    flowstats_source="adl_etc",
                    overwrite=overwrite,
                )
                writers[week] = writer
            for chunk in _read_csv_chunks(path, usecols=USE_COLUMNS, chunksize=chunksize):
                rows_processed += len(chunk)
                cols = _row_arrays(
                    chunk, label_map=label_map, category_map=category_map, session_id=session_id
                )
                batch = _to_batch(cols)
                n_zero = int(np.sum(batch["ppi_len"] == 0))
                dropped_zero_ppi += n_zero
                week_flow_counts[week] = week_flow_counts.get(week, 0) + (
                    len(batch["ppi_len"]) - n_zero
                )
                writer.add_batch(batch)
        for writer in writers.values():
            writer.close()
    except BaseException:
        # ShardWriter.close() is documented safe to call more than once, so no
        # need to track which writers already closed before the failure.
        for writer in writers.values():
            writer.close()
        raise
    wall = time.perf_counter() - t0

    summary = ExportSummary(
        dataset=dataset,
        n_files=len(csv_paths),
        n_files_skipped_unlabeled=0,
        weeks=week_flow_counts,
        dropped_zero_ppi=dropped_zero_ppi,
        rows_processed=rows_processed,
        wall_seconds=wall,
    )
    log(summary.render())
    return summary


# --- verification --------------------------------------------------------------


@dataclass
class VerifyResult:
    path: Path
    ok: bool
    problems: list[str] = field(default_factory=list)


def _stats_path_for(csv_path: Path) -> Path:
    """``flows-20220101.csv.xz`` -> ``stats-20220101.json`` in the same dir."""
    date_part = date_from_filename(csv_path).replace("-", "")
    return csv_path.with_name(f"stats-{date_part}.json")


def verify_day(csv_path: Path, stats_path: Path | None = None) -> VerifyResult:
    """Spec 001's mandatory Path-B checks, for one day: the row count matches
    ``stats-*.json``'s ``global.total-saved``, and every ``APP`` value in the
    CSV is one of ``stats-*.json``'s ``apps`` keys. The third check (a sampled
    comparison against the DataZoo HDF5) is skipped here and noted as such:
    this project has not downloaded a DataZoo HDF5 to compare against, per
    spec 001's own "when it is available locally" wording."""
    stats_path = stats_path or _stats_path_for(csv_path)
    problems: list[str] = []
    if not stats_path.exists():
        return VerifyResult(csv_path, ok=False, problems=[f"missing stats file {stats_path}"])
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    expected_total = stats.get("global", {}).get("total-saved")
    expected_apps = set(stats.get("apps", {}))

    n_rows = 0
    seen_apps: set[str] = set()
    for chunk in _read_csv_chunks(csv_path, usecols=["APP"], chunksize=200_000):
        n_rows += len(chunk)
        seen_apps.update(chunk["APP"].unique().tolist())

    if expected_total is not None and n_rows != expected_total:
        problems.append(f"row count {n_rows} != stats total-saved {expected_total}")
    unexpected_apps = seen_apps - expected_apps
    if unexpected_apps:
        problems.append(f"APP values not in stats.apps: {sorted(unexpected_apps)}")
    problems.append("DataZoo field-level comparison skipped: no local HDF5 to compare against")

    ok = n_rows == expected_total and not unexpected_apps
    return VerifyResult(csv_path, ok=ok, problems=problems)
