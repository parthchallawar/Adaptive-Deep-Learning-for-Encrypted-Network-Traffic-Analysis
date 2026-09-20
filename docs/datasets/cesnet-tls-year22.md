# CESNET-TLS-Year22 (D1)

- **Role:** primary dataset — supervised training, drift study, open-set (spec 001).
- **Manifest entry:** `cesnet-tls-year22-probe` (`data/manifest.json`) for the one real day exported so far; the full corpus will register as `cesnet-tls-year22` once exported.
- **Source and retrieval:** Path B (spec 001) — the third-party Kaggle mirror `pranjalkar99/cesnet-22` of the original CESNET release, pulled per file/day with the Kaggle CLI:
  ```
  kaggle datasets download -d pranjalkar99/cesnet-22 \
      -f CESNET-TLS-Year22/<WEEK-YYYY-NN>/<date>/flows-<date>.csv.xz -p data/raw/cesnet-tls-year22
  ```
  Path A (the canonical `cesnet-datazoo` HDF5 download) is the documented fallback and verification source (spec 001) but has not been used in this project yet.
- **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International) — confirmed on the dataset's own Zenodo record, not just the paper.
- **Citation:** Hynek, K., Luxemburk, J., Pešek, J., Čejka, T. & Šiška, P. "CESNET-TLS-Year22: A year-spanning TLS network traffic dataset from backbone lines." *Scientific Data* (2024). DOI: [10.5281/zenodo.10608607](https://doi.org/10.5281/zenodo.10608607).
- **Size:** ~30.5 GB compressed (main archive, Zenodo); spec 001's raw-CSV-mirror estimate is ~30 GB, consistent.
- **Time structure:** all of 2022, partitioned by the mirror's own `WEEK-2022-NN` folders (not plain ISO calendar week at the year boundary — see "Week labeling" below).
- **Mirror trust:** this is a **third-party re-upload**, not the CESNET/Zenodo publication directly. Spec 001 makes `--verify` (per-day count vs. `stats-*.json`, class list, and a sampled DataZoo comparison) mandatory before any result depends on it. `--verify` has passed on the one real day exported so far (`scripts/export_raw_csv.py --verify`, 2026-09-20); the DataZoo field-level comparison is still skipped (no local HDF5 downloaded).

## Class list and support (real, one day only)

The full 180-app / 24-category vocabulary (spec 001) needs the whole year;
what's been measured for real, from `WEEK-2022-00/2022-01-01`:

| | value |
|---|---|
| apps seen | 179 (of 180 — a single day naturally omits the rarest few) |
| categories seen | 23 (of 24) |
| flows | 487,081 (all of them; zero dropped for zero-PPI) |

Per-class/per-category support counts for the full corpus are not available
yet — they depend on the weeks-11-52 export (T5b's "done when," still
pending a Kaggle-kernel run, see below) and on `scripts/make_unknown_split.py`
being run for real once that export exists (plan T6).

## Known issues

- **Week-10 exporter artefact** (spec 004): the dataset's own documentation flags weeks 1-9 and 11-52 as separate regimes; this project's splits (`configs/splits/d1_main.yaml`) start at week 11 for exactly this reason, not because of anything found independently here.
- **`WEEK-2022-NN` is not plain ISO calendar week** at the start of the year — see "Week labeling" below. A naive `isocalendar()`-based exporter would silently disagree with the mirror's own directory layout for a handful of dates every year.
- **`TIME_FIRST` (UTC) disagrees with the mirror's own (local-time) file grouping** for ~3.7% of any given day's rows, right at the day boundary — see "`TIME_FIRST` is UTC" below. This is a real, measured bug this project hit and fixed (plan T5b), not a hypothetical.
- **Some days are genuinely empty** (zero flows, zero-byte compressed file) — see "A day can be entirely empty" below.
- **Third-party mirror** (see "Mirror trust" above): every result that depends on this data needs `--verify` to have passed for the specific days/weeks it uses.

## Privacy notes

Per spec 001: D1 is already anonymised by CESNET (source IPs anonymised,
no payload). This project additionally never reads `SRC_IP`, `DST_IP`,
`DST_ASN`, `DST_PORT`, `TLS_SNI`, or `TLS_JA3` from the CSV at all (`usecols`
excludes them at parse time, not a later filter — see "Identifier columns"
below), so none of them ever enter this process's memory, let alone a shard.

## Decisions this project made

- **Path B over Path A** for the drift study (spec 001): weekly/daily granularity is what the drift curves need, and it avoids a 30 GB download plus a ~4 GB shard upload. Path A stays the verification fallback.
- **Splits start at week 11**, per the dataset's own documented week-10 exporter change (spec 004), not a project-specific choice.
- **Period labels follow the mirror's own `WEEK-2022-NN` folder names** exactly (`cesnet_csv.iso_week`), not a recomputed ISO week, because the two disagree at the year boundary (see below) and the mirror's own layout is the ground truth for cross-referencing.
- **A shard's period is decided per source file, not per row** (`cesnet_csv.date_from_filename`) — see "`TIME_FIRST` is UTC" below for why the per-row alternative is actually wrong, not just a stylistic choice.

## Schema record (plan T5a)

Recorded from a real file, not the DataZoo docs: `pranjalkar99/cesnet-22`,
`CESNET-TLS-Year22/WEEK-2022-00/2022-01-01/flows-20220101.csv.xz` (26.9 MB
compressed, 487,081 rows) plus its `stats-20220101.json` and the week-level
`stats-week.json`, pulled 2026-09-20 via
`kaggle datasets download -d pranjalkar99/cesnet-22 -f <path> -p data/raw/_probe`.
Every claim below was read out of these three files, not assumed from spec
001 or the DataZoo package.

### Header (45 columns, in this exact order)

```
ID, SRC_IP, DST_IP, DST_ASN, DST_PORT, PROTOCOL, TLS_SNI, TLS_JA3,
TIME_FIRST, TIME_LAST, DURATION, BYTES, BYTES_REV, PACKETS, PACKETS_REV,
PPI_LEN, PPI_DURATION, PPI_ROUNDTRIPS, APP, CATEGORY,
FLAG_CWR, FLAG_CWR_REV, FLAG_ECE, FLAG_ECE_REV, FLAG_URG, FLAG_URG_REV,
FLAG_ACK, FLAG_ACK_REV, FLAG_PSH, FLAG_PSH_REV, FLAG_RST, FLAG_RST_REV,
FLAG_SYN, FLAG_SYN_REV, FLAG_FIN, FLAG_FIN_REV,
FLOW_ENDREASON_IDLE, FLOW_ENDREASON_ACTIVE, FLOW_ENDREASON_END, FLOW_ENDREASON_OTHER,
PPI, PHIST_SRC_SIZES, PHIST_DST_SIZES, PHIST_SRC_IPT, PHIST_DST_IPT
```

Plain CSV, comma-delimited, one header row. `PROTOCOL` is `6` (TCP) for
every row sampled (this is the TLS dataset; QUIC is the separate D2
mirror) — so this file carries no UDP rows to cross-check `udp_idle_timeout`
against; that stays untested until D2's schema is probed the same way.

### The `PPI` column — encoding, nesting, channel order

One field, a Python-literal string of 4 same-length lists:

```
PPI = "[[<ipt_ms>, ...], [<dir>, ...], [<size_bytes>, ...], [<push_flag>, ...]]"
```

Example (`PPI_LEN=21`):

```
[[0, 1, 18, 7, 0, 15, ...],       # IPT, ms, int, first value always 0
 [1, -1, 1, 1, -1, 1, ...],       # DIR, +1 / -1
 [517, 146, 51, 139, 77, ...],    # SIZE, bytes, L4 payload
 [1, 1, 1, 1, 1, 1, ...]]         # PUSH flag, 0/1
```

Parse with `json.loads`, not `ast.literal_eval`: despite looking like a
Python list literal, every value in it (ints, `+1`/`-1`, nested brackets)
is also valid JSON, and `json.loads` was measured at ~17x faster on 50,000
real cells from this file (1.3 s vs 22.1 s) with zero parse failures —
worth confirming rather than assuming, since PPI parsing is the
per-row-unavoidable cost in this exporter (the one thing that can't be
skipped by dropping columns) and 17x there is the difference between an
overnight run and a multi-day one across the full 42-week corpus.

**This is already `adl_etc.data.ppi`'s exact channel order and sign
convention** (`IPT_POS, DIR_POS, SIZE_POS, PUSH_POS = 0, 1, 2, 3`,
`DIR_FWD = +1`, `DIR_REV = -1`) — no reordering or sign flip needed in the
parser, only clipping through the existing `clip_ipt`/`clip_size`
(confirmed necessary: sampled IPT values up to 300,000 ms were seen,
which exceeds `IPT_MAX_MS = 32_767` and will clip; sampled sizes topped
out at 1,460 bytes, under `SIZE_MAX = 1_500`, but that's a 60k-row sample
on one day, not a bound).

**No trailing padding inside the CSV.** Every one of the 4 inner lists has
exactly `PPI_LEN` elements — the parser must not assume a fixed-30 array
on read; it pads to `K_MAX` itself when writing into `ppi.py`'s
`(30, 4)` int16 layout, same as `flows.py`'s `FlowRecord` already does.
`PPI_LEN` was observed in `[3, 30]` across the full day (min 3 — CESNET's
own short-flow floor; max 30 — already capped at our `K_MAX`, so no flow
in this mirror carries more than 30 packets worth of PPI to begin with).

**`PHIST_SRC_SIZES`, `PHIST_DST_SIZES`, `PHIST_SRC_IPT`, `PHIST_DST_IPT`**
are CESNET's own precomputed 8-bin packet histograms, same nested-list
style (`"[0, 1, 2, 0, 4, 1, 1, 0]"`). **Not used.** Project rule ("one
definition, one place") says our flow-statistics histograms come from
`ppi.py`'s own `SIZE_HIST_EDGES`/`IPT_HIST_EDGES` computed over the parsed
`PPI` column, exactly as `FlowRecord.flowstats()` already does for D3/D4 —
mixing in CESNET's differently-binned histograms would silently produce a
second, incompatible definition of the same feature.

### Label columns and vocabulary

- **`APP`** — the class label. 179 distinct values in this one day (spec
  001's "180 apps" is the full-year vocabulary; a single day naturally
  omits the rarest few). Values are lowercase-hyphenated slugs
  (`"the-weather-channel"`, `"microsoft-diagnostic"`, `"doh"`, ...).
- **`CATEGORY`** — 23 distinct values this day (spec says 24 over the
  year), same slug style but human-readable
  (`"Weather"`, `"Instant messaging"`, `"Software updates"`, ...).
- Both are **only present in the CSV rows**, not in `stats-*.json` as a
  separate `categories` block — the stats files carry only
  `{"global": {"total-saved": N}, "apps": {<app>: <count>, ...}}`, keyed
  by `APP`, not `CATEGORY`. The class-list check in `--verify` (spec
  001) therefore has to compare against `stats-*.json`'s `apps` keys, and
  any per-category check has to be derived by aggregating rows' own
  `CATEGORY` values, not read off a shipped list.

### Row count vs. `stats-*.json`

Exact match, all three ways, for `2022-01-01`:

| | value |
|---|---|
| `stats-20220101.json` → `global.total-saved` | 487,081 |
| actual CSV data rows (header excluded) | 487,081 |
| `sum(stats-20220101.json["apps"].values())` | 487,081 |

`stats-week.json` (`WEEK-2022-00`, i.e. 2022-01-01 + 2022-01-02) reports
`total-saved: 1,075,104`, consistent with a two-day sum. This confirms
the `--verify` per-day count check (spec 001) is a straightforward
`len(csv_rows) == stats["global"]["total-saved"]`, and that `stats.apps`
can double as the ground-truth class list and per-class support count for
that file, with no separate reconciliation logic needed.

### Identifier columns to drop at parse time

Per spec 001's privacy rule (never store or use SNI/JA3/IP/ASN/port as
model inputs), dropped **before** anything is buffered, not filtered
later:

`SRC_IP`, `DST_IP`, `DST_ASN`, `DST_PORT`, `TLS_SNI`, `TLS_JA3`.

`ID` (a per-file row index, not a network identifier) and `PROTOCOL` are
kept — `ID` is discarded after use (not a network identifier, just a
row-order artifact) and `PROTOCOL` because the value's needed to route
TCP-vs-UDP flowstats treatment, same as `flows.py` already does for
PCAP-sourced flows.

`FLOW_ENDREASON_*` (four booleans, one-hot) map onto the same
`EndReason` vocabulary `flows.py` already produces (`IDLE`, `ACTIVE`,
`FIN`→`END` naming differs, `OTHER`→ no PCAP-side equivalent seen yet);
worth carrying into `flows.parquet`'s audit sidecar for D1 the same way
`end_reason` already is for D3/D4, not into the model-facing arrays.

### Timestamps

`TIME_FIRST`/`TIME_LAST` are ISO-8601-shaped but **naive** (no `Z` or UTC
offset, e.g. `2021-12-31T22:00:00`, microsecond precision on
`TIME_LAST`). Parsed with `pandas.to_datetime(..., utc=True)` on the
assumption the mirror's clock is already UTC (CESNET's own published
convention); this assumption is not independently confirmed by anything
in this file and should be treated as provisional until cross-checked
against the DataZoo HDF5's own timestamp column during `--verify`'s
field-level comparison. `DURATION` (seconds, float) is
`TIME_LAST - TIME_FIRST` and agrees with the two timestamps in the one
row checked by hand.

### Week labeling — not plain ISO calendar week

The mirror's own `WEEK-2022-NN` folder names were checked directly, not
assumed to be Python's `isocalendar()`. They agree for almost the whole
year, but not at the edges:

- **2022-01-01 and 2022-01-02** sit in `WEEK-2022-00/`. Their true ISO week
  is `(2021, 52)` (`pd.Timestamp("2022-01-01").isocalendar()` →
  `(2021, 52, 6)`) — the mirror folds this stub into year 2022's "week 00"
  instead of carrying it into a `WEEK-2021-52` that doesn't otherwise exist
  in this dataset. True ISO week 1 of 2022 (`WEEK-2022-01/`) starts the
  following Monday, confirmed to contain exactly 2022-01-03 .. 2022-01-09.
- **The end of the year does not mirror this.** `WEEK-2022-52/` was
  confirmed (via `kaggle datasets download -f`) to hold 2022-12-26 through
  2022-12-31 — a plain, unshifted ISO week 52 — and neither
  `WEEK-2022-53/` nor `WEEK-2023-00/` exists (both 404). 2022 happens not
  to need the symmetric case (a December date whose ISO week rolls forward
  into next year), so that case is **unverified** and must be re-checked
  the same way before this convention is reused for another year (spec
  001's optional CESNET-QUICEXT-25, for instance).

`cesnet_csv.iso_week()` implements exactly this (ISO week, with the
start-of-year stub folded into `WEEK-<calendar_year>-00`), not a naive
`isocalendar()` call — the naive version would have put 2022-01-01/02 in a
`WEEK-2021-52` shard set that disagrees with the mirror's own directory it
came from.

### A day can be entirely empty — not just zero matching rows

`stats-20221231.json` reports `{"global": {"total-saved": 0}, "apps": {}}`
— a real zero-flow day in the mirror, not a hypothetical. Its
`flows-20221231.csv.xz` is not a header-only CSV; decompressed, it is
**zero bytes**, which makes `pandas.read_csv` raise `EmptyDataError`
rather than returning zero rows. Every place this exporter streams a CSV
goes through `cesnet_csv._read_csv_chunks`, which catches exactly this and
treats it as zero rows, not a crash — found by downloading and decompressing
the file, not by guessing that "0 flows" would behave like an empty
dataframe.

### `TIME_FIRST` is UTC; the mirror's own file grouping is not

Found by running the exporter against this real file, not by inspection:
partitioning shards by each row's own `TIME_FIRST` (converted to its ISO
week) put 17,850 of the file's 487,081 rows (3.7%) into a `WEEK-2021-52`
shard set that has no counterpart anywhere in the real mirror. Checked
directly:

```
>>> pd.to_datetime(df["TIME_FIRST"]).dt.date.value_counts()
2022-01-01    469231
2021-12-31     17850
```

`flows-20220101.csv.xz` — named and located for calendar day 2022-01-01 —
contains rows whose `TIME_FIRST` falls on 2021-12-31 UTC: the first ~hour of
CET-local New Year's Day. The mirror groups files by local time; `TIME_FIRST`
is stored in UTC. The two disagree right at the edge of any day.

**Fix:** a shard's period is decided once per *file*, from the file's own
name (`cesnet_csv.date_from_filename`), never recomputed per row from
`TIME_FIRST`. Every row from one file lands in that file's one period,
matching the mirror's actual layout, and `TIME_FIRST` itself is left
untouched as the flow's true (UTC) start time for the `ts` array column
(spec 004's temporal-split ordering needs the real value, not the
file's nominal date). A regression test
(`test_period_is_the_files_own_date_not_each_rows_time_first`) reproduces
this exact scenario at 2-row scale.

This also turned out to be the bigger performance cost: moving the ISO week
computation from once-per-row to once-per-file was a **2.6x** speedup on
this same file (measured, not assumed) — 172.6s → 66.4s, 2,822 → 7,339
rows/s — because `pd.Timestamp(...).isocalendar()` on 487,081 individual
rows was expensive relative to the rest of the per-row cost (JSON-parsing
`PPI`, building a `FlowRecord`, computing `flowstats()`).

### What this fixes for T5b

- Parser reads only `cesnet_csv.USE_COLUMNS` (never the 6 identifier
  columns, via `usecols`), parses `PPI` via `json.loads` into
  `[ipt, dir, size, push]` — same order as `ppi.py`, no remapping — pads to
  `(K_MAX, 4)`, clips through `clip_ipt`/`clip_size`, and cross-checks the
  parsed length against the row's own `PPI_LEN` column.
- `APP` → label (via the shard's `label_map`, same pattern as
  `export_pcap.py`'s file-level labels). `CATEGORY` → category map, same
  pattern.
- A shard's period comes from its source file's own name
  (`cesnet_csv.date_from_filename` → `iso_week`), never from individual
  rows' `TIME_FIRST` — see the `TIME_FIRST` finding above.
- `--verify`'s per-day count check is exact-equality against
  `stats-*.json["global"]["total-saved"]`; its class-list check is
  exact-equality against `stats-*.json["apps"].keys()`, not a hand-written
  180-app list (spec 001 already flagged that list needs the official
  servicemap; `stats-*.json` supersedes needing that separately per file).
- No UDP rows exist in this file to validate `udp_idle_timeout`'s CESNET
  path — that stays open until D2 (CESNET-QUIC22) gets its own schema
  probe, which this project has not done yet.
