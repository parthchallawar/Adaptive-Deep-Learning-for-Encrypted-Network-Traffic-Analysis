# Plan: Phase 1, the data pipeline

- **Specs:** [001](../specs/001-datasets-and-acquisition.md), [002](../specs/002-pcap-flow-pipeline.md), [003](../specs/003-feature-representation-and-preprocessing.md), [004](../specs/004-splits-and-evaluation-protocol.md) (build steps 1 to 4)
- **Status:** in-progress (9 of 14 items done)
- **Exit gate:** phase 2 (specs 014, 015, 005) cannot start until [Exit criteria](#exit-criteria) are all green.

## Approach

Build from the bottom of the dependency graph. Every model, baseline, policy and service call in every later phase reads the same arrays, so a defect here is a defect in every result the project will ever report. Nothing in this phase needs a GPU, so the whole thing is done on the laptop before a single Kaggle hour is spent.

Three rules hold throughout:

1. **Correctness is decided by hand-written expectations, not by eyeballing real traffic.** Every transform gets a fixture whose output was computed by hand or by an independent method.
2. **One definition, one place.** Bin edges, column orders, clipping rules and the shard schema exist once, in `src/adl_etc/data/`, and everything else imports them. A constant that appears twice will diverge.
3. **Provenance travels with the data.** Every shard set carries the exporter's git commit, the source manifest hash and the normalisation-stats hash. A result that cannot name the bytes it came from is not a result.

## Where phase 1 stands

Done and under test (19 tests, `ruff` clean):

- [x] **Environment.** `pyproject.toml`, phase-scoped extras, `.venv` on Python 3.12, pytest/ruff/mypy. Torch is deliberately *not* a dependency of this phase.
- [x] **Package layout.** `src/adl_etc/` src-layout with `data/`, `models/`, `training/`, `evaluation/`, `inference/`, `service/`, `dashboard/`, `utils/`.
- [x] **[`data/ppi.py`](../src/adl_etc/data/ppi.py).** PPI schema, value ranges, clipping, histogram bins, padding mask, flow-statistics column list.
- [x] **[`data/flows.py`](../src/adl_etc/data/flows.py).** Flow key, `FlowBuilder` with inactive/active/FIN-linger timeouts, PPI extraction, flow statistics.
- [x] **[`data/pcap_source.py`](../src/adl_etc/data/pcap_source.py).** dpkt backend for pcap and pcapng; Ethernet/VLAN/IPv4/IPv6/TCP/UDP; sizes derived from header length fields, not captured bytes.
- [x] **Golden tests.** [`tests/data/test_flows.py`](../tests/data/test_flows.py) covers the reference capture, padding, clipping, timeouts, truncation invariance and skipped-packet accounting.

- [x] **[`data/tensors.py`](../src/adl_etc/data/tensors.py) (T1).** `ShardWriter`/`ShardSet`, mmap-backed, sharded `.npy` + `meta.json` + `flows.parquet`. 11 tests.
- [x] **[`data/features.py`](../src/adl_etc/data/features.py) (T2).** Tokeniser, continuous view, prefix/mask, `Standardizer`, five augmentations, `StreamTensorizer`. 41 tests, including the stream/offline equivalence invariant (spec 020 #5) and frozen bin-edge drift detection.
- [x] **[`data/download.py`](../src/adl_etc/data/download.py), [`manifest.py`](../src/adl_etc/data/manifest.py), [`ustc_download.py`](../src/adl_etc/data/ustc_download.py), [`iscx_download.py`](../src/adl_etc/data/iscx_download.py), [`iscx_labels.py`](../src/adl_etc/data/iscx_labels.py) (T3).** Resumable/retrying downloader engine, dataset manifest, and both D3/D4 acquisition pipelines. USTC (D4) verified against the live GitHub API (`--dry-run` lists all 20 real classes, 387 MB); the bulk download itself has not been run yet, pending the note in this task's log. 63 new tests (full suite 223/223).

Remaining: tasks **T4 to T8** below.

## Scope decisions

Made once, here, so they are not relitigated mid-task.

| Decision | Choice | Reason |
|---|---|---|
| Live capture (Npcap/Scapy sniffer) | **deferred to phase 5** (spec 016) | It has no consumer until the inference service exists, and its only test is a manual smoke test. The *streaming interface* is built now (T2) so 016 is a driver, not a rewrite. |
| NFStream / ipfixprobe backends | **deferred, possibly permanently** | Both are Linux-only, the dev machine is Windows, and spec 002 states neither is needed to reach any success criterion. Parity is enforced against the documented PPI definition instead. |
| Shard container | **directory of `.npy` + `meta.json`**, not `.npz` | `.npz` cannot be memory-mapped. Kaggle has ~29 GB RAM against a ~4.3 GB shard set plus a model; `mmap_mode="r"` keeps that headroom. |
| D1 acquisition | **Path B (Kaggle mirror) in a CPU kernel**, Path A for verification | Weekly granularity is what the drift claim needs, and it removes a 30 GB download and a 4 GB upload entirely. Already recorded in spec 001. |
| ISCX volume | **measure first, subset if needed** | 28 GB through a pure-Python parser is unmeasured. T4 sets a concrete gate. |
| Parquet audit sidecar | **yes**, `flows.parquet` per export | Needed to debug an export without re-running it, and to answer "which capture file did this flow come from" during leakage checks. |

## Tasks

### T1 — `data/tensors.py`: shard writer and reader

**Why first:** every other task in this phase writes shards or reads them. It is the only task with no upstream dependency left.

**Files:** `src/adl_etc/data/tensors.py`, `src/adl_etc/utils/provenance.py`, `tests/data/test_tensors.py`

**On-disk layout**

```
data/processed/<dataset>/<period>/
  meta.json
  shard-00000/{ppi,ppi_len,flowstats,label,category,session_id,ts}.npy
  shard-00001/...
  flows.parquet          # audit sidecar, never a model input
```

**Array contract** (single source of truth, exported as `ARRAY_SPEC`):

| Array | dtype | per-flow shape |
|---|---|---|
| `ppi` | `int16` | `(30, 4)` |
| `ppi_len` | `int8` | `()` |
| `flowstats` | `float32` | `(FLOWSTATS_DIM,)` |
| `label` | `int16` | `()` — `-1` means unknown |
| `category` | `int8` | `()` |
| `session_id` | `int32` | `()` |
| `ts` | `int64` | `()` — flow start, unix ms |

**Interface**

```python
class ShardWriter:
    def __init__(self, root: Path, dataset: str, period: str, *,
                 label_map: dict[str, int], category_map: dict[str, int] | None = None,
                 max_flows: int = 500_000, allow_unknown: bool = False,
                 source_manifest_hash: str | None = None) -> None: ...
    def add(self, rec: FlowRecord, *, label: int, category: int = -1,
            session_id: int = 0) -> None: ...
    def add_batch(self, arrays: dict[str, np.ndarray]) -> None: ...   # vectorised path for T5
    def close(self) -> dict: ...                                       # returns meta, writes meta.json
    def __enter__/__exit__                                             # close on exception too

class ShardSet:
    @classmethod
    def open(cls, root: Path, *, mmap: bool = True) -> "ShardSet": ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> dict[str, np.ndarray]: ...
    def column(self, name: str) -> np.ndarray: ...   # concatenated view across shards
    def batches(self, size: int, *, shuffle: bool = False,
                rng: np.random.Generator | None = None) -> Iterator[dict]: ...
    meta: dict
```

**Implementation notes**

- The writer preallocates buffers of `max_flows` rows and flushes when full, so memory is bounded regardless of dataset size. `close()` truncates the final partial shard before writing.
- `column()` on an mmapped set returns a lazy concatenation; materialise with `np.concatenate` only when the caller asks. For the Kaggle path, the whole set is small enough to materialise once.
- `flows.parquet` holds `key_hash` (salted SHA-1 of the 5-tuple, **never** the tuple), `start_ts`, `end_reason`, `session_id`, `shard_index`, `row_index`. The salt is generated per export and written to `meta.json`, so flows can be joined within an export but not re-identified across exports.
- `utils/provenance.py`: `git_commit()` (via `git rev-parse HEAD`, falling back to `"unknown"` outside a checkout), `stable_hash(obj)` (sorted-key JSON to SHA-256), `now_iso()`.

**`meta.json` fields:** `schema_version`, `dataset`, `period`, `n_flows`, `n_shards`, `shard_sizes`, `label_map`, `category_map`, `ppi_columns`, `flowstats_columns`, `flowstats_source` (`"adl_etc"` or `"datazoo"`), `k_max`, `exporter_git_commit`, `source_manifest_hash`, `key_salt`, `created_at`, and a `counters` block (`dropped_zero_ppi`, `clipped_size`, `clipped_ipt`, `skipped_packets`).

**Edge cases**

- `ppi_len == 0`: dropped at write, counted in `counters.dropped_zero_ppi`. Such flows cannot be classified by definition.
- `allow_unknown=False` and a `label == -1` arrives: raise. Train shards must never contain unknown classes (spec 004 leakage rule), and this is the cheapest place to enforce it.
- Reopening an existing non-empty `period` directory: refuse unless `overwrite=True`. Half-overwritten shard sets are silent corruption.
- `flowstats_source` mismatch between two shard sets being loaded together: `ShardSet.open` raises. Our 46-column vector is not DataZoo's 43-column vector (see [Spec corrections](#spec-corrections-this-plan-requires)).

**Tests**

- Round trip: write 1 000 synthetic flows, reopen, assert every array is bit-identical and dtypes match `ARRAY_SPEC`.
- Sharding: `max_flows=100` over 250 flows gives 3 shards sized 100/100/50, and `column("ts")` returns all 250 in insertion order.
- mmap: `ShardSet.open(mmap=True)` returns `np.memmap`-backed arrays and does not load the file into RAM (`arr.base` check).
- Unknown guard: `add(label=-1)` with `allow_unknown=False` raises; with `True` it is stored.
- Zero-PPI flow is dropped and the counter increments.
- `meta.json` records the git commit and a stable `stats`-independent hash; two writes of the same data produce the same `n_flows` and label map.

**Done when:** all of the above pass, `ruff` and `mypy` clean, and `ShardWriter` can consume a `FlowRecord` from `flows.py` without an adapter.

---

### T2 — `data/features.py`: tokeniser, continuous view, prefixes, augmentations

**Why here:** it is the contract between the data and every model. Written now, it is tested against hand-computed values; written later, it gets tested against whatever the model happens to produce.

**Files:** `src/adl_etc/data/features.py`, `tests/data/test_features.py`

**Bin edges — frozen, not computed at import**

Log-spaced edges are generated once by a helper, then pasted into the module as literal arrays. A test asserts the literals still equal the generator's output. This costs nothing and makes it impossible for a NumPy version change to silently move every token in the project.

```python
SIZE_BIN_EDGES: np.ndarray   # 64 log-spaced bins over [1, 1500]
IPT_BIN_EDGES:  np.ndarray   # 32 log-spaced bins over [0, 65535] ms
```

**Token channel layout** — index `0` is the padding token in *every* channel:

| Channel | Values | Cardinality |
|---|---|---|
| `size_bin` | pad, then 64 bins | 65 |
| `ipt_bin` | pad, then 32 bins | 33 |
| `dir` | pad, `+1`, `-1` | 3 |
| `push` | pad, `0`, `1` | 3 |

Giving `push` its own pad index costs one embedding row and means a single `PAD_INDEX = 0` rule holds everywhere; a channel that is the sole exception is a bug waiting to happen. (Spec 003 currently says 2 values for `push`; see [Spec corrections](#spec-corrections-this-plan-requires).)

**Interface**

```python
def tokenize(ppi: np.ndarray, ppi_len: np.ndarray) -> np.ndarray:
    """[N,30,4] int16 raw PPI -> [N,30,4] int64 token indices. Padding positions are 0."""

def continuous(ppi: np.ndarray, ppi_len: np.ndarray) -> np.ndarray:
    """-> [N,30,4] float32: [log1p(size)*dir, log1p(ipt), dir, push]. Padding rows are 0."""

def prefix(arrays: dict, k: int) -> dict:
    """Zero positions > k and rebuild the mask. ppi_len is left unchanged."""

def padding_mask(ppi_len: np.ndarray, k: int = K_MAX) -> np.ndarray:
    """[N,30] bool, mask[i,j] = j < min(k, ppi_len[i])."""

class Standardizer:
    @classmethod
    def fit(cls, shards: ShardSet) -> "Standardizer": ...   # training periods only
    def transform_continuous(self, cont: np.ndarray) -> np.ndarray: ...
    def transform_flowstats(self, fs: np.ndarray) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...                 # stats.json + hash
    @classmethod
    def load(cls, path: Path) -> "Standardizer": ...
    hash: str
```

**Flow-statistics treatment:** `log1p` on count-like columns (bytes, packets, duration, ppi_duration), histograms renormalised to proportions per group, flags left as 0/1, then per-column standardisation. The column groups are derived from `FLOWSTATS_COLUMNS` by prefix, not hard-coded twice.

**Augmentations** (raw integer PPI in, raw integer PPI out, so they are backend-independent):

| Name | Rule | Invariant the test asserts |
|---|---|---|
| `drop(p)` | remove each packet w.p. `p`, shift left, **new IPT = sum of the removed gaps** | total PPI duration is preserved; packet order is preserved; `ipt[0] == 0` |
| `reorder(q)` | swap adjacent packets w.p. `q` | multiset of `(size, dir, push)` is unchanged |
| `ipt_jitter(s)` | `ipt *= exp(N(0, s))`, re-clip | result stays in `[0, 65535]`; `ipt[0] == 0` |
| `size_jitter(sd)` | `size += N(0, sd)`, clip to `[1, 1500]` | no size becomes 0 (that would fake a padding slot) |
| `crop()` | `K ~ U{2..30}` | equals `prefix(x, K)` exactly |

If `drop` removes the first packet, the new first packet's IPT becomes 0 and the flow's origin shifts — that is correct (it models an observer who joined late) and is asserted rather than left implicit.

**Streaming tensoriser** (built now, driven in phase 5):

```python
class StreamTensorizer:
    def reset(self) -> None: ...
    def push(self, ipt_ms: int, direction: int, size: int, push_flag: int) -> dict:
        """Append one packet, return the tokens/mask for the current prefix."""
```

**Tests**

- Frozen edges equal the generator's output; edges are strictly increasing and span the full range.
- Hand-computed tokenisation of the reference flow from `tests/conftest.py` (the same 6-packet PPI already golden-tested in `test_flows.py`).
- `prefix(x, k)` then `padding_mask` agree for every `k` in 1..30, including flows with `ppi_len < k`.
- Shortcut exclusion: the tensoriser's output dimension equals the documented feature count, and no field named in spec 003's exclusion list is reachable from a `Dataset` item.
- Each augmentation's invariant above, over 1 000 random flows and a fixed seed.
- `Standardizer` round trip: `fit` → `save` → `load` gives an identical hash and identical transformed output.
- **Stream/offline equivalence:** feeding the reference flow packet-by-packet through `StreamTensorizer` equals `tokenize` of the full flow at each `k`. (This is spec 020's invariant 5 and is cheap to satisfy now.)

**Done when:** all tests pass and `tokenize` runs 1M flows in under 5 s on the laptop (vectorised `searchsorted`, no Python loop over flows).

---

### T3 — Manifest and dataset downloaders (D3, D4)

**Why here:** the ISCX download is measured in hours, so it starts early and runs while T4 is being written against the fixture pcap.

**Files:** `src/adl_etc/data/manifest.py`, `scripts/download_iscx.py`, `scripts/download_ustc.py`, `scripts/download_all.py`, `data/manifest.json`

**Manifest schema** (`data/manifest.json`, one entry per downloaded artefact):

```json
{
  "iscx-vpn-2016": {
    "source_url": "...", "retrieved": "2026-09-19T...Z",
    "files": [{"name": "...", "bytes": 0, "sha256": "..."}],
    "extracted_to": "data/raw/iscx-vpn-2016/pcap",
    "exporter_git_commit": "..."
  }
}
```

`manifest.py` provides `register(name, **fields)`, `verify(name) -> bool` (re-hash on demand) and `require(name)` (raises with a message naming the download script). Shard `meta.json` stores the manifest entry's hash, so a shard set can always name its source bytes.

**Downloader behaviour (both scripts):** resumable HTTP with `Range`, retry with backoff on 5xx and connection resets, SHA-256 on completion, extraction into `data/raw/<dataset>/pcap/`, and a generated `labels.csv` mapping *file name* to class. `--files <glob>` limits the download to a subset (one capture per class) for development; `--dry-run` prints the plan and total bytes without fetching.

USTC file names differ between mirrors, so labelling goes through `labels.csv` built from a directory walk, never from hard-coded names.

**Real findings from building this (not assumptions):**

- **D3 is registration-gated, not direct HTTP.** Confirmed by fetching `cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/` directly: it serves a form collecting name/email/institution/job title/country, not a file listing. This project does not automate that submission — see the correction in spec 001 and `iscx_download.py`'s module docstring for the reasoning. `download_iscx.py` accepts `--base-url` (the post-registration listing) or `--files-from` (a local URL list) and discovers `*.pcap` links generically from whichever page it's given.
- **ISCX has no per-flow label column** (spec 001 already says "none usable" for time structure, and the same is true of any class label), so `iscx_labels.py` infers the 7x2 class from the file name using the dataset's documented category grouping. This is a tested but **unverified heuristic** — the real file list is behind the gate above — so every row carries `label_confidence="heuristic"` and an unrecognised name gets an empty `class_name` rather than a guess.
- **USTC's `.7z` archives are not uniform.** Probed two real archives while building `extract_7z`: `Shifu.7z` holds one top-level file named after the class (`Shifu.pcap`); `SMB.7z` holds a subfolder with two numbered files (`SMB/SMB-1.pcap`, `SMB/SMB-2.pcap`). `extract_7z` handles both shapes generically (collect every matching member, flatten into the destination) rather than assuming either one. `py7zr` was added as a core dependency for this — about half of USTC's 20 classes are `.7z`-only.
- **USTC verified live, not just against a fixture:** `python scripts/download_ustc.py --dry-run` lists all 20 real classes (10 benign, 10 malware) from the actual GitHub API, 387.1 MB compressed, matching spec 001's "10 benign + 10 malware" and roughly its "3.7 GB pcap" once decompressed.

**Tests**

- `tests/data/test_manifest.py`: register/verify/require, including a truncated-file case that size alone would miss but the hash catches. (Deviates slightly from the original plan text, which anchored this to `configs/data/*.yaml`; no such config files exist yet at this point in the project, so the tests exercise the manifest API directly instead.)
- `tests/data/test_download.py`, `test_ustc_download.py`, `test_iscx_download.py`: resume, retry-then-succeed, retry-exhaustion, non-retriable 4xx, GitHub-API listing, HTML link discovery, both 7z archive shapes, and the full `plan()`/`run()` pipeline for each dataset — all against the local `http_fixture` in `conftest.py`, not the network. Marked `integration`.
- The real bulk downloads are not run in CI or by these tests; `--dry-run` against the live GitHub API (above) is the closest thing to an end-to-end check that ran here.

**Done when:** `python scripts/download_all.py --datasets d3 d4 --dry-run` prints a correct plan (done — verified against live GitHub API for d4; d3 correctly reports the registration requirement instead of a plan, which is the correct behaviour until the user has registered), the real run completes (open — see progress log), `data/manifest.json` validates, and `labels.csv` covers every extracted pcap.

---

### T4 — `scripts/export_pcap.py`: the first end-to-end export

**Why here:** it joins three tested components (`pcap_source` → `flows` → `tensors`) into the project's first real artefact, and it is the only task that produces evidence about pure-Python parsing speed.

**Files:** `scripts/export_pcap.py`, `configs/data/pcap.yaml`, `tests/data/test_export_pcap.py`

```
python scripts/export_pcap.py --dataset iscx-vpn-2016 \
    --config configs/data/pcap.yaml --out data/processed/iscx-vpn-2016/all
```

- `session_id` = index of the source capture file. All flows from one file share a session, which is what makes spec 004's grouped split leakage-free for D3.
- Labels come from `labels.csv` at file level (both D3 and D4 label whole captures).
- Writes shards plus `flows.parquet`, and prints a summary: flows, dropped zero-PPI flows, skipped packets by reason, packets/s, wall time.
- `configs/data/pcap.yaml` holds `inactive_timeout`, `active_timeout`, `fin_linger`, `udp_idle_timeout`, `k_max`, `backend` — the values currently hard-coded as `FlowBuilder` defaults, so they become configurable without changing the defaults.

**Throughput gate:** measure on one ISCX capture. If the extrapolated full-set time exceeds **60 minutes**, switch D3 to a per-class subset and record the decision in the dataset card. Either outcome is acceptable; an unmeasured guess is not.

**Tests**

- End-to-end on `tests/conftest.py`'s reference pcap: the exported shard's `ppi` equals `EXPECTED_PPI` from `test_flows.py`, proving the export path does not alter what the flow builder produced.
- Two capture files produce two distinct `session_id`s and no shared flows.
- `flows.parquet` row count equals the shard row count, and `key_hash` is never a readable address.

**Done when:** D4 (3.7 GB) is fully exported, D3 is exported at whatever volume the throughput gate allows, and the reference-pcap test passes.

---

### T5 — CESNET Path B: schema discovery, then `scripts/export_raw_csv.py`

**Why the split into two steps:** the raw CSV column names, the PPI string encoding and the label column are only knowable by opening a real file. Writing the parser before looking is guesswork that fails silently on a 30 GB mirror.

**T5a — schema discovery (do this first, it takes minutes)**

Pull exactly one day plus its statistics file using the Kaggle API's single-file flag, so nothing near 30 GB is downloaded:

```
kaggle datasets download -d pranjalkar99/cesnet-22 \
    -f CESNET-TLS-Year22/WEEK-2022-11/<date>/flows-<date>.csv.xz -p data/raw/_probe
kaggle datasets download -d pranjalkar99/cesnet-22 -f <matching stats-*.json> -p data/raw/_probe
```

Record in `docs/datasets/cesnet-tls-year22.md`: the exact header, the PPI encoding (delimiter, nesting, channel order), the label column and its vocabulary, the row count versus `stats-*.json`, and which columns are identifiers to be dropped. **Every later decision in T5 cites this record.**

**T5b — the exporter**

**Files:** `scripts/export_raw_csv.py`, `src/adl_etc/data/cesnet_csv.py`, `tests/data/test_cesnet_csv.py`

- Streams `flows-YYYYMMDD.csv.xz` in chunks (`pandas.read_csv(chunksize=...)`, never a whole-file read), parses the PPI columns into the `[ipt, dir, size, push]` order defined in `ppi.py`, computes flow statistics, and writes shards **partitioned by ISO week** via `ShardWriter.add_batch`.
- Drops every identifier column (SNI, JA3, IPs, ASN, ports, destination prefix) at parse time, before anything is buffered. They are not dropped later; they never enter the process's data structures.
- Runs unchanged locally or inside a Kaggle CPU kernel with the mirror mounted read-only at `/kaggle/input/cesnet-22/`, writing to `/kaggle/working/` for saving as an output dataset (spec 015 mounts it for training).
- `--verify` implements spec 001's three mandatory checks: per-day flow counts against the shipped `stats-*.json`, the class list against the official 180-service servicemap, and — when a DataZoo HDF5 is present locally — a sampled field-level comparison over 10 000 flows on the overlapping fields.

**Mirror trust:** `--verify` must pass before any result depends on these shards. If it fails, fall back to Path A and record the failure in the dataset card. This is a third-party re-upload, not a CESNET publication.

**Tests**

- Parser tests against a small hand-written CSV fixture (10 rows) built from the header recorded in T5a, with one row's PPI computed by hand.
- Week partitioning: rows spanning a week boundary land in the correct `WEEK-2022-NN` shard sets.
- Identifier columns are absent from the output arrays *and* from `meta.json`.
- `--verify` fails loudly on a deliberately corrupted count.

**Done when:** weeks 11 to 52 of D1 exist as shard sets, `--verify` passes on every exported day, and the total is within the spec-003 budget (< 12 GB with D2).

---

### T6 — Splits and the metric suite

**Files:** `src/adl_etc/evaluation/metrics.py`, `src/adl_etc/evaluation/protocol.py`, `configs/splits/{d1_main,d2_quic,d3_grouped,d4_anomaly}.yaml`, `scripts/make_unknown_split.py`, `tests/evaluation/test_metrics.py`, `tests/evaluation/test_protocol.py`

**What phase 1 delivers.** Every metric that is a pure function of arrays — which is all of them except the ones needing telemetry that does not exist yet:

| Group | Functions | In phase 1? |
|---|---|---|
| Classification @K | `acc_at_k`, `macro_f1`, `balanced_accuracy`, `per_class_f1`, `confusion` | yes |
| Earliness | `auc_k`, `k95`, `harmonic_earliness` | yes |
| Policy | `mean_k`, `p95_k`, `coverage`, `committed_accuracy`, `pareto_front` | yes — pure array functions, and spec 009 lands faster for it |
| Open-set | `auroc`, `aupr`, `fpr_at_tpr`, `open_set_f1`, `rejection_earliness` | yes |
| Calibration | `ece`, `reliability_curve` | yes |
| Budget tracking, drift slope | — | **no**: they need controller windows (spec 011) and per-period reports (spec 012). Spec 004 already flags itself as scaffolding. |

**Contract:** metrics consume saved arrays, never models. The canonical input is `logits[N, K, C]` plus `labels[N]` and `ppi_len[N]`, so any policy can be re-scored later without re-running a model. This is what makes the phase-3 threshold sweeps cheap.

**`protocol.py`** loads a split YAML into concrete shard paths, asserts the leakage rules before returning, and assembles a `Report` (JSON + a plot manifest) under `results/<run_id>/`.

**Leakage assertions, enforced at split-load time and not merely tested:**

1. No `session_id` appears in two splits (this is what makes D3 honest).
2. No unknown label appears in a train split.
3. The `Standardizer` hash attached to a val/test split equals the train split's hash.
4. Train `ts` strictly precedes test `ts` for the temporal splits.

A violated assertion raises. A leakage rule that only exists in a test file is a rule that gets skipped on the day it matters.

**`scripts/make_unknown_split.py`** draws 150 known / 30 unknown classes with seed 42, stratified across categories to include both frequent and rare apps, redrawing any unknown with fewer than 100 test flows (spec 004), and writes the lists into the split YAML. A second draw with seed 43 gives the variance estimate. **Depends on T5** for the real class list.

**Tests**

- Hand-computed tiny examples: `acc@K` with padding, `auroc` with ties, `k95` monotonicity, `ece` on a 3-bin example, `pareto_front` dominance.
- Short-flow rule: evaluating at `K=10` a flow with `ppi_len=6` uses 6 packets and is counted in the separate short-flow bucket.
- Each leakage assertion fires on a deliberately corrupted split.
- Determinism: the same seed gives the same report hash.

**Done when:** every metric has a hand-computed test, and `protocol.load_split("configs/splits/d3_grouped.yaml")` returns real shard paths with all four assertions passing.

---

### T7 — Dataset cards

**Files:** `docs/datasets/{cesnet-tls-year22,cesnet-quic22,iscx-vpn-2016,ustc-tfc2016,self-captured}.md`

One card per dataset: source and exact retrieval command, license and citation, size, class list and support counts, time structure, known issues, privacy notes, and the decisions this project made about it (which weeks, which subset, why). The CESNET card additionally carries the T5a schema record and the `--verify` outcome.

Written as each dataset lands, not batched at the end — the details that matter are the ones noticed during the export.

**Done when:** cards exist for D1 to D4 and each names its manifest entry.

---

### T8 — Phase close-out

- Update spec 001 to 004 status lines to `implemented` (or `partially implemented` with the deferred parts named explicitly).
- Apply the [spec corrections](#spec-corrections-this-plan-requires) below.
- Update `specs/README.md` build-order `State` column for steps 1 to 4.
- Record measured numbers (parse throughput, shard sizes, flow counts per week) in the progress log, so phase 2's Kaggle budgeting starts from measurements rather than estimates.

## Order and parallelism

```
T1 tensors ──► T2 features ──► T6 metrics/protocol ──► T8
   │                               ▲
   ├──► T3 downloads ──► T4 export_pcap ──┤
   │        (runs in background)          │
   └──► T5a schema probe ──► T5b export_raw_csv ──┘
                                          │
                              T7 cards (written alongside T3–T5)
```

T1 and T2 are strictly sequential and block everything. T3's download should be kicked off as soon as T1 starts. T5a is minutes of work and should be done early even though T5b comes late — the header it records is what T5b, T6's class list and T7's card all depend on.

## Spec corrections this plan requires

Six inconsistencies (three anticipated while planning, three found while implementing T2) surfaced against the code. Each is corrected when its task lands, not before, so the spec always describes shipped behaviour. All six are now landed.

| # | Where | Problem | Correction | Landed with |
|---|---|---|---|---|
| 1 | 002, 003 | Both say `flowstats: float32[43]`, quoting DataZoo's count. Our `FLOWSTATS_COLUMNS` has **46** entries. The two vectors are not the same vector. | Write `[N, FLOWSTATS_DIM]` and add `flowstats_source` to `meta.json`. Keep the standing rule from `ppi.py`: never mix the two sources in one model until the published column order is confirmed. | T1 |
| 2 | 003 | "Continuous sequence" defines 4 channels; the Outputs section said `cont[30,3]`. | 4 channels: `[log1p(size)*dir, log1p(ipt), dir, push]`. Outputs line corrected. | T2 |
| 3 | 003 | `push` was given 2 values while every other channel reserves index 0 for padding, so a real packet with `push=0` was numerically identical to a padding slot. | 3 values (pad, 0, 1), so `PAD_INDEX = 0` holds in every channel without exception. | T2 |
| 4 | 002, 003 | **Real bug, not a documentation gap.** `IPT_MAX_MS = 65_535` (DataZoo's own bound) was being stored in `PPI_DTYPE = int16`, a *signed* type whose true range tops out at 32767. Any inter-packet gap >= 32768 ms raised `OverflowError` on assignment; found because a hand-written test used the documented maximum. Unreachable under the *default* 30 s inactive timeout, but the type did not enforce that, and a longer configured timeout (or the 300 s active timeout accumulating one large gap) could still hit it in production. | `IPT_MAX_MS = 32_767`. `PPI_DTYPE` stays a single signed int16 shared across all four channels (direction needs the sign bit), so our storable ceiling is genuinely half of DataZoo's unsigned column. Documented in `ppi.py`'s docstring and in specs 002/003. | T2 |
| 5 | features.py (new) | The first draft of `SIZE_BIN_EDGES`/`IPT_BIN_EDGES` included the clipping ceiling itself as the final edge. With `side="left"` digitization, a value exactly at the ceiling then falls into the *second-to-last* bin, not the last one — the top bin was permanently unreachable, silently wasting 1 of 64 (or 32) bins. Found by a hand-computed test asserting the maximum value lands in the last bin. `ppi.py`'s own `SIZE_HIST_EDGES`/`IPT_HIST_EDGES` already avoid this (their edges stop below the ceiling); the new code didn't follow the existing pattern. | Edges regenerated as strictly interior points (`geomspace(lo, hi, n_bins+1)[1:-1]`), matching the existing histogram-edge convention. | T2 |
| 6 | features.py (new) | `prefix()` passed a hardcoded `k=P.K_MAX` to `padding_mask()` instead of the actual prefix length `k`, so a prefix-5 view still reported all 30 positions as unmasked. Root cause was a second, deeper bug: `padding_mask(ppi_len, k)`'s `k` parameter controlled the *returned array's length* rather than a *cutoff* on a fixed-30 mask, so it could never be combined with the fixed-30 `ppi`/`tokens` arrays it's meant to accompany once actually prefixed. | `padding_mask` now always returns `[N, K_MAX]` with `k` as a cutoff, matching spec 003's formula (`mask[k] = k <= min(K, ppi_len)`) literally; `prefix()` now passes its own `k`. | T2 |

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **The Kaggle mirror is a third-party re-upload.** | Every D1 result rests on it. | `--verify` is mandatory before any result depends on it (T5b). Path A is the documented fallback. |
| **Our 46 flow statistics are not DataZoo's 43.** | A flow-statistics baseline trained on one source and tested on the other would be meaningless. | `flowstats_source` in `meta.json`; `ShardSet.open` refuses to mix. Already tracked in `ppi.py`. |
| **Pure-Python parse speed on 28 GB of ISCX is unmeasured.** | T4 could stall for hours. | Explicit 60-minute gate on an extrapolated measurement; a per-class subset is a pre-approved outcome. |
| **The CESNET CSV schema is unknown until T5a.** | T5b cannot be written correctly without it. | T5a is scheduled early and costs one file download. No parser code is written before the header is recorded. |
| **Week-10 exporter artefact in D1.** | Silent distribution shift that is not the drift we mean to study. | Splits start at week 11 (spec 004); the card documents it; the split YAML excludes weeks 1 to 10 explicitly rather than by convention. |
| **Bin edges drifting with a NumPy version.** | Every token in the project would move, invalidating saved checkpoints. | Edges frozen as literals with a test against the generator (T2). |
| **`searchsorted` doesn't vectorise as well as it looks.** | `tokenize()` on 1M flows took 7.8 s with a straightforward `np.searchsorted` call — plausible-looking code that would have made every later epoch slower than it needed to be. | Both binned columns are bounded integers, so a precomputed lookup table (gather instead of per-element binary search) cut it to 3.1 s. Measured, not assumed: see the T2 progress log entry. |
| **D3's class labels are a heuristic, not ground truth.** | Registration-gated, so the real file list can't be checked yet (see T3). A wrong app/activity mapping would silently mislabel a chunk of the D3 category-level transfer experiment, whose results spec 004 already calls "secondary" but which still needs to be *correctly* secondary, not silently wrong. | `label_confidence="heuristic"` on every row; unresolved names get an empty class rather than a guess; the downloader prints every unresolved name; re-verify against real files once registered, before D3 results are reported. |

## Exit criteria

Phase 2 starts when all of these hold:

1. `pytest` green on the full phase-1 suite; `ruff` and `mypy` clean.
2. D1 shard sets exist for weeks 11 to 52 with `--verify` passing on every exported day.
3. D3 and D4 shard sets exist, with per-file `session_id`s and audit parquet.
4. `data/manifest.json` validates for D1 to D4.
5. A `Standardizer` fitted on the D1 train weeks is saved with its hash, and a val split loads with all four leakage assertions passing.
6. Dataset cards exist for D1 to D4.
7. Measured numbers recorded in the progress log: flows per week, shard sizes on disk, parse throughput, tokenise throughput.

What phase 2 inherits: a `ShardSet` it can mmap, a `Standardizer` it must not refit, a metric suite that takes `logits[N,K,C]`, and split YAMLs that refuse to leak.

## Progress log

- **2026-09-19.** Environment, package layout and the flow/PPI layer done; 19 tests pass, ruff clean. Two findings while testing: inter-packet times must be rounded rather than truncated (truncation biased every value down by up to 1 ms, visible as a true 5 ms gap recording as 4), and a flow closed by FIN at end-of-capture was being reported as an EOF close. Both fixed in `ppi.py` and `flows.py`.
- **2026-09-19.** Plan rewritten from a checklist into tasks T1 to T8 with interfaces, edge cases and per-task done-criteria. Three spec inconsistencies found while doing so and scheduled for correction (flowstats 43 vs 46; continuous channels 3 vs 4; `push` padding index). Live capture and the optional Linux backends explicitly deferred with reasons.
- **2026-09-19.** T1 done: `ShardWriter`/`ShardSet` in `data/tensors.py`, plus `utils/provenance.py` (`git_commit`, `stable_hash`, `file_sha256`, `now_iso`). 11 new tests, all passing; full suite is 30/30, ruff and mypy clean. One Windows-specific finding fixed along the way: `ShardSet` cached `np.memmap` handles with no way to release them, so on Windows a reader left open blocks deleting or `overwrite=True`-ing its own directory later in the same process — a real problem on the dev machine, not just a test artefact. Added `ShardSet.close()` (also usable as a context manager) that closes the underlying mmap objects. `add_batch()` (the vectorised path T5 will use) has no per-row 5-tuple, so its parquet audit rows carry `key_hash=None`/`end_reason=None`; documented in the docstring rather than left implicit.
- **2026-09-19.** T2 done: `data/features.py` — `tokenize`, `continuous`, `padding_mask`, `prefix`, `Standardizer`, the five augmentations, `StreamTensorizer`. 41 new tests (full suite 160/160), ruff and mypy clean. Four real bugs found and fixed while writing the hand-computed tests, not test artefacts — see [Spec corrections](#spec-corrections-this-plan-requires) items 4 to 6 for the first three:
  - `IPT_MAX_MS=65535` overflowed the signed `int16` PPI array on assignment (max 32767); fixed to `32_767`.
  - Bin edges included the clipping ceiling as their own final value, making the top bin of both the size and IPT tokenisers permanently unreachable under `side="left"` digitization; regenerated as strictly-interior edges, matching `ppi.py`'s existing histogram-edge convention.
  - `padding_mask`'s `k` parameter controlled the returned array's *length* instead of acting as a *cutoff* on a fixed-30 mask, and `prefix()` additionally passed a hardcoded `k=P.K_MAX` regardless of the actual prefix — together these meant a prefix-truncated view reported no positions as masked at all. Both fixed; `padding_mask` now always returns `[N, K_MAX]`.
  - Performance: `tokenize()` on 1M synthetic flows measured 7.8 s with `np.searchsorted`; since both binned columns are bounded integers, replaced with a precomputed lookup table (O(1) gather), cutting it to 3.1 s — under the 5 s target this task set for itself.
  - The augmentation test for `drop()` originally asserted an approximate duration-conservation bound that was actually just wrong (dropping leading packets legitimately discards the leading gap, by the same convention that makes position 0's IPT always 0); replaced with an exact per-merge conservation check using uniquely-sized synthetic packets to unambiguously recover which positions survived.
- **2026-09-20.** T3 done: `data/download.py` (resumable HTTP, GitHub contents API, generic 7z extraction), `data/manifest.py`, `data/ustc_download.py`, `data/iscx_download.py`, `data/iscx_labels.py`, and the three `scripts/download_*.py` CLIs. 63 new tests (full suite 223/223), ruff and mypy clean. `py7zr` added as a core dependency (justified in the risks table and in T3's findings above, not decorative). Three things researched and confirmed empirically rather than assumed, each changing the design:
  - D3 (ISCX) turned out to be registration-gated (`insert.php` form collecting personal details), not the "direct HTTP from cicresearch.ca" spec 001 originally said — corrected in spec 001, with the reasoning for not automating the registration recorded there and in `iscx_download.py`'s docstring.
  - USTC's `.7z` archives are not internally uniform (`Shifu.7z`: one top-level file; `SMB.7z`: a subfolder with two numbered files) — probed two real archives before writing `extract_7z`, which handles both shapes rather than assuming one.
  - `download_ustc.py --dry-run` was run against the live GitHub API (not just the test fixture) and correctly listed all 20 real classes, 387.1 MB compressed — the closest thing to an end-to-end check available without spending the bandwidth on the full download.

  One open item carried forward: the bulk USTC/ISCX downloads themselves have not been run (D4 is ~387 MB compressed / ~3.7 GB extracted; D3 additionally needs the user's one-time registration first). `data/manifest.json` does not exist yet as a result — it is created by `register()` on first real run, not pre-seeded.
- **2026-09-20.** D4's real download run: 24 pcap files (20 classes — SMB and Weibo's archives each held several numbered files, exactly the multi-file shape `extract_7z` was built to handle), 3.8 GB on disk, matching spec 001's "3.7 GB pcap". `data/manifest.json` created; `M.verify("ustc-tfc2016")` is `True` (every file's hash matches what was recorded at download time). Full suite still 223/223 afterward. D3 remains blocked on the user's one-time registration (see above); D4's exit-criterion item is done.
