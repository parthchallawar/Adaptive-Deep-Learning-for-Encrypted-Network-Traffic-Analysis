# Plan: Phase 1, the data pipeline

- **Specs:** [001](../specs/001-datasets-and-acquisition.md), [002](../specs/002-pcap-flow-pipeline.md), [003](../specs/003-feature-representation-and-preprocessing.md), [004](../specs/004-splits-and-evaluation-protocol.md) (build steps 1 to 4)
- **Status:** done (14 of 14 items) — see [Exit criteria](#exit-criteria): one remains genuinely blocked outside this repo (a Kaggle-kernel run for D1's full corpus); D3's registration gate was cleared 2026-09-21 and real D3 data now exists, partially
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

- [x] **[`data/export_pcap.py`](../src/adl_etc/data/export_pcap.py), [`configs/data/pcap.yaml`](../configs/data/pcap.yaml) (T4).** Joins `pcap_source` → `flows` → `tensors`. D4 fully exported for real: 403,394 flows, 24 files, 187 MB on disk, 4.8 minutes. D3 blocked on registration (spec 001); throughput measured on real D4 captures instead and used to make the subset-gate call anyway. Found and fixed a real performance bug along the way (`FlowBuilder._expire()`'s per-packet O(open flows) scan, 34% of wall time) — see the plan's progress log. 14 new tests (full suite 237/237).
- [x] **[`docs/datasets/cesnet-tls-year22.md`](../docs/datasets/cesnet-tls-year22.md) (T5a), [`data/cesnet_csv.py`](../src/adl_etc/data/cesnet_csv.py), [`scripts/export_raw_csv.py`](../scripts/export_raw_csv.py) (T5b).** CESNET raw-CSV (Path B) exporter. Schema, PPI encoding and label columns recorded from a real downloaded day before any parser code was written. Reuses `FlowRecord.flowstats()` rather than reimplementing it. Two real bugs found by running against the real file (not the fixture) — see the plan's progress log: `json.loads` vs `ast.literal_eval` (17x), and a genuine correctness bug where per-row `TIME_FIRST`-based partitioning scattered 3.7% of one real file's rows into a shard set (`WEEK-2021-52`) that doesn't exist in the mirror, fixed by partitioning per file instead (also a 2.6x speedup). 26 new tests (full suite 263/263).
- [x] **[`evaluation/metrics.py`](../src/adl_etc/evaluation/metrics.py), [`evaluation/protocol.py`](../src/adl_etc/evaluation/protocol.py), [`evaluation/unknown_split.py`](../src/adl_etc/evaluation/unknown_split.py), [`configs/splits/`](../configs/splits/), [`scripts/make_unknown_split.py`](../scripts/make_unknown_split.py) (T6).** Every phase-1 metric (classification, earliness, policy, open-set, calibration — pure NumPy/pandas, no scikit-learn, since that stays a phase-2 extra), `protocol.load_split` with all four leakage assertions enforced at load time, and the stratified known/unknown class draw. Two real bugs found and fixed — see the plan's progress log. 41 new tests, full suite 304/304.
- [x] **[`docs/datasets/{cesnet-tls-year22,cesnet-quic22,iscx-vpn-2016,ustc-tfc2016,self-captured}.md`](../docs/datasets/) (T7).** One card per dataset — retrieval command, real license and citation (researched, not assumed: CC BY 4.0 for D1/D2 confirmed on their own Zenodo records; D3's "no formal license, citation required" confirmed by fetching the dataset page directly; D4's mirror-repo `MPL-2.0` tag flagged as covering the mirror, not the underlying CTU/paper data), class list and support counts where real data exists (D4's full 20-class table; D1's one real day), known issues, privacy notes, and the decisions made. D2's card is explicit that nothing in it has been checked against real bytes, unlike every other card.
- [x] **Spec close-out (T8).** Specs 001-004 status lines updated to name exactly what's implemented and what's deferred and why (not blanket "implemented"); `specs/README.md`'s build-order `State` column updated for steps 1-4; the six spec corrections were already landed incrementally (T1-T2, see the table below) — none were outstanding at this step.

All tasks **T1 to T8** are done. Remaining risk lives in [Exit criteria](#exit-criteria) below, not in this checklist.

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

**Measured (not estimated), plan T4, 2026-09-20:** no ISCX capture exists yet (spec 001's registration gate), so the measurement ran on real D4 captures instead — the same `dpkt` backend, same `FlowBuilder`, real production-scale files, which is at least as representative as a hypothetical ISCX number would have been. First measurement (94 MB `Neris.pcap`): 79.5 s, 1.2 MB/s, 6,248 pkt/s. Profiling that run found `FlowBuilder._expire()` — an O(open flows) scan run on *every single packet* — consuming 34% of total wall time, ahead of all `dpkt` parsing combined. Fixed: the per-packet check now only re-examines the one flow the current packet actually belongs to (still exact, still checked on every packet); the sweep over every *other* open flow is throttled to `min(all four timeouts)/4` of capture time instead of running unconditionally. Re-measured after the fix: 54.4 s, 1.7 MB/s, 9,132 pkt/s — a 46% reduction, same output (identical flow/drop counts, confirming the fix changed nothing but speed). Extrapolated to ISCX's 28 GB: **~281 minutes, still far past the 60-minute gate.** D3 will be exported as a per-class subset once unblocked (spec 001 updated with this result, overriding its earlier "export all pcaps, disk allows it" default, which reasoned about disk space but never about time).

A subtlety the throttling had to get right, and a regression test locks in: a packet reusing the exact same 5-tuple as a flow that has *individually* timed out (but that the throttled sweep hasn't reached yet, because other flows' traffic has kept the sweep fresh) must still start a new flow, never get silently merged into the stale one. `test_key_reuse_is_caught_even_when_sweep_is_throttled` constructs exactly that interleaving and was confirmed to fail (0 expired instead of 1) when the per-packet check is removed, via a deliberate mutation — not just reasoned about, checked.

**Tests**

- End-to-end on `tests/conftest.py`'s reference pcap: the exported shard's `ppi` equals `EXPECTED_PPI` from `test_flows.py`, proving the export path does not alter what the flow builder produced.
- Two capture files produce two distinct `session_id`s and no shared flows.
- `flows.parquet` row count equals the shard row count, and `key_hash` is never a readable address.
- Unlabeled files (absent from `labels.csv`, or present with an empty `class_name` — the ISCX heuristic-label case) are skipped and counted, never silently assigned a fake label.
- `PcapConfig.load()` round-trips the real `configs/data/pcap.yaml`; an unknown config key raises rather than being silently ignored.
- `FlowBuilder`'s throttled-sweep correctness (above), plus the pre-existing UDP-vs-TCP idle timeout tests, live in `test_flows.py` since they're properties of `FlowBuilder` itself, not of the export path.

**Done when:** D4 (3.7 GB) is fully exported (**done** — 403,394 flows, 187 MB of shards, 4.8 minutes; see the plan's progress log), D3 is exported at whatever volume the throughput gate allows (**blocked** on spec 001's registration gate — code is ready, nothing left to build), and the reference-pcap test passes (**done** — 14 new tests, 11 in `test_export_pcap.py` and 3 in `test_flows.py` for the UDP timeout and throttled-sweep work, all passing).

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

- Streams `flows-YYYYMMDD.csv.xz` in chunks (`pandas.read_csv(chunksize=...)`, never a whole-file read), parses the PPI columns into the `[ipt, dir, size, push]` order defined in `ppi.py`, computes flow statistics, and writes shards **partitioned by ISO week, keyed off each file's own name** via `ShardWriter.add_batch` — not off individual rows' `TIME_FIRST` (see the progress log: `TIME_FIRST` is UTC, the mirror's own file grouping isn't, and partitioning per row put 3.7% of one real file's rows in a shard set the mirror doesn't have).
- Drops every identifier column (SNI, JA3, IPs, ASN, ports, destination prefix) at parse time, before anything is buffered. They are not dropped later; they never enter the process's data structures.
- Runs unchanged locally or inside a Kaggle CPU kernel with the mirror mounted read-only at `/kaggle/input/cesnet-22/`, writing to `/kaggle/working/` for saving as an output dataset (spec 015 mounts it for training).
- `--verify` implements spec 001's three mandatory checks: per-day flow counts against the shipped `stats-*.json`, the class list against the official 180-service servicemap, and — when a DataZoo HDF5 is present locally — a sampled field-level comparison over 10 000 flows on the overlapping fields.

**Mirror trust:** `--verify` must pass before any result depends on these shards. If it fails, fall back to Path A and record the failure in the dataset card. This is a third-party re-upload, not a CESNET publication.

**Tests**

- Parser tests against a small hand-written CSV fixture (10 rows) built from the header recorded in T5a, with one row's PPI computed by hand.
- Week partitioning: files near a week boundary land in the correct `WEEK-2022-NN` shard sets, keyed off each file's own name.
- A regression test for the real `TIME_FIRST`-vs-file-date bug above, at fixture scale.
- Identifier columns are absent from the output arrays *and* from `meta.json`.
- `--verify` fails loudly on a deliberately corrupted count.

**Done when:** weeks 11 to 52 of D1 exist as shard sets, `--verify` passes on every exported day, and the total is within the spec-003 budget (< 12 GB with D2). **Code done and verified against one real day** (487,081 real flows, `--verify` OK, spot-checked — see progress log); the full weeks-11-52 corpus run itself is not done locally — per this plan's own scope decision (D1 acquisition: Path B in a Kaggle CPU kernel, not a 30 GB local download), it belongs in a Kaggle kernel, not this laptop, the same way D3's bulk download is code-complete but blocked on a step outside this repo.

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

**Done when:** every metric has a hand-computed test, and `protocol.load_split("configs/splits/d3_grouped.yaml")` returns real shard paths with all four assertions passing. **Adapted like T4's throughput gate:** D3 has no real data yet (registration-gated, spec 001), so `d3_grouped.yaml`'s session_ids-filtering mechanism is instead proven against `configs/splits/d4_anomaly.yaml`, which **does** have real data (`data/processed/ustc-tfc2016/all`, 403,394 real flows, plan T4) — `protocol.load_split("configs/splits/d4_anomaly.yaml")` runs for real and is asserted against the exact real category counts (282,412 benign / 120,982 malware). `d3_grouped.yaml` itself is code-complete and tested against a synthetic fixture, same as D3's downloader.

---

### T7 — Dataset cards

**Files:** `docs/datasets/{cesnet-tls-year22,cesnet-quic22,iscx-vpn-2016,ustc-tfc2016,self-captured}.md`

One card per dataset: source and exact retrieval command, license and citation, size, class list and support counts, time structure, known issues, privacy notes, and the decisions this project made about it (which weeks, which subset, why). The CESNET card additionally carries the T5a schema record and the `--verify` outcome.

Written as each dataset lands, not batched at the end — the details that matter are the ones noticed during the export.

**Done when:** cards exist for D1 to D4 and each names its manifest entry. **Done** (plus a fifth card for D5, per the Files list above) — D1's and D4's cards name their real manifest entries (`cesnet-tls-year22-probe`, `ustc-tfc2016`); D2's and D3's cards have no manifest entry to name yet and say so explicitly rather than fabricating one.

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

Phase 2 starts when all of these hold. Status as of 2026-09-20, honestly
per-item rather than a blanket "phase 1 done": every task **T1 to T8** is
complete, but two criteria below depend on real data this plan always said
would come from outside this repo (a Kaggle kernel, and the user's own
ISCX registration), and neither of those steps has happened yet. Declaring
those criteria green without the data behind them would be exactly the
kind of unmeasured claim this plan's own first rule (correctness decided
by hand-written expectations, not eyeballing) exists to prevent.

**Updated 2026-09-21** (see the progress log entry below): the user
completed ISCX registration and supplied 2 of 5 real archives. Criterion 3
moves from "blocked" to "partially met, real data"; item 7 gains a real
D3 throughput number that reverses the D4-proxy estimate spec 001 recorded
on 2026-09-20.

1. **Met.** `pytest` green on the full phase-1 suite (304/304 at close-out, 310/310 after the 2026-09-21 D3 work below); `ruff` and `mypy` clean.
2. **Blocked, not met.** D1 shard sets exist for `WEEK-2022-00` only (one real day, plan T5). Weeks 11 to 52 need the full raw-CSV corpus exported, which this plan's own scope decision puts in a Kaggle CPU kernel (D1 acquisition: Path B, spec 001), not this laptop — at the measured 7,339 rows/s that's a multi-hour job. `--verify` passes on the one day that has been exported.
3. **Partially met, real data (updated 2026-09-21).** D4 shard sets exist for real (`data/processed/ustc-tfc2016/all`, 403,394 flows, per-file `session_id` 0-23, audit parquet written). D3 shard sets now also exist for real (`data/processed/iscx-vpn-2016/all`, 130,303 flows across 37 files/session_ids, 8 of 14 classes) — 2 of the 5 real archives have been downloaded and exported; the remaining 3 have not, so D3's real coverage is partial, not complete.
4. **Met for what's downloaded.** `data/manifest.json` validates (`M.verify(...)` is `True`) for `ustc-tfc2016`, the one real `cesnet-tls-year22-probe` file, and now `iscx-vpn-2016` (37 files). There is nothing to validate yet for D2 (not acquired).
5. **Blocked, not met.** No `Standardizer` has been fit for real — `Standardizer.fit` (spec 003, plan T2) needs a real D1 train-period `ShardSet`, and the only real D1 data on disk is one day, not the train weeks (11-26) spec 004 defines. `protocol.load_split`'s standardizer-hash assertion is tested (real round trip, `tests/evaluation/test_protocol.py`) but has not been exercised against a real fitted D1 Standardizer.
6. **Met.** Cards exist for D1 to D5 (plan T7); each names its real manifest entry where one exists, and says plainly where one doesn't. D3's card was rewritten 2026-09-21 with real numbers.
7. **Met for what's measured**, recorded through this progress log rather than gathered specially for this line: D4's real export (403,394 flows, 187 MB, 4.8 min, 18,875 pkt/s — T4); D1's real single-day export (487,081 flows, 7,339 rows/s after the per-file partitioning fix — T5); `tokenize()` on 1M synthetic flows, 3.1 s (T2); **D3's real export (2026-09-21): 130,303 flows, 1,786.8 MB in 64.6 s, 27.7 MB/s — 16x the 1.7 MB/s D4-proxy estimate spec 001 recorded the day before**, reversing that estimate's "full corpus needs 281 minutes" conclusion (see spec 001's Open questions and the progress log entry below). Flows-per-week for D1 and shard sizes for the full D1/D3 corpora are still not measured, since neither corpus is fully downloaded yet (criterion 2, and D3's remaining 3 archives).

**Net: 3 of 7 fully met, 3 partially met, 1 blocked outside this repo.**
The blocker that's fully outside this repo's control is now just one: the
D1 weeks-11-52 Kaggle-kernel run (which criterion 5's Standardizer also
depends on). D3's remaining blocker is now a bandwidth/time tradeoff for
the user (download 3 more archives, ~17 more minutes to export once
downloaded), not a registration gate or a throughput problem — both of
those were resolved on 2026-09-21. Phase 2's own early work (specs
014/015: experiment tracking, Kaggle pipeline) does not itself need D1's
full corpus to start — it needs a `ShardSet` it can mmap, which D4 and now
D3 both provide for real. Treat phase 2 as unblocked to *start* on that
basis, but not as fully exited from phase 1 until criteria 2, 3 and 5 are
re-checked for real.

What phase 2 inherits: a `ShardSet` it can mmap, a `Standardizer` it must not refit (once one exists), a metric suite that takes `logits[N,K,C]`, and split YAMLs that refuse to leak.

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
- **2026-09-20.** T4 done: `data/export_pcap.py` (`PcapConfig`, `export_dataset`, `ExportSummary`) plus `configs/data/pcap.yaml`. Also added `udp_idle_timeout` to `FlowBuilder`, a real gap (not just a doc gap): spec 002 always documented UDP/QUIC as needing its own idle timeout, but the constructor only ever accepted one timeout applied to every protocol. 14 new tests, full suite 237/237.

  The throughput gate produced a genuine finding, not a rubber-stamped measurement. First pass (94 MB real D4 capture): 79.5 s, 1.2 MB/s. Profiling it found `FlowBuilder._expire()` — an O(open flows) scan run on every single packet — was 34% of total wall time, ahead of all `dpkt` parsing combined. Root cause: only the one flow a packet actually belongs to needs an up-to-the-packet answer; every other open flow was being needlessly re-checked every packet regardless of whether anything about it could have changed. Fixed by keeping the per-packet check for that one flow exactly as before, and throttling the sweep over every *other* flow to `min(all four timeouts)/4` of capture time. Re-measured: 54.4 s, 1.7 MB/s — 46% faster, byte-for-byte identical flow output (same counts, confirming the fix changed only speed). A regression test (`test_key_reuse_is_caught_even_when_sweep_is_throttled`) was written for the one correctness risk this kind of throttling can introduce — a same-key reuse landing in the gap between sweeps — and confirmed to actually fail (0 expired instead of 1) when the per-packet check is deliberately removed, via mutation, not just reasoned about.

  Extrapolated to ISCX's 28 GB even after the fix: ~281 minutes, still far past the 60-minute gate. Spec 001's open question ("ISCX subset size... default: all pcaps, since disk allows it") is corrected — that reasoning only ever considered disk space, never time. D3 will be a per-class subset once unblocked.

  D4's full corpus then exported for real with the fixed code: all 24 files, 403,394 flows (232,348 dropped for zero PPI — mostly MySQL/FTP/SMB's control-heavy traffic), 187 MB of shards, in 4.8 minutes (18,875 pkt/s, 13.7 MB/s aggregate — faster than the single-file measurement, since the fix helps proportionally more on files with fewer concurrent flows than the botnet capture used to measure it). Spot-checked: 24 distinct `session_id`s, `ppi_len` in [1, 30] with none at 0, label/category maps match spec 001's 20 classes and 2 categories exactly, `M.verify("ustc-tfc2016")` still `True`.
- **2026-09-20.** T5a done: pulled one real day (`WEEK-2022-00/2022-01-01/flows-20220101.csv.xz`, 487,081 rows) plus its `stats-20220101.json` and the week stats via `kaggle datasets download -f`, recorded the real 45-column header, the `PPI` field's encoding, the label columns and the identifier columns in `docs/datasets/cesnet-tls-year22.md`. Two findings from reading real bytes, not the DataZoo docs:
  - `json.loads` measured ~17x faster than `ast.literal_eval` on 50,000 real `PPI` cells (1.3 s vs 22.1 s), zero parse failures either way — used `json.loads`.
  - The mirror's `WEEK-2022-NN` folders are not plain ISO calendar weeks. `2022-01-01`/`02` sit in `WEEK-2022-00` even though their true ISO week is `(2021, 52)` — confirmed directly against the mirror (`WEEK-2022-01/` starts the following Monday, 2022-01-03, exactly true ISO week 1). Checked the symmetric year-end case too: `WEEK-2022-52/` runs 2022-12-26 to 2022-12-31 with no rollover, and neither `WEEK-2022-53` nor `WEEK-2023-00` exists — that edge is unverified for other years. Also found a genuinely empty day (`stats-20221231.json`: 0 flows) whose `.csv.xz` decompresses to zero bytes, which crashes plain `pandas.read_csv` with `EmptyDataError` rather than yielding zero rows — handled explicitly.

  T5b done: `data/cesnet_csv.py`, `scripts/export_raw_csv.py`. Flow statistics are computed by building a `FlowRecord` per row and calling its existing `flowstats()` — the same 46-column definition D3/D4 use — not reimplemented. 26 new tests, full suite 263/263, ruff/mypy clean.

  Running the exporter against the *real* downloaded day (not just the fixture) found a second real bug, more serious than T5a's: partitioning shards by each row's own `TIME_FIRST` (converted to an ISO week) put 17,850 of the file's 487,081 real rows (3.7%) into a `WEEK-2021-52` shard set with no counterpart anywhere in the mirror. Root cause: `TIME_FIRST` is UTC; the mirror's own per-file grouping is local time (CET), so the first ~hour of any local day's flows carry a `TIME_FIRST` on the *previous* UTC calendar date. Fixed by deciding a shard's period once per file, from the file's own name (`date_from_filename`), never per row — every row from one file now lands in that file's one period, matching the mirror's real layout. This also turned out to be the bigger performance win: moving the ISO-week computation from once-per-row to once-per-file was a measured 2.6x speedup on the same file (172.6 s → 66.4 s, 2,822 → 7,339 rows/s), since `pd.Timestamp(...).isocalendar()` on 487,081 individual rows was costing more than the rest of the per-row work combined. A regression test (`test_period_is_the_files_own_date_not_each_rows_time_first`) reproduces the exact scenario at 2-row scale. The stale shard output produced by the buggy version was deleted and the file re-exported with the fix before being spot-checked: 487,081 flows (matches `stats-20220101.json`'s `total-saved` exactly), 179 classes / 23 categories, `ppi_len` in `[3, 30]` with none at 0, single `session_id`, `source_manifest_hash` recorded.

  The full weeks-11-52 corpus itself was not run locally — consistent with this plan's own scope decision (D1 acquisition: Path B in a Kaggle CPU kernel, not a 30 GB local download); at the measured 7,339 rows/s, the full year is a multi-hour job that belongs on Kaggle, not this laptop, the same way D3's bulk download is code-complete but blocked outside this repo.
- **2026-09-20.** T6 done: `evaluation/metrics.py` (classification, earliness, policy, open-set and calibration metrics — pure NumPy/pandas, deliberately not scikit-learn, which stays a phase-2 `train` extra per `pyproject.toml`; AUROC/AUPR are the standard rank-based formulas), `evaluation/protocol.py` (`load_split` plus all four of spec 004's leakage assertions, enforced at load time), `evaluation/unknown_split.py` (the stratified known/unknown class draw), the four `configs/splits/*.yaml`, and `scripts/make_unknown_split.py`. 41 new tests, every metric hand-computed by hand (not cross-checked against another library), full suite 304/304, ruff/mypy clean.

  Two real bugs found while building this, both by actually exercising the code rather than trusting the design on paper:
  - **`assert_no_session_overlap` initially rejected D3's own intended design.** The first version treated any period name claimed by two different splits as an automatic conflict — which is exactly what D3's grouped-CV split needs to do (`train` and `test` legitimately share the single `all` period, filtered by disjoint `session_ids`). Caught immediately by `test_session_ids_filter_restricts_one_shared_period` and three others failing on the first run. Fixed by only treating a period claim as exclusive when it isn't filtered by `session_ids` — a period claimed *whole* by one split can never be shared, but two splits sharing one period via disjoint `session_ids` is fine, with the actual overlap check happening at the `(period, session_id)` level instead.
  - **`draw_unknown_split`'s seed had no effect on real data.** The first version picked each category's unknown classes by alternating the rarest and most-frequent remaining class by support rank — a selection fully determined by the support ordering whenever support values have no ties, which is the normal case on real data (class flow counts are essentially never exactly equal). `rng.shuffle` before the sort was silently discarded by the subsequent sort-by-support, so seed 42 and seed 43 produced the *identical* draw. Not caught by the unit tests (their synthetic fixture had tied support values within each stratum, where the shuffle-then-stable-sort happens to matter) — caught by running `scripts/make_unknown_split.py` for real against the real D4 shard set (`ustc-tfc2016/all`), where both seeds gave the same 5-class unknown set. Fixed by replacing the rank-based alternation with a genuine random choice (`rng.choice(..., replace=False)`) split between each category's rarer and more-frequent halves; re-running the same real command afterward gave two different sets ([1, 3, 7, 11, 17] vs [5, 7, 11, 15, 18]).

  D3 has no real data yet (registration-gated), so `configs/splits/d3_grouped.yaml`'s session_ids mechanism is proven against a synthetic fixture instead; `configs/splits/d4_anomaly.yaml` **is** run for real, against the actual D4 shard set, and asserted against its exact real category counts (282,412 benign / 120,982 malware, matching plan T4's numbers).
- **2026-09-20.** T7 done: five dataset cards (`docs/datasets/{cesnet-tls-year22,cesnet-quic22,iscx-vpn-2016,ustc-tfc2016,self-captured}.md`). License and citation for every dataset were researched, not assumed — D1 (CC BY 4.0, Hynek et al., *Scientific Data* 2024) and D2 (CC BY 4.0, *Data in Brief* 2023) confirmed on their own Zenodo records, not just the paper; D3 confirmed to have **no formal open license**, only a mandatory-citation "publicly available for researchers" notice, by fetching `unb.ca/cic/datasets/vpn.html` directly; D4's GitHub mirror repository is tagged `MPL-2.0` via the GitHub API, flagged in the card as covering the mirror maintainer's own repo, not a relicensing of the underlying CTU/paper data — cite the paper (Wang et al., ICOIN 2017), not the repo tag. D4's card carries the full real 20-class support table (`data/processed/ustc-tfc2016/all`); D1's carries what the one real exported day shows (179/180 apps, 23/24 categories); D2's and D3's cards say plainly that nothing in them has been checked against real bytes yet, rather than presenting spec 001's target numbers as if they were measurements.

  T8 done: spec 001-004 status lines rewritten to name exactly what's implemented, what's real, and what's deferred and why — two were stale in a way worth noting (spec 003 still said "shard writer and tokeniser pending" though both landed in plan T2, and spec 004 was still "draft" though T6 landed its first pass), not just spec 001/002 which already tracked recent work. `specs/README.md`'s build-order `State` column updated to match, steps 1-4. The six spec corrections (table above) were already landed incrementally as each task shipped — none were outstanding at this step, confirmed by re-reading the table rather than assumed. [Exit criteria](#exit-criteria) reassessed per-item rather than declared green as a block: 3 of 7 fully met, 2 partially met (D4 real / D3 blocked; manifest valid for what's downloaded), 2 genuinely blocked outside this repo (D1's weeks-11-52 corpus needs a Kaggle kernel; the D1 `Standardizer` needs that corpus to fit on). Phase 1's tasks (T1-T8) are complete; phase 1's *data* is not, by design, since two steps were never this repo's to finish alone.
- **2026-09-21.** D3 unblocked for real, one day after phase 1's close-out — the user completed ISCX registration and supplied 2 of 5 real archives (`VPN-PCAPS-01.zip`, `NonVPN-PCAPs-01.zip`). Two real gaps found and fixed immediately by running the existing "code complete, data blocked" pipeline against real bytes for the first time:
  - **The real post-registration listing serves zip archives, not individual pcaps.** `iscx_download.py`'s `discover_files` was written assuming a flat pcap listing like USTC's (the only real precedent available at the time, spec 001's original docstring said so explicitly). The real `/PCAPs` page lists `VPN-PCAPs-01/02.zip` and `NonVPN-PCAPs-01/02/03.zip`. Fixed by adding `adl_etc.data.download.extract_zip` (stdlib `zipfile`, no new dependency, mirroring `extract_7z`'s flatten-and-collision-check design) and pointing `discover_files` at `*.zip` links instead.
  - **A third of the real NonVPN archive is `.pcapng`, not `.pcap`** (11 of 23 files) — the newer capture format `pcap_source.py` already parses via `dpkt.pcapng.Reader` (spec 002), confirmed unrelated to this bug. Two independent places needed fixing, not one: `extract_zip`'s default suffix now matches `(".pcap", ".pcapng")` instead of just `.pcap`, and a **separate** bug was found in `export_pcap.py`'s own file-discovery (`pcap_dir.glob("*.pcap")`, unrelated code, same silent-drop shape) — fixed to check both suffixes via `PCAP_SUFFIXES`. Neither bug raised an error; both would have silently exported zero of the affected files with a lower flow count and no indication why. Caught by testing the label heuristic and then the full export against real files, not by inspection.
  - The label-inference heuristic (`iscx_labels.py`, written months earlier against *reconstructed* file names, never a real listing) was checked against real file names for the first time: **all 37 real files across both archives classified correctly, zero unresolved.**
  - Ran the full real export of everything downloaded so far: 37 files, 130,303 flows (1,590 dropped for zero PPI), 1,786.8 MB in 64.6 s — **27.7 MB/s, 16x the 1.7 MB/s throughput spec 001 recorded the day before** (2026-09-20's ISCX subset-size resolution), because that earlier number was extrapolated from D4's `Neris.pcap` (a botnet capture, not representative of ISCX's chat/voip/p2p/video traffic) since no real ISCX file existed yet to measure directly. Extrapolated to the full 28 GB corpus: **~17 minutes, not ~281** — comfortably under the 60-minute gate. The "D3 will be a per-class subset, never the full corpus" decision this plan and spec 001 both recorded is **reversed**; whether to fetch the remaining 3 archives is now a bandwidth/time tradeoff for the user, not a throughput blocker.
  - 6 new tests (`extract_zip`'s pcapng-matching and existing coverage, `iscx_download.py`'s zip-discovery rewrite, `export_pcap.py`'s pcapng-discovery regression), full suite 310/310, ruff/mypy clean. `docs/datasets/iscx-vpn-2016.md` and spec 001 rewritten with the real numbers and both corrections.
