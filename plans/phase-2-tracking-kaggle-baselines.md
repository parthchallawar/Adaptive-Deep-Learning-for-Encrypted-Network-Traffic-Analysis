# Plan: Phase 2, tracking, the Kaggle pipeline and the first baselines

- **Specs:** [014](../specs/014-experiment-tracking-and-reproducibility.md), [015](../specs/015-kaggle-training-pipeline.md), [005](../specs/005-baseline-models.md) (build steps 5 to 7)
- **Status:** not-started
- **Entry gate:** phase 1's [exit criteria](phase-1-data-pipeline.md#exit-criteria) are 3 of 7 fully met. Phase 1's own assessment is that phase 2 may **start** on that basis (014 and 015 need a mmap-able `ShardSet`, which D3 and D4 both provide for real), and this plan closes the two outstanding criteria itself in [T3](#t3--kaggle-code-delivery-and-the-d1-corpus-export-closes-phase-1s-last-two-criteria).
- **Exit gate:** phase 3 (spec 006, the PAT backbone) cannot start until [Exit criteria](#exit-criteria) are all green.

## Approach

Phase 1 built things that are *pure functions of arrays*, so they could be tested against hand-computed expectations. Phase 2 is the first phase where that stops being enough: a training run is stochastic, runs on a machine we do not own, and produces numbers nobody can check by hand. The discipline has to change shape accordingly.

Four rules hold throughout:

1. **Infrastructure before models, and both before real data.** 014 (tracking) lands before 015 (Kaggle), and 015's plumbing lands before 005's baselines, because an untracked first run is a run that gets repeated. This is already the specs' own build order; the reason is recorded here so it is not relitigated when the baselines look more interesting than the config loader.
2. **Every model is proven on real local data before it is given a GPU-hour.** D4 (403,394 real flows) and D3 (130,303 real flows) are on disk now. Any baseline that cannot overfit 512 real D4 flows on a laptop CPU is not ready for Kaggle, and finding that out locally costs minutes instead of a quota slot.
3. **The leakage surface moved.** Phase 1's leakage rules were about *splits*. Phase 2's are about *time within a flow*: a model or a feature that sees packet K+1 while claiming to decide at K invalidates the earliness claim that the whole project rests on. [T4](#t4--prefix_flowstats-the-honest-early-tabular-feature) is where this is decided, and it gets the same treatment phase 1 gave its leakage assertions — enforced in code and tested, not documented and trusted.
4. **A number that cannot name its run, config hash and commit is not a number.** This is spec 014's whole purpose. It is a rule, not a feature: no metric enters `results/summaries/` by hand.

## What this phase inherits, and what it must not touch

From phase 1: a `ShardSet` it can mmap, a `Standardizer` it **must not refit**, a metric suite that takes `logits[N, K, C]`, and split YAMLs that refuse to leak. The `ARRAY_SPEC` shard schema, the frozen bin edges and `PAD_INDEX = 0` are settled; changing any of them means re-exporting every shard, so phase 2 does not.

Phase 2 is also where **`torch` enters the project** (`pip install -e ".[train]"`, which also brings `scikit-learn`, `xgboost`, `mlflow`, `onnxruntime`). Phase 1's deliberate torch-free core stays torch-free: nothing under `src/adl_etc/data/` or `src/adl_etc/evaluation/` may import torch, so the metric suite and the exporters keep installing and running in seconds. [T5](#t5--the-training-skeleton-label-space-loaders-loop-losses) is the boundary.

## Findings that change the plan

Five things were checked against the code and the real data before this plan was written, rather than taken from the specs. Each changes a task below.

### F1 — `prefix_flowstats(K)` does not exist anywhere

Spec 005's B1 reads "statistics are recomputed from the first K packets (`prefix_flowstats(K)`)". A search of the entire repository finds that name **only in that sentence** — no implementation, no test, no caller. It is a forward reference that was never built, and B1 cannot exist without it. [T4](#t4--prefix_flowstats-the-honest-early-tabular-feature) builds it, and it is scheduled before the baselines rather than inside them because its definition is a research decision, not a coding detail (see F2).

### F2 — the honest-prefix definition is a leakage trap, and the stored vector cannot be reused

The 46 columns in `FLOWSTATS_COLUMNS` are computed from the **whole flow**: `BYTES`, `PACKETS`, `DURATION` and the six TCP flag columns are totals. An "early tabular baseline" that reads those at K=1 is reading the future, would look extraordinary at small K, and would invalidate every earliness comparison in the project — the exact failure the accuracy-vs-K curve exists to measure.

Two consequences, both of which must be explicit in code:

- `prefix_flowstats` recomputes **every** column from the first K PPI entries only. Nothing is copied from the stored `flowstats` array.
- `prefix_flowstats(K=30)` therefore **does not equal** the stored `flowstats` vector, and must not be expected to. The PPI holds only payload-carrying packets (spec 002), so a prefix statistic counts payload packets and payload bytes, while the stored vector counts every packet including pure ACKs. These are different quantities that happen to share a column name. A test asserts they differ on a real D4 flow, so a future reader cannot mistake the discrepancy for a bug.

### F3 — spec 005 still carries DataZoo's 43-column count

Spec 005 B1 says "43 standardised flowstats (spec 003)". Phase 1's [spec correction 1](phase-1-data-pipeline.md#spec-corrections-this-plan-requires) established that our vector is `FLOWSTATS_DIM` = **46** and DataZoo's is 43, that the two are not the same vector, and that `ShardSet.open` refuses to mix them. That correction was applied to specs 002 and 003 and **missed 005**. Corrected in [T9](#t9--tables-summaries-and-phase-close-out).

### F4 — `logits_at_k` assumes a dense 1..30 K axis, and spec 004 asks non-causal models for a 13-value grid

`metrics.logits_at_k` computes its index as `min(k, ppi_len) - 1` and indexes `logits[:, idx, :]` directly ([metrics.py:38-52](../src/adl_etc/evaluation/metrics.py#L38-L52)). That is only correct if axis 1 is K = 1, 2, ..., 30 densely. Spec 004 line 101 says non-causal baselines (B1, B2, B4) are evaluated at `K ∈ {1,2,3,4,5,6,8,10,12,15,20,25,30}` to bound cost — 13 values. Saving that 13-wide array into the canonical contract makes `logits_at_k(..., k=8)` silently return **grid position 8, which is K=12**. No exception, no shape error, just wrong numbers in the headline table.

Resolved in [T7](#t7--the-evaluation-entry-point-and-the-logits-artefact-contract): the artefact is always dense `[N, 30, C]`; a 13-value evaluation is **scattered into the dense array and the ungridded positions are recorded in a `k_evaluated` mask**, never left to be interpreted positionally. Metrics that read a masked position raise.

### F5 — Path B's full-year corpus is roughly 60 GB, which fits neither Kaggle nor the spec-003 budget

This is the finding that reshapes the phase. Measured, not estimated:

| Quantity | Measured value | Source |
|---|---|---|
| Shard bytes per flow | **445 B** | `data/processed/cesnet-tls-year22-probe` = 207 MB / 487,081 flows |
| Flows, 2022-01-01 | 487,081 | `stats-20220101.json` (`total-saved`) |
| Flows, 2022-12-26 | 387,815 | `stats-20221226.json` |
| Flows, WEEK-2022-00 (2 days) | 1,211,412 → ~606k/day | `stats-week.json` |

**Every day measured here is a public holiday** (Jan 1-2, Dec 26) — the three cheapest days to have probed, and the lowest-traffic days of the year. A mid-year working week is larger, not smaller, so these are lower bounds and the conclusion below only strengthens with real numbers.

At a conservative ~450k flows/day, one week is ~3.2M flows ≈ **1.4 GB**, and weeks 11 to 52 are ~132M flows ≈ **59 GB**. That exceeds Kaggle's 20 GB `/kaggle/working` cap (so no single kernel can even write it), the ~29 GB session RAM, and spec 003's "< 12 GB with D2" disk budget. Spec 015's throughput table further assumes "3M flows/epoch", which is spec 004's *XS* figure — Path B delivers the **full** release, about 18x XS, because the Kaggle mirror is the raw CSV publication and not the XS subset.

Spec 001 chose Path B for weekly granularity and recorded the disk cost of the *mirror*, but no document reconciles the exported corpus with either Kaggle's limits or spec 004's flow counts. **Path B must subsample at export time.** The rate and its justification are a scope decision below, and `export_raw_csv.py` gains a `--sample-rate` flag in [T3](#t3--kaggle-code-delivery-and-the-d1-corpus-export-closes-phase-1s-last-two-criteria).

## Scope decisions

Made once, here, so they are not relitigated mid-task.

| Decision | Choice | Reason |
|---|---|---|
| **D1 export sampling** | **Uniform 10% per day**, seeded from the file name, recorded in `meta.json` | Measured (F5): 10% of weeks 11-52 is ~13M flows ≈ **6.0 GB** — inside Kaggle's 20 GB working cap and spec 003's 12 GB budget — and gives ~5.1M train flows against spec 004's "about 3M in XS" target. Sample generously once: subsampling further at load time is free, re-exporting to get *more* costs another multi-hour kernel. |
| **Uniform, not per-class stratified** | uniform | Class-prior drift is part of what C4 and H4 study. Capping frequent classes would flatten the priors and delete the signal the drift benchmark exists to measure. The cost is rare classes, bounded by the check below. |
| **Rare-class floor** | verified post-export, not assumed | Measured per-app support on a real day: rarest `adobe-search` = **3 flows/day**, `redmine` = 8, then 36, 49, 51, 54, 59, 64; median app 1,029. Spec 004's floor is 100 test flows over a 4-week test period. At 10%, a class needs ≥36 flows/day to clear it, so ~2 of 179 apps drop out — and `draw_unknown_split` already excludes sub-floor classes by design (phase 1 T6). Note this floor **already excluded the rarest apps at full density** (3 × 28 = 84 < 100); sampling widens an existing gate, it does not open a new one. |
| **B5 (fixed-K Transformer) defers to phase 3** | deferred | B5 *is* spec 006's backbone, which is build step 8. Building a cut-down PAT in phase 2 and rewriting it in phase 3 is precisely the "tested against whatever the model happens to produce" failure phase 1's rule 1 exists to prevent. Spec 015's success criterion ("first real Kaggle run (baseline B5)") is corrected to name B2/B3 instead — the criterion's purpose is proving the pipeline end-to-end, which any real GPU job does. |
| **B4 (30pktTCNET) is optional in phase 2** | attempt last, may slip | Two blockers outside this repo: the `datazoo`/`cesnet-models` extras must install and fetch public weights, and spec 005's sanity check ("reproduces the public model's ballpark on **D2**") needs D2, which spec 001 lists as **not started**. Treated like phase 1 treated D3: build the adapter, test what can be tested, record honestly what could not. |
| **P-RL policy baseline defers to phase 3** | deferred | P-ECHO and P-CAPE are pure functions of saved logits and land in T7 for free. P-RL is a DQN training job whose only consumer is spec 009's comparison (step 12). Building it now means maintaining it for two phases before anything reads it. |
| **B3's multi-prefix loss** | minimal slice built now, in `training/losses.py` | Spec 005 says B3 trains with spec 008's multi-prefix loss, but 008 is step 10. The loss itself is a masked mean of per-position CE — small, and B3's per-step logits are the reason B3 is cheap to evaluate at every K. Spec 008 **extends** this function; it does not replace it. Named as a forward declaration so the ownership is unambiguous. |
| **Saved-logits evaluation cap** | 250,000 flows, uniform seeded subsample | Dense `[N, 30, C]` at fp16 with C=150 is 9 KB per flow; the full 4-week test period (~1.3M flows) would be 11.5 GB per run per seed. 250k flows is 2.25 GB and gives a 95% CI of about **±0.12%** on an accuracy near 0.9 — far tighter than any effect this project claims. The cap is a config value recorded in every report, not a hidden constant. |
| **MLflow store** | file store only, no server | Spec 014 already allows this and spec 019 (Docker) is deferred. A server is only needed for the registry API, which nothing in phase 2 uses. |
| **CI** | the local three-command check | Spec 015 calls the `--smoke` run "part of CI". There is no CI: no `.github/` exists in this repo. The smoke run is a pytest test instead, so `pytest` covers it, and spec 015 is corrected rather than left describing infrastructure that does not exist. |

## Tasks

### T1 — Config composition, seeding and run identity

**Why first:** every later task writes a config and needs a run name. Building it after the first model means retrofitting it onto a model that already hard-coded its hyperparameters.

**Files:** `src/adl_etc/utils/config.py`, `src/adl_etc/utils/seeding.py`, `src/adl_etc/utils/runinfo.py`, `configs/README.md`, `tests/utils/test_config.py`, `tests/utils/test_seeding.py`

**Interface**

```python
# config.py
def load_config(path: str | Path, overrides: Sequence[str] = ()) -> DictConfig:
    """Compose a YAML's `defaults:` list, apply CLI overrides, resolve, freeze."""
def config_hash(cfg: DictConfig) -> str:        # stable_hash of the resolved container
def save_resolved(cfg: DictConfig, dest: Path) -> None:   # config_resolved.yaml

# seeding.py
def seed_everything(seed: int) -> None:         # python, numpy, torch cpu+cuda, deterministic algos
def seeded_generator(seed: int, purpose: str) -> np.random.Generator
    """Independent streams per purpose (data order, augmentation, init) from one seed."""

# runinfo.py
@dataclass(frozen=True)
class RunInfo:
    run_name: str; seed: int; git_commit: str; git_dirty: bool
    config_hash: str; python: str; libraries: dict[str, str]; hardware: str
    def to_dict(self) -> dict: ...
def make_run_name(stage: str, model: str, variant: str, split: str, seed: int) -> str
```

**Decisions**

- `runinfo` reuses `utils/provenance.py`'s `git_commit`, `stable_hash` and `now_iso` rather than adding a second way to hash or timestamp things (phase 1's rule 2). `git_dirty` is new and belongs beside `git_commit`.
- Separate RNG streams per purpose: one global seed with a stream per consumer means adding an augmentation cannot shift the data order and silently change every earlier result.
- `configs/` gains `models/`, `train/` and `eval/` subdirectories alongside the existing `data/` and `splits/`. `configs/README.md` records the `defaults:` convention, because OmegaConf composition is not self-evident from the files.

**Edge cases**

- Unknown override key: raise. A typo'd `--override lr=0.1` against a config with `optim.lr` must not silently add a new key that nothing reads. (`configs/data/pcap.yaml` already set this precedent in phase 1.)
- `torch` absent: `seed_everything` seeds Python and NumPy and returns, so phase-1-only environments keep working.
- Non-git checkout (a Kaggle kernel has no `.git`): `git_commit` already returns `"unknown"`; `runinfo` must then read the commit from the code dataset's recorded metadata instead of reporting `"unknown"` for a run that genuinely knows its commit.

**Tests**

- Composition: a config with `defaults:` merges parent and child, child wins, and CLI overrides win over both.
- `config_hash` is invariant to key order and to comment/whitespace changes, and differs when any value changes.
- `seed_everything(0)` twice gives identical tensors from a 3-layer MLP's init and identical shuffle orders.
- Stream independence: drawing from the augmentation stream does not change the data-order stream's output.
- Unknown-key override raises.

**Done when:** the above pass, `ruff`/`mypy` clean, and a `RunInfo` round-trips to JSON and back.

---

### T2 — MLflow wrapper and the cross-machine merge

**Why here:** T3's Kaggle kernel writes MLflow runs on a machine we do not own. The import path has to exist before the first remote run, or its results are stranded.

**Files:** `src/adl_etc/utils/tracking.py`, `scripts/mlflow_import.py`, `tests/utils/test_tracking.py`

**Interface**

```python
class Tracker:
    @classmethod
    def start(cls, run: RunInfo, cfg: DictConfig, *, uri: str | None = None,
              enabled: bool = True) -> "Tracker": ...
    def log_params(self, flat: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None: ...
    def log_artifact(self, path: Path, subdir: str | None = None) -> None: ...
    def set_status(self, status: str) -> None: ...      # FINISHED | KILLED | PAUSED
    def __enter__/__exit__                              # KILLED on exception, always closes
```

**Decisions**

- `enabled=False` is a real, tested mode, not a debug afterthought: the `--smoke` run and every unit test use it, so no test writes to a tracking store.
- The tracking URI defaults to `file:results/mlruns` locally and `file:/kaggle/working/mlruns` when `/kaggle/working` exists. Detected once, in one place.
- `log_params` flattens the resolved config with dotted keys, and always logs `git_commit`, `git_dirty`, `config_hash` and `seed` as params so a run can be traced without opening artefacts.

**`scripts/mlflow_import.py`** copies run folders from a pulled Kaggle output into the local store and rewrites artifact URIs from `/kaggle/working/...` to the local path. Two hazards it must handle, because both silently corrupt a merged store:

- **Run-id collision.** Two machines generate UUIDs independently; a collision is unlikely but a *re-import of the same run* is not. Import is idempotent — re-importing an already-present run id updates it in place rather than duplicating it, and a genuine id collision with a different `config_hash` raises.
- **Partial runs.** Spec 014 requires a Kaggle run killed mid-flight to import with status `KILLED` and be excluded from tables. A run folder with no terminal status is imported as `KILLED`, not silently as `FINISHED`.

**Tests**

- Round trip: write a fixture run to a temp "remote" store, import it, assert params, metrics, artefact bytes and status all survive, and that artifact URIs point at existing local files.
- Idempotence: importing twice leaves one run.
- A status-less run imports as `KILLED`.
- `Tracker` with `enabled=False` writes nothing to disk and still returns working no-op handles.
- An exception inside the context manager sets `KILLED` and does not swallow the exception.

**Done when:** the above pass, and `mlflow ui --backend-store-uri file:results/mlruns` renders an imported fixture run.

---

### T3 — Kaggle code delivery and the D1 corpus export (closes phase 1's last two criteria)

**Why this early, out of spec order:** this is a multi-hour job on someone else's machine that blocks the baselines' real training data, and it is the one task whose latency cannot be compressed by working harder. It starts as soon as T1 is in — exactly as phase 1 kicked off its downloads during T1 — and runs in the background while T4 to T7 are built against D3/D4. It also needs **no GPU quota**: the export is a CPU kernel.

It closes phase 1's [exit criteria 2 and 5](phase-1-data-pipeline.md#exit-criteria), the last two genuinely outside this repo.

**Files:** `scripts/kaggle_sync.sh` (verify `push-code`), `kernel/export-d1/kernel-metadata.json`, `kernel/export-d1/kernel.py`, `src/adl_etc/data/cesnet_csv.py` (+`--sample-rate`), `scripts/export_raw_csv.py`, `docs/datasets/cesnet-tls-year22.md`

**Steps, in order**

1. **Re-verify Kaggle's constraints in the UI.** Spec 015's own table says to, "since they change" — quota, session cap, working-disk cap, RAM, whether this account may enable internet in kernels. Record what is actually observed in the spec, replacing the Sept 2026 figures. This costs ten minutes and every sizing decision below depends on it.
2. **Deliver the code.** `./scripts/kaggle_sync.sh push-code` packages `src/` + `configs/` as a private dataset; the kernel prepends it to `sys.path`. Route 2 of spec 015's code delivery, chosen as default because it does not depend on the unresolved internet question. The pushed commit hash is recorded in the dataset's description so `RunInfo` can report it from a checkout-less kernel (T1's edge case).
3. **Add `--sample-rate` to the exporter.** Uniform per-row sampling with `np.random.default_rng(seed_from_filename)`, applied **after** parsing and **before** buffering, so a sampled export is cheaper in memory and not merely smaller on disk. The rate and the seed go into `meta.json` — a shard set must be able to say it is a sample, or someone will compare a 10% week against a 100% week and report the difference as drift.
4. **Run the export kernel** over weeks 11 to 52 with `--sample-rate 0.10 --verify`. `--verify` compares each day against its shipped `stats-*.json` *before* sampling, so the check still tests the mirror rather than our own sampler.
5. **Fit the `Standardizer`** on the train weeks (11 to 26) only, save to `results/standardizer.json`, and wire its path into `configs/splits/d1_main.yaml`'s four `standardizer:` fields — which phase 1 left as `null` precisely because no real corpus existed to fit on. This closes criterion 5.

**Edge cases**

- 20 GB working cap: if the measured corpus approaches it, split the export into two kernels by week range and produce two dataset versions. Decide from the *measured* first-week output, not from F5's estimate.
- A day whose `.csv.xz` decompresses to zero bytes (phase 1 found a real one, 2022-12-31): already handled in `cesnet_csv.py`; the sampling path must not reintroduce a crash there.
- `--verify` failing on any day: stop and fall back to Path A, per spec 001's mandatory-verification rule. A third-party mirror that fails its own shipped statistics is not a corpus to train on.
- Sampling must be *per day*, seeded from the file name, so re-running one day reproduces that day's sample exactly and a partial re-export is consistent with the rest.

**Tests**

- `--sample-rate 0.5` on the 10-row fixture keeps a deterministic subset for a fixed seed, and the same rows on a re-run.
- `--sample-rate 1.0` is bit-identical to no sampling at all (the flag must not perturb the default path).
- `meta.json` records `sample_rate` and `sample_seed`; a set exported at 1.0 records `1.0`, not absence.
- `--verify` still passes against the *unsampled* day counts when sampling is on.

**Done when:** weeks 11 to 52 exist as shard sets on Kaggle and are mounted by a training kernel; `--verify` passed on every exported day; `results/standardizer.json` is fit on weeks 11-26 and referenced by `d1_main.yaml`; the rare-class floor check from the scope table is run and its real numbers recorded in the dataset card; and `protocol.load_split("configs/splits/d1_main.yaml")` loads for real with all four leakage assertions passing.

---

### T4 — `prefix_flowstats`: the honest early tabular feature

**Why separate from B1:** F1 and F2 above. Its definition decides whether the project's central earliness claim is measured honestly, which makes it a research decision that deserves its own tests, not a helper inside a baseline.

**Files:** `src/adl_etc/data/prefix_stats.py`, `tests/data/test_prefix_stats.py`

**Interface**

```python
def prefix_flowstats(ppi: np.ndarray, ppi_len: np.ndarray, k: int) -> np.ndarray:
    """[N,30,4] raw PPI -> [N, FLOWSTATS_DIM] float32, computed **only** from the
    first min(k, ppi_len) PPI entries. Never reads the stored flowstats array."""
```

**Decisions**

- Stays in `src/adl_etc/data/` and pure NumPy, so it keeps working in the torch-free core and can be called by the exporters later if prefix statistics are ever materialised.
- Column *identity* is preserved — the output has the same 46 columns in the same order, so `Standardizer.transform_flowstats` applies unchanged — but column *meaning* is "as observed through packet K". Documented at the top of the module in the terms of F2.
- Directional columns (`BYTES_REV`, `PACKETS_REV`, the `_REV` histograms) come from the PPI's `DIR` channel, which is relative to the flow initiator (spec 002). No addressing shortcut is reachable.
- TCP flag columns: the PPI carries only `PUSH`, so `FLAG_PSH` is derivable and the other five are not. They are emitted as **0**, and the docstring says so explicitly. Emitting the stored whole-flow flags would be exactly the leak F2 describes; leaving the columns out would break `Standardizer` compatibility.

**Edge cases**

- `k=1`: duration 0, one packet, empty IPT histogram. Spec 005 already calls this degenerate and acceptable — the curve starting low at K=1 is the honest result, not a bug to paper over.
- `k > ppi_len`: clamps to `ppi_len`, the same `min(k, ppi_len)` rule `metrics._index_at_k` uses. Reuse that helper rather than restating the rule.
- `ppi_len = 0` cannot occur (dropped at write, phase 1 T1) but is asserted rather than assumed.

**Tests**

- Hand-computed against `tests/conftest.py`'s reference 6-packet flow at k=1, 3, 6 and 30 — the same fixture phase 1 golden-tested, so the expected PPI is already known by hand.
- **Monotone-prefix property:** for a flow with `ppi_len=6`, the count-like columns are non-decreasing in k, and every column is *constant* for k ≥ 6.
- **The leakage test:** on a real D4 flow whose totals differ between K=5 and full length, `prefix_flowstats(k=5)` differs from `prefix_flowstats(k=30)`. A regression here means the future leaked in.
- **The F2 caveat, asserted:** `prefix_flowstats(k=30)` differs from the stored `flowstats` vector on a real D4 flow that has non-payload packets, so the discrepancy is locked in as intended behaviour rather than discovered later as a bug.
- Flags: only `FLAG_PSH` is ever non-zero.

**Done when:** the above pass, and `prefix_flowstats` over 100k real D4 flows at one k runs in under 2 s (vectorised, no per-flow Python loop — the B1 evaluation calls it 13 times).

---

### T5 — The training skeleton: label space, loaders, loop, losses

**Why here:** three baselines share it. Written once, the comparison between them isolates the architecture; written three times, it isolates nothing.

**Files:** `src/adl_etc/training/labels.py`, `datasets.py`, `loop.py`, `losses.py`, `checkpoint.py`, `tests/training/*`

**`LabelSpace` — the piece nothing has built and everything needs**

Shard `label` values are the dataset's own ids (0 to 179 for D1) and include classes held out as unknown. A model emits `C` contiguous logits. Without one authoritative mapping, every saved `logits[N,K,C]` array is uninterpretable and two runs can disagree silently about what column 7 means.

```python
@dataclass(frozen=True)
class LabelSpace:
    known: tuple[int, ...]           # dataset ids, sorted — model index i == known[i]
    unknown: tuple[int, ...]
    names: dict[int, str]
    def to_model(self, y: np.ndarray) -> np.ndarray:   # dataset id -> 0..C-1, -1 for unknown
    def to_dataset(self, y: np.ndarray) -> np.ndarray
    def save(self, path: Path) -> None; hash: str
```

Saved with every run and logged as a param. `evaluate` refuses to score logits whose `LabelSpace` hash differs from the checkpoint's.

**The rest**

- `datasets.py`: a torch `Dataset`/sampler over a `ShardSet`, mmap-backed, applying `tokenize`/`continuous` and a **loaded** `Standardizer`. It has no `fit` path at all — the one way to refit is to not have written the method (phase 1: "a `Standardizer` it must not refit"). Class-balanced sampling capped at 10x per spec 005.
- `loop.py`: AdamW, OneCycle, AMP, early stopping on val macro-F1 (patience 4), per-epoch checkpointing, and **resume from `state.json` + last checkpoint**, which T8's Kaggle guard depends on. No DataLoader workers (spec 015).
- `losses.py`: `cross_entropy_ls` (label smoothing 0.1) and `multi_prefix_ce(logits[B,30,C], y, mask)` — the masked mean over valid positions, declared as the minimal slice of spec 008 that B3 needs (scope decision above).
- `checkpoint.py`: atomic write (temp + rename, spec 015's edge case), storing model state, optimiser state, epoch, `RunInfo`, `LabelSpace` hash and `Standardizer` hash.

**Edge cases**

- A batch containing an unknown-class flow in *train*: raise. `ShardWriter` already refuses to write one (`allow_unknown=False`), and `protocol.load_split` asserts it, so this is the third and last gate.
- Standardizer hash mismatch between checkpoint and data: raise at load, not at first bad metric.
- Resume with a different config hash: raise. Resuming into a changed config produces a run whose name and metrics describe neither config.
- A class with zero flows in a training period: the balanced sampler must not divide by zero.

**Tests**

- `LabelSpace` round trip; `to_model(to_dataset(x)) == x`; unknown ids map to -1; hash changes when the known set changes.
- **Overfit test** (spec 005's own): each model reaches > 99% train accuracy on 512 real D4 flows within 200 steps. Marked `slow`.
- Resume: train 2 epochs, kill, resume, and assert the metric sequence continues rather than restarting — the same property T8 relies on for Kaggle.
- `multi_prefix_ce` equals plain CE when the mask selects one position; equals the hand-computed mean on a 2-flow, 3-position example.
- No refit: `datasets.py` exposes no `fit`, asserted by an API test so a future edit that adds one fails loudly.
- Determinism: same seed, same loss curve to 6 decimal places on CPU.

**Done when:** the above pass, and a 2-epoch B2 run on real D4 completes on the laptop CPU, writing a checkpoint that resumes.

---

### T6 — Baselines B1, B2, B3 (and B4 if it unblocks)

**Files:** `src/adl_etc/models/baselines/{xgb,cnn,rnn,tcnet_adapter}.py`, `configs/models/baselines/*.yaml`, `tests/models/*`

| Model | Input | Notes |
|---|---|---|
| B1 XGBoost | `prefix_flowstats(K)`, standardised | CPU, local. 13 fits, one per K in spec 004's grid — an honest early tabular baseline needs a model trained at each K, not one whole-flow model evaluated early. |
| B2 1D-CNN | `continuous` [30,4] | 3 Conv1d blocks (128/192/256, kernels 5/5/3), BN, GELU, masked avg+max pool, MLP head, ~0.4M params. Non-causal: one pass per K on zero-padded prefixes, 13 K values. |
| B3 GRU / LSTM | `continuous` [30,4] | Linear stem 4→128, 2-layer, hidden 256, **per-step logits**, trained with `multi_prefix_ce`. Causal: one pass gives all 30 K. ~0.6M params. Mirrors CAPE-Net's backbone, which is what makes the P-CAPE policy baseline faithful. |
| B4 30pktTCNET | PPI [30,3] in DataZoo scaling | Optional (scope decision). The adapter reproduces DataZoo's `ppi_transform`; our PPI is [30,4] with our own scaling, so this is a real conversion, not a reshape. |

**Decisions**

- B2 and B3 read `continuous`, which is **4** channels, not 3 — phase 1's [spec correction 2](phase-1-data-pipeline.md#spec-corrections-this-plan-requires) settled that and spec 005 inherits it.
- B1 consumes `FLOWSTATS_DIM` = 46 columns, not 43 (F3).
- Bounded random search, 20 trials on val, per spec 005's non-goal of exhaustive tuning. The search seed is part of the config, so a search is reproducible.
- **Every baseline is proven on real D4 before any Kaggle hour is spent** (rule 2). D4 is a 2-category, 20-class problem — an easier task than D1, which is the point: a model that cannot learn D4 is broken, not undertrained.

**B4's honest failure modes**, recorded in advance so they are not discovered as surprises: the `cesnet-models` weights may not download in this environment; the `ppi_transform` cannot be verified "against a DataZoo-produced batch" (spec 005's own test) without the Path A HDF5, which is not on disk; and its sanity criterion needs D2, which is not acquired. If B4 stalls, it lands in phase 3 with the reason written down, exactly as phase 1 handled D3.

**Tests**

- Shape/forward test per model with a batch of 8.
- Overfit test per model (in T5's harness).
- B1 at K=1 does not beat B1 at K=30 by more than noise — a cheap tripwire for the F2 leak, from the model's side rather than the feature's.
- B4 adapter: a hand-computed `ppi_transform` case, plus the DataZoo-batch comparison marked `slow` and **skipped with an explicit reason** if no HDF5 is present, never silently passed.

**Done when:** B1, B2 and B3 train on real D4 and real D3, each logging an MLflow run with a resolved config, and their acc@K curves are sane (monotone-ish in K, above the majority-class rate at K=1).

---

### T7 — The evaluation entry point and the logits artefact contract

**Why here:** the baselines have produced predictions, and spec 004's `evaluate(model_or_policy, split) -> Report` entry point does not exist yet — phase 1 built the metric *functions* but never the harness that drives them. This is also where F4 is resolved.

**Files:** `src/adl_etc/evaluation/run.py`, `src/adl_etc/evaluation/report.py`, `src/adl_etc/evaluation/policies.py`, `src/adl_etc/utils/plotstyle.py`, `configs/eval/*.yaml`, `tests/evaluation/*`

**The logits artefact — the contract F4 requires**

```
results/<run_name>/eval/<split>/
  logits.npy        # [N, 30, C] float16, dense over K = 1..30
  k_evaluated.npy   # [30] bool — which K were actually computed
  labels.npy, ppi_len.npy, flow_index.npy
  label_space.json, report.json
```

Always dense over all 30 K, never a 13-wide grid, because `metrics.logits_at_k` indexes positionally and a grid silently returns the wrong K (F4). Non-causal models scatter their 13 computed values into the dense array and mark the rest `False` in `k_evaluated`; any metric asked for a `False` position **raises**. Causal models mark all 30 `True`. `flow_index` records which flows the 250k cap selected, so a report can be re-scored or joined back to `flows.parquet`.

**Policy baselines** (`policies.py`), pure functions of that artefact, needing no model:

- `p_echo(logits, ppi_len, tau)` — commit at the first K where max-prob ≥ tau; the tau sweep gives the Pareto front.
- `p_cape(logits, ppi_len, tau_per_class, theta)` — per-class thresholds from val quantiles at 0.95 precision, falling back to the global threshold for classes with < 20 val flows (spec 005's min-support rule), plus the energy gate at K=10. Documented as an approximation of Learn-Then-Test, per spec 005.

These exercise `pareto_front`, `mean_k`, `coverage` and `committed_accuracy` — phase-1 functions that have never been run against real model output — and they are the reason P-RL can wait.

**`plotstyle.py`**: one colour-blind-safe palette, consistent axis labels ("packets read K", "accuracy"). Imported by every figure so the paper's plots match without per-script styling.

**Tests**

- A `k_evaluated=False` position raises rather than returning a number.
- `p_echo(tau=0)` commits at K=1 for every flow; `tau=1` never commits before K_max (spec 005's own test).
- `p_cape` falls back to the global threshold for a class with 19 val flows and uses its own at 20.
- Report determinism: same seed and same logits give the same report hash (spec 004).
- Round trip: write a report, reload, and the numbers match to fp16 precision.

**Done when:** `python -m adl_etc.evaluation.run --config configs/eval/d4_anomaly.yaml` produces a complete report with plots from a real B3 checkpoint, and spec 004's success-criterion command works with the corrected module path (F-note in T9).

---

### T8 — The Kaggle training kernel and the first real GPU run

**Files:** `kernel/kernel.py`, `kernel/kernel-metadata.json`, `kernel/run_queue.yaml`, `tests/test_kernel_smoke.py`

Replaces the current placeholder `kernel/kernel.py`, which prints a message and exits.

**Behaviour** (spec 015): read the session time limit and hold an **11.5 h guard**; check `/kaggle/working/state.json` for a partially finished run and resume from the last checkpoint; after each epoch estimate the next epoch's cost and, if it would breach the guard, save and exit cleanly with status `PAUSED`; work through `run_queue.yaml` in priority order, stopping when under 40 minutes remain; write MLflow runs and reports to `/kaggle/working`.

`state.json` is written **atomically** (temp + rename) — spec 015's edge case, and the same discipline T5's checkpoints use.

**`--smoke`**: the same code path on CPU with 5k flows, a 2-epoch fine-tune, in under 2 minutes, producing a report. It runs as a pytest test marked `slow` (the CI correction), so `pytest` covers it and it is not a command only a human remembers to type.

**The first real GPU run:** B3 on D1, three seeds, tracked, pulled back with `kaggle_sync.sh pull-results`, imported with `scripts/mlflow_import.py`. This is spec 015's end-to-end success criterion, with B3 in place of B5 (scope decision).

**Then record the measurements spec 015 left as estimates:** real throughput (flows/s forward+backward), real epoch wall time, real GPU-hours per job, and P100 vs T4x2 — spec 015 explicitly defers its "about 30k to 40k flows/s" figure to "the first smoke run", and its weekly budget table is an example, not a measurement. Replace both with real numbers, which is what phase 3's SSL budgeting will be sized from.

**Tests**

- Smoke run completes and writes a report.
- **Resume test** (spec 015's own): kill the smoke run after epoch 1, rerun, assert it resumes and the metrics continue rather than restarting.
- The time guard triggers a `PAUSED` exit when the estimated next epoch exceeds the remaining budget (simulated clock, not a 12-hour test).
- `run_queue.yaml` ordering is respected and a completed entry is not re-run on resume.

**Done when:** three real B3 seeds on D1 appear in local MLflow with their Kaggle provenance, no week exceeded the quota, and the measured numbers are in spec 015.

---

### T9 — Tables, summaries and phase close-out

**Files:** `scripts/make_tables.py`, `results/summaries/`, `.gitignore`, specs 004/005/014/015, `specs/README.md`

- **`scripts/make_tables.py`**: queries MLflow, aggregates by run-name pattern, computes mean ± std over seeds and paired bootstrap CIs (1000 resamples, spec 004), writes Markdown + LaTeX tables and figures to `results/summaries/`. Refuses runs with `git_dirty=true` unless `--allow-dirty`, excludes `KILLED` runs, and **errors on config drift** — the same run name with a different `config_hash` (spec 014's edge cases, all three).
- **`.gitignore` fix.** `results/*` with only `!results/.gitkeep` currently ignores `results/summaries/` entirely, while spec 014 says "large artefacts are gitignored, summaries are committed" and CLAUDE.md says the same. Add `!results/summaries/` so the intent is implemented rather than merely stated.
- **Spec corrections** from the table below, each landed with its task.
- **`specs/README.md`** build-order `State` column updated for steps 5, 6, 7.
- **Measured numbers recorded** in the progress log, so phase 3's sizing starts from measurements — phase 1's T8 rule.

**Done when:** `python scripts/make_tables.py` regenerates the baseline table from MLflow with zero manual edits, and every number in it traces to a run name, config hash and commit.

## Order and parallelism

```
T1 config/seed/runinfo ──► T2 mlflow ──┬──► T5 training skeleton ──► T6 baselines ──► T7 eval+policies ──┐
        │                              │           ▲                                                     │
        │                              └──► T4 prefix_flowstats ──┘                                      │
        │                                                                                                ▼
        └──► T3 push-code ──► D1 export kernel (CPU, hours) ──► Standardizer ───────────► T8 GPU run ──► T9
                 (starts with T1, runs in the background)
```

T1 blocks everything. **T3's export kernel starts the moment T1 lands** and runs unattended for hours — it is the critical path to real D1 data, and nothing else in the phase can shorten it. T4 and T5 are independent of each other and both feed T6. T7 needs only T6's checkpoints. T8 needs T5's resume support and T3's exported corpus.

If T3 stalls (mirror verification fails, or the corpus will not fit), T6 and T7 still complete on real D3/D4 data and the phase exits with D1 results missing and the reason recorded — the same shape phase 1 took when D1 and D3 were blocked.

## Spec corrections this plan requires

Applied when the task lands, not before, so specs always describe shipped behaviour.

| # | Where | Problem | Correction | Lands with |
|---|---|---|---|---|
| 1 | 005 | "43 standardised flowstats" — DataZoo's count. Phase 1 correction 1 fixed 002 and 003 and missed 005 (F3). | `FLOWSTATS_DIM` (46), with the standing rule that the two sources are never mixed. | T6 |
| 2 | 005 | `prefix_flowstats(K)` is referenced but was never built, and its honest definition was never written down (F1, F2). | Name the module, state that every column is recomputed from the first K PPI entries, and record that `prefix_flowstats(30) != flowstats` by design. | T4 |
| 3 | 005, 015 | B5 is spec 006's backbone (step 8), but 005 is step 7 and 015's success criterion names B5 as phase 2's first Kaggle run — a dependency inversion in both. | B5 moves to phase 3 with spec 006; 015's criterion names B2/B3. | T8 |
| 4 | 005 | B3 "trained with multi-prefix loss (spec 008)", which is step 10. | The masked multi-prefix CE is declared in `training/losses.py` in phase 2; spec 008 extends it rather than replacing it. | T5 |
| 5 | 004 | Non-causal models are evaluated on a 13-value K grid, but `logits_at_k` indexes a dense 1..30 axis — a 13-wide array returns silently wrong K (F4). | The artefact is always dense `[N,30,C]` plus a `k_evaluated` mask; masked positions raise. | T7 |
| 6 | 000, 004, 014 | Module paths predate the src-layout rename: `src.training.finetune`, `src.evaluation.run`, `src/utils/plotstyle.py`. The package is `adl_etc`. | `adl_etc.*` throughout; spec 000's module map updated. | T9 |
| 7 | 015 | The `--smoke` run is called "part of CI"; there is no CI in this repo (no `.github/`). | It is a pytest test marked `slow`; CI stays spec 020's continuous concern. | T8 |
| 8 | 001, 003, 004, 015 | No document reconciles Path B's full-release corpus (~60 GB) with Kaggle's 20 GB working cap, spec 003's 12 GB budget or spec 004's "about 3M flows in XS" (F5). | Record the measured per-flow and per-day figures and the 10% uniform sampling decision, in spec 001 and the dataset card. | T3 |

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **The D1 export is the critical path and runs on someone else's machine.** | Every D1 number in the phase. | T3 starts with T1, not after T7. Every baseline is proven on real D3/D4 first, so a D1 stall costs the D1 *results*, not the phase. |
| **Prefix statistics leak the future into B1.** | Would invalidate the earliness claim that is the project's core contribution, while looking like a great result. | F2's definition enforced in code (nothing read from stored `flowstats`), the monotone-prefix and K=5-vs-K=30 tests, and T6's tripwire that B1@K=1 must not rival B1@K=30. |
| **A 13-wide logits array silently mis-indexes** (F4). | Wrong numbers in the headline table, no error raised. | Dense `[N,30,C]` + `k_evaluated` mask; masked reads raise. |
| **Sampling D1 at 10% distorts what the drift study measures.** | C4 and H4 rest on it. | Uniform, never per-class stratified, so priors and their drift survive; rate and seed in `meta.json`; the rare-class floor verified against real post-export counts, not assumed. |
| **`Standardizer` refit on val or test.** | Silent train/test contamination across every model at once. | The dataset class has no `fit` method to call; hashes are checked at checkpoint load; `protocol.load_split` already asserts hash equality across splits. |
| **Kaggle quota burned on a broken run.** | A week of GPU time. | `--smoke` on CPU is a test, not a habit; overfit-512 gates every model; the time guard exits `PAUSED` rather than dying mid-epoch. |
| **Kaggle's published constraints have changed since Sept 2026.** | Every sizing decision in spec 015. | T3 step 1 re-verifies them in the UI first and rewrites the spec's table. Spec 015 itself asks for this. |
| **MLflow stores diverge across two machines.** | Unreproducible tables. | Idempotent import, colliding-id-with-different-config raises, status-less runs import as `KILLED` and are excluded from tables. |
| **B4 cannot be verified** (no DataZoo HDF5, D2 not acquired). | The public-SOTA comparison for H1 is weaker. | Optional by scope decision; its DataZoo-batch test skips with an explicit reason rather than passing silently; slips to phase 3 with the reason recorded. |
| **`torch` enters the dependency tree.** | Phase 1's fast, torch-free core could rot. | Nothing under `data/` or `evaluation/` imports torch; `prefix_stats.py` is pure NumPy; an import test asserts it. |

## Exit criteria

Phase 3 (spec 006) starts when all of these hold. Assessed per item, honestly, the way phase 1 assessed its own.

1. `pytest`, `ruff check src/ tests/ scripts/` and `mypy src/` all clean, with the new `slow`-marked overfit, smoke and resume tests included.
2. Weeks 11 to 52 of D1 exist as shard sets, `--verify` passed on every day, and the sampling rate is recorded in `meta.json` and the dataset card. *(Closes phase 1 criterion 2.)*
3. `results/standardizer.json` is fit on D1 train weeks 11-26 only, referenced from `d1_main.yaml`, and its hash assertion exercised against real D1 shards. *(Closes phase 1 criterion 5.)*
4. B1, B2 and B3 each have three-seed runs on D1 and on D4, logged to MLflow with resolved configs and commits.
5. `python -m adl_etc.evaluation.run` produces a complete report with acc@K curves and a P-ECHO Pareto front from a real checkpoint.
6. `python scripts/make_tables.py` regenerates `results/summaries/` from MLflow with zero manual edits, and refuses dirty-tree and drifted-config runs.
7. A Kaggle run that is killed mid-flight resumes and completes, proven once for real and not only in the simulated-clock test.
8. Spec 015's throughput, epoch time and GPU-hour figures are **measured**, replacing its estimates, and its weekly budget table is rebuilt from them.
9. All eight spec corrections landed; `specs/README.md` build-order `State` updated for steps 5 to 7.

What phase 3 inherits: a tracked, resumable, quota-aware training path; a label space and standardizer it must not redefine; baseline numbers to beat; and a logits artefact contract that spec 006's causal backbone fills densely for free.

## Progress log

- **2026-09-21.** Plan written. Five findings checked against the code and the real data rather than taken from the specs, each changing a task: `prefix_flowstats` does not exist (F1); its honest definition is a leakage trap and the stored vector cannot be reused (F2); spec 005 still carries DataZoo's 43-column count that phase 1 corrected elsewhere (F3); `metrics.logits_at_k` indexes a dense 1..30 axis while spec 004 asks non-causal models for a 13-value grid, which would return wrong K values with no error (F4); and Path B's full-year corpus measures ~60 GB against Kaggle's 20 GB working cap and spec 003's 12 GB budget (F5, from 445 B/flow measured on the real probe export and real per-day counts of 487,081 / 387,815 / 605,706 — all holidays, so lower bounds). Two dependency inversions found in spec 005: B5 needs spec 006 (step 8) and B3's loss needs spec 008 (step 10), the first of which also contradicts spec 015's own success criterion. Eight spec corrections scheduled. D1 sampling set to uniform 10% on the measured per-app support distribution (rarest app 3 flows/day; ~2 of 179 classes fall below spec 004's 100-test-flow floor, which already excluded them at full density).
