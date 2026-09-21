# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AnytimeETC: a research project on metadata-only classification of encrypted network flows that decides *early* (after each of the first 30 packets), can answer "unknown", and holds a compute budget under drift. `specs/000-system-overview.md` has the thesis, contributions (C1-C4), hypotheses and module map; `docs/research-analysis.md` has the novelty argument and dataset/Kaggle strategy.

**Current state:** phase 1 (data pipeline, specs 001-004) is done; phase 2 (`plans/phase-2-tracking-kaggle-baselines.md`: specs 014, 015, 005) is in progress. Built so far in phase 2: config/seeding/run identity, the run-directory `Tracker` and MLflow sync (`src/adl_etc/utils/`), `data/prefix_stats.py`, the shared training skeleton (`src/adl_etc/training/`: labels, losses, checkpoints, batches, loop), and the baselines B1 XGBoost / B2 CNN / B3 GRU-LSTM (`src/adl_etc/models/`; B4 is deferred). `src/adl_etc/{inference,service,dashboard}/` are still empty placeholders (next: make the metrics read through `DenseLogits`, then the evaluation entry point) `kernel/kernel.py` is a stub, and `kernel/export-d1/` (the D1 export kernel, tested against a synthetic mirror) is built but not yet pushed. Specs 005-020 describe planned work unless their plan says otherwise. The plans hold the build logs and what is blocked (full-year D1 export via a Kaggle kernel; D3 needs the user's manual ISCX registration).

## Commands

Setup: `python -m venv .venv`, activate (`.venv\Scripts\activate` on Windows), `pip install -e ".[dev]"`. The core install deliberately has no `torch`; the `train`, `datazoo`, `service`, `capture` extras are installed only when their phase starts (`train` is installed in the current `.venv`). Don't import torch/scikit-learn/mlflow from `data/`, `evaluation/` or `utils/tracking.py` (`evaluation/metrics.py` is pure NumPy/pandas; `tracking.py` must work on a Kaggle kernel with no mlflow, and a test enforces it). Tests that need torch/mlflow skip with a reason when the extra is missing. **Kaggle's image is older than this machine** (numpy 2.0.2, pandas 2.3.3, pyarrow 24, torch 2.10 vs local numpy 2.5.3, pandas 3.0.6, torch 2.14) and has no `mlflow`, `dpkt` or `py7zr`; see spec 015 for the verified list. Code a kernel imports must work on those versions without those three packages. Before relying on a new dependency or API, re-run the suite in a venv pinned to Kaggle's versions (`pip install numpy==2.0.2 pandas==2.3.3 pyarrow==24.0.0 omegaconf==2.3.0 scikit-learn==1.6.1 xgboost==3.2.0 dpkt py7zr pytest`, then `PYTHONPATH=src pytest`).

```
pytest                                   # whole suite
pytest tests/data/test_flows.py          # one file
pytest tests/data/test_flows.py::test_name   # one test
pytest -m "not integration"              # skip tests that open loopback sockets (download tests)
ruff check src/ tests/ scripts/          # line length 100; rules E,F,I,UP,B,SIM
mypy src/
```

pytest, ruff and mypy must all be clean before a task counts as done. Markers: `slow`, `gpu`, `integration`.

Data CLIs (thin wrappers in `scripts/` over logic in `src/adl_etc/data/`):

```
python scripts/download_all.py --datasets d3 d4 [--dry-run] [--files GLOB]   # D3 ISCX, D4 USTC
python scripts/download_iscx.py --base-url <post-registration URL> | --files-from <list>
python scripts/export_pcap.py ...        # PCAP -> shards (configs/data/pcap.yaml)
python scripts/export_raw_csv.py --files data/raw/cesnet-tls-year22/flows-*.csv.xz --out-root data/processed --dataset cesnet-tls-year22   # --verify checks against stats-*.json instead of exporting; --sample-rate 0.03 [--sample-seed N] keeps a uniform seeded sample (recorded in meta.json); --check-stats verifies each day against its stats file while exporting
python scripts/make_unknown_split.py     # one-off open-set class draw for configs/splits/
./scripts/kaggle_sync.sh check|push-dataset|version-dataset|push-code|push-kernel [dir]|pull-results   # see docs/kaggle-workflow.md; `check` fails (correctly) when private endpoints answer "Authentication required", even though public downloads still work
python scripts/mlflow_import.py --src <pulled-run-dirs> [--dry-run]   # run dirs -> results/mlflow.db (idempotent)
mlflow ui --backend-store-uri sqlite:///results/mlflow.db            # browse; on Windows, killing this leaves uvicorn workers holding the port
```

## Architecture (phase 1)

Data flows: **capture/CSV → flows → PPI arrays → shards on disk → tokenised/continuous views → evaluation**. Each stage's contract is one module; read these to understand the rest.

- `data/ppi.py` — single source of truth for the numeric flow layout: `K_MAX = 30` packets, each `(ipt_ms, dir, size, push)`, in the same column order as CESNET DataZoo so PCAP-derived and published flows are interchangeable. Changing any constant here means re-exporting every shard.
- `data/pcap_source.py` (dpkt, header-only, sizes from IP/L4 length fields rather than captured bytes) → `data/flows.py` (`FlowBuilder`: packets → `FlowRecord`). Two rules matter: only payload-carrying packets enter the PPI (ACK/SYN/FIN-only packets count toward flow stats but not PPI slots), and direction is relative to the flow initiator, never to IP/port. Use a fresh `FlowBuilder` per capture file.
- `data/cesnet_csv.py` — second source (raw CESNET weekly CSVs). It builds a `FlowRecord` per row and calls the same `FlowRecord.flowstats()` (46-column vector, *not* DataZoo's 43) so both sources share one flow-statistics definition. Its column/encoding assumptions were verified against a real file and are recorded in `docs/datasets/cesnet-tls-year22.md`.
- `data/tensors.py` — `ShardWriter`/`ShardSet`. A shard is a directory of `.npy` arrays (mmap-able, unlike `.npz`) plus `meta.json` and a `flows.parquet` audit sidecar that is never a model input. `ARRAY_SPEC` is the schema; add a column by editing that dict only. Both exporters write every array, so models never care which backend produced a flow. `ShardSet.open` refuses to combine shard sets with mismatched `flowstats_source`.
- `data/features.py` — the model-facing contract: `tokenize` (Transformer input), `continuous` (CNN/RNN input), `prefix`, `Standardizer`, augmentations, `StreamTensorizer`. **Token/index 0 always means "no packet"** in every channel (push included); real values start at 1. Bin edges are frozen literals, with tests asserting they match the generator functions — don't recompute them at import.
- `evaluation/metrics.py` — pure functions of saved arrays (`logits[N, K, C]`, `labels[N]`, `ppi_len[N]`), never of a model, so thresholds/policies can be re-scored offline. For flows shorter than K the effective K is `min(k, ppi_len)`, computed once in `_index_at_k`; reuse it rather than re-deriving.
- `evaluation/protocol.py` + `configs/splits/*.yaml` — `load_split` turns a split YAML into shard sets and enforces spec 004's leakage rules at load time (`LeakageError`): no session overlap, consistent standardizer hash, no unknown classes in train, temporal ordering. `evaluation/unknown_split.py` draws the known/unknown class partition (design-time, not per-load).
- `utils/` (phase 2): `config.load_config` composes `defaults:` lists (paths relative to the naming file), applies dotted CLI overrides and **raises on unknown keys**, and returns a read-only resolved config; `config_hash` hashes resolved values. `seeding.seeded_generator(seed, purpose)` gives independent streams so adding a draw never shifts another. `runinfo.RunInfo` records commit, dirtiness (**unknown counts as dirty**; Kaggle reads `GIT_COMMIT`/`GIT_DIRTY` sidecars that `push-code` writes), config hash and hardware.
- Tracking is two layers on purpose: `tracking.Tracker` writes a dependency-free **run directory** (`run.json`; `params.json` write-once; `metrics.jsonl` append-only), and `mlflow_sync` pushes those into a local **SQLite** MLflow store, idempotently and resume-aware. MLflow 3.x raises on the file store, so never point it at `file:` URIs. `atomic.py` is the one write-temp-then-rename helper.
- `training/` (needs torch): `labels.LabelSpace` maps shard label ids to contiguous model indices (an id in neither known nor unknown **raises**; its hash is stored in every checkpoint). `datasets.FlowBatches` yields index-slice batches from in-memory `ArrayData` (no DataLoader), re-zeroes padded positions after standardising, takes a *loaded* `Standardizer` and has no fit path, and refuses unknown-class flows when `train=True`. `loop.fit` is the one training loop for every baseline: a model returns logits `[B,C]` or `[B,K,C]` (the latter uses `losses.multi_prefix_ce`). **Resume is bit-exact** because each epoch's data order and dropout derive from `(seed, epoch)`; the checkpoint is written before `state.json`; resuming under a different config, label space or standardizer raises. `pause_check(epochs_done, seconds)` is the hook for Kaggle's time guard. Checkpoints load with `weights_only=True`.
- `evaluation/dense_logits.py`: saved logits are a `DenseLogits` that carries its `indexing`. **Causal** models (GRU) are `effective`: position `min(K, ppi_len) - 1`, every K available. **Per-K** models (XGBoost, the CNN) are `nominal`: position `K - 1`, only `evaluated_k` exist, no clamping, and another K raises `NotEvaluatedError`. Never file per-K outputs at the effective slot (the K=10 and K=12 models' answers for a short flow collide) or read a 13-wide grid positionally. `evaluation/metrics.py` still takes a raw `[N,K,C]` array and would misread a `nominal` one; converting it is the next task.
- `models/baselines/`: `build_model(cfg.model, n_classes)` for `cnn`/`gru`/`lstm` (unknown names or args raise); `xgb.XGBBaseline` is separate (one booster per K on `prefix_flowstats(K)`, class-weighted, best-iteration predictions). `predict.predict_dense` turns a torch baseline into a `DenseLogits` from its own `causal` flag (GRU: one pass; CNN: one `ArrayData.prefix(K)` pass per grid K). The CNN uses `MaskedBatchNorm1d` because real flows average ~7 real packets in 30 slots. Configs live in `configs/models/baselines/` with a `search:` block for the 20-trial `models/search.py` sampler (not yet run; it needs D1).
- `data/prefix_stats.py`: `prefix_flowstats(ppi, ppi_len, k)` is the honest early tabular feature. It recomputes every column from the first K PPI entries and **never reads the stored whole-flow `flowstats`**, so its k=30 output intentionally differs from the stored vector (only the PPI-derived columns match); only `FLAG_PSH` is recoverable, the other flag columns are 0.
- `data/manifest.py` + `utils/provenance.py` — every download is registered in `data/manifest.json` with per-file sha256; every artefact should be able to name the git commit and source bytes that made it. Use these helpers rather than hashing or timestamping ad hoc.
- Downloaders: `data/download.py` holds the shared resumable/retrying HTTP (stdlib `urllib` only), GitHub listing and zip/7z extraction; `ustc_download.py`, `iscx_download.py`, `iscx_labels.py` hold dataset-specific knowledge. ISCX labels are inferred from file names heuristically (`confidence="heuristic"` in `labels.csv`). D3 is registration-gated and intentionally not automated.

Datasets: D1 = CESNET-TLS-Year22 (primary, temporal splits by ISO week), D2 = CESNET-QUIC22, D3 = ISCX VPN-nonVPN 2016 (grouped-by-file CV), D4 = USTC-TFC2016 (anomaly). `docs/datasets/<name>.md` records each one's real status; trust `data/manifest.json` over specs for what actually exists.

## Testing notes

**D4 is a trap for naive experiments** (measured; see `docs/datasets/ustc-tfc2016.md`): 90% of its flows are exact PPI duplicates (label-majority ceiling 0.9126 for any model), and shards are ordered by capture file, so a contiguous train/val slice has disjoint classes and a random flow-level split leaks duplicates. Shuffle, or split by `session_id`, and never report a flow-level-split score as a result. **D3 is 81% one class** (105,633 of 130k flows; the smallest class has 130): use balanced accuracy or macro-F1, since class-balanced training deliberately trades plain accuracy away (the GRU's D3 accuracy is 0.18 while its balanced accuracy is 0.62). The CNN's K=1 accuracy is at chance by construction (trained on full flows only), which is a finding, not a bug. `tests/training/test_real_d4.py` (marked `slow`, skipped without the shards) encodes what D4 does allow.

`tests/` mirrors `src/adl_etc/`. `tests/conftest.py` builds a synthetic capture whose expected PPI is hand-computable; it is the ground truth for flow-construction tests, so extend it rather than mocking flows. Download tests use a loopback HTTP server and are marked `integration`.

## Project structure

- `specs/` — feature/requirement specifications, one file per feature, written before implementation (index in `specs/README.md`, use `template.md`).
- `plans/` — implementation plans derived from specs (`plans/README.md` indexes them). Phase plans are `phase-<N>-<name>.md`; a plan is only numbered `NNN-` if it is the plan for spec `NNN`.
- `agents/` — role definitions for the project's own multi-agent pipeline (data/training/evaluation agents), distinct from `.claude/agents/` (Claude Code subagents). `agent-memory/<agent-name>/` holds each agent's persistent logs.
- `.claude/` — Claude Code config (`settings.json`, `agents/`, `commands/`).
- `data/raw/`, `data/processed/` — gitignored except `data/manifest.json` and `data/processed/dataset-metadata.json`. `results/` is gitignored. Checkpoints (`*.pt`, `*.ckpt`, `*.onnx`, `*.h5`) are never committed.
- `configs/` (`data/`, `splits/`), `kernel/` (Kaggle kernel), `notebooks/`, `docs/`.

## Workflow conventions

1. New feature/capability → write a spec in `specs/` first.
2. Before implementing → write a plan in `plans/` derived from the spec.
3. Multi-agent or long-running work → define the agent's role in `agents/` and let it persist findings/state in `agent-memory/<agent-name>/`.
4. Keep specs and plans updated as living documents, not one-off snapshots. When code diverges from a spec, record the correction (the phase-1 plan has a "Spec corrections" table) and update the spec's status line to say exactly what is and isn't implemented.

## Scope decisions

- Containerised deployment (spec 019) is deferred: the service, dashboard, DB (SQLite) and MLflow all run as local processes. Don't add Docker artefacts.
- Kaggle CLI credentials live at `~/.kaggle/kaggle.json` (gitignored); they're needed for the CESNET (D1/D2) mirrors (`pranjalkar99/cesnet-22`, `zilinpeng/cesnet-quic22`). Kaggle GPU training goes through `docs/kaggle-workflow.md`.
- Windows is the primary dev platform: prefer pure-Python backends (dpkt, py7zr) over Linux-only tools.
