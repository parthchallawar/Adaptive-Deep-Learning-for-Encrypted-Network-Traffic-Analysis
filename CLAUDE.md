# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AnytimeETC: a research project on metadata-only classification of encrypted network flows that decides *early* (after each of the first 30 packets), can answer "unknown", and holds a compute budget under drift. `specs/000-system-overview.md` has the thesis, contributions (C1-C4), hypotheses and module map; `docs/research-analysis.md` has the novelty argument and dataset/Kaggle strategy.

**Current state:** only phase 1 (data pipeline, specs 001-004) is implemented. `src/adl_etc/{models,training,inference,service,dashboard}/` are empty placeholders, and `kernel/kernel.py` is a stub. Specs 005-020 describe planned work, not existing code. `plans/phase-1-data-pipeline.md` is the build log and lists what is still blocked (full-year D1 export via a Kaggle kernel; D3 needs the user's manual ISCX registration).

## Commands

Setup: `python -m venv .venv`, activate (`.venv\Scripts\activate` on Windows), `pip install -e ".[dev]"`. The core install deliberately has no `torch`; the `train`, `datazoo`, `service`, `capture` extras are installed only when their phase starts. Don't import torch/scikit-learn from phase-1 modules (`evaluation/metrics.py` is pure NumPy/pandas for this reason).

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
python scripts/export_raw_csv.py --files data/raw/cesnet-tls-year22/flows-*.csv.xz --out-root data/processed --dataset cesnet-tls-year22   # add --verify to check against stats-*.json instead of exporting
python scripts/make_unknown_split.py     # one-off open-set class draw for configs/splits/
./scripts/kaggle_sync.sh check|push-dataset|version-dataset|push-code|push-kernel|pull-results   # see docs/kaggle-workflow.md
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
- `data/manifest.py` + `utils/provenance.py` — every download is registered in `data/manifest.json` with per-file sha256; every artefact should be able to name the git commit and source bytes that made it. Use these helpers rather than hashing or timestamping ad hoc.
- Downloaders: `data/download.py` holds the shared resumable/retrying HTTP (stdlib `urllib` only), GitHub listing and zip/7z extraction; `ustc_download.py`, `iscx_download.py`, `iscx_labels.py` hold dataset-specific knowledge. ISCX labels are inferred from file names heuristically (`confidence="heuristic"` in `labels.csv`). D3 is registration-gated and intentionally not automated.

Datasets: D1 = CESNET-TLS-Year22 (primary, temporal splits by ISO week), D2 = CESNET-QUIC22, D3 = ISCX VPN-nonVPN 2016 (grouped-by-file CV), D4 = USTC-TFC2016 (anomaly). `docs/datasets/<name>.md` records each one's real status; trust `data/manifest.json` over specs for what actually exists.

## Testing notes

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
