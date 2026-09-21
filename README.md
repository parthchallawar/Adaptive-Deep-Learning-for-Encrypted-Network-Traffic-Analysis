# Adaptive Deep Learning for Encrypted Network Traffic Analysis

Research project applying adaptive deep learning techniques to the analysis of encrypted network traffic.

## Structure

See [CLAUDE.md](./CLAUDE.md) for the full project layout and working conventions.

```
.claude/          Claude Code config, subagents, commands
specs/            Feature specifications
plans/            Implementation plans
agents/           Project multi-agent role definitions
agent-memory/     Per-agent persistent memory logs
src/              Source code (data, models, training, evaluation, utils)
data/             Raw and processed datasets
notebooks/        Exploratory notebooks
tests/            Tests
configs/          Experiment configs
results/          Experiment outputs (metrics, checkpoints, plots)
scripts/          Utility/CLI scripts
docs/             Documentation
```

## Getting started

Requires Python 3.12+.

```
python -m venv .venv
.venv\Scripts\activate        # Windows; `source .venv/bin/activate` on Linux/macOS
pip install -e ".[dev]"
```

Run the checks:

```
pytest
ruff check src/ tests/ scripts/
mypy src/
```

Datasets are downloaded and exported via `scripts/download_*.py` and
`scripts/export_*.py`; see [`docs/datasets/`](./docs/datasets/) for what
each one actually is, its license, and its real download/processing
status, and [`data/manifest.json`](./data/manifest.json) for what has
been downloaded and verified so far. The Kaggle CLI (with
`~/.kaggle/kaggle.json`) is needed for the CESNET (D1/D2) mirrors — see
[`docs/kaggle-workflow.md`](./docs/kaggle-workflow.md).

## Status

Phase 1 (the data pipeline, specs 001-004) is done: PCAP-to-shard and
CESNET-CSV-to-shard exporters, the tokeniser/feature layer, evaluation
metrics, and split-loading with leakage checks are all built and tested.
Real data exists for D3 (ISCX) and D4 (USTC); D1 (CESNET, the primary
training dataset) has one real day exported as a proof of correctness,
with the full year still queued as a Kaggle-kernel job. See
[`plans/phase-1-data-pipeline.md`](./plans/phase-1-data-pipeline.md) for
the full build log and exit-criteria status, and
[`plans/README.md`](./plans/README.md) for every phase.
