# Adaptive Deep Learning for Encrypted Network Traffic Analysis

Project guidance for Claude Code. Keep this file updated as conventions solidify.

## Project structure

- `specs/` — feature/requirement specifications. One file per feature, written before implementation.
- `plans/` — implementation plans derived from specs. One file per plan, updated as work progresses.
- `agents/` — role/persona definitions for the project's own multi-agent pipeline (e.g. data-agent, training-agent, evaluation-agent), distinct from `.claude/agents/` which holds Claude Code subagent tool definitions.
- `agent-memory/` — persistent memory logs written by each agent in `agents/`, one subfolder per agent.
- `.claude/` — Claude Code configuration: `settings.json`, custom `agents/` (Claude Code subagents), `commands/` (slash commands).
- `src/adl_etc/` — Python source (src-layout; the installable package is `adl_etc`): `data/`, `models/`, `training/`, `evaluation/`, `inference/`, `service/`, `dashboard/`, `utils/`.
- `data/` — `raw/` (untouched input) and `processed/` (feature-engineered/cleaned) datasets. Large files should not be committed — see `.gitignore`.
- `notebooks/` — exploratory analysis notebooks.
- `tests/` — unit/integration tests, mirroring `src/` layout.
- `configs/` — experiment/config files (e.g. YAML) for model and training runs.
- `results/` — metrics, logs, checkpoints, plots produced by experiments (gitignored by default except summaries).
- `scripts/` — one-off or CLI utility scripts (data download, preprocessing, etc).
- `docs/` — longer-form documentation.

## Workflow conventions

1. New feature/capability → write a spec in `specs/` first.
2. Before implementing → write a plan in `plans/` derived from the spec.
3. Multi-agent or long-running work → define the agent's role in `agents/` and let it persist findings/state in `agent-memory/<agent-name>/`.
4. Keep specs and plans updated as living documents, not one-off snapshots.

## Notes

- Research direction, novelty statement, dataset and Kaggle strategy: `docs/research-analysis.md`. Specs `specs/000`..`020` (index in `specs/README.md`) define every module; keep them in sync with the code.

- Kaggle CLI workflow (dataset upload, kernel push/pull for GPU training) — see `docs/kaggle-workflow.md` and `scripts/kaggle_sync.sh`.

## Environment and setup

- Python 3.12+, `pyproject.toml` (setuptools, src-layout). Phase 1 (data pipeline) installs with only the core dependencies — deliberately no `torch`, so it stays fast; phase-specific extras (`train`, `datazoo`, `service`, `capture`, `dev`) are declared in `pyproject.toml` and pulled in only when that phase starts.
- Setup: `python -m venv .venv`, then `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate`, then `pip install -e ".[dev]"`.
- Checks: `pytest`, `ruff check src/ tests/ scripts/`, `mypy src/`. All three must be clean before a task is considered done.
- Dataset downloaders/exporters live in `scripts/*.py` (thin CLI wrappers) over `src/adl_etc/data/*.py` (the actual logic) — see `docs/datasets/<name>.md` for each dataset's real status, and `data/manifest.json` for what's actually been downloaded and verified. Kaggle CLI credentials (`~/.kaggle/kaggle.json`, gitignored) are needed for the CESNET (D1/D2) mirrors.
- Current build status: `plans/phase-1-data-pipeline.md` (data pipeline, specs 001-004) — done as of 2026-09-21, one real-data milestone (D1's full-year Kaggle-kernel export) still pending. `plans/README.md` indexes every phase's plan.
