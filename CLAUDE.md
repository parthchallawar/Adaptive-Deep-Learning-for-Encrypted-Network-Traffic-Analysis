# Adaptive Deep Learning for Encrypted Network Traffic Analysis

Project guidance for Claude Code. Keep this file updated as conventions solidify.

## Project structure

- `specs/` — feature/requirement specifications. One file per feature, written before implementation.
- `plans/` — implementation plans derived from specs. One file per plan, updated as work progresses.
- `agents/` — role/persona definitions for the project's own multi-agent pipeline (e.g. data-agent, training-agent, evaluation-agent), distinct from `.claude/agents/` which holds Claude Code subagent tool definitions.
- `agent-memory/` — persistent memory logs written by each agent in `agents/`, one subfolder per agent.
- `.claude/` — Claude Code configuration: `settings.json`, custom `agents/` (Claude Code subagents), `commands/` (slash commands).
- `src/` — Python source: `data/`, `models/`, `training/`, `evaluation/`, `utils/`.
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

- Fill in setup/run instructions here once the environment (Python version, dependency manager, dataset sources) is decided.
