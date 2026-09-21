# Spec 014: Experiment Tracking, Configuration and Reproducibility

- **Status:** partially implemented (phase 2 plan T1-T2, 2026-09-21: config composition and hashing, seeding with independent streams, `RunInfo`/run names, the run-directory `Tracker`, the SQLite MLflow sync and `scripts/mlflow_import.py`, all built and tested; **not yet built:** `scripts/make_tables.py`, `plotstyle.py` and the model registry, plan T7/T9). MLflow design corrected from file store to run directory + SQLite, see the Correction under "MLflow"
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 5 of 18 (moved ahead of 005: the first trained model must be tracked, or every baseline is rerun)
- **Depends on:** none. **Used by:** all training and evaluation specs; 015.

## Problem

The project will produce dozens of runs across two machines (laptop CPU, Kaggle GPU) over months. Without a strict scheme for configs, seeds, artefacts and result tables, the final comparison table cannot be trusted or regenerated, and the paper cannot be defended.

## Goals

- Every run fully described by one YAML config plus a git commit; every result traceable to both.
- MLflow as the single local registry of runs, metrics and artefacts, fed from plain run directories that any machine (laptop, Kaggle) can write with no dependencies and that are merged into one store on import. No server process is required (deployment is deferred, spec 019): a local SQLite store plus the `mlflow ui` command when browsing is wanted.
- Deterministic training where feasible; documented nondeterminism otherwise.
- `results/summaries/` regenerated from MLflow by scripts, never edited by hand.

## Non-goals

- A hosted or containerised MLflow server; hyperparameter-optimisation frameworks.

## Design

### Configuration

- OmegaConf YAML under `configs/`: `data/`, `splits/`, `models/`, `pretrain/`, `finetune/`, `policy/`, `controller/`, `eval/`. Composition by `defaults:` lists and CLI overrides (`python -m src.training.finetune --config configs/finetune/pat_ssl.yaml seed=1 data.label_fraction=0.1`).
- The resolved config is saved with every run as `config_resolved.yaml`, together with `git_commit`, `git_dirty`, Python and library versions, and the hardware description.

### Naming

`run_name = <stage>-<model>-<variant>-<split>-s<seed>`, e.g. `ft-pat-ssl_npp_pfc-d1m3to6-s0`. Checkpoints `results/<run_name>/ckpt_epoch<N>.pt` and `best.pt`; large artefacts are gitignored, summaries are committed.

### MLflow

**Correction (2026-09-21, found while building it).** This section originally specified MLflow's *file store* (`results/mlruns`). MLflow 3.16 puts the filesystem backend in maintenance mode and **raises by default** ("set `MLFLOW_ALLOW_FILE_STORE=true` to opt out"). The original premise, that avoiding a server means using the file store, is also false: MLflow's SQLite backend needs no server process either. Separately, whether Kaggle's image has mlflow, and whether a kernel may pip-install it, was never established. The design is therefore two layers:

- **Run directory (any machine, no dependencies).** `adl_etc.utils.tracking.Tracker` writes `<run_dir>/{run.json, params.json, metrics.jsonl, config_resolved.yaml, artifacts/}` and imports nothing from mlflow, so a Kaggle kernel records its results whether or not mlflow is installed. Append-only metrics, write-once params, atomic status writes; a killed writer leaves at worst a truncated last metrics line, which the reader tolerates. Statuses `RUNNING`, `FINISHED`, `KILLED`, `PAUSED`; an exception exiting the `with` block marks the run `KILLED` and records why.
- **Local MLflow registry (`adl_etc.utils.mlflow_sync`).** `sqlite:///results/mlflow.db`, artifacts under `results/mlartifacts`, browsed with `mlflow ui --backend-store-uri sqlite:///results/mlflow.db`. `scripts/mlflow_import.py --src <dir>` finds every run directory under `<dir>` and creates or extends its MLflow run. It is idempotent: a run's identity is `sha256(run_name, config_hash, started_at_ms)` (tag `adl.run_key`), so re-importing adds nothing and a run PAUSED on Kaggle, resumed on a later push and pulled again extends the same MLflow run. MLflow has no `PAUSED`, so `PAUSED` and never-finalised runs import as `KILLED` with the real state in the `adl.status` tag; an older copy of a run directory is refused rather than synced over a newer one.
- Kaggle: the kernel writes run directories under `/kaggle/working/`; they are pulled with `kaggle kernels output` and imported as above. No `MLFLOW_TRACKING_URI` is set on Kaggle.
- Logged: params (flattened config, plus `run_name`, `seed`, `git_commit`, `git_dirty`, `config_hash` always), per-epoch metrics, final report JSON (spec 004), figures, checkpoint *path* (never the checkpoint), efficiency JSON (spec 013), dataset manifest hashes (spec 001).
- Model registry: `AnytimeETC-PAT` with stages `candidate`, `staging`, `production`; the service (spec 016) loads by stage name when a tracking server is running, and otherwise from a plain checkpoint path recorded in `results/models/production.json`. The service must work without MLflow running.

### Seeds and determinism

- `seed_everything(seed)` sets Python, NumPy, PyTorch (CPU/GPU) seeds and `torch.use_deterministic_algorithms(True, warn_only=True)`; data order derived from the seed; augmentations use a seeded generator on GPU.
- Documented nondeterminism: AMP with SDPA kernels on GPU can produce tiny run-to-run differences; the three-seed protocol absorbs this.

### Tables and figures

- `scripts/make_tables.py` queries MLflow, aggregates by run name pattern, computes mean ± std over seeds and paired bootstrap CIs (spec 004), writes Markdown and LaTeX tables and PNG/HTML figures to `results/summaries/`.
- Figures follow a single style file (`src/utils/plotstyle.py`): colour-blind-safe palette, consistent axis labels ("packets read K", "accuracy").

## Inputs and outputs

- Inputs: configs, git state, run outputs.
- Outputs: MLflow store, `results/<run>/`, `results/summaries/`.

## Edge cases

- Dirty git tree at run time: allowed for development but flagged `git_dirty=true`; the final table script refuses runs with dirty trees unless `--allow-dirty`. A Kaggle kernel has no `.git`, so `scripts/kaggle_sync.sh push-code` writes `GIT_COMMIT` and `GIT_DIRTY` (dirtiness of the *shipped* `src/` and `configs/` only) beside `src/`, and `RunInfo` reads them. When dirtiness cannot be established either way it is recorded as **dirty**: unknown provenance must not pass the clean-tree check. Resuming a run on different code (commit or dirtiness changed) or under a different config hash is refused.
- Kaggle run interrupted: a run directory that never reached a terminal status, and a `PAUSED` one, are imported with status `KILLED` (real state in tag `adl.status`) and excluded from tables until a later resume finishes them.
- Config drift (same run name, different config): the table script hashes resolved configs and errors on mismatch.

## Testing

- Config composition tests; `seed_everything` determinism test on a tiny model; run-directory round trip, resume and crash-truncation tests; MLflow sync idempotence, resume-extension and stale-copy tests on a SQLite store; a fresh-interpreter test that `tracking` never imports mlflow. Implemented in `tests/utils/`.

## Interactions

- Used by every training/eval entry point; spec 015 handles the Kaggle side; spec 016 loads models from the registry.

## Success criteria

- `python scripts/make_tables.py` reproduces the main table from the store with zero manual edits.
- Any reported number can be traced to a run name, config hash and commit.

## Open questions

- Whether to use DVC for data versioning in addition to the manifest (default: no; manifest + Kaggle dataset versions are enough).
