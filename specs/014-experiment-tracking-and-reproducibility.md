# Spec 014: Experiment Tracking, Configuration and Reproducibility

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 5 of 18 (moved ahead of 005: the first trained model must be tracked, or every baseline is rerun)
- **Depends on:** none. **Used by:** all training and evaluation specs; 015.

## Problem

The project will produce dozens of runs across two machines (laptop CPU, Kaggle GPU) over months. Without a strict scheme for configs, seeds, artefacts and result tables, the final comparison table cannot be trusted or regenerated, and the paper cannot be defended.

## Goals

- Every run fully described by one YAML config plus a git commit; every result traceable to both.
- MLflow as the single registry of runs, metrics, artefacts and model versions, working offline (file store) on Kaggle and locally, merged into one store. No server process is required (deployment is deferred, spec 019): a local file store plus the `mlflow ui` command when browsing is wanted.
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

- Local: file store at `results/mlruns` (`MLFLOW_TRACKING_URI=file:///.../results/mlruns`), browsed on demand with `mlflow ui --backend-store-uri file:results/mlruns`. A SQLite-backed tracking server is optional and only needed if the model registry API is used; if so, run `mlflow server --backend-store-uri sqlite:///results/mlflow.db --artifacts-destination results/mlruns` as a plain local process.
- Kaggle: `MLFLOW_TRACKING_URI=file:///kaggle/working/mlruns`; the run directory is pulled with `kaggle kernels output` and imported with `scripts/mlflow_import.py` (copies run folders and rewrites artifact URIs).
- Logged: params (flattened config), per-epoch metrics, final report JSON (spec 004), figures, checkpoint path, efficiency JSON (spec 013), dataset manifest hashes (spec 001).
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

- Dirty git tree at run time: allowed for development but flagged `git_dirty=true`; the final table script refuses runs with dirty trees unless `--allow-dirty`.
- Kaggle run interrupted: partial MLflow runs are imported with status `KILLED` and excluded from tables.
- Config drift (same run name, different config): the table script hashes resolved configs and errors on mismatch.

## Testing

- Config composition tests; `seed_everything` determinism test on a tiny model; MLflow import round-trip test on a fixture run.

## Interactions

- Used by every training/eval entry point; spec 015 handles the Kaggle side; spec 016 loads models from the registry.

## Success criteria

- `python scripts/make_tables.py` reproduces the main table from the store with zero manual edits.
- Any reported number can be traced to a run name, config hash and commit.

## Open questions

- Whether to use DVC for data versioning in addition to the manifest (default: no; manifest + Kaggle dataset versions are enough).
