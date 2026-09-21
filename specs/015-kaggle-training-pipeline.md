# Spec 015: Kaggle GPU Training Pipeline

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 6 of 18 (moved ahead of 005, same reason as 014)
- **Depends on:** 003, 014. **Used by:** 005, 007, 008. **Related:** `docs/kaggle-workflow.md`, `scripts/kaggle_sync.sh`, `kernel/`.

## Problem

All GPU work runs on Kaggle's free tier. The constraints that shape the design (re-verify in the Kaggle UI before phase 2, since they change):

| Constraint | Value (Sept 2026, per Kaggle docs/announcements) |
|---|---|
| GPU quota | 30 h per week, shared across P100 and T4 x2. **Verified 2026-09-21** on the account's Quotas page: 30 hrs (00:00 used). The page shows the total but not the reset period, so "per week" is still Kaggle's documented behaviour, not observed |
| Max session | 12 h for CPU/GPU notebooks ("Save & Run All" runs in background, no idle timeout, 12 h cap). **Verified 2026-09-21:** the interactive session panel shows "12 hours" as the session maximum |
| Interactive idle timeout | about 60 min |
| GPU | P100 16 GB, or 2 x T4 (fp16 tensor cores on T4). **Verified 2026-09-21:** a "GPU T4 x2" session shows two GPUs at 15 GiB each; the P100 option was not checked. Note a GPU session counts against the quota while it is open, even when idle |
| RAM | about 29 GB in GPU sessions. **Verified 2026-09-21:** max 30 GiB |
| Disk | 20 GB persisted in `/kaggle/working` (stated in the notebook template itself); datasets mounted read-only under `/kaggle/input`. **Verified 2026-09-21:** the session panel shows 57.6 GiB max total disk, so the 20 GB limit is on saved *output*, and scratch beyond it can live in `/kaggle/temp/` |
| Datasets | private quota **214.75 GB** (= 200 GiB; verified 2026-09-21, 0 B used, so no private dataset has been uploaded yet); a dataset version is immutable. Private models: same 214.75 GB, unused. TPU 20 h, unused, irrelevant |
| Internet | must be enabled per notebook; requires a phone-verified account. **Partly verified 2026-09-21:** in an interactive notebook the Settings menu offers "Turn off internet", i.e. internet is currently on for this account. Not yet verified: that a CLI-pushed kernel gets it, and which packages the image already has |

### What the Kaggle image actually has (verified 2026-09-21, interactive GPU T4 x2 notebook)

Python 3.12.13, torch 2.10.0+cu128, numpy 2.0.2, pandas 2.3.3, pyarrow 24.0.0, scikit-learn 1.6.1, xgboost 3.2.0, omegaconf 2.3.0, tqdm 4.67.3. **Not installed: `mlflow`, `dpkt`, `py7zr`.** Outbound HTTPS works (pypi.org and github.com both returned 200). `/kaggle/working` had 19.5 GiB free, matching the 20 GB output cap. Two consequences:

- **The image is older than the dev machine** (local: numpy 2.5.3, pandas 3.0.6, pyarrow 25, torch 2.14). `pyproject.toml`'s floors (numpy >= 2.0, pandas >= 2.2, pyarrow >= 16, torch >= 2.4, scikit-learn >= 1.5, xgboost >= 2.1) all hold. The full suite was run in a throwaway venv pinned to Kaggle's exact versions (no torch, no mlflow, as on Kaggle): 453 passed, 16 skipped, and the skips are exactly the tests that need torch or mlflow. Re-run this check when the image changes or a new dependency is added.
- **Nothing on the training path needs the missing packages.** Every module a kernel imports (`data/{ppi,flows,tensors,features,prefix_stats,manifest,cesnet_csv}`, `evaluation/*`, `utils/*`) imports with `dpkt`, `py7zr` and `mlflow` blocked. Only the PCAP and downloader modules need `dpkt`/`py7zr`. This is also why `tracking.py` never imports mlflow (spec 014).

Not yet verified: that a kernel pushed with the CLI gets internet the way an interactive notebook does, and the P100 option.

A training job that assumes more than this (long sessions, workers, big models) will fail or burn the quota. The pipeline must be resumable, quota-aware, and reproducible from the repository.

## Goals

- One `kernel/kernel.py` entry point that makes the repo importable at a pinned commit, loads pre-tensorised shards from `/kaggle/input`, runs a named config, checkpoints every epoch, resumes automatically, and writes MLflow + report artefacts to `/kaggle/working`.
- Job sizes that fit in one session with >= 25% margin; longer jobs split into resumable stages (SSL epochs 1 to 5, 6 to 10).
- A weekly GPU budget plan and a run queue tracked in `plans/`.
- Local dry-run mode (`--smoke`) that runs the same code on CPU with 5k flows in under 2 minutes.

## Non-goals

- Multi-GPU data parallelism (models are small; the second T4 is used only to run two independent seeds concurrently when the session is otherwise idle).
- Kaggle competitions/notebooks as the source of truth; the repository is.

## Design

### Credentials (confirmed 2026-09-17)

`~/.kaggle/kaggle.json` holds the token for account `parthrchallawar`; Kaggle CLI 2.2.4 is installed and authenticates. `data/processed/dataset-metadata.json` and `kernel/kernel-metadata.json` carry the real slugs (`parthrchallawar/adl-encrypted-traffic-processed`, `parthrchallawar/adl-encrypted-traffic-train`). The token is gitignored and must never be printed or committed.

### Data packaging

Two routes, matching spec 001's acquisition paths:

- **Route B (default, no local download or upload):** a **CPU** kernel mounts the public mirror `pranjalkar99/cesnet-22` read-only, runs `scripts/export_raw_csv.py --verify`, and writes weekly shards to `/kaggle/working/`; saving that kernel produces an output dataset which training kernels then mount. CPU kernels do not consume the GPU quota, so the whole preparation costs zero GPU-hours. One export run per dataset; re-run only when the shard schema changes.
- **Route A (fallback):** shards are produced locally (`scripts/export_datazoo.py` or the PCAP pipeline) and pushed with `scripts/kaggle_sync.sh push-dataset` as the private dataset `adl-encrypted-traffic-processed` (about 4 GB for D1 XS + D2 XS). This is also the route for the PCAP-derived D3/D4 shards, which have no public mirror.

Either way the kernel pins the dataset *version*, and loads shards with `np.load(mmap_mode="r")` then copies to RAM (`np.ascontiguousarray`) once; batches are index slices; augmentations run on the GPU.

### Kernel structure

```
kernel/
  kernel-metadata.json     # id, enable_gpu, enable_internet, dataset_sources (pinned), kernel_sources
  kernel.py                # bootstrap: pip install git+<repo>@<commit>; run_config(CONFIG, overrides)
  run_queue.yaml           # ordered list of configs to run in this push (stops when < 40 min left)
```

`kernel.py` reads `KAGGLE_RUN_TIME_LIMIT` (11.5 h guard), checks `/kaggle/working/state.json` for a partially finished run, resumes from the last checkpoint, and after each epoch estimates remaining time; if the next epoch would exceed the guard it saves and exits cleanly with status `PAUSED` so that the next push resumes.

**Code delivery.** Two supported ways, chosen by a kernel flag, so that the pipeline does not depend on whether the account may use internet in kernels:

1. `pip install git+https://github.com/<owner>/<repo>@<commit>` (needs internet enabled in the kernel).
2. The repository's `src/` tree pushed as a small private dataset by `scripts/kaggle_sync.sh push-code` and mounted alongside the data; `kernel.py` prepends it to `sys.path`. No internet needed.

Route 2 is the safer default until internet-in-kernels is confirmed to work; both pin the same commit hash, which is recorded in the run's config (spec 014).

### Training efficiency

- AMP fp16 (`torch.autocast` + `GradScaler`), `torch.backends.cuda.matmul.allow_tf32` irrelevant on T4/P100; channels-last not needed.
- Batch 4096 (SSL two views: 2 x 4096 sequences of 31 tokens): a few GB of activations; fits with margin.
- No DataLoader workers; a single producer thread prefetches index batches to GPU.
- `torch.compile` optional (first-epoch compile cost about 2 min; enabled for runs > 1 h).
- Expected throughput on T4 (to be measured in the first smoke run and recorded here): about 30k to 40k flows/s forward+backward for PAT → 3M flows/epoch in about 90 s of pure compute; real epochs 3 to 5 min including heads and augmentation. This means a 10-epoch SSL run on D1 XS train is well under 1 h, and the 12 h cap is not the binding constraint; the weekly 30 h is.

### Weekly budget plan (example for phase 3)

| Job | Est. GPU-h |
|---|---|
| SSL pretraining NPP+PFC, 10 epochs, 3 seeds | 3 |
| SSL ablations (NPP only, PFC only, MPM), 1 seed each | 2 |
| Fine-tuning PAT (SSL/scratch) x 3 seeds x {100%, 10%, 1%} labels | 6 |
| Baselines B4/B5 on GPU, 3 seeds | 3 |
| Evaluation passes (all K, all periods) | 2 |
| Slack for failures | 4 |
| **Total** | **20** of 30 |

### Reproducibility

- `kernel-metadata.json` pins `dataset_sources` versions; `kernel.py` pins the repo commit; configs are in the repo; seeds in the queue.
- Outputs (`/kaggle/working/mlruns`, `results/<run>/`) are pulled with `scripts/kaggle_sync.sh pull-results` and imported into local MLflow (spec 014).

## Inputs and outputs

- Inputs: Kaggle dataset version, repo commit, `run_queue.yaml`.
- Outputs: checkpoints, reports, MLflow run folders in `/kaggle/working`, pulled into `results/`.

## Edge cases

- Session killed before a checkpoint: lose at most one epoch; `state.json` written atomically (write temp + rename).
- `pip install` from GitHub fails (internet disabled): the kernel falls back to the mounted `src/` dataset described under Code delivery; `kaggle_sync.sh push-code` creates it.
- Mirror dataset withdrawn or altered by its uploader: `--verify` fails loudly and Route A takes over; this is the reason verification is mandatory rather than advisory.
- Quota exhausted mid-week: the queue is ordered by priority; CPU-runnable jobs (baselines B1 to B3) are never queued on Kaggle.
- Output > 20 GB: never store full logits on Kaggle (spec 008 keeps top-10 logits + scores).
- P100 vs T4 differences (fp16 speed, memory): configs are identical; throughput logged per run.

## Testing

- `python kernel/kernel.py --smoke` locally on CPU: runs a 2-epoch fine-tune on 5k flows and a 1-epoch SSL on 5k flows, produces a report; part of CI.
- Resume test: kill the smoke run after epoch 1, rerun, verify it resumes and metrics continue.

## Interactions

- Uses spec 003 shards, spec 014 tracking; runs specs 005, 007, 008 configs.

## Success criteria

- First real Kaggle run (baseline B5) completes end-to-end, results pulled and appear in local MLflow, within phase 2.
- No week exceeds the quota; every run is resumable.

## Open questions

- Whether kernels on this account may enable internet. **Partly answered 2026-09-21:** an interactive notebook has internet on and reaches pypi.org and github.com. Still to confirm: that a CLI-pushed kernel (`enable_internet: true` in `kernel-metadata.json`) gets it too. Not blocking: code-as-dataset needs none, and the only missing packages on the training path are ones it doesn't import.
- Preferred GPU (T4 x2 to run two seeds concurrently vs P100 single): T4 x2 is confirmed available (two 15 GiB GPUs, 30 GiB RAM); the P100 option has not been checked. Decide after measuring throughput in the first smoke run.
