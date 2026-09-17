# Spec 015: Kaggle GPU Training Pipeline

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Depends on:** 003, 014. **Used by:** 005, 007, 008. **Related:** `docs/kaggle-workflow.md`, `scripts/kaggle_sync.sh`, `kernel/`.

## Problem

All GPU work runs on Kaggle's free tier. The constraints that shape the design (re-verify in the Kaggle UI before phase 2, since they change):

| Constraint | Value (Sept 2026, per Kaggle docs/announcements) |
|---|---|
| GPU quota | 30 h per week, shared across P100 and T4 x2 |
| Max session | 12 h for CPU/GPU notebooks ("Save & Run All" runs in background, no idle timeout, 12 h cap) |
| Interactive idle timeout | about 60 min |
| GPU | P100 16 GB, or 2 x T4 16 GB (fp16 tensor cores on T4) |
| RAM | about 29 GB in GPU sessions |
| Disk | 20 GB persisted in `/kaggle/working`; datasets mounted read-only under `/kaggle/input` |
| Datasets | private quota 200 GB; a dataset version is immutable |
| Internet | must be enabled per notebook; requires a phone-verified account |

A training job that assumes more than this (long sessions, workers, big models) will fail or burn the quota. The pipeline must be resumable, quota-aware, and reproducible from the repository.

## Goals

- One `kernel/kernel.py` entry point that installs the repo at a pinned commit, loads pre-tensorised shards from `/kaggle/input`, runs a named config, checkpoints every epoch, resumes automatically, and writes MLflow + report artefacts to `/kaggle/working`.
- Job sizes that fit in one session with >= 25% margin; longer jobs split into resumable stages (SSL epochs 1 to 5, 6 to 10).
- A weekly GPU budget plan and a run queue tracked in `plans/`.
- Local dry-run mode (`--smoke`) that runs the same code on CPU with 5k flows in under 2 minutes.

## Non-goals

- Multi-GPU data parallelism (models are small; the second T4 is used only to run two independent seeds concurrently when the session is otherwise idle).
- Kaggle competitions/notebooks as the source of truth; the repository is.

## Design

### Data packaging

- `scripts/export_datazoo.py` and the PCAP pipeline write shards (spec 003) locally; `scripts/kaggle_sync.sh push-dataset` uploads `data/processed/` as the private dataset `adl-encrypted-traffic-processed` (about 4 GB for D1 XS + D2 XS). New shards = new dataset version; the kernel pins the version.
- Kernel loads shards with `np.load(mmap_mode="r")` then copies to RAM (`np.ascontiguousarray`) once; batches are index slices; augmentations on GPU.

### Kernel structure

```
kernel/
  kernel-metadata.json     # id, enable_gpu, enable_internet, dataset_sources (pinned), kernel_sources
  kernel.py                # bootstrap: pip install git+<repo>@<commit>; run_config(CONFIG, overrides)
  run_queue.yaml           # ordered list of configs to run in this push (stops when < 40 min left)
```

`kernel.py` reads `KAGGLE_RUN_TIME_LIMIT` (11.5 h guard), checks `/kaggle/working/state.json` for a partially finished run, resumes from the last checkpoint, and after each epoch estimates remaining time; if the next epoch would exceed the guard it saves and exits cleanly with status `PAUSED` so that the next push resumes.

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
- `pip install` from GitHub fails (internet disabled): kernel falls back to a `src/` copy attached as a Kaggle "utility script" or dataset; `kaggle_sync.sh push-code` creates it.
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

- Is the owner's Kaggle account phone-verified (internet-enabled kernels)? If not, the "utility script" code path is the default.
- Preferred GPU (T4 x2 to run two seeds concurrently vs P100 single): decide after measuring throughput in the first smoke run.
