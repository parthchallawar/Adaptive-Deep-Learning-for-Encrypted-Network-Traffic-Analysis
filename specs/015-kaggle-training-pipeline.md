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

- Whether kernels on this account may enable internet (phone verification). Not blocking: the code-as-dataset route and the mirror-mount route both work without it. To be settled by the first kernel push.
- Preferred GPU (T4 x2 to run two seeds concurrently vs P100 single): decide after measuring throughput in the first smoke run.
