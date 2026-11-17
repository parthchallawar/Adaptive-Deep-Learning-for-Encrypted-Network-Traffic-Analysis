# Kaggle workflow

Used as a workaround for the Claude Code Kaggle connector's OAuth failure
(`client_secret_basic authentication requires a client_secret`). This drives
Kaggle directly via its CLI instead: upload `data/processed/` as a dataset,
push `kernel/` to run training on Kaggle's hosted GPUs, then pull results
back down.

The training strategy that sits on top of this workflow (quotas, run sizing,
resumable kernels, the two data routes) is specified in
[`specs/015-kaggle-training-pipeline.md`](../specs/015-kaggle-training-pipeline.md).

## Setup

**Status (2026-09-22): authenticated by OAuth.** Kaggle CLI 2.2.4 is installed and
`kaggle auth login` has cached credentials in `~/.kaggle/credentials.json` for account
`parthchallawar`. The metadata files carry that account's slugs, so the steps below are
for reference or for setting this up on another machine.

**Trap (hit on 2026-09-21):** a legacy `~/.kaggle/kaggle.json` that is no longer valid
takes precedence over the OAuth login. Public downloads still work with it, but every
private call (`datasets status`, `kernels list --mine`, any push) answers "Authentication
required". If that happens, move the old file away, then run `./scripts/kaggle_sync.sh check`.

```
pip install kaggle
```

Either `kaggle auth login` (OAuth, preferred), or create an API token at Kaggle →
Settings → API → Create New API Token and place it at `~/.kaggle/kaggle.json`
(`%USERPROFILE%\.kaggle\kaggle.json` on Windows). Never commit credentials — both are
excluded via `.gitignore`.

Verify it's working:

```
./scripts/kaggle_sync.sh check
```

## Two routes for getting data onto Kaggle

1. **Mount a public mirror and prepare shards on Kaggle (preferred for the
   CESNET data).** `pranjalkar99/cesnet-22` carries the full 53-week
   CESNET-TLS-Year22 release and `zilinpeng/cesnet-quic22` carries QUIC22. A
   **CPU** kernel mounts one of them, runs the shard exporter with `--verify`,
   and its saved output becomes the dataset that training kernels mount. CPU
   kernels don't consume the GPU quota, and nothing is downloaded or uploaded
   from this machine.
2. **Push locally built shards (needed for the PCAP datasets).** The flow below.

## Dataset: data/processed/ → Kaggle

`data/processed/dataset-metadata.json` holds the dataset's `title`/`id`
(already set to `parthchallawar/adl-encrypted-traffic-processed`).

```
./scripts/kaggle_sync.sh push-dataset          # first upload only
./scripts/kaggle_sync.sh version-dataset "msg" # subsequent updates
```

## Kernel: kernel/ → Kaggle

`kernel/kernel-metadata.json` defines the kernel (id, GPU/internet flags,
which dataset(s) it mounts): `dataset_sources` are the D1 export kernel's own
output (`parthchallawar/adl-export-d1`) and the code dataset
(`parthchallawar/adl-encrypted-traffic-code`, from `push-code` below).
`kernel/kernel.py` (plan T8) works through `kernel/run_queue.yaml` in order,
training each entry with `adl_etc.training.run.run_training` under an 11.5h
wall-clock guard. Within a session, a paused or finished entry resumes or is
skipped via its run's own `state.json`. **Across sessions** that only works if
the previous session's output is mounted: see "Resuming across sessions" below.
Each entry trains on the split its own config names (`d1_open.yaml` for
phase 3), never a hard-coded one.
`ADL_TRAIN_MODE=smoke` (the default is `queue`) runs the same code on
~5,000 synthetic flows with no Kaggle mount needed, in well under 2 minutes —
`tests/test_kernel_smoke.py` runs this locally on every `pytest`.

If kernels can't use internet on this account, push the code as a dataset with
`./scripts/kaggle_sync.sh push-code` and let `kernel.py` add it to `sys.path`
instead of pip-installing from GitHub.

```
./scripts/kaggle_sync.sh push-kernel              # kernel/  (training)
./scripts/kaggle_sync.sh push-kernel kernel/export-d1   # the D1 export kernel (CPU)
```

`kernel/export-d1/` is the CPU-only export of CESNET-TLS-Year22 weeks 11-52 (3% uniform
sample, every day checked against its `stats-*.json`, Standardizer fit on weeks 11-26, the
real `d1_main.yaml` loaded). It runs first as a three-day probe (`ADL_EXPORT_MODE`
defaults to `probe` in the file), which reports speed, size and whether a CLI-pushed
kernel has internet; switch the default to `full` for the real run.
`tests/test_kernel_export_d1.py` runs the whole script against a synthetic mirror.

`./scripts/kaggle_sync.sh check` needs private-endpoint access, so it fails when the
token is stale even though public downloads still work; fix with `kaggle auth login`.

## Resuming across sessions (plan phase 3, T2)

Every `kaggle kernels push` starts with an empty `/kaggle/working`, so a new
session cannot see the previous one's run directories on its own (phase 2 saw a
finished seed retrained from scratch). At startup the queue kernel looks for run
directories (named like runs, holding a `state.json`) anywhere under
`/kaggle/input`, and copies each one its queue needs into place before deciding
what to do: a finished run is skipped, a paused one continues from its last
checkpoint. A `state.json` that cannot be read, or that names a missing
checkpoint, stops the kernel rather than being silently retrained over. Two ways
to mount the previous session's output:

1. **The kernel's own previous output** (first choice): add the training
   kernel's own slug, `parthchallawar/adl-encrypted-traffic-train`, to
   `kernel_sources` in `kernel/kernel-metadata.json`. `kernel_sources` mounts a
   kernel's *last successful* version, so this works whenever the previous
   session ended cleanly (a time-guard pause does). **Verified 2026-11-14**
   (phase 3, Session 2, kernel v15): Kaggle accepts a kernel listing itself; the
   mount appears at `/kaggle/input/adl-encrypted-traffic-train/results/`, all 8
   of Session 1's run directories were found, and the SSL run was restored and
   skipped. Only the *latest* version is mounted, so runs spread over several
   sessions need option 2.
2. **A dataset made from it** (fallback, needs nothing new from Kaggle):
   `pull-results` the previous session, version the pulled `results/` folder as
   a small private dataset, and add that dataset to `dataset_sources`.

Tested locally (`tests/test_kernel_smoke.py`): a finished run restored and
skipped, a paused run restored and continued to the end, a damaged one refused.

## Evaluation-only entries (plan phase 3, T8)

A queue entry can evaluate an already-trained run instead of training one:

```yaml
- eval: configs/eval/d1_open_robust.yaml
  run: ft-pat-scratch-d1open-s0
  overrides: ["device=cuda"]
```

The run directory is looked up in this session's `results/`, then in any mounted
input (anything holding a `state.json` and a `best.pt`). Output goes to
`results/robust/<run>/eval/<split>/`; an entry whose `robustness.json` (or
`report.json`, for a config without `perturb:`) already exists is skipped. To
make runs from several sessions available at once, ship them as a slim dataset
and add `parthchallawar/adl-encrypted-traffic-checkpoints` to `dataset_sources`:

```
./scripts/kaggle_sync.sh push-checkpoints results/<pulled>/<run> [...]
```

It copies only `best.pt`, `config_resolved.yaml`, `label_space.json`,
`calibration.json` and `state.json` (~15 MB a run), never `logits.npy`.

**Pulling output.** Kaggle's output download is slow (~0.6 MB/s measured
2026-11-14) and drops connections on large files (`IncompleteRead`), and a
training run's `eval/val/logits.npy` is ~2.25 GB. Pull the small files first
and large ones only when needed: `kaggle kernels output <user>/<slug> -p <dir>
--file-pattern '\.(json|jsonl|yaml|log|png)$' --page-size 200`, then
`'best\.pt$'`, retrying on `IncompleteRead`. While a newer version of the
kernel is running, its earlier versions' output cannot be listed or pulled.
`kaggle kernels logs <slug>` prints nothing for a running session; add `-f`
to stream it.

## Pulling a file from an older kernel version

`kaggle kernels output <owner>/<slug>/<version>` parses the version and then drops it, so it
always returns the **latest** version's files, and the API's listing endpoints return 404 for any
explicit version. That made an older version's checkpoints look lost once a newer version existed
(2026-11-17, phase 4 T0). The download request does take a version number, so name the files:

```
python scripts/kaggle_pull_version.py parthchallawar/adl-encrypted-traffic-train 16 \
    results/<run>/best.pt --out results/<where>
```

Run it with the Python that has the `kaggle` package (the project `.venv` does not). Verify what
you pull (for a PAT checkpoint, its SHA-256 must equal `checkpoint_hash` in the run's
`calibration.json`). `push-checkpoints` is the durable fix: it rebuilds the checkpoint dataset
from **exactly** the runs named, so always name every run you want kept.

## Results: Kaggle → results/

Once the kernel finishes running on Kaggle, pull its output down:

```
./scripts/kaggle_sync.sh pull-results <your-kaggle-username>/<kernel-slug>
```

Output lands in `results/`, which is gitignored except for summaries (see
`.gitignore`).
