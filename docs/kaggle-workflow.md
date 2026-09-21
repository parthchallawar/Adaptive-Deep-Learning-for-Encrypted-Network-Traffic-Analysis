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
which dataset(s) it mounts); `id` and `dataset_sources` are already set for
this account. `kernel/kernel.py` is the script Kaggle runs — wire it up to
`src/training` once a training entry point exists.

If kernels can't use internet on this account, push the code as a dataset with
`./scripts/kaggle_sync.sh push-code` and let `kernel.py` add it to `sys.path`
instead of pip-installing from GitHub.

```
./scripts/kaggle_sync.sh push-kernel              # kernel/  (training)
./scripts/kaggle_sync.sh push-kernel kernel/export-d1   # the D1 export kernel (CPU)
```

`kernel/export-d1/` is the CPU-only export of CESNET-TLS-Year22 weeks 11-52 (10% uniform
sample, every day checked against its `stats-*.json`, Standardizer fit on weeks 11-26, the
real `d1_main.yaml` loaded). It runs first as a three-day probe (`ADL_EXPORT_MODE`
defaults to `probe` in the file), which reports speed, size and whether a CLI-pushed
kernel has internet; switch the default to `full` for the real run.
`tests/test_kernel_export_d1.py` runs the whole script against a synthetic mirror.

`./scripts/kaggle_sync.sh check` needs private-endpoint access, so it fails when the
token is stale even though public downloads still work; fix with `kaggle auth login`.

## Results: Kaggle → results/

Once the kernel finishes running on Kaggle, pull its output down:

```
./scripts/kaggle_sync.sh pull-results <your-kaggle-username>/<kernel-slug>
```

Output lands in `results/`, which is gitignored except for summaries (see
`.gitignore`).
