# Kaggle workflow

Used as a workaround for the Claude Code Kaggle connector's OAuth failure
(`client_secret_basic authentication requires a client_secret`). This drives
Kaggle directly via its CLI instead: upload `data/processed/` as a dataset,
push `kernel/` to run training on Kaggle's hosted GPUs, then pull results
back down.

## Setup

```
pip install kaggle
```

Create an API token at Kaggle → Settings → API → Create New API Token, and
place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json` (`%USERPROFILE%\.kaggle\kaggle.json`
on Windows). Never commit this file — it's already excluded via `.gitignore`.

Verify it's working:

```
./scripts/kaggle_sync.sh check
```

## Dataset: data/processed/ → Kaggle

`data/processed/dataset-metadata.json` holds the dataset's `title`/`id`.
Edit `id` to `<your-kaggle-username>/<slug>` before the first push.

```
./scripts/kaggle_sync.sh push-dataset          # first upload only
./scripts/kaggle_sync.sh version-dataset "msg" # subsequent updates
```

## Kernel: kernel/ → Kaggle

`kernel/kernel-metadata.json` defines the kernel (id, GPU/internet flags,
which dataset(s) it mounts). Edit `id` and `dataset_sources` to match your
username and the dataset pushed above. `kernel/kernel.py` is the script
Kaggle runs — wire it up to `src/training` once a training entry point
exists.

```
./scripts/kaggle_sync.sh push-kernel
```

## Results: Kaggle → results/

Once the kernel finishes running on Kaggle, pull its output down:

```
./scripts/kaggle_sync.sh pull-results <your-kaggle-username>/<kernel-slug>
```

Output lands in `results/`, which is gitignored except for summaries (see
`.gitignore`).
