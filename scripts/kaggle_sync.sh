#!/usr/bin/env bash
# Wrapper around the Kaggle CLI for this project's dataset/kernel workflow.
# See docs/kaggle-workflow.md for full setup and usage notes.
set -euo pipefail

DATASET_DIR="data/processed"
KERNEL_DIR="kernel"
RESULTS_DIR="results"
CODE_STAGE_DIR=".kaggle-code"

usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [args]

Commands:
  check                    Verify the kaggle CLI is installed and authenticated
  push-dataset             Create/upload data/processed as a Kaggle dataset
  version-dataset <msg>    Push a new version of an existing dataset
  push-code [msg]          Package src/ + configs/ as a Kaggle dataset so kernels
                           can import the project without internet access
  push-kernel [dir]        Push a kernel folder to Kaggle (default: kernel/;
                           e.g. kernel/export-d1). It needs a kernel-metadata.json
  pull-results <user>/<slug>   Download kernel output into results/
EOF
}

kaggle_username() {
  python -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'].lower())"
}

check() {
  command -v kaggle >/dev/null 2>&1 || {
    echo "kaggle CLI not found. Install with: pip install kaggle" >&2
    exit 1
  }
  [ -f "$HOME/.kaggle/kaggle.json" ] || {
    echo "Missing $HOME/.kaggle/kaggle.json. Create an API token at" >&2
    echo "Kaggle → Settings → API → Create New API Token, then place it there." >&2
    exit 1
  }
  # A public listing succeeds without credentials, so it proves nothing. `kernels list
  # --mine` needs them, and prints "Authentication required" (exit 0) when they are bad.
  out="$(kaggle kernels list --mine --page-size 1 2>&1)" || true
  if printf '%s' "$out" | grep -qi "authentication required"; then
    echo "The kaggle CLI is installed but not authenticated for private endpoints." >&2
    echo "Run: kaggle auth login   (or regenerate the token: Kaggle -> Settings -> API)." >&2
    exit 1
  fi
  echo "Kaggle CLI OK and authenticated."
}

push_dataset() {
  [ -f "$DATASET_DIR/dataset-metadata.json" ] || {
    echo "Missing $DATASET_DIR/dataset-metadata.json — set title/id before creating the dataset." >&2
    exit 1
  }
  kaggle datasets create -p "$DATASET_DIR"
}

version_dataset() {
  local msg="${1:?usage: version-dataset <message>}"
  kaggle datasets version -p "$DATASET_DIR" -m "$msg"
}

push_code() {
  local msg="${1:-code sync $(date -u +%Y-%m-%dT%H:%M:%SZ)}"
  local user slug
  user="$(kaggle_username)"
  slug="adl-encrypted-traffic-code"

  rm -rf "$CODE_STAGE_DIR"
  mkdir -p "$CODE_STAGE_DIR"
  # Ship only importable project code and configs — no data, no credentials.
  cp -r src "$CODE_STAGE_DIR/src"
  [ -d configs ] && cp -r configs "$CODE_STAGE_DIR/configs"
  find "$CODE_STAGE_DIR" -name '__pycache__' -type d -prune -exec rm -rf {} +
  # Provenance sidecars, read by adl_etc.utils.runinfo.code_provenance() since a
  # kernel has no .git. GIT_DIRTY covers only what is shipped (src/, configs/):
  # 1 if it differs from the commit, 0 if identical. When git can't say, both
  # files are left out and the run is treated as dirty rather than as clean.
  if git rev-parse HEAD > "$CODE_STAGE_DIR/GIT_COMMIT" 2>/dev/null; then
    if [ -n "$(git status --porcelain -- src configs)" ]; then
      echo 1 > "$CODE_STAGE_DIR/GIT_DIRTY"
    else
      echo 0 > "$CODE_STAGE_DIR/GIT_DIRTY"
    fi
  else
    rm -f "$CODE_STAGE_DIR/GIT_COMMIT"
  fi

  cat > "$CODE_STAGE_DIR/dataset-metadata.json" <<EOF
{
  "title": "ADL Encrypted Traffic - Code",
  "id": "$user/$slug",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

  if kaggle datasets status "$user/$slug" >/dev/null 2>&1; then
    kaggle datasets version -p "$CODE_STAGE_DIR" -m "$msg" --dir-mode zip
  else
    kaggle datasets create -p "$CODE_STAGE_DIR" --dir-mode zip
  fi
  echo "Code pushed as $user/$slug — mount it in the kernel and add it to sys.path."
}

push_kernel() {
  local dir="${1:-$KERNEL_DIR}"
  [ -f "$dir/kernel-metadata.json" ] || {
    echo "Missing $dir/kernel-metadata.json — set id/title/code_file before pushing." >&2
    exit 1
  }
  kaggle kernels push -p "$dir"
}

pull_results() {
  local slug="${1:?usage: pull-results <user>/<slug>}"
  mkdir -p "$RESULTS_DIR"
  kaggle kernels output "$slug" -p "$RESULTS_DIR"
}

cmd="${1:-}"
shift || true
case "$cmd" in
  check) check ;;
  push-dataset) push_dataset ;;
  version-dataset) version_dataset "$@" ;;
  push-code) push_code "$@" ;;
  push-kernel) push_kernel "$@" ;;
  pull-results) pull_results "$@" ;;
  *) usage; exit 1 ;;
esac
