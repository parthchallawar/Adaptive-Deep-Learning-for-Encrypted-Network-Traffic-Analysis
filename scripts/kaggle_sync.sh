#!/usr/bin/env bash
# Wrapper around the Kaggle CLI for this project's dataset/kernel workflow.
# See docs/kaggle-workflow.md for full setup and usage notes.
set -euo pipefail

DATASET_DIR="data/processed"
KERNEL_DIR="kernel"
RESULTS_DIR="results"

usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [args]

Commands:
  check                    Verify the kaggle CLI is installed and authenticated
  push-dataset             Create/upload data/processed as a Kaggle dataset
  version-dataset <msg>    Push a new version of an existing dataset
  push-kernel              Push kernel/ to Kaggle (kernel-metadata.json required)
  pull-results <user>/<slug>   Download kernel output into results/
EOF
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
  kaggle datasets list -s test-connection >/dev/null 2>&1 || {
    echo "kaggle.json is present but authentication failed." >&2
    echo "Regenerate the token from Kaggle → Settings → API and try again." >&2
    exit 1
  }
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

push_kernel() {
  [ -f "$KERNEL_DIR/kernel-metadata.json" ] || {
    echo "Missing $KERNEL_DIR/kernel-metadata.json — set id/title/code_file before pushing." >&2
    exit 1
  }
  kaggle kernels push -p "$KERNEL_DIR"
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
  push-kernel) push_kernel ;;
  pull-results) pull_results "$@" ;;
  *) usage; exit 1 ;;
esac
