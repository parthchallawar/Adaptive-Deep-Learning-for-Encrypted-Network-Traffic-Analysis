#!/usr/bin/env python
"""Import run directories (e.g. pulled from a Kaggle kernel) into local MLflow.

    python scripts/mlflow_import.py --src results/kaggle-out
    python scripts/mlflow_import.py --src results/kaggle-out --dry-run

Finds every directory under ``--src`` that holds a ``run.json`` (written by
``adl_etc.utils.tracking.Tracker``) and creates or extends its MLflow run in
``<results>/mlflow.db``. Safe to repeat: importing the same directory twice
leaves one run, and a run that was resumed on Kaggle and pulled again is
extended rather than duplicated. Runs that never reached FINISHED are imported
as KILLED (spec 014) and excluded from the results tables.

Browse with:  mlflow ui --backend-store-uri sqlite:///results/mlflow.db
See ``adl_etc.utils.mlflow_sync`` for why this is SQLite and not the file store.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from adl_etc.utils import mlflow_sync as S  # noqa: E402
from adl_etc.utils.tracking import TrackingError, read_run_dir  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--src", required=True, type=Path, help="directory to search for run dirs")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="results dir holding mlflow.db and mlartifacts/ (default: results)",
    )
    parser.add_argument("--experiment", default=S.DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be imported; touch nothing (does not import mlflow)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="leave RUNNING runs RUNNING instead of importing them as KILLED",
    )
    args = parser.parse_args(argv)

    if not args.src.is_dir():
        print(f"error: --src {args.src} is not a directory", file=sys.stderr)
        return 2

    run_dirs = S.find_run_dirs(args.src)
    if not run_dirs:
        print(f"no run directories (run.json) found under {args.src}")
        return 0

    if args.dry_run:
        bad = 0
        for rd in run_dirs:
            try:
                rec = read_run_dir(rd)
            except TrackingError as e:
                bad += 1
                print(f"UNREADABLE  {rd}: {e}")
                continue
            print(f"{rec.status:9s}  {rec.info.run_name}  metrics={len(rec.metrics)}  {rd}")
        print(f"{len(run_dirs)} run dir(s), {bad} unreadable (dry run: nothing imported)")
        return 1 if bad else 0

    uri, artifact_root = S.local_store(args.results)
    args.results.mkdir(parents=True, exist_ok=True)
    results = S.sync_tree(
        args.src,
        tracking_uri=uri,
        artifact_root=artifact_root,
        experiment=args.experiment,
        finalize_running=not args.live,
    )

    failures = 0
    for item in results:
        if isinstance(item, S.SyncResult):
            verb = "created" if item.created else "updated"
            print(
                f"{verb:8s}  {item.adl_status:9s} -> {item.mlflow_status:8s}  "
                f"+{item.metrics_logged} metrics  +{item.artifacts_logged} artifacts  "
                f"{item.run_dir}"
            )
        else:
            failures += 1
            print(f"FAILED    {item[0]}: {item[1]}", file=sys.stderr)
    print(f"{len(results) - failures} synced, {failures} failed  ({uri})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
