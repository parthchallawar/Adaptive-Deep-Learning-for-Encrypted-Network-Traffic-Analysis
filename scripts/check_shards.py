"""Check that a shard export is complete: every period's arrays exist and have exactly the size
their ``meta.json`` implies. Run after every pull from Kaggle.

    python scripts/check_shards.py --root results/kaggle-full/shards --dataset cesnet-tls-year22

Exit code 1 if anything is missing or the wrong size (and each problem is printed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adl_etc.data.shard_check import check_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", default="cesnet-tls-year22")
    args = parser.parse_args(argv)
    dataset_dir = args.root / args.dataset
    if not dataset_dir.is_dir():
        print(f"no such directory: {dataset_dir}", file=sys.stderr)
        return 2
    n_periods = sum(1 for p in dataset_dir.iterdir() if p.is_dir())
    flows, problems = check_dataset(dataset_dir)
    for p in problems:
        print(p)
    bad_periods = len({p.period for p in problems})
    print(
        f"{n_periods - bad_periods}/{n_periods} periods healthy, {flows:,} flows in them"
        + (f"; {len(problems)} problem(s) in {bad_periods} period(s)" if problems else "")
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
