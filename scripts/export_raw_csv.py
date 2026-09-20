#!/usr/bin/env python
"""CLI wrapper around adl_etc.data.cesnet_csv (spec 001 Path B, plan T5b).

    python scripts/export_raw_csv.py \
        --files data/raw/cesnet-tls-year22/flows-20220101.csv.xz ... \
        --out-root data/processed --dataset cesnet-tls-year22

    python scripts/export_raw_csv.py --verify \
        --files data/raw/cesnet-tls-year22/flows-20220101.csv.xz ...

Reads `flows-YYYYMMDD.csv.xz` files from the `pranjalkar99/cesnet-22` Kaggle
mirror (or an equivalent local layout) and writes shards partitioned by ISO
week. `--verify` runs spec 001's per-day checks against each file's sibling
`stats-YYYYMMDD.json` instead of exporting. See
`adl_etc.data.cesnet_csv` and `docs/datasets/cesnet-tls-year22.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from adl_etc.data import cesnet_csv as C
from adl_etc.data import manifest as M


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--files", nargs="+", required=True, type=Path, help="flows-YYYYMMDD.csv.xz file(s)"
    )
    parser.add_argument("--dataset", default="cesnet-tls-year22")
    parser.add_argument("--out-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest", type=Path, default=M.DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="JSON app -> int, for a stable corpus-wide id space",
    )
    parser.add_argument("--category-map", type=Path, default=None, help="JSON category -> int")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-flows", type=int, default=500_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--register", action="store_true", help="hash the input files into data/manifest.json"
    )
    parser.add_argument(
        "--verify", action="store_true", help="run spec 001's checks instead of exporting"
    )
    args = parser.parse_args(argv)

    if args.verify:
        ok = True
        for path in args.files:
            result = C.verify_day(path)
            status = "OK" if result.ok else "FAILED"
            print(f"{path}: {status}")
            for problem in result.problems:
                print(f"  - {problem}")
            ok = ok and result.ok
        return 0 if ok else 1

    if args.register:
        C.register_manifest(args.files, dataset=args.dataset, manifest_path=args.manifest)

    label_map = (
        json.loads(args.label_map.read_text(encoding="utf-8")) if args.label_map else None
    )
    category_map = (
        json.loads(args.category_map.read_text(encoding="utf-8")) if args.category_map else None
    )

    C.export_dataset(
        csv_paths=args.files,
        dataset=args.dataset,
        out_root=args.out_root,
        label_map=label_map,
        category_map=category_map,
        chunksize=args.chunksize,
        max_flows=args.max_flows,
        manifest_path=args.manifest,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
