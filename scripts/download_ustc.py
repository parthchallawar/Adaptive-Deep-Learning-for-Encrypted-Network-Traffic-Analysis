#!/usr/bin/env python
"""CLI wrapper around adl_etc.data.ustc_download (spec 001, D4).

    python scripts/download_ustc.py [--out DIR] [--files GLOB] [--dry-run]

Fully automated: no registration needed. See ustc_download.py's module
docstring for what this actually does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adl_etc.data import manifest as M
from adl_etc.data.download import DownloadError
from adl_etc.data.ustc_download import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/ustc-tfc2016"))
    parser.add_argument("--manifest", type=Path, default=M.DEFAULT_MANIFEST_PATH)
    parser.add_argument("--files", default=None, help="glob to limit which archives are fetched")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        run(
            out_dir=args.out,
            manifest_path=args.manifest,
            files_glob=args.files,
            dry_run=args.dry_run,
        )
    except DownloadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
