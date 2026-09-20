#!/usr/bin/env python
"""Dispatches to the per-dataset downloaders (spec 001).

    python scripts/download_all.py --datasets d3 d4 [--dry-run] [--files GLOB]

D1/D2 (CESNET) are not dispatched here: they go through
``scripts/export_raw_csv.py`` (spec 001, Path B) or the DataZoo API (Path A),
since the raw release is too large to mirror file-by-file the way D3/D4 are.
D3 (ISCX) additionally needs a one-time manual registration; see
``adl_etc.data.iscx_download``'s module docstring and pass ``--iscx-base-url``
here (or run ``download_iscx.py`` directly with ``--files-from``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adl_etc.data import iscx_download, ustc_download
from adl_etc.data import manifest as M
from adl_etc.data.download import DownloadError

DATASETS = ("d3", "d4")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--out-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=M.DEFAULT_MANIFEST_PATH)
    parser.add_argument("--files", default=None, help="glob applied to every requested dataset")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--iscx-base-url", default=None, help="see download_iscx.py --base-url")
    parser.add_argument("--iscx-files-from", type=Path, default=None)
    args = parser.parse_args(argv)

    exit_code = 0
    for name in args.datasets:
        print(f"=== {name} ===")
        try:
            if name == "d3":
                iscx_download.run(
                    base_url=args.iscx_base_url,
                    files_from=args.iscx_files_from,
                    out_dir=args.out_root / "iscx-vpn-2016",
                    manifest_path=args.manifest,
                    files_glob=args.files,
                    dry_run=args.dry_run,
                )
            elif name == "d4":
                ustc_download.run(
                    out_dir=args.out_root / "ustc-tfc2016",
                    manifest_path=args.manifest,
                    files_glob=args.files,
                    dry_run=args.dry_run,
                )
        except iscx_download.RegistrationRequired as e:
            print(f"error: {e}\nregister at: {iscx_download.REGISTRATION_URL}", file=sys.stderr)
            exit_code = 2
        except DownloadError as e:
            print(f"error: {e}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
