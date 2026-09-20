#!/usr/bin/env python
"""CLI wrapper around adl_etc.data.iscx_download (spec 001, D3).

    python scripts/download_iscx.py --base-url <post-registration URL> [--files GLOB] [--dry-run]
    python scripts/download_iscx.py --files-from urls.txt [...]

**Requires a one-time manual registration first** — see
iscx_download.py's module docstring for why this project does not automate
that step, and what --base-url should point to once you have it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adl_etc.data import manifest as M
from adl_etc.data.download import DownloadError
from adl_etc.data.iscx_download import REGISTRATION_URL, RegistrationRequired, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default=None, help="the post-registration file-listing URL")
    parser.add_argument(
        "--files-from", type=Path, default=None, help="local file of pcap URLs, one per line"
    )
    parser.add_argument("--out", type=Path, default=Path("data/raw/iscx-vpn-2016"))
    parser.add_argument("--manifest", type=Path, default=M.DEFAULT_MANIFEST_PATH)
    parser.add_argument("--files", default=None, help="glob to limit which files are fetched")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        run(
            base_url=args.base_url,
            files_from=args.files_from,
            out_dir=args.out,
            manifest_path=args.manifest,
            files_glob=args.files,
            dry_run=args.dry_run,
        )
    except RegistrationRequired as e:
        print(f"error: {e}\nregister at: {REGISTRATION_URL}", file=sys.stderr)
        return 2
    except DownloadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
