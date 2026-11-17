"""Download named files from a specific *version* of a Kaggle kernel's output.

    python scripts/kaggle_pull_version.py parthchallawar/adl-encrypted-traffic-train 16 \
        results/ft-pat-tuned-d1open-s0/best.pt --out results/kaggle-p3-s3

Why this exists: ``kaggle kernels output <owner>/<slug>/<version>`` parses the version and then
never uses it, so it always returns the *latest* version's files, and once a newer version exists
an older version's checkpoints look unreachable from the command line (found 2026-11-17, plan
phase 4 T0). The API underneath does take a version number for a download; the *listing* endpoints
do not accept one, so you must name each file path. Files are written to ``<out>/<file path>``.

Uses the ``kaggle`` package's own authentication (``kaggle auth login``). Run it with the Python
that has ``kaggle`` installed (the project's ``.venv`` does not).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("kernel", help="<owner>/<slug>")
    parser.add_argument("version", type=int, help="kernel version number")
    parser.add_argument("files", nargs="+", help="output file paths, e.g. results/<run>/best.pt")
    parser.add_argument("--out", type=Path, default=Path("."), help="directory to write into")
    args = parser.parse_args(argv)
    owner, _, slug = args.kernel.partition("/")
    if not slug:
        parser.error("kernel must be <owner>/<slug>")

    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

    api = KaggleApi()
    api.authenticate()
    failures = 0
    with api.build_kaggle_client() as kaggle:
        client = kaggle.kernels.kernels_api_client
        for file_path in args.files:
            req = ApiDownloadKernelOutputRequest()
            req.owner_slug, req.kernel_slug = owner, slug
            req.version_number, req.file_path = args.version, file_path
            try:
                body = client.download_kernel_output(req).content
            except Exception as err:  # noqa: BLE001 - report and carry on with the other files
                print(
                    f"FAILED {file_path}: {type(err).__name__}: {str(err)[:160]}", file=sys.stderr
                )
                failures += 1
                continue
            if not body:
                print(f"FAILED {file_path}: empty response", file=sys.stderr)
                failures += 1
                continue
            dest = args.out / file_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            print(f"{dest}  ({len(body):,} bytes)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
