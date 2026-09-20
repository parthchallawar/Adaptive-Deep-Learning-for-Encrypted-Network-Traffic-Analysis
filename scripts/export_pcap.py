#!/usr/bin/env python
"""CLI wrapper around adl_etc.data.export_pcap (spec 002, plan T4).

    python scripts/export_pcap.py --dataset iscx-vpn-2016 \
        --config configs/data/pcap.yaml --out data/processed/iscx-vpn-2016/all

Joins the pcap_source -> flows -> tensors pipeline into shards, reading
labels from data/raw/<dataset>/labels.csv (written by the spec-001
downloaders). See export_pcap.py's module docstring for what this does.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adl_etc.data import manifest as M
from adl_etc.data.export_pcap import PcapConfig, export_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="directory name under data/raw/")
    parser.add_argument("--config", type=Path, default=None, help="configs/data/pcap.yaml")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="full period directory, e.g. data/processed/<dataset>/all "
        "(overrides --out-root/--period; must end in <out-root>/<dataset>/<period>)",
    )
    parser.add_argument("--out-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--period", default="all")
    parser.add_argument("--manifest", type=Path, default=M.DEFAULT_MANIFEST_PATH)
    parser.add_argument("--files", default=None, help="glob to limit which captures are exported")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.out is not None:
        if args.out.parent.name != args.dataset:
            parser.error(f"--out {args.out} must end in <out-root>/{args.dataset}/<period>")
        out_root, period = args.out.parent.parent, args.out.name
    else:
        out_root, period = args.out_root, args.period

    export_dataset(
        dataset=args.dataset,
        raw_root=args.raw_root,
        out_root=out_root,
        period=period,
        config=PcapConfig.load(args.config),
        files_glob=args.files,
        manifest_path=args.manifest,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
