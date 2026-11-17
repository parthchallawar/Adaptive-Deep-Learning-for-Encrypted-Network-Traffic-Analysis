"""Phase 4's summary tables from the per-run ``phase4.json`` files.

    python scripts/make_phase4_tables.py --root results/phase4 --out results/summaries

``--root`` holds one directory per run (``<run>/eval/phase4.json``, as pulled from the Kaggle
score session: ``kaggle kernels output ... --file-pattern 'phase4\\.json$'``). Writes
``phase4_table.md`` and ``phase4_criteria.json``. See ``adl_etc.evaluation.phase4_tables``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adl_etc.evaluation.phase4_tables import criteria, load_reports, render_markdown
from adl_etc.utils.atomic import atomic_write_json, atomic_write_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("results/phase4"))
    parser.add_argument("--out", type=Path, default=Path("results/summaries"))
    parser.add_argument("--headline", default="ft-pat-ssl_tuned-d1open")
    args = parser.parse_args(argv)

    groups = load_reports(args.root)
    if not groups:
        raise SystemExit(f"no <run>/eval/phase4.json files under {args.root}")
    atomic_write_text(args.out / "phase4_table.md", render_markdown(groups))
    atomic_write_json(args.out / "phase4_criteria.json", criteria(groups, headline=args.headline))
    listing = ", ".join(f"{g} x{len(m)}" for g, m in sorted(groups.items()))
    print(f"wrote {args.out}/phase4_table.md and phase4_criteria.json")
    print(f"{len(groups)} group(s): {listing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
