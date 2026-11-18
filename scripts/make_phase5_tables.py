"""Phase 5's summary tables from the per-run JSON files.

    python scripts/make_phase5_tables.py --root results/phase5 --out results/summaries

``--root`` holds ``<run>/{drift,drift_monitor,policy_cost}.json`` per run and an optional
``efficiency.json``. Writes ``phase5_table.md`` and ``phase5_criteria.json``. See
``adl_etc.evaluation.phase5_tables``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adl_etc.evaluation.phase5_tables import (
    criteria,
    load_efficiency,
    load_group,
    render_markdown,
)
from adl_etc.utils.atomic import atomic_write_json, atomic_write_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("results/phase5"))
    parser.add_argument("--out", type=Path, default=Path("results/summaries"))
    parser.add_argument("--headline", default="ft-pat-ssl_tuned-d1open")
    args = parser.parse_args(argv)

    decay = load_group(args.root, "drift.json")
    monitor = load_group(args.root, "drift_monitor.json")
    cost = load_group(args.root, "policy_cost.json")
    eff = load_efficiency(args.root)
    if not (decay or monitor or cost or eff):
        raise SystemExit(f"no phase 5 result files under {args.root}")
    crit = criteria(decay, monitor, eff, headline=args.headline)
    atomic_write_text(
        args.out / "phase5_table.md", render_markdown(decay, monitor, cost, eff, crit)
    )
    atomic_write_json(args.out / "phase5_criteria.json", crit)
    print(f"wrote {args.out}/phase5_table.md and phase5_criteria.json")
    print(
        f"decay x{len(decay)}, monitor x{len(monitor)}, policy cost x{len(cost)}, "
        f"efficiency {'yes' if eff else 'no'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
