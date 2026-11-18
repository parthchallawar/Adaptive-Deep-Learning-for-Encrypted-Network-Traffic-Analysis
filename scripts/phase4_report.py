"""H3, H2 and H4 for one trained run, from its saved score files (plan phase 4, T6).

    python scripts/phase4_report.py \
        --run-dir results/kaggle-p3-s4/results/ft-pat-ssl_tuned-d1open-s1

Reads ``<run>/eval/{val,test_id,test_drift}/scores/`` and ``<run>/unknown.{npz,json}`` (written by
``python -m adl_etc.evaluation.score_run``) and writes ``<run>/eval/phase4.json``. Nothing here
needs torch or a model. Every fit uses validation scores only; see ``adl_etc.evaluation.phase4``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adl_etc.evaluation.phase4 import REPORT_KS, report_for_run
from adl_etc.utils.atomic import atomic_write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: <run>/eval/phase4.json")
    parser.add_argument(
        "--label-space",
        type=Path,
        default=None,
        help="default: <run>/label_space.json (pass a local run's file when only the score "
        "files were pulled from Kaggle; every d1_open run shares one label space)",
    )
    parser.add_argument("--window", type=int, default=2000, help="controller window, in flows")
    parser.add_argument("--skip-h4", action="store_true", help="skip the (slowest) controller part")
    args = parser.parse_args(argv)

    run = args.run_dir
    try:
        report = report_for_run(
            run,
            args.label_space or run / "label_space.json",
            window=args.window,
            skip_h4=args.skip_h4,
        )
    except ValueError as err:  # e.g. score sets from different checkpoints
        raise SystemExit(str(err)) from err
    out = args.out or run / "eval" / "phase4.json"
    atomic_write_json(out, report)

    print(f"wrote {out}")
    if "h3" in report:
        auroc = report["h3"]["auroc"]
        print("unknown AUROC (energy_raw / u) at K =", ", ".join(map(str, REPORT_KS)))
        for sig in ("energy_raw", "u"):
            vals = [auroc[sig].get(str(k)) for k in REPORT_KS]
            print(f"  {sig:<11}", " ".join("  -  " if v is None else f"{v:.3f}" for v in vals))
    print("closed-world accuracy at a mean of 4 / 6 / 8 packets:")
    for name, summ in report["h2"]["closed_world"].items():
        b = summ["acc_at_budget"]
        accs = f"{b['4.0']:.3f} {b['6.0']:.3f} {b['8.0']:.3f}"
        print(f"  {name:<10} {accs}  area {summ['pareto_area']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
