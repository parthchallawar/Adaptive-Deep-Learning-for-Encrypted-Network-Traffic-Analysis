"""H3, H2 and H4 for one trained run, from its saved score files (plan phase 4, T6).

    python scripts/phase4_report.py \
        --run-dir results/kaggle-p3-s4/results/ft-pat-ssl_tuned-d1open-s1

Reads ``<run>/eval/{val,test_id,test_drift}/scores/`` and ``<run>/unknown.{npz,json}`` (written by
``python -m adl_etc.evaluation.score_run``) and writes ``<run>/eval/phase4.json``. Nothing here
needs torch or a model. Every fit uses validation scores only; see ``adl_etc.evaluation.phase4``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adl_etc.evaluation.phase4 import REPORT_KS, run_report
from adl_etc.evaluation.scores import ScoreArrays
from adl_etc.inference.unknown import UnknownModel
from adl_etc.utils.atomic import atomic_write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: <run>/eval/phase4.json")
    parser.add_argument("--window", type=int, default=2000, help="controller window, in flows")
    parser.add_argument("--skip-h4", action="store_true", help="skip the (slowest) controller part")
    args = parser.parse_args(argv)

    run = args.run_dir
    val = ScoreArrays.load(run / "eval" / "val" / "scores")
    test = ScoreArrays.load(run / "eval" / "test_id" / "scores")
    drift_dir = run / "eval" / "test_drift" / "scores"
    drift = ScoreArrays.load(drift_dir) if drift_dir.is_dir() else None
    um = UnknownModel.load(run / "unknown", checkpoint_hash=val.meta.get("checkpoint_hash"))
    for name, s in (("test_id", test), ("test_drift", drift)):
        if s is not None and s.meta.get("checkpoint_hash") != val.meta.get("checkpoint_hash"):
            raise SystemExit(f"{name} scores come from a different checkpoint than val's")
    n_classes = len(json.loads((run / "label_space.json").read_text(encoding="utf-8"))["known"])

    report = run_report(val, test, drift, um, n_classes, window=args.window, skip_h4=args.skip_h4)
    out = args.out or run / "eval" / "phase4.json"
    atomic_write_json(out, report)

    auroc = report["h3"]["auroc"]
    print(f"wrote {out}")
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
