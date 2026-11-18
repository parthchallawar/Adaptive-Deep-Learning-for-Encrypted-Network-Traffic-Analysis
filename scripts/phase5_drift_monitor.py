"""Run the drift monitor over one scored run and validate it against the measured decay
(plan phase 5, T2; spec 012).

    python scripts/phase5_drift_monitor.py \
        --run-dir results/kaggle-p4-scores/results/scores/ft-pat-ssl_tuned-d1open-s0 \
        --out results/phase5/ft-pat-ssl_tuned-d1open-s0/drift_monitor.json

Reads ``<run>/eval/{val,test_id,test_drift}/scores/`` (and ``<run>/unknown.{npz,json}`` when the
model has one). NumPy only. See ``adl_etc.evaluation.drift_monitor``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adl_etc.evaluation.drift_curves import weekly_curves
from adl_etc.evaluation.drift_monitor import monitor_report
from adl_etc.evaluation.phase4 import jsonable
from adl_etc.evaluation.scores import ScoreArrays
from adl_etc.inference.unknown import UnknownModel
from adl_etc.utils.atomic import atomic_write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--out", type=Path, default=None, help="default: <run>/eval/drift_monitor.json"
    )
    args = parser.parse_args(argv)

    run = args.run_dir
    sets = {
        n: ScoreArrays.load(run / "eval" / n / "scores") for n in ("val", "test_id", "test_drift")
    }
    ckpt = sets["val"].meta.get("checkpoint_hash")
    um = (
        UnknownModel.load(run / "unknown", checkpoint_hash=ckpt)
        if (run / "unknown.npz").is_file()
        else None
    )
    curves = weekly_curves({k: sets[k] for k in ("test_id", "test_drift")}, sets["val"], um)
    report = jsonable({"run": sets["val"].meta.get("run"), **monitor_report(sets, um, curves)})
    out = args.out or run / "eval" / "drift_monitor.json"
    atomic_write_json(out, report)
    print(f"wrote {out}")
    for mode in ("static", "controller"):
        m = report[mode]
        lead = m["lead"]
        print(
            f"[{mode}] h={m['h']}  F1 floor {lead['macro_f1_floor']:.3f} first reached "
            f"{lead['first_week_below_floor']}"
        )
        print(
            f"   first warning {lead['first_warning'] and lead['first_warning']['week']}  "
            f"first alarm {lead['first_alarm'] and lead['first_alarm']['week']}  "
            f"lead (weeks) warning={lead.get('lead_weeks_warning')} "
            f"alarm={lead.get('lead_weeks_alarm')}"
        )
        for s, c in m["correlation"].items():
            first = m["per_signal_first_alarm"][s]
            print(
                f"   {s:22s} spearman vs macro-F1 {c['spearman_vs_policy_macro_f1']:+.2f}  "
                f"first alarm {first and first['week']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
