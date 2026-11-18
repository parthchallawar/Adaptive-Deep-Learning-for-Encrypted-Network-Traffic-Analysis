"""Weekly decay curves for one scored run (plan phase 5, T1).

    python scripts/phase5_drift_curves.py \
        --run-dir results/kaggle-p4-scores/results/scores/ft-pat-ssl_tuned-d1open-s0 \
        --out results/phase5/ft-pat-ssl_tuned-d1open-s0/drift.json

Reads ``<run>/eval/{val,test_id,test_drift}/scores/`` (and ``<run>/unknown.{npz,json}`` when the
model has an unknown score) and writes ``drift.json``: per week accuracy and macro-F1 at fixed K
and under Safe-Stop with a validation-calibrated threshold, plus the decay summary. NumPy only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adl_etc.evaluation.drift_curves import decay_summary, weekly_curves
from adl_etc.evaluation.phase4 import jsonable
from adl_etc.evaluation.scores import ScoreArrays
from adl_etc.inference.unknown import UnknownModel
from adl_etc.utils.atomic import atomic_write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: <run>/eval/drift.json")
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
    report = jsonable(
        {"run": sets["val"].meta.get("run"), **curves, "summary": decay_summary(curves)}
    )
    out = args.out or run / "eval" / "drift.json"
    atomic_write_json(out, report)
    s = report["summary"]
    print(f"wrote {out}")
    for key in ("acc_k5", "acc_k8", "acc_k30", "macro_f1_k8", "policy_accuracy", "policy_mean_k"):
        if key in s:
            print(
                f"  {key}: {s[key]['test_id']:.3f} -> {s[key]['last_four_drift']:.3f} "
                f"({s[key]['change']:+.3f}, {s[key]['slope_per_week']:+.4f}/week)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
