"""Write ``results/models/production.json``: the model bundle the service loads (plan phase 6,
T2, correction S2).

    python scripts/make_production_bundle.py                       # the headline run, seed 0
    python scripts/make_production_bundle.py --run ft-pat-tuned-d1open-s0 --run-dir <dir>

It names files, it copies none: the training run directory (``best.pt``, ``calibration.json``,
``label_space.json``, ``config_resolved.yaml``), the phase-4 unknown model, the standardizer, and
the operating point that phases 4 and 5 used: the commit threshold calibrated to a validation mean
K of 6, ``f = 0.02``, and both of phase 5's drift-monitor references (static and controller mode)
with the validation class histogram behind ``prior_kl`` (recomputed here from the validation
scores with the same function phase 5 used). Every checkpoint hash is checked against the others.
NumPy only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_RUN = "ft-pat-ssl_tuned-d1open-s0"
DEFAULT_RUN_DIR = "results/kaggle-p3-s3/results/ft-pat-ssl_tuned-d1open-s0"
SCORES_ROOT = "results/kaggle-p4-scores/results/scores"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--scores-root", default=SCORES_ROOT)
    parser.add_argument("--standardizer", default="results/standardizer.json")
    parser.add_argument("--out", type=Path, default=Path("results/models/production.json"))
    args = parser.parse_args(argv)

    from adl_etc.evaluation.drift_monitor import F_REJECT, reference_histogram
    from adl_etc.evaluation.scores import ScoreArrays
    from adl_etc.inference.unknown import UnknownModel
    from adl_etc.utils.atomic import atomic_write_json
    from adl_etc.utils.provenance import file_sha256, now_iso

    run_dir = Path(args.run_dir)
    ckpt = file_sha256(run_dir / "best.pt")
    unknown = Path("results/phase4") / args.run / "unknown"
    phase4 = Path("results/phase4") / args.run / "eval" / "phase4.json"
    drift = Path("results/phase5") / args.run / "drift_monitor.json"
    for p in (unknown.with_suffix(".npz"), phase4, drift):
        if not p.is_file():
            raise SystemExit(f"missing {p}")
    um = UnknownModel.load(unknown, checkpoint_hash=ckpt)  # raises on a hash mismatch
    p4 = json.loads(phase4.read_text(encoding="utf-8"))
    p5 = json.loads(drift.read_text(encoding="utf-8"))
    if p4["meta"]["checkpoint_hash"] != ckpt:
        raise SystemExit("phase4.json was computed on another checkpoint")

    val = ScoreArrays.load(Path(args.scores_root) / args.run / "eval" / "val" / "scores")
    if val.meta.get("checkpoint_hash") != ckpt:
        raise SystemExit("validation scores come from another checkpoint")
    theta = float(p5["theta_commit"])
    n_classes = len(json.loads((run_dir / "label_space.json").read_text())["known"])
    hist = reference_histogram(val, um, theta, n_classes)

    def mode(name: str) -> dict:
        m = p5[name]
        return {"signals": m["signals"], "reference": m["reference"], "h": m["h"]}

    prod = {
        "created": now_iso(),
        "run": args.run,
        "run_dir": run_dir.as_posix(),
        "checkpoint_sha256": ckpt,
        "unknown": unknown.as_posix(),
        "standardizer": Path(args.standardizer).as_posix(),
        "offline": {"phase4": phase4.as_posix(), "phase5": drift.as_posix()},
        "reference": {
            "theta_commit": theta,
            "theta_commit_target_mean_k": 6.0,
            "f": F_REJECT,
            "drift_window": int(p5["window"]),
            "drift_static": mode("static"),
            "drift_controller": mode("controller"),
            "reference_hist": hist.tolist(),
        },
    }
    atomic_write_json(args.out, prod)
    print(f"wrote {args.out}: run {args.run}, checkpoint {ckpt[:12]}, theta {theta:.4f}")
    print(f"reference histogram: {int(hist.sum())} committed validation flows, {n_classes} classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
