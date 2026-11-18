"""Packets-to-decision and time-to-decision of every policy for one scored run (plan phase 5,
T3; spec 013).

    python scripts/phase5_policy_cost.py \
        --run-dir results/kaggle-p4-scores/results/scores/ft-pat-ssl_tuned-d1open-s0 \
        --out results/phase5/ft-pat-ssl_tuned-d1open-s0/policy_cost.json

Needs the run's score files and the D1 shards (for each flow's inter-packet times, looked up by
``flow_index``). Policies are compared at one mean K, chosen on validation. NumPy only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from adl_etc.evaluation.efficiency import policy_cost
from adl_etc.evaluation.phase4 import jsonable
from adl_etc.evaluation.protocol import load_split
from adl_etc.evaluation.scores import ScoreArrays
from adl_etc.training.datasets import ArrayData
from adl_etc.utils.atomic import atomic_write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--split-file", default="configs/splits/d1_open.yaml")
    parser.add_argument("--shard-root", default="results/kaggle-full/shards")
    parser.add_argument("--target-mean-k", type=float, default=6.0)
    parser.add_argument("--n-classes", type=int, default=150)
    args = parser.parse_args(argv)

    run = args.run_dir
    val = ScoreArrays.load(run / "eval" / "val" / "scores")
    test = ScoreArrays.load(run / "eval" / "test_id" / "scores")
    loaded = load_split(args.split_file, root=args.shard_root)
    data = ArrayData.from_shards(loaded.shard_sets["test_id"], loaded.masks["test_id"])
    ipt = data.ppi[test.flow_index][:, :, 0].astype(np.float32)  # channel 0 = ipt_ms
    # the stored length must agree, or the flow_index lookup is pointing at the wrong flows
    if not np.array_equal(data.ppi_len[test.flow_index], test.ppi_len):
        raise SystemExit("flow_index does not line up with the shards (ppi_len differs)")
    report = jsonable(
        {
            "run": test.meta.get("run"),
            **policy_cost(val, test, ipt, args.n_classes, target_mean_k=args.target_mean_k),
        }
    )
    out = args.out or run / "eval" / "policy_cost.json"
    atomic_write_json(out, report)
    print(f"wrote {out}")
    for name in ("safe_stop", "p_echo", "p_cape", "p_rl"):
        r = report[name]
        pk = r["packets"]
        print(
            f"  {r['label']:10s} packets mean {pk['mean']:.2f} p95 {pk['p95']:.0f}"
            f"  time mean {r['time_ms']['mean']:.0f} ms ({r['time_share_of_flow']:.0%} of the flow)"
            f"  acc {r['accuracy']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
