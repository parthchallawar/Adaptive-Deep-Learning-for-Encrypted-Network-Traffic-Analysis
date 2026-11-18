"""Efficiency benchmark (plan phase 5, T3; spec 013): model cost and policy cost.

    python scripts/benchmark.py --out results/phase5/efficiency.json

Model cost needs only the model configs (weights do not change timing); policy cost needs a
scored run (``--run-dir``) and the shards for the inter-packet times. Wall-clock numbers are for
the machine this runs on; ``hardware`` in the output says which.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from adl_etc.evaluation import efficiency as E
from adl_etc.evaluation.phase4 import jsonable
from adl_etc.utils.atomic import atomic_write_json

#: name -> a run whose ``config_resolved.yaml`` has the model section (architecture only).
MODELS = {
    "PAT (B5 / SSL / scratch share this architecture)": "ft-pat-ssl_tuned-d1open-s1",
    "B3 GRU": "bl-gru-main-d1open-s1",
    "B2 CNN": "bl-cnn-main-d1open-s0",
}


def find_config(name: str, roots: list[Path]) -> Path:
    for r in roots:
        hits = list(r.glob(f"*/results/{name}/config_resolved.yaml")) + list(
            r.glob(f"{name}/config_resolved.yaml")
        )
        if hits:
            return hits[0]
    raise SystemExit(f"no config_resolved.yaml for {name} under {roots}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("results/phase5/efficiency.json"))
    parser.add_argument("--roots", nargs="*", type=Path, default=[Path("results")])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--stream-flows", type=int, default=200)
    parser.add_argument("--n-classes", type=int, default=150)
    args = parser.parse_args(argv)

    import torch

    from adl_etc.models.registry import build_model

    torch.set_num_threads(args.threads)
    models = {}
    for label, run in MODELS.items():
        cfg = OmegaConf.load(find_config(run, args.roots))
        models[label] = build_model(cfg.model, args.n_classes).eval()
    bench = E.bench_all(
        models, threads=args.threads, iters=args.iters, stream_flows=args.stream_flows
    )
    report = {"hardware": E.hardware_info(args.threads), "models": bench}
    atomic_write_json(args.out, jsonable(report))
    print(json.dumps(jsonable(report["hardware"]), indent=1))
    for label, r in bench.items():
        print(
            f"{label}: {r['params']:,} params, {r['flops_full_flow'] / 1e6:.1f} MFLOP/flow, "
            f"batch-1 {r['latency_batch1']['median_ms']:.2f} ms, "
            f"stream step {r['stream_step']['median_ms']:.3f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
