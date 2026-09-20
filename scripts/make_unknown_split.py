#!/usr/bin/env python
"""CLI wrapper around adl_etc.evaluation.unknown_split (spec 004, plan T6).

    python scripts/make_unknown_split.py \
        --dataset cesnet-tls-year22 --test-period WEEK-2022-31 \
        --out configs/splits/d1_main.unknown_classes.yaml

Reads the given test-ID period's real per-class support directly from its
shard set (`label_map`/`category_map` in `meta.json`, flow counts from the
`label` column), draws the seed-42 known/unknown split spec 004 defines,
and a second seed-43 draw recorded alongside it for the variance estimate.
Writes both into their own small YAML file rather than editing a primary
`configs/splits/*.yaml` in place, so the split file's hand-written comments
are never at risk of being clobbered by a round-tripped YAML dump.

Depends on plan T5's real class list: needs a test-ID period's shard set to
already exist on disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from adl_etc.data.tensors import ShardSet
from adl_etc.evaluation.unknown_split import draw_unknown_split


def class_stats(shards: ShardSet) -> tuple[dict[int, str], dict[int, int]]:
    """``(class -> category name, class -> flow count)`` read from a real
    shard set's own ``label_map``/``category_map`` and ``label`` column."""
    label_map: dict[str, int] = shards.meta["label_map"]
    category_map: dict[str, int] = shards.meta.get("category_map") or {}
    category_by_id = {v: k for k, v in category_map.items()}

    labels = shards.column("label")
    categories = shards.column("category")
    class_categories: dict[int, str] = {}
    support: dict[int, int] = {}
    for class_id in label_map.values():
        mask = labels == class_id
        support[class_id] = int(mask.sum())
        cats = categories[mask]
        # A class should map to one category; if not, the most common one
        # still gives a usable stratum rather than raising mid-draw.
        class_categories[class_id] = category_by_id.get(
            int(np.bincount(cats).argmax()) if cats.size else -1, "unknown"
        )
    return class_categories, support


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--test-period", required=True, help="the test-ID period, e.g. WEEK-2022-31"
    )
    parser.add_argument("--root", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-known", type=int, default=150)
    parser.add_argument("--n-unknown", type=int, default=30)
    parser.add_argument("--min-test-support", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variance-seed", type=int, default=43)
    args = parser.parse_args(argv)

    shards = ShardSet.open(args.root / args.dataset / args.test_period)
    try:
        class_categories, support = class_stats(shards)
    finally:
        shards.close()

    known, unknown = draw_unknown_split(
        class_categories,
        support,
        n_known=args.n_known,
        n_unknown=args.n_unknown,
        min_unknown_test_support=args.min_test_support,
        seed=args.seed,
    )
    _, unknown_variance = draw_unknown_split(
        class_categories,
        support,
        n_known=args.n_known,
        n_unknown=args.n_unknown,
        min_unknown_test_support=args.min_test_support,
        seed=args.variance_seed,
    )

    out: dict[str, Any] = {
        "dataset": args.dataset,
        "test_period": args.test_period,
        "seed": args.seed,
        "known_classes": known,
        "unknown_classes": unknown,
        "variance_seed": args.variance_seed,
        "unknown_classes_variance": unknown_variance,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(out), args.out)
    print(f"{len(known)} known, {len(unknown)} unknown classes written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
