#!/usr/bin/env python
"""CLI wrapper around adl_etc.evaluation.unknown_split (spec 004; plan T6, phase 3 T1a).

    python scripts/make_unknown_split.py \
        --dataset cesnet-tls-year22 \
        --test-period WEEK-2022-31 WEEK-2022-32 WEEK-2022-33 WEEK-2022-34 \
        --out configs/splits/d1_open.classes.yaml \
        --base-split configs/splits/d1_main.yaml --split-out configs/splits/d1_open.yaml

Reads the test-ID periods' real per-class support (summed over every period
given, since spec 004's 100-flow floor is over the whole test period) directly
from their shard sets (`label_map`/`category_map` in `meta.json`, flow counts
from the `label` column), draws the seed-42 known/unknown split spec 004
defines, and a second seed-43 draw recorded alongside it for the variance
estimate. Writes both, with class names and supports, to `--out`.

With `--base-split`/`--split-out` it also writes the open-set split file: the
base split's own text, unchanged (so its hand-written comments survive), plus a
generated `unknown_classes:` block that `protocol.load_split` reads. Regenerate
that file with this script rather than editing it by hand.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from adl_etc.data.tensors import ShardSet
from adl_etc.evaluation.unknown_split import draw_unknown_split


def class_stats(
    shard_sets: list[ShardSet],
) -> tuple[dict[int, str], dict[int, int], dict[int, str]]:
    """``(class -> category name, class -> flow count, class -> name)`` summed
    over every given shard set, from their own ``label_map``/``category_map``
    and ``label``/``category`` columns. A class's category is its most common
    one across all of them."""
    label_map: dict[str, int] = shard_sets[0].meta["label_map"]
    category_map: dict[str, int] = shard_sets[0].meta.get("category_map") or {}
    for ss in shard_sets[1:]:
        if ss.meta["label_map"] != label_map:
            raise ValueError("the test periods do not share one label_map")
    category_by_id = {v: k for k, v in category_map.items()}

    labels = np.concatenate([ss.column("label") for ss in shard_sets])
    categories = np.concatenate([ss.column("category") for ss in shard_sets])
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
    names = {int(v): str(k) for k, v in label_map.items()}
    return class_categories, support, names


def open_set_block(
    command: str,
    unknown: list[int],
    variance: list[int],
    n_known: int,
    names: dict[int, str],
    support: dict[int, int],
    categories: dict[int, str],
    seed: int,
    variance_seed: int,
) -> str:
    """The text appended to the base split: a commented, generated block."""
    lines = [
        "",
        "# --- open set (spec 004; plan phase 3, T1a) -------------------------------",
        "# GENERATED -- do not edit by hand; regenerate with:",
        f"#   {command}",
        f"# {n_known} known / {len(unknown)} unknown classes, seed {seed}, stratified by",
        "# category, every unknown class at or above the test-period support floor.",
        "# protocol.load_split drops these classes' flows from `train` and keeps them",
        "# everywhere else, so validation and test hold real unknown flows.",
        "unknown_classes:",
    ]
    for c in unknown:
        lines.append(
            f"  - {c}  # {names.get(c, c)} ({categories.get(c)}, {support[c]} test flows)"
        )
    lines += [
        f"# The seed-{variance_seed} draw, kept for spec 004's variance estimate (not used",
        "# by load_split):",
        f"unknown_classes_variance_seed{variance_seed}: [{', '.join(map(str, variance))}]",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--test-period",
        required=True,
        nargs="+",
        help="the test-ID period(s), e.g. WEEK-2022-31 ... WEEK-2022-34; support is summed",
    )
    parser.add_argument("--root", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-split", type=Path, default=None)
    parser.add_argument("--split-out", type=Path, default=None)
    parser.add_argument("--n-known", type=int, default=150)
    parser.add_argument("--n-unknown", type=int, default=30)
    parser.add_argument("--min-test-support", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variance-seed", type=int, default=43)
    args = parser.parse_args(argv)
    if (args.base_split is None) != (args.split_out is None):
        parser.error("--base-split and --split-out go together")
    base = args.base_split.read_text(encoding="utf-8") if args.base_split is not None else ""
    if "unknown_classes" in base:
        parser.error(f"{args.base_split} already has an open-set block")

    opened = [ShardSet.open(args.root / args.dataset / period) for period in args.test_period]
    try:
        class_categories, support, names = class_stats(opened)
    finally:
        for ss in opened:
            ss.close()

    def draw(seed: int) -> tuple[list[int], list[int]]:
        return draw_unknown_split(
            class_categories,
            support,
            n_known=args.n_known,
            n_unknown=args.n_unknown,
            min_unknown_test_support=args.min_test_support,
            seed=seed,
        )

    known, unknown = draw(args.seed)
    _, unknown_variance = draw(args.variance_seed)

    out: dict[str, Any] = {
        "dataset": args.dataset,
        "test_periods": list(args.test_period),
        "min_test_support": args.min_test_support,
        "seed": args.seed,
        "known_classes": known,
        "unknown_classes": unknown,
        "unknown_names": {c: names[c] for c in unknown},
        "unknown_test_support": {c: support[c] for c in unknown},
        "variance_seed": args.variance_seed,
        "unknown_classes_variance": unknown_variance,
        "classes_below_floor": sorted(c for c, n in support.items() if n < args.min_test_support),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(out), args.out)
    print(f"{len(known)} known, {len(unknown)} unknown classes written to {args.out}")

    if args.split_out is not None:
        command = (
            f"python scripts/make_unknown_split.py --dataset {args.dataset} --test-period "
            f"{' '.join(args.test_period)} --seed {args.seed} --base-split "
            f"{args.base_split.as_posix()} --split-out {args.split_out.as_posix()}"
        )
        block = open_set_block(
            command,
            unknown,
            unknown_variance,
            len(known),
            names,
            support,
            class_categories,
            args.seed,
            args.variance_seed,
        )
        header = (
            f"# {args.split_out.name}: {args.base_split.name} plus the open-set block at the end.\n"
        )
        args.split_out.write_text(header + base.rstrip("\n") + "\n" + block, encoding="utf-8")
        print(f"open-set split written to {args.split_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
