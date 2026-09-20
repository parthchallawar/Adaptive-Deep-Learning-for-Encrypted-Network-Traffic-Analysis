"""Open-set known/unknown class draw (spec 004, plan T6).

Draws `n_unknown` classes to hold out as "unknown" (never seen in train,
present only in val/test for open-set evaluation) and the remaining
`n_known` classes for supervised training, stratified across categories so
the unknown set is not accidentally all-rare or all-one-category, and
excluding any class whose test-ID support is below spec 004's 100-flow
floor before it is ever drawn (equivalent to "redraw" without the wasted
draws).

Separated from `protocol.py`: this is a design-time tool run once per split
(by `scripts/make_unknown_split.py`), not something `load_split` runs on
every load.
"""

from __future__ import annotations

import numpy as np


def draw_unknown_split(
    class_categories: dict[int, str],
    test_support: dict[int, int],
    *,
    n_known: int = 150,
    n_unknown: int = 30,
    min_unknown_test_support: int = 100,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Returns ``(known_classes, unknown_classes)``, both sorted.

    ``class_categories`` maps every candidate class id to its category name;
    ``test_support`` maps a class id to its flow count in the test-ID period
    (spec 004's "redraw any unknown with fewer than 100 test flows" rule --
    implemented by never drawing an ineligible class in the first place,
    which is equivalent and needs no retry loop).
    """
    all_classes = sorted(class_categories)
    if len(all_classes) < n_known + n_unknown:
        raise ValueError(
            f"need at least {n_known + n_unknown} classes, have {len(all_classes)}"
        )

    rng = np.random.default_rng(seed)
    by_category: dict[str, list[int]] = {}
    for c in all_classes:
        by_category.setdefault(class_categories[c], []).append(c)
    categories = sorted(by_category)

    base, extra = divmod(n_unknown, len(categories))
    target_per_category = {
        cat: base + (1 if i < extra else 0) for i, cat in enumerate(categories)
    }

    unknown: list[int] = []
    for cat in categories:
        eligible = sorted(
            (c for c in by_category[cat] if test_support.get(c, 0) >= min_unknown_test_support),
            key=lambda c: test_support.get(c, 0),
        )
        picked = _stratified_pick(rng, eligible, target_per_category[cat])
        unknown.extend(picked)

    if len(unknown) < n_unknown:
        # Categories that ran out of eligible classes: top up randomly from
        # whatever eligible classes remain anywhere.
        remaining_eligible = [
            c
            for c in all_classes
            if c not in unknown and test_support.get(c, 0) >= min_unknown_test_support
        ]
        rng.shuffle(remaining_eligible)
        unknown.extend(remaining_eligible[: n_unknown - len(unknown)])

    if len(unknown) < n_unknown:
        raise ValueError(
            f"only {len(unknown)} classes have >= {min_unknown_test_support} test "
            f"flows; need {n_unknown} for the unknown split"
        )
    unknown = unknown[:n_unknown]

    remaining = [c for c in all_classes if c not in unknown]
    rng.shuffle(remaining)
    if len(remaining) < n_known:
        raise ValueError(f"only {len(remaining)} classes left for {n_known} known slots")
    known = sorted(remaining[:n_known])
    return known, sorted(unknown)


def _stratified_pick(
    rng: np.random.Generator, sorted_by_support_asc: list[int], take: int
) -> list[int]:
    """Randomly picks ``take`` classes from ``sorted_by_support_asc``, split
    roughly evenly between its rarer and more frequent halves, so a
    stratum's draw is not all long-tail apps or all head apps (spec 004:
    "include both frequent and rare apps") while still depending on ``rng``
    -- a rank-based alternating pick would be fully determined by the
    support ordering whenever support values have no ties, which is the
    normal case on real data, making two different seeds draw identically.
    Confirmed by running this against real D4 support counts before this
    fix: seed 42 and seed 43 produced the exact same unknown set."""
    n = len(sorted_by_support_asc)
    if n <= take:
        return list(sorted_by_support_asc)
    mid = n // 2
    rare_half = sorted_by_support_asc[:mid] or sorted_by_support_asc[:1]
    frequent_half = sorted_by_support_asc[mid:] or sorted_by_support_asc[-1:]
    n_rare = min((take + 1) // 2, len(rare_half))
    n_frequent = min(take - n_rare, len(frequent_half))

    picked: list[int] = []
    picked.extend(rng.choice(rare_half, size=n_rare, replace=False).tolist())
    picked.extend(rng.choice(frequent_half, size=n_frequent, replace=False).tolist())

    if len(picked) < take:
        remaining = [c for c in sorted_by_support_asc if c not in picked]
        rng.shuffle(remaining)
        picked.extend(remaining[: take - len(picked)])
    return picked[:take]
