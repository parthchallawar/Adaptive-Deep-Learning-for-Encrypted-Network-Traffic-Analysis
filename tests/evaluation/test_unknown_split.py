"""Open-set known/unknown class draw (spec 004, plan T6)."""

from __future__ import annotations

import pytest

from adl_etc.evaluation.unknown_split import draw_unknown_split

# 4 categories x 10 classes = 40 classes, support alternating low/high within
# each category so both "frequent" and "rare" apps are always available.
_CATEGORIES = {}
_SUPPORT = {}
for cat_i, cat in enumerate(["chat", "video", "web", "mail"]):
    for j in range(10):
        class_id = cat_i * 10 + j
        _CATEGORIES[class_id] = cat
        # Half the classes per category are "rare" (below the 100 floor).
        _SUPPORT[class_id] = 50 if j < 5 else 500


def test_known_and_unknown_are_disjoint_and_correctly_sized():
    known, unknown = draw_unknown_split(
        _CATEGORIES, _SUPPORT, n_known=8, n_unknown=4, min_unknown_test_support=100, seed=42
    )
    assert len(known) == 8
    assert len(unknown) == 4
    assert set(known).isdisjoint(unknown)
    assert set(known) | set(unknown) <= set(_CATEGORIES)


def test_unknown_never_includes_a_class_below_the_support_floor():
    _, unknown = draw_unknown_split(
        _CATEGORIES, _SUPPORT, n_known=8, n_unknown=4, min_unknown_test_support=100, seed=42
    )
    assert all(_SUPPORT[c] >= 100 for c in unknown)


def test_unknown_is_stratified_across_categories():
    # 4 unknown slots, 4 categories -> one per category with an even split.
    _, unknown = draw_unknown_split(
        _CATEGORIES, _SUPPORT, n_known=8, n_unknown=4, min_unknown_test_support=100, seed=42
    )
    seen_categories = {_CATEGORIES[c] for c in unknown}
    assert seen_categories == {"chat", "video", "web", "mail"}


def test_same_seed_is_deterministic():
    a = draw_unknown_split(_CATEGORIES, _SUPPORT, n_known=8, n_unknown=4, seed=42)
    b = draw_unknown_split(_CATEGORIES, _SUPPORT, n_known=8, n_unknown=4, seed=42)
    assert a == b


def test_different_seed_gives_a_different_draw():
    a = draw_unknown_split(_CATEGORIES, _SUPPORT, n_known=8, n_unknown=4, seed=42)
    b = draw_unknown_split(_CATEGORIES, _SUPPORT, n_known=8, n_unknown=4, seed=43)
    assert a != b


def test_raises_when_not_enough_classes():
    with pytest.raises(ValueError, match="need at least"):
        draw_unknown_split(_CATEGORIES, _SUPPORT, n_known=150, n_unknown=30, seed=42)


def test_raises_when_not_enough_eligible_classes_for_unknown():
    # Every class is below the floor -- no valid unknown draw exists.
    low_support = dict.fromkeys(_CATEGORIES, 1)
    with pytest.raises(ValueError, match="test flows"):
        draw_unknown_split(
            _CATEGORIES, low_support, n_known=8, n_unknown=4, min_unknown_test_support=100, seed=42
        )
