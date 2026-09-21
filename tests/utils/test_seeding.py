"""Seeding and independent streams (spec 014, plan phase 2 T1)."""

from __future__ import annotations

import random
import sys
from typing import Any

import numpy as np
import pytest

from adl_etc.utils.seeding import seed_everything, seeded_generator


def test_seed_everything_makes_python_and_numpy_repeatable() -> None:
    seed_everything(123)
    first = (random.random(), np.random.rand(3).tolist())
    seed_everything(123)
    second = (random.random(), np.random.rand(3).tolist())

    assert first == second


@pytest.mark.parametrize("bad", [-1, True, "3", 1.5, None])
def test_bad_seeds_are_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="non-negative int"):
        seed_everything(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative int"):
        seeded_generator(bad, "x")  # type: ignore[arg-type]


def test_seed_everything_works_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase-1-only environments have no torch; seeding must not require it."""
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError

    seed_everything(5)
    a = np.random.rand()
    seed_everything(5)

    assert np.random.rand() == a


# --- independent streams ------------------------------------------------------------


def test_same_seed_and_purpose_give_the_same_stream() -> None:
    a = seeded_generator(3, "data_order").integers(0, 10**9, 8)
    b = seeded_generator(3, "data_order").integers(0, 10**9, 8)

    assert a.tolist() == b.tolist()


def test_different_purpose_or_seed_gives_a_different_stream() -> None:
    base = seeded_generator(3, "data_order").integers(0, 10**9, 8).tolist()

    assert seeded_generator(3, "augment").integers(0, 10**9, 8).tolist() != base
    assert seeded_generator(4, "data_order").integers(0, 10**9, 8).tolist() != base


def test_drawing_from_one_stream_does_not_move_another() -> None:
    """The reason streams exist: adding an augmentation must not change data order."""
    quiet = seeded_generator(0, "data_order").permutation(100).tolist()

    seed_everything(0)
    augment = seeded_generator(0, "augment")
    augment.random(10_000)  # heavy use of a sibling stream
    order = seeded_generator(0, "data_order").permutation(100).tolist()

    assert order == quiet


def test_empty_purpose_is_rejected() -> None:
    with pytest.raises(ValueError, match="purpose"):
        seeded_generator(0, "")


def test_stream_bits_are_frozen() -> None:
    """A change detector, not an independent check: these were recorded from
    this implementation. If NumPy's SeedSequence/PCG64 ever changes, every
    saved run's data order silently moves, so it must fail here instead."""
    expected = {
        (0, "data_order"): [4108906603365761282, 16292477498906473834, 14012030248116818216],
        (0, "augment"): [5988614991066744620, 9115439040383427510, 322538432924758408],
        (1, "data_order"): [17420762975951181301, 16279849795082762234, 12097210782269329176],
    }
    for (seed, purpose), raw in expected.items():
        g = seeded_generator(seed, purpose)
        assert [int(g.bit_generator.random_raw()) for _ in range(3)] == raw


# --- torch (skipped, with a reason, when the `train` extra isn't installed) ----------


@pytest.fixture
def torch_mod() -> Any:
    return pytest.importorskip("torch", reason="needs the `train` extra: pip install -e '.[train]'")


def _tiny_mlp(torch: Any) -> Any:
    nn = torch.nn
    return nn.Sequential(
        nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 4)
    )


def test_seed_everything_makes_torch_init_and_shuffles_identical(torch_mod: Any) -> None:
    torch = torch_mod
    seed_everything(0)
    m1, perm1 = _tiny_mlp(torch), torch.randperm(50)
    seed_everything(0)
    m2, perm2 = _tiny_mlp(torch), torch.randperm(50)

    for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):
        assert torch.equal(p1, p2)
    assert torch.equal(perm1, perm2)


def test_different_seeds_give_different_torch_init(torch_mod: Any) -> None:
    torch = torch_mod
    seed_everything(0)
    m1 = _tiny_mlp(torch)
    seed_everything(1)
    m2 = _tiny_mlp(torch)

    assert not torch.equal(next(m1.parameters()), next(m2.parameters()))


def test_seed_everything_requests_deterministic_algorithms(torch_mod: Any) -> None:
    seed_everything(0)

    assert torch_mod.are_deterministic_algorithms_enabled()
    assert torch_mod.is_deterministic_algorithms_warn_only_enabled()
