"""Batched augmentations equal the per-flow references flow by flow when both get
the same random draws (plan phase 3, F4/T5)."""

from __future__ import annotations

import numpy as np
import pytest

from adl_etc.data import augment_batch as AB
from adl_etc.data import features as F
from adl_etc.data import ppi as P


def random_flows(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ppi = np.zeros((n, P.K_MAX, P.PPI_CHANNELS), dtype=P.PPI_DTYPE)
    length = rng.integers(1, P.K_MAX + 1, n).astype(np.int8)
    for i, k in enumerate(length):
        ppi[i, :k, P.IPT_POS] = rng.integers(0, 4000, k)
        ppi[i, 0, P.IPT_POS] = 0
        ppi[i, :k, P.DIR_POS] = rng.choice([P.DIR_FWD, P.DIR_REV], k)
        ppi[i, :k, P.SIZE_POS] = rng.integers(1, P.SIZE_MAX + 1, k)
        ppi[i, :k, P.PUSH_POS] = rng.integers(0, 2, k)
    return ppi, length


class Replay:
    """A stand-in RNG that hands the per-flow reference pre-drawn values."""

    def __init__(self, uniform=None, normal=None, reorder_q=None):
        self.uniform, self.normal_draws, self.q = uniform, normal, reorder_q
        self.pos = 0

    def random(self, size=None):
        if size is not None:  # drop: one draw per real packet
            return self.uniform[:size]
        value = float(self.uniform[self.pos])  # reorder: one draw per visited position
        self.pos += 2 if value < self.q else 1
        return value

    def integers(self, low, high=None):
        raise AssertionError("tests only use draws that keep at least one packet")

    def normal(self, loc, scale, size):
        return loc + scale * self.normal_draws[:size]


def test_drop_matches_the_reference_flow_by_flow() -> None:
    ppi, length = random_flows(300, 0)
    rng = np.random.default_rng(1)
    u = rng.random((300, P.K_MAX))
    p = rng.uniform(0, 0.6, 300)
    u[np.arange(300), 0] = 1.0  # always keep packet 0, so the reference never needs a rescue
    keep = u >= p[:, None]

    out, new_len = AB.drop_given(ppi, length, keep)

    for i in range(300):
        want, want_len = F.drop(ppi[i], int(length[i]), float(p[i]), Replay(uniform=u[i]))
        assert new_len[i] == want_len
        np.testing.assert_array_equal(out[i], want, err_msg=f"flow {i}")


def test_drop_never_empties_a_flow_and_keeps_padding_zero() -> None:
    ppi, length = random_flows(500, 2)
    out, new_len = AB.drop(ppi, length, 1.0, np.random.default_rng(3))  # p = 1: drop everything

    assert (new_len == 1).all()
    assert (out[:, 1:] == 0).all() and (out[:, 0, P.IPT_POS] == 0).all()
    assert (out[:, 0, P.SIZE_POS] > 0).all()


def test_drop_preserves_elapsed_time_from_the_first_survivor_on() -> None:
    ppi, length = random_flows(400, 4)
    out, new_len = AB.drop(ppi, length, 0.3, np.random.default_rng(5))

    for i in range(400):
        # the last survivor's arrival time equals the original last-packet arrival
        # whenever the original last packet survived (gaps are folded forward)
        kept = out[i, : new_len[i]]
        orig_total = int(ppi[i, : length[i], P.IPT_POS].sum())
        assert int(kept[:, P.IPT_POS].sum()) <= orig_total


def test_reorder_matches_the_reference_flow_by_flow() -> None:
    ppi, length = random_flows(300, 6)
    u = np.random.default_rng(7).random((300, P.K_MAX - 1))
    q = 0.35

    out = AB.reorder_given(ppi, length, u, q)

    for i in range(300):
        want = F.reorder(ppi[i], int(length[i]), q, Replay(uniform=u[i], reorder_q=q))
        np.testing.assert_array_equal(out[i], want, err_msg=f"flow {i}")


def test_ipt_jitter_matches_the_reference_flow_by_flow() -> None:
    ppi, length = random_flows(300, 8)
    z = np.random.default_rng(9).standard_normal((300, P.K_MAX))

    out = AB.ipt_jitter_given(ppi, length, z, 0.2)

    for i in range(300):
        want = F.ipt_jitter(ppi[i], int(length[i]), 0.2, Replay(normal=z[i]))
        np.testing.assert_array_equal(out[i], want, err_msg=f"flow {i}")


def test_size_jitter_matches_the_reference_flow_by_flow() -> None:
    ppi, length = random_flows(300, 10)
    z = np.random.default_rng(11).standard_normal((300, P.K_MAX))

    out = AB.size_jitter_given(ppi, length, 8.0 * z)

    for i in range(300):
        want = F.size_jitter(ppi[i], int(length[i]), 8.0, Replay(normal=z[i]))
        np.testing.assert_array_equal(out[i], want, err_msg=f"flow {i}")


@pytest.mark.parametrize("fn", ["reorder", "ipt_jitter", "size_jitter"])
def test_padding_is_never_touched(fn: str) -> None:
    ppi, length = random_flows(200, 12)
    rng = np.random.default_rng(13)
    out = {
        "reorder": lambda: AB.reorder(ppi, length, 0.5, rng),
        "ipt_jitter": lambda: AB.ipt_jitter(ppi, length, 0.5, rng),
        "size_jitter": lambda: AB.size_jitter(ppi, length, 50.0, rng),
    }[fn]()
    pad = np.arange(P.K_MAX)[None, :] >= length[:, None]
    assert (out[pad] == 0).all()


def test_the_prefix_crop_is_always_shorter_than_its_flow() -> None:
    length = np.array([1, 2, 3, 4, 5, 10, 30, 30])
    ks = np.stack([AB.prefix_crop(length, np.random.default_rng(s))[0] for s in range(500)])
    eligible = AB.prefix_crop(length, np.random.default_rng(0))[1]

    assert eligible.tolist() == [False, False, False, True, True, True, True, True]
    assert (ks[:, eligible] >= 3).all() and (ks[:, eligible] <= (length[eligible] - 1)).all()
    assert set(ks[:, 3].tolist()) == {3} and set(ks[:, 6].tolist()) == set(range(3, 30))
    assert (ks[:, ~eligible] == length[~eligible]).all()


def test_the_batched_versions_are_fast_enough_for_ssl() -> None:
    """F4's point: a 4,096-flow batch through every augmentation in well under a
    second (the per-flow references take several seconds)."""
    import time

    ppi, length = random_flows(4096, 14)
    rng = np.random.default_rng(15)
    t0 = time.perf_counter()
    a, n = AB.drop(ppi, length, rng.uniform(0, 0.2, 4096), rng)
    a = AB.reorder(a, n, 0.05, rng)
    a = AB.ipt_jitter(a, n, 0.2, rng)
    a = AB.size_jitter(a, n, 8.0, rng)
    AB.prefix_crop(n, rng)
    assert time.perf_counter() - t0 < 1.0
