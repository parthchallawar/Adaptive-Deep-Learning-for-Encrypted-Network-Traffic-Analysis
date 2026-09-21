"""Batches, the balanced sampler and the no-refit guarantee (plan phase 2 T5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adl_etc.data import features as F
from adl_etc.data import ppi as P
from adl_etc.data.features import Standardizer
from adl_etc.training import datasets as D
from adl_etc.training.datasets import (
    ArrayData,
    FlowBatches,
    balanced_weights,
    epoch_indices,
    to_torch,
)
from adl_etc.training.labels import LabelSpace
from tests.training.util import make_flows, write_shards

SPACE = LabelSpace.create(known=range(4), unknown=[4, 5])


@pytest.fixture
def fitted(tmp_path: Path):
    data = make_flows(600, 4, seed=1)
    shards = write_shards(tmp_path, data)
    std = Standardizer.fit(shards)
    shards.close()
    return data, std


# --- balanced sampling ------------------------------------------------------------------


def class_mass(y: np.ndarray, w: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(y, weights=w, minlength=n_classes)


def test_balanced_weights_match_the_hand_example() -> None:
    """Counts 100/10/1, cap 10: mean = 111/3 = 37, so the factors are
    min(10, 0.37)=0.37, min(10, 3.7)=3.7, min(10, 37)=10 and the class masses
    are 100*.37, 10*3.7, 1*10 = 37 : 37 : 10."""
    y = np.array([0] * 100 + [1] * 10 + [2])

    mass = class_mass(y, balanced_weights(y, 3, cap=10), 3)

    np.testing.assert_allclose(mass / mass.sum(), np.array([37, 37, 10]) / 84)


def test_a_huge_cap_gives_exactly_equal_class_mass_and_cap_one_the_natural_mix() -> None:
    y = np.array([0] * 100 + [1] * 10 + [2])

    even = class_mass(y, balanced_weights(y, 3, cap=1e9), 3)
    natural = balanced_weights(y, 3, cap=1)

    np.testing.assert_allclose(even, [1 / 3] * 3)
    # cap=1: no class is oversampled, so the rare ones stay rare; big classes are
    # only ever scaled *down* toward the mean, which keeps their order.
    assert (np.diff(class_mass(y, natural, 3)) < 0).all()


def test_weights_sum_to_one_and_absent_classes_do_not_divide_by_zero() -> None:
    y = np.array([0, 0, 2, 2, 2])  # classes 1 and 3 never occur

    w = balanced_weights(y, n_classes=4)

    assert np.isfinite(w).all() and w.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "y, cap, message",
    [
        (np.array([0, -1]), 10, "known-class"),
        (np.array([], dtype=int), 10, "empty"),
        (np.array([0, 1]), 0.5, "cap"),
    ],
)
def test_balanced_weights_reject_bad_input(y, cap, message) -> None:
    with pytest.raises(ValueError, match=message):
        balanced_weights(y, 2, cap=cap)


def test_epoch_indices_without_weights_visit_every_flow_once_per_pass() -> None:
    rng = np.random.default_rng(0)

    once = epoch_indices(50, 50, rng)
    two_and_a_half = epoch_indices(50, 125, rng)

    assert sorted(once.tolist()) == list(range(50))
    counts = np.bincount(two_and_a_half, minlength=50)
    assert set(counts.tolist()) <= {2, 3} and counts.sum() == 125


def test_epoch_indices_are_deterministic_for_a_seed_and_follow_the_weights() -> None:
    y = np.array([0] * 900 + [1] * 100)
    w = balanced_weights(y, 2, cap=1e9)  # equal class mass

    a = epoch_indices(1000, 200_000, np.random.default_rng(7), w)
    b = epoch_indices(1000, 200_000, np.random.default_rng(7), w)

    np.testing.assert_array_equal(a, b)
    assert (y[a] == 1).mean() == pytest.approx(0.5, abs=0.01)  # 10% of flows, 50% of draws


def test_epoch_indices_reject_an_empty_epoch() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        epoch_indices(10, 0, np.random.default_rng(0))


# --- ArrayData ----------------------------------------------------------------------------------


def test_from_shards_concatenates_masks_and_outlives_the_shard_sets(tmp_path: Path) -> None:
    a = make_flows(100, 4, seed=1)
    b = make_flows(60, 4, seed=2)
    sa = write_shards(tmp_path, a, name="a")
    sb = write_shards(tmp_path, b, name="b")
    keep = np.arange(60) % 2 == 0

    data = ArrayData.from_shards([sa, sb], [None, keep])
    sa.close()
    sb.close()  # the arrays must not depend on the (closed) memory maps

    assert len(data) == 100 + 30
    np.testing.assert_array_equal(data.ppi[:100], a.ppi)
    np.testing.assert_array_equal(data.label[100:], b.label[keep])


def test_from_shards_validates_its_arguments(tmp_path: Path) -> None:
    ss = write_shards(tmp_path, make_flows(10, 4, seed=1))
    with pytest.raises(ValueError, match="no shard sets"):
        ArrayData.from_shards([])
    with pytest.raises(ValueError, match="one to one"):
        ArrayData.from_shards([ss], [None, None])
    ss.close()


def test_array_data_rejects_inconsistent_shapes() -> None:
    good = make_flows(5, 4, seed=1)

    with pytest.raises(ValueError, match="ppi must be"):
        ArrayData(good.ppi[:, :10], good.ppi_len, good.label)
    with pytest.raises(ValueError, match="label must be"):
        ArrayData(good.ppi, good.ppi_len, good.label[:3])


def test_subsample_is_seeded_sorted_and_without_replacement() -> None:
    data = make_flows(500, 4, seed=1)

    a = data.subsample(100, np.random.default_rng(3))
    b = data.subsample(100, np.random.default_rng(3))

    assert len(a) == 100
    np.testing.assert_array_equal(a.ppi, b.ppi)
    assert data.subsample(1000, np.random.default_rng(0)) is data


# --- FlowBatches ---------------------------------------------------------------------------------


def test_batch_holds_exactly_what_the_feature_layer_produces(fitted) -> None:
    data, std = fitted
    fb = FlowBatches(data, SPACE, standardizer=std, views=("continuous", "tokens"), train=True)
    idx = np.array([5, 0, 17, 3])

    b = fb.batch(idx)

    assert b["cont"].shape == (4, P.K_MAX, P.PPI_CHANNELS) and b["cont"].dtype == np.float32
    assert b["tokens"].shape == (4, P.K_MAX, P.PPI_CHANNELS) and b["tokens"].dtype == np.int64
    assert b["mask"].dtype == bool and b["y"].dtype == np.int64
    np.testing.assert_array_equal(b["tokens"], F.tokenize(data.ppi[idx], data.ppi_len[idx]))
    np.testing.assert_array_equal(b["mask"], F.padding_mask(data.ppi_len[idx]))
    np.testing.assert_array_equal(b["y"], data.label[idx])  # ids 0..3 map to themselves
    np.testing.assert_array_equal(b["ppi_len"], data.ppi_len[idx])


def test_padded_positions_are_exactly_zero_after_standardising(fitted) -> None:
    """Standardising alone would move a pad from 0 to -mean/std."""
    data, std = fitted
    fb = FlowBatches(data, SPACE, standardizer=std, train=True)

    b = fb.batch(np.arange(len(data)))

    assert not b["cont"][~b["mask"]].any()
    raw = std.transform_continuous(F.continuous(data.ppi, data.ppi_len))
    np.testing.assert_allclose(b["cont"][b["mask"]], raw[b["mask"]])  # real packets untouched


def test_standardised_real_packets_have_zero_mean_and_unit_variance(fitted) -> None:
    data, std = fitted
    fb = FlowBatches(data, SPACE, standardizer=std, train=True)

    b = fb.batch(np.arange(len(data)))
    real = b["cont"][b["mask"]]

    np.testing.assert_allclose(real.mean(axis=0), 0, atol=1e-4)
    np.testing.assert_allclose(real.std(axis=0), 1, atol=1e-3)


def test_train_mode_refuses_unknown_class_flows(fitted) -> None:
    _, std = fitted
    mixed = make_flows(100, 6, seed=2, class_ids=[0, 1, 4])  # class 4 is held out

    with pytest.raises(ValueError, match="never be trained on"):
        FlowBatches(mixed, SPACE, standardizer=std, train=True)

    val = FlowBatches(mixed, SPACE, standardizer=std, train=False)  # fine for evaluation
    assert (val.y < 0).any() and set(val.y[val.known_indices]) <= {0, 1}
    assert val.class_counts().sum() == len(val.known_indices)


def test_a_label_from_another_dataset_is_an_error_not_unknown(fitted) -> None:
    _, std = fitted
    stray = make_flows(20, 12, seed=2, class_ids=[0, 9])

    with pytest.raises(Exception, match="neither the known nor the unknown"):
        FlowBatches(stray, SPACE, standardizer=std)


def test_continuous_view_requires_a_standardizer_but_tokens_do_not(fitted) -> None:
    data, _ = fitted

    with pytest.raises(ValueError, match="needs a Standardizer"):
        FlowBatches(data, SPACE, standardizer=None, views=("continuous",))
    tokens_only = FlowBatches(data, SPACE, standardizer=None, views=("tokens",), train=True)
    assert set(tokens_only.batch(np.arange(3))) == {"mask", "ppi_len", "y", "tokens"}
    assert tokens_only.standardizer_hash is None


@pytest.mark.parametrize("views", [(), ("nonsense",), ("continuous", "nonsense")])
def test_bad_views_are_rejected(fitted, views) -> None:
    data, std = fitted

    with pytest.raises(ValueError, match="views"):
        FlowBatches(data, SPACE, standardizer=std, views=views)


def test_iter_batches_covers_every_flow_once_and_known_only_skips_unknown(fitted) -> None:
    _, std = fitted
    mixed = make_flows(103, 6, seed=2, class_ids=[0, 1, 4])
    fb = FlowBatches(mixed, SPACE, standardizer=std)

    seen = np.concatenate([idx for idx, _ in fb.iter_batches(16)])
    known = np.concatenate([idx for idx, _ in fb.iter_batches(16, known_only=True)])

    np.testing.assert_array_equal(seen, np.arange(103))
    np.testing.assert_array_equal(known, fb.known_indices)
    assert (fb.y[known] >= 0).all()


def test_to_torch_keeps_dtypes(fitted) -> None:
    import torch

    data, std = fitted
    b = to_torch(
        FlowBatches(data, SPACE, standardizer=std, views=("continuous", "tokens")).batch(
            np.arange(4)
        )
    )

    assert b["cont"].dtype == torch.float32 and b["tokens"].dtype == torch.int64
    assert b["mask"].dtype == torch.bool and b["y"].dtype == torch.int64


# --- the standardizer is never fit here ----------------------------------------------------------


def test_nothing_in_the_data_layer_can_refit_a_standardizer(
    fitted, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behavioural, not just nominal: with ``Standardizer.fit`` booby-trapped,
    building batches and iterating them must still work, so no path in the
    data layer calls it."""
    data, std = fitted

    def boom(*a, **k):
        raise AssertionError("the data layer tried to fit a Standardizer")

    monkeypatch.setattr(Standardizer, "fit", boom)

    fb = FlowBatches(data, SPACE, standardizer=std, views=("continuous", "tokens"), train=True)
    list(fb.iter_batches(64))
    fb.batch(np.arange(8))
    epoch_indices(len(fb), len(fb), np.random.default_rng(0), balanced_weights(fb.y, 4))

    assert not [n for n in dir(D) if "fit" in n.lower()]
    assert not [n for n in (*dir(FlowBatches), *dir(ArrayData)) if "fit" in n.lower()]


# --- ArrayData.prefix (how per-K models see a flow at nominal K) ----------------------------------


def test_prefix_zeroes_later_packets_and_caps_the_length_and_leaves_short_flows_whole() -> None:
    data = make_flows(60, 4, seed=1, min_len=2, max_len=12)
    k = 6

    cut = data.prefix(k)

    assert not cut.ppi[:, k:, :].any()  # nothing after packet k survives
    np.testing.assert_array_equal(cut.ppi_len, np.minimum(data.ppi_len, k))
    np.testing.assert_array_equal(cut.ppi[:, :k], data.ppi[:, :k])  # the first k are untouched
    short = data.ppi_len <= k
    assert short.any() and (~short).any()
    np.testing.assert_array_equal(cut.ppi[short], data.ppi[short])  # a shorter flow is whole
    np.testing.assert_array_equal(cut.label, data.label)
    assert cut.ppi_len.dtype == data.ppi_len.dtype
    assert data.ppi_len.max() > k, "the original was mutated"  # prefix() must not modify in place


def test_prefix_of_a_long_flow_equals_a_native_flow_of_that_length() -> None:
    """A 12-packet flow cut to K=5 is indistinguishable from a 5-packet flow."""
    data = make_flows(1, 4, seed=3, min_len=12, max_len=12)
    native = ArrayData(data.ppi.copy(), np.array([5], dtype=data.ppi_len.dtype), data.label)
    native.ppi[:, 5:, :] = 0

    cut = data.prefix(5)

    np.testing.assert_array_equal(cut.ppi, native.ppi)
    np.testing.assert_array_equal(cut.ppi_len, native.ppi_len)


@pytest.mark.parametrize("bad", [0, -1, 31, 2.5, True, "3"])
def test_prefix_rejects_a_bad_k(bad) -> None:
    with pytest.raises(ValueError, match="k must be"):
        make_flows(3, 4, seed=1).prefix(bad)
