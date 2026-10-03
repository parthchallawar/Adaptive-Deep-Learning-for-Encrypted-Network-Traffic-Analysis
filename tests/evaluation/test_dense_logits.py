"""DenseLogits: the two silent-wrong-number failure modes it exists to prevent
(plan phase 2 F4, and the per-K/short-flow refinement found while building T6)."""

from __future__ import annotations

import numpy as np
import pytest

from adl_etc.evaluation.dense_logits import (
    K_GRID,
    DenseLogits,
    NotEvaluatedError,
    from_causal,
    from_per_k,
    grid_from,
)
from adl_etc.evaluation.metrics import logits_at_k


def test_the_grid_is_specs_thirteen_values() -> None:
    assert K_GRID == (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30) and len(K_GRID) == 13


# --- failure mode 1: a grid read positionally ----------------------------------------------


def test_a_request_for_a_k_outside_the_grid_raises_instead_of_returning_another_k() -> None:
    per_k = {k: np.full((2, 3), float(k)) for k in K_GRID}
    dense = from_per_k(per_k)
    ppi_len = np.array([30, 30])

    assert dense.at(8, ppi_len)[0, 0] == 8.0  # K=8 is in the grid and returns K=8's row
    with pytest.raises(NotEvaluatedError, match="K=7 was not evaluated"):
        dense.at(7, ppi_len)
    # The bug this prevents: reading a 13-wide grid positionally (index k-1, as
    # metrics.logits_at_k does) makes a request for K=8 return K=10's row.
    assert np.asarray([per_k[k][0, 0] for k in K_GRID])[8 - 1] == 10.0


# --- failure mode 2: per-K models and short flows ---------------------------------------------


def test_per_k_models_keep_their_own_answer_for_a_short_flow() -> None:
    """A 7-packet flow under the K=10 and K=12 models: both see all 7 packets, but
    they are different models. Position must be the nominal K, so neither overwrites
    the other. (With the effective-K rule both would land on slot 6.)"""
    ppi_len = np.array([7])
    k10 = np.array([[5.0, 0.0, 0.0]])
    k12 = np.array([[0.0, 5.0, 0.0]])

    dense = from_per_k({10: k10, 12: k12})

    assert dense.at(10, ppi_len).argmax() == 0
    assert dense.at(12, ppi_len).argmax() == 1

    naive = np.zeros((1, 30, 3))
    for k, out in ((10, k10), (12, k12)):
        naive[0, min(k, 7) - 1] = out
    assert logits_at_k(naive, ppi_len, 10).argmax() == 1  # the old contract reads the wrong model


def test_causal_arrays_use_the_effective_position_for_short_flows() -> None:
    """A causal model's output at packet e is the prefix-e answer, so a 4-packet
    flow at nominal K=10 correctly reads position 3."""
    logits = np.arange(2 * 30 * 2, dtype=float).reshape(2, 30, 2)
    dense = from_causal(logits)

    got = dense.at(10, np.array([4, 30]))

    np.testing.assert_array_equal(got[0], logits[0, 3])  # min(10, 4) - 1
    np.testing.assert_array_equal(got[1], logits[1, 9])  # min(10, 30) - 1
    assert dense.evaluated_k == tuple(range(1, 31))
    # and it agrees with the metrics module's own rule
    np.testing.assert_array_equal(got, logits_at_k(logits, np.array([4, 30]), 10))


def test_nominal_arrays_ignore_flow_length() -> None:
    dense = from_per_k({5: np.ones((2, 2)), 30: np.full((2, 2), 2.0)})

    assert dense.at(30, np.array([3, 30])).tolist() == [[2.0, 2.0], [2.0, 2.0]]


# --- construction and validation ------------------------------------------------------------------


def test_from_per_k_stores_each_k_at_its_own_position_and_leaves_the_rest_unreadable() -> None:
    dense = from_per_k({3: np.full((1, 2), 3.0), 9: np.full((1, 2), 9.0)})

    assert dense.evaluated_k == (3, 9) and dense.logits.shape == (1, 30, 2)
    assert dense.logits[0, 2, 0] == 3.0 and dense.logits[0, 8, 0] == 9.0
    assert not dense.logits[0, 3].any()  # unevaluated slots are zeros, and not readable
    with pytest.raises(NotEvaluatedError):
        dense.at(4, np.array([30]))


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: DenseLogits(np.zeros((2, 10, 3)), (1,), "nominal"), "K_MAX|must be"),
        (lambda: DenseLogits(np.zeros((2, 30, 3)), (1,), "sideways"), "indexing"),
        (lambda: DenseLogits(np.zeros((2, 30, 3)), (), "nominal"), "evaluated_k"),
        (lambda: DenseLogits(np.zeros((2, 30, 3)), (5, 2), "nominal"), "evaluated_k"),
        (lambda: DenseLogits(np.zeros((2, 30, 3)), (31,), "nominal"), "evaluated_k"),
        (lambda: from_per_k({}), "empty"),
        (lambda: from_per_k({1: np.zeros((2, 3)), 2: np.zeros((2, 4))}), "expected"),
    ],
)
def test_invalid_arrays_are_rejected(build, message) -> None:
    with pytest.raises(ValueError, match=message):
        build()


def test_at_validates_its_arguments() -> None:
    dense = from_per_k({5: np.zeros((3, 2))})

    with pytest.raises(ValueError, match="ppi_len must be"):
        dense.at(5, np.array([1, 2]))
    with pytest.raises(TypeError, match="int"):
        dense.at(5.0, np.array([1, 2, 3]))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        dense.at(True, np.array([1, 2, 3]))


def test_grid_from_defaults_dedupes_and_bounds() -> None:
    assert grid_from(None) == K_GRID
    assert grid_from([5, 1, 5, 3]) == (1, 3, 5)
    for bad in ([], [0], [31]):
        with pytest.raises(ValueError, match="prefix lengths"):
            grid_from(bad)


def test_predictions_and_class_count() -> None:
    dense = from_causal(np.eye(3)[None, None, :, :].repeat(30, axis=1).reshape(3, 30, 3)[:, :, :])

    assert dense.n_classes == 3 and len(dense) == 3
    assert dense.predictions_at(5, np.array([30, 30, 30])).shape == (3,)
