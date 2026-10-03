"""EvalArrays persistence and the 250k eval-flow cap (spec 004, plan T7)."""

from __future__ import annotations

import numpy as np
import pytest

from adl_etc.data import ppi as P
from adl_etc.evaluation import artifact as A
from adl_etc.evaluation.dense_logits import from_causal, from_per_k


def make_eval_arrays(n: int = 5, c: int = 3, *, per_k: bool = False) -> A.EvalArrays:
    rng = np.random.default_rng(0)
    if per_k:
        grid = (1, 5, 10, 30)
        dense = from_per_k({k: rng.normal(size=(n, c)) for k in grid})
    else:
        dense = from_causal(rng.normal(size=(n, P.K_MAX, c)))
    return A.EvalArrays(
        dense=dense,
        labels=rng.integers(-1, c, size=n).astype(np.int64),
        ppi_len=rng.integers(1, P.K_MAX + 1, size=n).astype(np.int64),
        flow_index=np.arange(n, dtype=np.int64) * 7,  # a non-trivial mapping
    )


# --- the eval cap ----------------------------------------------------------------------


def test_subsample_keeps_everything_under_the_cap():
    idx = A.subsample_for_eval(100, cap=250_000, seed=0)
    np.testing.assert_array_equal(idx, np.arange(100))


def test_subsample_keeps_everything_exactly_at_the_cap():
    idx = A.subsample_for_eval(100, cap=100, seed=0)
    np.testing.assert_array_equal(idx, np.arange(100))


def test_subsample_is_a_sorted_seeded_draw_over_the_cap():
    idx_a = A.subsample_for_eval(1000, cap=100, seed=0)
    idx_b = A.subsample_for_eval(1000, cap=100, seed=0)
    idx_c = A.subsample_for_eval(1000, cap=100, seed=1)

    assert len(idx_a) == 100
    assert list(idx_a) == sorted(idx_a)  # ascending, not draw order
    np.testing.assert_array_equal(idx_a, idx_b)  # same seed -> same draw
    assert not np.array_equal(idx_a, idx_c)  # a different seed changes it
    assert len(set(idx_a.tolist())) == 100  # no replacement


def test_subsample_never_stratifies_by_class():
    """A uniform draw over 90/10 classes keeps roughly 90/10 -- a stratified
    sampler would flatten it, which the drift study (this cap's whole reason
    for existing) must never see happen silently."""
    n = 20_000
    labels = np.array([0] * 18_000 + [1] * 2_000)
    idx = A.subsample_for_eval(n, cap=2_000, seed=3)
    kept = labels[idx]
    assert abs(float(np.mean(kept == 0)) - 0.9) < 0.02


@pytest.mark.parametrize("bad", [0, -1])
def test_subsample_rejects_a_non_positive_cap(bad):
    with pytest.raises(ValueError, match="cap"):
        A.subsample_for_eval(10, cap=bad, seed=0)


def test_subsample_rejects_a_negative_n():
    with pytest.raises(ValueError, match="n"):
        A.subsample_for_eval(-1, cap=10, seed=0)


def test_subsample_of_zero_flows_is_empty_not_an_error():
    assert A.subsample_for_eval(0, cap=10, seed=0).size == 0


# --- round trip --------------------------------------------------------------------------


@pytest.mark.parametrize("per_k", [False, True])
def test_round_trip_preserves_everything_but_logits_precision(tmp_path, per_k):
    ea = make_eval_arrays(per_k=per_k)
    ea.save(tmp_path)
    loaded = A.EvalArrays.load(tmp_path)

    assert loaded.dense.indexing == ea.dense.indexing
    assert loaded.dense.evaluated_k == ea.dense.evaluated_k
    np.testing.assert_array_equal(loaded.labels, ea.labels)
    np.testing.assert_array_equal(loaded.ppi_len, ea.ppi_len)
    np.testing.assert_array_equal(loaded.flow_index, ea.flow_index)
    # fp16 round trip: close, not bit-exact.
    np.testing.assert_allclose(loaded.dense.logits, ea.dense.logits, atol=1e-2, rtol=1e-2)


def test_round_trip_preserves_argmax_decisions():
    """The thing that actually matters: fp16 must never flip which class
    wins, given a real margin (spec 004 only claims this, not exact logit
    fidelity). A plain random draw can land two logits within fp16's own
    precision purely by chance, so the winner here is built with a fixed,
    much larger gap instead of hoping for one."""
    rng = np.random.default_rng(1)
    n, k_max, c = 200, P.K_MAX, 20
    logits = rng.normal(scale=1.0, size=(n, k_max, c))
    winners = rng.integers(0, c, size=(n, k_max))
    idx = np.indices((n, k_max))
    logits[idx[0], idx[1], winners] += 10.0  # a gap far beyond fp16's precision at this scale
    ea = A.EvalArrays(
        dense=from_causal(logits),
        labels=rng.integers(0, 20, size=200).astype(np.int64),
        ppi_len=np.full(200, P.K_MAX, dtype=np.int64),
        flow_index=np.arange(200, dtype=np.int64),
    )
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        ea.save(d)
        loaded = A.EvalArrays.load(d)
    before = ea.dense.logits.argmax(axis=-1)
    after = loaded.dense.logits.argmax(axis=-1)
    np.testing.assert_array_equal(before, after)


def test_logits_are_stored_as_float16_not_float32(tmp_path):
    """The module docstring's whole sizing story (9 KB/flow, ~2.25 GB for the
    250k cap) depends on this; a silent upgrade to float32 would double it
    without any test noticing via value closeness alone."""
    make_eval_arrays().save(tmp_path)
    on_disk = np.load(tmp_path / A.LOGITS_FILE)
    assert on_disk.dtype == np.float16


def test_writing_creates_exactly_the_documented_files(tmp_path):
    make_eval_arrays().save(tmp_path)
    on_disk = {p.name for p in tmp_path.iterdir()}
    assert on_disk == {
        A.LOGITS_FILE,
        A.DENSE_LOGITS_FILE,
        A.LABELS_FILE,
        A.PPI_LEN_FILE,
        A.FLOW_INDEX_FILE,
    }


def test_dense_logits_json_is_plain_and_human_readable(tmp_path):
    make_eval_arrays(per_k=True).save(tmp_path)
    import json

    meta = json.loads((tmp_path / A.DENSE_LOGITS_FILE).read_text(encoding="utf-8"))
    assert meta == {"indexing": "nominal", "evaluated_k": [1, 5, 10, 30]}


def test_a_mismatched_array_length_is_rejected():
    dense = from_causal(np.zeros((3, P.K_MAX, 2)))
    with pytest.raises(ValueError, match="labels"):
        A.EvalArrays(
            dense=dense,
            labels=np.zeros(2, dtype=np.int64),  # wrong length
            ppi_len=np.zeros(3, dtype=np.int64),
            flow_index=np.zeros(3, dtype=np.int64),
        )
