"""LabelSpace and the losses (plan phase 2 T5). Expected values are worked out
by hand or with a NumPy reimplementation, not by calling the code under test."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from adl_etc.training.labels import LabelSpace, LabelSpaceError
from adl_etc.training.losses import cross_entropy_ls, multi_prefix_ce

# --- LabelSpace -----------------------------------------------------------------------------


def space() -> LabelSpace:
    return LabelSpace.create(known=[10, 3, 7, 3], unknown=[5, 99], names={3: "c", 7: "g", 5: "u"})


def test_create_sorts_and_dedupes() -> None:
    s = space()

    assert s.known == (3, 7, 10) and s.unknown == (5, 99) and s.n_classes == 3


def test_to_model_maps_known_to_contiguous_indices_and_unknown_to_minus_one() -> None:
    s = space()

    got = s.to_model(np.array([3, 7, 10, 5, 99, -1, 3]))

    assert got.tolist() == [0, 1, 2, -1, -1, -1, 0]


def test_to_dataset_is_the_inverse_on_known_classes() -> None:
    s = space()
    ids = np.array([10, 3, 7, 3])

    assert s.to_dataset(s.to_model(ids)).tolist() == ids.tolist()
    assert s.to_dataset(np.array([-1, 0])).tolist() == [-1, 3]


def test_an_id_in_neither_set_raises_rather_than_becoming_unknown() -> None:
    """A label map from the wrong dataset must not silently read as 'unknown'."""
    with pytest.raises(LabelSpaceError, match=r"\[42\]"):
        space().to_model(np.array([3, 42]))


def test_to_dataset_rejects_out_of_range_indices() -> None:
    with pytest.raises(LabelSpaceError, match="out of range"):
        space().to_dataset(np.array([3]))
    with pytest.raises(LabelSpaceError, match="out of range"):
        space().to_dataset(np.array([-2]))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(known=[]), "at least one"),
        (dict(known=[1, 2], unknown=[2]), "both known and unknown"),
        (dict(known=[-1]), "non-negative"),
        (dict(known=[1], names={9: "x"}), "not in the label space"),
    ],
)
def test_invalid_label_spaces_are_rejected(kwargs, message) -> None:
    with pytest.raises(LabelSpaceError, match=message):
        LabelSpace.create(**kwargs)


def test_from_label_map_holds_out_the_requested_classes() -> None:
    s = LabelSpace.from_label_map({"a": 0, "b": 1, "c": 2}, unknown=[1])

    assert s.known == (0, 2) and s.unknown == (1,)
    assert s.class_names == ["a", "c"]  # model order, not dataset order
    with pytest.raises(LabelSpaceError, match="not in the label map"):
        LabelSpace.from_label_map({"a": 0}, unknown=[5])


def test_hash_changes_with_any_part_of_the_space() -> None:
    base = space()

    assert base.hash == space().hash
    assert LabelSpace.create([3, 7, 10, 11], [5, 99], base.names).hash != base.hash
    assert LabelSpace.create([3, 7, 10], [5], base.names).hash != base.hash
    assert LabelSpace.create([3, 7, 10], [5, 99], {3: "renamed"}).hash != base.hash


def test_save_and_load_round_trip_and_detect_tampering(tmp_path: Path) -> None:
    path = tmp_path / "label_space.json"
    space().save(path)

    assert LabelSpace.load(path) == space()

    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["known"] = [3, 7, 11]  # edited by hand, hash left stale
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(LabelSpaceError, match="hash"):
        LabelSpace.load(path)

    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(LabelSpaceError, match="unreadable"):
        LabelSpace.load(path)


# --- losses ---------------------------------------------------------------------------------


def np_log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def np_ce(logits: np.ndarray, y: np.ndarray, smoothing: float) -> np.ndarray:
    """Label-smoothed CE per row: (1-e) * -logp[y] + e * mean_c(-logp[c])."""
    logp = np_log_softmax(logits)
    nll = -logp[np.arange(len(y)), y]
    uniform = -logp.mean(axis=-1)
    return (1 - smoothing) * nll + smoothing * uniform


def test_cross_entropy_matches_the_hand_formula() -> None:
    logits = np.array([[2.0, 0.0, -1.0], [0.1, 0.2, 0.3]])
    y = np.array([0, 2])

    got = cross_entropy_ls(torch.tensor(logits), torch.tensor(y), smoothing=0.1)

    assert float(got) == pytest.approx(np_ce(logits, y, 0.1).mean())


def test_zero_smoothing_is_plain_negative_log_likelihood() -> None:
    logits = np.array([[1.0, 2.0]])

    got = cross_entropy_ls(torch.tensor(logits), torch.tensor([1]), smoothing=0.0)

    assert float(got) == pytest.approx(-math.log(math.exp(2) / (math.exp(1) + math.exp(2))))


def test_multi_prefix_ce_is_the_pooled_masked_mean() -> None:
    """2 flows x 3 positions, flow 1 has only 2 real packets."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2, 3, 4))
    y = np.array([1, 3])
    mask = np.array([[True, True, True], [True, True, False]])

    per_pos = np.stack([np_ce(logits[i], np.full(3, y[i]), 0.1) for i in range(2)])
    expected = per_pos[mask].mean()  # 5 valid terms, each counted once

    got = multi_prefix_ce(torch.tensor(logits), torch.tensor(y), torch.tensor(mask), 0.1)

    assert float(got) == pytest.approx(expected)


def test_one_valid_position_equals_plain_cross_entropy() -> None:
    logits = np.random.default_rng(1).normal(size=(3, 5, 4))
    y = np.array([0, 2, 3])
    mask = np.zeros((3, 5), dtype=bool)
    mask[:, 2] = True  # only position 2 counts

    got = multi_prefix_ce(torch.tensor(logits), torch.tensor(y), torch.tensor(mask), 0.1)
    plain = cross_entropy_ls(torch.tensor(logits[:, 2]), torch.tensor(y), 0.1)

    assert float(got) == pytest.approx(float(plain))


def test_masked_positions_have_no_effect_and_no_gradient() -> None:
    logits = torch.randn(2, 4, 3, requires_grad=True)
    y = torch.tensor([0, 1])
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    multi_prefix_ce(logits, y, mask, 0.1).backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    assert torch.count_nonzero(logits.grad[mask]) > 0

    # Garbage in a padded slot changes nothing.
    tampered = logits.detach().clone()
    tampered[~mask] = 1e9
    a = multi_prefix_ce(logits.detach(), y, mask, 0.1)
    b = multi_prefix_ce(tampered, y, mask, 0.1)
    assert float(a) == pytest.approx(float(b))


def test_a_fully_masked_batch_gives_zero_not_nan() -> None:
    got = multi_prefix_ce(
        torch.randn(2, 3, 4), torch.tensor([0, 1]), torch.zeros(2, 3, dtype=torch.bool), 0.1
    )

    assert float(got) == 0.0


def test_loss_shapes_are_validated() -> None:
    with pytest.raises(ValueError, match=r"logits\[B, C\]"):
        cross_entropy_ls(torch.zeros(2, 3, 4), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match=r"logits\[B, K, C\]"):
        multi_prefix_ce(torch.zeros(2, 4), torch.tensor([0, 1]), torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="mask must be"):
        multi_prefix_ce(
            torch.zeros(2, 3, 4), torch.tensor([0, 1]), torch.ones(2, 2, dtype=torch.bool)
        )
