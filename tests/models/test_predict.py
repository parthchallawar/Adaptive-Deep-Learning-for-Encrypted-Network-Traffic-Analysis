"""predict_dense (plan phase 2 T6)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from adl_etc.data.features import Standardizer
from adl_etc.evaluation.dense_logits import K_GRID, NotEvaluatedError
from adl_etc.models.baselines import CNNBaseline, RNNBaseline
from adl_etc.models.baselines.predict import predict_dense
from adl_etc.training.datasets import ArrayData
from adl_etc.training.labels import LabelSpace
from tests.training.util import make_flows, write_shards

C = 4
SPACE = LabelSpace.create(range(C))


@pytest.fixture(scope="module")
def std(tmp_path_factory: pytest.TempPathFactory) -> Standardizer:
    root = tmp_path_factory.mktemp("std")
    shards = write_shards(root, make_flows(600, C, seed=1))
    out = Standardizer.fit(shards)
    shards.close()
    return out


@pytest.fixture(scope="module")
def flows() -> ArrayData:
    return make_flows(120, C, seed=7, min_len=3, max_len=12)


def test_a_causal_model_is_run_once_and_holds_every_k(flows: ArrayData, std: Standardizer) -> None:
    torch.manual_seed(0)
    model = RNNBaseline(C, hidden=32, stem=16)

    dense = predict_dense(model, flows, SPACE, std, batch_size=32)

    assert dense.indexing == "effective" and dense.evaluated_k == tuple(range(1, 31))
    assert dense.logits.shape == (120, 30, C)
    # it agrees with running the whole thing through the model directly
    from adl_etc.training.datasets import FlowBatches, to_torch

    fb = FlowBatches(flows, SPACE, standardizer=std)
    with torch.no_grad():
        direct = model.eval()(to_torch(fb.batch(np.arange(120)))).numpy()
    np.testing.assert_allclose(dense.logits, direct, atol=1e-5)


def test_a_non_causal_model_is_run_once_per_grid_k(flows: ArrayData, std: Standardizer) -> None:
    torch.manual_seed(0)

    dense = predict_dense(CNNBaseline(C, channels=(8, 8, 8)), flows, SPACE, std)

    assert dense.indexing == "nominal" and dense.evaluated_k == K_GRID
    with pytest.raises(NotEvaluatedError):
        dense.at(7, flows.ppi_len)  # 7 is not on the grid


def test_a_custom_grid_is_respected(flows: ArrayData, std: Standardizer) -> None:
    dense = predict_dense(CNNBaseline(C, channels=(8, 8, 8)), flows, SPACE, std, ks=[2, 9])

    assert dense.evaluated_k == (2, 9)


def test_a_per_k_prediction_depends_only_on_the_first_k_packets(
    flows: ArrayData, std: Standardizer
) -> None:
    """Truncation is real: scrambling every packet after K leaves K's logits unchanged."""
    torch.manual_seed(0)
    model = CNNBaseline(C, channels=(8, 8, 8))
    k = 5
    scrambled = ArrayData(flows.ppi.copy(), flows.ppi_len, flows.label)
    rng = np.random.default_rng(0)
    scrambled.ppi[:, k:, 2] = rng.integers(1, 1500, scrambled.ppi[:, k:, 2].shape)  # sizes

    a = predict_dense(model, flows, SPACE, std, ks=[k])
    b = predict_dense(model, scrambled, SPACE, std, ks=[k])

    np.testing.assert_allclose(a.at(k, flows.ppi_len), b.at(k, flows.ppi_len), atol=1e-5)


def test_prediction_includes_unknown_class_flows_and_restores_the_model_mode(
    std: Standardizer,
) -> None:
    space = LabelSpace.create(range(C), unknown=[9])
    mixed = make_flows(40, 10, seed=3, class_ids=[0, 1, 9])
    model = CNNBaseline(C, channels=(8, 8, 8)).train()

    dense = predict_dense(model, mixed, space, std, ks=[3])

    assert len(dense) == 40  # an open-set evaluation needs the unknown flows too
    assert model.training is True  # left as it was found


def test_a_model_emitting_the_wrong_shape_is_rejected(flows: ArrayData, std: Standardizer) -> None:
    class BadCausal(RNNBaseline):
        def forward(self, batch):  # type: ignore[no-untyped-def]
            return super().forward(batch)[:, :10]

    class BadPerK(CNNBaseline):
        def forward(self, batch):  # type: ignore[no-untyped-def]
            return super().forward(batch)[:, None, :]

    with pytest.raises(ValueError, match="causal model must emit"):
        predict_dense(BadCausal(C, hidden=8, stem=8), flows, SPACE, std)
    with pytest.raises(ValueError, match="per-K model must emit"):
        predict_dense(BadPerK(C, channels=(8, 8, 8)), flows, SPACE, std, ks=[3])
