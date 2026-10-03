"""Turn a trained torch baseline into a :class:`~adl_etc.evaluation.dense_logits.DenseLogits`.

A model declares ``causal``. A causal model (the GRU) is run once and every prefix
comes out of that one pass. A non-causal one (the CNN) is run once per nominal K
on flows truncated to their first K packets. Getting this wrong is silent, so the
choice lives in one place and not in each caller.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from adl_etc.data import ppi as P
from adl_etc.data.features import Standardizer
from adl_etc.evaluation.dense_logits import DenseLogits, from_causal, from_per_k, grid_from
from adl_etc.training.datasets import ArrayData, FlowBatches, to_torch
from adl_etc.training.labels import LabelSpace


@torch.no_grad()
def predict_dense(
    model: torch.nn.Module,
    data: ArrayData,
    label_space: LabelSpace,
    standardizer: Standardizer | None,
    *,
    ks: Iterable[int] | None = None,
    batch_size: int = 4096,
    device: str = "cpu",
    dtype: type = np.float32,
) -> DenseLogits:
    """Logits for every flow in ``data`` (unknown-class flows included: an
    open-set evaluation needs them). ``ks`` only matters for non-causal models."""
    dev = torch.device(device)
    views = tuple(getattr(model, "views", ("continuous",)))
    causal = bool(getattr(model, "causal"))  # noqa: B009 - required attribute, fail loudly
    was_training = model.training
    model.eval()
    model.to(dev)
    n_classes = label_space.n_classes

    def run(flows: ArrayData) -> np.ndarray:
        fb = FlowBatches(flows, label_space, standardizer=standardizer, views=views)
        out = None
        for idx, np_batch in fb.iter_batches(batch_size):
            logits = model(to_torch(np_batch, dev)).float().cpu().numpy()
            if out is None:
                out = np.zeros((len(flows), *logits.shape[1:]), dtype=dtype)
            out[idx] = logits
        assert out is not None
        return out

    try:
        if causal:
            dense = run(data)
            if dense.shape[1:] != (P.K_MAX, n_classes):
                raise ValueError(f"a causal model must emit [B, {P.K_MAX}, C], got {dense.shape}")
            return from_causal(dense)
        grid = grid_from(ks)
        per_k = {}
        for k in grid:
            arr = run(data.prefix(k))
            if arr.shape[1:] != (n_classes,):
                raise ValueError(f"a per-K model must emit [B, C], got {arr.shape}")
            per_k[k] = arr
        return from_per_k(per_k, dtype=dtype)
    finally:
        model.train(was_training)
