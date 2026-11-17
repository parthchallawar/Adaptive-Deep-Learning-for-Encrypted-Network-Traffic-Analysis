"""The one model pass behind a :class:`~adl_etc.evaluation.scores.ScoreArrays` (plan phase 4, T1).

For every flow in chunks (default 1,024) it runs the model once, reduces the per-position heads
to the scalar columns the policies need, and discards them: nothing larger than one chunk's heads
is ever held, and the 128-d projection is never stored (plan P4-F2, ADR-012). Needs torch; the
NumPy-only logic lives in :mod:`adl_etc.evaluation.scores` and :mod:`adl_etc.inference.unknown`.

Two entry points:

- :func:`score_flows`: all score columns for a sample of flows (optionally including the
  relative-Mahalanobis column, given a fitted :class:`~adl_etc.inference.unknown.GaussianBank`).
- :func:`accumulate_projection_sums`: the running sums the Gaussian bank is fitted from, over
  known-class training flows.

A model without the PAT's heads (the GRU baseline) still gets every column it can: its class
logits give ``pred``/``conf``/``margin``/the energies; ``p_safe`` falls back to ``conf``, and
``mahal``, ``pkt_nll`` and ``next_entropy`` are zeros (``meta["heads"] == "cls-only"``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from adl_etc.data import ppi as P
from adl_etc.data.features import Standardizer
from adl_etc.evaluation import scores as S
from adl_etc.evaluation.calibration import Calibration
from adl_etc.inference.unknown import GaussianBank, GaussianSums
from adl_etc.training.datasets import ArrayData, FlowBatches, to_torch
from adl_etc.training.labels import LabelSpace

PAT_HEADS = ("cls", "safe", "proj", "next_size", "next_ipt", "next_dir")


def has_pat_heads(model: torch.nn.Module) -> bool:
    return hasattr(model, "head_safe") and hasattr(model, "head_proj")


def identity_calibration(n_classes: int, checkpoint_hash: str = "") -> Calibration:
    """No prior adjustment, temperature 1, and a safety map that is the identity on
    ``sigmoid(safe)``: for a model that has no calibration file (the baselines)."""
    from adl_etc.evaluation.calibration import SafeMap

    ident = SafeMap(kind="isotonic", thresholds=[0.0, 1.0], values=[0.0, 1.0])
    return Calibration(
        checkpoint_hash, [0.0] * n_classes, [1.0] * S.K_MAX, [ident] * S.K_MAX, [0] * S.K_MAX
    )


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of ``softmax(logits)`` divided by ``log(V)`` (the model's own normalisation)."""
    logp = logits.float().log_softmax(-1)
    return -(logp.exp() * logp).sum(-1) / math.log(logits.shape[-1])


def next_packet_columns(
    out: dict[str, torch.Tensor], tokens: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """``(pkt_nll, next_entropy)``, both ``[B, 30]`` float32.

    ``pkt_nll[:, j]`` (j >= 1) is the negative log-likelihood of packet ``j + 1`` under the
    next-packet heads read after ``j`` packets (head index ``j - 1``), summed over the size, IPT
    and direction channels; 0 where that packet does not exist, and at ``j = 0`` (no head reads
    before packet 1).
    ``next_entropy[:, k-1]`` is the mean normalised entropy of the three heads after ``k`` packets,
    exactly the feature the ``safe`` head sees."""
    heads = (
        (out["next_size"], P.SIZE_POS),
        (out["next_ipt"], P.IPT_POS),
        (out["next_dir"], P.DIR_POS),
    )
    nll = torch.zeros(tokens.shape[0], S.K_MAX, device=tokens.device)
    exists = tokens[:, 1:, P.SIZE_POS] != 0  # packet j + 1 is real (token 0 = no packet)
    for logits, pos in heads:
        logp = logits[:, :-1].float().log_softmax(-1)  # position j - 1 predicts packet j + 1
        target = tokens[:, 1:, pos].long()
        picked = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        nll[:, 1:] += torch.where(exists, -picked, torch.zeros_like(picked))
    entropy = (_entropy(heads[0][0]) + _entropy(heads[1][0]) + _entropy(heads[2][0])) / 3
    return nll.cpu().numpy(), entropy.cpu().numpy()


@torch.no_grad()
def score_flows(
    model: torch.nn.Module,
    data: ArrayData,
    label_space: LabelSpace,
    standardizer: Standardizer | None,
    calibration: Calibration,
    *,
    bank: GaussianBank | None = None,
    flow_index: np.ndarray | None = None,
    period: np.ndarray | None = None,
    meta: dict[str, Any] | None = None,
    batch_size: int = 1024,
    device: str = "cpu",
) -> S.ScoreArrays:
    """Score every flow of ``data``. ``flow_index``/``period`` default to ``arange``/zeros."""
    if not bool(getattr(model, "causal", False)):
        raise ValueError("a score pass needs a causal model (one pass gives every prefix)")
    dev = torch.device(device)
    views = tuple(getattr(model, "views", ("continuous",)))
    key = getattr(model, "logits_key", None)
    pat = has_pat_heads(model)
    if bank is not None and not pat:
        raise ValueError("a Gaussian bank needs the PAT's projection head")
    was_training = model.training
    model.eval()
    model.to(dev)

    n = len(data)
    cols = {name: np.zeros((n, S.K_MAX), dtype=dt) for name, dt in S.FIELDS_2D.items()}
    labels = label_space.to_model(data.label).astype(np.int64)
    ppi_len = np.asarray(data.ppi_len, dtype=np.int64)
    adjustment = np.asarray(calibration.adjustment)
    temps = np.asarray(calibration.temperatures)

    try:
        fb = FlowBatches(data, label_space, standardizer=standardizer, views=views)
        for idx, np_batch in fb.iter_batches(batch_size):
            batch = to_torch(np_batch, dev)
            result = model(batch)
            cls = (result[key] if key is not None else result).float().cpu().numpy()
            c = S.class_scores(cls, adjustment, temps)
            plen = ppi_len[idx]
            zeros = np.zeros((len(idx), S.K_MAX))
            block: dict[str, np.ndarray] = {**c, "mahal": zeros}
            if pat:
                safe = result["safe"].float().cpu().numpy()
                block["p_safe"] = calibration.safe_probability(safe)
                block["pkt_nll"], block["next_entropy"] = next_packet_columns(
                    result, batch["tokens"]
                )
                if bank is not None:
                    block["mahal"] = bank.mahal(result["proj"].float().cpu().numpy())
            else:
                block.update(p_safe=c["conf"], pkt_nll=zeros, next_entropy=zeros)
            for name, dt in S.FIELDS_2D.items():
                cols[name][idx] = S.fill_forward(np.asarray(block[name]), plen).astype(dt)
    finally:
        model.train(was_training)

    info = {"heads": "pat" if pat else "cls-only", **(meta or {})}
    return S.ScoreArrays(
        **cols,
        labels=labels,
        ppi_len=ppi_len,
        flow_index=np.arange(n, dtype=np.int64) if flow_index is None else np.asarray(flow_index),
        period=np.zeros(n, dtype=np.int16) if period is None else np.asarray(period),
        meta=info,
    )


@torch.no_grad()
def accumulate_projection_sums(
    model: torch.nn.Module,
    data: ArrayData,
    label_space: LabelSpace,
    standardizer: Standardizer | None,
    sums: GaussianSums,
    *,
    batch_size: int = 1024,
    device: str = "cpu",
) -> None:
    """Feed ``data`` (known-class flows only) through ``model`` and add every chunk's projection
    to ``sums``. ``sums.valid`` must already be set from the same sample's labels and lengths."""
    if not has_pat_heads(model):
        raise ValueError("the Gaussian fit needs the PAT's projection head")
    dev = torch.device(device)
    views = tuple(getattr(model, "views", ("continuous",)))
    was_training = model.training
    model.eval()
    model.to(dev)
    labels = label_space.to_model(data.label).astype(np.int64)
    if (labels < 0).any():
        raise ValueError("unknown-class flows must never reach the Gaussian fit")
    try:
        fb = FlowBatches(data, label_space, standardizer=standardizer, views=views)
        for idx, np_batch in fb.iter_batches(batch_size):
            proj = model(to_torch(np_batch, dev))["proj"].float().cpu().numpy()
            sums.add(proj, labels[idx], np.asarray(data.ppi_len)[idx])
    finally:
        model.train(was_training)


def sample_known_per_class(
    labels: np.ndarray, per_class: int, rng: np.random.Generator
) -> np.ndarray:
    """Sorted indices of at most ``per_class`` flows of each known class (labels are model
    indices; ``-1`` flows are never drawn)."""
    keep = []
    for c in np.unique(labels[labels >= 0]):
        rows = np.flatnonzero(labels == c)
        if len(rows) > per_class:
            rows = rng.choice(rows, size=per_class, replace=False)
        keep.append(rows)
    return np.sort(np.concatenate(keep)) if keep else np.array([], dtype=np.int64)


__all__ = [
    "PAT_HEADS",
    "accumulate_projection_sums",
    "has_pat_heads",
    "identity_calibration",
    "next_packet_columns",
    "sample_known_per_class",
    "score_flows",
]
