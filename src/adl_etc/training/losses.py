"""Training losses (spec 005; the multi-prefix loss is spec 008's, in minimal form).

Spec 005 says B3 trains with spec 008's multi-prefix loss, but 008 is build step
10 and B3 is step 7. The loss itself is a masked mean of per-position
cross-entropy, so the minimal slice B3 needs is declared here and spec 008
*extends* :func:`multi_prefix_ce` (weighting, the safe head) rather than
replacing it.

Targets must already be valid class indices ``0..C-1``. That is guaranteed by
:class:`~adl_etc.training.datasets.FlowBatches` in train mode, which refuses
unknown-class flows, so the loss does not spend a device sync per step
re-checking.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_ls(logits: torch.Tensor, y: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """Label-smoothed cross-entropy for one prediction per flow, ``logits[B, C]``."""
    if logits.ndim != 2:
        raise ValueError(f"expected logits[B, C], got shape {tuple(logits.shape)}")
    return F.cross_entropy(logits, y, label_smoothing=smoothing)


def multi_prefix_ce(
    logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, smoothing: float = 0.1
) -> torch.Tensor:
    """Mean cross-entropy over every *valid* (flow, position) pair.

    ``logits[B, K, C]`` holds a prediction after each of the first K packets;
    every position is supervised with the flow's one label. ``mask[B, K]`` is
    True where a real packet sits, so padded positions contribute nothing (their
    logits are arbitrary and their gradient is exactly zero).

    Pooled rather than per-flow: a flow contributes as many terms as it has
    packets, so short prefixes, which every flow has, are weighted by how many
    flows reach them. Returns 0 (not NaN) if the whole batch is masked.
    """
    if logits.ndim != 3:
        raise ValueError(f"expected logits[B, K, C], got shape {tuple(logits.shape)}")
    b, k, c = logits.shape
    if mask.shape != (b, k):
        raise ValueError(f"mask must be [{b}, {k}], got {tuple(mask.shape)}")
    per_position = F.cross_entropy(
        logits.reshape(b * k, c),
        y[:, None].expand(b, k).reshape(-1),
        label_smoothing=smoothing,
        reduction="none",
    ).reshape(b, k)
    weight = mask.to(per_position.dtype)
    # torch.where, not multiplication: a masked position's loss may be inf/NaN
    # (garbage logits) and 0 * inf is NaN.
    total = torch.where(mask, per_position, torch.zeros_like(per_position)).sum()
    return total / weight.sum().clamp_min(1.0)
