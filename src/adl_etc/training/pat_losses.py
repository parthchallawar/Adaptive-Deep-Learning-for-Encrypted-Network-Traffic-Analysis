"""Losses over the PAT's heads (specs 007 and 008; plan phase 3, T5-T6).

Every function takes the PAT's per-packet outputs (``[B, 30, ...]``, index
``k - 1`` = after k packets) and the batch's ``mask``/``ppi_len``/``tokens``.
Padded positions never contribute: targets there are ignored, never zero.

- :func:`npp_loss` (spec 007, also kept alive at weight 0.1 in spec 008):
  predict packet k+1's size, ipt and direction tokens from position k, plus
  ``end``, "packet k is the last one". Targets come from ``batch["tokens"]``,
  the un-augmented flow.
- :func:`prefix_ce` (spec 008's ``L_cls``): weighted, label-smoothed CE at every
  real position, weights ``w_k = min(1, k / ramp)``; ``ramp = 1`` is uniform.
- :func:`self_distillation` (``L_sd``): KL from the flow's last real position,
  **detached**, to every position, at temperature T.
- :func:`safe_bce` (``L_safe``): BCE of ``safe`` against "is ``argmax cls`` right
  here", targets from detached logits, positives and negatives balanced per K so
  that K = 1 (mostly wrong) does not dominate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from adl_etc.data import features as FT
from adl_etc.data import ppi as P


def _masked_mean(values: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` where ``keep``; 0 (not NaN) when nothing is kept. A
    dropped position's value may be inf/NaN, so it is replaced, not multiplied."""
    total = torch.where(keep, values, torch.zeros_like(values)).sum()
    return total / keep.sum().clamp_min(1)


def end_targets(ppi_len: torch.Tensor, k_max: int = P.K_MAX) -> tuple[torch.Tensor, torch.Tensor]:
    """``(target, keep)`` for the ``end`` head, both ``[B, k_max]``: 1 at the last
    real packet of a flow that ended before the cap, 0 at its earlier packets;
    kept nowhere else (padding, and the last packet of a flow truncated at
    ``k_max``, whose continuation is unknown)."""
    positions = torch.arange(1, k_max + 1, device=ppi_len.device)[None, :]
    length = ppi_len[:, None]
    target = (positions == length).float()
    keep = (positions < length) | ((positions == length) & (length < k_max))
    return target, keep


def npp_loss(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Next-packet prediction: ``CE(size) + CE(ipt) + CE(dir) + BCE(end)``, each a
    mean over its valid targets."""
    tokens = batch["tokens"]
    nxt = torch.cat([tokens[:, 1:], torch.zeros_like(tokens[:, :1])], dim=1)  # packet k+1

    def ce(head: str, channel: int) -> torch.Tensor:
        logits = outputs[head].float()
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            nxt[..., channel].reshape(-1),
            ignore_index=FT.PAD_INDEX,  # no next packet -> padding token -> ignored
        )

    has_next = nxt[..., P.SIZE_POS] != FT.PAD_INDEX
    parts = {
        "npp_size": ce("next_size", P.SIZE_POS),
        "npp_ipt": ce("next_ipt", P.IPT_POS),
        "npp_dir": ce("next_dir", P.DIR_POS),
    }
    if not bool(has_next.any()):  # every flow in the batch is one packet long
        parts = {k: torch.zeros((), device=tokens.device) for k in parts}
    target, keep = end_targets(batch["ppi_len"])
    end = F.binary_cross_entropy_with_logits(
        outputs["next_end"].float(), target, reduction="none"
    )
    parts["npp_end"] = _masked_mean(end, keep)
    return sum(parts.values(), torch.zeros((), device=tokens.device)), parts


def ramp_weights(k_max: int, ramp: float, device: torch.device | str = "cpu") -> torch.Tensor:
    """``w_k = min(1, k / ramp)`` for k = 1..k_max (``ramp = 1``: all ones)."""
    if ramp < 1:
        raise ValueError(f"ramp must be >= 1, got {ramp}")
    k = torch.arange(1, k_max + 1, dtype=torch.float32, device=device)
    return torch.clamp(k / ramp, max=1.0)


def prefix_ce(
    cls: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    *,
    ramp: float = 8.0,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """``Σ w_k · CE(cls_k, y) / Σ w_k`` over real positions. With ``ramp = 1``
    this is exactly ``losses.multi_prefix_ce``."""
    b, k, c = cls.shape
    per_position = F.cross_entropy(
        cls.float().reshape(b * k, c),
        y[:, None].expand(b, k).reshape(-1),
        label_smoothing=smoothing,
        reduction="none",
    ).reshape(b, k)
    w = ramp_weights(k, ramp, cls.device)[None, :].expand(b, k)
    weighted = torch.where(mask, per_position * w, torch.zeros_like(per_position))
    return weighted.sum() / torch.where(mask, w, torch.zeros_like(w)).sum().clamp_min(1e-12)


def final_index(ppi_len: torch.Tensor) -> torch.Tensor:
    return (ppi_len - 1).clamp_min(0)


def self_distillation(
    cls: torch.Tensor,
    mask: torch.Tensor,
    ppi_len: torch.Tensor,
    *,
    ramp: float = 8.0,
    temperature: float = 2.0,
) -> torch.Tensor:
    """``Σ w_k · KL(softmax(stopgrad(cls_len) / T) || softmax(cls_k / T)) / Σ w_k``.
    The teacher is each flow's last real position and gets no gradient."""
    b, k, _ = cls.shape
    logits = cls.float() / temperature
    teacher = logits[torch.arange(b, device=cls.device), final_index(ppi_len)].detach()
    log_q = logits.log_softmax(-1)
    log_p = teacher.log_softmax(-1)[:, None, :]
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)  # [B, K]
    w = ramp_weights(k, ramp, cls.device)[None, :].expand(b, k)
    weighted = torch.where(mask, kl * w, torch.zeros_like(kl))
    return weighted.sum() / torch.where(mask, w, torch.zeros_like(w)).sum().clamp_min(1e-12)


def safe_bce(
    safe: torch.Tensor, cls: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """BCE of ``safe_k`` against ``1[argmax cls_k == y]`` (detached), balanced per
    K: at each position, positives and negatives each count half (or all, if only
    one kind is present there); the per-K values are then averaged over the K
    that have any real packet."""
    correct = cls.detach().argmax(-1) == y[:, None]
    bce = F.binary_cross_entropy_with_logits(safe.float(), correct.float(), reduction="none")
    pos, neg = mask & correct, mask & ~correct
    n_pos, n_neg = pos.sum(0), neg.sum(0)
    pos_mean = torch.where(pos, bce, torch.zeros_like(bce)).sum(0) / n_pos.clamp_min(1)
    neg_mean = torch.where(neg, bce, torch.zeros_like(bce)).sum(0) / n_neg.clamp_min(1)
    both = (n_pos > 0) & (n_neg > 0)
    per_k = torch.where(both, 0.5 * (pos_mean + neg_mean), pos_mean + neg_mean)
    present = (n_pos + n_neg) > 0
    return torch.where(present, per_k, torch.zeros_like(per_k)).sum() / present.sum().clamp_min(1)
