"""B2: 1D-CNN baseline (spec 005).

Three Conv1d blocks (channels 128, 192, 256; kernels 5, 5, 3), batch norm, GELU,
masked global average and max pooling, and an MLP head, about 0.45M parameters at
150 classes. It reads the continuous view ``[B, 30, 4]``.

**Non-causal.** A temporal convolution with global pooling gives a different
representation for every prefix length, so it is evaluated at fixed K by
truncating each flow to its first K packets (``ArrayData.prefix``) and running one
pass per K, on spec 004's 13-value grid. It is therefore a *per-K* model and its
saved logits are ``nominal``-indexed (see ``evaluation.dense_logits``).

**Masked batch norm.** Real flows are short: the mean is about 7 real packets in
30 slots, so an ordinary ``BatchNorm1d`` would compute its statistics mostly over
padding. :class:`MaskedBatchNorm1d` counts only real packets, and every block
re-zeroes padded positions so that the convolutions see the same zero padding
whether a flow is short by nature or truncated to K.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from adl_etc.data import ppi as P


class MaskedBatchNorm1d(nn.Module):
    """Batch norm over ``[B, C, L]`` whose statistics use only positions where
    ``mask`` is 1. Equivalent to ``nn.BatchNorm1d`` when the mask is all ones."""

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: [B, 1, L] of 0/1, same dtype as x
        if self.training:
            xf, mf = x.float(), mask.float()
            n = mf.sum().clamp_min(1.0) * 1.0
            mean = (xf * mf).sum(dim=(0, 2)) / n
            var = (((xf - mean[None, :, None]) ** 2) * mf).sum(dim=(0, 2)) / n
            with torch.no_grad():
                unbiased = var * (n / (n - 1.0).clamp_min(1.0))
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(unbiased, self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        inv = torch.rsqrt(var + self.eps) * self.weight
        shift = self.bias - mean * inv
        return (x * inv[None, :, None].to(x.dtype)) + shift[None, :, None].to(x.dtype)


class CNNBaseline(nn.Module):
    causal = False
    views = ("continuous",)

    def __init__(
        self,
        n_classes: int,
        *,
        channels: Sequence[int] = (128, 192, 256),
        kernels: Sequence[int] = (5, 5, 3),
        head_hidden: int = 256,
        dropout: float = 0.1,
        in_channels: int = P.PPI_CHANNELS,
    ) -> None:
        super().__init__()
        if len(channels) != len(kernels):
            raise ValueError("channels and kernels must have the same length")
        if any(k % 2 == 0 for k in kernels):
            raise ValueError(f"kernels must be odd so the length is preserved, got {list(kernels)}")
        widths = [in_channels, *channels]
        self.convs = nn.ModuleList(
            nn.Conv1d(widths[i], widths[i + 1], kernels[i], padding=kernels[i] // 2)
            for i in range(len(channels))
        )
        self.norms = nn.ModuleList(MaskedBatchNorm1d(c) for c in channels)
        self.act = nn.GELU()
        self.head = nn.Sequential(
            nn.Linear(2 * channels[-1], head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        mask = batch["mask"].unsqueeze(1)  # [B, 1, L] bool
        m = mask.to(batch["cont"].dtype)
        x = batch["cont"].transpose(1, 2) * m  # [B, C, L]
        for conv, norm in zip(self.convs, self.norms, strict=True):
            x = self.act(norm(conv(x), m)) * m  # padded positions back to exactly 0
        count = m.sum(dim=-1).clamp_min(1.0)
        avg = x.sum(dim=-1) / count
        mx = x.masked_fill(~mask, torch.finfo(x.dtype).min).amax(dim=-1)
        return self.head(torch.cat([avg, mx], dim=-1))  # [B, C]
