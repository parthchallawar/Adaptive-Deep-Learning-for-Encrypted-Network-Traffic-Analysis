"""B3: GRU / LSTM baseline (spec 005).

A linear stem (4 to 128), a two-layer recurrent encoder (hidden 256) and a
per-step classifier, so one pass yields a prediction after every packet. It
mirrors CAPE-Net's backbone, which is what makes the CAPE-style policy baseline
faithful. Trained with the multi-prefix loss (``training.losses``) so that the
comparison to the Transformer isolates the architecture and not the recipe.

**Causal.** The output at packet ``k`` depends only on packets ``1..k``, so a
single pass gives every prefix and its saved logits are ``effective``-indexed
(see ``evaluation.dense_logits``): no per-K passes are needed.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from adl_etc.data import ppi as P


class RNNBaseline(nn.Module):
    causal = True
    views = ("continuous",)

    def __init__(
        self,
        n_classes: int,
        *,
        cell: Literal["gru", "lstm"] = "gru",
        stem: int = 128,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
        in_channels: int = P.PPI_CHANNELS,
    ) -> None:
        super().__init__()
        if cell not in ("gru", "lstm"):
            raise ValueError(f"cell must be 'gru' or 'lstm', got {cell!r}")
        self.stem = nn.Sequential(nn.Linear(in_channels, stem), nn.GELU(), nn.Dropout(dropout))
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            stem,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        out, _ = self.rnn(self.stem(batch["cont"]))  # [B, K, hidden]
        return self.head(self.drop(out))  # [B, K, C]
