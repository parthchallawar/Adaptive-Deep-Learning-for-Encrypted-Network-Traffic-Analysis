"""Exponential moving average of a model's weights, with warm-up (spec 008).

A plain EMA with decay 0.999 averages over roughly the last 1,000 updates, so a
short run's average is mostly its *untrained starting weights*: after 220
updates, 0.999^220 ≈ 80% of the EMA is still the initial model (the 1% label
regime as spec 008 first sized it; plan phase 3, F11). The warm-up lets the
average start fast and settle to ``decay``:

    decay_t = min(decay, (1 + t) / (10 + t)),   t = updates already applied

so the first update keeps only 10% of the start, and ``decay`` is reached after
about ``9 / (1 - decay)`` updates.

Floating-point state (parameters and float buffers) is averaged; integer
buffers (``num_batches_tracked``) are copied. The averaged copy lives in its own
module, kept in eval mode, so validation can run on it directly.
"""

from __future__ import annotations

import copy
from typing import Any

import torch


def warmup_decay(decay: float, updates_done: int) -> float:
    """The decay applied by the next update, after ``updates_done`` updates."""
    return min(decay, (1.0 + updates_done) / (10.0 + updates_done))


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.updates = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = warmup_decay(self.decay, self.updates)
        ema_state = self.module.state_dict()
        for key, value in model.state_dict().items():
            target = ema_state[key]
            if target.dtype.is_floating_point:
                target.mul_(d).add_(value.detach(), alpha=1.0 - d)
            else:
                target.copy_(value)
        self.updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "updates": self.updates, "model": self.module.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError(f"EMA decay {state['decay']} in the checkpoint, {self.decay} here")
        self.updates = int(state["updates"])
        self.module.load_state_dict(state["model"])
