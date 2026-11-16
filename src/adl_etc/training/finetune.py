"""Starting a fine-tuning run from a pretrained PAT (spec 008; plan phase 3, T6).

The backbone and the SSL heads (``next_*``, ``proj``) come from the checkpoint;
``cls`` and ``safe`` start fresh, because pretraining never trained them (and
the class count may differ). The load is strict: every other tensor must exist
on both sides with the same shape, and the tokenizer fingerprint (a buffer
every PAT carries, see :func:`adl_etc.models.pat.tokenizer_fingerprint`) must
match, so a checkpoint trained on different tokens, a different ``K_MAX`` or a
different architecture is refused rather than half-loaded. The EMA weights are
used when the pretraining run kept them.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from adl_etc.data import augment_batch as AB
from adl_etc.training.checkpoint import CheckpointError, inference_state_dict, load_checkpoint
from adl_etc.training.datasets import FlowBatches

FRESH_HEADS = ("head_cls.", "head_safe.")


def load_pretrained(model: nn.Module, path: str | Path) -> dict[str, Any]:
    """Load ``path``'s weights into ``model`` except the fresh heads; returns
    what to record about the initialisation (for the run's params)."""
    payload = load_checkpoint(path, map_location="cpu")
    theirs = inference_state_dict(payload)
    ours = model.state_dict()

    def fresh(key: str) -> bool:
        return key.startswith(FRESH_HEADS)

    missing = sorted(k for k in ours if k not in theirs and not fresh(k))
    unexpected = sorted(k for k in theirs if k not in ours)
    mismatched = sorted(
        k for k in theirs if k in ours and not fresh(k) and theirs[k].shape != ours[k].shape
    )
    if missing or unexpected or mismatched:
        raise CheckpointError(
            f"{path} does not fit this model: missing {missing[:5]}, unexpected "
            f"{unexpected[:5]}, shape mismatches {mismatched[:5]}"
        )
    if "tokenizer_id" in ours and not torch.equal(theirs["tokenizer_id"], ours["tokenizer_id"]):
        raise CheckpointError(
            f"{path} was trained with different tokens (bin edges, vocabulary or K_MAX)"
        )
    ours.update({k: v for k, v in theirs.items() if not fresh(k)})
    model.load_state_dict(ours)
    return {
        "init_from": Path(path).as_posix(),
        "init_epoch": int(payload.get("epoch", 0)),
        "init_config_hash": str(payload.get("config_hash")),
        "init_used_ema": "ema" in payload and payload["ema"] is not None,
    }


class AugmentedBatches(FlowBatches):
    """Training batches with spec 008's low-intensity robustness augmentation:
    each real packet dropped with probability ``drop_p`` (gaps folded forward) and
    every IPT jittered by ``exp(N(0, ipt_s))``, drawn per batch from ``(seed, the
    batch's flow indices)`` so a resume repeats them exactly. Validation never
    uses this class. Spec 008 turns it on by default (p = 0.05, s = 0.1) and
    ablates it."""

    def __init__(self, *args: Any, seed: int, drop_p: float, ipt_s: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= drop_p < 1.0 or ipt_s < 0.0:
            raise ValueError(f"need 0 <= drop_p < 1 and ipt_s >= 0, got {drop_p}, {ipt_s}")
        self.seed, self.drop_p, self.ipt_s = seed, drop_p, ipt_s

    def batch(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        key = zlib.crc32(np.ascontiguousarray(idx, dtype=np.int64).tobytes())
        rng = np.random.default_rng([self.seed, key])
        ppi, length = AB.drop(self.data.ppi[idx], self.data.ppi_len[idx], self.drop_p, rng)
        ppi = AB.ipt_jitter(ppi, length, self.ipt_s, rng)
        return self.encode(ppi, length, self.y[idx])
