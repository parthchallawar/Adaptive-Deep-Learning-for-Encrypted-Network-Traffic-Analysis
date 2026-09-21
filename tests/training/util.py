"""Shared helpers for the training tests: learnable synthetic flows, real shard
sets built from them, and a tiny model that trains in well under a second."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from adl_etc.data import ppi as P
from adl_etc.data.tensors import ShardSet, ShardWriter
from adl_etc.training.datasets import ArrayData
from adl_etc.utils.config import load_config
from adl_etc.utils.runinfo import RunInfo


def make_flows(
    n: int,
    n_classes: int,
    seed: int,
    *,
    class_ids: list[int] | None = None,
    min_len: int = 3,
    max_len: int = 12,
) -> ArrayData:
    """Flows whose class is readable from packet sizes and timing, so a small
    model can learn them: class ``c`` sends packets around ``100 + 120c`` bytes
    with gaps around ``5 + 8c`` ms."""
    rng = np.random.default_rng(seed)
    ids = class_ids if class_ids is not None else list(range(n_classes))
    label = np.array(ids)[rng.integers(0, len(ids), n)].astype(np.int64)
    ppi = np.zeros((n, P.K_MAX, P.PPI_CHANNELS), dtype=P.PPI_DTYPE)
    ppi_len = rng.integers(min_len, max_len + 1, n).astype(np.int8)
    for i in range(n):
        k = int(ppi_len[i])
        c = int(label[i])
        ppi[i, :k, P.SIZE_POS] = np.clip(rng.normal(100 + 120 * c, 20, k), 1, P.SIZE_MAX)
        ppi[i, :k, P.IPT_POS] = np.clip(rng.normal(5 + 8 * c, 2, k), 0, P.IPT_MAX_MS)
        ppi[i, 0, P.IPT_POS] = 0
        ppi[i, :k, P.DIR_POS] = np.where(np.arange(k) % 2 == 0, P.DIR_FWD, P.DIR_REV)
        ppi[i, :k, P.PUSH_POS] = rng.integers(0, 2, k)
    return ArrayData(ppi=ppi, ppi_len=ppi_len, label=label, session_id=np.zeros(n, np.int32))


def write_shards(
    root: Path, data: ArrayData, *, name: str = "synthetic", period: str = "all", n_classes: int = 8
) -> ShardSet:
    """A real, opened ShardSet holding ``data``."""
    n = len(data)
    label_map = {f"class{i}": i for i in range(n_classes)}
    with ShardWriter(root, name, period, label_map=label_map, allow_unknown=True) as w:
        w.add_batch(
            {
                "ppi": data.ppi,
                "ppi_len": data.ppi_len,
                "flowstats": np.zeros((n, P.FLOWSTATS_DIM), np.float32),
                "label": data.label.astype(np.int16),
                "category": np.zeros(n, np.int8),
                "session_id": (
                    data.session_id if data.session_id is not None else np.zeros(n, np.int32)
                ),
                "ts": np.arange(n, dtype=np.int64),
            }
        )
    return ShardSet.open(root / name / period)


class TinyMLP(torch.nn.Module):
    """One prediction per flow from the flattened standardised sequence."""

    def __init__(self, n_classes: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(P.K_MAX * P.PPI_CHANNELS, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_classes),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.net(batch["cont"].flatten(1))


class TinyDropoutMLP(TinyMLP):
    """TinyMLP with dropout, so a resume is only exact if the dropout stream is."""

    def __init__(self, n_classes: int, hidden: int = 64) -> None:
        super().__init__(n_classes, hidden)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(P.K_MAX * P.PPI_CHANNELS, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden, n_classes),
        )


class TinyPerStep(torch.nn.Module):
    """One prediction after every packet, and causal: each packet is embedded on
    its own, then a masked running mean over the packets so far feeds the head, so
    the output at position k depends only on packets 1..k. (Embedding first
    matters: averaging the raw signed size channel would cancel across the
    alternating directions and throw the size signal away.)"""

    def __init__(self, n_classes: int, hidden: int = 32) -> None:
        super().__init__()
        self.embed = torch.nn.Sequential(torch.nn.Linear(P.PPI_CHANNELS, hidden), torch.nn.ReLU())
        self.head = torch.nn.Linear(hidden, n_classes)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        mask = batch["mask"].unsqueeze(-1).to(batch["cont"].dtype)  # [B, K, 1]
        h = self.embed(batch["cont"]) * mask  # padded positions contribute nothing
        seen = mask.cumsum(dim=1).clamp_min(1.0)
        return self.head(h.cumsum(dim=1) / seen)


def make_cfg(tmp_path: Path, extra: str = "") -> DictConfig:
    path = tmp_path / "cfg.yaml"
    path.write_text("seed: 0\ntrain:\n  epochs: 3\n" + extra, encoding="utf-8")
    return load_config(path)


def make_info(
    tmp_path: Path, cfg: DictConfig, *, name: str = "bl-tiny-v-synth-s0", seed: int = 0
) -> RunInfo:
    root = tmp_path / "code"
    root.mkdir(exist_ok=True)
    (root / "GIT_COMMIT").write_text("abc1234\n", encoding="utf-8")
    (root / "GIT_DIRTY").write_text("0\n", encoding="utf-8")
    return RunInfo.collect(name, seed, cfg, code_root=root)
