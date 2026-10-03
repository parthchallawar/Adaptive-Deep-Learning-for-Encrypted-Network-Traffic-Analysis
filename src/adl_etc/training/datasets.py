"""Batches over shard data for the training loop (spec 005, spec 003).

Deliberately **no torch ``Dataset``/``DataLoader``**: spec 015 forbids DataLoader
workers on Kaggle, and a batch here is an index slice into in-memory arrays run
through the vectorised tokeniser. :class:`FlowBatches` turns an index array into
a dict of NumPy arrays; :func:`to_torch` moves one to a device.

**The standardizer is loaded, never fit.** :class:`FlowBatches` takes a
:class:`~adl_etc.data.features.Standardizer` that already exists and has no
``fit`` of any kind. The one way to refit on val or test data by accident is to
not have written the method, and ``test_datasets.py`` asserts the API stays that
way.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from adl_etc.data import features as F
from adl_etc.data import ppi as P
from adl_etc.data.features import Standardizer
from adl_etc.data.tensors import ShardSet
from adl_etc.training.labels import LabelSpace

VIEWS = ("continuous", "tokens")


# --- the arrays --------------------------------------------------------------------------


@dataclass
class ArrayData:
    """Flows held in memory: raw PPI, lengths, and *dataset* label ids.

    Independent of any :class:`ShardSet` (the arrays are copies), so the shard
    sets can be closed after loading, which on Windows is what allows their
    directories to be deleted or overwritten again.
    """

    ppi: np.ndarray  # [N, K_MAX, PPI_CHANNELS] int16
    ppi_len: np.ndarray  # [N]
    label: np.ndarray  # [N] dataset ids, -1 = unknown
    session_id: np.ndarray | None = None  # [N]

    def __post_init__(self) -> None:
        n = len(self.ppi_len)
        if self.ppi.shape != (n, P.K_MAX, P.PPI_CHANNELS):
            raise ValueError(
                f"ppi must be [{n}, {P.K_MAX}, {P.PPI_CHANNELS}], got {self.ppi.shape}"
            )
        if self.label.shape != (n,):
            raise ValueError(f"label must be [{n}], got {self.label.shape}")
        if self.session_id is not None and self.session_id.shape != (n,):
            raise ValueError(f"session_id must be [{n}], got {self.session_id.shape}")

    def __len__(self) -> int:
        return len(self.ppi_len)

    @classmethod
    def from_shards(
        cls,
        shard_sets: Sequence[ShardSet],
        masks: Sequence[np.ndarray | None] | None = None,
    ) -> ArrayData:
        """Concatenate shard sets, optionally restricted by per-set boolean
        masks (the ``session_ids`` restriction ``protocol.load_split`` builds)."""
        if not shard_sets:
            raise ValueError("no shard sets given")
        masks = list(masks) if masks is not None else [None] * len(shard_sets)
        if len(masks) != len(shard_sets):
            raise ValueError("masks must match shard_sets one to one")

        def column(name: str) -> np.ndarray:
            parts = []
            for ss, mask in zip(shard_sets, masks, strict=True):
                col = ss.column(name)
                parts.append(col if mask is None else col[mask])
            return np.concatenate(parts)  # always a copy, detached from the memmaps

        return cls(
            ppi=column("ppi"),
            ppi_len=column("ppi_len"),
            label=column("label").astype(np.int64),
            session_id=column("session_id"),
        )

    def prefix(self, k: int) -> ArrayData:
        """Every flow as an observer sees it after its first ``k`` packets: later
        positions zeroed and ``ppi_len`` capped at ``k``. This is how a per-K
        (non-causal) model is evaluated at nominal K; a flow shorter than ``k`` is
        left whole."""
        if isinstance(k, bool) or not isinstance(k, int | np.integer) or not 1 <= k <= P.K_MAX:
            raise ValueError(f"k must be an int in 1..{P.K_MAX}, got {k!r}")
        ppi = self.ppi.copy()
        ppi[:, k:, :] = 0
        return ArrayData(
            ppi=ppi,
            ppi_len=np.minimum(self.ppi_len, k).astype(self.ppi_len.dtype),
            label=self.label,
            session_id=self.session_id,
        )

    def subsample(self, n: int, rng: np.random.Generator) -> ArrayData:
        """Uniform sample of ``n`` flows without replacement, original order kept."""
        if n >= len(self):
            return self
        idx = np.sort(rng.choice(len(self), size=n, replace=False))
        return ArrayData(
            ppi=self.ppi[idx],
            ppi_len=self.ppi_len[idx],
            label=self.label[idx],
            session_id=None if self.session_id is None else self.session_id[idx],
        )


# --- batches ---------------------------------------------------------------------------------


class FlowBatches:
    """Index-addressable batches of model inputs and targets.

    ``train=True`` refuses any flow that is not a known class: an unknown-class
    flow in training is a leakage bug (spec 004 rule 2), and this is the third
    and last gate after ``ShardWriter`` and ``protocol.load_split``.
    """

    def __init__(
        self,
        data: ArrayData,
        label_space: LabelSpace,
        *,
        standardizer: Standardizer | None,
        views: Sequence[str] = ("continuous",),
        train: bool = False,
    ) -> None:
        bad = [v for v in views if v not in VIEWS]
        if bad or not views:
            raise ValueError(f"views must be a non-empty subset of {VIEWS}, got {list(views)}")
        if "continuous" in views and standardizer is None:
            raise ValueError(
                "the continuous view needs a Standardizer (fit on training data once and "
                "loaded here): a model must not read unnormalised features"
            )
        self.data = data
        self.label_space = label_space
        self.standardizer = standardizer
        self.views = tuple(views)
        self.train = train
        self.y = label_space.to_model(data.label)

        if train and bool((self.y < 0).any()):
            n_bad = int((self.y < 0).sum())
            raise ValueError(
                f"{n_bad} flow(s) in a training set are not known classes; unknown-class "
                "flows must never be trained on (spec 004 leakage rule 2)"
            )

    def __len__(self) -> int:
        return len(self.data)

    @property
    def standardizer_hash(self) -> str | None:
        return None if self.standardizer is None else self.standardizer.hash

    @property
    def known_indices(self) -> np.ndarray:
        """Indices of flows whose class is known (all of them, in training)."""
        return np.flatnonzero(self.y >= 0)

    def class_counts(self) -> np.ndarray:
        """Flows per model class, length ``n_classes`` (unknown flows excluded)."""
        return np.bincount(self.y[self.y >= 0], minlength=self.label_space.n_classes)

    def batch(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        ppi = self.data.ppi[idx]
        ppi_len = self.data.ppi_len[idx]
        mask = F.padding_mask(ppi_len)
        out: dict[str, np.ndarray] = {
            "mask": mask,
            "ppi_len": ppi_len.astype(np.int64),
            "y": self.y[idx],
        }
        if "continuous" in self.views:
            assert self.standardizer is not None
            cont = self.standardizer.transform_continuous(F.continuous(ppi, ppi_len))
            # Standardising moves a padded position from 0 to -mean/std; the project's
            # rule is that padding is exactly 0 in every channel, so restore it.
            out["cont"] = cont * mask[:, :, None]
        if "tokens" in self.views:
            out["tokens"] = F.tokenize(ppi, ppi_len)
        return out

    def iter_batches(
        self, batch_size: int, *, known_only: bool = False
    ) -> Iterator[tuple[np.ndarray, dict[str, np.ndarray]]]:
        """Sequential ``(indices, batch)`` pairs covering the data once, for
        evaluation. ``known_only`` skips unknown-class flows."""
        order = self.known_indices if known_only else np.arange(len(self))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            yield idx, self.batch(idx)


def to_torch(batch: dict[str, np.ndarray], device: Any = "cpu") -> dict[str, Any]:
    import torch

    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in batch.items()}


# --- class-balanced sampling ---------------------------------------------------------------------


def balanced_weights(y: np.ndarray, n_classes: int, cap: float = 10.0) -> np.ndarray:
    """Per-flow sampling probabilities (sum to 1) for class-balanced sampling
    with oversampling capped at ``cap`` (spec 005: "capped at 10x").

    Each present class ``c`` is sampled ``min(cap, mean_count / n_c)`` times as
    often as it occurs naturally, where ``mean_count`` is the average flows per
    *present* class. So a class at least ``1/cap`` of the mean size ends up with
    equal total mass, and a rarer class is oversampled by exactly ``cap`` rather
    than a thousand times. Classes with no flows never appear, so nothing
    divides by zero.
    """
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")
    y = np.asarray(y)
    if y.size == 0:
        raise ValueError("cannot balance an empty label array")
    if y.min() < 0:
        raise ValueError("balanced sampling needs known-class labels only (no -1)")
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    present = counts > 0
    mean_count = y.size / present.sum()
    factor = np.zeros(n_classes)
    factor[present] = np.minimum(cap, mean_count / counts[present])
    w = factor[y]
    return w / w.sum()


def epoch_indices(
    n_data: int,
    n_samples: int,
    rng: np.random.Generator,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Flow indices for one epoch. With ``weights``: ``n_samples`` draws with
    replacement. Without: fresh permutations (each flow once per pass), taking
    ``n_samples`` from as many passes as needed."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if weights is not None:
        return rng.choice(n_data, size=n_samples, replace=True, p=weights)
    passes = -(-n_samples // n_data)
    return np.concatenate([rng.permutation(n_data) for _ in range(passes)])[:n_samples]
