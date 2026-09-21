"""The saved-logits contract: a dense ``[N, K_MAX, C]`` array that says how it is
indexed and which prefix lengths it actually holds (spec 004, plan phase 2 F4).

Two failure modes this exists to prevent, both silent (no exception, wrong
numbers in the headline table):

1. **A grid stored positionally.** Spec 004 evaluates non-causal models at 13
   values of K. Saving that 13-wide array and reading it as if axis 1 were
   ``K = 1..30`` returns K=10's row for a request for K=8 (index ``k - 1`` of the 13-wide grid).

2. **Per-K models and short flows.** :func:`~adl_etc.evaluation.metrics.logits_at_k`
   reads position ``min(K, ppi_len) - 1``, which is right for a *causal* model
   (its output at packet ``e`` is the prefix-``e`` answer). It is wrong for a model
   trained per K (XGBoost, the CNN): for a 7-packet flow the K=10 model's answer
   and the K=12 model's answer would both be filed at position 6, and a request
   for K=10 would read whichever was written last.

So an array carries its ``indexing``:

* ``"effective"`` (causal models): one pass yields every prefix; position
  ``min(K, ppi_len) - 1`` holds the answer for nominal K. All K are available.
* ``"nominal"`` (per-K models): position ``K - 1`` holds that K's model applied to
  the flow (a flow shorter than K is fed whole). Only ``evaluated_k`` positions
  exist, and asking for another raises :class:`NotEvaluatedError`.

Pure NumPy: the evaluation layer stays torch-free.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from adl_etc.data import ppi as P

#: Spec 004: prefix lengths at which non-causal (per-K) models are evaluated.
K_GRID: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30)

Indexing = Literal["effective", "nominal"]


class NotEvaluatedError(KeyError):
    """A prefix length that this array does not hold."""


@dataclass(frozen=True)
class DenseLogits:
    logits: np.ndarray  # [N, K_MAX, C]
    evaluated_k: tuple[int, ...]
    indexing: Indexing

    def __post_init__(self) -> None:
        if self.logits.ndim != 3 or self.logits.shape[1] != P.K_MAX:
            raise ValueError(f"logits must be [N, {P.K_MAX}, C], got {self.logits.shape}")
        if self.indexing not in ("effective", "nominal"):
            raise ValueError(f"indexing must be 'effective' or 'nominal', got {self.indexing!r}")
        ks = tuple(self.evaluated_k)
        if not ks or list(ks) != sorted(set(ks)) or ks[0] < 1 or ks[-1] > P.K_MAX:
            raise ValueError(f"evaluated_k must be sorted unique values in 1..{P.K_MAX}: {ks}")

    @property
    def n_classes(self) -> int:
        return int(self.logits.shape[2])

    def __len__(self) -> int:
        return int(self.logits.shape[0])

    def at(self, k: int, ppi_len: np.ndarray) -> np.ndarray:
        """``[N, C]`` logits for nominal prefix length ``k``, one row per flow.

        Raises :class:`NotEvaluatedError` for a ``k`` this array does not hold,
        instead of returning whatever happens to sit at that position.
        """
        if isinstance(k, bool) or not isinstance(k, int | np.integer):
            raise TypeError(f"k must be an int, got {k!r}")
        if int(k) not in self.evaluated_k:
            raise NotEvaluatedError(
                f"K={k} was not evaluated; this array holds K in {list(self.evaluated_k)}"
            )
        ppi_len = np.asarray(ppi_len)
        if ppi_len.shape != (len(self),):
            raise ValueError(f"ppi_len must be [{len(self)}], got {ppi_len.shape}")
        if self.indexing == "effective":
            idx = np.clip(np.minimum(int(k), ppi_len), 1, P.K_MAX) - 1
        else:
            idx = np.full(len(self), int(k) - 1)
        return self.logits[np.arange(len(self)), idx, :]

    def predictions_at(self, k: int, ppi_len: np.ndarray) -> np.ndarray:
        return self.at(k, ppi_len).argmax(axis=-1)


def from_causal(logits: np.ndarray) -> DenseLogits:
    """From a causal model's per-step output ``[N, K_MAX, C]``: every K is held."""
    return DenseLogits(logits, tuple(range(1, P.K_MAX + 1)), "effective")


def from_per_k(
    per_k: Mapping[int, np.ndarray], *, dtype: np.dtype | type = np.float32
) -> DenseLogits:
    """From ``{K: [N, C]}`` produced by a model applied at each nominal K.

    Positions for K values not in ``per_k`` are zero and are never readable:
    they are not in ``evaluated_k``.
    """
    if not per_k:
        raise ValueError("per_k is empty")
    ks = sorted(int(k) for k in per_k)
    first = np.asarray(per_k[ks[0]])
    n, c = first.shape
    logits = np.zeros((n, P.K_MAX, c), dtype=dtype)
    for k in ks:
        arr = np.asarray(per_k[k])
        if arr.shape != (n, c):
            raise ValueError(f"K={k}: expected {(n, c)}, got {arr.shape}")
        logits[:, k - 1, :] = arr
    return DenseLogits(logits, tuple(ks), "nominal")


def grid_from(ks: Iterable[int] | None) -> tuple[int, ...]:
    """Validate a requested set of prefix lengths (default: spec 004's grid)."""
    out = tuple(sorted({int(k) for k in (K_GRID if ks is None else ks)}))
    if not out or out[0] < 1 or out[-1] > P.K_MAX:
        raise ValueError(f"prefix lengths must lie in 1..{P.K_MAX}, got {out}")
    return out
