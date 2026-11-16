"""Batched augmentations over raw PPI (spec 003's five, vectorised; plan phase 3,
F4/T5).

:mod:`adl_etc.data.features` defines each augmentation for **one flow**
(``ppi [K_MAX, 4]``, ``ppi_len: int``), which is the readable reference. SSL
pretraining needs two augmented views of ~5.4M flows every epoch, about 11M
Python-level calls per augmentation per epoch, so these versions do the same
thing to a whole batch (``ppi [B, K_MAX, 4]``, ``ppi_len [B]``) with array
operations.

Each function draws its randomness up front and hands it to a ``*_given``
function that is deterministic in those draws; the tests feed the same draws to
the per-flow reference and require **identical** output, flow by flow. Like the
references, they act on raw integer PPI before tokenisation, and padding stays
all-zero.
"""

from __future__ import annotations

import numpy as np

from adl_etc.data import ppi as P

#: spec 003's defaults
DROP_P_MAX = 0.2
REORDER_Q = 0.05
IPT_JITTER_S = 0.2
SIZE_JITTER_SD = 8.0


def _valid(ppi_len: np.ndarray, k: int = P.K_MAX) -> np.ndarray:
    return np.arange(k)[None, :] < np.asarray(ppi_len)[:, None]


# --- drop ----------------------------------------------------------------------------------


def drop_given(
    ppi: np.ndarray, ppi_len: np.ndarray, keep: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the packets where ``keep`` is False (padding positions are ignored),
    shift the survivors left, and fold every removed packet's gap into the next
    survivor's IPT; the first survivor's IPT is 0. ``keep`` must keep at least
    one real packet of every non-empty flow (``drop`` guarantees it)."""
    b, k, _ = ppi.shape
    valid = _valid(ppi_len, k)
    keep = keep & valid
    ipt = ppi[..., P.IPT_POS].astype(np.int64) * valid
    cum = np.cumsum(ipt, axis=1)
    # index of the last kept packet strictly before each position (-1: none)
    marks = np.where(keep, np.arange(k)[None, :], -1)
    last_kept = np.maximum.accumulate(marks, axis=1)
    prev = np.concatenate([np.full((b, 1), -1), last_kept[:, :-1]], axis=1)
    prev_cum = np.take_along_axis(cum, np.maximum(prev, 0), axis=1)
    new_ipt = np.where(prev >= 0, cum - prev_cum, 0)  # own gap + every dropped gap since
    new_ipt = np.minimum(new_ipt, P.IPT_MAX_MS)

    rows = ppi.copy()
    rows[..., P.IPT_POS] = new_ipt.astype(ppi.dtype)
    new_len = keep.sum(axis=1)
    dest = np.cumsum(keep, axis=1) - 1  # where each kept packet lands
    out = np.zeros_like(ppi)
    bi, ki = np.nonzero(keep)
    out[bi, dest[bi, ki]] = rows[bi, ki]
    return out, new_len.astype(np.asarray(ppi_len).dtype)


def drop(
    ppi: np.ndarray, ppi_len: np.ndarray, p: np.ndarray | float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Each real packet is removed with probability ``p`` (scalar or per flow);
    a flow that would lose every packet keeps one at random."""
    b, k, _ = ppi.shape
    valid = _valid(ppi_len, k)
    p_arr = np.broadcast_to(np.asarray(p, dtype=np.float64), (b,))
    keep = (rng.random((b, k)) >= p_arr[:, None]) & valid
    empty = (keep.sum(axis=1) == 0) & (np.asarray(ppi_len) > 0)
    if empty.any():
        rescue = (rng.random(int(empty.sum())) * np.asarray(ppi_len)[empty]).astype(np.int64)
        keep[np.flatnonzero(empty), rescue] = True
    return drop_given(ppi, ppi_len, keep)


# --- reorder --------------------------------------------------------------------------------


def reorder_given(ppi: np.ndarray, ppi_len: np.ndarray, u: np.ndarray, q: float) -> np.ndarray:
    """The reference's left-to-right rule with draws ``u [B, K_MAX - 1]``: at a
    visited position i < ppi_len - 1, swap packets i and i+1 if ``u[i] < q`` and
    skip to i + 2, else move to i + 1. Position 0's IPT is reset to 0."""
    out = ppi.copy()
    b, k, _ = ppi.shape
    length = np.asarray(ppi_len)
    skip = np.zeros(b, dtype=bool)
    rows = np.arange(b)
    for i in range(k - 1):
        swap = ~skip & (i < length - 1) & (u[:, i] < q)
        if swap.any():
            sel = rows[swap]
            first = out[sel, i].copy()
            out[sel, i] = out[sel, i + 1]
            out[sel, i + 1] = first
        skip = swap
    out[:, 0, P.IPT_POS] = 0
    return out


def reorder(
    ppi: np.ndarray, ppi_len: np.ndarray, q: float, rng: np.random.Generator
) -> np.ndarray:
    return reorder_given(ppi, ppi_len, rng.random((ppi.shape[0], ppi.shape[1] - 1)), q)


# --- jitter ---------------------------------------------------------------------------------


def ipt_jitter_given(
    ppi: np.ndarray, ppi_len: np.ndarray, normal: np.ndarray, s: float
) -> np.ndarray:
    """IPT times ``exp(s * normal)``, rounded and clipped like ``ppi.clip_ipt``;
    position 0 stays 0, padding stays 0."""
    out = ppi.copy()
    valid = _valid(ppi_len, ppi.shape[1])
    new = ppi[..., P.IPT_POS].astype(np.float64) * np.exp(s * normal)
    new = np.minimum(np.floor(new + 0.5), P.IPT_MAX_MS)
    new[:, 0] = 0
    out[..., P.IPT_POS] = np.where(valid, new, ppi[..., P.IPT_POS]).astype(ppi.dtype)
    return out


def ipt_jitter(
    ppi: np.ndarray, ppi_len: np.ndarray, s: float, rng: np.random.Generator
) -> np.ndarray:
    return ipt_jitter_given(ppi, ppi_len, rng.standard_normal(ppi.shape[:2]), s)


def size_jitter_given(
    ppi: np.ndarray, ppi_len: np.ndarray, noise: np.ndarray
) -> np.ndarray:
    """Sizes plus ``noise`` bytes, rounded and clipped to ``[1, SIZE_MAX]`` (a real
    packet never becomes padding); padding untouched."""
    out = ppi.copy()
    valid = _valid(ppi_len, ppi.shape[1])
    new = np.clip(np.round(ppi[..., P.SIZE_POS].astype(np.float64) + noise), 1, P.SIZE_MAX)
    out[..., P.SIZE_POS] = np.where(valid, new, ppi[..., P.SIZE_POS]).astype(ppi.dtype)
    return out


def size_jitter(
    ppi: np.ndarray, ppi_len: np.ndarray, sd: float, rng: np.random.Generator
) -> np.ndarray:
    return size_jitter_given(ppi, ppi_len, rng.normal(0.0, sd, ppi.shape[:2]))


# --- crop for the prefix view ---------------------------------------------------------------


def prefix_crop(
    ppi_len: np.ndarray, rng: np.random.Generator, *, low: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Spec 007's PFC crop: ``K ~ U{low .. min(K_MAX, ppi_len - 1)}`` per flow,
    so the prefix is always shorter than the flow it must identify. Returns
    ``(k, eligible)``; a flow with fewer than ``low + 1`` packets is ineligible
    (its ``k`` is its own length, and PFC skips it)."""
    length = np.asarray(ppi_len).astype(np.int64)
    high = np.minimum(P.K_MAX, length - 1)
    eligible = high >= low
    span = np.maximum(high - low + 1, 1)
    k = low + (rng.random(len(length)) * span).astype(np.int64)
    return np.where(eligible, k, length), eligible
