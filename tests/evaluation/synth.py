"""Synthetic score streams with a known structure, for testing the phase-4 evaluation protocol.

Unknown-class flows have higher energy and surprise; later flows are harder (their safety
probability rises more slowly with K), so a fixed commit threshold reads more packets as time
goes on, which is exactly the drift the budget controller exists for.
"""

from __future__ import annotations

import numpy as np

from adl_etc.evaluation.scores import K_MAX, ScoreArrays, fill_forward
from adl_etc.inference.unknown import GaussianBank, UnknownModel

N_CLASSES = 12


def make_set(
    n_windows: int,
    window: int,
    seed: int,
    *,
    drift: float = 0.0,
    unknown_share: float = 0.1,
) -> ScoreArrays:
    rng = np.random.default_rng(seed)
    n = n_windows * window
    t = np.repeat(np.linspace(0.0, 1.0, n_windows), window)
    ppi_len = rng.integers(8, 45, n)
    unk = rng.random(n) < unknown_share
    labels = np.where(unk, -1, rng.integers(0, N_CLASSES, n))
    speed = rng.uniform(0.15, 0.6, n) * (1.0 - drift * t * 0.6)
    ks = np.arange(1, K_MAX + 1)[None, :]
    p = np.clip(1 - np.exp(-ks * speed[:, None]) + rng.normal(0, 0.02, (n, K_MAX)), 0, 1)
    pred = np.where(rng.random((n, K_MAX)) < p, labels[:, None], (labels[:, None] + 1) % N_CLASSES)
    energy = rng.normal(-8.0, 1.0, (n, K_MAX)) + 3.0 * unk[:, None]
    nll = rng.random((n, K_MAX)) + 0.8 * unk[:, None]

    def f(a: np.ndarray, dt: type) -> np.ndarray:
        return fill_forward(a, ppi_len).astype(dt)

    z = np.zeros((n, K_MAX))
    return ScoreArrays(
        pred=f(pred, np.int16),
        conf=f(p, np.float16),
        margin=f(p * 0.8, np.float16),
        p_safe=f(p, np.float16),
        energy=f(energy, np.float32),
        energy_raw=f(energy, np.float32),
        mahal=f(z, np.float32),
        pkt_nll=f(nll, np.float32),
        next_entropy=f(z, np.float16),
        labels=labels,
        ppi_len=ppi_len,
        flow_index=np.arange(n),
        period=(np.arange(n) // window).astype(np.int16),
        meta={"checkpoint_hash": "h"},
    )


def fitted_unknown(val: ScoreArrays, weights=(1.0, 0.0, 0.5)) -> UnknownModel:
    """An unknown model with no Gaussians (the synthetic stream has no projection)."""
    d = 4
    bank = GaussianBank(
        means=np.zeros((K_MAX, N_CLASSES, d)),
        precision=np.tile(np.eye(d), (K_MAX, 1, 1)),
        bg_mean=np.zeros((K_MAX, d)),
        bg_precision=np.tile(np.eye(d), (K_MAX, 1, 1)),
        valid=np.zeros((K_MAX, N_CLASSES), dtype=bool),
    )
    return UnknownModel.fit(bank, val, weights=weights, checkpoint_hash="h")
