"""Seeding and determinism (spec 014).

One user-facing seed per run, fanned out into independent streams per purpose.
The point of the fan-out: if data order, augmentation and initialisation all
drew from one shared generator, adding a single augmentation call would shift
every later draw and silently change the data order of every existing run.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def _check_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (when installed) PyTorch on CPU and GPU, and ask
    PyTorch for deterministic algorithms.

    Works without torch, so phase-1-only environments keep functioning.
    Determinism is requested with ``warn_only=True``: some GPU kernels have no
    deterministic implementation, and spec 014 documents the residual
    run-to-run noise (AMP with SDPA on GPU) rather than failing on it. The
    three-seed protocol absorbs it.
    """
    _check_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    # Must be set before the first CUDA matmul for use_deterministic_algorithms
    # to be honoured by cuBLAS; harmless on CPU.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def seeded_generator(seed: int, purpose: str) -> np.random.Generator:
    """An independent NumPy generator for ``purpose`` (e.g. ``"data_order"``,
    ``"augment"``, ``"init"``) derived from ``seed``.

    The purpose string is hashed with SHA-256 rather than Python's ``hash()``,
    which is salted per process and would make streams differ between runs.
    """
    _check_seed(seed)
    if not purpose:
        raise ValueError("purpose must be a non-empty string")
    tag = int.from_bytes(hashlib.sha256(purpose.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(np.random.SeedSequence([seed, tag]))
