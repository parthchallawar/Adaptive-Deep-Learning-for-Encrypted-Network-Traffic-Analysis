"""Spec 010's per-prefix unknown score (plan phase 4, T2; ``docs/design-phase-4.md`` s.1).

Three signals per prefix length ``k``:

- **energy** ``E_k`` (already in the score files),
- **relative Mahalanobis** ``M_k``: distance of the L2-normalised projection to the
  nearest class-conditional Gaussian (shared covariance per K) minus its distance to
  a class-free background Gaussian,
- **surprise** ``S_k``: the mean next-packet negative log-likelihood over packets 2..k.

``u_k = w_E z(E_k) + w_M z(M_k) + w_S z(S_k)`` with each ``z`` standardised by **known
validation** flows that reached K, and the per-K reject threshold is a quantile of
``u_k`` over those same flows (so one operating point means the same false-positive
rate at every K).

**Nothing here is ever fitted on unknown-class flows** (labels ``< 0``). The held-out
classes are the same in validation and test, so doing so would leak (plan P4-F4);
``tests/inference/test_unknown.py`` pins that down.

Torch-free by design (plan D2): this module is shared by offline evaluation, which may
not import torch, and by the service. The model pass that feeds it lives in
``adl_etc.training.score_pass``.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adl_etc.evaluation.scores import K_MAX, ScoreArrays, fill_forward
from adl_etc.utils.atomic import atomic_write_bytes, atomic_write_json

N_QUANTILES = 1001
QUANTILE_GRID = np.linspace(0.0, 1.0, N_QUANTILES)
DEFAULT_WEIGHTS = (1.0, 1.0, 0.5)  # energy, Mahalanobis, surprise (beta = 0.5)
SIGNALS = ("energy", "mahal", "surprise")
SCHEMA_VERSION = 1


class UnknownModelError(ValueError):
    pass


# --- fitting the Gaussians from running sums ----------------------------------------------------


class GaussianSums:
    """Running first and second moments of the projection, per K and class.

    ``valid[k-1, c]`` says which classes get a Gaussian at K: those with at least
    ``min_class_flows`` training flows that reached K. It is decided **before** the model
    pass from the sample's labels and lengths (:func:`class_validity`), because the shared
    covariance must only see flows of valid classes. float64 throughout.
    """

    def __init__(self, n_classes: int, valid: np.ndarray, dim: int = 128) -> None:
        if valid.shape != (K_MAX, n_classes):
            raise ValueError(f"valid must be [{K_MAX}, {n_classes}], got {valid.shape}")
        self.n_classes, self.dim, self.valid = n_classes, dim, valid.astype(bool)
        self.sum_c = np.zeros((K_MAX, n_classes, dim))
        self.n_c = np.zeros((K_MAX, n_classes), dtype=np.int64)
        self.sum_all = np.zeros((K_MAX, dim))
        self.n_all = np.zeros(K_MAX, dtype=np.int64)
        self.outer_all = np.zeros((K_MAX, dim, dim))
        self.outer_valid = np.zeros((K_MAX, dim, dim))

    def add(self, proj: np.ndarray, labels: np.ndarray, ppi_len: np.ndarray) -> None:
        """One chunk: ``proj`` ``[B, 30, dim]``, ``labels`` ``[B]`` (known classes only)."""
        labels = np.asarray(labels)
        if (labels < 0).any():
            raise UnknownModelError("unknown-class flows must never reach the Gaussian fit")
        x = np.asarray(proj, dtype=np.float64)
        for k in range(K_MAX):
            m = ppi_len >= k + 1
            if not m.any():
                continue
            xk, yk = x[m, k], labels[m]
            np.add.at(self.sum_c[k], yk, xk)
            self.n_c[k] += np.bincount(yk, minlength=self.n_classes)
            self.sum_all[k] += xk.sum(0)
            self.n_all[k] += len(yk)
            self.outer_all[k] += xk.T @ xk
            v = self.valid[k, yk]
            if v.any():
                self.outer_valid[k] += xk[v].T @ xk[v]

    def bank(self, shrinkage: float = 0.1) -> GaussianBank:
        d = self.dim
        eye = np.eye(d)
        means = np.zeros((K_MAX, self.n_classes, d))
        prec = np.zeros((K_MAX, d, d))
        bg_mean = np.zeros((K_MAX, d))
        bg_prec = np.zeros((K_MAX, d, d))
        for k in range(K_MAX):
            cv = self.valid[k]
            n_v = int(self.n_c[k][cv].sum())
            if cv.any() and n_v > int(cv.sum()):
                mu = self.sum_c[k][cv] / self.n_c[k][cv, None]
                means[k][cv] = mu
                scatter = self.outer_valid[k] - (mu * self.n_c[k][cv, None]).T @ mu
                cov = scatter / (n_v - int(cv.sum()))
                cov = (1 - shrinkage) * cov + shrinkage * (np.trace(cov) / d) * eye
                prec[k] = np.linalg.inv(cov)
            else:
                prec[k] = eye
            n = int(self.n_all[k])
            if n > 1:
                bg_mean[k] = self.sum_all[k] / n
                cov = self.outer_all[k] / n - np.outer(bg_mean[k], bg_mean[k])
                cov = (1 - shrinkage) * cov + shrinkage * (np.trace(cov) / d) * eye
                bg_prec[k] = np.linalg.inv(cov)
            else:
                bg_prec[k] = eye
        return GaussianBank(means, prec, bg_mean, bg_prec, self.valid.copy())


def class_validity(
    labels: np.ndarray, ppi_len: np.ndarray, n_classes: int, min_class_flows: int = 50
) -> np.ndarray:
    """``[30, C]`` bool: the class has at least ``min_class_flows`` of these flows that
    reached K. ``labels`` are known-class model indices."""
    valid = np.zeros((K_MAX, n_classes), dtype=bool)
    for k in range(K_MAX):
        m = ppi_len >= k + 1
        counts = np.bincount(labels[m], minlength=n_classes)
        valid[k] = counts >= min_class_flows
    return valid


@dataclass(frozen=True)
class GaussianBank:
    """Per-K class means, a shared precision matrix, and a background Gaussian."""

    means: np.ndarray  # [30, C, d]
    precision: np.ndarray  # [30, d, d]
    bg_mean: np.ndarray  # [30, d]
    bg_precision: np.ndarray  # [30, d, d]
    valid: np.ndarray  # [30, C] bool

    def mahal(self, proj: np.ndarray) -> np.ndarray:
        """Relative Mahalanobis ``M_k`` for ``proj`` ``[B, 30, d]`` -> ``[B, 30]``.
        A K with no valid class scores 0 (the standardisation then ignores it)."""
        x = np.asarray(proj, dtype=np.float64)
        out = np.zeros(x.shape[:2])
        for k in range(K_MAX):
            cv = self.valid[k]
            if not cv.any():
                continue
            xk, p, mu = x[:, k], self.precision[k], self.means[k][cv]
            xp = xk @ p
            d = (xp * xk).sum(1)[:, None] - 2.0 * (xp @ mu.T) + ((mu @ p) * mu).sum(1)[None, :]
            xb = xk - self.bg_mean[k]
            out[:, k] = d.min(1) - ((xb @ self.bg_precision[k]) * xb).sum(1)
        return out

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "means": self.means,
            "precision": self.precision,
            "bg_mean": self.bg_mean,
            "bg_precision": self.bg_precision,
            "valid": self.valid,
        }


# --- the score and its thresholds -------------------------------------------------------------


@dataclass(frozen=True)
class UnknownModel:
    """The Gaussian bank plus everything needed to turn three signals into ``u_k``
    and a per-K threshold. Fitted on known validation flows only."""

    bank: GaussianBank
    z_mean: np.ndarray  # [30, 3] energy, mahal, surprise
    z_std: np.ndarray  # [30, 3] (inf where a K has no usable flow, so z = 0 there)
    quantiles: np.ndarray  # [30, N_QUANTILES] quantiles of u_k over known val flows
    weights: tuple[float, float, float]
    checkpoint_hash: str

    # -- scoring ---------------------------------------------------------------

    def components(self, energy: np.ndarray, mahal: np.ndarray, surprise: np.ndarray) -> np.ndarray:
        """Standardised signals, ``[N, 30, 3]``."""
        raw = np.stack(
            [np.asarray(a, dtype=np.float64) for a in (energy, mahal, surprise)], axis=-1
        )
        return (raw - self.z_mean[None]) / self.z_std[None]

    def score(self, energy: np.ndarray, mahal: np.ndarray, surprise: np.ndarray) -> np.ndarray:
        """``u_k``, ``[N, 30]``. Higher = more unknown."""
        z = self.components(energy, mahal, surprise)
        return z @ np.asarray(self.weights, dtype=np.float64)

    def score_of(self, s: ScoreArrays) -> np.ndarray:
        return self.score(s.energy, s.mahal, s.surprise)

    def theta_reject(self, f: float) -> np.ndarray:
        """Per-K threshold, ``[30]``: the ``1 - f`` quantile of ``u_k`` over known
        validation flows, so about a fraction ``f`` of known flows is rejected at
        every K."""
        if not 0.0 < f < 1.0:
            raise ValueError(f"f must be in (0, 1), got {f}")
        return np.array([np.interp(1.0 - f, QUANTILE_GRID, q) for q in self.quantiles])

    # -- fitting ---------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        bank: GaussianBank,
        val: ScoreArrays,
        *,
        weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
        checkpoint_hash: str,
    ) -> UnknownModel:
        """Standardise and take quantiles from the **known** flows of ``val`` that reached
        each K. ``val.mahal`` must have been computed with ``bank``."""
        known = val.labels >= 0
        surprise = val.surprise
        signals = (val.energy, val.mahal, surprise)
        z_mean = np.zeros((K_MAX, 3))
        z_std = np.full((K_MAX, 3), np.inf)
        reached = [known & (val.ppi_len >= k + 1) for k in range(K_MAX)]
        for k in range(K_MAX):
            m = reached[k]
            if m.sum() < 2:
                continue
            for j, sig in enumerate(signals):
                col = np.asarray(sig[m, k], dtype=np.float64)
                z_mean[k, j] = col.mean()
                sd = col.std()
                z_std[k, j] = sd if sd > 1e-8 else np.inf  # a constant signal contributes 0
        proto = cls(
            bank, z_mean, z_std, np.full((K_MAX, N_QUANTILES), np.inf), weights, checkpoint_hash
        )
        u = proto.score(val.energy, val.mahal, surprise)
        quantiles = np.full((K_MAX, N_QUANTILES), np.inf)
        for k in range(K_MAX):
            m = reached[k]
            if m.sum() >= 2:
                quantiles[k] = np.quantile(u[m, k], QUANTILE_GRID)
        return cls(bank, z_mean, z_std, quantiles, weights, checkpoint_hash)

    # -- persistence -----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Writes ``<path>.npz`` and ``<path>.json`` (``path`` without a suffix)."""
        path = Path(path)
        buf = io.BytesIO()
        arrays = {
            **self.bank.to_arrays(),
            "z_mean": self.z_mean,
            "z_std": self.z_std,
            "quantiles": self.quantiles,
        }
        np.savez(buf, **arrays)  # type: ignore[arg-type]  # numpy's stub reads **kw as allow_pickle
        atomic_write_bytes(path.with_suffix(".npz"), buf.getvalue())
        atomic_write_json(
            path.with_suffix(".json"),
            {
                "schema_version": SCHEMA_VERSION,
                "checkpoint_hash": self.checkpoint_hash,
                "weights": list(self.weights),
                "signals": list(SIGNALS),
                # a K with few classes here has a weak Mahalanobis signal: check before trusting it
                "classes_with_gaussian": self.bank.valid.sum(1).tolist(),
            },
        )

    @classmethod
    def load(cls, path: str | Path, *, checkpoint_hash: str | None = None) -> UnknownModel:
        path = Path(path)
        meta: dict[str, Any] = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise UnknownModelError(f"unknown-model schema {meta.get('schema_version')}")
        if checkpoint_hash is not None and checkpoint_hash != meta["checkpoint_hash"]:
            raise UnknownModelError(
                f"unknown model was fitted for checkpoint {meta['checkpoint_hash'][:12]}, "
                f"not {checkpoint_hash[:12]}"
            )
        with np.load(path.with_suffix(".npz")) as z:
            bank = GaussianBank(
                z["means"], z["precision"], z["bg_mean"], z["bg_precision"], z["valid"]
            )
            return cls(
                bank,
                z["z_mean"],
                z["z_std"],
                z["quantiles"],
                tuple(meta["weights"]),
                meta["checkpoint_hash"],
            )


# --- evaluation helpers ----------------------------------------------------------------------


def rejection_k(u: np.ndarray, theta: np.ndarray, last_k: np.ndarray, k_ood: int = 3) -> np.ndarray:
    """First K in ``[k_ood, last_k]`` where ``u_k >= theta_k``; 0 where none. ``u`` is
    ``[N, 30]``, ``theta`` ``[30]``, ``last_k`` ``[N]`` (= ``min(30, ppi_len)``)."""
    ks = np.arange(1, K_MAX + 1)[None, :]
    hit = (u >= theta[None, :]) & (ks >= k_ood) & (ks <= last_k[:, None])
    first = hit.argmax(1) + 1
    return np.where(hit.any(1), first, 0)


def energy_gate_threshold(
    energy_raw: np.ndarray, labels: np.ndarray, ppi_len: np.ndarray, k: int, f: float
) -> float:
    """The fixed-K baseline of spec 010: the energy value that rejects a fraction ``f`` of
    known validation flows that reached ``k``."""
    m = (labels >= 0) & (ppi_len >= k)
    if m.sum() < 2:
        return float("inf")
    return float(np.quantile(energy_raw[m, k - 1], 1.0 - f))


__all__ = [
    "DEFAULT_WEIGHTS",
    "GaussianBank",
    "GaussianSums",
    "UnknownModel",
    "UnknownModelError",
    "class_validity",
    "energy_gate_threshold",
    "fill_forward",
    "rejection_k",
]
