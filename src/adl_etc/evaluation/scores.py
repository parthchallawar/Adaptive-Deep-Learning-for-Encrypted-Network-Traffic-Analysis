"""The compact per-flow score artefact every phase-4 experiment replays from (plan
phase 4, T1; ``docs/design-phase-4.md``).

One model pass per (run, split) writes a :class:`ScoreArrays`: for every flow and
every prefix length ``k`` (index ``k - 1``) a handful of scalars, never the full
logits (~2 GB per run) and never the 128-d projection. Policies, the unknown score
and the budget controller are then NumPy over these arrays, so a threshold sweep
takes seconds and needs no model and no torch.

Every array is ``[N, 30]`` and **filled forward**: a position at or beyond a flow's
``ppi_len`` repeats the flow's last real value, so a scan over ``k`` never needs a
mask (a policy still stops at ``min(30, ppi_len)`` on its own).

NumPy only, like the rest of ``adl_etc.evaluation``.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from adl_etc.utils.atomic import atomic_write_bytes, atomic_write_json
from adl_etc.utils.provenance import stable_hash

K_MAX = 30
SCHEMA_VERSION = 1
META_FILE = "scores.json"

#: ``[N, 30]`` fields and their on-disk dtypes.
FIELDS_2D: dict[str, np.dtype] = {
    "pred": np.dtype(np.int16),  # argmax of the prior-adjusted logits (what the policies use)
    "pred_raw": np.dtype(np.int16),  # argmax of the raw logits (what phase 3's reports score)
    "conf": np.dtype(np.float16),  # largest calibrated class probability
    "margin": np.dtype(np.float16),  # top-1 minus top-2 calibrated probability
    "p_safe": np.dtype(np.float16),  # calibrated P(correct | commit now)
    "energy": np.dtype(np.float32),  # -T_k * logsumexp(adjusted / T_k)
    "energy_raw": np.dtype(np.float32),  # energy of the raw logits (P-CAPE's gate)
    "mahal": np.dtype(np.float32),  # relative Mahalanobis (zeros if no unknown model)
    "pkt_nll": np.dtype(np.float32),  # NLL of packet j+1 given the first j (index j)
    "next_entropy": np.dtype(np.float16),  # the safe head's own next-packet entropy feature
}
#: ``[N]`` fields.
FIELDS_1D: dict[str, np.dtype] = {
    "labels": np.dtype(np.int64),  # model class index; -1 = unknown-class flow
    "ppi_len": np.dtype(np.int64),
    "flow_index": np.dtype(np.int64),  # the index the sampler drew, within the pre-sample split
    "period": np.dtype(np.int16),  # index into meta["periods"]
}


def fill_forward(values: np.ndarray, ppi_len: np.ndarray) -> np.ndarray:
    """Positions ``>= ppi_len`` repeat the flow's last real value (``values`` is
    ``[N, K]``). A flow with ``ppi_len`` above ``K`` is left untouched."""
    n, k = values.shape
    last = np.clip(np.asarray(ppi_len, dtype=np.int64) - 1, 0, k - 1)
    idx = np.minimum(np.arange(k)[None, :], last[:, None])
    return np.take_along_axis(values, idx, axis=1)


def running_surprise(pkt_nll: np.ndarray, ppi_len: np.ndarray) -> np.ndarray:
    """Spec 010's surprise ``S_k``: the mean next-packet negative log-likelihood
    over packets ``2..k``; ``S_1 = 0``. ``pkt_nll[:, j]`` is the NLL of packet
    ``j + 1`` given the first ``j``, so it only exists for ``1 <= j <= ppi_len - 1``.
    Filled forward past a flow's length. float64 internally, float32 out."""
    n, k_max = pkt_nll.shape
    j = np.arange(k_max)[None, :]
    valid = (j >= 1) & (j <= (np.asarray(ppi_len)[:, None] - 1))
    total = np.cumsum(np.where(valid, pkt_nll.astype(np.float64), 0.0), axis=1)
    count = np.cumsum(valid, axis=1)
    out = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    return fill_forward(out, ppi_len).astype(np.float32)


@dataclass(frozen=True)
class ScoreArrays:
    """Per-flow, per-prefix scores of one model on one sample of one split."""

    pred: np.ndarray
    conf: np.ndarray
    margin: np.ndarray
    p_safe: np.ndarray
    energy: np.ndarray
    energy_raw: np.ndarray
    mahal: np.ndarray
    pkt_nll: np.ndarray
    next_entropy: np.ndarray
    labels: np.ndarray
    ppi_len: np.ndarray
    flow_index: np.ndarray
    period: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    pred_raw: np.ndarray = None  # type: ignore[assignment]  # unset = same as ``pred``

    def __post_init__(self) -> None:
        if self.pred_raw is None:
            object.__setattr__(self, "pred_raw", self.pred)
        n = len(self.labels)
        for name in FIELDS_2D:
            arr = getattr(self, name)
            if arr.shape != (n, K_MAX):
                raise ValueError(f"{name} must be [{n}, {K_MAX}], got {arr.shape}")
        for name in FIELDS_1D:
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} must be [{n}], got {arr.shape}")
        if n and int(self.ppi_len.min()) < 1:
            raise ValueError("every flow needs at least one packet (ppi_len >= 1)")

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def is_unknown(self) -> np.ndarray:
        return self.labels < 0

    @property
    def last_k(self) -> np.ndarray:
        """``min(30, ppi_len)``: the last prefix a flow can be decided at."""
        return np.minimum(self.ppi_len, K_MAX)

    @property
    def surprise(self) -> np.ndarray:
        return running_surprise(self.pkt_nll, self.ppi_len)

    def take(self, idx: np.ndarray) -> ScoreArrays:
        """The flows at ``idx`` (integer indices or a boolean mask)."""
        kw = {name: getattr(self, name)[idx] for name in (*FIELDS_2D, *FIELDS_1D)}
        return ScoreArrays(**kw, meta=dict(self.meta))

    def with_mahal(self, mahal: np.ndarray, meta: dict[str, Any] | None = None) -> ScoreArrays:
        return replace(
            self,
            mahal=np.asarray(mahal, dtype=FIELDS_2D["mahal"]),
            meta={**self.meta, **(meta or {})},
        )

    # -- identity ----------------------------------------------------------------

    def content_hash(self) -> str:
        """SHA-256 over every array's bytes and the metadata, so two passes that
        should be identical can be compared in one line."""
        import hashlib

        h = hashlib.sha256()
        for name in (*FIELDS_2D, *FIELDS_1D):
            arr = np.ascontiguousarray(getattr(self, name))
            h.update(name.encode())
            h.update(str(arr.dtype).encode())
            h.update(arr.tobytes())
        h.update(stable_hash(self.meta).encode())
        return h.hexdigest()

    # -- persistence -------------------------------------------------------------

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, dtype in {**FIELDS_2D, **FIELDS_1D}.items():
            buf = io.BytesIO()
            np.save(buf, np.asarray(getattr(self, name)).astype(dtype, copy=False))
            atomic_write_bytes(out_dir / f"{name}.npy", buf.getvalue())
        atomic_write_json(
            out_dir / META_FILE,
            {
                "schema_version": SCHEMA_VERSION,
                "n_flows": len(self),
                "content_hash": self.content_hash(),
                **self.meta,
            },
        )

    @classmethod
    def load(cls, out_dir: str | Path) -> ScoreArrays:
        out_dir = Path(out_dir)
        meta = json.loads((out_dir / META_FILE).read_text(encoding="utf-8"))
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{out_dir}: score schema {meta.get('schema_version')}, expected {SCHEMA_VERSION}"
            )
        arrays = {name: np.load(out_dir / f"{name}.npy") for name in (*FIELDS_2D, *FIELDS_1D)}
        stored_hash = meta.pop("content_hash")
        meta.pop("schema_version")
        meta.pop("n_flows")
        scores = cls(**arrays, meta=meta)
        if scores.content_hash() != stored_hash:
            raise ValueError(f"{out_dir}: contents do not match the recorded hash (damaged copy?)")
        return scores


def concat(parts: list[ScoreArrays], meta: dict[str, Any] | None = None) -> ScoreArrays:
    """Stack score sets (e.g. one per drift week). ``meta`` defaults to the first part's."""
    if not parts:
        raise ValueError("nothing to concatenate")
    kw = {
        name: np.concatenate([getattr(p, name) for p in parts]) for name in (*FIELDS_2D, *FIELDS_1D)
    }
    return ScoreArrays(**kw, meta=dict(meta if meta is not None else parts[0].meta))


# --- turning one model chunk into score columns (NumPy) ---------------------------------------


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def class_scores(
    cls_logits: np.ndarray,
    adjustment: np.ndarray,
    temperatures: np.ndarray,
) -> dict[str, np.ndarray]:
    """From raw class logits ``[B, 30, C]``: the prior-adjusted argmax, calibrated
    confidence and margin, and the two energies. ``temperatures`` is per K."""
    raw = np.asarray(cls_logits, dtype=np.float64)
    adjusted = raw + np.asarray(adjustment, dtype=np.float64)
    temp = np.asarray(temperatures, dtype=np.float64)[None, :, None]
    logp = _log_softmax(adjusted / temp)
    p = np.exp(logp)
    top2 = np.partition(p, -2, axis=-1)[..., -2:]  # the two largest, unordered
    top1 = top2.max(axis=-1)
    second = top2.min(axis=-1)
    energy = -temp[..., 0] * _logsumexp(adjusted / temp)
    return {
        "pred": adjusted.argmax(-1),
        "pred_raw": raw.argmax(-1),
        "conf": top1,
        "margin": top1 - second,
        "energy": energy,
        "energy_raw": -_logsumexp(raw),
    }


def _logsumexp(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))[..., 0]
