"""Self-supervised pretraining of the PAT (spec 007; plan phase 3, T5).

Two objectives on unlabelled flows, plus the masked-packet comparison:

- **NPP**, next-packet prediction (:func:`~adl_etc.training.pat_losses.npp_loss`).
- **PFC**, prefix-to-flow contrastive alignment: the representation of a short,
  augmented prefix must pick out its own full flow among the batch (symmetric
  InfoNCE, temperature 0.1), with **twin masking**: flows whose raw token
  sequences are identical are removed from each other's negatives (3.9% of
  anchors per 4,096 batch on D1's real training weeks, plan F12).
- **MPM**, masked packet modelling on the bidirectional PAT (``causal: false``),
  spec 007's "conventional SSL" comparison.

**Views** (built by :class:`SSLBatches` from the raw PPI, with
:mod:`adl_etc.data.augment_batch`):

- ``a``, the *prefix* view: drop (p ~ U[0, 0.2] per flow), reorder (q = 0.05),
  ipt and size jitter, then a crop to K ~ U{3 .. len - 1}. Its ``proj`` at
  position K is PFC's anchor. Drop and reorder live here, where the contrastive
  target makes the representation invariant to them (H5).
- ``b``, the *full* view: ipt and size jitter only, full length. Its ``proj`` at
  the last packet is PFC's positive, and NPP is computed on it against the
  **raw** next tokens (spec 007: "targets are the raw next packet ... so that the
  model learns to denoise"). Leaving drop/reorder out of this view keeps packet k
  of the view aligned with packet k of the raw flow, which is what makes "the
  raw next packet" well defined.

Both views go through one forward pass (concatenated along the batch). All the
randomness of a batch is seeded from ``(seed, the batch's flow indices)``, and
the loop draws those indices from ``(seed, epoch)``, so a resumed run makes
exactly the same views (bit-exact resume, as for every other run).

:class:`PretrainPAT` is a :class:`~adl_etc.models.pat.PAT` with the same
parameters (its checkpoint loads straight into fine-tuning) whose forward
handles the views; given an ordinary batch it is an ordinary PAT.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from adl_etc.data import augment_batch as AB
from adl_etc.data import features as FT
from adl_etc.data import ppi as P
from adl_etc.models.pat import PAT
from adl_etc.training.datasets import FlowBatches, to_torch
from adl_etc.training.pat_losses import npp_loss

VIEW_KEYS = ("tokens", "cont", "mask", "ppi_len")


# --- the model ------------------------------------------------------------------------------


class PretrainPAT(PAT):
    """The PAT, with a forward that understands SSL views: a batch holding
    ``tokens_a``/``tokens_b`` returns ``{"a": ..., "b": ...}``, one holding
    ``tokens_m`` returns ``{"m": ...}``, an ordinary batch returns the PAT's
    usual heads. Same parameters, same ``state_dict`` keys as :class:`PAT`."""

    def _view(self, batch: dict[str, Any], suffix: str) -> dict[str, Any]:
        return {key: batch[f"{key}_{suffix}"] for key in VIEW_KEYS if f"{key}_{suffix}" in batch}

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        if "tokens_a" in batch:
            a, b = self._view(batch, "a"), self._view(batch, "b")
            n = a["tokens"].shape[0]
            both = {key: torch.cat([a[key], b[key]]) for key in a}
            out = super().forward(both)
            return {
                "a": {k: v[:n] for k, v in out.items()},
                "b": {k: v[n:] for k, v in out.items()},
            }
        if "tokens_m" in batch:
            return {"m": super().forward(self._view(batch, "m"))}
        return super().forward(batch)


# --- the views ------------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentSettings:
    """Spec 003's augmentation strengths for the prefix view."""

    drop_p_max: float = AB.DROP_P_MAX
    reorder_q: float = AB.REORDER_Q
    ipt_s: float = AB.IPT_JITTER_S
    size_sd: float = AB.SIZE_JITTER_SD
    mask_fraction: float = 0.3  # MPM


def token_hashes(tokens: np.ndarray) -> np.ndarray:
    """A 64-bit fingerprint of each flow's token sequence: equal sequences get
    equal values, so twins can be found by comparing one integer."""
    flat = tokens.reshape(len(tokens), -1).astype(np.uint64)
    weights = np.random.default_rng(20261004).integers(
        1, 2**63, size=flat.shape[1], dtype=np.uint64
    ) | np.uint64(1)
    with np.errstate(over="ignore"):
        return (flat * weights[None, :]).sum(axis=1).view(np.int64)


class SSLBatches(FlowBatches):
    """:class:`FlowBatches` whose ``batch(idx)`` also carries the SSL views.

    ``mode="npp_pfc"`` adds views ``a`` and ``b`` (see the module docstring),
    ``eligible_a`` (PFC skips flows too short to crop) and ``twin`` (token
    fingerprints); ``mode="mpm"`` adds view ``m`` (30% of each flow's real
    packets blanked to the padding token, still attended to) and ``mpm_mask``.
    """

    def __init__(
        self,
        *args: Any,
        mode: str = "npp_pfc",
        seed: int = 0,
        augment: AugmentSettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if mode not in ("npp_pfc", "mpm"):
            raise ValueError(f"mode must be 'npp_pfc' or 'mpm', got {mode!r}")
        if "tokens" not in self.views:
            raise ValueError("SSL needs the tokens view")
        self.mode = mode
        self.seed = seed
        self.augment = augment or AugmentSettings()

    def _rng(self, idx: np.ndarray) -> np.random.Generator:
        key = zlib.crc32(np.ascontiguousarray(idx, dtype=np.int64).tobytes())
        return np.random.default_rng([self.seed, key])

    def _encode(self, ppi: np.ndarray, ppi_len: np.ndarray, suffix: str) -> dict[str, np.ndarray]:
        mask = FT.padding_mask(ppi_len)
        out = {
            f"tokens_{suffix}": FT.tokenize(ppi, ppi_len),
            f"mask_{suffix}": mask,
            f"ppi_len_{suffix}": np.asarray(ppi_len).astype(np.int64),
        }
        if "continuous" in self.views:
            assert self.standardizer is not None
            cont = self.standardizer.transform_continuous(FT.continuous(ppi, ppi_len))
            out[f"cont_{suffix}"] = cont * mask[:, :, None]
        return out

    def batch(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        out = super().batch(idx)
        if "tokens" not in out:
            out["tokens"] = FT.tokenize(self.data.ppi[idx], self.data.ppi_len[idx])
        ppi, length = self.data.ppi[idx], self.data.ppi_len[idx]
        rng, aug = self._rng(idx), self.augment
        if self.mode == "npp_pfc":
            p = rng.uniform(0.0, aug.drop_p_max, len(idx))
            a, a_len = AB.drop(ppi, length, p, rng)
            a = AB.reorder(a, a_len, aug.reorder_q, rng)
            a = AB.ipt_jitter(a, a_len, aug.ipt_s, rng)
            a = AB.size_jitter(a, a_len, aug.size_sd, rng)
            k, eligible = AB.prefix_crop(a_len, rng)
            a[np.arange(P.K_MAX)[None, :] >= k[:, None]] = 0
            b = AB.size_jitter(AB.ipt_jitter(ppi, length, aug.ipt_s, rng), length, aug.size_sd, rng)
            out.update(self._encode(a, k, "a"))
            out.update(self._encode(b, length, "b"))
            out["eligible_a"] = eligible
            out["twin"] = token_hashes(out["tokens"])
        else:
            valid = FT.padding_mask(length)
            scores = np.where(valid, rng.random(valid.shape), np.inf)
            n_mask = np.maximum(1, np.round(aug.mask_fraction * length)).astype(np.int64)
            ranks = scores.argsort(axis=1).argsort(axis=1)
            masked = valid & (ranks < n_mask[:, None])
            enc = self._encode(ppi, length, "m")
            enc["tokens_m"] = np.where(masked[..., None], FT.PAD_INDEX, enc["tokens_m"])
            if "cont_m" in enc:
                enc["cont_m"] = np.where(masked[..., None], 0.0, enc["cont_m"])
            enc["mask_m"] = valid  # a blanked packet is still there, just unseen
            out.update(enc)
            out["mpm_mask"] = masked
        return out


# --- losses ---------------------------------------------------------------------------------


def pfc_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    twin: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric InfoNCE between L2-normalised ``anchors[i]`` and
    ``positives[i]`` with in-batch negatives, twins (equal ``twin`` keys, i != j)
    masked out of both directions. Returns ``(loss, top-1 retrieval accuracy)``."""
    n = anchors.shape[0]
    if n < 2:
        zero = anchors.sum() * 0.0
        return zero, zero.detach()
    sim = anchors.float() @ positives.float().T / temperature
    same = twin[:, None] == twin[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(same & ~eye, float("-inf"))
    target = torch.arange(n, device=sim.device)
    loss = 0.5 * (F.cross_entropy(sim, target) + F.cross_entropy(sim.T, target))
    top1 = (sim.argmax(dim=1) == target).float().mean()
    return loss, top1.detach()


def _last(t: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    return t[torch.arange(t.shape[0], device=t.device), (length - 1).clamp_min(0)]


def uniformity(proj: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """Wang & Isola's ``log E exp(-t ||u - v||^2)`` over distinct pairs: about
    -4 for well-spread unit vectors in 128 dimensions, 0 for a collapse."""
    sq = torch.cdist(proj.float(), proj.float()).pow(2)
    off = ~torch.eye(len(proj), dtype=torch.bool, device=proj.device)
    return torch.logsumexp(-t * sq[off], dim=0) - torch.log(torch.tensor(float(off.sum())))


COLLAPSE_UNIFORMITY = -0.5  # above this, `proj` has (nearly) collapsed


@dataclass
class SSLObjective:
    """``L = npp_weight · L_NPP + pfc_weight · L_PFC`` (spec 007: ``λ_PFC`` is
    ``pfc_weight``; NPP-only is ``pfc_weight = 0``, PFC-only ``npp_weight = 0``).
    Label-free; the best checkpoint is the lowest validation loss on fixed
    (seeded) validation views."""

    npp_weight: float = 1.0
    pfc_weight: float = 1.0
    temperature: float = 0.1
    monitor: str = "val_ssl_loss"
    higher_is_better: bool = False
    uses_labels: bool = False

    def loss(
        self, outputs: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b_out = outputs["b"]
        npp, npp_parts = npp_loss(b_out, {"tokens": batch["tokens"], "ppi_len": batch["ppi_len"]})
        rows = batch["eligible_a"]
        anchors = _last(outputs["a"]["proj"], batch["ppi_len_a"])[rows]
        positives = _last(b_out["proj"], batch["ppi_len_b"])[rows]
        pfc, top1 = pfc_loss(anchors, positives, batch["twin"][rows], temperature=self.temperature)
        total = self.npp_weight * npp + self.pfc_weight * pfc
        return total, {"npp": npp, "pfc": pfc, "pfc_top1": top1}

    @torch.no_grad()
    def validate(
        self, model: nn.Module, data: FlowBatches, *, batch_size: int,
        device: torch.device, amp: bool,
    ) -> dict[str, float]:  # fmt: skip
        model.eval()
        sums: dict[str, float] = {}
        n = 0
        knn_z: list[np.ndarray] = []
        knn_y: list[np.ndarray] = []
        for _, np_batch in data.iter_batches(batch_size):
            batch = to_torch(np_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(batch)
            _, parts = self.loss(out, batch)
            rows = batch["eligible_a"]
            anchors = _last(out["a"]["proj"], batch["ppi_len_a"])[rows]
            positives = _last(out["b"]["proj"], batch["ppi_len_b"])[rows]
            m = len(np_batch["y"])
            for key, value in {
                **{k: float(v) for k, v in parts.items()},
                "align": float((anchors.float() - positives.float()).pow(2).sum(-1).mean())
                if len(anchors) else 0.0,
                "uniform": float(uniformity(positives[:2048])) if len(positives) > 1 else 0.0,
            }.items():
                sums[key] = sums.get(key, 0.0) + value * m
            n += m
            final = _last(out["b"]["proj"], batch["ppi_len_b"]).float().cpu().numpy()
            known = np_batch["y"] >= 0
            knn_z.append(final[known])
            knn_y.append(np_batch["y"][known])
        metrics = {f"val_{k}": v / n for k, v in sums.items()}
        metrics["val_ssl_loss"] = (
            self.npp_weight * metrics["val_npp"] + self.pfc_weight * metrics["val_pfc"]
        )
        metrics["val_knn_acc"] = knn_accuracy(np.concatenate(knn_z), np.concatenate(knn_y))
        metrics["val_collapsed"] = float(metrics["val_uniform"] > COLLAPSE_UNIFORMITY)
        return metrics


@dataclass
class MPMObjective:
    """Masked packet modelling (spec 007's comparison): reconstruct the blanked
    packets' size, ipt and direction tokens from the bidirectional context, read
    through the ``next_*`` heads at each blanked position. Their weights carry
    over to fine-tuning, where ``L_NPP`` retrains the heads for their causal job."""

    monitor: str = "val_mpm"
    higher_is_better: bool = False
    uses_labels: bool = False

    def loss(
        self, outputs: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out, raw, masked = outputs["m"], batch["tokens"], batch["mpm_mask"]
        parts = {}
        for head, channel in (("next_size", P.SIZE_POS), ("next_ipt", P.IPT_POS),
                              ("next_dir", P.DIR_POS)):  # fmt: skip
            logits = out[head].float()[masked]
            parts[f"mpm_{head.removeprefix('next_')}"] = F.cross_entropy(
                logits, raw[..., channel][masked]
            )
        return sum(parts.values(), torch.zeros((), device=raw.device)), parts

    @torch.no_grad()
    def validate(
        self, model: nn.Module, data: FlowBatches, *, batch_size: int,
        device: torch.device, amp: bool,
    ) -> dict[str, float]:  # fmt: skip
        model.eval()
        total, n = 0.0, 0
        for _, np_batch in data.iter_batches(batch_size):
            batch = to_torch(np_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(batch)
            loss, _ = self.loss(out, batch)
            total += float(loss) * len(np_batch["y"])
            n += len(np_batch["y"])
        return {"val_mpm": total / n}


def knn_accuracy(z: np.ndarray, y: np.ndarray, *, k: int = 20, cap: int = 20000) -> float:
    """kNN (cosine, majority of ``k``) accuracy of the ``proj`` embedding: the
    first half of up to ``cap`` labelled flows are references, the second half
    queries (spec 007's kNN probe, cheap enough to log every epoch)."""
    z, y = z[:cap], y[:cap]
    half = len(y) // 2
    if half < k or len(np.unique(y)) < 2:
        return 0.0
    ref, query = z[:half], z[half:]
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True).clip(1e-12)
    query = query / np.linalg.norm(query, axis=1, keepdims=True).clip(1e-12)
    correct = 0
    for start in range(0, len(query), 2048):
        sims = query[start : start + 2048] @ ref.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        votes = y[:half][top]
        pred = np.array([np.bincount(row).argmax() for row in votes])
        correct += int((pred == y[half:][start : start + 2048]).sum())
    return correct / len(query)
