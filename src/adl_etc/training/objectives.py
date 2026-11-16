"""Training objectives by name, from a config's ``objective:`` section (plan
phase 3).

The :class:`~adl_etc.training.loop.Objective` protocol and the default
:class:`~adl_etc.training.loop.ClassifyObjective` live in ``loop.py``; this
module maps configs onto them and will hold the PAT's own objectives as they
are built (spec 007's SSL in T5, spec 008's prefix training in T6).

``objective: {name: classify, final_only: true}`` is B5 (spec 005): the PAT
trained with plain cross-entropy at each flow's last real packet only. With no
``objective:`` section a run is ``classify`` with the model's own logits, which
is every phase-2 baseline unchanged. ``objective: {name: prefix, ...}`` is spec
008's prefix training (:class:`PrefixObjective`); its label smoothing is the
config's ``train.label_smoothing``, so the design ablation flips one knob.
``ssl`` and ``mpm`` are spec 007's pretraining objectives
(:mod:`adl_etc.training.ssl`), label-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from adl_etc.evaluation.metrics import accuracy, causal_accuracy_curve, macro_f1
from adl_etc.training.datasets import FlowBatches, to_torch
from adl_etc.training.loop import ClassifyObjective, Objective, TrainSettings
from adl_etc.training.pat_losses import npp_loss, prefix_ce, safe_bce, self_distillation
from adl_etc.training.ssl import MPMObjective, SSLObjective

_CLASSIFY_OPTIONS = {"final_only"}


@dataclass
class PrefixObjective:
    """Spec 008's supervised prefix training of the PAT:
    ``L = L_cls + sd_weight·L_sd + safe_weight·L_safe + npp_weight·L_NPP``.

    Early stopping follows val ``AUC_K`` (mean acc@K over K = 1..30), because
    earliness is the objective, not acc@30. Validation also reports macro-F1 and
    accuracy at the last real packet and acc@K at K = 1, 3, 5, 8, all from the
    model the loop hands it (the EMA copy when the run keeps one).
    """

    n_classes: int
    label_smoothing: float = 0.1
    ramp: float = 8.0
    sd_weight: float = 0.2
    sd_temperature: float = 2.0
    safe_weight: float = 1.0
    npp_weight: float = 0.1
    monitor: str = "val_auc_k"
    higher_is_better: bool = True
    uses_labels: bool = True

    def loss(
        self, outputs: Any, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cls = outputs["cls"]
        if cls.shape[-1] != self.n_classes:
            raise ValueError(
                f"the model emits {cls.shape[-1]} logits but the label space has "
                f"{self.n_classes} known classes"
            )
        y, mask = batch["y"], batch["mask"]
        parts = {
            "cls": prefix_ce(cls, y, mask, ramp=self.ramp, smoothing=self.label_smoothing),
            "sd": self_distillation(
                cls, mask, batch["ppi_len"], ramp=self.ramp, temperature=self.sd_temperature
            ),
            "safe": safe_bce(outputs["safe"], cls, y, mask),
            "npp": npp_loss(outputs, batch)[0],
        }
        total = (
            parts["cls"]
            + self.sd_weight * parts["sd"]
            + self.safe_weight * parts["safe"]
            + self.npp_weight * parts["npp"]
        )
        return total, parts

    @torch.no_grad()
    def validate(
        self, model: nn.Module, data: FlowBatches, *, batch_size: int,
        device: torch.device, amp: bool,
    ) -> dict[str, float]:  # fmt: skip
        model.eval()
        preds: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        lengths: list[np.ndarray] = []
        loss_sum, n = 0.0, 0
        for _, np_batch in data.iter_batches(batch_size, known_only=True):
            batch = to_torch(np_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                cls = model(batch)["cls"]
            cls = cls.float()
            loss = prefix_ce(cls, batch["y"], batch["mask"], ramp=self.ramp, smoothing=0.0)
            loss_sum += float(loss) * len(np_batch["y"])
            n += len(np_batch["y"])
            preds.append(cls.argmax(-1).to(torch.int16).cpu().numpy())
            targets.append(np_batch["y"])
            lengths.append(np_batch["ppi_len"])
        p, t, lens = np.concatenate(preds), np.concatenate(targets), np.concatenate(lengths)
        curve = causal_accuracy_curve(p, t, lens)
        final = p[np.arange(len(t)), lens - 1]
        return {
            "val_loss": loss_sum / n,
            "val_acc": accuracy(final, t),
            "val_macro_f1": macro_f1(final, t),
            "val_auc_k": float(curve.mean()),
            **{f"val_acc_k{k}": float(curve[k - 1]) for k in (1, 3, 5, 8)},
        }


def _options(cls: type, name: str, raw: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown option(s) for objective {name!r}: {sorted(unknown)}")
    return raw


_PREFIX_OPTIONS = {
    f.name
    for f in fields(PrefixObjective)
    if f.name not in {"n_classes", "label_smoothing", "monitor", "higher_is_better", "uses_labels"}
}


def build_objective(
    section: Mapping[str, Any] | DictConfig | None,
    *,
    model: nn.Module,
    n_classes: int,
    settings: TrainSettings,
) -> Objective:
    raw: dict[str, Any]
    if section is None:
        raw = {}
    elif isinstance(section, DictConfig):
        container = OmegaConf.to_container(section, resolve=True)
        assert isinstance(container, dict)
        raw = {str(k): v for k, v in container.items()}
    else:
        raw = {str(k): v for k, v in section.items()}
    name = raw.pop("name", "classify")

    if name == "classify":
        _options(ClassifyObjective, name, raw, _CLASSIFY_OPTIONS)
        return ClassifyObjective(
            n_classes,
            settings.label_smoothing,
            key=getattr(model, "logits_key", None),
            final_only=bool(raw.get("final_only", False)),
        )
    if name == "prefix":
        _options(PrefixObjective, name, raw, _PREFIX_OPTIONS)
        return PrefixObjective(n_classes, settings.label_smoothing, **raw)
    if name == "ssl":
        _options(SSLObjective, name, raw, {"npp_weight", "pfc_weight", "temperature"})
        return SSLObjective(**raw)
    if name == "mpm":
        _options(MPMObjective, name, raw, set())
        return MPMObjective()
    raise ValueError(
        f"unknown objective {name!r}; expected one of ['classify', 'prefix', 'ssl', 'mpm']"
    )
