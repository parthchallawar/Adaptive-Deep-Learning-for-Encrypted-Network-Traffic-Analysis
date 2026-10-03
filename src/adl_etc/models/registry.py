"""One place that turns a config's ``model:`` section into a model: the PAT
(spec 006) by name ``pat``, the same PAT with SSL views for pretraining (spec
007) by name ``pat_pretrain``, every torch baseline (spec 005) otherwise."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch import nn

from adl_etc.models import baselines


def build_model(model_cfg: Mapping[str, Any] | DictConfig, n_classes: int) -> nn.Module:
    """``{name: pat|cnn|gru|lstm, ...constructor args}`` to a model. An unknown
    name or argument raises, so a typo cannot fall back to a default."""
    raw = (
        OmegaConf.to_container(model_cfg, resolve=True)
        if isinstance(model_cfg, DictConfig)
        else dict(model_cfg)
    )
    assert isinstance(raw, dict)
    spec: dict[str, Any] = {str(k): v for k, v in raw.items()}
    name = spec.get("name")
    if name not in ("pat", "pat_pretrain"):
        return baselines.build_model(spec, n_classes)

    from adl_etc.models.pat import PAT

    cls: type[PAT] = PAT
    if name == "pat_pretrain":
        from adl_etc.training.ssl import PretrainPAT

        cls = PretrainPAT
    kwargs = {k: v for k, v in spec.items() if k != "name"}
    try:
        return cls(n_classes, **kwargs)
    except TypeError as e:
        raise ValueError(f"bad arguments for model {name!r}: {e}") from e
