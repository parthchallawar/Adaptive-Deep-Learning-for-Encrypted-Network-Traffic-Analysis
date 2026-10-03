"""Baseline models (spec 005): B2 CNN, B3 GRU/LSTM. (B1 XGBoost was removed
2026-10-03; see spec 005.)

``build_model`` turns a config's ``model:`` section into a torch baseline.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch import nn

from adl_etc.models.baselines.cnn import CNNBaseline
from adl_etc.models.baselines.rnn import RNNBaseline

TORCH_MODELS = ("cnn", "gru", "lstm")


def build_model(model_cfg: Mapping[str, Any] | DictConfig, n_classes: int) -> nn.Module:
    """``{name: cnn|gru|lstm, ...constructor args}`` to a model. An unknown name
    or constructor argument raises, so a typo cannot silently fall back to a
    default."""
    raw = (
        OmegaConf.to_container(model_cfg, resolve=True)
        if isinstance(model_cfg, DictConfig)
        else dict(model_cfg)
    )
    assert isinstance(raw, dict)
    kwargs: dict[str, Any] = {str(k): v for k, v in raw.items()}
    name = kwargs.pop("name", None)
    try:
        if name == "cnn":
            return CNNBaseline(n_classes, **kwargs)
        if name in ("gru", "lstm"):
            return RNNBaseline(n_classes, cell=name, **kwargs)
    except TypeError as e:
        raise ValueError(f"bad arguments for model {name!r}: {e}") from e
    raise ValueError(f"unknown model name {name!r}; expected one of {list(TORCH_MODELS)}")


__all__ = ["TORCH_MODELS", "CNNBaseline", "RNNBaseline", "build_model"]
