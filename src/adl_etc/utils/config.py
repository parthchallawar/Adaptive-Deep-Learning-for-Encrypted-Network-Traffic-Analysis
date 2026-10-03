"""Experiment configuration: composition, overrides, hashing (spec 014).

Every run is described by one YAML file plus a git commit. This module turns
that file into a frozen, fully-resolved config and a stable hash of it.

OmegaConf alone does not compose ``defaults:`` lists (that is Hydra, which this
project does not depend on), so the small composer here does it: each entry of
a file's ``defaults`` list is a path **relative to the file that names it**,
loaded recursively, later entries overriding earlier ones, and the file's own
keys overriding all of its defaults. ``defaults`` itself never appears in the
resolved config.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError, OmegaConfBaseException

from adl_etc.utils.provenance import stable_hash

DEFAULTS_KEY = "defaults"


class ConfigError(ValueError):
    """A config file or override that cannot be composed as written."""


def _load_one(path: Path, chain: tuple[Path, ...]) -> DictConfig:
    path = path.resolve()
    if path in chain:
        cycle = " -> ".join(str(p) for p in (*chain, path))
        raise ConfigError(f"circular `defaults`: {cycle}")
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")

    raw = OmegaConf.load(path)
    if not isinstance(raw, DictConfig):
        raise ConfigError(f"{path}: expected a mapping at the top level")

    defaults = raw.pop(DEFAULTS_KEY, None)
    if defaults is None:
        entries: list[str] = []
    elif isinstance(defaults, str):
        # A bare string would otherwise be iterated character by character.
        raise ConfigError(f"{path}: `{DEFAULTS_KEY}` must be a list of paths, not a string")
    else:
        entries = [str(e) for e in defaults]

    merged: DictConfig = OmegaConf.create({})
    for entry in entries:
        merged = OmegaConf.merge(merged, _load_one(path.parent / entry, (*chain, path)))  # type: ignore[assignment]
    return OmegaConf.merge(merged, raw)  # type: ignore[return-value]


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> DictConfig:
    """Compose ``path``'s ``defaults:`` list, apply ``overrides`` (dotted
    ``key=value`` strings, as on a command line), resolve interpolations and
    return a read-only config.

    An override naming a key the composed config does not already have raises
    :class:`ConfigError`. A typo such as ``optim.learning_rate=0.1`` against a
    config that has ``optim.lr`` must fail loudly, not add a new key that
    nothing reads and let the run proceed with the old value.
    """
    cfg = _load_one(Path(path), ())
    OmegaConf.set_struct(cfg, True)
    try:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))  # type: ignore[assignment]
        OmegaConf.resolve(cfg)
    except ConfigKeyError as e:
        raise ConfigError(f"override names a key that is not in the config: {e}") from e
    except OmegaConfBaseException as e:
        raise ConfigError(f"could not compose {path} with overrides {list(overrides)}: {e}") from e
    OmegaConf.set_readonly(cfg, True)
    return cfg


def to_container(cfg: DictConfig) -> dict[str, Any]:
    """The resolved config as plain, JSON-serialisable Python."""
    out = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(out, dict):
        raise ConfigError("config root is not a mapping")
    return {str(k): v for k, v in out.items()}


def config_hash(cfg: DictConfig) -> str:
    """SHA-256 of the *resolved* config.

    Invariant to key order, comments and whitespace (it hashes values, not
    text), so two files that mean the same thing hash the same; the results
    table uses this to catch one run name carrying two different configs.
    """
    return stable_hash(to_container(cfg))


def flatten(cfg: DictConfig) -> dict[str, Any]:
    """Dotted-key view of the resolved config, for logging as run params."""
    flat: dict[str, Any] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}.{key}" if prefix else str(key), value)
        else:
            flat[prefix] = node

    walk("", to_container(cfg))
    return flat


def save_resolved(cfg: DictConfig, dest: str | Path) -> None:
    """Write ``config_resolved.yaml``-style output: every interpolation
    substituted, so the file stands alone without its ``defaults`` chain."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, dest, resolve=True)
