"""Bounded random search (spec 005: 20 trials per baseline, tuned on val).

Only the sampler lives here: it draws trials and renders them as config
overrides, and the caller runs each one through the normal ``fit`` path. Each
trial has its own random stream, so trial ``i`` is the same whether the search
runs 5 trials or 20, and adding a parameter to the space changes no earlier
draw's other parameters.

A search space maps a dotted config key to one distribution::

    train.lr:          {loguniform: [0.0005, 0.01]}
    model.dropout:     {choice: [0.0, 0.1, 0.2]}
    model.hidden:      {int: [128, 512]}          # inclusive
    train.label_smoothing: {uniform: [0.0, 0.2]}
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from adl_etc.utils.seeding import seeded_generator

DEFAULT_TRIALS = 20
_KINDS = ("loguniform", "uniform", "int", "choice")


class SearchSpaceError(ValueError):
    pass


def _validate(space: Mapping[str, Mapping[str, Any]]) -> None:
    if not space:
        raise SearchSpaceError("the search space is empty")
    for key, dist in space.items():
        kinds = [k for k in dist if k in _KINDS]
        if len(dist) != 1 or len(kinds) != 1:
            raise SearchSpaceError(f"{key}: give exactly one of {_KINDS}, got {dict(dist)}")
        kind, arg = kinds[0], dist[kinds[0]]
        if kind == "choice":
            if not isinstance(arg, Sequence) or isinstance(arg, str) or len(arg) == 0:
                raise SearchSpaceError(f"{key}: choice needs a non-empty list")
            continue
        if not isinstance(arg, Sequence) or len(arg) != 2 or not arg[0] < arg[1]:
            raise SearchSpaceError(f"{key}: {kind} needs [low, high] with low < high, got {arg}")
        if kind == "loguniform" and arg[0] <= 0:
            raise SearchSpaceError(f"{key}: loguniform needs low > 0, got {arg[0]}")
        if kind == "int" and not all(float(v).is_integer() for v in arg):
            raise SearchSpaceError(f"{key}: int bounds must be integers, got {arg}")


def sample_trials(
    space: Mapping[str, Mapping[str, Any]], n_trials: int = DEFAULT_TRIALS, seed: int = 0
) -> list[dict[str, Any]]:
    """``n_trials`` reproducible draws from ``space``, each ``{dotted_key: value}``."""
    _validate(space)
    if n_trials < 1:
        raise SearchSpaceError(f"n_trials must be >= 1, got {n_trials}")
    trials = []
    for i in range(n_trials):
        rng = seeded_generator(seed, f"search-trial-{i}")
        trial: dict[str, Any] = {}
        for key in sorted(space):  # fixed order, so the draw does not depend on dict order
            ((kind, arg),) = space[key].items()
            if kind == "loguniform":
                trial[key] = float(math.exp(rng.uniform(math.log(arg[0]), math.log(arg[1]))))
            elif kind == "uniform":
                trial[key] = float(rng.uniform(arg[0], arg[1]))
            elif kind == "int":
                trial[key] = int(rng.integers(int(arg[0]), int(arg[1]) + 1))
            else:
                trial[key] = arg[int(rng.integers(0, len(arg)))]
        trials.append(trial)
    return trials


def to_overrides(trial: Mapping[str, Any]) -> list[str]:
    """A trial as ``key=value`` strings for ``utils.config.load_config``.
    ``repr`` keeps floats exact (a rounded learning rate is a different run)."""
    return [f"{k}={v!r}" if isinstance(v, float) else f"{k}={v}" for k, v in trial.items()]
