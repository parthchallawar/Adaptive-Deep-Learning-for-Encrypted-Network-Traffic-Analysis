"""Split loading and leakage assertions (spec 004, plan T6).

Loads a split YAML (``configs/splits/*.yaml``) into concrete shard paths and
opened :class:`~adl_etc.data.tensors.ShardSet` objects, and enforces spec
004's four leakage rules *at load time*, not merely in a test file -- "a
leakage rule that only exists in a test file is a rule that gets skipped on
the day it matters" (plan T6).

**Split YAML schema**::

    dataset: cesnet-tls-year22
    temporal: true                # whether rule 4 (ts ordering) applies
    splits:
      train:
        periods: [WEEK-2022-11, WEEK-2022-12, ...]
        standardizer: null        # path to a saved Standardizer, or omit
      val:
        periods: [WEEK-2022-27, ...]
        standardizer: results/standardizer.json
      test:
        periods: [all]
        session_ids: [0, 3, 7]    # optional: restrict one shared period by
                                  # session_id (D3's grouped-CV fold design)

``session_ids`` is only ever compared *within* one period's own shard set,
never across periods: each period may come from an independent export run
with its own locally-scoped session_id numbering (true of D1/D2's per-week
CESNET exports, spec 001/T5 -- session_id there is only unique within one
`export_dataset()` call), so a bare cross-period session_id comparison would
raise false leakage errors for two unrelated files that happen to share a
number. Comparing (period, session_id) instead avoids that trap while still
catching D3's real grouped-CV leak.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adl_etc.data.tensors import ShardSet


class LeakageError(Exception):
    """A split violates one of spec 004's four leakage rules."""


def load_split_spec(path: str | Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(OmegaConf.load(path))
    if not isinstance(raw, dict) or "dataset" not in raw or "splits" not in raw:
        raise ValueError(f"{path}: expected a mapping with 'dataset' and 'splits' keys")
    return {str(k): v for k, v in raw.items()}


# --- rule 1: no period, and no (period, session_id), in two splits ----------


def assert_no_session_overlap(spec: dict[str, Any]) -> None:
    """A period claimed *whole* (no ``session_ids`` filter) can never be
    shared with another split. Two splits *can* legitimately share one
    period via disjoint ``session_ids`` (D3's grouped-CV design) -- that
    case is only a conflict if the specific session_ids themselves overlap,
    checked separately below."""
    period_claims: dict[str, list[tuple[str, bool]]] = {}
    session_owner: dict[tuple[str, int], str] = {}
    for split_name, cfg in spec["splits"].items():
        session_ids = cfg.get("session_ids")
        whole = session_ids is None
        for period in cfg["periods"]:
            for other_name, other_whole in period_claims.get(period, []):
                if other_name != split_name and (whole or other_whole):
                    raise LeakageError(
                        f"period {period!r} is assigned to both {other_name!r} and {split_name!r}"
                    )
            period_claims.setdefault(period, []).append((split_name, whole))

            if session_ids is None:
                continue
            for sid in session_ids:
                key = (period, int(sid))
                sid_owner = session_owner.get(key)
                if sid_owner is not None and sid_owner != split_name:
                    raise LeakageError(
                        f"session_id {sid} in period {period!r} is assigned to both "
                        f"{sid_owner!r} and {split_name!r}"
                    )
                session_owner[key] = split_name


# --- rule 3: a val/test split's Standardizer must be the one fit on train --


def assert_standardizer_hash_consistent(spec: dict[str, Any]) -> None:
    from adl_etc.data.features import Standardizer

    hashes: dict[str, str] = {}
    for split_name, cfg in spec["splits"].items():
        path = cfg.get("standardizer")
        if path is None:
            continue
        hashes[split_name] = Standardizer.load(Path(path)).hash
    if len(set(hashes.values())) > 1:
        raise LeakageError(f"splits reference standardizers with different hashes: {hashes}")


# --- loaded split, rules 2 and 4 (need opened shard data) -------------------


@dataclass
class LoadedSplit:
    dataset: str
    paths: dict[str, list[Path]]
    shard_sets: dict[str, list[ShardSet]]
    masks: dict[str, list[np.ndarray | None]]
    spec: dict[str, Any] = field(repr=False)
    boundary_trimmed: dict[str, int] = field(default_factory=dict)
    """Flows :func:`trim_period_boundaries` dropped from each split, keyed by split
    name (0 for a split it did not touch, e.g. every split when ``temporal`` is
    false). Kept for reports rather than discarded, even though it is always tiny."""

    def close(self) -> None:
        for shard_sets in self.shard_sets.values():
            for ss in shard_sets:
                ss.close()

    def __enter__(self) -> LoadedSplit:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _masked_column(ss: ShardSet, mask: np.ndarray | None, name: str) -> np.ndarray:
    col = ss.column(name)
    return col if mask is None else col[mask]


def assert_no_unknown_in_train(
    loaded: LoadedSplit, *, train_splits: tuple[str, ...] = ("train",)
) -> None:
    """No flow with the shard schema's unknown sentinel (``label == -1``,
    `tensors.ARRAY_SPEC`) may appear in a training split (spec 004 rule 2)."""
    for name in train_splits:
        for ss, mask in zip(
            loaded.shard_sets.get(name, []), loaded.masks.get(name, []), strict=True
        ):
            labels = _masked_column(ss, mask, "label")
            if labels.size and bool(np.any(labels == -1)):
                raise LeakageError(f"split {name!r} contains unknown-labelled flows (label=-1)")


def trim_period_boundaries(
    loaded: LoadedSplit, *, max_trim_fraction: float = 0.001
) -> dict[str, int]:
    """Drops the handful of flows a day's own file misfiles across a period
    boundary, before :func:`assert_temporal_order` gets to see them.

    The dataset card's "``TIME_FIRST`` is UTC; the mirror's own file grouping is
    not" finding (spec 001/T5b) is about day files, but the same effect lands on
    *period* boundaries too: a period's first day can carry a few rows whose real
    ``ts`` (its true UTC start time) falls before the period's own start, because
    the mirror groups that day's file by *local* midnight. Measured on the real
    weeks-11-52 D1 export (2026-09-22): 30 of 197,505 val flows (0.015%) and a
    similar handful at the val/test_id and test_id/test_drift boundaries.

    Only runs when the split YAML sets ``temporal: true``; D3's grouped CV has no
    meaningful time order. Processes splits in the YAML's own declared order
    (train, then val, then test_id, ...), which is why that order must be
    chronological for a temporal split -- exactly what spec 004 already asks for.
    Each split's own floor is the maximum ``ts`` actually *kept* in every split
    named before it, so a split's stray rows never leak into the next split's
    floor. Mutates ``loaded.masks`` in place (ANDed with the leakage-rule 1
    ``session_ids`` mask, if any) and returns what it dropped.

    ``max_trim_fraction`` is the tripwire that keeps this a fix for a known,
    tiny artifact rather than a general licence to drop overlapping flows: more
    than that fraction of a split raises :class:`LeakageError` instead of being
    silently absorbed, because a violation that large is not the boundary
    artifact -- it is a real ordering bug that deserves to fail loudly.
    """
    dropped: dict[str, int] = {}
    if not loaded.spec.get("temporal", False):
        return dropped
    floor = -1  # ts is milliseconds since the epoch: always >= 0 for real data
    for name in loaded.spec["splits"]:
        shard_sets = loaded.shard_sets.get(name, [])
        existing = loaded.masks.get(name, [None] * len(shard_sets))
        new_masks: list[np.ndarray | None] = []
        n_before = n_after = 0
        for ss, mask in zip(shard_sets, existing, strict=True):
            ts = ss.column("ts")
            base = mask if mask is not None else np.ones(len(ts), dtype=bool)
            n_before += int(base.sum())
            keep = base & (ts > floor)
            n_after += int(keep.sum())
            new_masks.append(keep)
        n_dropped = n_before - n_after
        if n_before and n_dropped / n_before > max_trim_fraction:
            raise LeakageError(
                f"split {name!r}: {n_dropped}/{n_before} flows have ts at or before "
                f"the previous split's own ts -- {100 * n_dropped / n_before:.2f}% is too "
                "many to be the known day-boundary artifact (cap "
                f"{100 * max_trim_fraction:.3f}%); this looks like a real ordering bug"
            )
        dropped[name] = n_dropped
        loaded.masks[name] = new_masks
        kept_ts = [_masked_column(ss, m, "ts") for ss, m in zip(shard_sets, new_masks, strict=True)]
        kept_ts = [t for t in kept_ts if t.size]
        if kept_ts:
            floor = max(floor, int(max(t.max() for t in kept_ts)))
    loaded.boundary_trimmed = dropped
    return dropped


def assert_temporal_order(
    loaded: LoadedSplit, *, train_split: str = "train"
) -> None:
    """Train ``ts`` strictly precedes every other split's ``ts`` (spec 004
    rule 4). Only enforced when the split YAML sets ``temporal: true`` --
    D3's grouped CV has no meaningful time order."""
    if not loaded.spec.get("temporal", False):
        return
    train_sets = loaded.shard_sets.get(train_split, [])
    if not train_sets:
        return
    train_ts = [
        _masked_column(ss, mask, "ts")
        for ss, mask in zip(train_sets, loaded.masks.get(train_split, []), strict=True)
    ]
    train_ts = [t for t in train_ts if t.size]
    if not train_ts:
        return
    max_train_ts = int(max(t.max() for t in train_ts))

    for name, shard_sets in loaded.shard_sets.items():
        if name == train_split:
            continue
        for ss, mask in zip(shard_sets, loaded.masks.get(name, []), strict=True):
            ts = _masked_column(ss, mask, "ts")
            if ts.size and int(ts.min()) <= max_train_ts:
                raise LeakageError(
                    f"split {name!r} has a flow with ts <= max(train ts); "
                    f"train must strictly precede every other split"
                )


# --- top-level entry point ---------------------------------------------------


def load_split(
    path: str | Path,
    *,
    root: str | Path = "data/processed",
    mmap: bool = True,
) -> LoadedSplit:
    """Loads a split YAML into opened shard sets, asserting all four leakage
    rules before returning -- never after. A violated assertion raises
    :class:`LeakageError` and nothing is left open on that failure path.

    For a ``temporal: true`` split, :func:`trim_period_boundaries` runs first and
    drops the tiny, known day-boundary artifact before rule 4 gets a look; a real
    ordering bug still raises (see that function)."""
    spec = load_split_spec(path)
    assert_no_session_overlap(spec)
    assert_standardizer_hash_consistent(spec)

    dataset = spec["dataset"]
    root = Path(root)
    paths: dict[str, list[Path]] = {}
    shard_sets: dict[str, list[ShardSet]] = {}
    masks: dict[str, list[np.ndarray | None]] = {}

    try:
        for split_name, cfg in spec["splits"].items():
            split_paths = [root / dataset / period for period in cfg["periods"]]
            opened = [ShardSet.open(p, mmap=mmap) for p in split_paths]
            session_ids = cfg.get("session_ids")
            split_masks: list[np.ndarray | None] = (
                [None] * len(opened)
                if session_ids is None
                else [np.isin(ss.column("session_id"), session_ids) for ss in opened]
            )
            paths[split_name] = split_paths
            shard_sets[split_name] = opened
            masks[split_name] = split_masks
    except BaseException:
        for opened in shard_sets.values():
            for ss in opened:
                ss.close()
        raise

    loaded = LoadedSplit(
        dataset=dataset, paths=paths, shard_sets=shard_sets, masks=masks, spec=spec
    )
    try:
        trim_period_boundaries(loaded)
        assert_no_unknown_in_train(loaded)
        assert_temporal_order(loaded)
    except BaseException:
        loaded.close()
        raise
    return loaded
