"""The label space: which classes a model knows, and which column is which.

Shard ``label`` values are the *dataset's* ids (0..179 for D1) and include
classes held out as unknown. A model emits ``C`` contiguous logits. Without one
authoritative mapping between the two, every saved ``logits[N, K, C]`` array is
uninterpretable and two runs can disagree silently about what column 7 means.

A :class:`LabelSpace` is that mapping. It is saved with every run, its hash is
stored in every checkpoint, and evaluation refuses logits whose hash differs
from the checkpoint's.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from adl_etc.utils.atomic import atomic_write_json
from adl_etc.utils.provenance import stable_hash

#: The shard schema's sentinel for "unknown class" (``tensors.ARRAY_SPEC``).
UNKNOWN_SENTINEL = -1


class LabelSpaceError(ValueError):
    """A label that does not belong to the label space, or a corrupt saved one."""


@dataclass(frozen=True)
class LabelSpace:
    """``known`` are the dataset ids the model is trained on, sorted; model
    index ``i`` is dataset id ``known[i]``. ``unknown`` are ids held out for
    open-set evaluation: they appear only in val/test and map to ``-1``.
    """

    known: tuple[int, ...]
    unknown: tuple[int, ...] = ()
    names: Mapping[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.known:
            raise LabelSpaceError("a label space needs at least one known class")
        if list(self.known) != sorted(set(self.known)):
            raise LabelSpaceError("`known` must be sorted and unique")
        if list(self.unknown) != sorted(set(self.unknown)):
            raise LabelSpaceError("`unknown` must be sorted and unique")
        if min((*self.known, *self.unknown)) < 0:
            raise LabelSpaceError(
                f"class ids must be non-negative ({UNKNOWN_SENTINEL} is the unknown sentinel)"
            )
        overlap = set(self.known) & set(self.unknown)
        if overlap:
            raise LabelSpaceError(f"classes cannot be both known and unknown: {sorted(overlap)}")
        stray = set(self.names) - set(self.known) - set(self.unknown)
        if stray:
            raise LabelSpaceError(f"names given for ids not in the label space: {sorted(stray)}")

    @classmethod
    def create(
        cls,
        known: Iterable[int],
        unknown: Iterable[int] = (),
        names: Mapping[int, str] | None = None,
    ) -> LabelSpace:
        """Sorts and de-duplicates, so callers need not."""
        return cls(
            known=tuple(sorted({int(k) for k in known})),
            unknown=tuple(sorted({int(u) for u in unknown})),
            names=dict(names or {}),
        )

    @classmethod
    def from_label_map(
        cls, label_map: Mapping[str, int], unknown: Iterable[int] = ()
    ) -> LabelSpace:
        """From a shard set's ``meta["label_map"]`` (``name -> dataset id``) and
        the ids to hold out. Every id not held out is known."""
        held_out = {int(u) for u in unknown}
        by_id = {int(i): str(n) for n, i in label_map.items()}
        missing = held_out - set(by_id)
        if missing:
            raise LabelSpaceError(f"unknown ids not in the label map: {sorted(missing)}")
        return cls.create(
            known=set(by_id) - held_out,
            unknown=held_out,
            names={i: n for i, n in by_id.items()},
        )

    @property
    def n_classes(self) -> int:
        return len(self.known)

    @property
    def class_names(self) -> list[str]:
        """Names in *model* order (index ``i`` first), falling back to the id."""
        return [self.names.get(i, str(i)) for i in self.known]

    def to_model(self, y: np.ndarray) -> np.ndarray:
        """Dataset ids -> model indices ``0..C-1``. Held-out ids and the ``-1``
        sentinel map to ``-1``.

        An id that is in neither ``known`` nor ``unknown`` raises rather than
        becoming ``-1``: silently treating an unrecognised class as "unknown"
        would hide a label map from the wrong dataset.
        """
        y = np.asarray(y)
        known = np.asarray(self.known, dtype=np.int64)
        pos = np.searchsorted(known, y)
        pos_clipped = np.minimum(pos, len(known) - 1)
        is_known = known[pos_clipped] == y
        out = np.where(is_known, pos_clipped, UNKNOWN_SENTINEL).astype(np.int64)

        held_out = np.isin(y, np.asarray(self.unknown, dtype=np.int64)) | (y == UNKNOWN_SENTINEL)
        stray = ~is_known & ~held_out
        if stray.any():
            bad = sorted({int(v) for v in np.unique(y[stray])})
            raise LabelSpaceError(
                f"label id(s) {bad[:10]} are in neither the known nor the unknown classes "
                "of this label space (a label map from a different dataset?)"
            )
        return out

    def to_dataset(self, index: np.ndarray) -> np.ndarray:
        """Model indices -> dataset ids. ``-1`` stays ``-1``."""
        index = np.asarray(index)
        known = np.asarray(self.known, dtype=np.int64)
        if index.size and (index.max() >= len(known) or index.min() < UNKNOWN_SENTINEL):
            raise LabelSpaceError(f"model index out of range 0..{len(known) - 1} (or -1)")
        return np.where(index >= 0, known[np.maximum(index, 0)], UNKNOWN_SENTINEL)

    # -- identity ---------------------------------------------------------------

    def _payload(self) -> dict[str, object]:
        return {
            "known": list(self.known),
            "unknown": list(self.unknown),
            "names": {str(k): v for k, v in sorted(self.names.items())},
        }

    @property
    def hash(self) -> str:
        return stable_hash(self._payload())

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, {**self._payload(), "hash": self.hash})

    @classmethod
    def load(cls, path: str | Path) -> LabelSpace:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            space = cls.create(
                raw["known"], raw["unknown"], {int(k): v for k, v in raw["names"].items()}
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            raise LabelSpaceError(f"unreadable label space {path}: {e}") from e
        if raw.get("hash") != space.hash:
            raise LabelSpaceError(f"{path}: stored hash does not match its contents (edited?)")
        return space
