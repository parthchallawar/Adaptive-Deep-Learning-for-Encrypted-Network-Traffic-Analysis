"""Robustness evaluation: the same flows, deliberately damaged (plan phase 3, T8;
spec 007's H5).

A condition is one perturbation from :mod:`adl_etc.data.augment_batch` (the
batched, reference-checked versions of spec 003's augmentations) applied to the
raw PPI of an evaluation set before the model sees it. ``clean`` is the
undamaged set. Everything here is NumPy: the model pass lives in
``evaluation/run.py``'s ``main``, so this module stays importable without torch
(``tests/test_import_boundary.py``).

**Paired by construction.** A condition's randomness comes from
``seeded_generator(seed, "perturb/<name>")`` and nothing else, so every model
evaluated with the same seed on the same (capped) flow set sees exactly the
same damaged flows. Scratch-vs-SSL differences under damage are then model
differences, not draw differences.

**What "after K packets" means under damage.** Metrics use the damaged
``ppi_len``: after a 10% drop, "K = 5" is the fifth packet the observer
actually received. That is the deployment question (what can be decided from
what arrived), not "how many of the original first five survived".

**Known limitation.** PPI keeps only a flow's first 30 payload packets, so a
drop can only remove packets from inside that window; on the wire, packets 31+
would slide in to replace them. Flows longer than 30 packets therefore come out
of a drop shorter than they really would. Small K (the regime this project
cares about) is unaffected unless the flow has fewer than K packets left.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from adl_etc.data import augment_batch as A
from adl_etc.utils.seeding import seeded_generator

#: the parameter each kind takes (``value`` in a config entry)
KINDS = {
    "clean": None,
    "drop": "p",  # each real packet removed with probability p
    "reorder": "q",  # adjacent-pair swap probability
    "ipt_jitter": "s",  # IPT x exp(s * N(0, 1))
    "size_jitter": "sd",  # size + N(0, sd) bytes
}

#: K values copied into the compact summary (the full curves stay in each report)
SUMMARY_KS = (1, 2, 3, 5, 8, 10, 15, 30)


@dataclass(frozen=True)
class Perturbation:
    name: str
    kind: str
    value: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown perturbation kind {self.kind!r}; known: {sorted(KINDS)}")
        if (KINDS[self.kind] is None) != (self.value is None):
            need = "no value" if KINDS[self.kind] is None else f"a value ({KINDS[self.kind]})"
            raise ValueError(f"perturbation {self.name!r} ({self.kind}) takes {need}")
        if self.value is not None and not self.value >= 0:
            raise ValueError(f"perturbation {self.name!r}: value must be >= 0, got {self.value}")
        if self.kind in ("drop", "reorder") and self.value is not None and self.value > 1:
            raise ValueError(f"perturbation {self.name!r}: a probability must be <= 1")
        if not self.name or "/" in self.name or "\\" in self.name:
            raise ValueError(f"perturbation name {self.name!r} must be a plain directory name")


CLEAN = Perturbation("clean", "clean")


def parse_perturbations(entries: Iterable[Mapping[str, Any]]) -> list[Perturbation]:
    """Config entries (``{name, kind, value}``) to perturbations. ``clean`` is
    always first, added if absent: every summary is relative to it."""
    out = []
    for e in entries:
        unknown = set(e) - {"name", "kind", "value"}
        if unknown:
            raise ValueError(f"perturbation entry has unknown keys {sorted(unknown)}: {dict(e)}")
        value = e.get("value")
        out.append(
            Perturbation(str(e["name"]), str(e["kind"]), None if value is None else float(value))
        )
    names = [p.name for p in out]
    if len(set(names)) != len(names):
        raise ValueError(f"perturbation names must be unique, got {names}")
    if "clean" in names:
        if out[names.index("clean")] != CLEAN:
            raise ValueError("the name 'clean' is reserved for the undamaged set")
        out.remove(CLEAN)
    return [CLEAN, *out]


def perturb(
    ppi: np.ndarray, ppi_len: np.ndarray, p: Perturbation, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """The damaged copy of ``(ppi [N, 30, 4], ppi_len [N])``; inputs untouched.
    Deterministic in ``(seed, p.name)``."""
    if p.kind == "clean":
        return ppi, ppi_len
    rng = seeded_generator(seed, f"perturb/{p.name}")
    assert p.value is not None
    if p.kind == "drop":
        return A.drop(ppi, ppi_len, p.value, rng)
    if p.kind == "reorder":
        return A.reorder(ppi, ppi_len, p.value, rng), ppi_len.copy()
    if p.kind == "ipt_jitter":
        return A.ipt_jitter(ppi, ppi_len, p.value, rng), ppi_len.copy()
    if p.kind == "size_jitter":
        return A.size_jitter(ppi, ppi_len, p.value, rng), ppi_len.copy()
    raise AssertionError(p.kind)  # unreachable: __post_init__ checks the kind


def _pick(curve: Mapping[Any, float] | None, ks: Iterable[int]) -> dict[str, float] | None:
    if curve is None:
        return None
    by_int = {int(k): float(v) for k, v in curve.items()}
    return {str(k): by_int[k] for k in ks if k in by_int}


def summarize(
    reports: Mapping[str, Mapping[str, Any]],
    perturbations: Iterable[Perturbation],
    *,
    ks: Iterable[int] = SUMMARY_KS,
) -> dict[str, Any]:
    """One compact table over conditions, from each condition's
    ``Report.to_dict()``: acc@K at a few K, AUC_K, macro-F1, k95, open-set
    AUROC@K, and each one's change from ``clean`` (negative = worse)."""
    ks = tuple(ks)
    perts = {p.name: p for p in perturbations}
    if "clean" not in reports:
        raise ValueError("summarize needs the 'clean' report")
    if set(reports) != set(perts):
        raise ValueError(f"reports {sorted(reports)} do not match perturbations {sorted(perts)}")

    def row(r: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "acc_at_k": _pick(r["acc_at_k"], ks),
            "auc_k": float(r["auc_k"]),
            "macro_f1": float(r["macro_f1"]),
            "k95": int(r["k95"]),
            "auroc_at_k": _pick(r.get("auroc_at_k"), ks),
        }

    clean = row(reports["clean"])
    conditions: dict[str, Any] = {}
    for name in perts:  # config order, clean first
        cur = row(reports[name])
        delta: dict[str, Any] = {
            "acc_at_k": {k: cur["acc_at_k"][k] - clean["acc_at_k"][k] for k in clean["acc_at_k"]},
            "auc_k": cur["auc_k"] - clean["auc_k"],
            "macro_f1": cur["macro_f1"] - clean["macro_f1"],
        }
        if cur["auroc_at_k"] is not None and clean["auroc_at_k"] is not None:
            delta["auroc_at_k"] = {
                k: cur["auroc_at_k"][k] - clean["auroc_at_k"][k] for k in clean["auroc_at_k"]
            }
        conditions[name] = {
            "kind": perts[name].kind,
            "value": perts[name].value,
            **cur,
            "delta_vs_clean": delta,
        }
    first = reports["clean"]
    return {
        "split": first["split"],
        "n_flows": int(first["n_flows"]),
        "n_known": int(first["n_known"]),
        "n_unknown": int(first["n_unknown"]),
        "ks": list(ks),
        "conditions": conditions,
        "report_hashes": {name: reports[name].get("hash", "") for name in perts},
    }
