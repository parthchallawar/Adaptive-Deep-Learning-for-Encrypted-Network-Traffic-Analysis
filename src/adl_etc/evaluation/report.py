"""`Report`: spec 004's `evaluate(model_or_policy, split) -> Report` output, minus
the model (plan T7). A pure function of a :class:`~adl_etc.evaluation.dense_logits.DenseLogits`
plus labels/ppi_len -- so a report can be rebuilt from saved logits (`EvalArrays`)
without a model, and re-run to prove determinism.

Deliberately smaller than spec 004's full field list for now: drift-slope and
budget-tracking metrics are specs 011/012/013's own build steps (phase 2's own
scope table), so this report has none of them. What it does have: the
accuracy-vs-K curve, k95/auc_k, a reliability diagram, P-ECHO's Pareto front,
and (given a fit calibration) P-CAPE's single operating point, plus an
open-set AUROC-vs-K curve when the split actually has unknown-labelled flows.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adl_etc.evaluation import metrics as M
from adl_etc.evaluation import policies as PL
from adl_etc.evaluation.dense_logits import DenseLogits
from adl_etc.utils.atomic import atomic_write_json
from adl_etc.utils.provenance import now_iso, stable_hash

DEFAULT_ECHO_TAUS: tuple[float, ...] = tuple(round(t, 2) for t in np.linspace(0.0, 1.0, 21))
UNKNOWN_LABEL = -1


@dataclass(frozen=True)
class CapeConfig:
    """A fit P-CAPE calibration, ready to evaluate (:func:`policies.fit_cape_thresholds`)."""

    tau_c: Sequence[float]
    theta: float
    energy_k: int = 10


def _subset(dense: DenseLogits, mask: np.ndarray) -> DenseLogits:
    """The same array restricted to the flows ``mask`` selects; ``evaluated_k``
    and ``indexing`` never depend on N, so they carry over unchanged."""
    return DenseLogits(dense.logits[mask], dense.evaluated_k, dense.indexing)


def _acc_curve(
    dense: DenseLogits, labels: np.ndarray, ppi_len: np.ndarray, ks: list[int]
) -> dict[int, float]:
    return {k: M.acc_at_k(dense, labels, ppi_len, k) for k in ks}


def _reliability(
    dense: DenseLogits, labels: np.ndarray, ppi_len: np.ndarray, k: int, n_bins: int
) -> dict[str, Any]:
    probs = M.softmax(dense.at(k, ppi_len))
    confidence = probs.max(axis=-1)
    correct = probs.argmax(axis=-1) == labels
    centers, bin_acc, bin_conf = M.reliability_curve(confidence, correct, n_bins=n_bins)
    return {
        "k": k,
        "n_bins": n_bins,
        "bin_centers": centers.tolist(),
        "bin_accuracy": [None if np.isnan(v) else float(v) for v in bin_acc],
        "bin_confidence": [None if np.isnan(v) else float(v) for v in bin_conf],
        "ece": M.ece(confidence, correct, n_bins=n_bins),
    }


def _auroc_curve(
    dense: DenseLogits, is_unknown: np.ndarray, ppi_len: np.ndarray, ks: list[int]
) -> dict[int, float]:
    out = {}
    for k in ks:
        score = PL.energy_score(dense.at(k, ppi_len))
        out[k] = M.auroc(score, is_unknown)
    return out


@dataclass(frozen=True)
class Report:
    """JSON-serialisable; ``hash`` covers everything except ``created_at``, so
    two reports built from identical inputs hash identically regardless of
    when they were built (spec 004's determinism requirement)."""

    split: str
    n_classes: int
    n_flows: int
    n_known: int
    n_unknown: int
    ks: list[int]
    acc_at_k: dict[int, float]
    macro_f1: float
    balanced_accuracy: float
    k95: int
    auc_k: float
    reliability: dict[str, Any]
    echo: dict[str, Any]
    cape: dict[str, Any] | None
    auroc_at_k: dict[int, float] | None
    meta: dict[str, Any]
    hash: str = field(default="")
    created_at: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "n_classes": self.n_classes,
            "n_flows": self.n_flows,
            "n_known": self.n_known,
            "n_unknown": self.n_unknown,
            "ks": self.ks,
            "acc_at_k": {str(k): v for k, v in self.acc_at_k.items()},
            "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "k95": self.k95,
            "auc_k": self.auc_k,
            "reliability": self.reliability,
            "echo": self.echo,
            "cape": self.cape,
            "auroc_at_k": (
                None
                if self.auroc_at_k is None
                else {str(k): v for k, v in self.auroc_at_k.items()}
            ),
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> None:
        payload = self.to_dict()
        payload["hash"] = self.hash
        payload["created_at"] = self.created_at
        atomic_write_json(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> Report:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            split=raw["split"],
            n_classes=raw["n_classes"],
            n_flows=raw["n_flows"],
            n_known=raw["n_known"],
            n_unknown=raw["n_unknown"],
            ks=raw["ks"],
            acc_at_k={int(k): v for k, v in raw["acc_at_k"].items()},
            macro_f1=raw["macro_f1"],
            balanced_accuracy=raw["balanced_accuracy"],
            k95=raw["k95"],
            auc_k=raw["auc_k"],
            reliability=raw["reliability"],
            echo=raw["echo"],
            cape=raw["cape"],
            auroc_at_k=None
            if raw["auroc_at_k"] is None
            else {int(k): v for k, v in raw["auroc_at_k"].items()},
            meta=raw["meta"],
            hash=raw["hash"],
            created_at=raw["created_at"],
        )


def build_report(
    dense: DenseLogits,
    labels: np.ndarray,
    ppi_len: np.ndarray,
    *,
    n_classes: int,
    split: str = "",
    unknown_label: int = UNKNOWN_LABEL,
    ks: Sequence[int] | None = None,
    echo_taus: Sequence[float] = DEFAULT_ECHO_TAUS,
    cape: CapeConfig | None = None,
    ece_bins: int = 15,
    seed: int = 0,
) -> Report:
    """Everything above assembled into one :class:`Report`.

    ``labels``/``ppi_len`` are model-space (``unknown_label`` for a held-out
    class, spec 004). The accuracy curve, F1, balanced accuracy, k95, auc_k,
    the reliability diagram and P-ECHO's Pareto front are computed on known
    flows only -- none of those metrics has a defined answer for a class the
    model was never trained on. The open-set AUROC-vs-K curve (only emitted
    when the split actually holds unknown-labelled flows) and P-CAPE's own
    summary (only emitted when ``cape`` is given) use the full set, since
    handling unknown flows is their entire point.
    """
    ks_list = sorted(ks) if ks is not None else sorted(dense.evaluated_k)
    labels = np.asarray(labels)
    known_mask = labels != unknown_label
    n_flows, n_known = len(labels), int(known_mask.sum())
    n_unknown = n_flows - n_known

    dense_known = _subset(dense, known_mask)
    labels_known, ppi_len_known = labels[known_mask], np.asarray(ppi_len)[known_mask]

    acc_curve = _acc_curve(dense_known, labels_known, ppi_len_known, ks_list)
    k_final = ks_list[-1]
    preds_final = M.predictions_at_k(dense_known, ppi_len_known, k_final)
    macro_f1 = M.macro_f1(preds_final, labels_known)
    balanced_acc = M.balanced_accuracy(preds_final, labels_known)
    k95_value = M.k95(dense_known, labels_known, ppi_len_known, ks=ks_list)
    auc_k_value = M.auc_k(dense_known, labels_known, ppi_len_known, ks=ks_list)
    reliability = _reliability(dense_known, labels_known, ppi_len_known, k_final, ece_bins)

    echo_front = PL.echo_pareto_front(
        dense_known, labels_known, ppi_len_known, echo_taus, ks=ks_list
    )
    echo = {"taus": list(echo_taus), "pareto_front": [list(p) for p in echo_front]}

    cape_summary: dict[str, Any] | None = None
    if cape is not None:
        result = PL.p_cape(
            dense,
            np.asarray(ppi_len),
            np.asarray(cape.tau_c),
            cape.theta,
            energy_k=cape.energy_k,
            ks=ks_list,
        )
        rejected = result.prediction == unknown_label
        cape_summary = {
            "mean_k": M.mean_k(result.k),
            "coverage": M.coverage(result.committed),
            "committed_accuracy": M.committed_accuracy(
                result.prediction, labels, result.committed
            ),
            "n_rejected": int(rejected.sum()),
            "rejection_earliness": M.rejection_earliness(
                result.k, labels == unknown_label, rejected
            ),
        }

    auroc_curve: dict[int, float] | None = None
    if n_unknown > 0:
        auroc_curve = _auroc_curve(dense, labels == unknown_label, np.asarray(ppi_len), ks_list)

    meta = {
        "indexing": dense.indexing,
        "evaluated_k": list(dense.evaluated_k),
        "seed": seed,
        "unknown_label": unknown_label,
        "ece_bins": ece_bins,
    }

    payload_for_hash = {
        "split": split,
        "n_classes": n_classes,
        "n_flows": n_flows,
        "n_known": n_known,
        "n_unknown": n_unknown,
        "ks": ks_list,
        "acc_at_k": {str(k): v for k, v in acc_curve.items()},
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "k95": k95_value,
        "auc_k": auc_k_value,
        "reliability": reliability,
        "echo": echo,
        "cape": cape_summary,
        "auroc_at_k": None if auroc_curve is None else {str(k): v for k, v in auroc_curve.items()},
        "meta": meta,
    }
    report_hash = stable_hash(payload_for_hash)

    return Report(
        split=split,
        n_classes=n_classes,
        n_flows=n_flows,
        n_known=n_known,
        n_unknown=n_unknown,
        ks=ks_list,
        acc_at_k=acc_curve,
        macro_f1=macro_f1,
        balanced_accuracy=balanced_acc,
        k95=k95_value,
        auc_k=auc_k_value,
        reliability=reliability,
        echo=echo,
        cape=cape_summary,
        auroc_at_k=auroc_curve,
        meta=meta,
        hash=report_hash,
        created_at=now_iso(),
    )
