"""Per-flow predictions and the paired bootstrap between two models (spec 004;
plan phase 3, T8).

``EvalArrays`` keeps full logits (~2.25 GB for a 250k-flow PAT pass), too big
to pull from Kaggle routinely or to store in MLflow. A comparison between two
models only needs each flow's predicted class at each K, so every evaluation
also writes ``predictions.npz`` (``int16 [N, len(ks)]``, ~15 MB at 250k x 30):

- ``flow_index``, ``labels`` (model space, unknown = ``UNKNOWN_LABEL``),
  ``ppi_len``: which flows, so two files can be checked to be the same flows;
- ``ks`` and ``preds``: the argmax at each evaluated K, read through
  ``DenseLogits.at`` (effective K for a causal model, nominal for a per-K one).

:func:`paired_bootstrap` resamples **flows** (known-class flows only, like every
accuracy in ``report.build_report``) and recomputes both models' acc@K and AUC_K
on the same resample, so the interval is for the *difference* on these flows.
It measures flow-sampling uncertainty, not seed-to-seed variance; across seeds,
``scripts/make_tables.py`` reports mean +/- std separately.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from adl_etc.evaluation.dense_logits import DenseLogits
from adl_etc.evaluation.metrics import predictions_at_k
from adl_etc.evaluation.report import UNKNOWN_LABEL

PREDICTIONS_FILE = "predictions.npz"
N_BOOTSTRAP = 1000
_FIELDS = ("flow_index", "labels", "ppi_len", "ks", "preds")


@dataclass(frozen=True)
class Predictions:
    flow_index: np.ndarray  # [N] int64
    labels: np.ndarray  # [N] int64, model space
    ppi_len: np.ndarray  # [N] int64
    ks: np.ndarray  # [J] int64
    preds: np.ndarray  # [N, J] int16

    @classmethod
    def from_dense(
        cls, dense: DenseLogits, labels: np.ndarray, ppi_len: np.ndarray, flow_index: np.ndarray
    ) -> Predictions:
        ks = np.array(sorted(dense.evaluated_k), dtype=np.int64)
        preds = np.stack([predictions_at_k(dense, ppi_len, int(k)) for k in ks], axis=1)
        if preds.max(initial=0) > np.iinfo(np.int16).max:
            raise ValueError("more classes than int16 can hold")
        return cls(
            flow_index=np.asarray(flow_index, dtype=np.int64),
            labels=np.asarray(labels, dtype=np.int64),
            ppi_len=np.asarray(ppi_len, dtype=np.int64),
            ks=ks,
            preds=preds.astype(np.int16),
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            flow_index=self.flow_index,
            labels=self.labels,
            ppi_len=self.ppi_len,
            ks=self.ks,
            preds=self.preds,
        )

    @classmethod
    def load(cls, path: str | Path) -> Predictions:
        with np.load(Path(path)) as z:
            return cls(**{name: z[name] for name in _FIELDS})

    def correct(self, ks: Sequence[int]) -> np.ndarray:
        """``bool [N_known, len(ks)]``: right at each K, known-class flows only."""
        cols = [int(np.flatnonzero(self.ks == k)[0]) for k in ks]
        known = self.labels != UNKNOWN_LABEL
        return self.preds[known][:, cols] == self.labels[known][:, None]


def paired_bootstrap(
    a: Predictions,
    b: Predictions,
    *,
    ks: Sequence[int] | None = None,
    report_ks: Sequence[int] = (1, 2, 3, 5, 8, 30),
    n: int = N_BOOTSTRAP,
    seed: int = 0,
) -> dict[str, Any]:
    """``b`` minus ``a`` on the same flows: the observed difference in acc@K (for
    ``report_ks``) and in AUC_K (mean acc over ``ks``, default every K both hold),
    each with a 95% percentile interval over ``n`` flow resamples and the share of
    resamples in which ``b`` is not better (a one-sided bootstrap p-value)."""
    if not (
        np.array_equal(a.flow_index, b.flow_index)
        and np.array_equal(a.labels, b.labels)
        and np.array_equal(a.ppi_len, b.ppi_len)
    ):
        raise ValueError("the two prediction sets are not over the same flows")
    common = sorted(set(a.ks.tolist()) & set(b.ks.tolist()))
    ks = sorted(ks) if ks is not None else common
    missing = sorted(set(ks) - set(common))
    if missing:
        raise ValueError(f"K values {missing} are not in both prediction sets")
    report_ks = [k for k in report_ks if k in ks]

    ca = a.correct(ks).astype(np.float64)
    cb = b.correct(ks).astype(np.float64)
    n_flows = ca.shape[0]
    if n_flows == 0:
        raise ValueError("no known-class flows to compare")
    col = {k: i for i, k in enumerate(ks)}

    def stats(weights: np.ndarray) -> np.ndarray:
        # acc per K for each model under one resample (flow weights = draw counts)
        return np.stack([weights @ ca, weights @ cb]) / weights.sum()

    observed = stats(np.ones(n_flows))
    rng = np.random.default_rng(seed)
    diffs = np.empty((n, len(ks)))
    for i in range(n):
        w = np.bincount(rng.integers(0, n_flows, n_flows), minlength=n_flows).astype(np.float64)
        s = stats(w)
        diffs[i] = s[1] - s[0]
    auc_diffs = diffs.mean(axis=1)

    def summary(obs_a: float, obs_b: float, resampled: np.ndarray) -> dict[str, float]:
        lo, hi = np.percentile(resampled, [2.5, 97.5])
        return {
            "a": float(obs_a),
            "b": float(obs_b),
            "diff": float(obs_b - obs_a),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "p_not_better": float(np.mean(resampled <= 0)),
        }

    return {
        "n_flows": int(n_flows),
        "n_resamples": int(n),
        "bootstrap_seed": int(seed),  # not "seed": callers label rows with the run's seed
        "ks": list(ks),
        "acc_at_k": {
            str(k): summary(observed[0, col[k]], observed[1, col[k]], diffs[:, col[k]])
            for k in report_ks
        },
        "auc_k": summary(observed[0].mean(), observed[1].mean(), auc_diffs),
    }
