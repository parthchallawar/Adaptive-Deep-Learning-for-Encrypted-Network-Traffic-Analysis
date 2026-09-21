"""B1: XGBoost on flow statistics (spec 005).

The classical tabular baseline, made *honest about earliness*: one model per K on
spec 004's grid, each trained on :func:`~adl_etc.data.prefix_stats.prefix_flowstats`
(statistics recomputed from the first K packets only). A single whole-flow model
evaluated early would read the future.

A per-K model, so its saved logits are ``nominal``-indexed
(``evaluation.dense_logits``). CPU, local. Class weights are inverse-frequency,
capped, on the same rule as the neural baselines' sampler.

Not every class need appear in training (a rare class can have no flows in the
train period). XGBoost needs contiguous labels, so the classes present are
remapped for fitting and their probabilities scattered back into the full
``C``-wide output; an absent class gets a very low logit and is never predicted.
Fewer than two present classes raises.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf

from adl_etc.data.prefix_stats import prefix_flowstats
from adl_etc.evaluation.dense_logits import DenseLogits, from_per_k, grid_from
from adl_etc.evaluation.metrics import macro_f1
from adl_etc.training.datasets import ArrayData
from adl_etc.training.labels import LabelSpace
from adl_etc.utils.atomic import atomic_write_json
from adl_etc.utils.seeding import seeded_generator
from adl_etc.utils.tracking import Tracker

_LOG_FLOOR = float(np.log(1e-12))
META_JSON = "meta.json"


class XGBError(RuntimeError):
    pass


def class_weights(y: np.ndarray, n_present: int, cap: float) -> np.ndarray:
    """Per-sample weights, mean 1: ``min(cap, mean_count / n_c)`` for each sample's
    class ``c``, the same capped inverse-frequency rule the neural sampler uses."""
    counts = np.bincount(y, minlength=n_present).astype(np.float64)
    factor = np.minimum(cap, (len(y) / n_present) / counts)
    w = factor[y]
    return w / w.mean()


class XGBBaseline:
    def __init__(
        self,
        label_space: LabelSpace,
        *,
        ks: Iterable[int] | None = None,
        n_estimators: int = 2000,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        class_weight_cap: float = 10.0,
        n_jobs: int = -1,
        max_train_flows: int | None = None,
        seed: int = 0,
    ) -> None:
        self.label_space = label_space
        self.ks = grid_from(ks)
        self.params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "early_stopping_rounds": early_stopping_rounds,
            "class_weight_cap": class_weight_cap,
            "n_jobs": n_jobs,
            "max_train_flows": max_train_flows,
            "seed": seed,
        }
        self.boosters: dict[int, xgb.Booster] = {}
        self.best_iteration: dict[int, int] = {}
        self.present: np.ndarray | None = None  # model indices seen in training

    @classmethod
    def from_config(
        cls, model_cfg: Mapping[str, Any] | DictConfig, label_space: LabelSpace, *, seed: int = 0
    ) -> XGBBaseline:
        """From a config's ``model:`` section (``name: xgb`` plus constructor
        arguments). An unknown name or argument raises."""
        raw = (
            OmegaConf.to_container(model_cfg, resolve=True)
            if isinstance(model_cfg, DictConfig)
            else dict(model_cfg)
        )
        assert isinstance(raw, dict)
        kwargs: dict[str, Any] = {str(k): v for k, v in raw.items()}
        name = kwargs.pop("name", None)
        if name != "xgb":
            raise ValueError(f"expected model name 'xgb', got {name!r}")
        try:
            return cls(label_space, seed=seed, **kwargs)
        except TypeError as e:
            raise ValueError(f"bad arguments for model 'xgb': {e}") from e

    # -- training ---------------------------------------------------------------

    def fit(
        self, train: ArrayData, val: ArrayData, tracker: Tracker | None = None
    ) -> dict[int, dict[str, float]]:
        """Fit one booster per K. Returns ``{K: {val_macro_f1, best_iteration}}``."""
        cap = self.params["max_train_flows"]
        if cap is not None and len(train) > cap:
            train = train.subsample(
                int(cap), seeded_generator(self.params["seed"], "xgb-subsample")
            )
        y = self.label_space.to_model(train.label)
        if (y < 0).any():
            raise ValueError("training data contains unknown-class flows (spec 004 rule 2)")
        present = np.unique(y)
        if len(present) < 2:
            raise XGBError(f"need at least 2 classes in training, found {len(present)}")
        self.present = present
        y_fit = np.searchsorted(present, y)

        vy = self.label_space.to_model(val.label)
        keep = (vy >= 0) & np.isin(vy, present)  # early stopping needs labels the model has
        if not keep.any():
            raise ValueError("no validation flows belong to a class seen in training")
        val_kept = ArrayData(val.ppi[keep], val.ppi_len[keep], val.label[keep])
        vy_fit = np.searchsorted(present, vy[keep])

        weights = class_weights(y_fit, len(present), self.params["class_weight_cap"])
        p = self.params
        summary: dict[int, dict[str, float]] = {}
        for k in self.ks:
            clf = xgb.XGBClassifier(
                objective="multi:softprob",
                tree_method="hist",
                n_estimators=p["n_estimators"],
                learning_rate=p["learning_rate"],
                max_depth=p["max_depth"],
                subsample=p["subsample"],
                colsample_bytree=p["colsample_bytree"],
                early_stopping_rounds=p["early_stopping_rounds"],
                eval_metric="mlogloss",
                n_jobs=p["n_jobs"],
                random_state=p["seed"],
                verbosity=0,
            )
            x_tr = prefix_flowstats(train.ppi, train.ppi_len, k)
            x_va = prefix_flowstats(val_kept.ppi, val_kept.ppi_len, k)
            clf.fit(x_tr, y_fit, sample_weight=weights, eval_set=[(x_va, vy_fit)], verbose=False)
            self.boosters[k] = clf.get_booster()
            self.best_iteration[k] = int(clf.best_iteration)

            probs = self._proba(k, x_va)
            f1 = macro_f1(probs.argmax(axis=1), vy_fit)
            summary[k] = {"val_macro_f1": f1, "best_iteration": float(self.best_iteration[k])}
            if tracker is not None:
                tracker.log_metrics(summary[k], step=k)  # x-axis: packets read K
        return summary

    # -- inference ---------------------------------------------------------------

    def _proba(self, k: int, features: np.ndarray) -> np.ndarray:
        booster = self.boosters[k]
        return np.asarray(
            booster.predict(xgb.DMatrix(features), iteration_range=(0, self.best_iteration[k] + 1))
        )

    def predict_dense(self, data: ArrayData, *, dtype: type = np.float32) -> DenseLogits:
        """Log-probabilities as logits, nominal-indexed over ``self.ks``."""
        if self.present is None or not self.boosters:
            raise XGBError("the model has not been fitted or loaded")
        c = self.label_space.n_classes
        per_k: dict[int, np.ndarray] = {}
        for k in self.ks:
            probs = self._proba(k, prefix_flowstats(data.ppi, data.ppi_len, k))
            logits = np.full((len(data), c), _LOG_FLOOR, dtype=np.float64)
            logits[:, self.present] = np.log(np.clip(probs, 1e-12, None))
            per_k[k] = logits.astype(dtype)
        return from_per_k(per_k, dtype=dtype)

    # -- persistence ---------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        if self.present is None or not self.boosters:
            raise XGBError("nothing to save: the model has not been fitted")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for k, booster in self.boosters.items():
            booster.save_model(str(d / f"model_k{k}.json"))
        atomic_write_json(
            d / META_JSON,
            {
                "ks": list(self.ks),
                "present": self.present.tolist(),
                "best_iteration": {str(k): v for k, v in self.best_iteration.items()},
                "label_space_hash": self.label_space.hash,
                "params": self.params,
            },
        )

    @classmethod
    def load(cls, directory: str | Path, label_space: LabelSpace) -> XGBBaseline:
        d = Path(directory)
        try:
            meta: Mapping[str, Any] = json.loads((d / META_JSON).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise XGBError(f"unreadable XGBoost baseline in {d}: {e}") from e
        if meta["label_space_hash"] != label_space.hash:
            raise XGBError(
                "this baseline was trained under a different label space "
                f"({meta['label_space_hash'][:12]} vs {label_space.hash[:12]})"
            )
        params = {k: v for k, v in meta["params"].items()}
        model = cls(label_space, ks=meta["ks"], **params)
        model.present = np.asarray(meta["present"], dtype=np.int64)
        for k in model.ks:
            booster = xgb.Booster()
            booster.load_model(str(d / f"model_k{k}.json"))
            model.boosters[k] = booster
            model.best_iteration[k] = int(meta["best_iteration"][str(k)])
        return model
