"""The model bundle the service serves (spec 016; plan phase 6, T2, correction S2).

A bundle is named by ``results/models/production.json`` (written by
``scripts/make_production_bundle.py``): a training run directory (``best.pt``,
``calibration.json``, ``label_space.json``, ``config_resolved.yaml``), the phase-4 unknown model
(``unknown.{npz,json}``), the standardizer, and the phase-4/5 operating point (commit threshold,
``f``, the drift monitor's validation reference and class histogram).

:class:`ModelBundle` is the engine's :class:`~adl_etc.service.engine.Scorer`. Its
:meth:`~ModelBundle.score` runs the model once over a batch of prefixes and computes the score
columns **only at the positions each flow has just reached**, using the same functions as the
offline score pass (``evaluation.scores.class_scores``, ``Calibration.safe_maps``,
``GaussianBank.mahal_at``, ``running_surprise``, ``UnknownModel.score_at``) and rounding each
column to the dtype the offline score files store, so a streamed decision equals the offline
replay's. Scoring all 30 positions as the offline pass does was measured at ~0.5 s per 64-flow
batch on this laptop, 40% of it in the Mahalanobis term.

Needs torch: import this module only where torch is installed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adl_etc.data import ppi as P
from adl_etc.data.features import Standardizer
from adl_etc.evaluation import scores as S
from adl_etc.evaluation.calibration import Calibration
from adl_etc.inference.unknown import UnknownModel
from adl_etc.training.datasets import ArrayData, FlowBatches, to_torch
from adl_etc.training.labels import LabelSpace
from adl_etc.training.score_pass import next_packet_columns
from adl_etc.utils.provenance import file_sha256

K_MAX = P.K_MAX


class BundleError(RuntimeError):
    pass


def _as(name: str, x: np.ndarray) -> np.ndarray:
    """Round a column to its stored dtype (as the offline score file holds it), back to float64."""
    return np.asarray(x).astype(S.FIELDS_2D[name]).astype(np.float64)


@dataclass
class ModelBundle:
    version: str
    run: str
    run_dir: Path
    checkpoint_hash: str
    model: torch.nn.Module
    label_space: LabelSpace
    standardizer: Standardizer
    calibration: Calibration
    unknown: UnknownModel
    reference: dict[str, Any]
    production: dict[str, Any]
    loaded_at: float = field(default_factory=time.time)
    load_seconds: float = 0.0

    # -- loading ----------------------------------------------------------------------------

    @classmethod
    def load(cls, production_path: str | Path, *, root: str | Path = ".") -> ModelBundle:
        """Load and cross-check every file. Raises :class:`BundleError` naming what is missing."""
        from omegaconf import OmegaConf

        from adl_etc.models.registry import build_model
        from adl_etc.training.checkpoint import inference_state_dict, load_checkpoint

        t0 = time.time()
        root = Path(root)
        production_path = Path(production_path)
        if not production_path.is_file():
            raise BundleError(
                f"{production_path} not found: run `python scripts/make_production_bundle.py` first"
            )
        prod = json.loads(production_path.read_text(encoding="utf-8"))
        run_dir = root / prod["run_dir"]
        needed = [
            run_dir / "best.pt",
            run_dir / "calibration.json",
            run_dir / "label_space.json",
            run_dir / "config_resolved.yaml",
            (root / prod["unknown"]).with_suffix(".npz"),
            root / prod["standardizer"],
        ]
        missing = [str(p) for p in needed if not p.is_file()]
        if missing:
            raise BundleError(f"model bundle incomplete, missing: {missing}")
        ckpt_hash = file_sha256(run_dir / "best.pt")
        if prod.get("checkpoint_sha256") and prod["checkpoint_sha256"] != ckpt_hash:
            raise BundleError(
                f"best.pt hash {ckpt_hash[:12]} differs from production.json's "
                f"{prod['checkpoint_sha256'][:12]}"
            )
        label_space = LabelSpace.load(run_dir / "label_space.json")
        model_cfg = OmegaConf.load(run_dir / "config_resolved.yaml").model
        model = build_model(model_cfg, label_space.n_classes)
        model.load_state_dict(
            inference_state_dict(load_checkpoint(run_dir / "best.pt", map_location="cpu"))
        )
        model.eval()
        if not hasattr(model, "head_proj"):
            raise BundleError("the service needs a PAT (projection and next-packet heads)")
        calibration = Calibration.load(run_dir / "calibration.json", checkpoint_hash=ckpt_hash)
        unknown = UnknownModel.load(root / prod["unknown"], checkpoint_hash=ckpt_hash)
        standardizer = Standardizer.load(root / prod["standardizer"])
        return cls(
            version=ckpt_hash[:12],
            run=str(prod.get("run", run_dir.name)),
            run_dir=run_dir,
            checkpoint_hash=ckpt_hash,
            model=model,
            label_space=label_space,
            standardizer=standardizer,
            calibration=calibration,
            unknown=unknown,
            reference=dict(prod.get("reference", {})),
            production=prod,
            load_seconds=time.time() - t0,
        )

    @property
    def class_names(self) -> list[str]:
        return self.label_space.class_names

    @property
    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    # -- the forward pass ---------------------------------------------------------------------

    #: Prefix-length buckets: rows are grouped so each forward runs only as many positions as
    #: its longest prefix (``PAT.forward(n_packets=)``), instead of all 31 for every row.
    BUCKETS = (3, 6, 10, 16, K_MAX)

    @torch.inference_mode()
    def _forward(self, ppi: np.ndarray, ppi_len: np.ndarray) -> dict[str, np.ndarray]:
        n = len(ppi_len)
        ppi_len = np.asarray(ppi_len, dtype=np.int64)
        data = ArrayData(
            ppi=np.ascontiguousarray(ppi, dtype=np.int16),
            ppi_len=ppi_len,
            label=np.full(n, -1, dtype=np.int64),
        )
        views = tuple(getattr(self.model, "views", ("tokens",)))
        fb = FlowBatches(data, self.label_space, standardizer=self.standardizer, views=views)
        c = self.label_space.n_classes
        out = {
            "cls": np.zeros((n, K_MAX, c), np.float32),
            "safe": np.zeros((n, K_MAX), np.float32),
            "proj": np.zeros((n, K_MAX, self.unknown.bank.means.shape[-1]), np.float32),
            "pkt_nll": np.zeros((n, K_MAX), np.float32),
        }
        lo = 0
        for hi in self.BUCKETS:
            rows = np.flatnonzero((ppi_len > lo) & (ppi_len <= hi))
            lo = hi
            if not len(rows):
                continue
            t = int(ppi_len[rows].max())
            batch = to_torch(fb.batch(rows))
            res = self.model(batch, n_packets=t)
            padded = {
                name: torch.nn.functional.pad(res[name], (0, 0, 0, K_MAX - t))
                for name in ("next_size", "next_ipt", "next_dir")
            }
            nll, _ = next_packet_columns(padded, batch["tokens"])
            out["cls"][rows, :t] = res["cls"].float().numpy()
            out["safe"][rows, :t] = res["safe"].float().numpy()
            out["proj"][rows, :t] = res["proj"].float().numpy()
            out["pkt_nll"][rows] = nll
        return out

    def score(
        self, ppi: np.ndarray, ppi_len: np.ndarray, k_from: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Score columns at 1-based positions ``k_from[i]..ppi_len[i]`` of each row (``[B, 30]``,
        NaN elsewhere). ``ppi`` holds each flow's packets so far, zero beyond ``ppi_len``."""
        ppi_len = np.asarray(ppi_len, dtype=np.int64)
        k_from = np.asarray(k_from, dtype=np.int64)
        if (ppi_len < 1).any() or (k_from < 1).any() or (k_from > ppi_len).any():
            raise ValueError("need 1 <= k_from <= ppi_len for every row")
        b = len(ppi_len)
        h = self._forward(ppi, ppi_len)
        # surprise from the stored-precision NLL, exactly as ScoreArrays.surprise derives it
        pkt_nll = S.fill_forward(h["pkt_nll"], ppi_len).astype(S.FIELDS_2D["pkt_nll"])
        surprise_all = S.running_surprise(pkt_nll, ppi_len).astype(np.float64)

        rows = np.concatenate(
            [
                np.full(int(t - f + 1), i)
                for i, (f, t) in enumerate(zip(k_from, ppi_len, strict=True))
            ]
        )
        pos = np.concatenate([np.arange(f - 1, t) for f, t in zip(k_from, ppi_len, strict=True)])
        out = {
            name: np.full((b, K_MAX), np.nan)
            for name in ("conf", "p_safe", "energy", "mahal", "surprise", "u")
        }
        out["pred"] = np.zeros((b, K_MAX), dtype=np.int64)
        adjustment = np.asarray(self.calibration.adjustment)
        temps = np.asarray(self.calibration.temperatures)
        for p in np.unique(pos):
            sel = pos == p
            r = rows[sel]
            c = S.class_scores(h["cls"][r, p : p + 1], adjustment, temps[p : p + 1])
            out["pred"][r, p] = np.asarray(c["pred"][:, 0]).astype(S.FIELDS_2D["pred"])
            out["conf"][r, p] = _as("conf", c["conf"][:, 0])
            out["energy"][r, p] = _as("energy", c["energy"][:, 0])
            out["p_safe"][r, p] = _as(
                "p_safe", self.calibration.safe_maps[p](h["safe"][r, p].astype(np.float64))
            )
            out["mahal"][r, p] = _as(
                "mahal", self.unknown.bank.mahal_at(h["proj"][r, p], np.full(len(r), p))
            )
            out["surprise"][r, p] = surprise_all[r, p]
            out["u"][r, p] = self.unknown.score_at(
                out["energy"][r, p], out["mahal"][r, p], out["surprise"][r, p], np.full(len(r), p)
            )
        return out

    def theta_reject(self, f: float) -> np.ndarray:
        return self.unknown.theta_reject(f)

    # -- the flow inspector -------------------------------------------------------------------

    def explain(self, ppi: np.ndarray, ppi_len: int, top: int = 3) -> dict[str, Any]:
        """Per-K detail for one flow: the top classes' calibrated probabilities, ``p_safe``,
        ``u`` and the per-packet next-packet surprise (spec 017's inspector)."""
        n = int(ppi_len)
        if n < 1:
            return {"k": [], "top": [], "p_safe": [], "u": [], "surprise": [], "pkt_nll": []}
        ppi = np.asarray(ppi, dtype=np.int16).reshape(1, K_MAX, P.PPI_CHANNELS)
        h = self._forward(ppi, np.array([n]))
        cols = self.score(ppi, np.array([n]), np.array([1]))
        probs = self.calibration.probabilities(h["cls"][:, :n])[0]  # [n, C]
        names = self.class_names
        order = np.argsort(-probs, axis=1)[:, :top]
        top_rows = [
            [{"class": names[j], "p": float(probs[k, j])} for j in order[k]] for k in range(n)
        ]
        return {
            "k": list(range(1, n + 1)),
            "top": top_rows,
            "pred": [names[int(c)] for c in cols["pred"][0, :n]],
            "p_safe": [float(x) for x in cols["p_safe"][0, :n]],
            "u": [float(x) for x in cols["u"][0, :n]],
            "surprise": [float(x) for x in cols["surprise"][0, :n]],
            "pkt_nll": [float(x) for x in h["pkt_nll"][0, :n]],
        }

    def describe(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run": self.run,
            "run_dir": str(self.run_dir),
            "checkpoint_sha256": self.checkpoint_hash,
            "n_classes": self.label_space.n_classes,
            "n_parameters": self.n_parameters,
            "loaded_at": self.loaded_at,
            "load_seconds": round(self.load_seconds, 2),
            "unknown_weights": list(self.unknown.weights),
            "theta_commit_calibrated": self.reference.get("theta_commit"),
            "f": self.reference.get("f"),
            "torch_threads": torch.get_num_threads(),
        }
