"""``python -m adl_etc.evaluation.score_run --config configs/eval/d1_open_scores.yaml
--override run_dir=results/<run>`` (plan phase 4, T1-T2).

Runs a trained model once over the configured splits and writes, per split, a
:class:`~adl_etc.evaluation.scores.ScoreArrays` to ``<out_dir>/eval/<split>/scores/``. For a PAT
run it first fits the Gaussian bank on a per-class sample of known training flows, scores
``val`` with it, fits the unknown model on the **known** validation flows
(:class:`~adl_etc.inference.unknown.UnknownModel`) and writes it to
``<out_dir>/unknown.{npz,json}``. Every later split reuses the bank. A GRU run (no projection
head) gets class-logit columns only.

Like ``evaluation/run.py`` this file sits under ``adl_etc.evaluation``, which
``tests/test_import_boundary.py`` proves torch-free, so every torch-touching import is inside
:func:`main`.

Config (see ``configs/eval/d1_open_scores.yaml``):

- ``run_dir`` / ``out_dir``: the training run, and where ``eval/`` is written (default: the run).
- ``splits``: ``{name: {cap: N}}`` (one uniform seeded sample of the whole split, the same draw
  phase 3's evaluations used, so flows line up with their ``predictions.npz``) or
  ``{name: {per_period: N}}`` (that many flows from every period, for the drift weeks).
- ``unknown``: ``enabled``, ``per_class``, ``min_class_flows``, ``shrinkage``, ``weights``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

CAL_FILE = "calibration.json"
UNKNOWN_STEM = "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--override", nargs="*", default=(), help="dotted key=value overrides")
    args = parser.parse_args(argv)

    import time

    import numpy as np
    from omegaconf import OmegaConf

    from adl_etc.data.features import Standardizer
    from adl_etc.evaluation.artifact import subsample_for_eval
    from adl_etc.evaluation.calibration import Calibration
    from adl_etc.evaluation.protocol import load_split
    from adl_etc.evaluation.scores import concat
    from adl_etc.inference.unknown import GaussianSums, UnknownModel, class_validity
    from adl_etc.models.registry import build_model
    from adl_etc.training.checkpoint import inference_state_dict, load_checkpoint
    from adl_etc.training.datasets import ArrayData
    from adl_etc.training.labels import LabelSpace
    from adl_etc.training.score_pass import (
        accumulate_projection_sums,
        has_pat_heads,
        identity_calibration,
        sample_known_per_class,
        score_flows,
    )
    from adl_etc.utils.config import load_config
    from adl_etc.utils.provenance import file_sha256
    from adl_etc.utils.seeding import seeded_generator

    cfg = load_config(args.config, overrides=args.override)
    if not cfg.get("run_dir"):
        raise ValueError("`run_dir` is required")
    run_dir = Path(str(cfg.run_dir))
    out_dir = Path(str(cfg.out_dir)) if cfg.get("out_dir") else run_dir
    checkpoint = run_dir / "best.pt"
    seed = int(cfg.get("seed", 0))
    device = str(cfg.get("device", "cpu"))
    batch = int(cfg.get("batch_size", 1024))

    label_space = LabelSpace.load(run_dir / "label_space.json")
    standardizer = Standardizer.load(cfg.standardizer)
    model_cfg = OmegaConf.load(run_dir / "config_resolved.yaml").model
    model = build_model(model_cfg, label_space.n_classes)
    ckpt_hash = file_sha256(checkpoint)
    model.load_state_dict(inference_state_dict(load_checkpoint(checkpoint, map_location="cpu")))
    cal_path = run_dir / CAL_FILE
    calibration = (
        Calibration.load(cal_path, checkpoint_hash=ckpt_hash)
        if cal_path.is_file()
        else identity_calibration(label_space.n_classes, ckpt_hash)
    )
    pat = has_pat_heads(model)
    ucfg = cfg.get("unknown") or {}
    want_unknown = bool(ucfg.get("enabled", True)) and pat

    loaded = load_split(cfg.split, root=cfg.get("data_root", "data/processed"))
    base_meta = {
        "run": run_dir.name,
        "model": str(model_cfg.get("name", "model")),
        "checkpoint_hash": ckpt_hash,
        "calibrated": cal_path.is_file(),
        "split_file": str(cfg.split),
        "seed": seed,
    }
    try:
        bank = None
        if want_unknown:
            t0 = time.time()
            train = ArrayData.from_shards(loaded.shard_sets["train"], loaded.masks["train"])
            y = label_space.to_model(train.label).astype(np.int64)
            rng = seeded_generator(seed, "unknown_fit")
            idx = sample_known_per_class(y, int(ucfg.get("per_class", 1500)), rng)
            sample = ArrayData(
                ppi=train.ppi[idx],
                ppi_len=train.ppi_len[idx],
                label=train.label[idx],
                session_id=None,
            )
            del train
            valid = class_validity(
                y[idx], sample.ppi_len, label_space.n_classes, int(ucfg.get("min_class_flows", 50))
            )
            sums = GaussianSums(label_space.n_classes, valid, dim=128)
            accumulate_projection_sums(
                model, sample, label_space, standardizer, sums, batch_size=batch, device=device
            )
            bank = sums.bank(float(ucfg.get("shrinkage", 0.1)))
            have = bank.valid.sum(1)
            print(
                f"gaussian bank from {len(idx)} known training flows ({time.time() - t0:.0f}s); "
                f"classes with a Gaussian at K=1/5/10/20/30: "
                f"{have[0]}/{have[4]}/{have[9]}/{have[19]}/{have[29]} of {label_space.n_classes}"
            )
            del sample

        names = list(cfg.splits.keys())
        if want_unknown and "val" in names:  # val first: the unknown model is fitted on it
            names = ["val"] + [n for n in names if n != "val"]
        for name in names:
            scfg = cfg.splits[name]
            period_names = list(loaded.spec["splits"][name]["periods"])
            shard_sets, masks = loaded.shard_sets[name], loaded.masks[name]
            t0 = time.time()
            if scfg.get("per_period"):
                parts, offset = [], 0
                for p, (ss, mask) in enumerate(zip(shard_sets, masks, strict=True)):
                    d = ArrayData.from_shards([ss], [mask])
                    sel = subsample_for_eval(len(d), cap=int(scfg.per_period), seed=seed + p)
                    parts.append(
                        score_flows(
                            model,
                            _take(d, sel),
                            label_space,
                            standardizer,
                            calibration,
                            bank=bank,
                            flow_index=sel + offset,
                            period=np.full(len(sel), p, dtype=np.int16),
                            meta={**base_meta, "split": name, "periods": period_names},
                            batch_size=batch,
                            device=device,
                        )
                    )
                    offset += len(d)
                    del d
                scores = concat(parts)
            else:
                data = ArrayData.from_shards(shard_sets, masks)
                sel = subsample_for_eval(len(data), cap=int(scfg.cap), seed=seed)
                lengths = np.array(
                    [
                        int(m.sum()) if m is not None else len(s)
                        for s, m in zip(shard_sets, masks, strict=True)
                    ]
                )
                period = np.searchsorted(np.cumsum(lengths), sel, side="right").astype(np.int16)
                scores = score_flows(
                    model,
                    _take(data, sel),
                    label_space,
                    standardizer,
                    calibration,
                    bank=bank,
                    flow_index=sel,
                    period=period,
                    meta={
                        **base_meta,
                        "split": name,
                        "periods": period_names,
                        "cap": int(scfg.cap),
                    },
                    batch_size=batch,
                    device=device,
                )
                del data
            if name == "val" and want_unknown:
                assert bank is not None  # want_unknown implies the bank was fitted above
                w_e, w_m, w_s = (float(w) for w in ucfg.get("weights", (1.0, 1.0, 0.5)))
                unknown = UnknownModel.fit(
                    bank, scores, weights=(w_e, w_m, w_s), checkpoint_hash=ckpt_hash
                )
                unknown.save(out_dir / UNKNOWN_STEM)
                base_meta["unknown_model_sha256"] = file_sha256(
                    (out_dir / UNKNOWN_STEM).with_suffix(".npz")
                )
                scores = _with_meta(scores, base_meta)
            scores.save(out_dir / "eval" / name / "scores")
            print(
                f"{name}: {len(scores)} flows, {len(period_names)} periods, "
                f"hash {scores.content_hash()[:12]} ({time.time() - t0:.0f}s)"
            )
    finally:
        loaded.close()
    return 0


def _take(data, idx):
    from adl_etc.training.datasets import ArrayData

    return ArrayData(
        ppi=data.ppi[idx], ppi_len=data.ppi_len[idx], label=data.label[idx], session_id=None
    )


def _with_meta(scores, extra):
    from dataclasses import replace

    return replace(scores, meta={**scores.meta, **extra})


if __name__ == "__main__":
    raise SystemExit(main())
