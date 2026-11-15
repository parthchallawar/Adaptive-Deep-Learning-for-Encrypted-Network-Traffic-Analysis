"""``python -m adl_etc.evaluation.run --config configs/eval/<name>.yaml``

Spec 004's ``evaluate(model_or_policy, split) -> Report`` entry point (plan T7).
Loads a trained baseline checkpoint, runs it over a real split, and writes
``results/<out_dir>/eval/<split>/``: the `EvalArrays` (``logits.npy`` +
friends), ``report.json``, and a handful of PNG figures.

This file lives under ``adl_etc.evaluation``, which ``tests/test_import_boundary.py``
walks to prove torch/mlflow-freedom stays intact, so every torch-touching
import (the model, the checkpoint, the split loader that opens real shard
memmaps) is inside :func:`main`, not at module level -- the same pattern
``plotstyle.py`` uses for matplotlib. Nothing above the ``if __name__`` guard
needs torch to exist.

Optional keys (plan phase 3, T8):

- ``run_dir``: a training run's directory. The model is rebuilt from its
  ``config_resolved.yaml``, and ``checkpoint``/``label_space`` default to its
  ``best.pt``/``label_space.json``, so one eval config serves every model.
  ``out_dir`` stays separate (on Kaggle a previous session's run directory is a
  read-only mount).
- ``save_arrays`` (default true): false skips ``logits.npy`` and friends (~2.25 GB
  per 250k-flow PAT pass) and keeps only reports, summaries and figures.
  ``predictions.npz`` (:mod:`.bootstrap`, ~15 MB) is written either way: it is
  what a paired bootstrap between two models needs.
- ``perturb``: a list of ``{name, kind, value}`` (:mod:`.robustness`). After the
  clean pass, each condition damages the same flows with its own fixed-seed draw,
  re-runs the model, and writes ``perturb/<name>/report.json``; ``robustness.json``
  then summarises every condition against ``clean``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

CAL_FILE = "calibration.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path, help="configs/eval/*.yaml")
    parser.add_argument("--override", nargs="*", default=(), help="dotted key=value overrides")
    args = parser.parse_args(argv)

    import numpy as np
    from omegaconf import OmegaConf

    from adl_etc.data.features import Standardizer
    from adl_etc.evaluation.artifact import DEFAULT_EVAL_CAP, EvalArrays, subsample_for_eval
    from adl_etc.evaluation.bootstrap import PREDICTIONS_FILE, Predictions
    from adl_etc.evaluation.calibration import Calibration, ece_per_k
    from adl_etc.evaluation.dense_logits import from_causal
    from adl_etc.evaluation.plotstyle import (
        plot_accuracy_vs_k,
        plot_auroc_vs_k,
        plot_pareto_front,
        plot_reliability_diagram,
    )
    from adl_etc.evaluation.policies import fit_cape_thresholds
    from adl_etc.evaluation.protocol import load_split
    from adl_etc.evaluation.report import CapeConfig, build_report
    from adl_etc.evaluation.robustness import parse_perturbations, perturb, summarize
    from adl_etc.models.baselines.predict import predict_dense, predict_heads
    from adl_etc.models.registry import build_model
    from adl_etc.training.checkpoint import (
        check_compatible,
        inference_state_dict,
        load_checkpoint,
    )
    from adl_etc.training.datasets import ArrayData
    from adl_etc.training.labels import LabelSpace
    from adl_etc.utils.atomic import atomic_write_json
    from adl_etc.utils.config import load_config
    from adl_etc.utils.provenance import file_sha256

    cfg = load_config(args.config, overrides=args.override)

    if not cfg.get("out_dir"):
        raise ValueError("`out_dir` is required (where eval/<split>/ is written)")
    run_dir = Path(str(cfg.run_dir)) if cfg.get("run_dir") else None
    if run_dir is not None:
        model_cfg = OmegaConf.load(run_dir / "config_resolved.yaml").model
        checkpoint = Path(str(cfg.get("checkpoint") or run_dir / "best.pt"))
        label_space_path = Path(str(cfg.get("label_space") or run_dir / "label_space.json"))
    else:
        model_cfg = cfg.model
        checkpoint = Path(str(cfg.checkpoint))
        label_space_path = Path(str(cfg.label_space))
    save_arrays = bool(cfg.get("save_arrays", True))
    # Parsed before any data is loaded, so a bad entry fails in seconds, not after a pass.
    perturbations = None
    if cfg.get("perturb"):
        raw = OmegaConf.to_container(cfg.perturb)
        if not isinstance(raw, list):
            raise ValueError(f"`perturb` must be a list of {{name, kind, value}}, got {raw!r}")
        perturbations = parse_perturbations(raw)

    label_space = LabelSpace.load(label_space_path)
    standardizer = Standardizer.load(cfg.standardizer) if cfg.get("standardizer") else None

    loaded = load_split(cfg.split, root=cfg.get("data_root", "data/processed"))
    try:
        split_name = str(cfg.eval_split)
        data = ArrayData.from_shards(loaded.shard_sets[split_name], loaded.masks[split_name])
    finally:
        loaded.close()

    seed = int(cfg.get("seed", 0))
    cap = int(cfg.get("eval_cap", DEFAULT_EVAL_CAP))
    idx = subsample_for_eval(len(data), cap=cap, seed=seed)
    sub = _at(data, idx)

    model = build_model(model_cfg, label_space.n_classes)
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    check_compatible(
        ckpt,
        label_space_hash=label_space.hash,
        standardizer_hash=(standardizer.hash if standardizer is not None else None),
    )
    model.load_state_dict(inference_state_dict(ckpt))

    device = str(cfg.get("device", "cpu"))
    labels_model = label_space.to_model(sub.label).astype(np.int64)

    # A prefix-trained PAT carries calibration.json beside its checkpoint (spec 008):
    # apply it and report calibrated ECE per K on this split (plan phase 3, T7).
    cal_path = Path(str(cfg.get("calibration") or checkpoint.parent / CAL_FILE))
    calibration = None
    safe = None
    if hasattr(model, "head_safe") and cal_path.is_file():
        calibration = Calibration.load(cal_path, checkpoint_hash=file_sha256(checkpoint))
        heads = predict_heads(model, sub, label_space, standardizer, device=device)
        dense = from_causal(heads["cls"])
        safe = heads["safe"]
    else:
        dense = predict_dense(model, sub, label_space, standardizer, device=device)

    out_dir = Path(cfg.out_dir) / "eval" / split_name
    ea = EvalArrays(
        dense=dense,
        labels=labels_model,
        ppi_len=sub.ppi_len.astype(np.int64),
        flow_index=idx.astype(np.int64),
    )
    if save_arrays:
        ea.save(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    # Always (~15 MB at 250k flows): what a paired comparison between two models needs.
    Predictions.from_dense(dense, labels_model, sub.ppi_len, idx).save(out_dir / PREDICTIONS_FILE)
    label_space.save(out_dir / "label_space.json")
    if calibration is not None and safe is not None:
        if save_arrays:
            np.save(out_dir / "safe.npy", safe)
        uncalibrated = Calibration(
            calibration.checkpoint_hash,
            [0.0] * label_space.n_classes,
            [1.0] * len(calibration.temperatures),
            calibration.safe_maps,
            calibration.n_fit,
        )
        cal_report = {
            "split": split_name,
            "calibration": cal_path.as_posix(),
            **ece_per_k(calibration, dense.logits, labels_model, sub.ppi_len, safe_logits=safe),
            "uncalibrated": ece_per_k(uncalibrated, dense.logits, labels_model, sub.ppi_len),
        }
        atomic_write_json(out_dir / "calibration_report.json", cal_report)

    cape_cfg = None
    cape_section = cfg.get("cape")
    if cape_section:
        cal_loaded = load_split(cfg.split, root=cfg.get("data_root", "data/processed"))
        try:
            cal_name = str(cape_section.calibration_split)
            cal_data = ArrayData.from_shards(
                cal_loaded.shard_sets[cal_name], cal_loaded.masks[cal_name]
            )
        finally:
            cal_loaded.close()
        cal_dense = predict_dense(model, cal_data, label_space, standardizer, device=device)
        cal_labels = label_space.to_model(cal_data.label).astype(np.int64)
        tau_c = fit_cape_thresholds(
            cal_dense,
            cal_labels,
            cal_data.ppi_len,
            n_classes=label_space.n_classes,
            precision=float(cape_section.get("precision", 0.95)),
            min_support=int(cape_section.get("min_support", 20)),
        )
        cape_cfg = CapeConfig(
            tau_c=tau_c.tolist(),
            theta=float(cape_section.theta),
            energy_k=int(cape_section.get("energy_k", 10)),
        )

    report = build_report(
        dense,
        labels_model,
        sub.ppi_len,
        n_classes=label_space.n_classes,
        split=split_name,
        cape=cape_cfg,
        seed=seed,
    )
    report.save(out_dir / "report.json")

    model_name = str(model_cfg.get("name", "model"))
    plot_accuracy_vs_k({model_name: report.acc_at_k}, out_dir / "acc_vs_k.png", title=split_name)
    plot_pareto_front({"P-ECHO": report.echo["pareto_front"]}, out_dir / "echo_pareto.png")
    plot_reliability_diagram(report.reliability, out_dir / "reliability.png")
    if report.auroc_at_k is not None:
        plot_auroc_vs_k(report.auroc_at_k, out_dir / "auroc_vs_k.png", title=split_name)

    print(f"wrote {out_dir}/report.json (hash {report.hash[:12]}, {report.n_flows} flows)")

    if perturbations is not None:
        # The clean pass's logits (float32, ~4.5 GB at 250k PAT flows) are not needed
        # again; free them before each damaged pass allocates its own.
        del dense, ea, safe
        reports = {"clean": {**report.to_dict(), "hash": report.hash}}
        curves = {"clean": report.acc_at_k}
        for p in perturbations[1:]:
            ppi_p, len_p = perturb(sub.ppi, sub.ppi_len, p, seed=seed)
            damaged = ArrayData(
                ppi=ppi_p, ppi_len=len_p, label=sub.label, session_id=sub.session_id
            )
            dense_p = predict_dense(model, damaged, label_space, standardizer, device=device)
            r = build_report(
                dense_p,
                labels_model,
                damaged.ppi_len,
                n_classes=label_space.n_classes,
                split=f"{split_name}/{p.name}",
                seed=seed,
            )
            del dense_p
            r.save(out_dir / "perturb" / p.name / "report.json")
            reports[p.name] = {**r.to_dict(), "hash": r.hash}
            curves[p.name] = r.acc_at_k
            print(f"  {p.name}: auc_k {r.auc_k:.4f} (clean {report.auc_k:.4f})")
        summary = summarize(reports, perturbations)
        summary["model"] = model_name
        summary["checkpoint"] = checkpoint.as_posix()
        summary["seed"] = seed
        atomic_write_json(out_dir / "robustness.json", summary)
        plot_accuracy_vs_k(
            curves, out_dir / "robustness_acc_vs_k.png", title=f"{split_name}, damaged"
        )
        print(f"wrote {out_dir}/robustness.json ({len(perturbations) - 1} conditions)")
    return 0


def _at(data, idx):
    """``ArrayData`` restricted to explicit indices (its own ``subsample`` only
    takes a count + RNG, not the eval cap's specific draw)."""
    from adl_etc.training.datasets import ArrayData

    return ArrayData(
        ppi=data.ppi[idx],
        ppi_len=data.ppi_len[idx],
        label=data.label[idx],
        session_id=None if data.session_id is None else data.session_id[idx],
    )


if __name__ == "__main__":
    raise SystemExit(main())
