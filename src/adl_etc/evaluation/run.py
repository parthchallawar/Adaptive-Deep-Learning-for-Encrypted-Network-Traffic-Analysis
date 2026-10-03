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

    from adl_etc.data.features import Standardizer
    from adl_etc.evaluation.artifact import DEFAULT_EVAL_CAP, EvalArrays, subsample_for_eval
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

    label_space = LabelSpace.load(cfg.label_space)
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

    model = build_model(cfg.model, label_space.n_classes)
    ckpt = load_checkpoint(cfg.checkpoint, map_location="cpu")
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
    cal_path = Path(str(cfg.get("calibration") or Path(str(cfg.checkpoint)).parent / CAL_FILE))
    calibration = None
    safe = None
    if hasattr(model, "head_safe") and cal_path.is_file():
        calibration = Calibration.load(cal_path, checkpoint_hash=file_sha256(cfg.checkpoint))
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
    ea.save(out_dir)
    label_space.save(out_dir / "label_space.json")
    if calibration is not None and safe is not None:
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

    model_name = str(cfg.model.get("name", "model"))
    plot_accuracy_vs_k({model_name: report.acc_at_k}, out_dir / "acc_vs_k.png", title=split_name)
    plot_pareto_front({"P-ECHO": report.echo["pareto_front"]}, out_dir / "echo_pareto.png")
    plot_reliability_diagram(report.reliability, out_dir / "reliability.png")
    if report.auroc_at_k is not None:
        plot_auroc_vs_k(report.auroc_at_k, out_dir / "auroc_vs_k.png", title=split_name)

    print(f"wrote {out_dir}/report.json (hash {report.hash[:12]}, {report.n_flows} flows)")
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
