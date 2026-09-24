"""``python -m adl_etc.training.run_xgb --config configs/train/b1_xgb_d1.yaml``

B1's training entry point (spec 005). A separate, small module rather than a
branch inside :mod:`adl_etc.training.run`: :class:`XGBBaseline` is not an
``nn.Module``, needs no :class:`~adl_etc.data.features.Standardizer` (its
features come straight from ``prefix_flowstats``, computed from the raw PPI),
and trains one booster per K directly rather than through
:func:`adl_etc.training.loop.fit`'s epoch loop, so none of that module's
machinery applies. ``run_name_for``/``run_dir_for``/``is_finished`` are still
shared (both are generic over the config's ``stage``/``model.name``/
``variant``/``split_name``/``results_root`` fields).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from adl_etc.evaluation.artifact import DEFAULT_EVAL_CAP, EvalArrays, subsample_for_eval
from adl_etc.evaluation.protocol import load_split
from adl_etc.evaluation.report import build_report
from adl_etc.models.baselines.xgb import XGBBaseline
from adl_etc.training.datasets import ArrayData
from adl_etc.training.labels import LabelSpace
from adl_etc.training.run import is_finished, run_dir_for, run_name_for
from adl_etc.utils.atomic import atomic_write_json
from adl_etc.utils.config import load_config
from adl_etc.utils.runinfo import RunInfo
from adl_etc.utils.tracking import FINISHED, RUN_JSON, Tracker

STATE_JSON = "state.json"


def load_train_val(cfg: DictConfig) -> tuple[ArrayData, ArrayData, LabelSpace]:
    loaded = load_split(cfg.split, root=cfg.data_root)
    try:
        train_name, val_name = str(cfg.train_split), str(cfg.val_split)
        label_map = loaded.shard_sets[train_name][0].meta["label_map"]
        label_space = LabelSpace.from_label_map(label_map)
        train_data = ArrayData.from_shards(
            loaded.shard_sets[train_name], loaded.masks[train_name]
        )
        val_data = ArrayData.from_shards(loaded.shard_sets[val_name], loaded.masks[val_name])
    finally:
        loaded.close()
    return train_data, val_data, label_space


def _evaluate_after_training(
    run_dir: Path,
    model: XGBBaseline,
    val: ArrayData,
    label_space: LabelSpace,
    *,
    seed: int,
    eval_cap: int,
) -> None:
    out_dir = run_dir / "eval" / "val"
    if (out_dir / "report.json").is_file():
        return

    idx = subsample_for_eval(len(val), cap=eval_cap, seed=seed)
    sub = ArrayData(ppi=val.ppi[idx], ppi_len=val.ppi_len[idx], label=val.label[idx])
    dense = model.predict_dense(sub)
    labels_model = label_space.to_model(sub.label).astype(np.int64)

    ea = EvalArrays(
        dense=dense,
        labels=labels_model,
        ppi_len=sub.ppi_len.astype(np.int64),
        flow_index=idx.astype(np.int64),
    )
    ea.save(out_dir)
    report = build_report(
        dense, labels_model, sub.ppi_len, n_classes=label_space.n_classes, split="val", seed=seed
    )
    report.save(out_dir / "report.json")


def run_training(
    cfg: DictConfig,
    *,
    seed: int,
    code_root: str | Path | None = None,
    tracker_enabled: bool = True,
    evaluate_after: bool = True,
    eval_cap: int = DEFAULT_EVAL_CAP,
) -> dict[int, dict[str, float]] | None:
    """Train B1 for one seed. Returns ``None`` (no data loaded, no GPU/CPU
    spent) if ``run_dir`` already holds a finished attempt -- XGBoost has no
    partial-K resume, so "resume" here is all-or-nothing, unlike the torch
    baselines' epoch-level resume."""
    run_dir = run_dir_for(cfg, seed)
    run_name = run_name_for(cfg, seed)

    if is_finished(run_dir):
        return None

    train_data, val_data, label_space = load_train_val(cfg)
    model = XGBBaseline.from_config(cfg.model, label_space, seed=seed)
    run_info = RunInfo.collect(run_name, seed, cfg, code_root=code_root)

    run_dir.mkdir(parents=True, exist_ok=True)
    label_space.save(run_dir / "label_space.json")

    resuming = (run_dir / RUN_JSON).is_file()
    tracker = Tracker.start(run_info, cfg, run_dir, resume=resuming, enabled=tracker_enabled)
    with tracker:
        summary = model.fit(train_data, val_data, tracker=tracker)
        model.save(run_dir / "model")
        best_f1 = max(v["val_macro_f1"] for v in summary.values())
        tracker.log_metrics({"best_val_macro_f1": best_f1})
        tracker.close(FINISHED)

    # "finished", not Tracker's "FINISHED": is_finished() (reused from
    # training.run) checks training.loop's own lowercase status vocabulary --
    # a distinct thing from Tracker's, exactly the mismatch make_tables.py's
    # first draft hit for the same reason (see its own history for the story).
    atomic_write_json(
        run_dir / STATE_JSON,
        {"status": "finished", "summary": {str(k): v for k, v in summary.items()}},
    )

    if evaluate_after:
        _evaluate_after_training(
            run_dir, model, val_data, label_space, seed=seed, eval_cap=eval_cap
        )

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--override", nargs="*", default=(), help="dotted key=value overrides")
    parser.add_argument("--seed", type=int, default=None, help="overrides the config's seed")
    parser.add_argument("--code-root", type=Path, default=None)
    parser.add_argument("--no-tracker", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.override)
    seed = args.seed if args.seed is not None else int(cfg.seed)

    summary = run_training(
        cfg, seed=seed, code_root=args.code_root, tracker_enabled=not args.no_tracker
    )
    run_dir = run_dir_for(cfg, seed)
    if summary is None:
        print(f"{run_name_for(cfg, seed)}: already finished -> {run_dir}")
    else:
        best = max(v["val_macro_f1"] for v in summary.values())
        print(f"{run_name_for(cfg, seed)}: finished, best_val_macro_f1={best:.4f} -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
