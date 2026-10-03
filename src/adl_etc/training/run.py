"""``python -m adl_etc.training.run --config configs/train/<name>.yaml``

Spec 015's training entry point (plan T8): composes a config, loads a real
split, builds a baseline model, and calls the shared loop
(:func:`adl_etc.training.loop.fit`). :mod:`kernel.kernel` (the Kaggle kernel)
imports :func:`run_training` directly and drives it from a queue of configs
under one shared wall-clock deadline; this file's own CLI runs exactly one
config, for local use and tests.

**Closed-set, by scope decision.** Every class in a split's ``label_map`` is
trained as a known class (``LabelSpace.from_label_map`` with no ``unknown``
held out). Spec 004's 150-known/30-unknown open-set draw
(``evaluation.unknown_split``) is a real, built module, but no committed split
file uses it yet, and wiring it into training would also mean *filtering*
held-out classes out of the training shards (not just relabelling them,
which ``FlowBatches(train=True)`` already refuses to do -- see its docstring).
That is future work, recorded in the phase-2 plan rather than done silently
here.

Unlike ``evaluation.run``/``evaluation.plotstyle``, this module is not walked
by ``tests/test_import_boundary.py`` (``adl_etc.training`` may import torch
freely, per CLAUDE.md), so imports are ordinary module-level ones.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from adl_etc.data.features import Standardizer
from adl_etc.evaluation.artifact import DEFAULT_EVAL_CAP, EvalArrays, subsample_for_eval
from adl_etc.evaluation.protocol import load_split
from adl_etc.evaluation.report import build_report
from adl_etc.models.baselines import build_model
from adl_etc.models.baselines.predict import predict_dense
from adl_etc.training.datasets import ArrayData, FlowBatches
from adl_etc.training.labels import LabelSpace
from adl_etc.training.loop import TrainResult, TrainSettings, fit
from adl_etc.utils.config import load_config
from adl_etc.utils.runinfo import RunInfo, make_run_name
from adl_etc.utils.tracking import FINISHED, PAUSED, RUN_JSON, Tracker

STATE_JSON = "state.json"
_DONE_STATUSES = ("finished", "early_stopped")  # adl_etc.training.loop's own status strings


def run_name_for(cfg: DictConfig, seed: int) -> str:
    return make_run_name(
        str(cfg.stage), str(cfg.model.name), str(cfg.variant), str(cfg.split_name), seed
    )


def run_dir_for(cfg: DictConfig, seed: int) -> Path:
    return Path(str(cfg.results_root)) / run_name_for(cfg, seed)


def is_finished(run_dir: Path) -> bool:
    """Peeks at ``state.json`` without loading any data, so a queue can skip a
    completed entry cheaply (plan T8's resume test)."""
    state_path = run_dir / STATE_JSON
    if not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return state.get("status") in _DONE_STATUSES


def _time_guard(
    deadline: float | None, *, safety: float = 1.2
) -> Callable[[int, float], bool] | None:
    """``fit``'s ``pause_check`` hook: after an epoch, assume the next one
    costs about what the last one did (spec 015's own estimation rule) and
    pause if that would run past ``deadline`` (a ``time.monotonic()`` value).
    ``safety`` leaves margin for the next epoch running slower and for the
    checkpoint write itself."""
    if deadline is None:
        return None

    def check(_epochs_done: int, last_epoch_seconds: float) -> bool:
        remaining = deadline - time.monotonic()
        return remaining < last_epoch_seconds * safety

    return check


def load_train_val(cfg: DictConfig) -> tuple[FlowBatches, FlowBatches, LabelSpace, Standardizer]:
    """Real shards for ``cfg.train_split``/``cfg.val_split``, from ``cfg.split``."""
    standardizer = Standardizer.load(cfg.standardizer)
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
    train = FlowBatches(train_data, label_space, standardizer=standardizer, train=True)
    val = FlowBatches(val_data, label_space, standardizer=standardizer)
    return train, val, label_space, standardizer


def _evaluate_after_training(
    run_dir: Path,
    model,
    val: FlowBatches,
    label_space: LabelSpace,
    standardizer: Standardizer,
    *,
    seed: int,
    device: str,
    eval_cap: int,
) -> None:
    """A real report from the best checkpoint on (a cap of) the val split, so a
    finished run is interpretable without a separate ``evaluation.run`` call.
    Skipped if a report is already there (a resumed, already-finished run)."""
    from adl_etc.training.checkpoint import load_checkpoint

    out_dir = run_dir / "eval" / "val"
    if (out_dir / "report.json").is_file():
        return

    ckpt = load_checkpoint(run_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])

    idx = subsample_for_eval(len(val.data), cap=eval_cap, seed=seed)
    sub = ArrayData(
        ppi=val.data.ppi[idx],
        ppi_len=val.data.ppi_len[idx],
        label=val.data.label[idx],
        session_id=None if val.data.session_id is None else val.data.session_id[idx],
    )
    dense = predict_dense(model, sub, label_space, standardizer, device=device)
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
    deadline: float | None = None,
    code_root: str | Path | None = None,
    tracker_enabled: bool = True,
    evaluate_after: bool = True,
    eval_cap: int = DEFAULT_EVAL_CAP,
) -> TrainResult:
    """Train one config for one seed, resuming ``run_dir`` if it already holds
    a partial attempt. Writes ``label_space.json`` at the start and, once
    training reaches ``finished``/``early_stopped`` (never on ``paused``), a
    real evaluation report under ``eval/val/``."""
    run_dir = run_dir_for(cfg, seed)
    run_name = run_name_for(cfg, seed)

    train, val, label_space, standardizer = load_train_val(cfg)
    model = build_model(cfg.model, label_space.n_classes)
    settings = TrainSettings.from_cfg(cfg.train)
    run_info = RunInfo.collect(run_name, seed, cfg, code_root=code_root)

    run_dir.mkdir(parents=True, exist_ok=True)
    label_space.save(run_dir / "label_space.json")

    resuming = (run_dir / RUN_JSON).is_file()
    tracker = Tracker.start(run_info, cfg, run_dir, resume=resuming, enabled=tracker_enabled)
    with tracker:
        result = fit(
            model,
            train,
            val,
            settings=settings,
            run_dir=run_dir,
            run_info=run_info,
            device=str(cfg.device),
            tracker=tracker,
            resume=True,
            pause_check=_time_guard(deadline),
        )
        tracker.close(PAUSED if result.status == "paused" else FINISHED)

    if evaluate_after and result.status != "paused":
        _evaluate_after_training(
            run_dir,
            model,
            val,
            label_space,
            standardizer,
            seed=seed,
            device=str(cfg.device),
            eval_cap=eval_cap,
        )

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--override", nargs="*", default=(), help="dotted key=value overrides")
    parser.add_argument("--seed", type=int, default=None, help="overrides the config's seed")
    parser.add_argument(
        "--time-limit-seconds", type=float, default=None, help="pause_check's wall-clock guard"
    )
    parser.add_argument("--code-root", type=Path, default=None)
    parser.add_argument("--no-tracker", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=args.override)
    seed = args.seed if args.seed is not None else int(cfg.seed)
    deadline = (
        None if args.time_limit_seconds is None else time.monotonic() + args.time_limit_seconds
    )

    result = run_training(
        cfg,
        seed=seed,
        deadline=deadline,
        code_root=args.code_root,
        tracker_enabled=not args.no_tracker,
    )
    run_dir = run_dir_for(cfg, seed)
    print(
        f"{run_name_for(cfg, seed)}: {result.status} ({result.epochs_done} epochs, "
        f"best_val_macro_f1={result.best_metric:.4f}) -> {run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
