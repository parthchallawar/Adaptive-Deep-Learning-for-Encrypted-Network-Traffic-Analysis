"""``python -m adl_etc.training.run --config configs/train/<name>.yaml``

Spec 015's training entry point (plan T8): composes a config, loads a real
split, builds a baseline model, and calls the shared loop
(:func:`adl_etc.training.loop.fit`). :mod:`kernel.kernel` (the Kaggle kernel)
imports :func:`run_training` directly and drives it from a queue of configs
under one shared wall-clock deadline; this file's own CLI runs exactly one
config, for local use and tests.

**Open-set when the split says so** (plan phase 3, T1a). A split YAML with
``unknown_classes`` (``configs/splits/d1_open.yaml``: spec 004's 150 known / 30
unknown draw) has those classes' flows dropped from its training split by
``protocol.load_split`` itself, and the label space holds them out, so they reach
validation as ``-1`` (unknown) and are never trained on. A split without it
(``d1_main.yaml``, phase 2's closed-set runs) trains every class, as before.

**Low-label regimes** (spec 008): ``label_fraction: 0.01`` keeps a seeded uniform
1% of the training flows (seeded by the run's seed, so each seed sees its own
subset); the label space does not change, so a class with no labelled flows is
simply never predicted and the 1%/10%/100% numbers measure the same task.

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
from omegaconf import DictConfig, OmegaConf

from adl_etc.data.features import Standardizer
from adl_etc.evaluation.artifact import DEFAULT_EVAL_CAP, EvalArrays, subsample_for_eval
from adl_etc.evaluation.calibration import fit_calibration
from adl_etc.evaluation.dense_logits import from_causal
from adl_etc.evaluation.protocol import load_split
from adl_etc.evaluation.report import build_report
from adl_etc.models.baselines.predict import predict_dense, predict_heads
from adl_etc.models.registry import build_model
from adl_etc.training.datasets import ArrayData, FlowBatches
from adl_etc.training.labels import LabelSpace
from adl_etc.training.loop import TrainResult, TrainSettings, fit
from adl_etc.training.objectives import PrefixObjective, build_objective
from adl_etc.training.ssl import AugmentSettings, MPMObjective, SSLBatches, SSLObjective
from adl_etc.utils.config import load_config
from adl_etc.utils.provenance import file_sha256
from adl_etc.utils.runinfo import RunInfo, make_run_name
from adl_etc.utils.seeding import seed_everything, seeded_generator
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
        label_space = LabelSpace.from_label_map(label_map, unknown=loaded.unknown_classes)
        train_data = ArrayData.from_shards(
            loaded.shard_sets[train_name], loaded.masks[train_name]
        )
        val_data = ArrayData.from_shards(loaded.shard_sets[val_name], loaded.masks[val_name])
    finally:
        loaded.close()
    train = FlowBatches(train_data, label_space, standardizer=standardizer, train=True)
    val = FlowBatches(val_data, label_space, standardizer=standardizer)
    return train, val, label_space, standardizer


def _declares(cfg: DictConfig, key: str) -> bool:
    """True if ``cfg`` names ``key`` at all, including as a missing (``???``) value,
    which ``key in cfg`` reports as absent."""
    try:
        return key in OmegaConf.to_container(cfg, resolve=False)  # type: ignore[operator]
    except Exception:
        return False


def label_subset(data: ArrayData, fraction: float, seed: int) -> ArrayData:
    """A seeded uniform ``fraction`` of ``data`` (spec 008's low-label regimes),
    original order kept; at least one flow."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"label_fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return data
    n = max(1, round(fraction * len(data)))
    return data.subsample(n, seeded_generator(seed, "label-fraction"))


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
    calibrate: bool = False,
    class_counts: np.ndarray | None = None,
    balanced_cap: float | None = None,
) -> None:
    """A real report from the best checkpoint on (a cap of) the val split, so a
    finished run is interpretable without a separate ``evaluation.run`` call.
    Skipped if a report is already there (a resumed, already-finished run).

    With ``calibrate`` (a prefix-trained PAT, spec 008), the same val pass also
    fits the per-K calibration and writes it as ``calibration.json`` beside
    ``best.pt``, bound to that file's SHA-256; ``eval/val/safe.npy`` keeps the
    ``safe`` logits for offline policy work (spec 009)."""
    from adl_etc.training.checkpoint import inference_state_dict, load_checkpoint

    out_dir = run_dir / "eval" / "val"
    if (out_dir / "report.json").is_file():
        return

    ckpt = load_checkpoint(run_dir / "best.pt", map_location="cpu")
    model.load_state_dict(inference_state_dict(ckpt))

    idx = subsample_for_eval(len(val.data), cap=eval_cap, seed=seed)
    sub = ArrayData(
        ppi=val.data.ppi[idx],
        ppi_len=val.data.ppi_len[idx],
        label=val.data.label[idx],
        session_id=None if val.data.session_id is None else val.data.session_id[idx],
    )
    labels_model = label_space.to_model(sub.label).astype(np.int64)
    safe = None
    if calibrate:
        heads = predict_heads(model, sub, label_space, standardizer, device=device)
        dense = from_causal(heads["cls"])
        safe = heads["safe"]
    else:
        dense = predict_dense(model, sub, label_space, standardizer, device=device)

    ea = EvalArrays(
        dense=dense,
        labels=labels_model,
        ppi_len=sub.ppi_len.astype(np.int64),
        flow_index=idx.astype(np.int64),
    )
    ea.save(out_dir)
    if safe is not None:
        np.save(out_dir / "safe.npy", safe)
        assert class_counts is not None
        cal = fit_calibration(
            dense.logits,
            safe,
            labels_model,
            sub.ppi_len,
            class_counts=class_counts,
            cap=balanced_cap,
            checkpoint_hash=file_sha256(run_dir / "best.pt"),
        )
        cal.save(run_dir / "calibration.json")
    report = build_report(
        dense, labels_model, sub.ppi_len, n_classes=label_space.n_classes, split="val", seed=seed
    )
    report.save(out_dir / "report.json")


def _probe_after_pretraining(
    run_dir: Path, model, train: FlowBatches, val: FlowBatches, *, seed: int, device: str
) -> None:
    """A pretrained model's classifier head is untrained, so instead of a
    classification report it gets spec 007's linear probes on frozen ``z_K``
    (``eval/val/probe.json``). Skipped if already written."""
    import json

    from adl_etc.training.checkpoint import inference_state_dict, load_checkpoint
    from adl_etc.training.probes import linear_probe

    out = run_dir / "eval" / "val" / "probe.json"
    if out.is_file():
        return
    model.load_state_dict(inference_state_dict(load_checkpoint(run_dir / "best.pt")))
    plain_val = FlowBatches(
        val.data, val.label_space, standardizer=val.standardizer, views=val.views
    )
    result = linear_probe(model, train, plain_val, seed=seed, device=device)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")


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
    fraction = float(cfg.get("label_fraction", 1.0))
    if fraction != 1.0:
        train = FlowBatches(
            label_subset(train.data, fraction, seed), label_space,
            standardizer=standardizer, views=train.views, train=True,
        )  # fmt: skip
    # Seed *before* building the model: its initial weights must depend on the seed
    # alone, not on whatever else ran in this process first (a Kaggle queue trains
    # several seeds in one process; found in plan phase 3, T5).
    seed_everything(seed)
    model = build_model(cfg.model, label_space.n_classes)
    views = tuple(getattr(model, "views", ("continuous",)))
    if views != train.views:  # the PAT reads tokens (and the continuous side channel)
        train = FlowBatches(
            train.data, label_space, standardizer=standardizer, views=views, train=True
        )
        val = FlowBatches(val.data, label_space, standardizer=standardizer, views=views)
    settings = TrainSettings.from_cfg(cfg.train)
    objective = build_objective(
        cfg.get("objective"), model=model, n_classes=label_space.n_classes, settings=settings
    )
    plain_train = train
    augment_ft = cfg.get("finetune_augment")
    if isinstance(objective, PrefixObjective) and augment_ft:
        from adl_etc.training.finetune import AugmentedBatches

        train = AugmentedBatches(
            train.data, label_space, standardizer=standardizer, views=views, train=True,
            seed=seed, drop_p=float(augment_ft.drop_p), ipt_s=float(augment_ft.ipt_s),
        )  # fmt: skip
    if isinstance(objective, SSLObjective | MPMObjective):
        # Spec 007: the views are built per batch, seeded so a resume is exact; the
        # validation views are fixed (seed 0) so every epoch is scored on the same ones.
        mode = "mpm" if isinstance(objective, MPMObjective) else "npp_pfc"
        augment = AugmentSettings(**dict(cfg.get("augment") or {}))
        train = SSLBatches(
            train.data, label_space, standardizer=standardizer, views=views, train=True,
            mode=mode, seed=seed, augment=augment,
        )  # fmt: skip
        val = SSLBatches(
            val.data, label_space, standardizer=standardizer, views=views,
            mode=mode, seed=0, augment=augment,
        )  # fmt: skip
    run_info = RunInfo.collect(run_name, seed, cfg, code_root=code_root)

    run_dir.mkdir(parents=True, exist_ok=True)
    label_space.save(run_dir / "label_space.json")

    # Fine-tuning from a pretrained PAT (spec 008). Only on a fresh start: a resume
    # continues from its own checkpoint, so the pretrained file need not even exist then.
    init_info: dict[str, object] = {}
    if _declares(cfg, "init_from") and OmegaConf.is_missing(cfg, "init_from"):
        # `init_from: ???` in a config means "required": cfg.get() would quietly return
        # None and the run would train from scratch under an SSL run name.
        raise ValueError(
            "this config needs `init_from` (a pretrained checkpoint) and none was given"
        )
    init_from = cfg.get("init_from")
    if init_from and not (run_dir / STATE_JSON).is_file():
        from adl_etc.training.finetune import load_pretrained

        init_info = load_pretrained(model, str(init_from))

    resuming = (run_dir / RUN_JSON).is_file()
    tracker = Tracker.start(run_info, cfg, run_dir, resume=resuming, enabled=tracker_enabled)
    with tracker:
        tracker.log_params(
            {"n_train_flows": len(train), "label_fraction": fraction,
             "n_unknown_classes": len(label_space.unknown)}
        )  # fmt: skip
        if init_info:
            tracker.log_params(init_info)
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
            objective=objective,
        )
        tracker.close(PAUSED if result.status == "paused" else FINISHED)

    if evaluate_after and result.status != "paused" and not objective.uses_labels:
        _probe_after_pretraining(
            run_dir, model, plain_train, val, seed=seed, device=str(cfg.device)
        )
    elif evaluate_after and result.status != "paused":
        _evaluate_after_training(
            run_dir,
            model,
            val,
            label_space,
            standardizer,
            seed=seed,
            device=str(cfg.device),
            eval_cap=eval_cap,
            calibrate=isinstance(objective, PrefixObjective),
            class_counts=train.class_counts(),
            balanced_cap=settings.balanced_cap,
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
