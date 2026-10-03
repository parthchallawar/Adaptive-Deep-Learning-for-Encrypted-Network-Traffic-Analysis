"""The supervised training loop shared by every baseline (spec 005, spec 015).

Written once so that a comparison between two models isolates the architecture
and not the recipe: AdamW, OneCycle, AMP on GPU, class-balanced sampling capped
at 10x, label smoothing, early stopping on validation macro-F1, and per-epoch
checkpoints. No DataLoader workers (spec 015).

**Resumable, and exactly so.** After every epoch a checkpoint is written and
then ``state.json``; a killed session loses at most the epoch in progress.
Every epoch's randomness, its data order and its dropout, is derived from
``(seed, epoch)`` and not carried as running RNG state, so a run that was
paused after epoch 2 and resumed produces bit-for-bit the same training as one
that was never interrupted (tested). A resume also refuses to proceed under a
different label space, standardizer or config.

A model is any ``nn.Module`` whose ``forward(batch: dict[str, Tensor])`` returns
logits ``[B, C]`` (one prediction per flow) or ``[B, K, C]`` (one after each
packet; trained with :func:`~adl_etc.training.losses.multi_prefix_ce`).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from adl_etc.evaluation.metrics import accuracy, macro_f1
from adl_etc.training import checkpoint as ckpt
from adl_etc.training.datasets import FlowBatches, balanced_weights, epoch_indices, to_torch
from adl_etc.training.losses import cross_entropy_ls, multi_prefix_ce
from adl_etc.utils.atomic import atomic_write_json
from adl_etc.utils.runinfo import RunInfo
from adl_etc.utils.seeding import seed_everything, seeded_generator
from adl_etc.utils.tracking import Tracker

STATE_JSON = "state.json"
BEST_CHECKPOINT = "best.pt"

RUNNING = "running"
FINISHED = "finished"
EARLY_STOPPED = "early_stopped"
PAUSED = "paused"
_DONE = (FINISHED, EARLY_STOPPED)


class TrainingError(RuntimeError):
    """Training cannot continue (for example a non-finite loss)."""


@dataclass(frozen=True)
class TrainSettings:
    """Spec 005's shared recipe. Defaults are the spec's; peak LR is per model
    family (3e-3 for CNN/RNN, 1e-3 for Transformers) and set by the config."""

    epochs: int = 20
    batch_size: int = 4096
    lr: float = 3e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    patience: int = 4
    balanced_cap: float | None = 10.0
    amp: bool = True
    grad_clip: float | None = None
    keep_last: int = 2
    samples_per_epoch: int | None = None
    pct_start: float = 0.3

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1 or self.keep_last < 1:
            raise ValueError("epochs, batch_size, patience and keep_last must be >= 1")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.balanced_cap is not None and self.balanced_cap < 1:
            raise ValueError(f"balanced_cap must be >= 1 or null, got {self.balanced_cap}")
        if self.samples_per_epoch is not None and self.samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be >= 1 or null")

    @classmethod
    def from_cfg(cls, section: Mapping[str, Any] | DictConfig) -> TrainSettings:
        """From a config's ``train`` section. An unknown key raises: a typo'd
        ``batchsize`` must not silently leave the default in place."""
        raw = (
            OmegaConf.to_container(section, resolve=True)
            if isinstance(section, DictConfig)
            else dict(section)
        )
        assert isinstance(raw, dict)
        kwargs: dict[str, Any] = {str(k): v for k, v in raw.items()}
        unknown = set(kwargs) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown train setting(s): {sorted(unknown)}")
        return cls(**kwargs)


@dataclass
class TrainResult:
    status: str  # finished | early_stopped | paused
    epochs_done: int
    epochs_run: int  # epochs run by *this call* (0 when resuming a completed run)
    best_epoch: int
    best_metric: float
    history: list[dict[str, float]] = field(default_factory=list)
    run_dir: Path = Path(".")

    @property
    def best_checkpoint(self) -> Path:
        return self.run_dir / BEST_CHECKPOINT


def _epoch_seed(seed: int, epoch: int) -> int:
    return int(seeded_generator(seed, f"torch-epoch-{epoch}").integers(0, 2**31 - 1))


def _loss(logits: torch.Tensor, batch: dict[str, torch.Tensor], smoothing: float) -> torch.Tensor:
    if logits.ndim == 3:
        return multi_prefix_ce(logits, batch["y"], batch["mask"], smoothing)
    return cross_entropy_ls(logits, batch["y"], smoothing)


def _final_logits(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """One row of logits per flow: the last real packet's, for per-step models."""
    if logits.ndim == 2:
        return logits
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, batch["ppi_len"] - 1]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: FlowBatches,
    *,
    batch_size: int,
    device: torch.device,
    amp: bool = False,
) -> dict[str, float]:
    """Closed-set loss, accuracy and macro-F1 at full flow length, over the
    known-class flows only (unknown-class flows have no target here)."""
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    loss_sum, n = 0.0, 0
    for _, np_batch in data.iter_batches(batch_size, known_only=True):
        batch = to_torch(np_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(batch)
        logits = logits.float()
        loss_sum += float(_loss(logits, batch, 0.0)) * len(np_batch["y"])
        n += len(np_batch["y"])
        preds.append(_final_logits(logits, batch).argmax(-1).cpu().numpy())
        targets.append(np_batch["y"])
    p, t = np.concatenate(preds), np.concatenate(targets)
    return {
        "val_loss": loss_sum / n,
        "val_acc": accuracy(p, t),
        "val_macro_f1": macro_f1(p, t),
    }


def _write_state(run_dir: Path, state: dict[str, Any]) -> None:
    atomic_write_json(run_dir / STATE_JSON, state)


def _prune(run_dir: Path, keep_last: int, protect: str) -> None:
    epoch_files = sorted(
        run_dir.glob("ckpt_epoch*.pt"), key=lambda p: int(p.stem.removeprefix("ckpt_epoch"))
    )
    for old in epoch_files[:-keep_last]:
        if old.name != protect:
            old.unlink(missing_ok=True)


def fit(
    model: torch.nn.Module,
    train: FlowBatches,
    val: FlowBatches,
    *,
    settings: TrainSettings,
    run_dir: str | Path,
    run_info: RunInfo,
    device: str = "cpu",
    tracker: Tracker | None = None,
    resume: bool = True,
    pause_check: Callable[[int, float], bool] | None = None,
) -> TrainResult:
    """Train ``model``, checkpointing every epoch, resuming from ``run_dir`` if
    it holds a previous attempt.

    ``pause_check(epochs_done, last_epoch_seconds)`` is asked after every epoch's
    checkpoint is safely on disk; returning True stops cleanly with status
    ``paused`` so a later call resumes. The Kaggle kernel's time guard (spec 015)
    is exactly this hook. ``tracker`` is optional and its lifecycle belongs to
    the caller; ``fit`` only logs to it.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if not train.train:
        raise ValueError("`train` must be built with FlowBatches(..., train=True)")
    label_hash = train.label_space.hash
    if val.label_space.hash != label_hash:
        raise ValueError("train and val use different label spaces")
    if val.standardizer_hash != train.standardizer_hash:
        raise ValueError(
            "train and val use different standardizers (spec 004 leakage rule 3: val and test "
            "must reuse the training statistics unchanged)"
        )
    if len(val.known_indices) == 0:
        raise ValueError("the validation set has no known-class flows to early-stop on")

    dev = torch.device(device)
    use_amp = settings.amp and dev.type == "cuda"
    n_classes = train.label_space.n_classes
    seed = run_info.seed
    seed_everything(seed)
    model.to(dev)

    n_epoch = settings.samples_per_epoch or len(train)
    steps_per_epoch = -(-n_epoch // settings.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=settings.lr,
        total_steps=settings.epochs * steps_per_epoch,
        pct_start=settings.pct_start,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    weights = (
        balanced_weights(train.y, n_classes, settings.balanced_cap)
        if settings.balanced_cap is not None
        else None
    )

    epochs_done, best_metric, best_epoch, bad_epochs = 0, -1.0, 0, 0
    history: list[dict[str, float]] = []
    state_path = run_dir / STATE_JSON

    if state_path.exists():
        if not resume:
            raise ckpt.CheckpointError(
                f"{run_dir} already holds a training run; pass resume=True to continue it"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        saved = ckpt.load_checkpoint(run_dir / state["last_checkpoint"], map_location=str(dev))
        ckpt.check_compatible(
            saved,
            label_space_hash=label_hash,
            standardizer_hash=train.standardizer_hash,
            config_hash=run_info.config_hash,
        )
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        epochs_done = int(saved["epoch"])
        best_metric = float(saved["best_metric"])
        best_epoch = int(saved["best_epoch"])
        bad_epochs = int(saved["bad_epochs"])
        history = [dict(h) for h in saved["history"]]
        if state["status"] in _DONE:
            return TrainResult(
                status=state["status"],
                epochs_done=epochs_done,
                epochs_run=0,
                best_epoch=best_epoch,
                best_metric=best_metric,
                history=history,
                run_dir=run_dir,
            )

    if tracker is not None:
        tracker.log_params({"checkpoint_dir": run_dir.as_posix()})

    status = RUNNING
    epochs_run = 0
    while status == RUNNING:
        epoch = epochs_done  # 0-based index of the epoch about to run
        started = time.perf_counter()
        torch.manual_seed(_epoch_seed(seed, epoch))
        order = epoch_indices(
            len(train), n_epoch, seeded_generator(seed, f"epoch-{epoch}"), weights
        )

        model.train()
        loss_sum = torch.zeros((), device=dev)
        for step in range(steps_per_epoch):
            idx = order[step * settings.batch_size : (step + 1) * settings.batch_size]
            batch = to_torch(train.batch(idx), dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(batch)
            if logits.shape[-1] != n_classes:
                raise ValueError(
                    f"the model emits {logits.shape[-1]} logits but the label space has "
                    f"{n_classes} known classes"
                )
            loss = _loss(logits.float(), batch, settings.label_smoothing)
            scaler.scale(loss).backward()
            if settings.grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.detach() * len(idx)

        train_loss = float(loss_sum) / len(order)
        if not math.isfinite(train_loss):
            raise TrainingError(f"non-finite training loss ({train_loss}) in epoch {epoch}")

        val_metrics = evaluate(model, val, batch_size=settings.batch_size, device=dev, amp=use_amp)
        seconds = time.perf_counter() - started
        epochs_done += 1
        epochs_run += 1
        row = {
            "epoch": float(epochs_done),
            "train_loss": train_loss,
            **val_metrics,
            "lr": float(opt.param_groups[0]["lr"]),
            "epoch_seconds": seconds,
            "flows_per_sec": len(order) / seconds if seconds > 0 else 0.0,
        }
        history.append(row)

        improved = val_metrics["val_macro_f1"] > best_metric + 1e-12
        if improved:
            best_metric, best_epoch, bad_epochs = val_metrics["val_macro_f1"], epochs_done, 0
        else:
            bad_epochs += 1

        payload = {
            "epoch": epochs_done,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "history": history,
            "run_info": run_info.to_dict(),
            "label_space_hash": label_hash,
            "standardizer_hash": train.standardizer_hash,
            "config_hash": run_info.config_hash,
        }
        name = f"ckpt_epoch{epochs_done}.pt"
        ckpt.save_checkpoint(run_dir / name, payload)  # checkpoint first, state second:
        if improved:
            ckpt.save_checkpoint(run_dir / BEST_CHECKPOINT, payload)

        # The epoch is now safely on disk, so this is the point to decide whether to
        # stop. (The pause hook is the Kaggle time guard, spec 015: it may take its
        # time, and a kill while it runs loses nothing.)
        if epochs_done >= settings.epochs:
            status = FINISHED
        elif bad_epochs >= settings.patience:
            status = EARLY_STOPPED
        elif pause_check is not None and pause_check(epochs_done, seconds):
            status = PAUSED

        _write_state(
            run_dir,
            {
                "epochs_done": epochs_done,
                "status": status,
                "last_checkpoint": name,
                "best_checkpoint": BEST_CHECKPOINT if best_epoch else None,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "config_hash": run_info.config_hash,
            },
        )  # a crash between the two leaves state pointing at the previous, intact checkpoint
        _prune(run_dir, settings.keep_last, protect=name)

        if tracker is not None:
            tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)

    if tracker is not None and status != PAUSED:
        tracker.log_metrics({"best_val_macro_f1": best_metric, "best_epoch": float(best_epoch)})
        tracker.log_params({"best_checkpoint": (run_dir / BEST_CHECKPOINT).as_posix()})

    return TrainResult(
        status=status,
        epochs_done=epochs_done,
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        best_metric=best_metric,
        history=history,
        run_dir=run_dir,
    )
