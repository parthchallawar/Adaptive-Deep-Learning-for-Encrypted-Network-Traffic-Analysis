"""Kaggle GPU kernel: trains the queued baseline configs (plan T8, spec 015).

Mounts the D1 export kernel's output (``parthchallawar/adl-export-d1``: the
weekly shard sets plus the Standardizer fit on weeks 11-26) and the project
code dataset (``scripts/kaggle_sync.sh push-code``), resolves
``configs/splits/d1_main.yaml``'s ``standardizer:`` field to the mounted copy
-- the same regex-substitution trick ``kernel/export-d1/kernel.py`` already
uses, because ``evaluation.protocol.assert_standardizer_hash_consistent``
loads that path literally -- then works through ``run_queue.yaml`` in order,
training each entry with ``adl_etc.training.run.run_training``.

**The time guard** is one wall-clock deadline (``ADL_TIME_LIMIT_SECONDS``,
default 11.5 h) shared two ways: ``training.run``'s own per-epoch pause hook
stops mid-run, after a checkpoint is safely on disk, if the next epoch would
breach it; this file's own queue loop stops *starting new work* once under 40
minutes remain (spec 015). Either way ``state.json`` is written atomically and
the next push resumes -- an already-finished queue entry is skipped without
repeating it (checked via ``state.json`` alone, no data loaded).

``ADL_TRAIN_MODE=smoke`` runs a fast, self-contained CPU path instead: about
5,000 synthetic flows with a learnable signal, a real (small) GRU, 2 epochs,
no Kaggle mount needed -- the same code (``training.run.run_training``,
``training.loop.fit``, the real ``RNNBaseline``) on fake data, so ``pytest``
can prove the whole pipeline runs before any Kaggle GPU hour is spent on it.
The default, ``queue``, is the real run.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

MODE = os.environ.get("ADL_TRAIN_MODE", "queue")  # "smoke" | "queue"
TIME_LIMIT_SECONDS = float(os.environ.get("ADL_TIME_LIMIT_SECONDS", 11.5 * 3600))
QUEUE_STOP_MARGIN_SECONDS = 40 * 60  # spec 015: stop starting new work under 40 min left

# Overridable by environment so tests/test_kernel_smoke.py can point this at a
# synthetic mirror, the same technique kernel/export-d1/kernel.py uses.
INPUT = Path(os.environ.get("ADL_KAGGLE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("ADL_KAGGLE_WORKING", "/kaggle/working"))
_DEFAULT_QUEUE = str(Path(__file__).resolve().parent / "run_queue.yaml")
QUEUE_PATH = Path(os.environ.get("ADL_QUEUE", _DEFAULT_QUEUE))

REPORT: dict[str, Any] = {"mode": MODE}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def probe_environment() -> None:
    log(f"python {platform.python_version()} on {platform.platform()}")
    for name in ("numpy", "torch", "omegaconf"):
        try:
            mod = __import__(name)
            v = getattr(mod, "__version__", "?")
            if name == "torch":
                import torch

                log(f"  torch {v}, cuda available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    log(f"  gpu: {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}")
            else:
                log(f"  {name} {v}")
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not judge
            log(f"  {name}: MISSING ({exc})")


# --- code and data discovery (mirrors kernel/export-d1/kernel.py) --------------------------


def _find_code_root() -> Path | None:
    """Looks for the code dataset (``scripts/kaggle_sync.sh push-code``) by
    ``src/adl_etc/__init__.py`` somewhere under ``INPUT`` and, if found, adds
    its ``src/`` to ``sys.path``. Returns ``None``, not an error, when nothing
    is mounted -- the smoke path's own use, see :func:`run_smoke`."""
    for pattern in ("*/src", "*/*/src", "*/*/*/src"):
        for src in sorted(INPUT.glob(pattern)):
            if (src / "adl_etc" / "__init__.py").exists():
                sys.path.insert(0, str(src))
                log(f"code: {src}")
                return src.parent
    return None


def add_code_to_path() -> Path:
    """The code dataset, required: the real Kaggle image has no other way to
    make ``adl_etc`` importable. Returns the directory that holds ``src/``
    (also where ``GIT_COMMIT``/``GIT_DIRTY`` live, read by
    ``adl_etc.utils.runinfo.code_provenance``)."""
    code_root = _find_code_root()
    if code_root is None:
        raise SystemExit("the project code dataset is not mounted (no src/adl_etc found)")
    return code_root


def find_shard_root() -> Path:
    """The directory directly holding ``cesnet-tls-year22/WEEK-2022-NN/``,
    mirroring the export kernel's own output layout (``shards/<dataset>/...``)."""
    for candidate in sorted(INPUT.rglob("cesnet-tls-year22")):
        if candidate.is_dir() and any(candidate.glob("WEEK-2022-*")):
            return candidate.parent
    raise SystemExit("no cesnet-tls-year22 shard set found under " + str(INPUT))


def resolve_split_config(code_root: Path) -> Path:
    """A copy of ``configs/splits/d1_main.yaml`` whose ``standardizer:``
    fields point at the mounted ``standardizer.json`` -- the committed value
    is a local path (``results/standardizer.json``) that does not exist on
    Kaggle, and ``assert_standardizer_hash_consistent`` loads it literally."""
    candidates = sorted(INPUT.rglob("standardizer.json"))
    if not candidates:
        raise SystemExit("no standardizer.json found under " + str(INPUT))
    std_path = candidates[0]
    spec_text = (code_root / "configs" / "splits" / "d1_main.yaml").read_text(encoding="utf-8")
    resolved = re.sub(
        r"^(\s*standardizer:).*$",
        lambda m: f"{m.group(1)} {std_path}",
        spec_text,
        flags=re.MULTILINE,
    )
    resolved_path = WORK / "d1_main.resolved.yaml"
    resolved_path.write_text(resolved, encoding="utf-8")
    return resolved_path


# --- the real run: work through run_queue.yaml ------------------------------------------------


def load_queue(path: Path) -> list[dict[str, Any]]:
    raw = OmegaConf.to_container(OmegaConf.load(path))
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: expected a list of {{config, overrides}} entries")
    return [dict(entry) for entry in raw]


def run_queue(deadline: float) -> None:
    # add_code_to_path() must run before any `adl_etc` import: on the real
    # Kaggle image nothing has installed this package, so `sys.path` only
    # gains it here. A run against a local dev install (this file's own
    # tests) cannot catch the order being wrong -- `adl_etc` is already
    # importable there regardless -- so this bug reached a real Kaggle push
    # (2026-09-23, kernel version 1) before it was caught. Real Kaggle runs
    # are the check for exactly this class of bug; see kernel/export-d1's
    # own notes for the same trap.
    code_root = add_code_to_path()

    from adl_etc.training.run import is_finished, run_dir_for, run_training
    from adl_etc.utils.config import load_config

    split_path = resolve_split_config(code_root)
    shard_root = find_shard_root()
    log(f"shards: {shard_root}")
    log(f"resolved split: {split_path}")

    entries = load_queue(QUEUE_PATH)
    log(f"queue: {len(entries)} entries from {QUEUE_PATH}")
    results: list[dict[str, Any]] = []

    for i, entry in enumerate(entries):
        cfg_path = code_root / str(entry["config"])
        overrides = [*entry.get("overrides", []), f"split={split_path}", f"data_root={shard_root}"]
        cfg = load_config(cfg_path, overrides=[str(o) for o in overrides])
        seed = int(cfg.seed)
        run_dir = run_dir_for(cfg, seed)
        tag = f"[{i + 1}/{len(entries)}] {run_dir.name}"

        if is_finished(run_dir):
            log(f"{tag}: already finished, skipping")
            results.append({"run_dir": str(run_dir), "status": "already_finished"})
            continue

        remaining = deadline - time.monotonic()
        if remaining < QUEUE_STOP_MARGIN_SECONDS:
            log(f"{tag}: only {remaining / 60:.0f} min left in the session, stopping the queue")
            break

        log(f"{tag}: starting ({remaining / 3600:.2f} h left)")
        result = run_training(cfg, seed=seed, deadline=deadline, code_root=code_root)
        log(
            f"{tag}: {result.status} after {result.epochs_done} epoch(s), "
            f"best_val_macro_f1={result.best_metric:.4f}"
        )
        results.append(
            {
                "run_dir": str(run_dir),
                "status": result.status,
                "epochs_done": result.epochs_done,
                "best_metric": result.best_metric,
            }
        )

    REPORT["queue"] = results
    (WORK / "train_report.json").write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
    log("wrote train_report.json")


# --- the smoke run: fast, self-contained, no Kaggle mount ---------------------------------

_SMOKE_CLASSES = 8


def _make_smoke_flows(n: int, seed: int):
    """A learnable synthetic flow set, self-contained (kernel.py must run
    without the test suite mounted, so this does not import from tests/)."""
    import numpy as np

    from adl_etc.data import ppi as P
    from adl_etc.training.datasets import ArrayData

    rng = np.random.default_rng(seed)
    label = rng.integers(0, _SMOKE_CLASSES, n).astype(np.int64)
    ppi = np.zeros((n, P.K_MAX, P.PPI_CHANNELS), dtype=P.PPI_DTYPE)
    ppi_len = rng.integers(3, 13, n).astype(np.int8)
    for i in range(n):
        k, c = int(ppi_len[i]), int(label[i])
        ppi[i, :k, P.SIZE_POS] = np.clip(rng.normal(100 + 120 * c, 20, k), 1, P.SIZE_MAX)
        ppi[i, :k, P.IPT_POS] = np.clip(rng.normal(5 + 8 * c, 2, k), 0, P.IPT_MAX_MS)
        ppi[i, 0, P.IPT_POS] = 0
        ppi[i, :k, P.DIR_POS] = np.where(np.arange(k) % 2 == 0, P.DIR_FWD, P.DIR_REV)
        ppi[i, :k, P.PUSH_POS] = rng.integers(0, 2, k)
    return ArrayData(ppi=ppi, ppi_len=ppi_len, label=label, session_id=np.zeros(n, np.int32))


def _write_smoke_shards(root: Path, data, *, name: str, period: str) -> None:
    import numpy as np

    from adl_etc.data import ppi as P
    from adl_etc.data.tensors import ShardWriter

    n = len(data)
    label_map = {f"class{i}": i for i in range(_SMOKE_CLASSES)}
    with ShardWriter(root, name, period, label_map=label_map, allow_unknown=True) as w:
        w.add_batch(
            {
                "ppi": data.ppi,
                "ppi_len": data.ppi_len,
                "flowstats": np.zeros((n, P.FLOWSTATS_DIM), np.float32),
                "label": data.label.astype(np.int16),
                "category": np.zeros(n, np.int8),
                "session_id": data.session_id,
                "ts": np.arange(n, dtype=np.int64),
            }
        )


def _build_smoke_split(root: Path, *, n_train: int, n_val: int, seed: int) -> tuple[Path, Path]:
    from adl_etc.data.features import Standardizer
    from adl_etc.data.tensors import ShardSet

    shard_root = root / "shards"
    _write_smoke_shards(shard_root, _make_smoke_flows(n_train, seed), name="smoke", period="train")
    _write_smoke_shards(
        shard_root, _make_smoke_flows(n_val, seed + 1), name="smoke", period="val"
    )

    train_ss = ShardSet.open(shard_root / "smoke" / "train")
    try:
        std = Standardizer.fit(train_ss)
    finally:
        train_ss.close()
    std_path = root / "standardizer.json"
    std.save(std_path)

    split_path = root / "smoke_split.yaml"
    split_path.write_text(
        "dataset: smoke\n"
        "temporal: false\n"
        "splits:\n"
        "  train:\n"
        "    periods: [train]\n"
        f"    standardizer: {std_path}\n"
        "  val:\n"
        "    periods: [val]\n"
        f"    standardizer: {std_path}\n",
        encoding="utf-8",
    )
    return split_path, shard_root


def _smoke_config(root: Path) -> Path:
    split_path, shard_root = _build_smoke_split(root, n_train=4000, n_val=1000, seed=0)
    cfg_text = (
        "model:\n"
        "  name: gru\n"
        "  stem: 32\n"
        "  hidden: 32\n"
        "  layers: 1\n"
        "  dropout: 0.1\n"
        "train:\n"
        "  epochs: 2\n"
        "  batch_size: 256\n"
        "  lr: 0.005\n"
        "  weight_decay: 0.0001\n"
        "  label_smoothing: 0.1\n"
        "  patience: 99\n"
        "  balanced_cap: null\n"
        "  amp: false\n"
        f"split: {split_path}\n"
        f"data_root: {shard_root}\n"
        f"standardizer: {shard_root.parent / 'standardizer.json'}\n"
        "train_split: train\n"
        "val_split: val\n"
        "device: cpu\n"
        f"results_root: {root / 'runs'}\n"
        "stage: bl\n"
        "variant: smoke\n"
        "split_name: smoke\n"
        "seed: 0\n"
    )
    cfg_path = root / "smoke_config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    return cfg_path


def run_smoke(*, deadline: float | None = None) -> None:
    # Best-effort, unlike run_queue's add_code_to_path(): smoke mode's whole
    # point is running with no Kaggle mount at all (a local dev install, or
    # these tests, already makes `adl_etc` importable), but if it *is*
    # pushed to Kaggle for real (kernel-metadata.json always mounts the code
    # dataset, regardless of mode), this still finds it rather than relying
    # on sys.path already being right.
    _find_code_root()

    from adl_etc.training.run import run_training
    from adl_etc.utils.config import load_config

    cfg_path = _smoke_config(WORK / "smoke")
    cfg = load_config(cfg_path)

    t0 = time.perf_counter()
    result = run_training(cfg, seed=int(cfg.seed), deadline=deadline)
    wall = time.perf_counter() - t0
    REPORT["smoke"] = {
        "status": result.status,
        "epochs_done": result.epochs_done,
        "best_val_macro_f1": result.best_metric,
        "wall_seconds": round(wall, 1),
    }
    log(f"smoke: {result.status} in {wall:.1f}s, best_val_macro_f1={result.best_metric:.4f}")
    (WORK / "train_report.json").write_text(json.dumps(REPORT, indent=2), encoding="utf-8")


def main() -> None:
    probe_environment()
    WORK.mkdir(parents=True, exist_ok=True)
    if MODE == "smoke":
        run_smoke()
    else:
        run_queue(time.monotonic() + TIME_LIMIT_SECONDS)


if __name__ == "__main__":
    main()
