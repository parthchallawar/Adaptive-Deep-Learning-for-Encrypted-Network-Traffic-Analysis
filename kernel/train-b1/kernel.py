"""Kaggle GPU kernel: trains B1 (XGBoost) on the real D1 export, 3 seeds
(spec 005, plan T8-follow-up).

Mounts the D1 export kernel's output (``kernel_sources``:
``parthchallawar/adl-export-d1``) and the project code dataset, resolves
``configs/splits/d1_main.yaml``'s ``standardizer:`` field to the mounted copy
-- the same trick ``kernel/kernel.py`` and ``kernel/export-d1/kernel.py``
both use, because ``assert_standardizer_hash_consistent`` loads that path
literally, even though B1 itself reads no Standardizer (its features come
straight from ``prefix_flowstats``) -- then trains seeds 0, 1, 2 in order via
``adl_etc.training.run_xgb.run_training``, with ``model.device=cuda`` forced
below so XGBoost's own histogram builder runs on the GPU instead of Kaggle's
4-core free CPU (a real run on CPU alone ran past 10 hours without finishing
even one seed and was cancelled; GPU histogram building is the fix, not a
smaller grid or fewer trees).

XGBoost has no epoch-level pause/resume -- an already-finished seed is
skipped (``is_finished``, no data loaded); one not yet started trains in full
within this push, expected to be fast enough on GPU not to need the time
guard the GPU *neural*-baseline kernel (``kernel/kernel.py``) needs.
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

MODE = os.environ.get("ADL_TRAIN_MODE", "queue")  # kept for parity with kernel/kernel.py
INPUT = Path(os.environ.get("ADL_KAGGLE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("ADL_KAGGLE_WORKING", "/kaggle/working"))
SEEDS = (0, 1, 2)
CONFIG = "configs/train/b1_xgb_d1.yaml"
DEVICE = "cuda"  # tests override this to "cpu" -- no GPU on a CI runner

REPORT: dict[str, Any] = {"mode": MODE}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def probe_environment() -> None:
    log(f"python {platform.python_version()} on {platform.platform()}")
    for name in ("numpy", "xgboost", "omegaconf"):
        try:
            mod = __import__(name)
            log(f"  {name} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not judge
            log(f"  {name}: MISSING ({exc})")
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        log(f"  gpu: {out.stdout.strip() or out.stderr.strip() or 'nvidia-smi returned nothing'}")
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not judge
        log(f"  gpu: nvidia-smi unavailable ({exc})")
    log_input_tree()


def log_input_tree(max_depth: int = 4, max_entries: int = 300) -> None:
    """What's actually mounted under ``INPUT`` -- see ``kernel/kernel.py``'s
    own note on why this is logged unconditionally rather than assumed."""
    if not INPUT.exists():
        log(f"{INPUT} does not exist")
        return
    entries: list[str] = []
    truncated = False
    for p in sorted(INPUT.rglob("*")):
        try:
            depth = len(p.relative_to(INPUT).parts)
        except ValueError:
            continue
        if depth > max_depth:
            continue
        entries.append(str(p.relative_to(INPUT)) + ("/" if p.is_dir() else ""))
        if len(entries) >= max_entries:
            truncated = True
            break
    log(f"under {INPUT}, depth<={max_depth}{' (truncated)' if truncated else ''}:")
    for e in entries:
        log(f"  {e}")


def add_code_to_path() -> Path:
    for pattern in ("*/src", "*/*/src", "*/*/*/src"):
        for src in sorted(INPUT.glob(pattern)):
            if (src / "adl_etc" / "__init__.py").exists():
                sys.path.insert(0, str(src))
                log(f"code: {src}")
                return src.parent
    raise SystemExit("the project code dataset is not mounted (no src/adl_etc found)")


def find_shard_root() -> Path:
    for candidate in sorted(INPUT.rglob("cesnet-tls-year22")):
        if candidate.is_dir() and any(candidate.glob("WEEK-2022-*")):
            return candidate.parent
    raise SystemExit("no cesnet-tls-year22 shard set found under " + str(INPUT))


def resolve_split_config(code_root: Path) -> tuple[Path, Path]:
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
    return resolved_path, std_path


def main() -> None:
    probe_environment()
    WORK.mkdir(parents=True, exist_ok=True)
    code_root = add_code_to_path()

    from adl_etc.training.run_xgb import is_finished, run_dir_for, run_training
    from adl_etc.utils.config import load_config

    split_path, std_path = resolve_split_config(code_root)
    shard_root = find_shard_root()
    log(f"shards: {shard_root}")
    log(f"resolved split: {split_path}")
    log(f"standardizer: {std_path}")

    results: list[dict[str, Any]] = []
    for i, seed in enumerate(SEEDS):
        overrides = [
            f"seed={seed}",
            f"split={split_path}",
            f"data_root={shard_root}",
            f"standardizer={std_path}",
            f"model.device={DEVICE}",
        ]
        cfg = load_config(code_root / CONFIG, overrides=overrides)
        run_dir = run_dir_for(cfg, seed)
        tag = f"[{i + 1}/{len(SEEDS)}] {run_dir.name}"

        if is_finished(run_dir):
            log(f"{tag}: already finished, skipping")
            results.append({"run_dir": str(run_dir), "status": "already_finished"})
            continue

        log(f"{tag}: starting")
        t0 = time.perf_counter()
        summary = run_training(cfg, seed=seed, code_root=code_root)
        wall = time.perf_counter() - t0
        best = max(v["val_macro_f1"] for v in summary.values()) if summary else None
        log(f"{tag}: finished in {wall:.1f}s, best_val_macro_f1={best}")
        results.append(
            {
                "run_dir": str(run_dir),
                "status": "finished",
                "wall_seconds": round(wall, 1),
                "best_val_macro_f1": best,
            }
        )

    REPORT["seeds"] = results
    (WORK / "train_b1_report.json").write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
    log("wrote train_b1_report.json")


if __name__ == "__main__":
    main()
