"""Kaggle CPU kernel: CESNET-TLS-Year22 weeks 11-52 -> 10%-sampled shard sets.

Phase-2 plan T3. Mounts the mirror (`pranjalkar99/cesnet-22`) and the project code
dataset, exports with ``--sample-rate 0.10`` while checking every day against its
own ``stats-*.json`` (before sampling), then fits the Standardizer on the training
weeks and loads ``configs/splits/d1_main.yaml`` for real, so the four leakage rules
are asserted against the actual corpus. Needs no GPU quota.

``MODE = "probe"`` does the same on three days to answer the cheap questions first
(does the mount look like we expect, is there internet, how fast is it, how big is a
week); ``MODE = "full"`` is the whole 42-week run. Outputs (under /kaggle/working):

    shards/cesnet-tls-year22/WEEK-2022-NN/   the shard sets
    standardizer.json                        fit on weeks 11-26 only
    export_report.json                       counts, timings, rare-class floor, checks
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
import urllib.request
from pathlib import Path

MODE = os.environ.get("ADL_EXPORT_MODE", "probe")  # "probe" | "full"

SAMPLE_RATE = 0.10
SAMPLE_SEED = 0
FIRST_WEEK, LAST_WEEK = 11, 52
TRAIN_WEEKS = range(11, 27)  # spec 004: the standardizer is fit on these only
TEST_FLOOR = 100  # spec 004: minimum test flows for a class to be scored
CHUNKSIZE = 100_000

# The environment overrides exist so tests/test_kernel_export_d1.py can run this file
# against a synthetic mirror; on Kaggle they are unset.
INPUT = Path(os.environ.get("ADL_KAGGLE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("ADL_KAGGLE_WORKING", "/kaggle/working"))
OUT = WORK / ("shards" if MODE == "full" else "probe-shards")
DATASET = "cesnet-tls-year22"

REPORT: dict = {"mode": MODE, "sample_rate": SAMPLE_RATE, "sample_seed": SAMPLE_SEED}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def probe_environment() -> None:
    log(f"python {platform.python_version()} on {platform.platform()}")
    for name in ("numpy", "pandas", "pyarrow", "omegaconf"):
        try:
            mod = __import__(name)
            log(f"  {name} {mod.__version__}")
        except Exception as exc:  # noqa: BLE001 - a probe reports, it does not judge
            log(f"  {name}: MISSING ({exc})")
    log(f"cpus: {os.cpu_count()}")
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        log(f"ram: {mem:.1f} GiB")
    except (ValueError, OSError, AttributeError):
        pass
    du = shutil.disk_usage(WORK)
    log(f"/kaggle/working: {du.free / 2**30:.1f} GiB free of {du.total / 2**30:.1f}")
    try:
        with urllib.request.urlopen("https://pypi.org/simple/", timeout=10) as resp:
            REPORT["internet"] = f"yes (HTTP {resp.status})"
    except Exception as exc:  # noqa: BLE001
        REPORT["internet"] = f"no ({type(exc).__name__}: {exc})"
    log(f"internet: {REPORT['internet']}")
    log(f"mounted under {INPUT}: " + ", ".join(sorted(p.name for p in INPUT.iterdir())))


def add_code_to_path() -> Path:
    """The code dataset is ``src/`` + ``configs/`` + provenance sidecars. Returns the
    directory that holds ``src/``."""
    for pattern in ("*/src", "*/*/src", "*/*/*/src"):
        for src in sorted(INPUT.glob(pattern)):
            if (src / "adl_etc" / "__init__.py").exists():
                sys.path.insert(0, str(src))
                log(f"code: {src}")
                return src.parent
    raise SystemExit("the project code dataset is not mounted (no src/adl_etc found)")


def find_mirror_files(cesnet_csv) -> list[Path]:
    """Every ``flows-*.csv.xz`` of weeks FIRST_WEEK..LAST_WEEK, in date order."""
    found = sorted(INPUT.rglob("flows-*.csv.xz"))
    log(f"mirror: {len(found)} flows files under {INPUT}")
    picked, missing_stats = [], []
    for path in found:
        week = cesnet_csv.iso_week(cesnet_csv.date_from_filename(path))
        if week.startswith("WEEK-2022-") and FIRST_WEEK <= int(week[-2:]) <= LAST_WEEK:
            picked.append(path)
            if not cesnet_csv._stats_path_for(path).exists():
                missing_stats.append(path.name)
    picked.sort(key=lambda p: p.name)
    weeks: dict[str, int] = {}
    for p in picked:
        w = cesnet_csv.iso_week(cesnet_csv.date_from_filename(p))
        weeks[w] = weeks.get(w, 0) + 1
    REPORT["mirror_files_selected"] = len(picked)
    REPORT["days_per_week"] = weeks
    REPORT["weeks_with_fewer_than_7_days"] = {w: n for w, n in weeks.items() if n != 7}
    REPORT["days_missing_stats"] = missing_stats
    log(f"weeks {FIRST_WEEK}-{LAST_WEEK}: {len(picked)} day files across {len(weeks)} weeks")
    if missing_stats:
        raise SystemExit(f"{len(missing_stats)} day(s) have no stats file: {missing_stats[:5]}")
    return picked


def choose_probe_days(files: list[Path]) -> list[Path]:
    by_name = {p.name: p for p in files}
    wanted = ["flows-20220314.csv.xz", "flows-20220615.csv.xz", "flows-20221231.csv.xz"]
    chosen = [by_name[n] for n in wanted if n in by_name]
    return chosen or files[:2]


def dir_gib(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30


def week_counts(cesnet_csv, tensors, root: Path) -> dict[str, dict]:
    out = {}
    for wdir in sorted((root / DATASET).glob("WEEK-*")):
        with tensors.ShardSet.open(wdir) as ss:
            n_before = ss.meta["counters"]["rows_before_sampling"]
            out[wdir.name] = {
                "n_flows": ss.meta["n_flows"],
                "rows_before_sampling": n_before,
                "kept_fraction": round(ss.meta["n_flows"] / n_before, 5) if n_before else None,
                "sample_rate": ss.meta["sample_rate"],
                "gib": round(dir_gib(wdir), 3),
            }
    return out


def main() -> None:
    probe_environment()
    code_root = add_code_to_path()

    from adl_etc.data import cesnet_csv, tensors
    from adl_etc.utils.runinfo import code_provenance

    commit, dirty = code_provenance(code_root)
    REPORT["code_commit"], REPORT["code_dirty"] = commit, dirty
    log(f"code commit {commit[:12]} dirty={dirty}")

    files = find_mirror_files(cesnet_csv)
    if not files:
        raise SystemExit("no mirror files found; is pranjalkar99/cesnet-22 attached?")
    if MODE == "probe":
        files = choose_probe_days(files)
        log("probe days: " + ", ".join(p.name for p in files))

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    summary = cesnet_csv.export_dataset(
        csv_paths=files,
        dataset=DATASET,
        out_root=OUT,
        chunksize=CHUNKSIZE,
        max_flows=500_000,
        overwrite=True,
        sample_rate=SAMPLE_RATE,
        sample_seed=SAMPLE_SEED,
        verify=True,  # raises VerificationError on the first day that disagrees with its stats
        manifest_path=WORK / "no-manifest.json",  # the mirror is not in this kernel's manifest
        log=log,
    )
    wall = time.perf_counter() - t0
    REPORT["export"] = {
        "wall_seconds": round(wall, 1),
        "rows_read": summary.rows_processed,
        "rows_per_second": round(summary.rows_per_second),
        "flows_written": summary.n_flows,
        "verified_days": summary.verified_days,
        "dropped_zero_ppi": summary.dropped_zero_ppi,
    }
    REPORT["weeks"] = week_counts(cesnet_csv, tensors, OUT)
    REPORT["shards_gib"] = round(dir_gib(OUT), 3)
    log(f"shards on disk: {REPORT['shards_gib']} GiB")

    if MODE == "probe":
        n_days = len(files)
        per_day = wall / max(n_days, 1)
        REPORT["probe_extrapolation"] = {
            "seconds_per_day": round(per_day, 1),
            "full_run_hours_if_all_days_alike": round(per_day * 294 / 3600, 2),
        }
        log(f"probe: {per_day:.1f}s/day -> ~{per_day * 294 / 3600:.1f} h for 294 days")
    else:
        finish_full(cesnet_csv, tensors, code_root)

    (WORK / "export_report.json").write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
    log("wrote export_report.json")


def finish_full(cesnet_csv, tensors, code_root: Path) -> None:
    import numpy as np

    from adl_etc.data.features import Standardizer
    from adl_etc.evaluation.protocol import load_split

    bad = {
        w: c["kept_fraction"]
        for w, c in REPORT["weeks"].items()
        if c["kept_fraction"] is None or abs(c["kept_fraction"] - SAMPLE_RATE) > 0.01
    }
    REPORT["weeks_off_the_sample_rate"] = bad
    if bad:
        log(f"WARNING: weeks whose kept fraction is not {SAMPLE_RATE} +/- 0.01: {bad}")

    # -- the Standardizer, fit on the training weeks only -------------------------------
    train = [tensors.ShardSet.open(OUT / DATASET / f"WEEK-2022-{w:02d}") for w in TRAIN_WEEKS]
    try:
        std = Standardizer.fit_many(train)
    finally:
        for s in train:
            s.close()
    std_path = WORK / "standardizer.json"
    std.save(std_path)
    REPORT["standardizer_hash"] = std.hash
    log(f"standardizer fit on weeks {TRAIN_WEEKS.start}-{TRAIN_WEEKS.stop - 1}: {std.hash[:12]}")

    # -- the real split, with all four leakage rules --------------------------------------
    spec_text = (code_root / "configs" / "splits" / "d1_main.yaml").read_text(encoding="utf-8")
    resolved = spec_text.replace("standardizer: null", f"standardizer: {std_path}")
    resolved_path = WORK / "d1_main.resolved.yaml"
    resolved_path.write_text(resolved, encoding="utf-8")
    loaded = load_split(resolved_path, root=OUT)
    try:
        REPORT["split_flows"] = {
            name: int(sum(len(s) for s in sets)) for name, sets in loaded.shard_sets.items()
        }
        log(f"load_split OK (four leakage rules asserted): {REPORT['split_flows']}")

        # -- the rare-class floor, on the real test period ---------------------------------
        label_map = loaded.shard_sets["train"][0].meta["label_map"]
        names = {v: k for k, v in label_map.items()}
        test = np.concatenate([s.column("label") for s in loaded.shard_sets["test_id"]])
        counts = np.bincount(test.astype(np.int64), minlength=len(label_map))
        REPORT["n_classes"] = len(label_map)
        REPORT["test_id_flows_per_class"] = {names[i]: int(c) for i, c in enumerate(counts)}
        below = {names[i]: int(c) for i, c in enumerate(counts) if c < TEST_FLOOR}
        REPORT["classes_below_test_floor"] = below
        log(f"{len(label_map)} classes; {len(below)} below the {TEST_FLOOR}-flow test floor")
    finally:
        loaded.close()


if __name__ == "__main__":
    main()
