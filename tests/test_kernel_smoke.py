"""``kernel/kernel.py`` (plan T8, spec 015), exercised end to end: the smoke
path (no Kaggle mount needed), the real queue path against a synthetic mirror
laid out like the D1 export kernel's own output, the resume behaviour, and
the time guard. The same technique ``tests/test_kernel_export_d1.py`` uses for
the export kernel: load the real file with ``importlib``, drive it entirely
through environment variables.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Correction 7 (spec 015): the --smoke run is "part of CI" only in the sense
# that pytest covers it; there is no CI in this repo. These are real (if
# small) training runs, so they belong under `slow`, not the default suite.
pytestmark = pytest.mark.slow

from adl_etc.data.features import Standardizer  # noqa: E402
from adl_etc.data.tensors import ShardSet  # noqa: E402
from tests.training.util import make_flows  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
KERNEL = REPO / "kernel" / "kernel.py"
N_CLASSES = 6
TRAIN_WEEKS = range(11, 27)


def write_week_shard(root: Path, data, *, period: str, ts_start: int, n_classes: int) -> None:
    """Like ``tests.training.util.write_shards``, but with a real, globally
    increasing ``ts`` (the real helper restarts each shard at 0, which is fine
    for most tests but fails ``d1_main.yaml``'s ``temporal: true`` ordering
    rule once every week is loaded together)."""
    import numpy as np

    from adl_etc.data import ppi as P
    from adl_etc.data.tensors import ShardWriter

    n = len(data)
    label_map = {f"class{i}": i for i in range(n_classes)}
    with ShardWriter(
        root, "cesnet-tls-year22", period, label_map=label_map, allow_unknown=True
    ) as w:
        w.add_batch(
            {
                "ppi": data.ppi,
                "ppi_len": data.ppi_len,
                "flowstats": np.zeros((n, P.FLOWSTATS_DIM), np.float32),
                "label": data.label.astype(np.int16),
                "category": np.zeros(n, np.int8),
                "session_id": data.session_id,
                "ts": ts_start + np.arange(n, dtype=np.int64),
            }
        )


def load_kernel(monkeypatch, tmp_path: Path, mode: str, **env):
    (tmp_path / "working").mkdir(exist_ok=True)
    monkeypatch.setenv("ADL_KAGGLE_INPUT", str(tmp_path / "input"))
    monkeypatch.setenv("ADL_KAGGLE_WORKING", str(tmp_path / "working"))
    monkeypatch.setenv("ADL_TRAIN_MODE", mode)
    monkeypatch.delenv("ADL_QUEUE", raising=False)
    monkeypatch.delenv("ADL_TIME_LIMIT_SECONDS", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))

    spec = importlib.util.spec_from_file_location(f"kernel_{mode}_{id(tmp_path)}", KERNEL)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.log = lambda _m: None
    return mod


@pytest.fixture(autouse=True)
def _restore_sys_path():
    saved = list(sys.path)
    yield
    sys.path[:] = saved


def build_queue_inputs(tmp_path: Path, results_root: Path) -> tuple[Path, Path]:
    """``<tmp_path>/input``: a D1-export-shaped mirror (42 tiny weeks + a
    Standardizer) and the project code dataset, plus a queue file that trains
    fast on CPU. Returns ``(input_dir, queue_path)``."""
    mirror = tmp_path / "input" / "adl-export-d1"
    shard_root = mirror / "shards"
    for w in range(11, 53):
        data = make_flows(24, N_CLASSES, seed=w)
        write_week_shard(
            shard_root,
            data,
            period=f"WEEK-2022-{w:02d}",
            ts_start=(w - 11) * 10_000,
            n_classes=N_CLASSES,
        )

    train_sets = [
        ShardSet.open(shard_root / "cesnet-tls-year22" / f"WEEK-2022-{w:02d}") for w in TRAIN_WEEKS
    ]
    try:
        std = Standardizer.fit_many(train_sets)
    finally:
        for s in train_sets:
            s.close()
    std.save(mirror / "standardizer.json")

    code = tmp_path / "input" / "adl-encrypted-traffic-code"
    (code / "src" / "adl_etc").mkdir(parents=True)
    shutil.copy(REPO / "src" / "adl_etc" / "__init__.py", code / "src" / "adl_etc")
    (code / "configs" / "splits").mkdir(parents=True)
    shutil.copy(REPO / "configs" / "splits" / "d1_main.yaml", code / "configs" / "splits")
    (code / "configs" / "train").mkdir(parents=True)
    shutil.copy(REPO / "configs" / "train" / "base.yaml", code / "configs" / "train")
    shutil.copy(REPO / "configs" / "train" / "b3_gru_d1.yaml", code / "configs" / "train")
    (code / "configs" / "models" / "baselines").mkdir(parents=True)
    shutil.copy(
        REPO / "configs" / "models" / "baselines" / "gru.yaml",
        code / "configs" / "models" / "baselines",
    )
    (code / "GIT_COMMIT").write_text("0123456789abcdef\n", encoding="utf-8")
    (code / "GIT_DIRTY").write_text("0\n", encoding="utf-8")

    overrides = (
        f'["seed={{seed}}", "device=cpu", "train.epochs=1", "train.patience=99", '
        f'"results_root={results_root.as_posix()}"]'
    )
    queue_path = tmp_path / "queue.yaml"
    queue_path.write_text(
        "- config: configs/train/b3_gru_d1.yaml\n"
        f"  overrides: {overrides.format(seed=0)}\n"
        "- config: configs/train/b3_gru_d1.yaml\n"
        f"  overrides: {overrides.format(seed=1)}\n",
        encoding="utf-8",
    )
    return tmp_path / "input", queue_path


# --- the smoke path: no Kaggle mount at all ------------------------------------------------


def test_smoke_completes_fast_and_writes_a_report(tmp_path, monkeypatch):
    mod = load_kernel(monkeypatch, tmp_path, "smoke")
    t0 = time.perf_counter()
    mod.main()
    wall = time.perf_counter() - t0

    report = json.loads((tmp_path / "working" / "train_report.json").read_text())
    assert report["mode"] == "smoke"
    assert report["smoke"]["status"] in ("finished", "early_stopped")
    assert report["smoke"]["epochs_done"] == 2
    assert wall < 120  # spec 015: "in under 2 minutes"

    runs = list((tmp_path / "working" / "smoke" / "runs").glob("bl-gru-smoke-smoke-s0"))
    assert runs and (runs[0] / "best.pt").is_file()
    assert (runs[0] / "eval" / "val" / "report.json").is_file()


def test_smoke_pauses_and_resumes_rather_than_restarting(tmp_path, monkeypatch):
    mod = load_kernel(monkeypatch, tmp_path, "smoke")
    from adl_etc.training.run import run_training

    cfg_path = mod._smoke_config(tmp_path / "working" / "smoke")
    from adl_etc.utils.config import load_config

    cfg = load_config(cfg_path)

    paused = run_training(cfg, seed=0, deadline=time.monotonic() - 1.0)
    assert paused.status == "paused" and paused.epochs_done == 1

    resumed = run_training(cfg, seed=0, deadline=None)
    assert resumed.status in ("finished", "early_stopped")
    assert resumed.epochs_run == 1  # config's epochs=2, one already done
    assert resumed.history[0]["train_loss"] == paused.history[0]["train_loss"]


# --- the real queue path, against a synthetic D1-shaped mirror ------------------------------


def test_each_entry_trains_on_the_split_its_own_config_names(tmp_path, monkeypatch):
    """An entry naming an open-set split must train open-set: the kernel resolves
    that entry's own split file, not a hard-coded d1_main.yaml (found while
    building plan phase 3's T1a, before any open-set run reached Kaggle)."""
    from adl_etc.training.labels import LabelSpace

    results_root = tmp_path / "results"
    input_dir, _ = build_queue_inputs(tmp_path, results_root)
    splits = input_dir / "adl-encrypted-traffic-code" / "configs" / "splits"
    main = (splits / "d1_main.yaml").read_text(encoding="utf-8")
    (splits / "d1_open_test.yaml").write_text(main + "\nunknown_classes: [5]\n", encoding="utf-8")
    queue_path = tmp_path / "open_queue.yaml"
    queue_path.write_text(
        "- config: configs/train/b3_gru_d1.yaml\n"
        '  overrides: ["seed=0", "device=cpu", "train.epochs=1", "train.patience=99", '
        f'"results_root={results_root.as_posix()}", "split=configs/splits/d1_open_test.yaml", '
        '"split_name=opentest"]\n',
        encoding="utf-8",
    )
    mod = load_kernel(
        monkeypatch, tmp_path, "queue", ADL_QUEUE=str(queue_path), ADL_TIME_LIMIT_SECONDS=3600
    )
    mod.main()

    run_dir = results_root / "bl-gru-main-opentest-s0"
    space = LabelSpace.load(run_dir / "label_space.json")
    assert space.unknown == (5,) and space.n_classes == N_CLASSES - 1
    assert (tmp_path / "working" / "d1_open_test.resolved.yaml").is_file()


def test_the_queue_trains_every_entry_in_order(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    input_dir, queue_path = build_queue_inputs(tmp_path, results_root)
    mod = load_kernel(
        monkeypatch, tmp_path, "queue", ADL_QUEUE=str(queue_path), ADL_TIME_LIMIT_SECONDS=3600
    )
    mod.main()

    report = json.loads((tmp_path / "working" / "train_report.json").read_text())
    statuses = [e["status"] for e in report["queue"]]
    assert statuses == ["finished", "finished"] or set(statuses) <= {"finished", "early_stopped"}
    assert len(report["queue"]) == 2

    run0 = Path(report["queue"][0]["run_dir"])
    run1 = Path(report["queue"][1]["run_dir"])
    assert run0 != run1
    assert (run0 / "best.pt").is_file() and (run1 / "best.pt").is_file()


def test_a_finished_queue_entry_is_skipped_on_a_second_push(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    input_dir, queue_path = build_queue_inputs(tmp_path, results_root)
    mod = load_kernel(
        monkeypatch, tmp_path, "queue", ADL_QUEUE=str(queue_path), ADL_TIME_LIMIT_SECONDS=3600
    )
    mod.main()
    first_run_dir = Path(
        json.loads((tmp_path / "working" / "train_report.json").read_text())["queue"][0]["run_dir"]
    )
    first_mtime = (first_run_dir / "best.pt").stat().st_mtime

    mod2 = load_kernel(
        monkeypatch, tmp_path, "queue", ADL_QUEUE=str(queue_path), ADL_TIME_LIMIT_SECONDS=3600
    )
    mod2.main()
    report2 = json.loads((tmp_path / "working" / "train_report.json").read_text())
    assert [e["status"] for e in report2["queue"]] == ["already_finished", "already_finished"]
    assert (first_run_dir / "best.pt").stat().st_mtime == first_mtime  # not retrained


def test_the_queue_stops_before_starting_new_work_once_the_deadline_has_passed(
    tmp_path, monkeypatch
):
    results_root = tmp_path / "results"
    input_dir, queue_path = build_queue_inputs(tmp_path, results_root)
    # A deadline already in the past: the very first (unfinished) entry must
    # not be started.
    mod = load_kernel(
        monkeypatch, tmp_path, "queue", ADL_QUEUE=str(queue_path), ADL_TIME_LIMIT_SECONDS=0.001
    )
    time.sleep(0.01)
    mod.main()

    report = json.loads((tmp_path / "working" / "train_report.json").read_text())
    assert report["queue"] == []
    assert not results_root.exists() or not any(results_root.iterdir())


# --- cross-session resume (plan phase 3, T2) ---------------------------------------------------


def _one_entry_queue(tmp_path: Path, results_root: Path, *, epochs: int) -> Path:
    queue_path = tmp_path / "one_queue.yaml"
    queue_path.write_text(
        "- config: configs/train/b3_gru_d1.yaml\n"
        f'  overrides: ["seed=0", "device=cpu", "train.epochs={epochs}", "train.patience=99", '
        f'"results_root={results_root.as_posix()}"]\n',
        encoding="utf-8",
    )
    return queue_path


def _end_session(tmp_path: Path, results_root: Path) -> Path:
    """Session 1 is over: its output becomes a read-only mount of the next session
    (as `kernel_sources` of this kernel would), and the working area is wiped."""
    mounted = tmp_path / "input" / "previous-session" / "results"
    shutil.copytree(results_root, mounted)
    shutil.rmtree(results_root)
    return mounted


def test_a_finished_run_from_a_previous_session_is_restored_and_skipped(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    _, _ = build_queue_inputs(tmp_path, results_root)
    queue = _one_entry_queue(tmp_path, results_root, epochs=1)
    load_kernel(monkeypatch, tmp_path, "queue", ADL_QUEUE=str(queue),
                ADL_TIME_LIMIT_SECONDS=3600).main()  # fmt: skip
    _end_session(tmp_path, results_root)

    load_kernel(monkeypatch, tmp_path, "queue", ADL_QUEUE=str(queue),
                ADL_TIME_LIMIT_SECONDS=3600).main()  # fmt: skip

    report = json.loads((tmp_path / "working" / "train_report.json").read_text())
    assert report["queue"][0]["status"] == "already_finished"  # not retrained
    assert (results_root / "bl-gru-main-d1main-s0" / "best.pt").is_file()


def test_a_paused_run_from_a_previous_session_continues_where_it_stopped(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    _, _ = build_queue_inputs(tmp_path, results_root)
    queue = _one_entry_queue(tmp_path, results_root, epochs=3)
    mod = load_kernel(
        monkeypatch, tmp_path, "queue", ADL_QUEUE=str(queue), ADL_TIME_LIMIT_SECONDS=1
    )
    mod.QUEUE_STOP_MARGIN_SECONDS = 0  # let the entry start, then the guard pauses it
    mod.main()
    paused = json.loads((tmp_path / "working" / "train_report.json").read_text())["queue"][0]
    assert paused["status"] == "paused" and paused["epochs_done"] == 1
    _end_session(tmp_path, results_root)

    second = load_kernel(monkeypatch, tmp_path, "queue", ADL_QUEUE=str(queue),
                         ADL_TIME_LIMIT_SECONDS=3600)  # fmt: skip
    logs: list[str] = []
    second.log = logs.append
    second.main()

    # Training is deterministic, so a silent retrain would end identically: the log
    # is what shows the run was restored and continued rather than redone.
    assert any("restored from the previous session" in line for line in logs)
    resumed = json.loads((tmp_path / "working" / "train_report.json").read_text())["queue"][0]
    assert resumed["status"] == "finished" and resumed["epochs_done"] == 3
    state = json.loads((results_root / "bl-gru-main-d1main-s0" / "state.json").read_text())
    assert state["epochs_done"] == 3


def test_a_damaged_previous_run_stops_the_kernel_instead_of_being_retrained(
    tmp_path, monkeypatch
):
    results_root = tmp_path / "results"
    _, _ = build_queue_inputs(tmp_path, results_root)
    queue = _one_entry_queue(tmp_path, results_root, epochs=1)
    load_kernel(monkeypatch, tmp_path, "queue", ADL_QUEUE=str(queue),
                ADL_TIME_LIMIT_SECONDS=3600).main()  # fmt: skip
    mounted = _end_session(tmp_path, results_root)
    (mounted / "bl-gru-main-d1main-s0" / "state.json").write_text("{not json", encoding="utf-8")

    mod = load_kernel(monkeypatch, tmp_path, "queue", ADL_QUEUE=str(queue),
                      ADL_TIME_LIMIT_SECONDS=3600)  # fmt: skip
    with pytest.raises(SystemExit, match="unreadable"):
        mod.main()
    assert not (results_root / "bl-gru-main-d1main-s0").exists()
