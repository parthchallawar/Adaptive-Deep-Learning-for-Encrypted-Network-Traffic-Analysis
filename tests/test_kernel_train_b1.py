"""``kernel/train-b1/kernel.py`` (B1, spec 005, plan T8-follow-up), run end to
end against a synthetic mirror laid out like the D1 export kernel's own
output -- the same technique ``tests/test_kernel_smoke.py`` uses for the GPU
training kernel.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("xgboost")

pytestmark = pytest.mark.slow  # real shard sets, real (if tiny) XGBoost fits

from adl_etc.data.features import Standardizer  # noqa: E402
from adl_etc.data.tensors import ShardSet, ShardWriter  # noqa: E402
from tests.training.util import make_flows  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
KERNEL = REPO / "kernel" / "train-b1" / "kernel.py"
N_CLASSES = 6
TRAIN_WEEKS = range(11, 27)


def write_week_shard(root: Path, data, *, period: str, ts_start: int, n_classes: int) -> None:
    """Same technique as ``tests/test_kernel_smoke.py``'s own helper: real,
    globally increasing ``ts`` across weeks, needed for ``d1_main.yaml``'s
    ``temporal: true`` ordering rule."""
    import numpy as np

    from adl_etc.data import ppi as P

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


def build_inputs(tmp_path: Path) -> Path:
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
    shutil.copy(REPO / "configs" / "train" / "b1_xgb_d1.yaml", code / "configs" / "train")
    (code / "configs" / "models" / "baselines").mkdir(parents=True)
    shutil.copy(
        REPO / "configs" / "models" / "baselines" / "xgb.yaml",
        code / "configs" / "models" / "baselines",
    )
    (code / "GIT_COMMIT").write_text("0123456789abcdef\n", encoding="utf-8")
    (code / "GIT_DIRTY").write_text("0\n", encoding="utf-8")
    return tmp_path / "input"


def load_kernel(monkeypatch, tmp_path: Path):
    (tmp_path / "working").mkdir(exist_ok=True)
    monkeypatch.setenv("ADL_KAGGLE_INPUT", str(tmp_path / "input"))
    monkeypatch.setenv("ADL_KAGGLE_WORKING", str(tmp_path / "working"))

    spec = importlib.util.spec_from_file_location(f"train_b1_{id(tmp_path)}", KERNEL)
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


def test_trains_all_three_seeds_and_writes_a_report(tmp_path, monkeypatch):
    build_inputs(tmp_path)
    mod = load_kernel(monkeypatch, tmp_path)
    mod.SEEDS = (0, 1)  # two is enough to prove the loop; keep the test fast

    mod.main()

    report = json.loads((tmp_path / "working" / "train_b1_report.json").read_text())
    statuses = [e["status"] for e in report["seeds"]]
    assert statuses == ["finished", "finished"]
    assert all(e["best_val_macro_f1"] is not None for e in report["seeds"])

    run0 = Path(report["seeds"][0]["run_dir"])
    assert (run0 / "model" / "meta.json").is_file()
    assert (run0 / "eval" / "val" / "report.json").is_file()


def test_a_finished_seed_is_skipped_on_a_second_push(tmp_path, monkeypatch):
    build_inputs(tmp_path)
    mod = load_kernel(monkeypatch, tmp_path)
    mod.SEEDS = (0,)
    mod.main()
    first_run_dir = Path(
        json.loads((tmp_path / "working" / "train_b1_report.json").read_text())["seeds"][0][
            "run_dir"
        ]
    )
    first_mtime = (first_run_dir / "model" / "meta.json").stat().st_mtime

    mod2 = load_kernel(monkeypatch, tmp_path)
    mod2.SEEDS = (0,)
    mod2.main()
    report2 = json.loads((tmp_path / "working" / "train_b1_report.json").read_text())
    assert report2["seeds"][0]["status"] == "already_finished"
    assert (first_run_dir / "model" / "meta.json").stat().st_mtime == first_mtime
