"""The D1 export kernel (kernel/export-d1/kernel.py), run end to end against a small
synthetic mirror laid out like ``pranjalkar99/cesnet-22`` under ``/kaggle/input``.

The kernel is the one piece of phase-2 code that cannot be exercised on Kaggle
before it is pushed, and a multi-hour run that dies at hour 3 on a path typo is the
expensive way to find a bug. So its paths are overridable by environment, and this
test is what stands in for that first run.
"""

from __future__ import annotations

import importlib.util
import json
import lzma
import shutil
import sys
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pytest

from tests.data.test_cesnet_sampling import write_days

REPO = Path(__file__).resolve().parents[1]
KERNEL = REPO / "kernel" / "export-d1" / "kernel.py"
ROWS_PER_DAY = 300


def week_monday(week: int) -> date:
    return date(2022, 1, 3) + timedelta(days=7 * (week - 1))  # WEEK-2022-01 starts Jan 3


def build_inputs(root: Path, weeks=range(11, 53)) -> Path:
    """``<root>/input``: a mirror (one day per week) and a code dataset."""
    mirror = root / "input" / "cesnet-22" / "CESNET-TLS-Year22"
    for w in weeks:
        day = week_monday(w)
        day_dir = mirror / f"WEEK-2022-{w:02d}" / day.isoformat()
        day_dir.mkdir(parents=True)
        write_days(day_dir, {day.strftime("%Y%m%d"): ROWS_PER_DAY})

    code = root / "input" / "adl-encrypted-traffic-code"
    (code / "src" / "adl_etc").mkdir(parents=True)
    shutil.copy(REPO / "src" / "adl_etc" / "__init__.py", code / "src" / "adl_etc")
    (code / "configs" / "splits").mkdir(parents=True)
    shutil.copy(REPO / "configs" / "splits" / "d1_main.yaml", code / "configs" / "splits")
    (code / "GIT_COMMIT").write_text("0123456789abcdef\n", encoding="utf-8")
    (code / "GIT_DIRTY").write_text("0\n", encoding="utf-8")
    return root / "input"


def load_kernel(monkeypatch, tmp_path: Path, mode: str):
    (tmp_path / "working").mkdir(exist_ok=True)
    monkeypatch.setenv("ADL_KAGGLE_INPUT", str(tmp_path / "input"))
    monkeypatch.setenv("ADL_KAGGLE_WORKING", str(tmp_path / "working"))
    monkeypatch.setenv("ADL_EXPORT_MODE", mode)

    def no_network(*_a, **_k):
        raise OSError("network disabled in tests")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    spec = importlib.util.spec_from_file_location(f"export_d1_{mode}", KERNEL)
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


def test_probe_reports_speed_and_size_without_running_the_whole_year(tmp_path, monkeypatch):
    build_inputs(tmp_path)
    # the probe's named days, one of them the mirror's real zero-flow day
    mirror = tmp_path / "input" / "cesnet-22" / "CESNET-TLS-Year22"
    empty_dir = mirror / "WEEK-2022-52" / "2022-12-31"
    empty_dir.mkdir(parents=True)
    with lzma.open(empty_dir / "flows-20221231.csv.xz", "wb"):
        pass
    (empty_dir / "stats-20221231.json").write_text(
        json.dumps({"global": {"total-saved": 0}, "apps": {}}), encoding="utf-8"
    )
    mod = load_kernel(monkeypatch, tmp_path, "probe")
    mod.main()

    report = json.loads((tmp_path / "working" / "export_report.json").read_text())
    assert report["mode"] == "probe"
    assert report["internet"].startswith("no")  # the probe records the answer, either way
    assert report["export"]["verified_days"] == 2  # 20220314 and 20221231; 0615 is week 24
    assert report["code_commit"] == "0123456789abcdef"
    assert report["code_dirty"] is False
    assert report["probe_extrapolation"]["full_run_hours_if_all_days_alike"] >= 0
    assert not (tmp_path / "working" / "standardizer.json").exists()  # full run only


def test_full_run_exports_verifies_fits_the_standardizer_and_loads_the_real_split(
    tmp_path, monkeypatch
):
    build_inputs(tmp_path)
    mod = load_kernel(monkeypatch, tmp_path, "full")
    mod.main()

    work = tmp_path / "working"
    report = json.loads((work / "export_report.json").read_text())
    assert report["mirror_files_selected"] == 42
    assert report["export"]["verified_days"] == 42
    assert sorted(report["weeks"]) == [f"WEEK-2022-{w:02d}" for w in range(11, 53)]
    assert all(c["sample_rate"] == mod.SAMPLE_RATE for c in report["weeks"].values())
    assert all(c["rows_before_sampling"] == ROWS_PER_DAY for c in report["weeks"].values())

    # the Standardizer exists, is fit, and is what the split config now points at
    assert (work / "standardizer.json").exists()
    saved = json.loads((work / "standardizer.json").read_text())
    assert report["standardizer_hash"] == saved["hash"]
    assert str(work / "standardizer.json") in (work / "d1_main.resolved.yaml").read_text()

    # load_split ran for real: four splits, spec 004's four leakage rules passed
    assert set(report["split_flows"]) == {"train", "val", "test_id", "test_drift"}
    assert report["split_flows"]["train"] > 0
    assert report["n_classes"] == 3
    assert "classes_below_test_floor" in report  # 300 rows/day is far below 100 per class


def test_the_standardizer_is_fit_on_training_weeks_only(tmp_path, monkeypatch):
    """The saved statistics are those of weeks 11-26, not of every exported week
    (each day's sample differs, so a fit over more weeks has a different hash)."""
    import numpy as np

    from adl_etc.data.features import Standardizer
    from adl_etc.data.tensors import ShardSet

    build_inputs(tmp_path)
    mod = load_kernel(monkeypatch, tmp_path, "full")
    mod.main()

    out = tmp_path / "working" / "shards" / "cesnet-tls-year22"
    train = [ShardSet.open(out / f"WEEK-2022-{w:02d}") for w in range(11, 27)]
    everything = [ShardSet.open(out / f"WEEK-2022-{w:02d}") for w in range(11, 53)]
    try:
        expected = Standardizer.fit_many(train)
        wider = Standardizer.fit_many(everything)
    finally:
        for s in (*train, *everything):
            s.close()
    saved = Standardizer.load(tmp_path / "working" / "standardizer.json")
    assert saved.hash == expected.hash
    np.testing.assert_array_equal(saved.cont_mean, expected.cont_mean)
    assert saved.hash != wider.hash


def test_a_day_that_disagrees_with_its_stats_stops_the_run_and_writes_no_report(
    tmp_path, monkeypatch
):
    build_inputs(tmp_path)
    bad = next((tmp_path / "input").rglob("stats-20220620.json"))  # week 25
    bad.write_text(json.dumps({"global": {"total-saved": 1}, "apps": {}}), encoding="utf-8")
    mod = load_kernel(monkeypatch, tmp_path, "full")
    with pytest.raises(Exception, match="stats total-saved 1"):
        mod.main()
    assert not (tmp_path / "working" / "export_report.json").exists()
    assert not (tmp_path / "working" / "standardizer.json").exists()


def test_a_missing_stats_file_is_refused_before_anything_is_exported(tmp_path, monkeypatch):
    build_inputs(tmp_path)
    next((tmp_path / "input").rglob("stats-20220620.json")).unlink()
    mod = load_kernel(monkeypatch, tmp_path, "full")
    with pytest.raises(SystemExit, match="no stats file"):
        mod.main()
    assert not (tmp_path / "working" / "shards").exists()


def test_a_missing_code_dataset_or_mirror_fails_with_a_clear_message(tmp_path, monkeypatch):
    build_inputs(tmp_path)
    shutil.rmtree(tmp_path / "input" / "adl-encrypted-traffic-code")
    mod = load_kernel(monkeypatch, tmp_path, "probe")
    with pytest.raises(SystemExit, match="code dataset is not mounted"):
        mod.main()
