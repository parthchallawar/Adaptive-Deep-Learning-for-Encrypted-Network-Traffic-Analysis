"""``export_dataset(sample_rate=...)`` and in-pass ``verify`` (phase-2 plan T3).

Rows are identified by their ``ts`` (each fixture row has its own ``TIME_FIRST``),
so a test can say exactly *which* rows a sample kept, not just how many.
"""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import numpy as np
import pytest

from adl_etc.data import cesnet_csv as C
from adl_etc.data.tensors import ARRAY_SPEC, ShardSet

from .test_cesnet_csv import write_fixture_csv

DATASET = "cesnet-tls-year22"
_APPS = ("app-a", "app-a", "app-a", "app-b", "app-c")  # 60 / 20 / 20 %


def make_rows(day: str, n: int) -> list[tuple]:
    """``n`` distinct one-packet rows on ``day`` (YYYY-MM-DD), unique TIME_FIRST."""
    rows = []
    for i in range(n):
        app = _APPS[i % len(_APPS)]
        cat = "cat-a" if app != "app-c" else "cat-b"
        stamp = f"{day}T{i // 3600 % 24:02d}:{i // 60 % 60:02d}:{i % 60:02d}"
        rows.append((stamp, 0.0, app, cat, [0], [1], [100 + i % 900], [1], 1, 0, 100, 0, 1, 0))
    return rows


def write_days(root: Path, days: dict[str, int]) -> list[Path]:
    """``{YYYYMMDD: n_rows}`` -> flows-*.csv.xz files (+ matching stats-*.json)."""
    paths = []
    for stamp, n in days.items():
        day = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}"
        rows = make_rows(day, n)
        paths.append(write_fixture_csv(root / f"flows-{stamp}.csv.xz", rows, compress=True))
        apps: dict[str, int] = {}
        for r in rows:
            apps[r[2]] = apps.get(r[2], 0) + 1
        stats = {"global": {"total-saved": n}, "apps": apps}
        (root / f"stats-{stamp}.json").write_text(json.dumps(stats), encoding="utf-8")
    return paths


def export(paths, out, **kw):
    kw.setdefault("chunksize", 300)
    kw.setdefault("log", lambda _msg: None)
    return C.export_dataset(csv_paths=paths, dataset=DATASET, out_root=out, **kw)


def _run(paths, out, **kw):
    export(paths, out, **kw)
    return out


def read_week(out: Path, week: str, name: str = "ts") -> np.ndarray:
    with ShardSet.open(out / DATASET / week) as ss:
        return np.array(ss.column(name))


# --- what a sample is ----------------------------------------------------------------


def test_a_half_sample_is_a_deterministic_subset_in_source_order(tmp_path):
    paths = write_days(tmp_path, {"20220615": 2000})
    all_ts = read_week(_run(paths, tmp_path / "full"), "WEEK-2022-24")

    a = read_week(_run(paths, tmp_path / "a", sample_rate=0.5), "WEEK-2022-24")
    b = read_week(_run(paths, tmp_path / "b", sample_rate=0.5), "WEEK-2022-24")

    np.testing.assert_array_equal(a, b)  # same seed -> the same rows on a re-run
    assert 0.45 * len(all_ts) < len(a) < 0.55 * len(all_ts)  # p=0.5, n=2000: ~11 sigma
    assert set(a) <= set(all_ts)
    np.testing.assert_array_equal(a, all_ts[np.isin(all_ts, a)])  # original order kept


def test_sampling_is_uniform_not_per_class(tmp_path):
    """Class priors must survive: the drift study measures them. A 60/20/20 mix
    stays 60/20/20 in a 25% sample (a stratified sampler would flatten it)."""
    paths = write_days(tmp_path, {"20220615": 4000})
    out = _run(paths, tmp_path / "o", sample_rate=0.25)
    label = read_week(out, "WEEK-2022-24", "label")
    with ShardSet.open(out / DATASET / "WEEK-2022-24") as ss:
        app_a = ss.meta["label_map"]["app-a"]
    assert 0.25 * 4000 * 0.9 < len(label) < 0.25 * 4000 * 1.1
    assert abs(float(np.mean(label == app_a)) - 0.6) < 0.05


def test_rate_one_is_bit_identical_to_no_sampling(tmp_path):
    paths = write_days(tmp_path, {"20220615": 300, "20220616": 200})
    plain = _run(paths, tmp_path / "plain")
    one = _run(paths, tmp_path / "one", sample_rate=1.0)
    with (
        ShardSet.open(plain / DATASET / "WEEK-2022-24") as p,
        ShardSet.open(one / DATASET / "WEEK-2022-24") as o,
    ):
        assert len(p) == len(o) == 500
        for name in ARRAY_SPEC:
            np.testing.assert_array_equal(p.column(name), o.column(name), err_msg=name)


def test_the_sample_does_not_depend_on_chunk_size(tmp_path):
    paths = write_days(tmp_path, {"20220615": 1500})
    small = read_week(_run(paths, tmp_path / "s", sample_rate=0.3, chunksize=7), "WEEK-2022-24")
    big = read_week(_run(paths, tmp_path / "b", sample_rate=0.3, chunksize=4096), "WEEK-2022-24")
    np.testing.assert_array_equal(small, big)


def test_a_day_reproduces_its_own_sample_wherever_it_is_exported(tmp_path):
    """Seeded from the file's name, per day: a partial re-export of one day, from a
    different directory (Kaggle's mount is not this machine's path), matches the
    full export's rows for that day."""
    src1, src2 = tmp_path / "src1", tmp_path / "src2"
    src1.mkdir()
    src2.mkdir()
    both = write_days(src1, {"20220615": 800, "20220616": 800})
    alone = write_days(src2, {"20220616": 800})

    full = read_week(_run(both, tmp_path / "full", sample_rate=0.4), "WEEK-2022-24")
    only = read_week(_run(alone, tmp_path / "only", sample_rate=0.4), "WEEK-2022-24")
    day16 = full[full >= np.min(only)]  # the second day's rows (later timestamps)
    np.testing.assert_array_equal(day16, only)


def test_seed_and_file_name_each_change_the_sample(tmp_path):
    paths = write_days(tmp_path, {"20220615": 1000, "20220616": 1000})
    s0 = read_week(_run(paths, tmp_path / "s0", sample_rate=0.5), "WEEK-2022-24")
    s1 = read_week(_run(paths, tmp_path / "s1", sample_rate=0.5, sample_seed=1), "WEEK-2022-24")
    assert not np.array_equal(s0, s1)

    a = C.sample_rng(Path("flows-20220615.csv.xz"), 0).random(50)
    b = C.sample_rng(Path("flows-20220616.csv.xz"), 0).random(50)
    same_name_elsewhere = C.sample_rng(Path("/kaggle/input/x/flows-20220615.csv.xz"), 0).random(50)
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(a, same_name_elsewhere)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan")])
def test_a_rate_outside_zero_one_is_rejected(tmp_path, bad):
    paths = write_days(tmp_path, {"20220615": 10})
    with pytest.raises(ValueError, match="sample_rate"):
        export(paths, tmp_path / "o", sample_rate=bad)


# --- what a shard set says about itself -------------------------------------------------


def test_meta_records_the_rate_the_seed_and_the_unsampled_row_count(tmp_path):
    paths = write_days(tmp_path, {"20220615": 700, "20220616": 300})
    out = _run(paths, tmp_path / "o", sample_rate=0.2, sample_seed=7)
    with ShardSet.open(out / DATASET / "WEEK-2022-24") as ss:
        assert ss.meta["sample_rate"] == 0.2
        assert ss.meta["sample_seed"] == 7
        assert ss.meta["counters"]["rows_before_sampling"] == 1000
        assert 0 < ss.meta["n_flows"] < 1000


def test_an_unsampled_export_records_rate_one_not_absence(tmp_path):
    paths = write_days(tmp_path, {"20220615": 20})
    out = _run(paths, tmp_path / "o")
    with ShardSet.open(out / DATASET / "WEEK-2022-24") as ss:
        assert ss.meta["sample_rate"] == 1.0
        assert ss.meta["counters"]["rows_before_sampling"] == 20


def test_a_day_that_samples_to_nothing_or_is_empty_does_not_crash(tmp_path):
    paths = write_days(tmp_path, {"20220615": 3})
    empty = tmp_path / "flows-20220616.csv.xz"
    with lzma.open(empty, "wb"):  # a valid xz stream that decompresses to zero bytes
        pass
    (tmp_path / "stats-20220616.json").write_text(
        json.dumps({"global": {"total-saved": 0}, "apps": {}}), encoding="utf-8"
    )
    summary = export([*paths, empty], tmp_path / "o", sample_rate=0.001, verify=True)
    assert summary.rows_processed == 3
    assert summary.verified_days == 2
    # a week made only of an empty day still says how many rows it saw
    with ShardSet.open(tmp_path / "o" / DATASET / "WEEK-2022-24") as ss:
        assert ss.meta["counters"]["rows_before_sampling"] == 3


# --- verification runs on the unsampled day ------------------------------------------------


def test_verify_checks_the_unsampled_count_while_sampling(tmp_path):
    paths = write_days(tmp_path, {"20220615": 500, "20220616": 400})
    summary = export(paths, tmp_path / "o", sample_rate=0.1, verify=True)
    assert summary.verified_days == 2
    assert summary.rows_processed == 900  # rows read, not rows kept
    assert summary.n_flows < 200


def test_verify_stops_on_a_count_that_disagrees_with_stats(tmp_path):
    paths = write_days(tmp_path, {"20220615": 100})
    stats = tmp_path / "stats-20220615.json"
    stats.write_text(json.dumps({"global": {"total-saved": 101}, "apps": {}}), encoding="utf-8")
    with pytest.raises(C.VerificationError, match="row count 100 != stats total-saved 101"):
        export(paths, tmp_path / "o", sample_rate=0.5, verify=True)


def test_verify_stops_on_an_app_the_stats_do_not_list(tmp_path):
    paths = write_days(tmp_path, {"20220615": 100})
    stats = tmp_path / "stats-20220615.json"
    stats.write_text(
        json.dumps({"global": {"total-saved": 100}, "apps": {"app-a": 60}}), encoding="utf-8"
    )
    with pytest.raises(C.VerificationError, match="APP values not in stats.apps"):
        export(paths, tmp_path / "o", verify=True)


def test_verify_stops_when_the_stats_file_is_missing(tmp_path):
    paths = write_days(tmp_path, {"20220615": 10})
    (tmp_path / "stats-20220615.json").unlink()
    with pytest.raises(C.VerificationError, match="missing stats file"):
        export(paths, tmp_path / "o", verify=True)


def test_without_verify_a_wrong_stats_file_is_ignored(tmp_path):
    paths = write_days(tmp_path, {"20220615": 10})
    (tmp_path / "stats-20220615.json").write_text("not json", encoding="utf-8")
    assert export(paths, tmp_path / "o").verified_days == 0


# --- writers are released per week without breaking non-contiguous input --------------------


def test_a_week_split_around_another_week_still_lands_in_one_shard_set(tmp_path):
    """Writers close as soon as their week's last file is done, to bound memory over
    a 42-week export; that must be the *last* file, not the first week change."""
    paths = write_days(tmp_path, {"20220613": 30, "20220620": 40, "20220614": 50})
    summary = export(paths, tmp_path / "o")
    assert summary.weeks == {"WEEK-2022-24": 80, "WEEK-2022-25": 40}
    assert len(read_week(tmp_path / "o", "WEEK-2022-24")) == 80


def test_a_week_that_is_only_an_empty_day_records_zero_rows_seen(tmp_path):
    empty = tmp_path / "flows-20220615.csv.xz"
    with lzma.open(empty, "wb"):
        pass
    (tmp_path / "stats-20220615.json").write_text(
        json.dumps({"global": {"total-saved": 0}, "apps": {}}), encoding="utf-8"
    )
    # nothing is written for a week with no flows at all, and nothing crashes
    summary = export([empty], tmp_path / "o", sample_rate=0.1, verify=True)
    assert summary.n_flows == 0
    with ShardSet.open(tmp_path / "o" / DATASET / "WEEK-2022-24") as ss:
        assert ss.meta["counters"]["rows_before_sampling"] == 0
        assert ss.meta["n_flows"] == 0
