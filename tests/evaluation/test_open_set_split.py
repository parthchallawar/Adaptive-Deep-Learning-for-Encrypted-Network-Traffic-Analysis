"""Open-set splits: held-out classes leave training at load time and stay
everywhere else (spec 004; plan phase 3, T1a), plus the script that draws them
and writes the split file."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from adl_etc.evaluation import protocol as PR
from tests.evaluation.test_protocol import _write_period, _write_yaml

ROOT = Path(__file__).resolve().parents[2]


def labels_of(loaded: PR.LoadedSplit, name: str) -> np.ndarray:
    return np.concatenate([
        ss.column("label") if m is None else ss.column("label")[m]
        for ss, m in zip(loaded.shard_sets[name], loaded.masks[name], strict=True)
    ])  # fmt: skip


def two_period_split(tmp_path: Path, **extra) -> Path:
    _write_period(tmp_path, "d", "p1", n=8, labels=[0, 1, 2, 3, 0, 1, 2, 3], ts=list(range(8)))
    _write_period(tmp_path, "d", "p2", n=4, labels=[0, 1, 2, 3], ts=list(range(100, 104)))
    path = tmp_path / "split.yaml"
    _write_yaml(path, {
        "dataset": "d", "temporal": True,
        "splits": {"train": {"periods": ["p1"]}, "val": {"periods": ["p2"]}}, **extra,
    })  # fmt: skip
    return path


def test_held_out_classes_leave_training_and_stay_in_validation(tmp_path: Path) -> None:
    with PR.load_split(two_period_split(tmp_path, unknown_classes=[3, 1]), root=tmp_path) as L:
        assert L.unknown_classes == (1, 3)  # sorted
        assert sorted(labels_of(L, "train").tolist()) == [0, 0, 2, 2]
        assert sorted(labels_of(L, "val").tolist()) == [0, 1, 2, 3]
        assert L.unknown_filtered == {"train": 4}


def test_a_closed_set_split_is_untouched(tmp_path: Path) -> None:
    with PR.load_split(two_period_split(tmp_path), root=tmp_path) as L:
        assert L.unknown_classes == () and L.unknown_filtered == {}
        assert len(labels_of(L, "train")) == 8


def test_filtering_combines_with_a_session_ids_mask(tmp_path: Path) -> None:
    _write_period(tmp_path, "d", "all", n=6, labels=[0, 1, 0, 1, 0, 1],
                  session_ids=[0, 0, 1, 1, 2, 2])  # fmt: skip
    path = tmp_path / "grouped.yaml"
    _write_yaml(path, {
        "dataset": "d", "temporal": False, "unknown_classes": [1],
        "splits": {"train": {"periods": ["all"], "session_ids": [0, 1]},
                   "test": {"periods": ["all"], "session_ids": [2]}},
    })  # fmt: skip

    with PR.load_split(path, root=tmp_path) as L:
        assert labels_of(L, "train").tolist() == [0, 0]  # sessions 0-1, class 1 dropped
        assert labels_of(L, "test").tolist() == [0, 1]
        assert L.unknown_filtered == {"train": 2}  # counted within the session mask only


def test_rule_2_catches_a_held_out_class_that_reaches_training(tmp_path: Path) -> None:
    with PR.load_split(two_period_split(tmp_path, unknown_classes=[1]), root=tmp_path) as L:
        L.masks["train"] = [None]  # undo the filter, as a bug elsewhere would
        with pytest.raises(PR.LeakageError, match="held-out unknown classes"):
            PR.assert_no_unknown_in_train(L)


@pytest.mark.parametrize("bad", [[-1], [2, 2], [1, -3]])
def test_malformed_unknown_classes_are_refused(tmp_path: Path, bad: list[int]) -> None:
    with pytest.raises(ValueError, match="distinct non-negative"):
        PR.load_split(two_period_split(tmp_path, unknown_classes=bad), root=tmp_path)


# --- the script ----------------------------------------------------------------------------


def test_the_script_draws_from_summed_support_and_writes_a_loadable_split(tmp_path: Path) -> None:
    """Six classes in two test weeks: class 5 has 60 flows per week, so it only
    clears a 100-flow floor when the weeks are summed."""
    rng = np.random.default_rng(0)
    for week in ("w1", "w2"):
        labels = np.repeat(np.arange(6), [80, 70, 90, 75, 85, 60])
        rng.shuffle(labels)
        _write_period(tmp_path / "data", "d", week, n=len(labels), labels=labels.tolist())
    base = tmp_path / "base.yaml"
    base.write_text("# a hand-written comment that must survive\ndataset: d\ntemporal: false\n"
                    "splits:\n  train: {periods: [w1]}\n  test: {periods: [w2]}\n",
                    encoding="utf-8")  # fmt: skip
    out, split_out = tmp_path / "classes.yaml", tmp_path / "open.yaml"
    cmd = [sys.executable, str(ROOT / "scripts/make_unknown_split.py"), "--dataset", "d",
           "--test-period", "w1", "w2", "--root", str(tmp_path / "data"), "--out", str(out),
           "--base-split", str(base), "--split-out", str(split_out),
           "--n-known", "4", "--n-unknown", "2", "--min-test-support", "130"]  # fmt: skip

    subprocess.run(cmd, check=True, capture_output=True, text=True)

    classes = OmegaConf.load(out)
    assert classes.test_periods == ["w1", "w2"]
    assert len(classes.known_classes) == 4 and len(classes.unknown_classes) == 2
    assert set(classes.unknown_classes) <= {0, 1, 2, 3, 4}  # 5 has 120 < 130 summed
    assert list(classes.classes_below_floor) == [5]
    text = split_out.read_text(encoding="utf-8")
    assert "# a hand-written comment that must survive" in text and "GENERATED" in text
    assert PR.load_split_spec(split_out)["unknown_classes"] == sorted(classes.unknown_classes)

    regenerate_on_top = [
        str(split_out) if a == str(base) else str(tmp_path / "x.yaml") if a == str(split_out)
        else a for a in cmd
    ]  # fmt: skip
    again = subprocess.run(regenerate_on_top, capture_output=True, text=True)
    assert again.returncode != 0 and "already has an open-set block" in again.stderr
    assert not (tmp_path / "x.yaml").exists()


def test_the_committed_open_split_matches_its_generated_record() -> None:
    """``d1_open.yaml``'s block and ``d1_open.classes.yaml`` came from one run."""
    spec = PR.load_split_spec(ROOT / "configs/splits/d1_open.yaml")
    record = OmegaConf.load(ROOT / "configs/splits/d1_open.classes.yaml")
    main = PR.load_split_spec(ROOT / "configs/splits/d1_main.yaml")

    assert spec["unknown_classes"] == list(record.unknown_classes)
    assert len(record.unknown_classes) == 30 and len(record.known_classes) == 150
    assert spec["splits"] == main["splits"]  # same periods and standardizer as d1_main
    assert not set(record.unknown_classes) & set(record.classes_below_floor)
