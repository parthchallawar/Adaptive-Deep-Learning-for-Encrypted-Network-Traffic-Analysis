"""``scripts/make_tables.py`` (plan T9): MLflow -> results/summaries/*, with
spec 014's three edge cases (dirty runs, config drift, unfinished runs).

Uses the same synthetic-run-directory technique as
``tests/utils/test_mlflow_sync.py``, but with real ``make_run_name`` run
names so ``make_tables``'s own grouping (seed stripped) has something real
to group.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from omegaconf import DictConfig

pytest.importorskip("mlflow", reason="needs the `train` extra: pip install -e '.[train]'")

from adl_etc.utils.config import load_config  # noqa: E402
from adl_etc.utils.mlflow_sync import local_store, sync_run_dir  # noqa: E402
from adl_etc.utils.runinfo import RunInfo, make_run_name  # noqa: E402
from adl_etc.utils.tracking import FINISHED, Tracker  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
make_tables = importlib.import_module("make_tables")


@dataclass
class Store:
    uri: str
    artifact_root: str
    client: Any


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    from mlflow import MlflowClient

    uri, artifact_root = local_store(tmp_path / "results")
    yield Store(uri, artifact_root, MlflowClient(tracking_uri=uri))


def make_cfg(work: Path, *, lr: float = 0.1) -> DictConfig:
    path = work / f"cfg-{lr}.yaml"
    path.write_text(f"seed: 0\noptim:\n  lr: {lr}\n", encoding="utf-8")
    return load_config(path)


def make_run(
    work: Path,
    *,
    stage: str = "bl",
    model: str = "gru",
    variant: str = "main",
    split: str = "synth",
    seed: int,
    best_val_macro_f1: float,
    dirty: bool = False,
    lr: float = 0.1,
    end: str | None = FINISHED,
    subdir: str = "runs",
) -> Path:
    root = work / f"code-{subdir}"
    root.mkdir(exist_ok=True)
    (root / "GIT_COMMIT").write_text("abc1234\n", encoding="utf-8")
    (root / "GIT_DIRTY").write_text(("1" if dirty else "0") + "\n", encoding="utf-8")
    cfg = make_cfg(work, lr=lr)
    name = make_run_name(stage, model, variant, split, seed)
    info = RunInfo.collect(name, seed, cfg, code_root=root)
    rd = work / subdir / name

    t = Tracker.start(info, cfg, rd)
    t.log_metrics({"train_loss": 1.0, "val_macro_f1": 0.5}, step=0)
    t.log_metrics({"best_val_macro_f1": best_val_macro_f1, "best_epoch": 1.0})
    if end is None:
        return rd  # never closed: a killed/unfinished session
    t.close(end)
    return rd


def sync(store: Store, rd: Path) -> None:
    sync_run_dir(rd, tracking_uri=store.uri, artifact_root=store.artifact_root)


def test_seeds_of_one_group_are_aggregated_into_one_row(store: Store, tmp_path: Path) -> None:
    for seed, f1 in enumerate([0.80, 0.81, 0.82]):
        sync(store, make_run(tmp_path, seed=seed, best_val_macro_f1=f1))

    runs = make_tables.fetch_runs(store.uri, "adl-etc")
    rows, dirty = make_tables.build_rows(runs, allow_dirty=False)

    assert dirty == []
    assert len(rows) == 1
    row = rows[0]
    assert row["run_group"] == "bl-gru-main-synth"
    assert row["seeds"] == [0, 1, 2]
    assert row["mean"] == pytest.approx(0.81, abs=1e-9)
    assert row["std"] > 0
    assert row["ci_lo"] <= row["mean"] <= row["ci_hi"]
    assert row["ci_lo"] >= 0.80 - 1e-9 and row["ci_hi"] <= 0.82 + 1e-9  # bootstrap of a mean
    assert not row["dirty"]


def test_different_variants_become_separate_rows(store: Store, tmp_path: Path) -> None:
    sync(store, make_run(tmp_path, variant="main", seed=0, best_val_macro_f1=0.8))
    sync(store, make_run(tmp_path, variant="ablation", seed=0, best_val_macro_f1=0.7))

    runs = make_tables.fetch_runs(store.uri, "adl-etc")
    rows, _ = make_tables.build_rows(runs, allow_dirty=False)

    assert {r["run_group"] for r in rows} == {"bl-gru-main-synth", "bl-gru-ablation-synth"}


def test_a_single_seed_group_has_a_degenerate_but_valid_ci(store: Store, tmp_path: Path) -> None:
    sync(store, make_run(tmp_path, seed=0, best_val_macro_f1=0.75))

    runs = make_tables.fetch_runs(store.uri, "adl-etc")
    rows, _ = make_tables.build_rows(runs, allow_dirty=False)

    assert len(rows) == 1
    assert rows[0]["mean"] == rows[0]["ci_lo"] == rows[0]["ci_hi"] == pytest.approx(0.75)
    assert rows[0]["std"] == 0.0


def test_a_dirty_run_is_refused_by_default(store: Store, tmp_path: Path) -> None:
    sync(store, make_run(tmp_path, seed=0, best_val_macro_f1=0.8, dirty=True))
    runs = make_tables.fetch_runs(store.uri, "adl-etc")

    with pytest.raises(make_tables.TableError, match="dirty"):
        make_tables.build_rows(runs, allow_dirty=False)


def test_allow_dirty_includes_and_flags_the_run(store: Store, tmp_path: Path) -> None:
    sync(store, make_run(tmp_path, seed=0, best_val_macro_f1=0.8, dirty=True))
    runs = make_tables.fetch_runs(store.uri, "adl-etc")

    rows, dirty = make_tables.build_rows(runs, allow_dirty=True)
    assert len(dirty) == 1
    assert rows[0]["dirty"] is True


def test_config_drift_under_the_same_run_name_raises(store: Store, tmp_path: Path) -> None:
    # Same stage/model/variant/split/seed (-> same run name), different lr ->
    # different config_hash: exactly spec 014's config-drift edge case. Two
    # different run dirs (as two separate machines/pushes would produce),
    # synced into the same store under the one run name.
    sync(store, make_run(tmp_path, seed=0, best_val_macro_f1=0.8, lr=0.1, subdir="run-a"))
    sync(store, make_run(tmp_path, seed=0, best_val_macro_f1=0.9, lr=0.2, subdir="run-b"))
    runs = make_tables.fetch_runs(store.uri, "adl-etc")

    with pytest.raises(make_tables.TableError, match="config drift"):
        make_tables.build_rows(runs, allow_dirty=False)


def test_a_run_that_never_finished_is_excluded(store: Store, tmp_path: Path) -> None:
    sync(store, make_run(tmp_path, seed=0, best_val_macro_f1=0.8))
    sync(store, make_run(tmp_path, seed=1, best_val_macro_f1=0.9, end=None))  # left running/killed

    runs = make_tables.fetch_runs(store.uri, "adl-etc")
    rows, _ = make_tables.build_rows(runs, allow_dirty=False)

    assert len(rows) == 1
    assert rows[0]["seeds"] == [0]  # seed 1's unfinished run is excluded, not averaged in


def test_a_run_name_outside_the_convention_is_skipped_not_crashed(
    store: Store, tmp_path: Path
) -> None:
    root = tmp_path / "code"
    root.mkdir(exist_ok=True)
    (root / "GIT_COMMIT").write_text("abc1234\n", encoding="utf-8")
    (root / "GIT_DIRTY").write_text("0\n", encoding="utf-8")
    cfg = make_cfg(tmp_path)
    info = RunInfo.collect("not-a-real-run-name", 0, cfg, code_root=root)
    rd = tmp_path / "runs" / "stray"
    t = Tracker.start(info, cfg, rd)
    t.log_metrics({"best_val_macro_f1": 0.5})
    t.close(FINISHED)
    sync(store, rd)

    runs = make_tables.fetch_runs(store.uri, "adl-etc")
    rows, _ = make_tables.build_rows(runs, allow_dirty=False)
    assert rows == []


def test_main_writes_markdown_and_latex_with_every_number_traceable(
    store: Store, tmp_path: Path
) -> None:
    for seed, f1 in enumerate([0.80, 0.81]):
        sync(store, make_run(tmp_path, seed=seed, best_val_macro_f1=f1))

    results_dir = Path(store.uri.removeprefix("sqlite:///")).parent
    rc = make_tables.main(["--results", str(results_dir)])
    assert rc == 0

    md = (results_dir / "summaries" / "main_table.md").read_text(encoding="utf-8")
    tex = (results_dir / "summaries" / "main_table.tex").read_text(encoding="utf-8")
    assert "bl-gru-main-synth" in md
    assert "0.805" in md  # the mean, to 3dp
    assert "abc1234" in md  # the commit
    assert r"\begin{tabular}" in tex
    assert "bl-gru-main-synth" in tex


def test_main_refuses_a_dirty_run_with_a_nonzero_exit(
    store: Store, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sync(store, make_run(tmp_path, seed=0, best_val_macro_f1=0.8, dirty=True))
    results_dir = Path(store.uri.removeprefix("sqlite:///")).parent

    rc = make_tables.main(["--results", str(results_dir)])
    assert rc != 0
    assert "dirty" in capsys.readouterr().err


def test_main_with_no_runs_reports_and_exits_nonzero(tmp_path: Path) -> None:
    rc = make_tables.main(["--results", str(tmp_path / "empty-results")])
    assert rc != 0
