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


# --- phase 3: evaluations attached to runs, the per-K table, the bootstrap ------------------

import json  # noqa: E402

import numpy as np  # noqa: E402

from adl_etc.evaluation.bootstrap import Predictions  # noqa: E402
from adl_etc.utils.mlflow_sync import SyncError, attach_eval, find_eval_dirs  # noqa: E402

mlflow_import = importlib.import_module("mlflow_import")


def write_eval(
    ed: Path,
    *,
    auc: float,
    acc5: float,
    h: str,
    drop10_auc: float | None = None,
    preds: Predictions | None = None,
) -> Path:
    ed.mkdir(parents=True, exist_ok=True)
    report = {
        "split": ed.name,
        "acc_at_k": {"1": 0.1, "3": 0.4, "5": acc5, "8": 0.8, "30": 0.9},
        "auroc_at_k": {"5": 0.85, "30": 0.9},
        "auc_k": auc,
        "macro_f1": 0.8,
        "balanced_accuracy": 0.8,
        "k95": 8,
        "n_flows": 1000,
        "n_unknown": 100,
        "reliability": {"k": 30, "ece": 0.03},
        "hash": h,
    }
    (ed / "report.json").write_text(json.dumps(report), encoding="utf-8")
    if drop10_auc is not None:
        rob = {
            "conditions": {
                "clean": {"auc_k": auc, "macro_f1": 0.8, "acc_at_k": {"8": 0.8}},
                "drop_10": {"auc_k": drop10_auc, "macro_f1": 0.7, "acc_at_k": {"8": 0.7}},
            },
            "report_hashes": {"clean": h, "drop_10": h + "-d"},
        }
        (ed / "robustness.json").write_text(json.dumps(rob), encoding="utf-8")
        # a per-condition report, which must not be attached on its own
        write_eval(ed / "perturb" / "drop_10", auc=drop10_auc, acc5=acc5, h=h + "-d")
    if preds is not None:
        preds.save(ed / "predictions.npz")
    return ed


def _preds(p_right: float, seed: int, n: int = 400) -> Predictions:
    rng = np.random.default_rng(0)  # same flows and labels for every model
    labels = rng.integers(0, 4, n)
    labels[:40] = -1
    ks = np.arange(1, 31)
    right = np.random.default_rng(seed).random((n, ks.size)) < p_right
    preds = np.where(right, labels[:, None], (labels[:, None] + 1) % 4).astype(np.int16)
    return Predictions(np.arange(n), labels, rng.integers(1, 31, n), ks, preds)


def _pat_run(work: Path, variant: str, seed: int) -> Path:
    return make_run(
        work, stage="ft", model="pat", variant=variant, split="d1open", seed=seed,
        best_val_macro_f1=0.8,
    )  # fmt: skip


def test_attach_eval_adds_split_metrics_once_and_refuses_a_different_one(
    store: Store, tmp_path: Path
) -> None:
    rd = _pat_run(tmp_path, "tuned", 0)
    sync(store, rd)
    ed = write_eval(rd / "eval" / "val", auc=0.83, acc5=0.78, h="aaa")

    first = attach_eval(ed, run_name=rd.name, split="val", tracking_uri=store.uri)
    assert first.attached and first.metrics_logged > 0
    m = store.client.get_run(first.run_id).data.metrics
    assert m["val.auc_k"] == pytest.approx(0.83) and m["val.acc_k5"] == pytest.approx(0.78)
    assert m["val.ece_k30"] == pytest.approx(0.03) and m["val.auroc_k30"] == pytest.approx(0.9)

    again = attach_eval(ed, run_name=rd.name, split="val", tracking_uri=store.uri)
    assert not again.attached and again.metrics_logged == 0

    other = write_eval(tmp_path / "other" / rd.name / "eval" / "val", auc=0.5, acc5=0.5, h="bbb")
    with pytest.raises(SyncError, match="different 'val' evaluation"):
        attach_eval(other, run_name=rd.name, split="val", tracking_uri=store.uri)


def test_reattaching_the_same_evaluation_adds_only_files_it_lacked(
    store: Store, tmp_path: Path
) -> None:
    """Phase 3 Session 4 re-ran seed 0's robustness pass only to get predictions.npz:
    same reports (same hashes), so no metrics are logged twice, but the new file lands."""
    rd = _pat_run(tmp_path, "tuned", 0)
    sync(store, rd)
    old = write_eval(tmp_path / "s3" / rd.name / "eval" / "test_id", auc=0.83, acc5=0.78,
                     h="aaa", drop10_auc=0.8)  # fmt: skip
    first = attach_eval(old, run_name=rd.name, split="test_id", tracking_uri=store.uri)
    assert "predictions.npz" not in first.files_added

    new = write_eval(tmp_path / "s4" / rd.name / "eval" / "test_id", auc=0.83, acc5=0.78,
                     h="aaa", drop10_auc=0.8, preds=_preds(0.5, seed=1))  # fmt: skip
    again = attach_eval(new, run_name=rd.name, split="test_id", tracking_uri=store.uri)
    assert not again.attached and again.metrics_logged == 0
    assert again.files_added == ("predictions.npz",)
    names = {Path(a.path).name for a in store.client.list_artifacts(first.run_id, "eval/test_id")}
    assert "predictions.npz" in names
    third = attach_eval(new, run_name=rd.name, split="test_id", tracking_uri=store.uri)
    assert third.files_added == ()


def test_attach_eval_needs_the_run_imported_first(store: Store, tmp_path: Path) -> None:
    sync(store, make_run(tmp_path, seed=0, best_val_macro_f1=0.8))  # creates the experiment
    name = "ft-pat-x-d1open-s0"
    ed = write_eval(tmp_path / "robust" / name / "eval" / "test_id", auc=0.8, acc5=0.7, h="c")
    with pytest.raises(SyncError, match="no finished MLflow run"):
        attach_eval(ed, run_name=name, split="test_id", tracking_uri=store.uri)


def test_find_eval_dirs_skips_per_condition_reports_and_foreign_names(tmp_path: Path) -> None:
    ed = tmp_path / "robust" / "ft-pat-tuned-d1open-s0" / "eval" / "test_id"
    write_eval(ed, auc=0.8, acc5=0.7, h="a", drop10_auc=0.77)
    write_eval(tmp_path / "not-a-run" / "eval" / "val", auc=0.8, acc5=0.7, h="b")
    found = find_eval_dirs(tmp_path)
    assert [(n, s) for n, s, _ in found] == [("ft-pat-tuned-d1open-s0", "test_id")]


def _phase3_store(store: Store, tmp_path: Path) -> None:
    """Two seeds each of the tuned scratch/SSL pair, with test_id evaluations
    (robustness + per-flow predictions) attached, plus one val-only ablation."""
    for seed in (0, 1):
        for variant, auc, drop, p in (("tuned", 0.81, 0.78, 0.5), ("ssl_tuned", 0.83, 0.80, 0.6)):
            rd = _pat_run(tmp_path, variant, seed)
            sync(store, rd)
            ed = write_eval(
                tmp_path / "robust" / rd.name / "eval" / "test_id",
                auc=auc + seed * 0.01,
                acc5=0.7,
                h=f"{variant}{seed}",
                drop10_auc=drop + seed * 0.01,
                preds=_preds(p, seed=seed * 10 + len(variant)),
            )
            attach_eval(ed, run_name=rd.name, split="test_id", tracking_uri=store.uri)
    rd = _pat_run(tmp_path, "abl_drop0", 0)
    sync(store, rd)
    ed = write_eval(rd / "eval" / "val", auc=0.80, acc5=0.72, h="abl")
    attach_eval(ed, run_name=rd.name, split="val", tracking_uri=store.uri)


def test_phase3_table_aggregates_seeds_robustness_and_bootstrap(
    store: Store, tmp_path: Path
) -> None:
    _phase3_store(store, tmp_path)
    runs = make_tables.fetch_runs(store.uri, "adl-etc")
    groups, _ = make_tables.checked_groups(runs, allow_dirty=False)
    p3 = make_tables.build_phase3(groups, store.client, tmp_path / "dl")

    test = {r["group"]: r for r in p3["test"]}
    assert list(test) == ["ft-pat-ssl_tuned-d1open", "ft-pat-tuned-d1open"]  # display order
    ssl = test["ft-pat-ssl_tuned-d1open"]
    assert ssl["seeds"] == [0, 1]
    assert ssl["auc_k"][0] == pytest.approx(0.835) and ssl["auc_k"][2] == 2
    assert ssl["robust"]["drop_10"][0] == pytest.approx(-0.03)  # per-seed delta, then mean
    assert [r["group"] for r in p3["val"]] == ["ft-pat-abl_drop0-d1open"]

    boot = [b for b in p3["bootstrap"] if b["split"] is not None]
    pair = ("tuned", "ssl_tuned")
    assert [(b["pair"], b["seed"]) for b in boot] == [(pair, 0), (pair, 1)]
    assert all(b["auc_k"]["diff"] > 0 for b in boot)  # the simulated SSL model is better
    # the other pairs have no runs at all here, so they get no row
    assert all(b["split"] is not None for b in p3["bootstrap"])

    md = make_tables.render_phase3_markdown(p3)
    assert "PAT tuned, SSL init" in md and "0.835 ± 0.007" in md
    for block in md.split("\n\n"):  # every table's separator matches its header
        rows = [ln for ln in block.splitlines() if ln.startswith("|")]
        if len(rows) >= 2:
            assert rows[0].count("|") == rows[1].count("|"), rows[0]
    assert r"\begin{tabular}" in make_tables.render_phase3_latex(p3)


def test_a_pair_without_predictions_is_reported_unavailable(store: Store, tmp_path: Path) -> None:
    for variant in ("scratch", "ssl"):
        rd = _pat_run(tmp_path, variant, 0)
        sync(store, rd)
        ed = write_eval(rd / "eval" / "val", auc=0.8, acc5=0.7, h=variant)
        attach_eval(ed, run_name=rd.name, split="val", tracking_uri=store.uri)
    runs = make_tables.fetch_runs(store.uri, "adl-etc")
    groups, _ = make_tables.checked_groups(runs, allow_dirty=False)
    p3 = make_tables.build_phase3(groups, store.client, tmp_path / "dl")
    assert p3["bootstrap"] == [{"pair": ("scratch", "ssl"), "split": None, "seed": None}]
    assert "no per-flow predictions" in make_tables.render_phase3_markdown(p3)


def test_import_script_attaches_evaluations_across_sources(tmp_path: Path) -> None:
    src_a, src_b = tmp_path / "s1", tmp_path / "s2"
    src_a.mkdir()
    rd = _pat_run(src_a, "tuned", 0)
    write_eval(rd / "eval" / "val", auc=0.8, acc5=0.7, h="v")
    ed = src_b / "robust" / rd.name / "eval" / "test_id"
    write_eval(ed, auc=0.79, acc5=0.7, h="t", drop10_auc=0.76)
    results = tmp_path / "results"

    argv = ["--src", str(src_a), str(src_b), "--results", str(results)]
    assert mlflow_import.main(argv) == 0
    assert mlflow_import.main(argv) == 0  # idempotent: the second pass attaches nothing new

    from mlflow import MlflowClient

    uri, _ = local_store(results)
    client = MlflowClient(tracking_uri=uri)
    run = client.search_runs([client.get_experiment_by_name("adl-etc").experiment_id])[0]
    assert run.data.metrics["val.auc_k"] == pytest.approx(0.8)
    assert run.data.metrics["test_id.drop_10.auc_k"] == pytest.approx(0.76)
