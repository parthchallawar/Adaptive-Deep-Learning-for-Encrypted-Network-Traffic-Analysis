"""Phase 4's table generator and acceptance criteria."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from adl_etc.evaluation import phase4 as P4
from adl_etc.evaluation import phase4_tables as T
from tests.evaluation.synth import N_CLASSES, fitted_unknown, make_set

HEAD = "ft-pat-ssl_tuned-d1open"
SCRATCH = "ft-pat-tuned-d1open"


def fake_report(
    *,
    u8=0.90,
    u30=0.93,
    e8=0.88,
    gain=0.02,
    kred=0.40,
    safe6=0.90,
    echo6=0.88,
    cape6=0.85,
    rl6=0.87,
    ctrl_err=0.03,
    static_err=0.20,
    osc=0,
    settle=(2, 3),
    front=((3.0, 0.70), (5.0, 0.82), (7.0, 0.88)),
    area=None,
):
    area = area or {"safe_stop": 0.80, "p_echo": 0.79, "p_cape": 0.75, "p_rl": 0.78}
    acc6 = {"safe_stop": safe6, "p_echo": echo6, "p_cape": cape6, "p_rl": rl6}
    cw = {
        p: {
            "acc_at_budget": {"4.0": 0.7, "6.0": acc6[p], "8.0": 0.9},
            "pareto_area": area[p],
            "smallest_mean_k": 3.0,
        }
        for p in T.POLICIES
    }
    h4 = {
        "budgets": {},
        "disturbances": {"target_step_8_5_8": {"controller": {"settling_windows": list(settle)}}},
    }
    for b in T.BUDGETS:
        h4["budgets"][b] = {
            "controller": {
                sp: {"mean_abs_error": ctrl_err, "oscillation_events": osc, "accuracy": 0.8}
                for sp in ("val", "test_id", "test_drift")
            },
            "static": {
                sp: {"mean_abs_error": static_err, "oscillation_events": 0, "accuracy": 0.8}
                for sp in ("val", "test_id", "test_drift")
            },
        }
    return {
        "meta": {"heads": "pat"},
        "h3": {
            "auroc": {
                # u = raw energy + `gain` at every K except 8 and 30, which are set directly
                "u": {
                    "8": u8,
                    "30": u30,
                    "5": 0.86 + gain,
                    "3": 0.79 + gain,
                    "2": 0.68 + gain,
                    "1": 0.5 + gain,
                },
                "energy_raw": {"8": e8, "30": 0.92, "5": 0.86, "3": 0.79, "2": 0.68, "1": 0.5},
            },
            "earliness": [{"k_reduction": kred, "gate": {}, "ours": {}}] * 2,
        },
        "h2": {
            "closed_world": cw,
            "fronts": {
                "safe_stop": {"mean_k": [m for m, _ in front], "accuracy": [a for _, a in front]}
            },
            "open_world": {
                "reject_first": {"unknown_committed": 0.2, "mean_k_known": 6.2},
                "commit_first": {"unknown_committed": 0.5, "mean_k_known": 6.0},
            },
        },
        "h4": h4,
    }


def write_runs(root: Path, runs: dict[str, dict]) -> None:
    for name, report in runs.items():
        d = root / name / "eval"
        d.mkdir(parents=True)
        (d / "phase4.json").write_text(json.dumps(report))


def test_load_reports_groups_by_seed_and_ignores_other_directories(tmp_path: Path) -> None:
    write_runs(
        tmp_path,
        {
            f"{HEAD}-s2": fake_report(),
            f"{HEAD}-s0": fake_report(),
            f"{SCRATCH}-s0": fake_report(),
            "not-a-run": fake_report(),
        },
    )
    groups = T.load_reports(tmp_path)
    assert set(groups) == {HEAD, SCRATCH}
    assert [s for s, _ in groups[HEAD]] == [0, 2]


def test_mean_std_and_fmt() -> None:
    assert T.mean_std([None, float("nan")]) is None and T.fmt(None) == "-"
    assert T.fmt(T.mean_std([0.5])) == "0.500 (n=1)"  # one seed shows no spread
    assert T.fmt(T.mean_std([0.4, 0.6]), pct=True) == "50.0 ± 10.0 (n=2)"


def test_min_mean_k_at_accuracy() -> None:
    front = {"mean_k": [3.0, 5.0, 8.0], "accuracy": [0.6, 0.8, 0.9]}
    assert T.min_mean_k_at_accuracy(front, 0.8) == 5.0
    assert (
        T.min_mean_k_at_accuracy(front, 0.95) is None
        and T.min_mean_k_at_accuracy(None, 0.5) is None
    )


def by_name(items: list[dict], needle: str) -> dict:
    return next(c for c in items if needle in c["criterion"])


def test_criteria_report_met_and_not_met_against_the_specs_bars() -> None:
    good = {
        HEAD: [(0, fake_report())],
        SCRATCH: [(0, fake_report(front=((4.0, 0.70), (6.5, 0.82), (9.0, 0.88))))],
    }
    c = T.criteria(good)
    assert by_name(c["H3"], "K=8")["met"] is True and by_name(c["H3"], "K=30")["met"] is True
    assert by_name(c["H3"], "best AUROC gain")["met"] is True  # 0.90 - 0.88 = 0.02
    assert all(x["met"] for x in c["H3"] if "rejection-K" in x["criterion"])
    assert by_name(c["H2"], "minus P-ECHO")["met"] is True  # 0.90 - 0.88 = 0.02 >= 0.015
    assert by_name(c["H2"], "largest Pareto area")["met"] is True
    assert by_name(c["H2"], "extra packets")["met"] is True  # 6.2 - 6.0 = 0.2 <= 0.5
    assert (
        by_name(c["H4"], "controller")["met"] is True and by_name(c["H4"], "static")["met"] is True
    )
    assert (
        by_name(c["H4"], "oscillation")["met"] is True and by_name(c["H4"], "settle")["met"] is True
    )
    assert all(x["met"] for x in c["H1_packets"])  # SSL needs ~20% fewer packets to reach 80%

    bad = {
        HEAD: [
            (
                0,
                fake_report(
                    u8=0.80,
                    e8=0.82,
                    gain=-0.02,
                    kred=0.1,
                    safe6=0.87,
                    ctrl_err=0.08,
                    static_err=0.03,
                    osc=2,
                    settle=(2, 9),
                    area={"safe_stop": 0.7, "p_echo": 0.79, "p_cape": 0.7, "p_rl": 0.7},
                ),
            )
        ]
    }
    c = T.criteria(bad)
    assert by_name(c["H3"], "K=8")["met"] is False
    assert by_name(c["H3"], "best AUROC gain")["met"] is False  # u is below raw energy
    assert all(x["met"] is False for x in c["H3"] if "rejection-K" in x["criterion"])
    assert by_name(c["H2"], "minus P-ECHO")["met"] is False
    assert by_name(c["H2"], "largest Pareto area")["met"] is False
    assert by_name(c["H4"], "controller")["met"] is False
    assert by_name(c["H4"], "static")["met"] is False  # static held the budget: no contrast
    assert (
        by_name(c["H4"], "oscillation")["met"] is False
        and by_name(c["H4"], "settle")["met"] is False
    )
    assert c["H1_packets"] == []  # no scratch group: cannot be evaluated, not guessed


def test_dig_walks_dicts_and_lists_and_returns_none_when_absent() -> None:
    rep = {"a": [{"x": 1}, {"x": 2}], "b": {"c": 3}}
    assert T.dig(rep, "a", "1", "x") == 2 and T.dig(rep, "b", "c") == 3
    assert T.dig(rep, "a", "5", "x") is None and T.dig(rep, "b", "z") is None
    assert T.dig(rep, "a", "x") is None


def test_the_earliness_table_is_populated_not_just_present(tmp_path: Path) -> None:
    write_runs(tmp_path, {f"{HEAD}-s0": fake_report()})
    text = T.render_markdown(T.load_reports(tmp_path))
    section = text.split("## 2.")[1].split("## 3.")[0]
    assert section.count("\n| ft-pat-ssl_tuned-d1open |") == 2  # one row per false-positive target


def test_a_group_without_h4_or_h3_is_not_evaluable_rather_than_failed() -> None:
    rep = fake_report()
    del rep["h4"], rep["h3"]
    c = T.criteria({HEAD: [(0, rep)]})
    assert c["H4"] == [] and all(x["met"] is None for x in c["H3"])


def test_a_missing_headline_group_gives_no_criteria() -> None:
    assert T.criteria({"other-d1open": [(0, fake_report())]})["H3"] == []


@pytest.fixture(scope="module")
def real_reports() -> dict:
    val, test = make_set(30, 500, seed=1), make_set(30, 500, seed=2)
    um = fitted_unknown(val)
    return P4.run_report(val, test, None, um, N_CLASSES, skip_h4=True)


def test_render_markdown_on_real_reports_has_every_section(real_reports, tmp_path: Path) -> None:
    write_runs(
        tmp_path,
        {f"{HEAD}-s0": real_reports, f"{HEAD}-s1": real_reports, f"{SCRATCH}-s0": real_reports},
    )
    text = T.render_markdown(T.load_reports(tmp_path))
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5."):
        assert heading in text
    assert f"{HEAD}" in text and "Safe-Stop" in text and "n=2" in text and "n=1" in text
    # these reports have no H4 section, so the table has no H4 rows (a rule that cannot be
    # evaluated is left out, never filled with a guess) while the H3 and H2 rows are there
    assert "| H4 |" not in text and "| H3 |" in text and "| H2 |" in text


def test_the_script_writes_the_table_and_the_criteria(real_reports, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_runs(runs, {f"{HEAD}-s0": real_reports})
    spec = importlib.util.spec_from_file_location(
        "make_phase4_tables", Path(__file__).parents[2] / "scripts" / "make_phase4_tables.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["make_phase4_tables"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert mod.main(["--root", str(runs), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "phase4_table.md").read_text().startswith("# Phase 4 tables")
    assert json.loads((tmp_path / "out" / "phase4_criteria.json").read_text())["headline"] == HEAD
    with pytest.raises(SystemExit, match="no <run>"):
        mod.main(["--root", str(tmp_path / "empty"), "--out", str(tmp_path / "o2")])
