"""Phase 5's table generator and acceptance criteria."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from adl_etc.evaluation import phase5_tables as T

HEAD = "ft-pat-ssl_tuned-d1open"
SIGNALS = ["mean_k", "prior_kl"]


def fake_decay() -> dict:
    s = {
        "acc_k5": {
            "test_id": 0.78,
            "last_four_drift": 0.71,
            "change": -0.07,
            "slope_per_week": -0.003,
        },
        "policy_accuracy": {
            "test_id": 0.89,
            "last_four_drift": 0.82,
            "change": -0.07,
            "slope_per_week": -0.003,
        },
        "early_minus_late": -0.007,
    }
    return {"run": "x", "summary": s, "weeks": []}


def fake_monitor(*, alarm=True, lead=-1, h=6.5) -> dict:
    first = {
        "week": "test_drift:WEEK-2022-38",
        "week_index": 8,
        "window": 40,
        "signals": ["prior_kl"],
    }
    mode = {
        "signals": SIGNALS,
        "h": h,
        "correlation": {
            "mean_k": {"spearman_vs_policy_macro_f1": -0.1, "spearman_vs_policy_accuracy": -0.1},
            "prior_kl": {"spearman_vs_policy_macro_f1": -0.9, "spearman_vs_policy_accuracy": -0.9},
        },
        "per_signal_first_alarm": {"mean_k": None, "prior_kl": first},
        "per_signal_lead_weeks": {"mean_k": None, "prior_kl": lead},
        "lead": {
            "first_week_below_floor": "test_drift:WEEK-2022-37",
            "first_warning": first if alarm else None,
            "first_alarm": first if alarm else None,
        },
        "summary": {
            "first_warning": {"window": 40} if alarm else None,
            "first_alarm": {"window": 40} if alarm else None,
            "n_alarm_windows": 1,
        },
    }
    return {"static": mode, "controller": mode}


def fake_efficiency(step=0.5, full=2.0, params=885_821) -> dict:
    return {
        "hardware": {
            "processor": "cpu",
            "cpu_physical": 4,
            "cpu_logical": 8,
            "ram_gb": 16,
            "torch": "2",
        },
        "models": {
            "PAT (x)": {
                "params": params,
                "flops_full_flow": 5.5e7,
                "flops_stream_total_30": 5.4e7,
                "latency_batch1": {"median_ms": full, "p95_ms": full * 1.2},
                "latency_batch64": {"median_ms": 64 * 0.3, "p95_ms": 30.0},
                "stream_step": {"median_ms": step, "p95_ms": step * 1.5},
                "stream_total_30_ms": step * 30,
            }
        },
    }


def groups(make, n=3, names=(HEAD,)) -> T.Groups:
    return {g: [(s, make()) for s in range(n)] for g in names}


def by_name(items: list[dict], needle: str) -> dict:
    return next(c for c in items if needle in c["criterion"])


def test_criteria_report_met_and_not_met() -> None:
    names = (HEAD, "ft-pat-tuned-d1open", "bl-gru-main-d1open")
    c = T.criteria(groups(fake_decay, names=names), groups(fake_monitor), fake_efficiency())
    assert by_name(c["004"], "three seeds")["met"] is True
    assert all(x["met"] for x in c["012"] if x["bar"].startswith("=="))
    assert by_name(c["012"], "strongest")["value"] == pytest.approx(-0.9)
    assert by_name(c["012"], "strongest")["met"] is None  # reported, not judged
    assert by_name(c["012"], "weeks between")["value"] == -1.0
    assert all(x["met"] for x in c["013"])

    bad = T.criteria(
        groups(fake_decay, n=2, names=names),
        groups(lambda: fake_monitor(alarm=False)),
        fake_efficiency(step=1.8, full=6.0, params=3_000_000),
    )
    assert by_name(bad["004"], "three seeds")["met"] is False
    assert all(x["met"] is False for x in bad["012"] if x["bar"].startswith("=="))
    assert all(x["met"] is False for x in bad["013"])


def test_missing_inputs_give_no_criteria_not_guesses() -> None:
    c = T.criteria({}, {}, None)
    assert c["004"] == [] and c["012"] == [] and c["013"] == []


def write(root: Path, name: str, filename: str, obj: dict) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_text(json.dumps(obj))


def test_load_group_groups_by_seed_and_ignores_other_directories(tmp_path: Path) -> None:
    for run in (f"{HEAD}-s2", f"{HEAD}-s0", "not-a-run"):
        write(tmp_path, run, "drift.json", fake_decay())
    g = T.load_group(tmp_path, "drift.json")
    assert set(g) == {HEAD} and [s for s, _ in g[HEAD]] == [0, 2]
    assert T.load_efficiency(tmp_path) is None


def test_render_markdown_has_every_section_and_populated_rows() -> None:
    decay, mon = groups(fake_decay), groups(fake_monitor)
    cost = groups(
        lambda: {
            p: {
                "packets": {"mean": 6.0, "median": 5.0, "p95": 14.0},
                "time_ms": {"mean": 1300.0, "median": 600.0, "p95": 5000.0},
                "time_share_of_flow": 0.16,
                "accuracy": 0.89,
            }
            for p in T.POLICY_LABEL
        }
    )
    crit = T.criteria(decay, mon, fake_efficiency())
    text = T.render_markdown(decay, mon, cost, fake_efficiency(), crit)
    for h in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5."):
        assert h in text
    sec2 = text.split("## 2.")[1].split("## 3.")[0]
    assert "class-prior KL" in sec2 and "-0.90" in sec2 and "wk38" in sec2
    sec4 = text.split("## 4.")[1].split("## 5.")[0]
    assert sec4.count("| Safe-Stop |") == 1 and "16.0" in sec4
    assert "PAT (x)" in text and "885,821" in text


def test_the_script_writes_table_and_criteria(tmp_path: Path) -> None:
    root = tmp_path / "r"
    for s in range(3):
        write(root, f"{HEAD}-s{s}", "drift.json", fake_decay())
        write(root, f"{HEAD}-s{s}", "drift_monitor.json", fake_monitor())
    (root / "efficiency.json").write_text(json.dumps(fake_efficiency()))
    spec = importlib.util.spec_from_file_location(
        "make_phase5_tables", Path(__file__).parents[2] / "scripts" / "make_phase5_tables.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["make_phase5_tables"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert mod.main(["--root", str(root), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "phase5_table.md").read_text().startswith("# Phase 5 tables")
    assert json.loads((tmp_path / "out" / "phase5_criteria.json").read_text())["headline"] == HEAD
    with pytest.raises(SystemExit, match="no phase 5"):
        mod.main(["--root", str(tmp_path / "empty"), "--out", str(tmp_path / "o2")])
