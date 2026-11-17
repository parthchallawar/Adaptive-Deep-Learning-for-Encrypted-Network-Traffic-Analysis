"""The JSON-safe report and the script that writes it."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from adl_etc.evaluation import phase4 as P4
from tests.evaluation.synth import N_CLASSES, fitted_unknown, make_set


def test_jsonable_handles_numpy_int_keys_and_infinities() -> None:
    obj = {1: np.float32(0.5), "a": [np.int64(3), np.array([1.0, np.inf])], "n": float("nan")}
    out = P4.jsonable(obj)
    assert json.loads(json.dumps(out, allow_nan=False)) == {
        "1": 0.5,
        "a": [3, [1.0, None]],
        "n": None,
    }


def test_the_script_writes_a_strict_json_report_from_saved_scores(tmp_path: Path) -> None:
    val = make_set(30, 500, seed=1)
    test = make_set(30, 500, seed=2)
    drift = make_set(30, 500, seed=3, drift=1.0)
    um = fitted_unknown(val)
    run = tmp_path / "run"
    for name, s in (("val", val), ("test_id", test), ("test_drift", drift)):
        s.save(run / "eval" / name / "scores")
    um.save(run / "unknown")
    (run / "label_space.json").write_text(json.dumps({"known": list(range(N_CLASSES))}))

    spec = importlib.util.spec_from_file_location(
        "phase4_report", Path(__file__).parents[2] / "scripts" / "phase4_report.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["phase4_report"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert mod.main(["--run-dir", str(run), "--window", "500"]) == 0

    report = json.loads((run / "eval" / "phase4.json").read_text())  # strict: no NaN/Infinity
    assert {"meta", "h3", "h2", "h4"} <= set(report)
    assert report["meta"]["checkpoint_hash"] == "h" and report["meta"]["n_drift"] == len(drift)
    assert "4.0" in report["h2"]["closed_world"]["safe_stop"]["acc_at_budget"]


def test_a_model_without_an_unknown_model_gets_the_closed_world_comparison_only() -> None:
    val = make_set(30, 500, seed=1)
    test = make_set(30, 500, seed=2)
    report = P4.run_report(val, test, make_set(30, 500, seed=3), None, N_CLASSES, window=500)
    assert set(report) == {"meta", "h2"}  # no H3, no H4, no open-world part
    assert set(report["h2"]) == {"closed_world", "fronts"}
    assert report["meta"]["unknown_weights"] is None
    assert set(report["h2"]["closed_world"]) == {"safe_stop", "p_echo", "p_cape", "p_rl"}


def test_a_score_set_from_another_checkpoint_is_refused(tmp_path: Path) -> None:
    import pytest

    val = make_set(30, 500, seed=1)
    other = replace(make_set(30, 500, seed=2), meta={"checkpoint_hash": "different"})
    run = tmp_path / "run"
    val.save(run / "eval" / "val" / "scores")
    other.save(run / "eval" / "test_id" / "scores")
    fitted_unknown(val).save(run / "unknown")
    (run / "label_space.json").write_text(json.dumps({"known": list(range(N_CLASSES))}))
    spec = importlib.util.spec_from_file_location(
        "phase4_report2", Path(__file__).parents[2] / "scripts" / "phase4_report.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    with pytest.raises(SystemExit, match="different checkpoint"):
        mod.main(["--run-dir", str(run), "--skip-h4"])
