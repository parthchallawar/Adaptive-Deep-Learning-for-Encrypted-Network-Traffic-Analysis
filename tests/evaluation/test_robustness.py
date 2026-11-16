"""``evaluation.robustness``: parsing, the damage itself, and the summary (plan
phase 3, T8). NumPy only, like the module."""

from __future__ import annotations

import numpy as np
import pytest

from adl_etc.data import ppi as P
from adl_etc.evaluation import robustness as R


def _flows(n: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ppi_len = rng.integers(1, P.K_MAX + 1, size=n).astype(np.int64)
    ppi = np.zeros((n, P.K_MAX, 4), dtype=np.int32)
    for i, length in enumerate(ppi_len):
        ppi[i, :length, P.IPT_POS] = rng.integers(0, 500, size=length)
        ppi[i, 0, P.IPT_POS] = 0
        ppi[i, :length, P.DIR_POS] = rng.choice([-1, 1], size=length)
        ppi[i, :length, P.SIZE_POS] = rng.integers(1, 1500, size=length)
    return ppi, ppi_len


# --- parsing -------------------------------------------------------------------------------


def test_parse_puts_clean_first_and_adds_it_when_absent():
    out = R.parse_perturbations([{"name": "d", "kind": "drop", "value": 0.1}])
    assert [p.name for p in out] == ["clean", "d"]
    out = R.parse_perturbations(
        [{"name": "d", "kind": "drop", "value": 0.1}, {"name": "clean", "kind": "clean"}]
    )
    assert [p.name for p in out] == ["clean", "d"]


@pytest.mark.parametrize(
    "entry, match",
    [
        ({"name": "x", "kind": "explode", "value": 1}, "unknown perturbation kind"),
        ({"name": "x", "kind": "drop"}, "takes a value"),
        ({"name": "x", "kind": "clean", "value": 0.1}, "takes no value"),
        ({"name": "x", "kind": "drop", "value": 1.5}, "<= 1"),
        ({"name": "x", "kind": "ipt_jitter", "value": -0.1}, ">= 0"),
        ({"name": "a/b", "kind": "drop", "value": 0.1}, "plain directory name"),
        ({"name": "x", "kind": "drop", "value": 0.1, "p": 0.2}, "unknown keys"),
        ({"name": "clean", "kind": "drop", "value": 0.1}, "reserved"),
    ],
)
def test_parse_rejects_bad_entries(entry, match):
    with pytest.raises(ValueError, match=match):
        R.parse_perturbations([entry])


def test_parse_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        R.parse_perturbations(
            [
                {"name": "d", "kind": "drop", "value": 0.1},
                {"name": "d", "kind": "drop", "value": 0.2},
            ]
        )


# --- the damage ----------------------------------------------------------------------------


def test_clean_is_the_identity():
    ppi, ppi_len = _flows()
    out, out_len = R.perturb(ppi, ppi_len, R.CLEAN, seed=0)
    assert np.array_equal(out, ppi) and np.array_equal(out_len, ppi_len)


@pytest.mark.parametrize(
    "p",
    [
        R.Perturbation("d", "drop", 0.2),
        R.Perturbation("r", "reorder", 0.3),
        R.Perturbation("j", "ipt_jitter", 0.5),
        R.Perturbation("s", "size_jitter", 20.0),
    ],
)
def test_damage_is_deterministic_in_seed_and_name_and_leaves_inputs_alone(p):
    ppi, ppi_len = _flows()
    before, before_len = ppi.copy(), ppi_len.copy()
    a, a_len = R.perturb(ppi, ppi_len, p, seed=0)
    b, b_len = R.perturb(ppi, ppi_len, p, seed=0)
    assert np.array_equal(a, b) and np.array_equal(a_len, b_len)
    assert np.array_equal(ppi, before) and np.array_equal(ppi_len, before_len)
    assert not np.array_equal(a, ppi), "the damage must actually change something"

    c, _ = R.perturb(ppi, ppi_len, p, seed=1)
    renamed = R.Perturbation(p.name + "2", p.kind, p.value)
    d, _ = R.perturb(ppi, ppi_len, renamed, seed=0)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, d)
    # padding stays all-zero, and a damaged flow is never empty
    valid = np.arange(P.K_MAX)[None, :] < a_len[:, None]
    assert not a[~valid].any()
    assert (a_len >= 1).all()


def test_drop_removes_about_the_requested_fraction():
    ppi, ppi_len = _flows(n=4000)
    _, out_len = R.perturb(ppi, ppi_len, R.Perturbation("d", "drop", 0.2), seed=0)
    removed = 1 - out_len.sum() / ppi_len.sum()
    assert 0.17 < removed < 0.23
    assert (out_len <= ppi_len).all()


def test_non_drop_damage_keeps_every_flow_length():
    ppi, ppi_len = _flows()
    for p in (R.Perturbation("r", "reorder", 0.3), R.Perturbation("j", "ipt_jitter", 0.5)):
        _, out_len = R.perturb(ppi, ppi_len, p, seed=0)
        assert np.array_equal(out_len, ppi_len)


# --- the summary ---------------------------------------------------------------------------


def _report(acc: dict[int, float], auc: float, f1: float, auroc=None, h="h") -> dict:
    return {
        "split": "test_id",
        "n_flows": 10,
        "n_known": 9,
        "n_unknown": 1,
        "acc_at_k": {str(k): v for k, v in acc.items()},
        "auc_k": auc,
        "macro_f1": f1,
        "k95": 8,
        "auroc_at_k": None if auroc is None else {str(k): v for k, v in auroc.items()},
        "hash": h,
    }


def test_summary_reports_each_condition_against_clean():
    perts = R.parse_perturbations([{"name": "d", "kind": "drop", "value": 0.1}])
    reports = {
        "clean": _report({1: 0.1, 3: 0.5, 30: 0.9}, 0.8, 0.85, {3: 0.7, 30: 0.9}, "c"),
        "d": _report({1: 0.1, 3: 0.4, 30: 0.85}, 0.75, 0.80, {3: 0.65, 30: 0.88}, "x"),
    }
    s = R.summarize(reports, perts, ks=(1, 3, 30))
    assert list(s["conditions"]) == ["clean", "d"]
    d = s["conditions"]["d"]
    assert d["kind"] == "drop" and d["value"] == 0.1
    assert d["delta_vs_clean"]["acc_at_k"]["3"] == pytest.approx(-0.1)
    assert d["delta_vs_clean"]["auc_k"] == pytest.approx(-0.05)
    assert d["delta_vs_clean"]["macro_f1"] == pytest.approx(-0.05)
    assert d["delta_vs_clean"]["auroc_at_k"]["30"] == pytest.approx(-0.02)
    clean_delta = s["conditions"]["clean"]["delta_vs_clean"]
    assert all(v == 0 for v in clean_delta["acc_at_k"].values())
    assert s["report_hashes"] == {"clean": "c", "d": "x"}
    assert s["n_unknown"] == 1


def test_summary_needs_clean_and_matching_conditions():
    perts = R.parse_perturbations([{"name": "d", "kind": "drop", "value": 0.1}])
    r = _report({1: 0.1}, 0.8, 0.8)
    with pytest.raises(ValueError, match="clean"):
        R.summarize({"d": r}, perts, ks=(1,))
    with pytest.raises(ValueError, match="do not match"):
        R.summarize({"clean": r}, perts, ks=(1,))
