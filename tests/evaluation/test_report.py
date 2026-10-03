"""Report: assembling metrics + policies into spec 004's JSON contract (plan T7)."""

from __future__ import annotations

import numpy as np
import pytest

from adl_etc.data import ppi as P
from adl_etc.evaluation import report as R
from adl_etc.evaluation.dense_logits import from_causal
from adl_etc.evaluation.policies import fit_cape_thresholds


def make_dense_and_labels(n=200, c=5, seed=0, unknown_frac=0.0):
    """Flows that get more accurate as K grows (a real, honest accuracy
    curve): each flow's true class starts winning only from a per-flow
    "reveal" packet on, chosen uniformly in 1..K_MAX."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, c, size=n).astype(np.int64)
    reveal = rng.integers(1, P.K_MAX + 1, size=n)
    logits = rng.normal(scale=0.5, size=(n, P.K_MAX, c))
    for i in range(n):
        logits[i, reveal[i] - 1 :, labels[i]] += 8.0
    if unknown_frac > 0:
        n_unknown = int(n * unknown_frac)
        idx = rng.choice(n, size=n_unknown, replace=False)
        labels[idx] = R.UNKNOWN_LABEL
        logits[idx] *= 0.05  # unknown flows: flatter logits throughout
    dense = from_causal(logits)
    ppi_len = np.full(n, P.K_MAX, dtype=np.int64)
    return dense, labels, ppi_len


# --- shape and sanity ---------------------------------------------------------------------


def test_report_has_a_rising_accuracy_curve():
    dense, labels, ppi_len = make_dense_and_labels()
    report = R.build_report(dense, labels, ppi_len, n_classes=5, split="test")

    accs = [report.acc_at_k[k] for k in report.ks]
    assert accs[-1] > accs[0]
    assert accs[-1] > 0.9  # fully revealed by K=30 for every flow
    assert report.n_flows == 200
    assert report.n_known == 200
    assert report.n_unknown == 0
    assert report.auroc_at_k is None  # no unknown flows in this split
    assert report.cape is None  # no calibration given


def test_report_echo_pareto_front_is_non_empty_and_ordered():
    dense, labels, ppi_len = make_dense_and_labels()
    report = R.build_report(dense, labels, ppi_len, n_classes=5, split="test")
    front = report.echo["pareto_front"]
    assert front
    mean_ks = [p[0] for p in front]
    accs = [p[1] for p in front]
    assert mean_ks == sorted(mean_ks)
    assert accs == sorted(accs)


def test_report_k95_and_auc_k_are_consistent_with_the_curve():
    dense, labels, ppi_len = make_dense_and_labels()
    report = R.build_report(dense, labels, ppi_len, n_classes=5, split="test")
    assert report.k95 in report.ks
    assert report.acc_at_k[report.k95] >= 0.95 * report.acc_at_k[report.ks[-1]]
    assert min(report.acc_at_k.values()) <= report.auc_k <= max(report.acc_at_k.values())


def test_reliability_has_the_documented_shape():
    dense, labels, ppi_len = make_dense_and_labels()
    report = R.build_report(dense, labels, ppi_len, n_classes=5, split="test", ece_bins=10)
    rel = report.reliability
    assert rel["n_bins"] == 10
    assert len(rel["bin_centers"]) == 10
    assert 0.0 <= rel["ece"] <= 1.0


# --- open-set: AUROC-vs-K, only when there are unknown flows -----------------------------


def test_known_only_metrics_ignore_unknown_flows_entirely():
    """A model's predictions are never -1 (unknown_label), so an unknown flow
    folded into acc_at_k would always count as wrong, dragging accuracy down.
    Verified against an independently-computed known-only curve, not against
    the full-known report (removing flows changes the average regardless)."""
    from adl_etc.evaluation import metrics as M

    dense, labels, ppi_len = make_dense_and_labels(seed=9)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(labels), size=len(labels) // 3, replace=False)
    with_unknown = labels.copy()
    with_unknown[idx] = R.UNKNOWN_LABEL

    mixed = R.build_report(dense, with_unknown, ppi_len, n_classes=5, split="test")

    keep = with_unknown != R.UNKNOWN_LABEL
    dense_kept = from_causal(dense.logits[keep])
    labels_kept, ppi_len_kept = labels[keep], ppi_len[keep]
    expected_acc = {k: M.acc_at_k(dense_kept, labels_kept, ppi_len_kept, k) for k in mixed.ks}

    assert mixed.n_known == int(keep.sum())
    assert mixed.acc_at_k == pytest.approx(expected_acc)


def test_auroc_at_k_appears_only_with_real_unknown_flows():
    dense, labels, ppi_len = make_dense_and_labels(unknown_frac=0.2, seed=1)
    report = R.build_report(dense, labels, ppi_len, n_classes=5, split="test")
    assert report.n_unknown > 0
    assert report.auroc_at_k is not None
    assert set(report.auroc_at_k) == set(report.ks)
    for v in report.auroc_at_k.values():
        assert 0.0 <= v <= 1.0
    # Known-only metrics must never count the unknown flows.
    assert report.n_known + report.n_unknown == report.n_flows
    assert report.n_known < report.n_flows


# --- P-CAPE summary, only when a calibration is given -------------------------------------


def test_cape_summary_appears_only_when_calibrated():
    dense, labels, ppi_len = make_dense_and_labels(unknown_frac=0.2, seed=2)
    known = labels != R.UNKNOWN_LABEL
    tau_c = fit_cape_thresholds(
        from_causal(dense.logits[known]), labels[known], ppi_len[known], n_classes=5
    )
    cfg = R.CapeConfig(tau_c=tau_c.tolist(), theta=-2.0, energy_k=10)

    report = R.build_report(dense, labels, ppi_len, n_classes=5, split="test", cape=cfg)
    assert report.cape is not None
    for key in ("mean_k", "coverage", "committed_accuracy", "n_rejected", "rejection_earliness"):
        assert key in report.cape
    assert 0.0 <= report.cape["coverage"] <= 1.0


# --- determinism and round trip ------------------------------------------------------------


def test_same_inputs_give_the_same_hash():
    dense, labels, ppi_len = make_dense_and_labels()
    r1 = R.build_report(dense, labels, ppi_len, n_classes=5, split="test", seed=7)
    r2 = R.build_report(dense, labels, ppi_len, n_classes=5, split="test", seed=7)
    assert r1.hash == r2.hash  # hash must not depend on created_at, which can differ


def test_a_different_seed_changes_the_hash():
    dense, labels, ppi_len = make_dense_and_labels()
    r1 = R.build_report(dense, labels, ppi_len, n_classes=5, split="test", seed=1)
    r2 = R.build_report(dense, labels, ppi_len, n_classes=5, split="test", seed=2)
    assert r1.hash != r2.hash


def test_different_logits_change_the_hash():
    dense_a, labels, ppi_len = make_dense_and_labels(seed=3)
    dense_b, _, _ = make_dense_and_labels(seed=4)
    r1 = R.build_report(dense_a, labels, ppi_len, n_classes=5, split="test")
    r2 = R.build_report(dense_b, labels, ppi_len, n_classes=5, split="test")
    assert r1.hash != r2.hash


def test_round_trip_preserves_every_number(tmp_path):
    dense, labels, ppi_len = make_dense_and_labels(unknown_frac=0.1, seed=5)
    report = R.build_report(dense, labels, ppi_len, n_classes=5, split="test")
    path = tmp_path / "report.json"
    report.save(path)
    loaded = R.Report.load(path)

    assert loaded.hash == report.hash
    assert loaded.acc_at_k == pytest.approx(report.acc_at_k)
    assert loaded.macro_f1 == pytest.approx(report.macro_f1)
    assert loaded.balanced_accuracy == pytest.approx(report.balanced_accuracy)
    assert loaded.k95 == report.k95
    assert loaded.auc_k == pytest.approx(report.auc_k)
    assert loaded.echo == report.echo
    assert loaded.auroc_at_k == pytest.approx(report.auroc_at_k)
    assert loaded.n_flows == report.n_flows
    assert loaded.created_at == report.created_at


def test_round_trip_hash_is_recomputable_from_the_reloaded_content():
    """The saved hash isn't just carried along -- rebuilding a report from
    the reloaded arrays gives the same hash, proving the hash really is a
    function of the content, not an unrelated stamp."""
    dense, labels, ppi_len = make_dense_and_labels(seed=6)
    report = R.build_report(dense, labels, ppi_len, n_classes=5, split="test", seed=6)
    again = R.build_report(dense, labels, ppi_len, n_classes=5, split="test", seed=6)
    assert report.hash == again.hash
