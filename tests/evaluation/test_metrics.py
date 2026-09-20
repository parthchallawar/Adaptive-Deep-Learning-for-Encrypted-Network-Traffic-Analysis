"""Evaluation metrics (spec 004, plan T6). Every case here is hand-computed,
not cross-checked against another implementation -- these numbers are the
ground truth the rest of the project will be judged against.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from adl_etc.evaluation import metrics as M

# --- K-indexing / padding ------------------------------------------------------


def _logits_from_thresholds(labels: np.ndarray, thresholds: np.ndarray, k_max: int) -> np.ndarray:
    """Builds logits[N, k_max, 2] where flow i's prediction is "wrong" (not
    `labels[i]`) for k < thresholds[i] and "correct" (== labels[i]) for
    k >= thresholds[i], with a clear margin."""
    n = len(labels)
    logits = np.zeros((n, k_max, 2), dtype=np.float64)
    for i in range(n):
        for k in range(1, k_max + 1):
            correct = k >= thresholds[i]
            pred = labels[i] if correct else 1 - labels[i]
            logits[i, k - 1, pred] = 10.0
            logits[i, k - 1, 1 - pred] = -10.0
    return logits


def test_acc_at_k_uses_the_flows_own_ppi_len_not_raw_padding():
    # Flow 0 is long enough for k=3; flow 1 has ppi_len=1, so its k=3
    # evaluation must use index 0 (its only real packet), not index 2 --
    # which is deliberately filled with a wrong-looking prediction to catch
    # a broken padding/capping rule.
    labels = np.array([1, 1])
    ppi_len = np.array([3, 1])
    logits = np.zeros((2, 3, 2))
    logits[0] = [[10, 0], [0, 10], [0, 10]]  # flow 0: correct from k=2 on
    logits[1] = [[0, 10], [10, 0], [10, 0]]  # flow 1: idx0 correct, idx1/2 wrong-looking

    assert M.acc_at_k(logits, labels, ppi_len, k=1) == pytest.approx(0.5)
    assert M.acc_at_k(logits, labels, ppi_len, k=2) == pytest.approx(1.0)
    # If capping were broken, flow 1 would read idx2 (class 0, wrong) here.
    assert M.acc_at_k(logits, labels, ppi_len, k=3) == pytest.approx(1.0)


def test_short_flow_mask_and_accuracy():
    labels = np.array([1, 1])
    ppi_len = np.array([3, 1])
    logits = np.zeros((2, 3, 2))
    logits[0] = [[10, 0], [0, 10], [0, 10]]
    logits[1] = [[0, 10], [10, 0], [10, 0]]

    np.testing.assert_array_equal(M.short_flow_mask(ppi_len, k=3), [False, True])
    # Flow 1 alone at k=3 (capped to idx0, class 1) matches its label.
    assert M.short_flow_accuracy(logits, labels, ppi_len, k=3) == pytest.approx(1.0)
    # No flow is shorter than k=1.
    assert M.short_flow_accuracy(logits, labels, ppi_len, k=1) is None


def test_softmax_sums_to_one_and_matches_hand_values():
    probs = M.softmax(np.array([[1.0, 2.0, 3.0]]))[0]
    assert probs.sum() == pytest.approx(1.0)
    total = math.exp(1) + math.exp(2) + math.exp(3)
    expected = [math.exp(1) / total, math.exp(2) / total, math.exp(3) / total]
    np.testing.assert_allclose(probs, expected, rtol=1e-6)


# --- classification -------------------------------------------------------------


def test_confusion_per_class_f1_macro_f1_balanced_accuracy():
    labels = np.array([0, 0, 1, 1, 2])
    preds = np.array([0, 1, 1, 1, 2])

    cm = M.confusion(preds, labels, classes=[0, 1, 2])
    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 2, 0], [0, 0, 1]])

    f1 = M.per_class_f1(preds, labels, classes=[0, 1, 2])
    assert f1[0] == pytest.approx(2 / 3)
    assert f1[1] == pytest.approx(0.8)
    assert f1[2] == pytest.approx(1.0)

    assert M.macro_f1(preds, labels, classes=[0, 1, 2]) == pytest.approx((2 / 3 + 0.8 + 1.0) / 3)
    expected_balanced_acc = (0.5 + 1.0 + 1.0) / 3
    assert M.balanced_accuracy(preds, labels, classes=[0, 1, 2]) == pytest.approx(
        expected_balanced_acc
    )


def test_classes_with_zero_support_excluded_from_macro_averages():
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])
    # Class 2 never appears in this period's labels.
    f1 = M.per_class_f1(preds, labels, classes=[0, 1, 2])
    assert 2 not in f1
    assert set(f1) == {0, 1}


# --- earliness --------------------------------------------------------------------


def test_k95_monotonicity():
    # 20 two-class flows; flow i "locks in" the correct prediction at k=t_i
    # and stays correct after. Cumulative correct counts by k: 12, 16, 19, 19, 20.
    thresholds = np.array([1] * 12 + [2] * 4 + [3] * 3 + [5] * 1)
    labels = np.ones(20, dtype=np.int64)
    logits = _logits_from_thresholds(labels, thresholds, k_max=5)
    ppi_len = np.full(20, 5)

    accs = [M.acc_at_k(logits, labels, ppi_len, k) for k in range(1, 6)]
    assert accs == pytest.approx([0.6, 0.8, 0.95, 0.95, 1.0])
    assert M.k95(logits, labels, ppi_len) == 3


def test_auc_k_is_mean_of_acc_at_k():
    thresholds = np.array([1, 1, 3])
    labels = np.array([1, 1, 1])
    logits = _logits_from_thresholds(labels, thresholds, k_max=3)
    ppi_len = np.full(3, 3)

    accs = [M.acc_at_k(logits, labels, ppi_len, k) for k in (1, 2, 3)]
    assert M.auc_k(logits, labels, ppi_len) == pytest.approx(np.mean(accs))


def test_harmonic_earliness():
    assert M.harmonic_earliness(0.8, mean_k_value=6, k_max=30) == pytest.approx(0.8)
    # Zero accuracy and zero earliness (mean_k == k_max): denominator is zero.
    assert M.harmonic_earliness(0.0, mean_k_value=30, k_max=30) == 0.0


# --- policy -----------------------------------------------------------------------


def test_policy_summary_statistics():
    k = np.array([1, 2, 3, 4, 5])
    committed = np.array([True, True, False, True, False])
    preds = np.array([0, 1, 1, 0, 1])
    labels = np.array([0, 1, 0, 0, 1])

    assert M.mean_k(k) == pytest.approx(3.0)
    assert M.p95_k(k) == pytest.approx(np.percentile(k, 95))
    assert M.coverage(committed) == pytest.approx(0.6)
    # Committed flows are indices 0, 1, 3 -- all correctly predicted.
    assert M.committed_accuracy(preds, labels, committed) == pytest.approx(1.0)


def test_committed_accuracy_with_no_committed_flows():
    committed = np.zeros(3, dtype=bool)
    assert M.committed_accuracy(np.zeros(3), np.zeros(3), committed) == 0.0


def test_pareto_front_dominance():
    # (12, 0.7) is dominated by (10, 0.9): lower mean_k and higher accuracy.
    mean_ks = [5, 10, 8, 12]
    accuracies = [0.8, 0.9, 0.85, 0.7]
    assert M.pareto_front(mean_ks, accuracies) == [(5, 0.8), (8, 0.85), (10, 0.9)]


# --- open-set / anomaly ------------------------------------------------------------


def test_auroc_with_ties():
    # scores [1,2,2,3], positives at 2 and 3 -- one tie between a positive
    # and a negative at score 2, resolved as half a win by average ranking.
    scores = np.array([1, 2, 2, 3])
    is_positive = np.array([False, True, False, True])
    assert M.auroc(scores, is_positive) == pytest.approx(0.875)


def test_auroc_degenerate_when_one_class_absent():
    assert math.isnan(M.auroc(np.array([1, 2, 3]), np.array([False, False, False])))


def test_aupr_hand_computed():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    is_positive = np.array([True, False, True, False, True])
    assert M.aupr(scores, is_positive) == pytest.approx(34 / 45)


def test_fpr_at_tpr_hand_computed():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    is_positive = np.array([True, False, True, False, True])
    assert M.fpr_at_tpr(scores, is_positive, tpr_target=0.6) == pytest.approx(0.5)
    # Catching every positive (3/3) requires the last threshold, by which
    # point both negatives are included too.
    assert M.fpr_at_tpr(scores, is_positive, tpr_target=1.0) == pytest.approx(1.0)


def test_open_set_f1_reuses_macro_f1():
    preds = np.array([0, -1, 1, -1])
    labels = np.array([0, -1, 1, 1])
    assert M.open_set_f1(preds, labels) == M.macro_f1(preds, labels)
    assert M.open_set_f1(preds, labels) == pytest.approx((2 / 3 + 1.0 + 2 / 3) / 3)


def test_rejection_earliness():
    k = np.array([3, 5, 7, 2])
    is_unknown_true = np.array([False, True, True, True])
    is_rejected = np.array([False, True, False, True])
    # Correctly-rejected unknowns are indices 1 and 3: k values 5 and 2.
    assert M.rejection_earliness(k, is_unknown_true, is_rejected) == pytest.approx(3.5)


def test_rejection_earliness_nan_when_nothing_correctly_rejected():
    k = np.array([1, 2, 3])
    assert math.isnan(
        M.rejection_earliness(k, np.array([True, True, True]), np.array([False, False, False]))
    )


# --- calibration --------------------------------------------------------------------


def test_ece_three_bin_example():
    confidence = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
    correct = np.array([True, False, True, False, True])
    assert M.ece(confidence, correct, n_bins=3) == pytest.approx(0.38)


def test_reliability_curve_empty_bins_are_nan():
    confidence = np.array([0.1, 0.2])
    correct = np.array([True, False])
    centers, acc, conf = M.reliability_curve(confidence, correct, n_bins=3)
    assert len(centers) == 3
    assert acc[0] == pytest.approx(0.5)
    assert math.isnan(acc[1])
    assert math.isnan(acc[2])
