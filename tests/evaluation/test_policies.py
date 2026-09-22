"""P-ECHO and P-CAPE (spec 005, plan T7)."""

from __future__ import annotations

import numpy as np
import pytest

from adl_etc.data import ppi as P
from adl_etc.evaluation import policies as PL
from adl_etc.evaluation.dense_logits import DenseLogits, from_causal, from_per_k


def _causal_step(pred_by_k: list[np.ndarray]) -> DenseLogits:
    """``pred_by_k[k-1]`` is ``[N]`` of one-hot-ish class indices (margin 10)
    for step K -- a convenient way to script a per-flow confidence schedule."""
    n, c = len(pred_by_k[0]), int(max(p.max() for p in pred_by_k) + 1)
    logits = np.full((n, P.K_MAX, c), -10.0)
    for k, preds in enumerate(pred_by_k, start=1):
        logits[np.arange(n), k - 1, preds] = 10.0
    if len(pred_by_k) < P.K_MAX:
        logits[:, len(pred_by_k) :, :] = logits[:, len(pred_by_k) - 1 : len(pred_by_k), :]
    return from_causal(logits)


# --- P-ECHO -----------------------------------------------------------------------


def test_echo_tau_zero_commits_at_the_first_k():
    dense = from_causal(np.zeros((3, P.K_MAX, 2)))
    ppi_len = np.full(3, P.K_MAX)
    result = PL.p_echo(dense, ppi_len, tau=0.0)
    assert np.all(result.k == 1)
    assert np.all(result.committed)


def test_echo_tau_one_never_commits_before_k_max():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, P.K_MAX, 3)) * 3
    dense = from_causal(logits)
    ppi_len = np.full(5, P.K_MAX)
    result = PL.p_echo(dense, ppi_len, tau=1.0)
    assert np.all(result.k == P.K_MAX)
    assert not np.any(result.committed)
    # Still a real prediction (the forced default decision), not garbage.
    last = PL.softmax(dense.at(P.K_MAX, ppi_len))
    np.testing.assert_array_equal(result.prediction, last.argmax(axis=-1))


def test_echo_commits_when_confidence_crosses_tau_and_not_before():
    # Flow 0: confident (~1.0) for class 0 from k=1. Flow 1: confident for
    # class 1 only from k=3 on; uncertain (50/50-ish) before that.
    dense = _causal_step(
        [
            np.array([0, 1]),  # k=1: flow 1 not yet confident -- see below
            np.array([0, 1]),  # k=2
            np.array([0, 1]),  # k=3
        ]
    )
    # Overwrite flow 1's k=1,2 with a low-margin (uncertain) logit pair.
    dense.logits[1, 0, :] = [0.1, 0.0]
    dense.logits[1, 1, :] = [0.1, 0.0]
    ppi_len = np.full(2, P.K_MAX)

    result = PL.p_echo(dense, ppi_len, tau=0.9)
    assert result.k[0] == 1  # confident immediately
    assert result.k[1] == 3  # only confident once the real margin appears
    np.testing.assert_array_equal(result.prediction, [0, 1])
    assert np.all(result.committed)


def test_echo_raises_on_tau_outside_zero_one():
    dense = from_causal(np.zeros((1, P.K_MAX, 2)))
    with pytest.raises(ValueError, match="tau"):
        PL.p_echo(dense, np.array([P.K_MAX]), tau=1.5)


def test_echo_defaults_to_the_arrays_own_grid_not_one_to_thirty():
    """A per-K model only ever offers its own evaluated K's to commit at, and
    p_echo must never call dense.at on a K outside that grid (it would
    raise)."""
    grid = (2, 5, 10, 30)  # deliberately excludes K=1
    per_k = {k: np.array([[10.0, -10.0]]) for k in grid}  # confident from the first grid point
    dense = from_per_k(per_k)
    result = PL.p_echo(dense, np.array([30]), tau=0.9)
    assert result.k[0] == 2  # the grid's own first value (2), never 1


def test_echo_pareto_front_is_non_dominated():
    dense = _causal_step([np.array([0, 1]), np.array([0, 1]), np.array([0, 1])])
    dense.logits[1, 0, :] = [0.0, 0.0]  # flow 1 uncertain at k=1
    ppi_len = np.full(2, P.K_MAX)
    labels = np.array([0, 1])

    front = PL.echo_pareto_front(dense, labels, ppi_len, taus=(0.0, 0.5, 0.99), ks=range(1, 4))
    assert front  # non-empty
    # Pareto-front points are sorted by mean_k, strictly increasing accuracy.
    mean_ks = [p[0] for p in front]
    accs = [p[1] for p in front]
    assert mean_ks == sorted(mean_ks)
    assert accs == sorted(accs)


# --- P-CAPE: threshold fitting ------------------------------------------------------


def test_threshold_for_precision_hand_computed():
    # By confidence descending: 0.9 (correct), 0.7 (wrong), 0.6 (correct), 0.5
    # (correct). Prefix precisions: [1]->1.0, [1,0]->0.5, [1,0,1]->0.667,
    # [1,0,1,1]->0.75. Only the size-1 prefix clears 0.95, so the threshold
    # is that prefix's own (highest) confidence.
    conf = np.array([0.9, 0.6, 0.7, 0.5])
    correct = np.array([True, True, False, True])
    t = PL._threshold_for_precision(conf, correct, precision=0.95)
    assert t == pytest.approx(0.9)


def test_threshold_for_precision_picks_the_most_permissive_qualifying_prefix():
    # Already sorted descending, four *distinct* confidences: three qualifying
    # prefixes (sizes 1, 2, 3, all at precision 1.0), one that doesn't (size
    # 4). The lowest-confidence prefix that still clears 0.95 (size 3, at
    # 0.85) is the answer -- the one that commits most often, not the
    # smallest, most-restrictive qualifying prefix (0.95).
    conf = np.array([0.95, 0.90, 0.85, 0.70])
    correct = np.array([True, True, True, False])
    t = PL._threshold_for_precision(conf, correct, precision=0.95)
    assert t == pytest.approx(0.85)


def test_threshold_for_precision_is_one_when_unreachable():
    conf = np.array([0.9, 0.8])
    correct = np.array([False, False])
    assert PL._threshold_for_precision(conf, correct, precision=0.95) == 1.0


def test_threshold_for_precision_is_one_on_no_evidence():
    assert PL._threshold_for_precision(np.array([]), np.array([]), precision=0.95) == 1.0


def test_fit_cape_thresholds_falls_back_below_min_support():
    """Class 0 has 19 val flows *predicted* as class 0 (spec's own "min
    support 20" example, one short); class 1 has 25. Below-support class 0
    must get exactly the pooled/global threshold, not its own (which, from
    its own noisy data alone, would be unreachable at 1.0); class 1 (25 >=
    20) must get its own class-conditional threshold, computed from just its
    own flows."""
    n0_wrong, n0_correct, n1 = 9, 10, 25  # class 0: 19 total, one short of min_support
    labels = np.array([1] * n0_wrong + [0] * n0_correct + [1] * n1)
    logits = np.zeros((n0_wrong + n0_correct + n1, 2))
    # Predicted class 0, high confidence, but *wrong* (true label 1).
    logits[:n0_wrong] = [[3.0, 0.0]] * n0_wrong
    # Predicted class 0, low confidence, correct (true label 0).
    logits[n0_wrong : n0_wrong + n0_correct] = [[0.3, 0.0]] * n0_correct
    # Predicted class 1, very high confidence, always correct.
    logits[n0_wrong + n0_correct :] = [[0.0, 10.0]] * n1
    n = len(labels)
    dense = from_causal(np.broadcast_to(logits[:, None, :], (n, P.K_MAX, 2)).copy())
    ppi_len = np.full(n, P.K_MAX)

    tau_c = PL.fit_cape_thresholds(dense, labels, ppi_len, n_classes=2, min_support=20)

    # Class 0 alone can never reach 95% precision (its wrong flows outrank its
    # correct ones by confidence): its own, unfallen-back, threshold is 1.0.
    own_class0 = PL.fit_cape_thresholds(dense, labels, ppi_len, n_classes=2, min_support=0)[0]
    assert own_class0 == pytest.approx(1.0)
    assert tau_c[0] != pytest.approx(1.0)  # min_support=20 caught this: not its own value

    # It got exactly the pooled/global value instead.
    probs = PL.softmax(dense.at(P.K_MAX, ppi_len))
    global_conf, global_correct = probs.max(axis=-1), probs.argmax(axis=-1) == labels
    global_tau = PL._threshold_for_precision(global_conf, global_correct, 0.95)
    assert tau_c[0] == pytest.approx(global_tau)

    # Class 1 (25 >= 20) gets its own value, computed from just its own
    # *predicted*-class-1 flows (not merely true-label-1, though they agree
    # here except for class 0's own wrong flows, which are predicted 0).
    mask1 = probs.argmax(axis=-1) == 1
    expected_class1 = PL._threshold_for_precision(global_conf[mask1], global_correct[mask1], 0.95)
    assert tau_c[1] == pytest.approx(expected_class1)


# --- P-CAPE: the policy itself -------------------------------------------------------


def test_cape_commits_to_a_class_only_past_its_own_threshold():
    # Two classes with different tau_c: class 0 needs 0.9, class 1 needs 0.5.
    tau_c = np.array([0.9, 0.5])
    dense = _causal_step([np.array([0, 1])])  # both flows confident (~1.0) from k=1
    ppi_len = np.full(2, P.K_MAX)
    result = PL.p_cape(dense, ppi_len, tau_c, theta=100.0)  # theta huge: energy gate never fires
    assert np.all(result.committed)
    assert np.all(result.k == 1)
    np.testing.assert_array_equal(result.prediction, [0, 1])


def test_cape_uses_the_predicted_classs_own_threshold_not_class_zeros():
    """Flow 0 (class 0, tau_c=0.9) and flow 1 (class 1, tau_c=0.5) both sit at
    confidence ~0.73 -- below class 0's bar, above class 1's. If the wrong
    (mismatched) threshold were ever used, one of these two would flip."""
    tau_c = np.array([0.9, 0.5])
    logits = np.zeros((2, P.K_MAX, 2))
    logits[0, :, :] = [1.0, 0.0]  # softmax ~= [0.731, 0.269]
    logits[1, :, :] = [0.0, 1.0]  # softmax ~= [0.269, 0.731]
    dense = from_causal(logits)
    ppi_len = np.full(2, P.K_MAX)

    result = PL.p_cape(dense, ppi_len, tau_c, theta=1000.0)  # energy gate never fires
    assert not result.committed[0]  # 0.731 < class 0's own 0.9
    assert result.committed[1]  # 0.731 >= class 1's own 0.5
    assert result.k[1] == 1
    assert result.prediction[1] == 1


def test_cape_energy_gate_rejects_a_flat_flow_at_the_fixed_k():
    # Flat logits have *higher* (less negative) energy than concentrated ones
    # (energy_score(flat=[0,0,0]) ~= -1.10, energy_score([10,0,0]) ~= -10.0);
    # theta=-5.0 sits cleanly between the two.
    n = 2
    logits = np.zeros((n, P.K_MAX, 3))
    logits[0] = 0.0  # flow 0: flat everywhere -- higher energy, should be rejected
    logits[1, :, 0] = 10.0  # flow 1: confident for class 0 throughout -- lower energy
    dense = from_causal(logits)
    ppi_len = np.full(n, P.K_MAX)
    tau_c = np.full(3, 2.0)  # unreachable (softmax max prob < 1): no class-cascade commit

    result = PL.p_cape(dense, ppi_len, tau_c, theta=-5.0, energy_k=10)
    assert result.prediction[0] == -1
    assert result.committed[0]
    assert result.k[0] == 10
    # Flow 1 also never clears tau_c=2.0, so it too reaches the energy gate,
    # but its concentrated logits give it low enough energy to pass it.
    assert not result.committed[1]
    assert result.prediction[1] == 0  # the forced default decision, from real logits


def test_cape_never_revokes_an_early_commit_with_the_later_energy_gate():
    """A flow that commits to a known class before energy_k keeps that
    commitment even though its k=10 logits are flat enough to trigger the
    energy gate on their own (energy_score([0,0]) ~= -0.69; theta=-1.0 means
    the gate *would* fire here if it were ever actually checked)."""
    n = 1
    logits = np.zeros((n, P.K_MAX, 2))
    logits[0, 0] = [10.0, -10.0]  # k=1: confident, class 0
    logits[0, 9] = [0.0, 0.0]  # k=10: flat (would trigger the energy gate alone)
    dense = from_causal(logits)
    ppi_len = np.full(n, P.K_MAX)
    tau_c = np.array([0.9, 0.9])

    result = PL.p_cape(dense, ppi_len, tau_c, theta=-1.0, energy_k=10)
    assert result.k[0] == 1
    assert result.prediction[0] == 0
    assert result.committed[0]


def test_cape_raises_when_energy_k_is_not_in_the_evaluated_grid():
    dense = from_per_k({1: np.zeros((1, 2)), 5: np.zeros((1, 2))})
    with pytest.raises(ValueError, match="energy_k"):
        PL.p_cape(dense, np.array([5]), np.array([0.5, 0.5]), theta=0.0, energy_k=10)


# --- energy score --------------------------------------------------------------------


def test_energy_score_is_lower_for_a_confident_flow():
    confident = np.array([[10.0, -10.0]])
    flat = np.array([[0.0, 0.0]])
    assert PL.energy_score(confident)[0] < PL.energy_score(flat)[0]
