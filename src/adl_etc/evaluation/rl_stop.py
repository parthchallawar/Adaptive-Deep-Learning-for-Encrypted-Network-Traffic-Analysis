"""P-RL: a learned stopper on the same backbone (plan phase 4, T3; ADR-017;
``docs/design-phase-4.md`` s.3).

Finite-horizon optimal stopping by backward induction. The state at prefix ``K`` is the pair
``(conf_K, margin_K)`` of the calibrated top probability and top-2 margin, each cut into
``n_bins`` equal bins. Fitted on **known validation flows only**; ``lam`` is the price of one
more packet (a packet costs ``lam / 30`` accuracy)::

    Q_commit(K, b) = P(correct | K, b) - lam * K / 30
    Q_cont(K, b)   = E[ V_{K+1} | K, b ]
    V_K            = max(Q_commit, Q_cont)          (last packet: Q_commit)
    stop at the first K where Q_commit >= Q_cont

This is **not** a re-implementation of FastFlow (no recurrent state in the stopper, no online
Q-learning). It answers one question: does an optimal stopper on confidence and margin beat the
learned safety head on the same backbone? The paper must describe it in those words.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from adl_etc.evaluation.replay import ReplayResult, summarize
from adl_etc.evaluation.scores import K_MAX, ScoreArrays
from adl_etc.inference.policy import COMMIT, FORCED_COMMIT


@dataclass(frozen=True)
class RLPolicy:
    q_commit: np.ndarray  # [30, n_bins^2]
    q_cont: np.ndarray  # [30, n_bins^2]; -inf where continuing is impossible
    n_bins: int
    lam: float


def _bins(conf: np.ndarray, margin: np.ndarray, n_bins: int) -> np.ndarray:
    c = np.minimum((np.asarray(conf, dtype=np.float64) * n_bins).astype(np.int64), n_bins - 1)
    m = np.minimum((np.asarray(margin, dtype=np.float64) * n_bins).astype(np.int64), n_bins - 1)
    return np.maximum(c, 0) * n_bins + np.maximum(m, 0)


def fit_rl(val: ScoreArrays, lam: float, *, n_bins: int = 50, min_count: int = 20) -> RLPolicy:
    """Fit on the known-class flows of ``val`` (unknown-class flows are ignored)."""
    known = val.take(val.labels >= 0)
    last = known.last_k
    b = _bins(known.conf, known.margin, n_bins)
    correct = (known.pred == known.labels[:, None]).astype(np.float64)
    s = n_bins * n_bins
    q_commit = np.zeros((K_MAX, s))
    q_cont = np.full((K_MAX, s), -np.inf)
    v_next = None
    for k in range(K_MAX - 1, -1, -1):  # k is 0-based; the prefix length is k + 1
        alive = last >= k + 1
        cnt = np.bincount(b[alive, k], minlength=s)
        hits = np.bincount(b[alive, k], weights=correct[alive, k], minlength=s)
        fallback = correct[alive, k].mean() if alive.any() else 0.0
        qc = np.where(cnt >= min_count, hits / np.maximum(cnt, 1), fallback) - lam * (k + 1) / K_MAX
        q_commit[k] = qc
        go_on = last > k + 1  # flows that have a next packet
        if k < K_MAX - 1 and go_on.any() and v_next is not None:
            c2 = np.bincount(b[go_on, k], minlength=s)
            qn = np.bincount(b[go_on, k], weights=v_next[go_on], minlength=s) / np.maximum(c2, 1)
            q_cont[k] = np.where(c2 >= min_count, qn, v_next[go_on].mean())
        v_here = np.maximum(q_commit[k][b[:, k]], q_cont[k][b[:, k]])
        v_next = np.where(last == k + 1, q_commit[k][b[:, k]], v_here)
    return RLPolicy(q_commit, q_cont, n_bins, float(lam))


def apply_rl(policy: RLPolicy, scores: ScoreArrays) -> ReplayResult:
    n = len(scores)
    last = scores.last_k
    b = _bins(scores.conf, scores.margin, policy.n_bins)
    kk = np.arange(K_MAX)[None, :]
    stop = policy.q_commit[kk, b] >= policy.q_cont[kk, b]
    forced = kk == (last[:, None] - 1)
    stop = (stop | forced) & (kk < last[:, None])
    k = stop.argmax(1) + 1
    early = ~forced[np.arange(n), k - 1]
    decision = np.where(early, COMMIT, FORCED_COMMIT).astype(np.int8)
    return ReplayResult(
        k.astype(np.int64), decision, scores.pred[np.arange(n), k - 1].astype(np.int64)
    )


def sweep_rl(
    val: ScoreArrays, test: ScoreArrays, lams: Sequence[float] | np.ndarray, **kw
) -> list[dict[str, float]]:
    """Fit on ``val`` at each ``lam`` and summarise on ``test``."""
    rows = []
    for lam in lams:
        r = apply_rl(fit_rl(val, float(lam), **kw), test)
        rows.append({"theta": float(lam), **summarize(r, test.labels)})
    return rows
