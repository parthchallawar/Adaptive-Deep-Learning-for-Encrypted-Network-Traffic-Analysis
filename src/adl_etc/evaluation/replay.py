"""Vectorised replay of stopping policies over saved scores (plan phase 4, T3;
``docs/design-phase-4.md`` s.2).

Everything here is a pure function of a :class:`~adl_etc.evaluation.scores.ScoreArrays`
(or its columns): no model, no torch, seconds per sweep.

- :func:`safe_stop`: spec 009's rule, the same one :func:`adl_etc.inference.policy.decide`
  applies per flow (a test asserts they agree decision for decision).
- :func:`echo`, :func:`cape`: P-ECHO and P-CAPE on the saved ``conf``/``pred``/``energy_raw``
  columns. With ``force_at="grid_end"`` they reproduce ``evaluation.policies`` exactly
  (a flow that never commits is reported at K = 30); the default ``"flow_end"`` reports it
  at its own last packet, which is what Safe-Stop does, so mean K is comparable.

Decision codes are those of :mod:`adl_etc.inference.policy`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from adl_etc.evaluation import metrics as M
from adl_etc.evaluation.scores import K_MAX, ScoreArrays
from adl_etc.inference.policy import COMMIT, FORCED_COMMIT, FORCED_REJECT, REJECT

UNKNOWN = -1


@dataclass(frozen=True)
class ReplayResult:
    k: np.ndarray  # [N] int64, 1-based prefix length of the decision
    decision: np.ndarray  # [N] int8, policy.COMMIT / REJECT / FORCED_COMMIT / FORCED_REJECT
    prediction: np.ndarray  # [N] int64, class index, -1 = rejected as unknown
    n_nan: int = 0  # decisions made on NaN scores (counted, never hidden)

    @property
    def rejected(self) -> np.ndarray:
        return (self.decision == REJECT) | (self.decision == FORCED_REJECT)

    @property
    def forced(self) -> np.ndarray:
        return (self.decision == FORCED_COMMIT) | (self.decision == FORCED_REJECT)

    @property
    def committed(self) -> np.ndarray:
        """Decided before being forced (what ``metrics.coverage`` counts)."""
        return ~self.forced

    def __len__(self) -> int:
        return len(self.k)


def _first(mask: np.ndarray) -> np.ndarray:
    """1-based K of the first True per row; ``K_MAX + 1`` where none."""
    return np.where(mask.any(1), mask.argmax(1) + 1, K_MAX + 1)


def safe_stop(
    p_safe: np.ndarray,
    u: np.ndarray,
    pred: np.ndarray,
    ppi_len: np.ndarray,
    theta_commit: float,
    theta_reject: np.ndarray,
    *,
    k_min: int = 2,
    k_ood: int = 3,
    order: str = "reject_first",
    k_cap: int = K_MAX,
) -> ReplayResult:
    """Spec 009's rule over ``[N, 30]`` score columns. ``theta_reject`` is the per-K
    threshold ``[30]`` (pass ``np.full(30, np.inf)`` to switch rejection off)."""
    if order not in ("reject_first", "commit_first"):
        raise ValueError(f"order must be 'reject_first' or 'commit_first', got {order!r}")
    n = len(ppi_len)
    rows = np.arange(n)
    last = np.minimum(np.asarray(ppi_len), k_cap).astype(np.int64)
    ks = np.arange(1, K_MAX + 1)[None, :]
    live = ks <= last[:, None]
    p_raw = np.asarray(p_safe, dtype=np.float64)
    u_raw = np.asarray(u, dtype=np.float64)
    n_nan = int((np.isnan(p_raw) & live).any(1).sum() + (np.isnan(u_raw) & live).any(1).sum())
    p = np.nan_to_num(p_raw, nan=0.0)
    uu = np.where(np.isnan(u_raw), np.inf, u_raw)
    theta_reject = np.asarray(theta_reject, dtype=np.float64)

    can_reject = live & (ks >= k_ood) & (uu >= theta_reject[None, :])
    can_commit = live & (ks >= k_min) & (p >= theta_commit)
    k_rej, k_com = _first(can_reject), _first(can_commit)
    reject_wins = (k_rej <= k_com) if order == "reject_first" else (k_rej < k_com)
    k_early = np.minimum(k_rej, k_com)
    early = k_early <= K_MAX

    k = np.where(early, k_early, last)
    forced_reject = ~early & (uu[rows, last - 1] >= theta_reject[last - 1])
    decision = np.where(
        early,
        np.where(reject_wins, REJECT, COMMIT),
        np.where(forced_reject, FORCED_REJECT, FORCED_COMMIT),
    ).astype(np.int8)
    rejected = (decision == REJECT) | (decision == FORCED_REJECT)
    prediction = np.where(rejected, UNKNOWN, np.asarray(pred)[rows, k - 1]).astype(np.int64)
    return ReplayResult(k.astype(np.int64), decision, prediction, n_nan)


def echo(
    conf: np.ndarray,
    pred: np.ndarray,
    ppi_len: np.ndarray,
    tau: float,
    *,
    k_cap: int = K_MAX,
    force_at: str = "flow_end",
) -> ReplayResult:
    """P-ECHO: commit at the first K whose calibrated top probability is ``>= tau``."""
    if force_at not in ("flow_end", "grid_end"):
        raise ValueError(f"force_at must be 'flow_end' or 'grid_end', got {force_at!r}")
    n = len(ppi_len)
    rows = np.arange(n)
    last = np.minimum(np.asarray(ppi_len), k_cap).astype(np.int64)
    ks = np.arange(1, K_MAX + 1)[None, :]
    hit = (ks <= last[:, None]) & (np.asarray(conf, dtype=np.float64) >= tau)
    k_hit = _first(hit)
    early = k_hit <= K_MAX
    k_forced = last if force_at == "flow_end" else np.full(n, k_cap)
    k = np.where(early, k_hit, k_forced)
    pred_k = np.asarray(pred)[rows, np.minimum(k, last) - 1]
    decision = np.where(early, COMMIT, FORCED_COMMIT).astype(np.int8)
    return ReplayResult(k.astype(np.int64), decision, pred_k.astype(np.int64))


def cape(
    conf: np.ndarray,
    pred: np.ndarray,
    energy_raw: np.ndarray,
    ppi_len: np.ndarray,
    tau_c: np.ndarray,
    theta: float,
    *,
    energy_k: int = 10,
    k_cap: int = K_MAX,
    force_at: str = "flow_end",
) -> ReplayResult:
    """P-CAPE: commit at the first K where the predicted class's own ``tau_c`` is cleared;
    a flow not yet committed at ``energy_k`` is rejected if its raw energy there is
    ``>= theta`` (an early commitment is never revisited, as in ``policies.p_cape``)."""
    if force_at not in ("flow_end", "grid_end"):
        raise ValueError(f"force_at must be 'flow_end' or 'grid_end', got {force_at!r}")
    n = len(ppi_len)
    rows = np.arange(n)
    last = np.minimum(np.asarray(ppi_len), k_cap).astype(np.int64)
    ks = np.arange(1, K_MAX + 1)[None, :]
    pred = np.asarray(pred)
    live = ks <= last[:, None]
    hit = live & (np.asarray(conf, dtype=np.float64) >= np.asarray(tau_c)[pred])
    k_hit = _first(hit)
    # the gate looks at position min(energy_k, L); it fires only for flows not committed by then
    gate_pos = np.minimum(energy_k, last)
    gated = (k_hit > gate_pos) & (np.asarray(energy_raw)[rows, gate_pos - 1] >= theta)
    if force_at == "grid_end":
        gate_k = np.full(n, energy_k)
        k_forced = np.full(n, k_cap)
    else:
        gate_k, k_forced = gate_pos, last
    early_commit = (k_hit <= K_MAX) & ~gated
    k = np.where(gated, gate_k, np.where(early_commit, k_hit, k_forced))
    decision = np.where(gated, REJECT, np.where(early_commit, COMMIT, FORCED_COMMIT)).astype(
        np.int8
    )
    pos = np.minimum(k, last) - 1
    prediction = np.where(gated, UNKNOWN, pred[rows, pos])
    return ReplayResult(k.astype(np.int64), decision, prediction.astype(np.int64))


# --- metrics on a result -----------------------------------------------------------------------


def summarize(result: ReplayResult, labels: np.ndarray) -> dict[str, float]:
    """The numbers every policy table reports. ``labels`` are model indices, ``-1`` for
    unknown-class flows. A rejected known flow counts as an error."""
    labels = np.asarray(labels)
    known = labels >= 0
    unk = ~known
    out: dict[str, float] = {
        "n": float(len(labels)),
        "mean_k": M.mean_k(result.k),
        "p95_k": M.p95_k(result.k),
        "forced_rate": float(result.forced.mean()),
    }
    if known.any():
        out["known_accuracy"] = float((result.prediction[known] == labels[known]).mean())
        out["known_reject_rate"] = float(result.rejected[known].mean())
        out["mean_k_known"] = M.mean_k(result.k[known])
    if unk.any():
        out["unknown_reject_rate"] = float(result.rejected[unk].mean())
        out["unknown_committed"] = 1.0 - out["unknown_reject_rate"]
        out["mean_k_unknown"] = M.mean_k(result.k[unk])
        out["rejection_k"] = M.rejection_earliness(result.k, unk, result.rejected)
    if known.any() and unk.any():
        out["open_set_macro_f1"] = M.open_set_f1(result.prediction, labels)
    return out


def closed_world(scores: ScoreArrays) -> ScoreArrays:
    """Known-class flows only (H2's main test turns rejection off on these)."""
    return scores.take(scores.labels >= 0)


NO_REJECTION = np.full(K_MAX, np.inf)


def sweep_theta_commit(
    scores: ScoreArrays,
    thetas: Sequence[float] | np.ndarray,
    *,
    theta_reject: np.ndarray = NO_REJECTION,
    u: np.ndarray | None = None,
    **kw,
) -> list[dict[str, float]]:
    """Safe-Stop at each ``theta_commit``; one summary dict per value (with ``theta``)."""
    uu = u if u is not None else np.zeros((len(scores), K_MAX))
    rows = []
    for t in thetas:
        r = safe_stop(scores.p_safe, uu, scores.pred, scores.ppi_len, float(t), theta_reject, **kw)
        rows.append({"theta": float(t), **summarize(r, scores.labels)})
    return rows


def sweep_echo(
    scores: ScoreArrays, taus: Sequence[float] | np.ndarray, **kw
) -> list[dict[str, float]]:
    rows = []
    for t in taus:
        r = echo(scores.conf, scores.pred, scores.ppi_len, float(t), **kw)
        rows.append({"theta": float(t), **summarize(r, scores.labels)})
    return rows


def logit_grid(n: int = 200, lo: float = 0.5, hi: float = 0.999) -> np.ndarray:
    """``n`` thresholds evenly spaced in logit space over ``[lo, hi]`` (spec 009)."""
    a, b = np.log(lo / (1 - lo)), np.log(hi / (1 - hi))
    return 1.0 / (1.0 + np.exp(-np.linspace(a, b, n)))
