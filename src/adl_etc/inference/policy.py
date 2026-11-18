"""Spec 009's three-way stopping rule, for one flow at one prefix length (plan phase 4,
T3; ``docs/design-phase-4.md`` s.2).

This is the single definition of the rule. Offline evaluation replays it vectorised in
:mod:`adl_etc.evaluation.replay` and the service will call :func:`decide` per packet;
``tests/inference/test_policy.py`` asserts the two agree on thousands of flows. No torch.

For a flow of ``L = min(k_cap, ppi_len)`` usable packets, at prefix ``k`` (1-based)::

    reject_now = k >= k_ood and u_k >= theta_reject(k)
    commit_now = k >= k_min and p_safe_k >= theta_commit
    reject_first (default): REJECT if reject_now, elif COMMIT if commit_now
    commit_first:           COMMIT if commit_now, elif REJECT if reject_now
    at k == L with nothing fired: forced REJECT if u_k >= theta_reject(k) else forced COMMIT

The forced branch ignores ``k_ood`` (spec 009), so it is the only way a flow shorter than
``k_ood`` packets can be rejected. Scores that are NaN count as ``p_safe = 0`` and
``u = +inf`` (the flow keeps reading, then is force-rejected).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

COMMIT = 0
REJECT = 1
FORCED_COMMIT = 2
FORCED_REJECT = 3
DECISION_NAMES = {
    COMMIT: "commit",
    REJECT: "reject",
    FORCED_COMMIT: "forced_commit",
    FORCED_REJECT: "forced_reject",
}

UNKNOWN_CLASS = -1


@dataclass(frozen=True)
class Decision:
    k: int
    code: int
    cls: int  # predicted class, or UNKNOWN_CLASS for a rejection
    p_safe: float
    u: float
    theta_commit: float
    theta_reject: float

    @property
    def rejected(self) -> bool:
        return self.code in (REJECT, FORCED_REJECT)

    @property
    def forced(self) -> bool:
        return self.code in (FORCED_COMMIT, FORCED_REJECT)

    @property
    def name(self) -> str:
        return DECISION_NAMES[self.code]


def decide(
    k: int,
    p_safe_k: float,
    u_k: float,
    pred_k: int,
    *,
    theta_commit: float,
    theta_reject_k: float,
    last_k: int,
    k_min: int = 2,
    k_ood: int = 3,
    order: str = "reject_first",
) -> Decision | None:
    """The decision after ``k`` packets, or ``None`` to keep reading. ``last_k`` is
    ``min(k_cap, ppi_len)``; at ``k == last_k`` a decision is always returned."""
    if order not in ("reject_first", "commit_first"):
        raise ValueError(f"order must be 'reject_first' or 'commit_first', got {order!r}")
    if not 1 <= k <= last_k:
        raise ValueError(f"k={k} outside 1..{last_k}")
    p = 0.0 if math.isnan(p_safe_k) else float(p_safe_k)
    u = math.inf if math.isnan(u_k) else float(u_k)

    def make(code: int) -> Decision:
        cls = UNKNOWN_CLASS if code in (REJECT, FORCED_REJECT) else int(pred_k)
        return Decision(k, code, cls, p, u, theta_commit, theta_reject_k)

    reject_now = k >= k_ood and u >= theta_reject_k
    commit_now = k >= k_min and p >= theta_commit
    if order == "reject_first":
        if reject_now:
            return make(REJECT)
        if commit_now:
            return make(COMMIT)
    else:
        if commit_now:
            return make(COMMIT)
        if reject_now:
            return make(REJECT)
    if k == last_k:
        return make(FORCED_REJECT if u >= theta_reject_k else FORCED_COMMIT)
    return None
