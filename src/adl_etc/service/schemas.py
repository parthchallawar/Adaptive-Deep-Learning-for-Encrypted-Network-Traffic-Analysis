"""Request and response bodies of the service API (spec 016), as Pydantic models.

``DecisionEvent`` is the wire form of a decision: the fields of
:class:`adl_etc.inference.policy.Decision` (``k``, the decision name, class, ``p_safe``, ``u`` and
both thresholds in force) plus the flow id, time and, for shard replay, the true class.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from adl_etc.data import ppi as P


class DecisionEvent(BaseModel):
    flow_id: int
    k: int
    decision: str
    forced: bool
    class_id: int
    class_name: str | None
    p_safe: float | None
    u: float | None
    theta_commit: float | None
    theta_reject: float | None
    ts: float
    time_to_decision_ms: float
    true_class: str | None = None
    correct: bool | None = None
    model_version: str


class ShardReplayRequest(BaseModel):
    period: str = Field(description="a D1 week, e.g. WEEK-2022-31 (test_id) or WEEK-2022-50")
    count: int = Field(2000, ge=1, le=200_000, description="flows to replay")
    rate: float = Field(30.0, gt=0, le=10_000, description="new flows per second")
    speed: float = Field(1.0, gt=0, le=1000, description="time compression of packet gaps")
    seed: int = 0


class BudgetRequest(BaseModel):
    mode: Literal["auto", "manual", "quality_first"] | None = None
    metric: Literal["mean_k", "p95_k"] | None = None
    target: float | None = Field(None, gt=0, le=30)
    theta_commit: float | None = Field(None, gt=0, lt=1)
    window: int | None = Field(None, ge=10, le=100_000)


class ScoreFlowRequest(BaseModel):
    ppi: list[list[float]] = Field(
        description="payload packets in order, each [ipt_ms, direction (+1/-1), size, push]"
    )

    @field_validator("ppi")
    @classmethod
    def _shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) > P.K_MAX:
            v = v[: P.K_MAX]  # the model reads the first 30 payload packets
        for row in v:
            if len(row) != P.PPI_CHANNELS:
                raise ValueError(f"each packet needs {P.PPI_CHANNELS} values")
        return v


class ModelReloadRequest(BaseModel):
    production: str | None = Field(None, description="another production.json to load")
