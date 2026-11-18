"""Spec 011's budget controller (plan phase 4, T5; ``docs/design-phase-4.md`` s.4).

A proportional-integral feedback loop that moves the commit threshold ``theta_commit`` so
the realised packets-per-flow tracks an operator's target, with no labels. It acts **between
windows**: every flow in a window is decided with the knobs in force when the window opened.

Control law, with ``e = (B_obs - B*) / B*`` (positive = over budget)::

    I   = clip(I + e, -i_max, i_max)                       # anti-windup
    phi = logit(theta) - (kp * e + ki * I)                 # over budget -> lower theta
    theta = clip(sigmoid(phi), theta_min, theta_max)

Metrics: ``"mean_k"`` moves ``theta``. ``"p95_k"`` instead moves the hard cap ``k_cap`` (one
packet per window: down while the window's p95 exceeds the target, back up while it is under).
An optional second loop of the same form holds a rejection-rate cap by moving the unknown
score's false-positive target ``f`` (``logit(f)``, lower gains).

**Defaults differ from spec 011's text** (``kp=0.5, ki=0.05, i_max=5``; correction C9). The update
is in velocity form, so ``kp`` already integrates the error and the ``ki * I`` term integrates it a
second time; with ``i_max = 5`` a large first error winds ``I`` up and keeps pushing ``theta`` after
the error has gone (measured on a synthetic plant with a known theta -> mean-K map: 37% worst error
after window 10, 16 windows to settle a target step). ``kp=1.0, ki=0.02, i_max=1.0`` gave 2% and 3
windows, no oscillation. Those numbers come from the synthetic plant, so they are starting values:
re-tune on the validation period (plan D4), never on test.

Safety rails (spec 011): ``theta`` stays in ``[theta_min, theta_max]``; a window with fewer than
``min_flows`` flows changes nothing; an unreachable target saturates ``theta`` and logs one
"budget infeasible" event; alternating-sign errors above 5% for more than 5 windows halve the
gains. Every update is logged as a :class:`ControllerEvent`. No torch, no NumPy needed beyond
the window array.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


@dataclass(frozen=True)
class ControllerEvent:
    window: int
    observed: float
    error: float
    integral: float
    theta: float
    k_cap: int
    f: float
    reason: str  # "update" | "frozen" | "infeasible" | "oscillation" | "update+oscillation"


class PIKnob:
    """One clipped PI loop in logit space (the commit threshold, or ``f``)."""

    def __init__(self, init: float, lo: float, hi: float, kp: float, ki: float, i_max: float):
        if not 0.0 < lo <= init <= hi < 1.0:
            raise ValueError(f"need 0 < lo <= init <= hi < 1, got lo={lo}, init={init}, hi={hi}")
        self.value, self.lo, self.hi = float(init), float(lo), float(hi)
        self.kp, self.ki, self.i_max = float(kp), float(ki), float(i_max)
        self.integral = 0.0

    def step(self, e: float) -> bool:
        """Apply one error; returns True when the knob is pinned on a rail while the error
        still pushes into it (the target cannot be reached)."""
        self.integral = max(-self.i_max, min(self.i_max, self.integral + e))
        phi = logit(self.value) - (self.kp * e + self.ki * self.integral)
        raw = sigmoid(phi)
        self.value = max(self.lo, min(self.hi, raw))
        return (raw <= self.lo and e > 0) or (raw >= self.hi and e < 0)


class BudgetController:
    def __init__(
        self,
        target: float,
        *,
        metric: str = "mean_k",
        theta_init: float = 0.9,
        kp: float = 1.0,
        ki: float = 0.02,
        i_max: float = 1.0,
        theta_min: float = 0.5,
        theta_max: float = 0.999,
        min_flows: int = 200,
        k_cap_max: int = 30,
        k_cap_min: int = 3,
        reject_cap: float | None = None,
        f_init: float = 0.02,
        f_bounds: tuple[float, float] = (0.001, 0.2),
        f_gains: tuple[float, float] = (0.2, 0.02),
    ) -> None:
        if metric not in ("mean_k", "p95_k"):
            raise ValueError(f"metric must be 'mean_k' or 'p95_k', got {metric!r}")
        if target <= 0:
            raise ValueError(f"target must be positive, got {target}")
        self.metric, self.target, self.min_flows = metric, float(target), int(min_flows)
        self.theta_knob = PIKnob(theta_init, theta_min, theta_max, kp, ki, i_max)
        self.k_cap_max, self.k_cap_min = int(k_cap_max), int(k_cap_min)
        self.k_cap = int(k_cap_max)
        self.reject_cap = reject_cap
        self.f_knob = PIKnob(f_init, f_bounds[0], f_bounds[1], f_gains[0], f_gains[1], i_max)
        self.window = 0
        self._recent: deque[float] = deque(maxlen=6)
        self._infeasible_reported = False
        self.events: list[ControllerEvent] = []

    # -- the knobs the policy reads -------------------------------------------------

    @property
    def theta(self) -> float:
        return self.theta_knob.value

    @property
    def f(self) -> float:
        return self.f_knob.value

    @property
    def integral(self) -> float:
        return self.theta_knob.integral

    def set_target(self, target: float) -> None:
        if target <= 0:
            raise ValueError(f"target must be positive, got {target}")
        self.target = float(target)
        self._recent.clear()
        self._infeasible_reported = False

    # -- one window -----------------------------------------------------------------

    def update(self, window_k: np.ndarray, rejected: np.ndarray | None = None) -> ControllerEvent:
        """Close a window: ``window_k`` is the decision K of every flow in it, ``rejected``
        (optional, bool) which of them were rejected, for the rejection-rate cap."""
        window_k = np.asarray(window_k)
        w = self.window
        self.window += 1
        if len(window_k) < self.min_flows:
            return self._log(w, float("nan"), 0.0, "frozen")

        obs = (
            float(np.mean(window_k))
            if self.metric == "mean_k"
            else float(np.percentile(window_k, 95))
        )
        e = (obs - self.target) / self.target
        reason = "update"
        if self.metric == "mean_k":
            pinned = self.theta_knob.step(e)
        else:
            self.k_cap = (
                max(self.k_cap_min, self.k_cap - 1)
                if e > 0
                else min(self.k_cap_max, self.k_cap + 1)
            )
            pinned = e > 0 and self.k_cap <= self.k_cap_min
        if self.reject_cap is not None and rejected is not None:
            rate = float(np.mean(rejected))
            self.f_knob.step((rate - self.reject_cap) / max(self.reject_cap, 1e-9))
        if pinned and not self._infeasible_reported:
            self._infeasible_reported = True
            reason = "infeasible"
        elif not pinned:
            self._infeasible_reported = False

        self._recent.append(e)
        if self._oscillating():
            self.theta_knob.kp *= 0.5
            self.theta_knob.ki *= 0.5
            self._recent.clear()
            reason = "update+oscillation" if reason == "update" else f"{reason}+oscillation"
        return self._log(w, obs, e, reason)

    def _oscillating(self) -> bool:
        if len(self._recent) < 6:
            return False
        r = list(self._recent)
        return all(abs(x) > 0.05 for x in r) and all(r[i] * r[i + 1] < 0 for i in range(len(r) - 1))

    def _log(self, window: int, obs: float, e: float, reason: str) -> ControllerEvent:
        ev = ControllerEvent(window, obs, e, self.integral, self.theta, self.k_cap, self.f, reason)
        self.events.append(ev)
        return ev

    # -- persistence ----------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "target": self.target,
            "window": self.window,
            "k_cap": self.k_cap,
            "theta": {**_knob_state(self.theta_knob)},
            "f": {**_knob_state(self.f_knob)},
            "recent": list(self._recent),
            "infeasible_reported": self._infeasible_reported,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["metric"] != self.metric:
            raise ValueError(
                f"state is for metric {state['metric']!r}, controller has {self.metric!r}"
            )
        self.target, self.window, self.k_cap = (
            float(state["target"]),
            int(state["window"]),
            int(state["k_cap"]),
        )
        _load_knob(self.theta_knob, state["theta"])
        _load_knob(self.f_knob, state["f"])
        self._recent = deque(state["recent"], maxlen=6)
        self._infeasible_reported = bool(state["infeasible_reported"])

    def event_log(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self.events]


def _knob_state(k: PIKnob) -> dict[str, float]:
    return {"value": k.value, "integral": k.integral, "kp": k.kp, "ki": k.ki}


def _load_knob(k: PIKnob, s: dict[str, float]) -> None:
    k.value, k.integral, k.kp, k.ki = s["value"], s["integral"], s["kp"], s["ki"]
