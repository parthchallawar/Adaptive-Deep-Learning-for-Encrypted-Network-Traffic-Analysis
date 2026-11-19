"""The streaming inference engine (spec 016; plan phase 6, T2).

Packets arrive one at a time as :class:`PacketEvent`\\ s. For every flow the engine keeps the
packets seen so far; when a flow gets new packets it scores the flow's current prefix with a
:class:`Scorer` and applies spec 009's rule (:func:`adl_etc.inference.policy.decide`) at each new
prefix length. A decided flow is evicted; later packets of it are ignored.

**Correction S3 (version 1 recomputes the prefix).** The scorer runs the full model forward on
the prefix rather than stepping a per-flow KV cache. The PAT is causal, so the output at
position ``k`` is the same either way, and there is no cache to evict or corrupt.

**Streaming vs offline.** Offline replay (:func:`adl_etc.evaluation.replay.safe_stop`) knows each
flow's length and forces a decision at its last packet. A stream does not know a flow has ended
until it is told (an end event, the flow's idle timeout, or the cap ``k_cap``), so the rule runs
with ``last_k = k_cap`` while packets arrive and the forced branch fires when the end is known,
on the scores of the last packet. Both give the same ``(k, decision, class)``;
``tests/service/test_engine.py`` checks it flow for flow.

The engine is torch-free: the model lives behind :class:`Scorer` (``service.bundle`` is the real
one; tests use a deterministic stub). The rule, controller and drift monitor are the phase-4 and
phase-5 modules, imported, never re-implemented.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from adl_etc.data import ppi as P
from adl_etc.inference import drift as DR
from adl_etc.inference.controller import BudgetController, ControllerEvent
from adl_etc.inference.policy import (
    COMMIT,
    FORCED_COMMIT,
    FORCED_REJECT,
    REJECT,
    UNKNOWN_CLASS,
    Decision,
    decide,
)
from adl_etc.service.privacy import KeyHasher

log = logging.getLogger(__name__)

K_MAX = P.K_MAX
UNDECIDABLE = "undecidable"
BUDGET_MODES = ("auto", "manual", "quality_first")
SCORE_FIELDS = ("pred", "conf", "p_safe", "energy", "mahal", "surprise", "u")


# --- what flows in ----------------------------------------------------------------------------


@dataclass
class FlowSpec:
    """One flow as a source knows it. The engine never reads ``ppi`` beyond the packets it has
    been sent; the source keeps the array so events need not copy it."""

    key: str  # unique within the source (a 5-tuple string, or "WEEK-2022-31#123")
    session_id: int
    ppi: np.ndarray  # int16 [30, 4]
    ppi_len: int
    start_ts: float = 0.0
    flowstats: np.ndarray | None = None  # float32 [46]
    true_class: str | None = None
    true_known: bool | None = None
    end_reason: str | None = None


@dataclass
class PacketEvent:
    flow: FlowSpec
    index: int  # 0-based payload packet index; END for "this flow has no more packets"
    ts: float = 0.0

    END = -1

    @property
    def is_end(self) -> bool:
        return self.index == PacketEvent.END


# --- the model interface ------------------------------------------------------------------------


class Scorer(Protocol):
    """What the engine needs from a model bundle."""

    version: str
    class_names: list[str]

    def score(
        self, ppi: np.ndarray, ppi_len: np.ndarray, k_from: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Score columns ``SCORE_FIELDS`` as ``[B, 30]`` float64 arrays (``pred`` as int), valid
        at 1-based positions ``k_from[i]..ppi_len[i]`` of row ``i`` (other cells undefined)."""
        ...

    def theta_reject(self, f: float) -> np.ndarray:
        """The per-K reject threshold ``[30]`` at false-positive target ``f``."""
        ...


# --- configuration --------------------------------------------------------------------------------


@dataclass
class EngineConfig:
    max_batch: int = 64
    max_wait_ms: float = 5.0
    idle_timeout_s: float = 60.0
    max_flows: int = 50_000
    queue_limit: int = 100_000
    order: str = "reject_first"
    k_min: int = 2
    k_ood: int = 3
    trace: bool = True


@dataclass
class BudgetConfig:
    mode: str = "auto"
    metric: str = "mean_k"
    target: float = 6.0
    window: int = 500
    min_flows: int = 200
    kp: float = 3.0
    ki: float = 0.0
    i_max: float = 1.0
    theta_min: float = 0.5
    theta_max: float = 0.999
    quality_theta: float = 0.98
    f: float = 0.02
    theta_init: float = 0.9


@dataclass
class DriftReference:
    """Phase 5's validation reference for one monitor mode."""

    signals: tuple[str, ...]
    reference: dict[str, tuple[float, float]]
    h: float
    window: int = 5000  # the window size the reference was measured on


@dataclass
class DriftConfig:
    window: int = 5000
    warmup: int = 5
    span: int = 3
    static: DriftReference | None = None
    controller: DriftReference | None = None
    reference_hist: np.ndarray | None = None


# --- what flows out -------------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    flow_id: int
    key: str
    session_id: int
    decision: Decision | None  # None for an undecidable flow
    class_name: str | None
    model_version: str
    scores: dict[str, float]  # energy, mahal, surprise, conf at the decision
    decided_at: float
    time_to_decision_ms: float
    true_class: str | None
    true_known: bool | None
    ppi_len_seen: int
    trace: dict[str, list[Any]] | None = None

    @property
    def name(self) -> str:
        return UNDECIDABLE if self.decision is None else self.decision.name

    @property
    def k(self) -> int:
        return 0 if self.decision is None else self.decision.k

    @property
    def rejected(self) -> bool:
        return self.decision is not None and self.decision.rejected

    @property
    def correct(self) -> bool | None:
        """Open-world correctness, when the true class is known (shard replay)."""
        if self.decision is None or self.true_known is None:
            return None
        if not self.true_known:
            return self.decision.rejected
        return (not self.decision.rejected) and self.class_name == self.true_class

    def to_event(self) -> dict[str, Any]:
        d = self.decision
        return {
            "flow_id": self.flow_id,
            "k": self.k,
            "decision": self.name,
            "forced": bool(d.forced) if d else False,
            "class_id": d.cls if d else UNKNOWN_CLASS,
            "class_name": self.class_name,
            "p_safe": _num(d.p_safe) if d else None,
            "u": _num(d.u) if d else None,
            "theta_commit": _num(d.theta_commit) if d else None,
            "theta_reject": _num(d.theta_reject) if d else None,
            "ts": self.decided_at,
            "time_to_decision_ms": self.time_to_decision_ms,
            "true_class": self.true_class,
            "correct": self.correct,
            "model_version": self.model_version,
        }


def _num(x: float) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


@dataclass
class _FlowState:
    flow_id: int
    spec: FlowSpec
    scorer: Scorer
    ppi: np.ndarray = field(default_factory=lambda: np.zeros((K_MAX, P.PPI_CHANNELS), np.int16))
    k: int = 0  # packets received
    scored_k: int = 0  # packets already run through the rule
    ended: bool = False
    first_wall: float = 0.0
    last_wall: float = 0.0
    last: dict[str, float] = field(default_factory=dict)  # scores at position scored_k
    trace: dict[str, list[Any]] = field(default_factory=lambda: {f: [] for f in SCORE_FIELDS})


# --- the engine -----------------------------------------------------------------------------------


class Engine:
    def __init__(
        self,
        scorer: Scorer,
        *,
        config: EngineConfig | None = None,
        budget: BudgetConfig | None = None,
        drift: DriftConfig | None = None,
        sink: Callable[[str, dict[str, Any]], None] | None = None,
        first_flow_id: int = 1,
        clock: Callable[[], float] = time.time,
        key_hasher: Callable[[str], str] | None = None,
    ) -> None:
        self.cfg = config or EngineConfig()
        if self.cfg.order not in ("reject_first", "commit_first"):
            raise ValueError(f"order must be reject_first or commit_first, got {self.cfg.order!r}")
        self.budget_cfg = budget or BudgetConfig()
        self.drift_cfg = drift or DriftConfig()
        self.scorer = scorer
        self.sink = sink  # (table, row) -> None; the database writer, or None
        self.clock = clock
        self.key_hasher = key_hasher or KeyHasher()
        self.listeners: list[Callable[[DecisionRecord], None]] = []
        self._lock = threading.RLock()
        self._flows: dict[str, _FlowState] = {}
        self._done: OrderedDict[str, None] = OrderedDict()
        self._done_cap = 500_000
        self._next_id = int(first_flow_id)
        self._theta_cache: dict[tuple[int, float], np.ndarray] = {}
        self._q: queue.Queue[PacketEvent] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False
        self.recent: deque[DecisionRecord] = deque(maxlen=2000)

        # budget
        self.mode = self.budget_cfg.mode
        if self.mode not in BUDGET_MODES:
            raise ValueError(f"budget mode must be one of {BUDGET_MODES}, got {self.mode!r}")
        self.manual_theta = self.budget_cfg.theta_init
        self.controller = self._new_controller(self.budget_cfg.metric, self.budget_cfg.theta_init)
        self._win_k: list[int] = []
        self._win_rej: list[bool] = []
        self.controller_events: deque[dict[str, Any]] = deque(maxlen=2000)

        # drift
        self.monitor: DR.DriftMonitor | None = None
        self.monitor_mode = ""
        self._drift_rows: list[tuple[int, int, int, float, float]] = []
        self.drift_events: deque[dict[str, Any]] = deque(maxlen=500)
        self.drift_last_stats: dict[str, float | None] = {}
        self._reset_monitor()

        # counters
        self.counters = {
            "events": 0,
            "ignored_events": 0,
            "dropped_flows": 0,
            "decisions": 0,
            "rejected": 0,
            "undecidable": 0,
            "batches": 0,
            "forwards": 0,
            "idle_evictions": 0,
            "labelled": 0,  # decisions whose true class is known (shard replay)
            "correct": 0,
        }
        self.batch_ms: deque[float] = deque(maxlen=2000)
        self._rate: deque[tuple[float, int]] = deque(maxlen=200)

    # -- budget -------------------------------------------------------------------------------

    def _new_controller(self, metric: str, theta: float) -> BudgetController:
        b = self.budget_cfg
        return BudgetController(
            b.target,
            metric=metric,
            theta_init=min(max(theta, b.theta_min), b.theta_max),
            kp=b.kp,
            ki=b.ki,
            i_max=b.i_max,
            theta_min=b.theta_min,
            theta_max=b.theta_max,
            min_flows=b.min_flows,
            f_init=b.f,
        )

    @property
    def theta_commit(self) -> float:
        if self.mode == "manual":
            return self.manual_theta
        if self.mode == "quality_first":
            return self.budget_cfg.quality_theta
        return self.controller.theta

    @property
    def k_cap(self) -> int:
        return self.controller.k_cap if self.mode == "auto" else K_MAX

    @property
    def f(self) -> float:
        return self.controller.f if self.mode == "auto" else self.budget_cfg.f

    def set_budget(
        self,
        *,
        mode: str | None = None,
        metric: str | None = None,
        target: float | None = None,
        theta_commit: float | None = None,
        window: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if mode is not None and mode not in BUDGET_MODES:
                raise ValueError(f"mode must be one of {BUDGET_MODES}")
            if metric is not None and metric not in ("mean_k", "p95_k"):
                raise ValueError("metric must be mean_k or p95_k")
            if target is not None and target <= 0:
                raise ValueError("target must be positive")
            if theta_commit is not None and not 0.0 < theta_commit < 1.0:
                raise ValueError("theta_commit must be in (0, 1)")
            if window is not None:
                if window < 10:
                    raise ValueError("window must be at least 10 decisions")
                self.budget_cfg.window = int(window)
            if metric is not None and metric != self.controller.metric:
                self.controller = self._new_controller(metric, self.theta_commit)
                self.budget_cfg.metric = metric
            if target is not None:
                self.budget_cfg.target = float(target)
                self.controller.set_target(float(target))
            if theta_commit is not None:
                self.manual_theta = float(theta_commit)
            if mode is not None and mode != self.mode:
                if mode == "auto":  # resume from the threshold in force, not from a stale one
                    self.controller.theta_knob.value = min(
                        max(self.theta_commit, self.budget_cfg.theta_min), self.budget_cfg.theta_max
                    )
                elif mode == "manual" and theta_commit is None:
                    self.manual_theta = self.theta_commit
                self.mode = mode
                self._reset_monitor()
            self._win_k, self._win_rej = [], []
            return self.budget_state()

    def budget_state(self) -> dict[str, Any]:
        c = self.controller
        return {
            "mode": self.mode,
            "metric": c.metric,
            "target": c.target,
            "theta_commit": self.theta_commit,
            "controller_theta": c.theta,
            "manual_theta": self.manual_theta,
            "quality_theta": self.budget_cfg.quality_theta,
            "k_cap": self.k_cap,
            "f": self.f,
            "integral": c.integral,
            "window": self.budget_cfg.window,
            "window_fill": len(self._win_k),
            "windows_closed": c.window,
            "gains": {"kp": c.theta_knob.kp, "ki": c.theta_knob.ki},
        }

    # -- drift --------------------------------------------------------------------------------

    def _reset_monitor(self) -> None:
        ref = self.drift_cfg.controller if self.mode == "auto" else self.drift_cfg.static
        self.monitor_mode = "controller" if self.mode == "auto" else "static"
        self._drift_rows = []
        if ref is None or self.drift_cfg.reference_hist is None:
            self.monitor = None
            return
        scale = math.sqrt(ref.window / self.drift_cfg.window) if self.drift_cfg.window else 1.0
        reference = {s: (mu, sd * scale) for s, (mu, sd) in ref.reference.items()}
        self.monitor = DR.DriftMonitor(
            reference,
            signals=ref.signals,
            h=ref.h,
            warmup=self.drift_cfg.warmup,
            span=self.drift_cfg.span,
        )

    def drift_state(self) -> dict[str, Any]:
        mon = self.monitor
        ref = self.drift_cfg.controller if self.mode == "auto" else self.drift_cfg.static
        return {
            "enabled": mon is not None,
            "mode": self.monitor_mode,
            "window": self.drift_cfg.window,
            "reference_window": ref.window if ref else None,
            "reference_rescaled": bool(ref and ref.window != self.drift_cfg.window),
            "window_fill": len(self._drift_rows),
            "windows_closed": mon.window if mon else 0,
            "warmup": self.drift_cfg.warmup,
            "h": mon.cusum[next(iter(mon.cusum))].h if mon and mon.cusum else None,
            "signals": list(mon.signals) if mon else [],
            "cusum": (
                {s: {"pos": c.pos, "neg": c.neg} for s, c in mon.cusum.items()} if mon else {}
            ),
            "reference": (
                {s: {"mean": m, "std": sd} for s, (m, sd) in mon.reference.items()} if mon else {}
            ),
            "last_stats": self.drift_last_stats,
            "level": self.drift_events[-1]["level"] if self.drift_events else "none",
            "events": list(self.drift_events)[-100:],
        }

    # -- model swap -----------------------------------------------------------------------------

    def swap_scorer(self, scorer: Scorer) -> None:
        """New flows use ``scorer``; flows already in flight finish on the one they started on."""
        with self._lock:
            self.scorer = scorer

    # -- the queue and the thread --------------------------------------------------------------

    def submit(self, event: PacketEvent) -> None:
        self._q.put(event)

    @property
    def queue_size(self) -> int:
        return self._q.qsize()

    @property
    def open_flows(self) -> int:
        return len(self._flows)

    def is_done(self, key: str) -> bool:
        return key in self._done

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="inference-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def drain(self, timeout: float = 30.0) -> None:
        """Block until the queue is empty and processed (tests, shutdown)."""
        deadline = time.monotonic() + timeout
        while (self._q.qsize() or self._busy) and time.monotonic() < deadline:
            time.sleep(0.005)

    _busy = False

    def _run(self) -> None:
        last_sweep = time.monotonic()
        while self._running:
            events = self._collect()
            if events:
                self._busy = True
                try:
                    self.process(events)
                except Exception:  # noqa: BLE001 - the loop must survive one bad batch
                    log.exception("inference batch failed")
                finally:
                    self._busy = False
            if time.monotonic() - last_sweep > 1.0:
                last_sweep = time.monotonic()
                try:
                    self.sweep_idle()
                except Exception:  # noqa: BLE001
                    log.exception("idle sweep failed")

    def _collect(self) -> list[PacketEvent]:
        try:
            first = self._q.get(timeout=0.1)
        except queue.Empty:
            return []
        events = [first]
        deadline = time.monotonic() + self.cfg.max_wait_ms / 1000.0
        while len(events) < self.cfg.max_batch:
            remaining = deadline - time.monotonic()
            try:
                events.append(
                    self._q.get(timeout=remaining) if remaining > 0 else self._q.get_nowait()
                )
            except queue.Empty:
                break
        return events

    # -- the core: one batch of events -------------------------------------------------------------

    def process(self, events: Sequence[PacketEvent]) -> list[DecisionRecord]:
        """Apply a batch of events synchronously; returns the decisions it produced."""
        t0 = time.perf_counter()
        out: list[DecisionRecord] = []
        with self._lock:
            now = self.clock()
            touched: dict[str, _FlowState] = {}
            ended: list[_FlowState] = []
            for ev in events:
                self.counters["events"] += 1
                key = ev.flow.key
                if key in self._done:
                    self.counters["ignored_events"] += 1
                    continue
                st = self._flows.get(key)
                if st is None:
                    if len(self._flows) >= self.cfg.max_flows:
                        self.counters["dropped_flows"] += 1
                        self._mark_done(key)
                        continue
                    st = _FlowState(self._next_id, ev.flow, self.scorer, first_wall=now)
                    self._next_id += 1
                    self._flows[key] = st
                st.last_wall = now
                if ev.is_end:
                    st.ended = True
                    ended.append(st)
                    continue
                if st.k >= K_MAX or ev.index != st.k:
                    # out of order or beyond the PPI: the PPI holds the first 30 payload packets
                    self.counters["ignored_events"] += 1
                    continue
                st.ppi[st.k] = ev.flow.ppi[ev.index]
                st.k += 1
                touched[key] = st

            # one forward per model version over every flow that got packets
            by_scorer: dict[int, list[_FlowState]] = {}
            for st in touched.values():
                by_scorer.setdefault(id(st.scorer), []).append(st)
            for group in by_scorer.values():
                out.extend(self._score_group(group, now))

            for st in ended:
                if st.spec.key in self._flows:  # not decided by its own last packet
                    out.append(self._finish(st, now))
        self._after(out)
        dt = (time.perf_counter() - t0) * 1000.0
        self.batch_ms.append(dt)
        self.counters["batches"] += 1
        self._rate.append((time.monotonic(), len(events)))
        return out

    def _score_group(self, group: list[_FlowState], now: float) -> list[DecisionRecord]:
        scorer = group[0].scorer
        ppi = np.stack([st.ppi for st in group])
        k_now = np.array([st.k for st in group], dtype=np.int64)
        k_from = np.array([st.scored_k + 1 for st in group], dtype=np.int64)
        cols = scorer.score(ppi, k_now, k_from)
        self.counters["forwards"] += 1
        theta_c = self.theta_commit
        theta_r = self._theta_reject(scorer, self.f)
        k_cap = self.k_cap
        out = []
        for i, st in enumerate(group):
            for k in range(int(k_from[i]), int(k_now[i]) + 1):
                vals = {f: float(cols[f][i, k - 1]) for f in SCORE_FIELDS}
                st.scored_k = k
                st.last = vals
                if self.cfg.trace:
                    for f in SCORE_FIELDS:
                        st.trace[f].append(vals[f])
                final = st.ended and k == st.k
                last_k = k if (final or k >= min(k_cap, K_MAX)) else min(k_cap, K_MAX)
                dec = decide(
                    k,
                    vals["p_safe"],
                    vals["u"],
                    int(vals["pred"]),
                    theta_commit=theta_c,
                    theta_reject_k=float(theta_r[k - 1]),
                    last_k=last_k,
                    k_min=self.cfg.k_min,
                    k_ood=self.cfg.k_ood,
                    order=self.cfg.order,
                )
                if dec is not None:
                    out.append(self._record(st, dec, now))
                    break
        return out

    def _finish(self, st: _FlowState, now: float) -> DecisionRecord:
        """A flow ended (or went idle) before the rule fired: force at its last packet, or
        store it as undecidable when it never carried a payload packet."""
        if st.scored_k == 0:
            return self._record(st, None, now)
        v = st.last
        theta_r = self._theta_reject(st.scorer, self.f)
        dec = decide(
            st.scored_k,
            v["p_safe"],
            v["u"],
            int(v["pred"]),
            theta_commit=self.theta_commit,
            theta_reject_k=float(theta_r[st.scored_k - 1]),
            last_k=st.scored_k,
            k_min=self.cfg.k_min,
            k_ood=self.cfg.k_ood,
            order=self.cfg.order,
        )
        assert dec is not None  # at k == last_k a decision is always returned
        return self._record(st, dec, now)

    def _theta_reject(self, scorer: Scorer, f: float) -> np.ndarray:
        key = (id(scorer), round(f, 9))
        th = self._theta_cache.get(key)
        if th is None:
            th = np.asarray(scorer.theta_reject(f), dtype=np.float64)
            if len(self._theta_cache) > 64:
                self._theta_cache.clear()
            self._theta_cache[key] = th
        return th

    def _record(self, st: _FlowState, dec: Decision | None, now: float) -> DecisionRecord:
        self._flows.pop(st.spec.key, None)
        self._mark_done(st.spec.key)
        names = st.scorer.class_names
        cls_name = None
        if dec is not None and not dec.rejected and 0 <= dec.cls < len(names):
            cls_name = names[dec.cls]
        v = st.last
        rec = DecisionRecord(
            flow_id=st.flow_id,
            key=st.spec.key,
            session_id=st.spec.session_id,
            decision=dec,
            class_name=cls_name,
            model_version=st.scorer.version,
            scores={f: v.get(f, float("nan")) for f in ("energy", "mahal", "surprise", "conf")},
            decided_at=now,
            time_to_decision_ms=(now - st.first_wall) * 1000.0,
            true_class=st.spec.true_class,
            true_known=st.spec.true_known,
            ppi_len_seen=st.k,
            trace=(
                {f: [_num(x) for x in st.trace[f]] for f in SCORE_FIELDS}
                if self.cfg.trace and dec is not None
                else None
            ),
        )
        rec_ppi = st.ppi
        self._persist(rec, st.spec, rec_ppi)
        return rec

    def _mark_done(self, key: str) -> None:
        self._done[key] = None
        if len(self._done) > self._done_cap:
            self._done.popitem(last=False)

    def sweep_idle(self, now: float | None = None) -> list[DecisionRecord]:
        """Force a decision on every flow silent for ``idle_timeout_s``."""
        out: list[DecisionRecord] = []
        with self._lock:
            now = self.clock() if now is None else now
            stale = [
                st for st in self._flows.values() if now - st.last_wall >= self.cfg.idle_timeout_s
            ]
            for st in stale:
                st.ended = True
                out.append(self._finish(st, now))
                self.counters["idle_evictions"] += 1
        self._after(out)
        return out

    # -- after a decision: controller, drift, persistence, listeners ------------------------------

    def _after(self, records: list[DecisionRecord]) -> None:
        if not records:
            return
        with self._lock:
            for rec in records:
                self.recent.append(rec)
                if rec.decision is None:
                    self.counters["undecidable"] += 1
                    continue
                self.counters["decisions"] += 1
                if rec.rejected:
                    self.counters["rejected"] += 1
                if rec.correct is not None:
                    self.counters["labelled"] += 1
                    self.counters["correct"] += int(rec.correct)
                self._win_k.append(rec.k)
                self._win_rej.append(rec.rejected)
                if len(self._win_k) >= self.budget_cfg.window:
                    self._close_controller_window()
                d = rec.decision
                self._drift_rows.append(
                    (d.k, d.code, d.cls, d.p_safe, rec.scores.get("surprise", float("nan")))
                )
                if self.monitor is not None and len(self._drift_rows) >= self.drift_cfg.window:
                    self._close_drift_window()
        for rec in records:
            for fn in list(self.listeners):
                try:
                    fn(rec)
                except Exception:  # noqa: BLE001 - a listener must not stop the engine
                    log.exception("decision listener failed")

    def _close_controller_window(self) -> None:
        ks = np.asarray(self._win_k)
        rej = np.asarray(self._win_rej)
        self._win_k, self._win_rej = [], []
        if self.mode == "auto":
            ev = self.controller.update(ks, rej)
        else:
            obs = (
                float(ks.mean())
                if self.controller.metric == "mean_k"
                else float(np.percentile(ks, 95))
            )
            tgt = self.controller.target
            ev = ControllerEvent(
                self.controller.window,
                obs,
                (obs - tgt) / tgt,
                self.controller.integral,
                self.theta_commit,
                self.k_cap,
                self.f,
                self.mode,
            )
            self.controller.window += 1
        row = {
            "window_id": ev.window,
            "ts": self.clock(),
            "budget_metric": self.controller.metric,
            "target": self.controller.target,
            "observed": _num(ev.observed),
            "error": _num(ev.error),
            "integral": _num(ev.integral),
            "theta_commit": float(ev.theta),
            "theta_reject_f": float(ev.f),
            "k_cap": int(ev.k_cap),
            "mode": self.mode,
            "reason": ev.reason,
        }
        self.controller_events.append(row)
        self._emit("controller_state", row)

    def _close_drift_window(self) -> None:
        assert self.monitor is not None and self.drift_cfg.reference_hist is not None
        rows = np.asarray(self._drift_rows, dtype=np.float64)
        self._drift_rows = []
        has_nll = "mean_pkt_nll" in self.monitor.signals
        stats = DR.window_stats(
            rows[:, 0],
            rows[:, 1].astype(np.int64),
            rows[:, 2].astype(np.int64),
            rows[:, 3],
            rows[:, 4] if has_nll else None,
            self.drift_cfg.reference_hist,
            theta_commit=self.theta_commit if self.monitor_mode == "controller" else None,
        )
        ev = self.monitor.update(stats)
        self.drift_last_stats = {k: _num(v) for k, v in stats.items()}
        ts = self.clock()
        event = {
            "window": ev.window,
            "ts": ts,
            "level": ev.level,
            "alarms": list(ev.alarms),
            "z": {k: _num(v) for k, v in ev.z.items()},
            "values": {s: _num(stats[s]) for s in self.monitor.signals if s in stats},
        }
        self.drift_events.append(event)
        for s in self.monitor.signals:
            if s not in stats:
                continue
            c = self.monitor.cusum[s]
            self._emit(
                "drift_signal",
                {
                    "window_id": ev.window,
                    "ts": ts,
                    "signal_name": s,
                    "value": _num(stats[s]),
                    "z": _num(ev.z.get(s, float("nan"))),
                    "cusum_pos": c.pos,
                    "cusum_neg": c.neg,
                    "alarm": s in ev.alarms,
                    "level": ev.level,
                },
            )

    def _persist(self, rec: DecisionRecord, spec: FlowSpec, ppi: np.ndarray) -> None:
        if self.sink is None:
            return
        self._emit(
            "flow",
            {
                "id": rec.flow_id,
                "session_id": spec.session_id,
                "key_hash": self.key_hasher(spec.key),
                "start_ts": float(spec.start_ts),
                "ppi": np.ascontiguousarray(ppi, dtype=np.int16).tobytes(),
                "ppi_len": int(rec.ppi_len_seen),
                "flowstats": (
                    None
                    if spec.flowstats is None
                    else np.asarray(spec.flowstats, dtype=np.float32).tobytes()
                ),
                "end_reason": spec.end_reason,
                "true_class": spec.true_class,
                "true_known": spec.true_known,
            },
        )
        d = rec.decision
        self._emit(
            "decision",
            {
                "flow_id": rec.flow_id,
                "k": rec.k,
                "decision": rec.name,
                "forced": bool(d.forced) if d else False,
                "class_id": d.cls if d else UNKNOWN_CLASS,
                "class_name": rec.class_name,
                "p_safe": _num(d.p_safe) if d else None,
                "u": _num(d.u) if d else None,
                "u_threshold": _num(d.theta_reject) if d else None,
                "energy": _num(rec.scores["energy"]) if d else None,
                "mahal": _num(rec.scores["mahal"]) if d else None,
                "surprise": _num(rec.scores["surprise"]) if d else None,
                "model_version": rec.model_version,
                "theta_commit": _num(d.theta_commit) if d else None,
                "decided_at": rec.decided_at,
                "time_to_decision_ms": rec.time_to_decision_ms,
                "correct": rec.correct,
            },
        )
        if rec.trace is not None:
            self._emit("decision_trace", {"flow_id": rec.flow_id, "payload": json.dumps(rec.trace)})

    def _emit(self, table: str, row: dict[str, Any]) -> None:
        if self.sink is not None:
            self.sink(table, row)

    # -- metrics --------------------------------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        lat = np.asarray(self.batch_ms) if self.batch_ms else np.zeros(1)
        now = time.monotonic()
        recent = [(t, n) for t, n in self._rate if now - t <= 5.0]
        events_per_s = sum(n for _, n in recent) / 5.0 if recent else 0.0
        return {
            **self.counters,
            "open_flows": self.open_flows,
            "queue_size": self.queue_size,
            "batch_ms_p50": float(np.percentile(lat, 50)),
            "batch_ms_p95": float(np.percentile(lat, 95)),
            "batch_ms_max": float(lat.max()),
            "events_per_s": events_per_s,
            "model_version": self.scorer.version,
        }


def drift_reference_from(d: Mapping[str, Any], window: int) -> DriftReference:
    """A :class:`DriftReference` from a phase-5 ``drift_monitor.json`` mode block."""
    return DriftReference(
        signals=tuple(d["signals"]),
        reference={s: (float(v["mean"]), float(v["std"])) for s, v in d["reference"].items()},
        h=float(d["h"]),
        window=int(window),
    )


__all__ = [
    "BUDGET_MODES",
    "COMMIT",
    "FORCED_COMMIT",
    "FORCED_REJECT",
    "REJECT",
    "BudgetConfig",
    "DecisionRecord",
    "DriftConfig",
    "DriftReference",
    "Engine",
    "EngineConfig",
    "FlowSpec",
    "PacketEvent",
    "Scorer",
    "UNDECIDABLE",
    "drift_reference_from",
]
