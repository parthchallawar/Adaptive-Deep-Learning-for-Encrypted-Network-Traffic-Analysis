"""Traffic sources for the service (spec 016; plan phase 6, T2).

Two sources, one replay path:

- **Shard replay** (:func:`flows_from_shards`), the main demo source. D1 is published as flow
  records, not PCAPs, so replaying stored PPI from a week of ``test_id`` / ``test_drift`` is the
  only way to show labelled in-distribution traffic, real unknown-class flows and real drift.
- **PCAP upload** (:func:`flows_from_pcap`): ``data.pcap_source`` + ``data.flows.FlowBuilder``,
  the phase-1 pipeline unchanged. A flow that carried no payload packet becomes an end event
  with no packets, which the engine stores as ``undecidable`` (spec 016).

Both produce :class:`~adl_etc.service.engine.FlowSpec`\\ s; :class:`Replayer` turns them into
:class:`~adl_etc.service.engine.PacketEvent`\\ s in time order. Packet ``j`` of a flow is sent
``ipt_ms[j]`` after packet ``j - 1`` (divided by ``speed`` and capped at ``max_gap_s``, so one
long-idle flow cannot stall a demo); an end event follows the last packet. A flow the engine has
already decided gets no more events. Live capture is not built (correction S1).
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from adl_etc.data import ppi as P
from adl_etc.service.engine import Engine, FlowSpec, PacketEvent

log = logging.getLogger(__name__)


# --- sources ------------------------------------------------------------------------------------


def period_dir(data_root: str | Path, dataset: str, period: str) -> Path:
    root = Path(data_root)
    for cand in (root / dataset / period, root / period):
        if (cand / "meta.json").is_file():
            return cand
    raise FileNotFoundError(
        f"no shards for {period!r} under {root} (looked for {dataset}/{period}/meta.json)"
    )


def flows_from_shards(
    data_root: str | Path,
    dataset: str,
    period: str,
    *,
    count: int,
    session_id: int,
    known_ids: set[int],
    seed: int = 0,
) -> list[FlowSpec]:
    """``count`` flows of one period, a seeded uniform sample kept in stored (time) order. The
    true class name comes from the shard's own label map; ``known_ids`` (the model's known
    dataset ids) says whether it is a class the model was trained on."""
    import ast
    import json

    from adl_etc.data.tensors import ShardSet

    d = period_dir(data_root, dataset, period)
    ss = ShardSet.open(d)
    try:
        n = len(ss)
        if n == 0:
            raise ValueError(f"{period} has no flows")
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=min(count, n), replace=False))
        ppi = np.asarray(ss.column("ppi")[idx])
        ppi_len = np.asarray(ss.column("ppi_len")[idx]).astype(np.int64)
        label = np.asarray(ss.column("label")[idx]).astype(np.int64)
        stats = np.asarray(ss.column("flowstats")[idx])
        raw_map = ss.meta.get("label_map", {})
    finally:
        ss.close()
    if isinstance(raw_map, str):  # older meta files store it as a repr
        try:
            raw_map = json.loads(raw_map)
        except ValueError:
            raw_map = ast.literal_eval(raw_map)
    names = {int(v): str(k) for k, v in dict(raw_map).items()}
    return [
        FlowSpec(
            key=f"{period}#{int(i)}",
            session_id=session_id,
            ppi=ppi[j],
            ppi_len=int(min(ppi_len[j], P.K_MAX)),
            flowstats=stats[j],
            true_class=names.get(int(label[j]), str(int(label[j]))),
            true_known=int(label[j]) in known_ids,
            end_reason="replay",
        )
        for j, i in enumerate(idx)
    ]


def flows_from_pcap(path: str | Path, *, session_id: int) -> tuple[list[FlowSpec], dict[str, int]]:
    """Every flow of a capture, by start time. The 5-tuple is used only as an in-memory key
    (the engine stores a salted hash of it, never the tuple)."""
    from adl_etc.data.flows import FlowBuilder, FlowRecord
    from adl_etc.data.pcap_source import PcapStats, parse_pcap

    stats = PcapStats()
    builder = FlowBuilder(session_id=session_id)
    records: list[FlowRecord] = []
    for pkt in parse_pcap(path, stats):
        records.extend(builder.add(pkt))
    records.extend(builder.flush())
    records.sort(key=lambda r: r.start_ts)
    specs = []
    for r in records:
        (a_ip, a_port), (b_ip, b_port), proto = r.key
        key = f"{a_ip.hex()}:{a_port}-{b_ip.hex()}:{b_port}/{proto}@{r.start_ts:.6f}"
        specs.append(
            FlowSpec(
                key=key,
                session_id=session_id,
                ppi=np.asarray(r.ppi, dtype=np.int16),
                ppi_len=int(min(r.ppi_len, P.K_MAX)),
                start_ts=float(r.start_ts),
                flowstats=r.flowstats(),
                end_reason=r.end_reason,
            )
        )
    return specs, stats.as_dict()


# --- jobs ----------------------------------------------------------------------------------------


@dataclass
class Job:
    type: str
    total: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    status: str = "queued"  # queued | running | done | failed | cancelled
    progress: int = 0  # flows finished (decided or ended)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    session_id: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "progress": self.progress,
            "total": self.total,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "session_id": self.session_id,
            "detail": self.detail,
        }


# --- the replayer --------------------------------------------------------------------------------


class Replayer:
    """Sends a list of flows to the engine as timed packet events, on its own thread."""

    def __init__(
        self,
        engine: Engine,
        flows: list[FlowSpec],
        starts: np.ndarray,
        *,
        job: Job,
        speed: float = 1.0,
        max_gap_s: float = 0.5,
        pace_limit: int = 256,
        on_done: Callable[[Job], None] | None = None,
    ) -> None:
        if len(flows) != len(starts):
            raise ValueError("one start time per flow")
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.engine, self.flows, self.job = engine, flows, job
        self.starts = np.asarray(starts, dtype=np.float64)
        self.speed, self.max_gap_s = float(speed), float(max_gap_s)
        self.pace_limit = int(pace_limit)
        self.on_done = on_done
        self.thread = threading.Thread(target=self._run, name=f"replay-{job.id}", daemon=True)

    def start(self) -> Replayer:
        self.thread.start()
        return self

    def _gap(self, flow: FlowSpec, j: int) -> float:
        ipt_ms = float(flow.ppi[j, P.IPT_POS]) if j > 0 else 0.0
        return min(max(ipt_ms, 0.0) / 1000.0 / self.speed, self.max_gap_s)

    def _run(self) -> None:
        job, engine = self.job, self.engine
        job.status = "running"
        heap: list[tuple[float, int, int, int]] = []
        seq = 0
        for i, (flow, t) in enumerate(zip(self.flows, self.starts, strict=True)):
            first = 0 if flow.ppi_len > 0 else PacketEvent.END
            heap.append((float(t), seq, i, first))
            seq += 1
        heapq.heapify(heap)
        t0 = time.monotonic()
        # Pace to the engine: a replay is a source we control, so when the engine falls behind
        # the replay clock slows down rather than queueing packets of flows that the backlog is
        # about to decide (which would only deepen the backlog). Spec 016's queue_limit stays
        # the hard ceiling for any source.
        limit = min(engine.cfg.queue_limit, self.pace_limit)
        lag = 0.0
        try:
            while heap and not job.cancel.is_set():
                due, _, i, j = heap[0]
                now = time.monotonic() - t0 - lag
                if due > now:
                    time.sleep(min(due - now, 0.02))
                    continue
                heapq.heappop(heap)
                flow = self.flows[i]
                if engine.is_done(flow.key):
                    job.progress += 1
                    continue
                engine.submit(PacketEvent(flow, j, ts=due))
                if j == PacketEvent.END:
                    job.progress += 1
                    continue
                nxt = j + 1 if j + 1 < flow.ppi_len else PacketEvent.END
                gap = self._gap(flow, nxt) if nxt != PacketEvent.END else 0.05 / self.speed
                heapq.heappush(heap, (due + gap, seq, i, nxt))
                seq += 1
                if engine.queue_size > limit:
                    t_wait = time.monotonic()
                    while engine.queue_size > limit // 2 and not job.cancel.is_set():
                        time.sleep(0.005)  # back-pressure (spec 016): pause, drop nothing
                    lag += time.monotonic() - t_wait  # and the replay clock waits too
            job.status = "cancelled" if job.cancel.is_set() else "done"
        except Exception as exc:  # noqa: BLE001 - reported through the job
            log.exception("replay %s failed", job.id)
            job.status, job.error = "failed", str(exc)
        finally:
            job.finished_at = time.time()
            if self.on_done is not None:
                self.on_done(job)


def arrival_times(n: int, rate: float, seed: int = 0) -> np.ndarray:
    """``n`` flow start times (seconds) of a Poisson stream at ``rate`` flows/s."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.exponential(1.0 / rate, size=n)) - 0.0


def pcap_start_times(flows: list[FlowSpec], speed: float) -> np.ndarray:
    if not flows:
        return np.zeros(0)
    t = np.array([f.start_ts for f in flows], dtype=np.float64)
    return (t - t.min()) / max(speed, 1e-9)
