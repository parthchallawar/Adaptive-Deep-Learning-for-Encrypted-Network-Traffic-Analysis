"""The streaming engine (spec 016; plan phase 6, T2), with a causal stub model.

The central claim: deciding packet by packet, in any interleaving of flows and any batching,
gives the same ``(k, decision, class)`` per flow as the offline vectorised replay
(:func:`adl_etc.evaluation.replay.safe_stop`) on the same scores. The real-model version of this
test is ``test_bundle.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adl_etc.evaluation import replay as R
from adl_etc.inference.policy import COMMIT, FORCED_COMMIT, UNKNOWN_CLASS
from adl_etc.service.engine import (
    BudgetConfig,
    DriftConfig,
    Engine,
    EngineConfig,
    FlowSpec,
    PacketEvent,
    drift_reference_from,
)
from adl_etc.service.ingest import flows_from_pcap
from tests.service.util import K, StubScorer, random_flows, stub_reference, write_multi_flow_pcap

TH_SYN, TH_FIN, TH_PUSH, TH_ACK = 0x02, 0x01, 0x08, 0x10


def manual_engine(scorer=None, *, order: str = "reject_first", theta: float = 0.8, **kw) -> Engine:
    return Engine(
        scorer or StubScorer(),
        config=EngineConfig(order=order, **kw),
        budget=BudgetConfig(mode="manual", theta_init=theta),
    )


def offline(flows: list[FlowSpec], scorer: StubScorer, theta: float, order: str):
    keep = [f for f in flows if f.ppi_len > 0]
    ppi = np.stack([f.ppi for f in keep])
    n = np.array([f.ppi_len for f in keep])
    cols = scorer.score(ppi, n, np.ones(len(keep), dtype=np.int64))
    res = R.safe_stop(
        cols["p_safe"], cols["u"], cols["pred"], n, theta, scorer.theta_reject(0.02), order=order
    )
    return {
        f.key: (int(res.k[i]), int(res.decision[i]), int(res.prediction[i]))
        for i, f in enumerate(keep)
    }


def stream(engine: Engine, flows: list[FlowSpec], seed: int, *, end_events: bool = True) -> dict:
    """Interleave every flow's packets randomly (each flow's own order kept), cut the stream into
    random batches, and send an end event after each flow's last packet (sometimes in the same
    batch, sometimes many batches later)."""
    rng = np.random.default_rng(seed)
    cursors = {f.key: 0 for f in flows}
    pending = list(flows)
    events: list[PacketEvent] = []
    while pending:
        f = pending[int(rng.integers(len(pending)))]
        j = cursors[f.key]
        if j < f.ppi_len:
            events.append(PacketEvent(f, j))
            cursors[f.key] += 1
        else:
            if end_events:
                events.append(PacketEvent(f, PacketEvent.END))
            pending.remove(f)
    out = {}
    i = 0
    while i < len(events):
        size = int(rng.integers(1, 80))
        for rec in engine.process(events[i : i + size]):
            out[rec.key] = rec
        i += size
    return out


@pytest.mark.parametrize("order", ["reject_first", "commit_first"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_streamed_decisions_equal_the_offline_replay(order: str, seed: int) -> None:
    flows = random_flows(400, seed, min_len=1)
    scorer = StubScorer()
    eng = manual_engine(scorer, order=order)
    got = stream(eng, flows, seed)
    want = offline(flows, scorer, 0.8, order)
    assert set(got) == set(want)
    mismatches = [
        (k, (r.k, r.decision.code, r.decision.cls), want[k])
        for k, r in got.items()
        if (r.k, r.decision.code, r.decision.cls) != want[k]
    ]
    assert not mismatches, mismatches[:5]
    codes = {r.decision.code for r in got.values()}
    assert len(codes) >= 3, codes  # the test exercises commits, rejections and forced ones


def test_each_packet_is_scored_once_however_the_stream_is_batched() -> None:
    flows = random_flows(50, 9, min_len=1)
    eng = manual_engine()
    stream(eng, flows, 9)
    assert eng.counters["ignored_events"] >= 0
    assert eng.open_flows == 0  # every flow decided and evicted


def test_a_flow_without_payload_packets_is_stored_as_undecidable() -> None:
    eng = manual_engine()
    empty = FlowSpec("empty", 1, np.zeros((K, 4), np.int16), 0)
    (rec,) = eng.process([PacketEvent(empty, PacketEvent.END)])
    assert rec.name == "undecidable" and rec.k == 0 and rec.decision is None
    assert eng.counters["undecidable"] == 1 and eng.counters["decisions"] == 0


def test_a_30_packet_flow_is_decided_at_30_without_an_end_event() -> None:
    eng = manual_engine(theta=0.995)
    (f,) = random_flows(1, 3, min_len=K, max_len=K)
    f.ppi[0, 2] = 13  # the stub's slowest p_safe (13 % 7 = 6: 0.992 at k = 30) and a low u
    recs = eng.process([PacketEvent(f, j) for j in range(K)])
    assert len(recs) == 1 and recs[0].k == K and recs[0].decision.code == FORCED_COMMIT


def test_events_after_a_decision_are_ignored() -> None:
    eng = manual_engine(theta=0.5)  # commits as soon as k_min allows
    (f,) = random_flows(1, 4, min_len=10, max_len=10)
    f.ppi[0, 2] = 22
    recs = eng.process([PacketEvent(f, j) for j in range(3)])
    assert recs and recs[0].k == 2 and recs[0].decision.code == COMMIT
    assert eng.process([PacketEvent(f, 3), PacketEvent(f, PacketEvent.END)]) == []
    assert eng.counters["ignored_events"] == 2


def test_an_idle_flow_is_forced_at_its_last_packet() -> None:
    clock = [1000.0]
    eng = Engine(
        StubScorer(),
        config=EngineConfig(idle_timeout_s=60),
        budget=BudgetConfig(mode="manual", theta_init=0.99999),
        clock=lambda: clock[0],
    )
    (f,) = random_flows(1, 5, min_len=12, max_len=12)
    f.ppi[0, 2] = 22
    assert eng.process([PacketEvent(f, j) for j in range(4)]) == []
    assert eng.sweep_idle(now=1059.0) == []
    (rec,) = eng.sweep_idle(now=1061.0)
    assert rec.k == 4 and rec.decision.forced and eng.counters["idle_evictions"] == 1


def test_a_model_swap_moves_only_new_flows_to_the_new_model() -> None:
    a, b = StubScorer("model-a"), StubScorer("model-b", shift=1)
    eng = manual_engine(a, theta=0.99999)
    old = random_flows(20, 6, min_len=8, max_len=8)
    early = eng.process([PacketEvent(f, j) for f in old for j in range(3)])
    eng.swap_scorer(b)
    new = random_flows(20, 7, min_len=5, max_len=5)
    recs = eng.process(
        [PacketEvent(f, j) for f in old for j in range(3, 8)]
        + [PacketEvent(f, PacketEvent.END) for f in old]
        + [PacketEvent(f, j) for f in new for j in range(5)]
        + [PacketEvent(f, PacketEvent.END) for f in new]
    )
    by_key = {r.key: r.model_version for r in early + recs}
    assert {by_key[f.key] for f in old} == {"model-a"}
    assert {by_key[f.key] for f in new} == {"model-b"}


def test_beyond_max_flows_new_flows_are_dropped_and_counted() -> None:
    eng = manual_engine(theta=0.99999, max_flows=5)
    flows = random_flows(8, 8, min_len=10, max_len=10)
    eng.process([PacketEvent(f, 0) for f in flows])
    assert eng.open_flows == 5 and eng.counters["dropped_flows"] == 3


def run_budget(target: float, seed: int, theta0: float = 0.95) -> Engine:
    eng = Engine(
        StubScorer(),
        budget=BudgetConfig(
            mode="auto", target=target, window=100, min_flows=50, theta_init=theta0
        ),
    )
    for s in range(6):
        stream(eng, random_flows(200, seed * 10 + s, min_len=K, max_len=K), s)
    return eng


def test_a_lower_budget_lowers_theta_commit_and_mean_k() -> None:
    high = run_budget(12.0, 1)
    low = run_budget(4.0, 1)
    assert low.controller.window >= 10 and high.controller.window >= 10
    assert low.theta_commit < high.theta_commit
    k_low = np.mean([r.k for r in list(low.recent)[-400:]])
    k_high = np.mean([r.k for r in list(high.recent)[-400:]])
    assert k_low < k_high


def test_budget_modes_and_validation() -> None:
    eng = Engine(StubScorer(), budget=BudgetConfig(mode="auto", theta_init=0.9))
    st = eng.set_budget(mode="manual", theta_commit=0.7)
    assert st["mode"] == "manual" and eng.theta_commit == 0.7
    st = eng.set_budget(mode="quality_first")
    assert eng.theta_commit == eng.budget_cfg.quality_theta
    st = eng.set_budget(mode="auto", metric="p95_k", target=5.0)
    assert st["metric"] == "p95_k" and st["target"] == 5.0
    for bad in ({"mode": "fast"}, {"metric": "x"}, {"target": -1.0}, {"theta_commit": 1.5}):
        with pytest.raises(ValueError):
            eng.set_budget(**bad)  # type: ignore[arg-type]


def test_decisions_rows_and_drift_windows_reach_the_sink() -> None:
    rows: list[tuple[str, dict]] = []
    ref = stub_reference()
    eng = Engine(
        StubScorer(),
        budget=BudgetConfig(mode="manual", theta_init=0.8, window=40),
        drift=DriftConfig(
            window=50,
            warmup=1,
            static=drift_reference_from(ref["drift_static"], 50),
            reference_hist=np.asarray(ref["reference_hist"]),
        ),
        sink=lambda t, r: rows.append((t, r)),
    )
    flows = random_flows(260, 11, min_len=1)
    flows.append(FlowSpec("none", 1, np.zeros((K, 4), np.int16), 0))
    stream(eng, flows, 11)
    tables = {t for t, _ in rows}
    assert {"flow", "decision", "decision_trace", "controller_state", "drift_signal"} <= tables
    decisions = [r for t, r in rows if t == "decision"]
    assert len(decisions) == 261
    assert sum(r["decision"] == "undecidable" for r in decisions) == 1
    flow_rows = [r for t, r in rows if t == "flow"]
    assert all(len(r["key_hash"]) == 40 and r["key_hash"] != "none" for r in flow_rows)
    drift = eng.drift_state()
    assert drift["windows_closed"] == 5 and drift["mode"] == "static"
    assert {r["signal_name"] for t, r in rows if t == "drift_signal"} >= {"prior_kl", "mean_k"}
    rejected = [
        r for t, r in rows if t == "decision" and r["decision"] in ("reject", "forced_reject")
    ]
    assert rejected and all(r["class_id"] == UNKNOWN_CLASS for r in rejected)


def test_the_threaded_loop_drains_the_queue() -> None:
    eng = manual_engine()
    eng.start()
    try:
        flows = random_flows(100, 12, min_len=1)
        for f in flows:
            for j in range(f.ppi_len):
                eng.submit(PacketEvent(f, j))
            eng.submit(PacketEvent(f, PacketEvent.END))
        eng.drain()
    finally:
        eng.stop()
    assert eng.counters["decisions"] == 100 and eng.open_flows == 0
    m = eng.metrics()
    assert m["batches"] >= 1 and m["batch_ms_p95"] >= 0


def test_a_pcap_streams_to_the_same_decisions_as_offline(tmp_path: Path) -> None:
    """Spec 016's fixture-PCAP test: three flows, one of them handshake-only."""
    handshake = [(0.0, 0, True, TH_SYN), (0.01, 0, False, TH_SYN | TH_ACK), (0.02, 0, True, TH_ACK)]
    talk = [(0.03 + 0.01 * i, 100 + 37 * i, i % 2 == 0, TH_ACK | TH_PUSH) for i in range(14)]
    close = [(0.5, 0, True, TH_FIN | TH_ACK), (0.51, 0, False, TH_FIN | TH_ACK)]
    flows_spec = [
        handshake + talk + close,
        handshake + [(x + 0.002, s + 5, d, f) for x, s, d, f in talk[:6]] + close,
        handshake + close,  # no payload at all
    ]
    pcap = write_multi_flow_pcap(tmp_path / "three.pcap", flows_spec)
    flows, stats = flows_from_pcap(pcap, session_id=1)
    assert len(flows) == 3 and stats["malformed"] == 0
    scorer = StubScorer()
    eng = manual_engine(scorer)
    got = stream(eng, flows, 0)
    want = offline(flows, scorer, 0.8, "reject_first")
    undecidable = [r for r in got.values() if r.decision is None]
    assert len(undecidable) == 1
    for key, (k, code, cls) in want.items():
        r = got[key]
        assert (r.k, r.decision.code, r.decision.cls) == (k, code, cls)
