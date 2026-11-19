"""Spec 018's database (plan phase 6, T1): schema, the queued writer, WAL reads, the version
guard, and the dashboard's query speed."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import insert, text

from adl_etc.service.db import models as m
from adl_etc.service.db.store import SchemaVersionError, Store, create_schema, make_engine
from adl_etc.service.privacy import KeyHasher


def decision_row(flow_id: int, t: float, k: int = 5, name: str = "commit") -> dict:
    return {
        "flow_id": flow_id,
        "k": k,
        "decision": name,
        "forced": name.startswith("forced"),
        "class_id": -1 if "reject" in name else 3,
        "class_name": None if "reject" in name else "class3",
        "p_safe": 0.95,
        "u": -1.0,
        "u_threshold": 1.5,
        "energy": -2.0,
        "mahal": 0.1,
        "surprise": 3.1,
        "model_version": "abc",
        "theta_commit": 0.93,
        "decided_at": t,
        "time_to_decision_ms": 12.0,
        "correct": True,
    }


def flow_row(i: int, session: int) -> dict:
    return {
        "id": i,
        "session_id": session,
        "key_hash": "0" * 40,
        "start_ts": 0.0,
        "ppi": np.zeros((30, 4), np.int16).tobytes(),
        "ppi_len": 5,
    }


@pytest.fixture
def store(tmp_path: Path):
    s = Store(f"sqlite:///{tmp_path / 'svc.db'}").start()
    yield s
    s.stop()


def test_a_new_database_has_every_table_and_its_version(store: Store) -> None:
    assert store.engine is not None
    with store.engine.connect() as c:
        names = {r[0] for r in c.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        mode = c.execute(text("PRAGMA journal_mode")).scalar()
    assert set(m.TABLE_ORDER) | {"schema_version"} <= names
    assert mode == "wal"
    assert store.count("schema_version") == 1 and store.next_flow_id() == 1


def test_fifty_thousand_decisions_through_the_queue_in_under_ten_seconds(store: Store) -> None:
    session = store.new_session("shards", "WEEK-2022-31")
    t0 = time.perf_counter()
    now = time.time()
    for i in range(1, 50_001):
        store.put("flow", flow_row(i, session))
        store.put("decision", decision_row(i, now))
    store.flush(timeout=60)
    elapsed = time.perf_counter() - t0
    assert store.count("decision") == 50_000 and store.count("flow") == 50_000
    assert elapsed < 10.0, f"{elapsed:.1f}s"
    assert store.next_flow_id() == 50_001
    s = store.stats(window_s=60)
    assert s["n"] == 50_000 and s["mean_k"] == 5 and s["commit_rate"] == 1.0


def test_a_reader_sees_rows_while_the_writer_is_busy(store: Store) -> None:
    session = store.new_session("shards", "x")
    stop = threading.Event()

    def write() -> None:
        i = 1
        while not stop.is_set():
            store.put("flow", flow_row(i, session))
            store.put("decision", decision_row(i, time.time()))
            i += 1
            if i % 500 == 0:
                time.sleep(0.01)

    th = threading.Thread(target=write)
    th.start()
    seen = []
    try:
        deadline = time.time() + 5
        while time.time() < deadline and len(seen) < 5:
            t0 = time.perf_counter()
            rows = store.recent_flows(0, 50)
            assert time.perf_counter() - t0 < 1.0  # never blocked behind the writer
            if rows:
                seen.append(rows[0]["decision_id"])
            time.sleep(0.05)
    finally:
        stop.set()
        th.join()
    assert len(seen) >= 3 and seen == sorted(seen) and seen[-1] > seen[0]


def test_stop_drains_everything_already_put(tmp_path: Path) -> None:
    s = Store(f"sqlite:///{tmp_path / 'd.db'}", flush_rows=10_000, flush_s=60).start()
    session = s.new_session("pcap", "a.pcap")
    for i in range(1, 301):
        s.put("flow", flow_row(i, session))
    s.stop()
    s2 = Store(f"sqlite:///{tmp_path / 'd.db'}").start()
    try:
        assert s2.count("flow") == 300
    finally:
        s2.stop()


def test_another_schema_version_raises(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'old.db'}"
    eng = make_engine(url)
    create_schema(eng)
    with eng.begin() as c:
        c.execute(text("UPDATE schema_version SET version = 99"))
    eng.dispose()
    with pytest.raises(SchemaVersionError, match="99"):
        Store(url).start()


def test_a_foreign_database_with_flows_but_no_version_raises(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'foreign.db'}"
    eng = make_engine(url)
    m.metadata.create_all(eng)
    with eng.begin() as c:
        c.execute(insert(m.capture_session), {"kind": "x", "source": "y", "started_at": 0.0})
        c.execute(insert(m.flow), flow_row(1, 1))
    eng.dispose()
    with pytest.raises(SchemaVersionError, match="no schema_version"):
        Store(url).start()


def test_unknown_tables_are_refused(store: Store) -> None:
    with pytest.raises(KeyError):
        store.put("packets", {})


def test_the_key_hash_is_salted_per_hasher() -> None:
    a, b = KeyHasher(), KeyHasher()
    assert a("10.0.0.5:1-1.2.3.4:443/6") == a("10.0.0.5:1-1.2.3.4:443/6")
    assert a("10.0.0.5:1-1.2.3.4:443/6") != b("10.0.0.5:1-1.2.3.4:443/6")
    assert len(a("x")) == 40


def test_series_rejected_and_histograms(store: Store) -> None:
    session = store.new_session("shards", "x")
    now = time.time()
    for i in range(1, 101):
        store.put("flow", flow_row(i, session))
        name = "reject" if i % 4 == 0 else "commit"
        store.put("decision", decision_row(i, now - (i % 10), k=1 + i % 6, name=name))
    store.flush()
    series = store.series(minutes=1, bucket_s=5, now=now)
    assert sum(b["n"] for b in series) == 100
    rej = store.rejected(10)
    assert len(rej) == 10 and all(r["decision"] == "reject" for r in rej)
    assert sum(store.rejection_k_histogram().values()) == 25
    assert sum(store.decision_k_histogram().values()) == 100


@pytest.mark.slow
def test_dashboard_queries_take_under_200_ms_with_a_million_decisions(tmp_path: Path) -> None:
    """Spec 018's success criterion, measured on the laptop. The million decisions are spread
    over four hours, about 70 per second, which is the engine's measured ceiling on this CPU
    (plan phase 6, T2); the windowed queries (stats, the strip chart) read only their window."""
    url = f"sqlite:///{tmp_path / 'big.db'}"
    eng = make_engine(url)
    create_schema(eng)
    n = 1_000_000
    now = time.time()
    rng = np.random.default_rng(0)
    with eng.begin() as c:
        c.execute(insert(m.capture_session), {"kind": "x", "source": "y", "started_at": 0.0})
        blob = np.zeros((30, 4), np.int16).tobytes()
        for start in range(0, n, 50_000):
            ids = range(start + 1, start + 50_001)
            c.execute(
                insert(m.flow),
                [
                    {
                        "id": i,
                        "session_id": 1,
                        "key_hash": "0" * 40,
                        "start_ts": 0.0,
                        "ppi": blob,
                        "ppi_len": 5,
                    }
                    for i in ids
                ],
            )
            ts = now - rng.uniform(0, 4 * 3600, 50_000)
            c.execute(
                insert(m.decision),
                [decision_row(i, float(t)) for i, t in zip(ids, ts, strict=True)],
            )
    eng.dispose()
    store = Store(url).start()
    try:
        timings = {}
        for name, fn in {
            "recent_flows": lambda: store.recent_flows(n - 5000, 500),
            "stats_10s": lambda: store.stats(10.0),
            "series_10min": lambda: store.series(10, 5),
            "rejected": lambda: store.rejected(200),
            "flow_detail": lambda: store.flow_detail(n // 2),
            "k_histogram": lambda: store.decision_k_histogram(),
            "rejection_histogram": lambda: store.rejection_k_histogram(),
        }.items():
            fn()
            t0 = time.perf_counter()
            fn()
            timings[name] = (time.perf_counter() - t0) * 1000
        print(timings)
        slow = {k: v for k, v in timings.items() if v >= 200}
        assert not slow, timings
    finally:
        store.stop()
