"""The service's database access (spec 018; plan phase 6, T1, correction S4).

One :class:`Store` per service process. Writes go through :meth:`Store.put` into a queue that a
single writer thread drains, flushing every ``flush_rows`` rows or ``flush_s`` seconds in one
transaction (``executemany`` per table). Reads are short-lived connections; on SQLite the
database runs in WAL mode so a reader never blocks the writer and vice versa.

:meth:`Store.stop` drains the queue before returning, so nothing put before ``stop`` is lost.
A write that fails is logged loudly and the batch is kept for the next attempt rather than
dropped (spec 018: decisions are never dropped silently); after ``max_retries`` failed attempts
the batch is counted in ``failed_rows`` and the error re-raised by :meth:`flush`.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy import Engine, Integer, case, create_engine, event, func, insert, select, update
from sqlalchemy.pool import StaticPool

from adl_etc.service.db import models as m

log = logging.getLogger(__name__)

_STOP = object()
UNDECIDABLE = "undecidable"
REJECTED = ("reject", "forced_reject")


class SchemaVersionError(RuntimeError):
    pass


def make_engine(url: str) -> Engine:
    """An engine with SQLite's WAL mode and a busy timeout switched on for every connection."""
    kw: dict[str, Any] = {"future": True}
    if url in ("sqlite://", "sqlite:///:memory:"):
        # one shared connection, so every reader sees the same in-memory database
        kw.update(connect_args={"check_same_thread": False}, poolclass=StaticPool)
    elif url.startswith("sqlite"):
        kw.update(connect_args={"check_same_thread": False, "timeout": 30})
    engine = create_engine(url, **kw)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn: Any, _record: Any) -> None:  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

    return engine


def create_schema(engine: Engine) -> None:
    """Create every table if missing and check (or record) the schema version."""
    m.metadata.create_all(engine)
    with engine.begin() as conn:
        rows = conn.execute(select(m.schema_version.c.version)).scalars().all()
        if not rows:
            n_flows = conn.execute(select(func.count()).select_from(m.flow)).scalar_one()
            if n_flows:
                raise SchemaVersionError(
                    "the database holds flows but no schema_version row; it was not made by "
                    "this service. Delete it or point the service at a new file."
                )
            conn.execute(insert(m.schema_version), {"version": m.SCHEMA_VERSION})
        elif rows != [m.SCHEMA_VERSION]:
            raise SchemaVersionError(
                f"database schema version {rows} but this code writes {m.SCHEMA_VERSION}; "
                "the database is disposable demo state: delete it and start again"
            )


class Store:
    def __init__(
        self,
        url: str,
        *,
        flush_rows: int = 200,
        flush_s: float = 0.1,
        max_retries: int = 5,
    ) -> None:
        self.url = url
        self.flush_rows, self.flush_s, self.max_retries = (
            int(flush_rows),
            float(flush_s),
            max_retries,
        )
        self.engine: Engine | None = None
        self._q: queue.Queue[Any] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.rows_written = 0
        self.failed_rows = 0
        self.flushes = 0

    # -- lifecycle ------------------------------------------------------------------

    def start(self) -> Store:
        if self.url.startswith("sqlite:///") and not self.url.endswith(":memory:"):
            path = self.url.removeprefix("sqlite:///")
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
        self.engine = make_engine(self.url)
        create_schema(self.engine)
        self._thread = threading.Thread(target=self._run, name="db-writer", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Drain everything queued so far, then stop the writer and dispose the engine."""
        if self._thread is not None:
            self._q.put(_STOP)
            self._thread.join()
            self._thread = None
        if self.engine is not None:
            self.engine.dispose()

    def flush(self, timeout: float = 30.0) -> None:
        """Block until every row put before this call is written (tests, shutdown)."""
        done = threading.Event()
        self._q.put(done)
        if not done.wait(timeout):
            raise TimeoutError("the database writer did not flush in time")
        if self._error is not None:
            err, self._error = self._error, None
            raise err

    @property
    def pending(self) -> int:
        return self._q.qsize()

    # -- writes ---------------------------------------------------------------------

    def put(self, table: str, row: Mapping[str, Any]) -> None:
        if table not in m.TABLE_ORDER:
            raise KeyError(f"unknown table {table!r}")
        self._q.put((table, dict(row)))

    def put_many(self, table: str, rows: Iterable[Mapping[str, Any]]) -> None:
        for r in rows:
            self.put(table, r)

    def execute_now(self, stmt: Any) -> Any:
        """A synchronous write outside the queue (session and job bookkeeping only)."""
        assert self.engine is not None, "call start() first"
        with self.engine.begin() as conn:
            return conn.execute(stmt)

    def new_session(
        self, kind: str, source: str, *, backend: str = "", config_hash: str = ""
    ) -> int:
        res = self.execute_now(
            insert(m.capture_session).values(
                kind=kind,
                source=source,
                started_at=time.time(),
                backend=backend,
                config_hash=config_hash,
            )
        )
        return int(res.inserted_primary_key[0])

    def end_session(self, session_id: int) -> None:
        self.execute_now(
            update(m.capture_session)
            .where(m.capture_session.c.id == session_id)
            .values(ended_at=time.time())
        )

    def _run(self) -> None:
        pending: list[tuple[str, dict[str, Any]]] = []
        waiters: list[threading.Event] = []
        last = time.monotonic()
        attempts = 0
        stopping = False
        while True:
            timeout = max(0.0, self.flush_s - (time.monotonic() - last))
            try:
                item = self._q.get(timeout=timeout if pending else None)
            except queue.Empty:
                item = None
            if item is _STOP:
                stopping = True
            elif isinstance(item, threading.Event):
                waiters.append(item)
            elif item is not None:
                pending.append(item)
                # take whatever else is already queued without waiting
                while len(pending) < self.flush_rows:
                    try:
                        nxt = self._q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is _STOP:
                        stopping = True
                        break
                    if isinstance(nxt, threading.Event):
                        waiters.append(nxt)
                        break
                    pending.append(nxt)
            due = (
                len(pending) >= self.flush_rows
                or (pending and time.monotonic() - last >= self.flush_s)
                or waiters
                or stopping
            )
            if due and pending:
                try:
                    self._write(pending)
                    self.rows_written += len(pending)
                    self.flushes += 1
                    pending = []
                    attempts = 0
                except Exception as exc:  # noqa: BLE001 - logged loudly, retried, then surfaced
                    attempts += 1
                    log.error("database write failed (attempt %d): %s", attempts, exc)
                    if attempts >= self.max_retries or stopping:
                        self.failed_rows += len(pending)
                        self._error = exc
                        pending = []
                        attempts = 0
                    else:
                        time.sleep(0.05 * attempts)
                        continue
            if due:
                last = time.monotonic()
            if not pending:
                for w in waiters:
                    w.set()
                waiters = []
            if stopping and not pending:
                return

    def _write(self, rows: list[tuple[str, dict[str, Any]]]) -> None:
        assert self.engine is not None
        by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for table, row in rows:
            by_table[table].append(row)
        with self.engine.begin() as conn:
            for name in m.TABLE_ORDER:
                if by_table.get(name):
                    conn.execute(insert(m.metadata.tables[name]), by_table[name])

    # -- reads ----------------------------------------------------------------------

    def _read(self, stmt: Any) -> list[dict[str, Any]]:
        assert self.engine is not None, "call start() first"
        with self.engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(stmt)]

    def next_flow_id(self) -> int:
        rows = self._read(select(func.max(m.flow.c.id).label("mx")))
        return int(rows[0]["mx"] or 0) + 1

    @staticmethod
    def _flow_rows() -> Any:
        """flow joined to decision, newest decision first."""
        f, d = m.flow, m.decision
        return (
            select(
                d.c.id.label("decision_id"),
                f.c.id.label("flow_id"),
                f.c.session_id,
                f.c.key_hash,
                f.c.ppi_len,
                f.c.true_class,
                d.c.k,
                d.c.decision,
                d.c.forced,
                d.c.class_id,
                d.c.class_name,
                d.c.p_safe,
                d.c.u,
                d.c.u_threshold,
                d.c.theta_commit,
                d.c.decided_at,
                d.c.time_to_decision_ms,
                d.c.correct,
                d.c.model_version,
            )
            .join(f, f.c.id == d.c.flow_id)
            .order_by(d.c.id.desc())
        )

    def flow_detail(self, flow_id: int) -> dict[str, Any] | None:
        rows = self._read(select(m.flow).where(m.flow.c.id == flow_id))
        if not rows:
            return None
        out = rows[0]
        dec = self._read(select(m.decision).where(m.decision.c.flow_id == flow_id))
        out["decision"] = dec[0] if dec else None
        trace = self._read(select(m.decision_trace).where(m.decision_trace.c.flow_id == flow_id))
        out["trace"] = trace[0]["payload"] if trace else None
        return out

    def stats(self, window_s: float = 10.0, now: float | None = None) -> dict[str, Any]:
        """Aggregates over decisions made in the last ``window_s`` seconds. Undecidable flows
        are counted separately and kept out of every rate (``k`` is 0 for them). Only the
        covering index on ``decided_at`` is read; whole-table totals come from the engine's
        counters, since a count over a million rows alone costs more than the 200 ms budget."""
        d = m.decision
        now = time.time() if now is None else now
        decided = d.c.decision != UNDECIDABLE

        def only(expr: Any) -> Any:
            return case((decided, expr), else_=None)

        stmt = select(
            func.sum(decided.cast(Integer)).label("n"),
            func.sum((~decided).cast(Integer)).label("undecidable"),
            func.avg(only(d.c.k)).label("mean_k"),
            func.avg(only((d.c.decision == "commit").cast(Integer))).label("commit_rate"),
            func.avg(only(d.c.decision.in_(REJECTED).cast(Integer))).label("reject_rate"),
            func.avg(only(d.c.forced.cast(Integer))).label("forced_rate"),
            func.avg(d.c.correct.cast(Integer)).label("accuracy"),
            func.avg(only(d.c.time_to_decision_ms)).label("mean_time_to_decision_ms"),
        ).where(d.c.decided_at >= now - window_s)
        row = self._read(stmt)[0]
        n = int(row["n"] or 0)
        return {
            "window_s": window_s,
            "n": n,
            "undecidable": int(row["undecidable"] or 0),
            "flows_per_s": n / window_s if window_s > 0 else 0.0,
            "mean_k": _f(row["mean_k"]),
            "commit_rate": _f(row["commit_rate"]),
            "reject_rate": _f(row["reject_rate"]),
            "forced_rate": _f(row["forced_rate"]),
            "accuracy": _f(row["accuracy"]),
            "mean_time_to_decision_ms": _f(row["mean_time_to_decision_ms"]),
        }

    def series(
        self, minutes: float = 10.0, bucket_s: float = 5.0, now: float | None = None
    ) -> list[dict[str, Any]]:
        """Mean K, reject rate, threshold and accuracy per ``bucket_s`` bucket over the last
        ``minutes`` (undecidable flows left out)."""
        d = m.decision
        now = time.time() if now is None else now
        bucket = func.floor(d.c.decided_at / bucket_s).label("b")
        decided = d.c.decision != UNDECIDABLE

        def only(expr: Any) -> Any:
            return case((decided, expr), else_=None)

        stmt = (
            select(
                bucket,
                func.sum(decided.cast(Integer)).label("n"),
                func.avg(only(d.c.k)).label("mean_k"),
                func.avg(only(d.c.decision.in_(REJECTED).cast(Integer))).label("reject_rate"),
                func.avg(only(d.c.theta_commit)).label("theta_commit"),
                func.avg(d.c.correct.cast(Integer)).label("accuracy"),
            )
            .where(d.c.decided_at >= now - minutes * 60)
            .group_by(bucket)
            .order_by(bucket)
        )
        return [
            {
                "t": float(r["b"]) * bucket_s,
                "n": int(r["n"] or 0),
                "mean_k": _f(r["mean_k"]),
                "reject_rate": _f(r["reject_rate"]),
                "theta_commit": _f(r["theta_commit"]),
                "accuracy": _f(r["accuracy"]),
            }
            for r in self._read(stmt)
        ]

    def recent_flows(self, since: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        """Decided flows with ``decision.id > since``, newest first, at most ``limit``."""
        return self._read(self._flow_rows().where(m.decision.c.id > since).limit(limit))

    def rejected(self, limit: int = 200) -> list[dict[str, Any]]:
        """The most recent flows rejected as unknown, newest first."""
        stmt = self._flow_rows().where(m.decision.c.decision.in_(("reject", "forced_reject")))
        return self._read(stmt.limit(limit))

    def rejection_k_histogram(self) -> dict[int, int]:
        d = m.decision
        stmt = (
            select(d.c.k, func.count().label("n"))
            .where(d.c.decision.in_(REJECTED))
            .group_by(d.c.k)
        )
        return {int(r["k"]): int(r["n"]) for r in self._read(stmt)}

    def decision_k_histogram(self, since: float | None = None) -> dict[int, int]:
        """Decision K counts since ``since`` (default: the last ten minutes)."""
        d = m.decision
        since = time.time() - 600 if since is None else since
        stmt = (
            select(d.c.k, func.count().label("n"))
            .where(d.c.decided_at >= since)
            .group_by(d.c.k)
        )
        return {int(r["k"]): int(r["n"]) for r in self._read(stmt) if int(r["k"]) > 0}

    def controller_history(self, n: int = 200) -> list[dict[str, Any]]:
        c = m.controller_state
        rows = self._read(select(c).order_by(c.c.id.desc()).limit(n))
        return rows[::-1]

    def drift_history(self, n_windows: int = 100) -> list[dict[str, Any]]:
        """The last ``n_windows`` monitor windows, one dict per window with every signal."""
        s = m.drift_signal
        rows = self._read(
            select(s).order_by(s.c.window_id.desc(), s.c.signal_name).limit(n_windows * 8)
        )
        windows: dict[int, dict[str, Any]] = {}
        for r in rows:
            w = windows.setdefault(
                r["window_id"],
                {"window": r["window_id"], "ts": r["ts"], "level": r["level"], "signals": {}},
            )
            w["signals"][r["signal_name"]] = {
                "value": r["value"],
                "z": r["z"],
                "cusum_pos": r["cusum_pos"],
                "cusum_neg": r["cusum_neg"],
                "alarm": bool(r["alarm"]),
            }
        return [windows[k] for k in sorted(windows)][-n_windows:]

    def count(self, table: str) -> int:
        t = m.metadata.tables[table]
        return int(self._read(select(func.count().label("n")).select_from(t))[0]["n"])


def _f(v: Any) -> float | None:
    return None if v is None else float(v)
