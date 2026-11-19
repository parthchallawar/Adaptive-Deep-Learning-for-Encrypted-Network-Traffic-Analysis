"""Spec 018's tables, as SQLAlchemy 2.x Core (plan phase 6, T1).

Engine-neutral types only (no SQLite-specific column types), so PostgreSQL stays a URL change.
Correction S4: no Alembic. :func:`create_schema` runs ``metadata.create_all`` and records
:data:`SCHEMA_VERSION` in ``schema_version``; opening a database written by another version
raises :class:`SchemaVersionError` (the database is disposable demo state: delete it).
Correction S5: ``flow.flowstats`` holds ``float32[46]`` (``FlowRecord.flowstats()``), not 43.

Privacy (spec 018): no IP address or port is ever stored. ``flow.key_hash`` is a salted SHA-1 of
the 5-tuple whose salt lives only in the service's memory for the session.

Three columns go beyond the spec, all for the demo: ``flow.true_class`` and ``flow.true_known``
(the dataset label and whether it is a known class, available only for shard replay) and
``decision.correct`` (open-world correctness: a known flow committed to its own class, or an
unknown-class flow rejected).
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
)

SCHEMA_VERSION = 1

metadata = MetaData()

schema_version = Table(
    "schema_version",
    metadata,
    Column("version", Integer, primary_key=True),
)

capture_session = Table(
    "capture_session",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("kind", String(16), nullable=False),  # "shards" | "pcap" | "score"
    Column("source", Text, nullable=False),  # pcap file name, or "WEEK-2022-31", ...
    Column("started_at", Float, nullable=False),
    Column("ended_at", Float),
    Column("backend", String(32)),
    Column("config_hash", String(64)),
)

flow = Table(
    "flow",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),  # assigned by the engine
    Column("session_id", Integer, ForeignKey("capture_session.id"), nullable=False),
    Column("key_hash", String(40), nullable=False),
    Column("start_ts", Float, nullable=False),
    Column("ppi", LargeBinary, nullable=False),  # int16[30, 4]
    Column("ppi_len", Integer, nullable=False),
    Column("flowstats", LargeBinary),  # float32[46], or NULL when the source has none
    Column("end_reason", String(16)),
    Column("true_class", String(64)),  # shard replay only
    Column("true_known", Boolean),  # shard replay only: is the true class a known class?
)

decision = Table(
    "decision",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("flow_id", Integer, ForeignKey("flow.id"), nullable=False),
    Column("k", Integer, nullable=False),  # 0 for an undecidable flow
    Column("decision", String(16), nullable=False),  # policy name, or "undecidable"
    Column("forced", Boolean, nullable=False),
    Column("class_id", Integer, nullable=False),  # model index, -1 = rejected / none
    Column("class_name", String(64)),
    Column("p_safe", Float),
    Column("u", Float),
    Column("u_threshold", Float),  # theta_reject(k) in force
    Column("energy", Float),
    Column("mahal", Float),
    Column("surprise", Float),
    Column("model_version", String(16), nullable=False),
    Column("theta_commit", Float),
    Column("decided_at", Float, nullable=False),
    Column("time_to_decision_ms", Float),
    Column("correct", Boolean),  # shard replay only: committed class == true class
)

decision_trace = Table(
    "decision_trace",
    metadata,
    Column("flow_id", Integer, ForeignKey("flow.id"), primary_key=True),
    Column("payload", Text, nullable=False),  # JSON: per-K p_safe, u, pred up to the decision
)

controller_state = Table(
    "controller_state",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("window_id", Integer, nullable=False),
    Column("ts", Float, nullable=False),
    Column("budget_metric", String(16), nullable=False),
    Column("target", Float, nullable=False),
    Column("observed", Float),
    Column("error", Float),
    Column("integral", Float),
    Column("theta_commit", Float, nullable=False),
    Column("theta_reject_f", Float),  # the unknown score's false-positive target f
    Column("k_cap", Integer),
    Column("mode", String(16), nullable=False),
    Column("reason", String(32)),
)

drift_signal = Table(
    "drift_signal",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("window_id", Integer, nullable=False),
    Column("ts", Float, nullable=False),
    Column("signal_name", String(32), nullable=False),
    Column("value", Float),
    Column("z", Float),
    Column("cusum_pos", Float),
    Column("cusum_neg", Float),
    Column("alarm", Boolean, nullable=False),
    Column("level", String(16), nullable=False),
)

model_version = Table(
    "model_version",
    metadata,
    Column("version", String(16), primary_key=True),
    Column("run_dir", Text, nullable=False),
    Column("checkpoint_hash", String(64), nullable=False),
    Column("loaded_at", Float, nullable=False),
    Column("config_hash", String(64)),
)

job = Table(
    "job",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("type", String(16), nullable=False),
    Column("status", String(16), nullable=False),  # queued | running | done | failed
    Column("progress", Integer, nullable=False, default=0),
    Column("total", Integer),
    Column("error", Text),
    Column("created_at", Float, nullable=False),
    Column("finished_at", Float),
    Column("session_id", Integer),
)

# Covering index for the dashboard's time-window aggregates (stats and the strip chart): they read
# only these columns, so SQLite answers them from the index without touching the table.
Index(
    "ix_decision_window",
    decision.c.decided_at,
    decision.c.decision,
    decision.c.k,
    decision.c.forced,
    decision.c.theta_commit,
    decision.c.correct,
    decision.c.time_to_decision_ms,
)
Index("ix_decision_flow_id", decision.c.flow_id)
Index("ix_decision_decision_decided_at", decision.c.decision, decision.c.decided_at)
Index("ix_drift_signal_name_ts", drift_signal.c.signal_name, drift_signal.c.ts)
Index("ix_flow_session_start", flow.c.session_id, flow.c.start_ts)
Index("ix_controller_window", controller_state.c.window_id)

#: Insert order inside one flush, so a row never precedes the row it references.
TABLE_ORDER = (
    "capture_session",
    "model_version",
    "job",
    "flow",
    "decision",
    "decision_trace",
    "controller_state",
    "drift_signal",
)
