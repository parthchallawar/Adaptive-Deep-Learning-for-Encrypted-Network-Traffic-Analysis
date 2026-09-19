# Spec 018: Storage and Database

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 16 of 18 (before 016, which writes to it)
- **Depends on:** none. **Used by:** 016, 017, 012.

## Problem

The online system must persist flows, decisions, controller and drift state so that the dashboard can query history and so that demos are reproducible. Offline artefacts (shards, checkpoints, MLflow) have their own storage (specs 003, 014); this spec covers the service database only. The synopsis lists PostgreSQL/SQLite.

**Decision (2026-09-17):** SQLite is the only engine in the current plan, since containerised deployment is deferred (spec 019). The schema and the data-access layer stay engine-neutral (SQLAlchemy + Alembic, no SQLite-only types) so that PostgreSQL is a configuration change if deployment is reactivated. PostgreSQL-specific work is not implemented or tested for now.

## Goals

- SQLAlchemy 2.x models with Alembic migrations; schema written to be engine-neutral (SQLite now, PostgreSQL later without a rewrite).
- Write path fast enough for 5k decisions/s in batches; read path indexed for the dashboard's queries.
- Retention policy and export to Parquet.

## Non-goals

- Storing packet payloads (never captured) or raw PPI of every packet beyond the 30-packet PPI (already small).

## Schema

| Table | Columns (key ones) | Notes |
|---|---|---|
| `capture_session` | id, source (pcap path / iface), started_at, ended_at, backend, config_hash | one per upload or live run |
| `flow` | id, session_id, key_hash (sha1 of 5-tuple, not the tuple itself), start_ts, ppi (BLOB int16[30,4]), ppi_len, flowstats (BLOB float32[43]), end_reason | no IPs stored in the DB; only a salted hash for de-duplication |
| `decision` | id, flow_id, k, decision (COMMIT/REJECT/FORCED), class_id, class_name, p_safe, u, energy, mahal, surprise, model_version, theta_commit, theta_reject, decided_at, time_to_decision_ms | one row per flow (the final decision); optional `decision_trace` table with per-K scores when tracing is enabled |
| `controller_state` | window_id, ts, budget_metric, target, observed, error, integral, theta_commit, theta_reject, mode | one row per window |
| `drift_signal` | window_id, ts, signal_name, value, cusum, alarm | long format |
| `model_version` | version, mlflow_run_id, stage, loaded_at, config_hash | |
| `job` | id, type, status, progress, error | pcap processing jobs |

Indexes: `decision(decided_at)`, `decision(flow_id)`, `decision(decision, decided_at)`, `drift_signal(signal_name, ts)`, `flow(session_id, start_ts)`.

## Design

- Async engine (`sqlite+aiosqlite`; `postgresql+asyncpg` reachable by config if deployment is reactivated) with batched inserts (executemany every 200 rows or 100 ms).
- Alembic migrations in `src/service/db/migrations/`; `scripts/db_init.py`.
- Retention: `scripts/db_prune.py --older-than 7d`; export `scripts/db_export.py --parquet` for analysis in notebooks.
- Privacy: 5-tuple stored only as a salted hash; salt per capture session, discarded at session end, so flows cannot be re-identified later.

## Inputs and outputs

- Inputs: rows from the service.
- Outputs: query results for the dashboard/API; Parquet exports.

## Edge cases

- SQLite write contention with the dashboard reading: WAL mode; readers never block the writer.
- Database file locked or corrupt: the service creates a new one and logs loudly; decisions are never dropped silently.
- Schema change with existing data: Alembic migration required; CI runs migrations on a fixture DB.

## Performance considerations

- 5k decisions/s x 200 bytes = 1 MB/s; SQLite in WAL mode sustains this on an SSD.

## Testing

- Model/migration tests on SQLite in memory; batch-insert throughput test.

## Interactions

- Written by 016 and 011/012 (via the service); read by 017.

## Success criteria

- Dashboard queries return in < 200 ms with 1M decision rows on SQLite; migrations apply cleanly from an empty database.

## Open questions

- Whether to keep `decision_trace` on by default in demos (default: on for the inspector page, off under load).
