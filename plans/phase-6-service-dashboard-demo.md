# Plan: Phase 6 — database, inference service, dashboard, demo

- **Specs:** `specs/018-storage-and-database.md`, `specs/016-inference-service-api.md`, `specs/017-dashboard.md` (build steps 16, 17, 18)
- **Status:** not-started. **Forward draft**, written 2026-11-17. Re-check against phases 4 and 5 before starting.
- **Starts when:** phase 5's exit criteria hold (the service can start after phase 4 if time is short; the drift page then waits).
- **Roadmap:** [`ROADMAP.md`](ROADMAP.md).

## Purpose

Turn the research code into something that can be shown: flows go in, early decisions come out, unknown flows are flagged, a budget knob visibly moves the mean number of packets read. It is a demonstration system for one laptop, not a product.

## Scope decisions (each simplifies a spec; recorded as corrections)

| # | Spec text | Decision | Reason |
|---|---|---|---|
| S1 | 016: live capture endpoints | Return `501 Not Implemented`; PCAP upload and shard replay only | Npcap is not installed; live capture is deferred by the phase-1 plan |
| S2 | 016: model from the MLflow registry | Load from a local run directory named in `results/models/production.json` (`best.pt`, `calibration.json`, `unknown.npz`, `label_space.json`) | No registry was ever built; a path is enough |
| S3 | 016: batched `PATStream.step` with per-flow caches | **Version 1 recomputes the prefix**: when a flow gets a packet, run the full PAT forward on its current prefix. Switch to `PATStream` only if the throughput target is missed | The model is causal and stream-equivalence is already tested, so the outputs are identical; recompute has no cache-eviction bugs. 30 tokens through 885,821 parameters is cheap |
| S4 | 018: async engine (`aiosqlite`), Alembic | Synchronous SQLAlchemy 2.x, one writer thread fed by a queue (flush every 200 rows or 100 ms), WAL mode; `metadata.create_all` plus a `schema_version` row that raises on mismatch | The database is disposable demo state on one machine. Fewer dependencies and fewer failure modes; PostgreSQL stays a URL change |
| S5 | 018: `flowstats` BLOB `float32[43]` | `float32[46]` | The project's flow-statistics vector has 46 columns (`FlowRecord.flowstats()`) |
| S6 | 017: WebSocket client thread in Streamlit | The dashboard polls `GET /flows?since=` and `GET /stats` once a second; the WebSocket endpoint stays in the service for other clients | A background socket thread inside Streamlit's rerun model is fragile |
| S7 | 017: Plotly, `src/evaluation/plots.py` | Offline figures reuse `adl_etc.evaluation.plotstyle` (matplotlib, `st.pyplot`); live charts use Streamlit's built-in charts | One plotting code path for the report and the dashboard, which is what the spec wants |
| S8 | 017: Experiments page queries MLflow | It reads `results/summaries/*.md` and each run's `report.json` | The MLflow UI already exists for browsing; the page only needs the final tables |
| S9 | 016: static API key | Bind to `127.0.0.1` only; no key | Local demo |

## Architecture

```
PCAP file ──► data/pcap_source + FlowBuilder ─┐
                                              ├─► PacketEvent queue ─► inference loop ─► decisions ─► writer thread ─► SQLite (WAL)
D1 shards ──► scripts/replay_shards.py ───────┘        │  per flow: prefix → PAT → Calibration → inference.unknown
   (stored PPI replayed as packets)                    │            → inference.policy.decide → inference.controller
                                                       └─► inference.drift (one WindowStats per 5,000 decisions)
FastAPI (127.0.0.1) ── GET /flows /stats /budget /drift /model /health ──► Streamlit dashboard (polls at 1 Hz)
```

The decision rule, unknown scorer, controller and drift monitor are the phase-4 and phase-5 `inference/` modules, imported, not re-implemented.

**Why shard replay is the main demo source.** D1 is published as flow records, not PCAPs, so there is no PCAP of in-distribution traffic. Replaying stored PPI from `test_id` and `test_drift` gives labelled, in-distribution flows, real unknown-class flows for the alert panel, and real drift for the drift page. A PCAP upload (a D4 capture or a self-captured one) is the second scene and shows the PCAP path working; its predictions may be poor because the capture point differs from the training data, and the demo script must say so.

## Tasks

### T1 — Database (spec 018)

`src/adl_etc/service/db/{models.py, store.py}`, `scripts/db_init.py`, tests `tests/service/test_db.py`.

- Tables as in spec 018 with S4 and S5 applied: `capture_session`, `flow`, `decision`, `decision_trace` (optional, off by default), `controller_state`, `drift_signal`, `model_version`, `job`, `schema_version`.
- `Store(url)`: `start()`, `stop()` (drains the queue), `put(row)`, read helpers `recent_flows(since, limit)`, `stats(window_s)`, `controller_history(n)`, `drift_history(n)`.
- The 5-tuple is stored only as a salted SHA-1; the salt lives in memory for the session and is never written.

Tests: create from empty; 50,000 decision rows written through the queue in under 10 s; a reader sees rows while the writer is active (WAL); schema-version mismatch raises.

### T2 — Service (spec 016)

`src/adl_etc/service/{app.py, engine.py, ingest.py, schemas.py}`, `configs/service.yaml`, `scripts/replay_shards.py`, tests `tests/service/test_api.py`, `tests/service/test_engine.py`.

- `engine.Engine`: owns the model bundle, per-flow state (`flow_id → prefix tensor, k, last_ts`), the controller and the monitor. `on_packet(event) -> Decision | None`. Evicts a flow on decision or after 60 s idle; caps at 50,000 concurrent flows and counts drops.
- Micro-batching: drain the queue every 5 ms or 64 events, group by flow, pad prefixes, one forward pass.
- Endpoints: `POST /ingest/pcap`, `POST /ingest/shards` (period, count, speed), `POST /score/flow`, `GET /flows`, `GET /stats`, `GET|POST /budget`, `GET /drift`, `GET|POST /model`, `GET /jobs/{id}`, `GET /health`, `GET /metrics`, `WS /stream/decisions`; live-capture endpoints return 501.
- `Decision` is the dataclass from `inference/policy.py`, serialised with Pydantic.

Tests (spec 016's list): for a fixture PCAP, the service's decisions equal offline `replay.safe_stop` on the same flows; posting a lower budget lowers `θ_commit` after replayed traffic; a model reload affects only new flows; a flow with no payload packets is stored as `undecidable`.

### T3 — Dashboard (spec 017)

`src/adl_etc/dashboard/{app.py, pages/*.py, client.py}`, tests with `streamlit.testing.v1.AppTest` against a stub client.

Build in this order; stop after page 4 if time is short:

1. **Monitor** — start a shard replay or upload a PCAP; table of the last 500 flows; counters and a strip chart of mean K and rejection rate.
2. **Budget** — set metric, target and mode; target vs realised; `θ` trajectory.
3. **Flow inspector** — one flow's packets, top-3 class probability per K, `p_safe(k)`, `u(k)`, decision marker, per-packet surprise.
4. **Unknown** — rejected flows, rejection-K histogram, AUROC(K) from the offline report.
5. **Drift** — phase-5 signals and alarms.
6. **Experiments** — the summary tables and figures.
7. **Model & system** — active run, config, latency.

### T4 — Demo script

`docs/demo.md`: a five-minute script with exact commands and what to say at each step. Scene 1: shard replay (decisions arrive early; unknown alerts; lower the budget and watch mean K fall; jump to a late drift week and watch the drift page). Scene 2 (optional): PCAP upload. Scene 3 (optional, roadmap DP-6a): the fraud-motivated scene from `docs/future-work.md` FEP-1.

A rehearsal checklist: start order (service, then dashboard), ports, what to do if the port is still held (on Windows, killing `uvicorn` can leave workers holding it).

### T5 — Paperwork

`pyproject.toml`: fill the `service` extra (fastapi, uvicorn, sqlalchemy, pydantic, streamlit, python-multipart, httpx). Corrections S1–S9 written into specs 016–018; index files and `CLAUDE.md` updated.

## Success criteria

| Spec | Criterion |
|---|---|
| 018 | Dashboard queries under 200 ms with 1M decision rows; database creates cleanly from empty |
| 016 | Service decisions equal the offline replay on a fixture; 5,000 packet events/s on the laptop with p95 batch latency under 5 ms, or the measured value reported; budget knob changes mean K |
| 017 | The five-minute script in `docs/demo.md` runs end to end on the laptop |

## Risks

| Risk | Mitigation |
|---|---|
| Prefix recompute is too slow for the target | Measure in T2; switch to `PATStream` with a per-flow cache; phase 5 already timed `PATStream.step` |
| Demo traffic looks wrong on a self-captured PCAP | Shard replay is scene 1; the PCAP scene is optional and its caveat is in the script |
| Windows leaves ports and SQLite locks behind | Rehearsal checklist; the store closes on shutdown; one database file per demo run |
| Scope creep in the dashboard | Page order above; pages 5–7 are optional |
| Torch in the service process slows start-up | Load the model once at start; `/health` reports "loading" until ready |

## Exit criteria

1. `pytest` (including `tests/service/`), `ruff`, `mypy` clean.
2. The demo script has been run start to finish twice on the laptop.
3. Corrections landed; index files updated.

## Progress log

- **2026-11-17.** Forward draft written. Not started.
