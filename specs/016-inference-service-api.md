# Spec 016: Inference Service (FastAPI)

- **Status:** built 2026-11-18 (plan phase 6, T2), with the corrections below. Implemented: every endpoint in the API table except live capture (501), plus `/ingest/shards`, `/ingest/periods`, `/stats`, `/stats/series`, `/unknown`, `/experiments`, `/system`, `/jobs`; streaming per-flow state, micro-batching, controller and drift monitor online; Prometheus `/metrics`. Not implemented: live capture, the MLflow registry, ONNX, batched `PATStream`. Throughput target missed (measured ~300 packet events/s, not 5,000).
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 17 of 18
- **Depends on:** 002, 003, 006, 009, 010, 011, 012, 014, 018. **Used by:** 017, 019.

## Problem

The research artefacts (model, policy, controller, monitors) must run as one online system that accepts PCAP uploads or live capture, emits per-flow decisions as packets arrive, exposes the budget and drift controls, and persists everything for the dashboard. It must be CPU-only, single-process-friendly, and honest about latency.

## Goals

- REST + WebSocket API: ingest (pcap upload, live start/stop, single-flow scoring), decisions stream, budget control, drift status, model management.
- Streaming inference with per-flow state (KV caches) and batched stepping across concurrent flows.
- Same tensorizer and policy code as offline evaluation (no re-implementation).
- Metrics endpoint (Prometheus text format) for latency and throughput.

## Non-goals

- Authentication beyond a static API key; horizontal scaling; GPU serving.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/ingest/pcap` | upload a pcap; returns `job_id`; processed in a background task through spec-002 backend |
| POST | `/ingest/live/start` `{iface}` / `/ingest/live/stop` | start/stop live capture (Scapy/Npcap or ipfixprobe) |
| POST | `/score/flow` | body: PPI array (any K) + optional flowstats; returns per-K decisions (batch, offline use) |
| WS | `/stream/decisions` | pushes `{flow_id, k, decision, class, p_safe, u, ts}` events as they happen |
| GET/POST | `/budget` | read/set `budget_metric`, `B*`, mode {auto, manual, quality-first}; returns controller state |
| GET | `/drift` | current signals, CUSUM states, level |
| GET/POST | `/model` | active model version (MLflow registry stage), reload |
| GET | `/flows?since=&limit=` | recent flows with decisions from the DB |
| GET | `/metrics` | Prometheus text |
| GET | `/health` | liveness |

Decision event schema is the same dataclass used by the offline replay (spec 009), serialised with Pydantic.

## Design

- **Process model:** FastAPI + uvicorn run directly (`uvicorn adl_etc.service.app:app`), one worker, no container (spec 019 deferred); a capture thread produces `PacketEvent`s into an asyncio queue; an inference loop drains the queue every 5 ms or 64 events, groups events by flow, runs `PATStream.step` batched (pad to max k), applies the policy and the controller, persists decisions, and broadcasts to WebSocket clients.
- **Flow state:** dict `flow_id → {kv_cache, k, last_ts, scores}`; evicted on decision or after 60 s idle; capped at 50k concurrent flows (LRU) with a dropped-flows counter.
- **Model loading:** from a local checkpoint path by default (`results/models/production.json`), or from the MLflow registry stage when a tracking server happens to be running (spec 014); ONNX Runtime full-forward for `/score/flow`, TorchScript/eager `PATStream` for streaming (ONNX with explicit cache I/O if the export works; measured in spec 013).
- **Controller and monitor:** singletons updated on every decision; state persisted to the DB every window.
- **Persistence:** spec 018 tables via SQLAlchemy; writes batched.
- **Config:** `configs/service.yaml` (model stage, thresholds initial values, capture backend, DB URL).

## Inputs and outputs

- Inputs: pcap files, interface packets, API calls.
- Outputs: decision events, DB rows, metrics, logs.

## Edge cases

- Upload of a very large pcap (> 2 GB): streamed to disk, processed incrementally; progress in `GET /jobs/{id}`.
- Non-payload packets: ignored by the tensorizer; flows with zero payload packets are stored as `undecidable`.
- Model reload while streaming: new flows use the new model; in-flight flows finish on the old one (two model instances at most).
- Capture permission errors (Npcap not installed / no root): clear 4xx with instructions.
- Back-pressure: if the queue exceeds 100k events, capture is paused and a warning is logged (never silently drop decisions).

## Performance considerations

- Target: 5k packet events/s on a laptop CPU with p95 step latency < 5 ms under batching; measured by spec 013 and exposed at `/metrics`.

## Testing

- API tests with `TestClient` and a fixture pcap: decisions appear, counts match the offline pipeline exactly (same tensorizer).
- Stream/offline equivalence: decisions from `/stream` equal replay of the same pcap through spec 009 offline.
- Controller endpoint test: setting `B*` changes θ in the expected direction after replayed traffic.
- Load test (locust or a simple asyncio script) for the target throughput.

## Interactions

- Consumes 002 backends, 003 tensorizer, 006 stream model, 009/010/011/012 logic, 014 registry; writes 018; serves 017.

## Success criteria

- Live demo end-to-end on the owner's laptop; decisions for browsing flows within the first 10 payload packets; budget knob visibly changes mean K; unknown alerts for a flow type not in training.

## Pre-build notes (added 2026-11-17)

Build plan (forward draft): [`plans/phase-6-service-dashboard-demo.md`](../plans/phase-6-service-dashboard-demo.md), task T2. These simplify the design above for a one-laptop demonstration; where they disagree with it, they win.

- **S1. No live capture.** `/ingest/live/start` and `/ingest/live/stop` return `501 Not Implemented`. Npcap is not installed and live capture has been deferred since phase 1.
- **S2. No MLflow registry.** The model bundle is a local run directory named in `results/models/production.json`: `best.pt`, `calibration.json`, `unknown.npz`, `label_space.json`. `/model` reads and reloads that.
- **S3. Version 1 recomputes the prefix.** When a flow receives a packet, the full PAT forward runs on its current prefix. The model is causal and stream-equivalence is tested, so the outputs equal `PATStream.step`'s, with no cache-eviction logic. Move to batched `PATStream` only if the throughput target is missed.
- **New endpoint `POST /ingest/shards` `{period, count, speed}`.** D1 has no PCAPs, so replaying stored PPI from `test_id` / `test_drift` is the only way to show in-distribution traffic, real unknown-class flows and real drift. This is the main demo source; PCAP upload is the second.
- **New endpoint `GET /stats?window_s=`** returning the aggregates the dashboard polls (flows/s, mean K, commit and reject rates).
- **S9. Bind to `127.0.0.1`; no API key.**
- **Shared code:** the rule, unknown scorer, controller and monitor are imported from `src/adl_etc/inference/` (torch-free, built in phases 4 and 5). The `Decision` dataclass is the one in `inference/policy.py`.
- **Dependencies:** the `service` extra gains fastapi, uvicorn, sqlalchemy, pydantic, streamlit, python-multipart and httpx.

## Open questions

- WebSocket vs. server-sent events for the dashboard (default: WebSocket).
- Whether `/score/flow` should also accept raw packet lists (default: yes, tensorised server-side).

## Corrections after building (phase 6, 2026-11-18)

Recorded in the plan's scope table; these describe what the code does.

| # | Spec text | What was built | Why |
|---|---|---|---|
| S1 | live capture endpoints | `501 Not Implemented` | Npcap not installed; deferred since phase 1 |
| S2 | model from the MLflow registry | `results/models/production.json`, written by `scripts/make_production_bundle.py`, names the run directory, unknown model, standardizer and the phase-4/5 operating point; `POST /model` reloads it | no registry was built |
| S3 | batched `PATStream.step` with KV caches | the full PAT forward on each flow's prefix, through a new `PAT.forward(n_packets=)` that runs only the first n positions (tested equal to the full pass), with flows grouped by prefix length; score columns computed only at the positions just reached (`GaussianBank.mahal_at`, `UnknownModel.score_at`) | outputs equal the offline score pass (0 of 400 real flows differ, `tests/service/test_bundle.py`); no cache-eviction logic |
| S9 | static API key | bound to `127.0.0.1`, no key; the entry point refuses another host | local demo |
| — | streaming forced decision | the rule runs with `last_k = k_cap` while packets arrive; the forced branch fires on an end event, the 60 s idle timeout or `k_cap` | a stream does not know a flow's length; gives the offline replay's `(k, decision, class)` exactly (`tests/service/test_engine.py`) |
| — | back-pressure | the hard ceiling `engine.queue_limit` (100k) is kept, and a replay additionally paces itself to the engine (queue held near 256 events) | without it a replay faster than the engine queues packets of flows the backlog is about to decide, and the lag grows without bound (seen on the first live run) |
| — | 5k packet events/s, p95 batch < 5 ms | measured ~300 packet events/s (~23 flows/s at mean K ≈ 6), batch p95 ~380 ms, on this laptop's CPU with 4 torch threads | prefix recompute is compute-bound (a 64-flow batch is ~1-3.5 GFLOP depending on prefix length); batched `PATStream` stepping is the next step (`docs/future-work.md`) |
