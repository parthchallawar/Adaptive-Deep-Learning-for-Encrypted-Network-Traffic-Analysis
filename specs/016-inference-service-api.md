# Spec 016: Inference Service (FastAPI)

- **Status:** draft
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

## Open questions

- WebSocket vs. server-sent events for the dashboard (default: WebSocket).
- Whether `/score/flow` should also accept raw packet lists (default: yes, tensorised server-side).
