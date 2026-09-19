# Spec 017: Interactive Dashboard (Streamlit)

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 18 of 18
- **Depends on:** 004, 016, 018. **Used by:** demo and report figures.

## Problem

The synopsis promises results "presented through an interactive dashboard". The dashboard must make the research visible: decisions arriving early, the accuracy-earliness trade-off, unknown alerts, the budget controller in action, drift signals, and the offline experiment results, all from one place, without re-implementing any logic.

## Goals

- Pages: Live/PCAP monitor, Flow inspector, Budget control, Unknown & anomalies, Drift, Experiments (offline results), Model & system.
- All data via the service API (spec 016) and the DB (spec 018) for live pages, and via MLflow (spec 014) for experiment pages.
- Plotly charts, consistent with the report figure style.

## Non-goals

- Editing configs or launching training from the UI; multi-user auth.

## Pages

1. **Monitor**: start/stop live capture or upload a pcap; live table of flows (flow id, class or unknown, decision K, p_safe, u, time-to-decision); running counters (flows/s, mean K, commit/reject rates); a strip chart of mean K and rejection rate over the last 10 minutes.
2. **Flow inspector**: pick a flow; timeline of its packets (size bars signed by direction, IPT gaps), overlaid per-K class probability of the top-3 classes, p_safe(k), u(k), the decision marker, and per-packet NPP surprise (spec 010 explanation).
3. **Budget control**: choose metric (mean K / compute / p95 K), set target, mode {auto, manual, quality-first}; plots of target vs realised over time and θ trajectory; accuracy-vs-budget curve from the offline replay for context.
4. **Unknown & anomalies**: rejected flows table, rejection K histogram, cluster view (optional, spec 012 stretch), AUROC(K) curve from offline evaluation.
5. **Drift**: the spec-012 signals with CUSUM states and alarms; offline drift curves (weekly macro-F1) for reference.
6. **Experiments**: main table, accuracy-vs-K curves per model, Pareto fronts, efficiency scatter, calibration diagrams, all read from MLflow with run selection.
7. **Model & system**: active model version, config summary, latency/throughput from `/metrics`, hardware.

## Design

- `src/dashboard/app.py` with one module per page; `st.cache_data` for MLflow queries (TTL 60 s); WebSocket client in a background thread feeding a `st.session_state` ring buffer for live events; `st.rerun` throttled to 1 Hz.
- Charts built by shared functions in `src/evaluation/plots.py` so that report figures and dashboard figures are the same code.

## Inputs and outputs

- Inputs: API/WS (spec 016), DB (spec 018), MLflow store (spec 014).
- Outputs: interactive views; "export figure" buttons write PNG/HTML to `results/summaries/figures/`.

## Edge cases

- Service down: pages show a clear status banner and fall back to DB/MLflow read-only content.
- Thousands of flows/s: live table shows the last 500; aggregates come from the service, not from the UI.
- No live capture permission on Windows: the page explains Npcap requirements and offers pcap replay.

## Performance considerations

- Streamlit reruns are cheap if heavy data stays in cache; live pages poll aggregates rather than streaming every event to the browser.

## Testing

- Smoke test with `streamlit.testing.v1.AppTest` for each page against a fixture DB and a stubbed API.
- Plot functions unit-tested on fixture reports.

## Interactions

- Reads 016/018/014; reuses 004's plotting code.

## Success criteria

- A five-minute demo script (in `docs/demo.md`) can be executed on the laptop: upload pcap → watch decisions → change budget → see mean K move → view unknown alerts → open experiment tables.

## Open questions

- Whether to embed the report's LaTeX table export here (default: no, `scripts/make_tables.py` does it).
