# Master roadmap

- **Written:** 2026-11-17, at the end of phase 3. **Owner:** Parth Challawar.
- **Purpose:** one page that says where the project is, what comes next, in what order, and what to do at each fork. Read this first, then the plan of the current phase.
- **Keep it current:** update the status table and the decision log whenever a phase or a decision point closes.

## The project in four sentences

AnytimeETC classifies encrypted network flows from packet sizes, directions and timing only. It gives an answer after every one of the first 30 packets, can stop early, and can answer "unknown". A controller holds a packet budget while traffic drifts over weeks. The research question is whether training, calibrating and controlling the classifier as one early-decision process beats doing those things separately.

## Status

| Phase | Specs | State | Plan |
|---|---|---|---|
| 1 Data pipeline | 001–004 | done | [phase-1](phase-1-data-pipeline.md) |
| 2 Tracking, Kaggle, baselines | 014, 015, 005 | done (two minor exit items left open, see the plan) | [phase-2](phase-2-tracking-kaggle-baselines.md) |
| 3 PAT, SSL, prefix training | 006–008 | done; three seeds for the headline comparison; H1 partly supported, H5 not supported | [phase-3](phase-3-pat-ssl-prefix-training.md), [results](../docs/results-phase-3.md) |
| 4 Unknown, stopping, budget | 010, 009, 011 | done; three seeds; H3 AUROC bars met, early-rejection not met; H2 not met against P-ECHO / P-RL; H4 holds the budget | [phase-4](phase-4-unknown-stopping-budget.md), [results](../docs/results-phase-4.md) |
| 5 Drift and efficiency | 012, 013, rest of 004 | done; decay measured, two label-free signals track it, streaming step misses 1 ms | [phase-5](phase-5-drift-efficiency.md), [results](../docs/results-phase-5.md) |
| 6 Service, dashboard, demo | 018, 016, 017 | built 2026-11-18 (branch `phase6`); decisions equal the offline replay; ~300 packet events/s, not the 5,000 target; web UI instead of Streamlit (S10); owner's demo run-through open | [phase-6](phase-6-service-dashboard-demo.md), [demo](../docs/demo.md) |
| **7 Report and paper** | (new spec 021) | **next** | [phase-7](phase-7-report-paper.md) |

The plans for phases 5 to 7 were drafted ahead of time on 2026-11-17 at the owner's request. `specs/README.md` says to write a plan just before its phase; treat those three as starting drafts and re-check them against real results when each phase begins.

## Flow

```
Phase 3 (done)
   │  inherits: calibrated per-K probabilities, safe head, proj, next-packet heads, 9 checkpoints
   ▼
Phase 4  T0 secure checkpoints → T1 scores → T2 unknown (H3) → T3 policy (H2, H1c) → T5 controller (H4)
   │        DP-4a after T2        DP-4b after T3                DP-4c after T5
   ▼
Phase 5  drift signals (012) + weekly accuracy decay + efficiency table (013)
   │        DP-5a: is label-free adaptation worth building?
   ▼
Phase 6  database (018) → service (016) → dashboard (017) → demo script
   │        DP-6a: which demo story (plain / fraud-motivated / TQH-C2 bonus)
   ▼
Phase 7  remaining seeds and baselines → figures and tables → report → paper draft
```

Phase 7's writing can start as soon as phase 4 closes; phases 5 and 6 add sections, they do not block the core argument.

## Dependencies between phases

| Needs | From | Why |
|---|---|---|
| Phase 4 | the nine phase-3 checkpoints | every score is computed from them; they are not on local disk yet (phase-4 T0) |
| Phase 5 monitors | phase 4's time-ordered scores and replay harness | drift signals are aggregates of the decision stream |
| Phase 5 efficiency | phase 4's policy replay | packets-to-decision and time-to-decision come from it |
| Phase 6 service | phase 4's `inference/` modules, phase 5's monitor | the service runs the same code online |
| Phase 6 dashboard | the service and the database | it only reads |
| Phase 7 | every results document | it quotes them and nothing else |

## Decision points

Each has a default so that work never stops to wait for a discussion.

| ID | When | Question | Default if unsure |
|---|---|---|---|
| DP-4a | after phase-4 T2 | Does the combined unknown score beat energy alone by ≥ 1 AUROC point at some K ≤ 8? | No → ship energy with per-K quantile thresholds, report the negative result, continue |
| DP-4b | after phase-4 T3 | Does the feature knock-out show the next-packet-entropy feature matters? Is the ELECTS comparison worth ~3 GPU-hours? | Retrain only if the knock-out moves accuracy ≥ 0.2 points; skip ELECTS and mark it descoped |
| DP-4c | after phase-4 T5 | Is the static threshold already within 5% of budget on the drift weeks? | Report honestly; lead the H4 section with the synthetic disturbances |
| DP-5a | start of phase 5 | Build label-free adaptation (spec 012 stretch)? | No. Build monitoring only; revisit only if phases 4–6 finish with time to spare |
| DP-5b | start of phase 5 | Add D2 (QUIC) cross-protocol evaluation? | No. D2 is not exported; it costs a new export kernel run. List it as future work |
| DP-6a | start of phase 6 | Which demo story? | Plain PCAP replay demo. Add the fraud-motivated demo (a mock payment API and scripted clients) only as a second scene if the first works; see `docs/future-work.md` FEP-1 |
| DP-6b | start of phase 6 | Live capture on Windows (needs Npcap)? | No. PCAP upload and replay only; live capture stays deferred |
| DP-7a | start of phase 7 | Three seeds for B5, B2 and the low-label pairs? | Yes for B5 and B2 (they appear in the main table); low-label pairs stay one seed and are labelled so |
| DP-7b | start of phase 7 | Reproduce 30pktTCNET and a CAPE-Net-style backbone as extra baselines? | 30pktTCNET only if its public weights load in under a day of work; CAPE-Net is represented by P-CAPE on our backbone, stated as such |

## What the paper claims under each outcome

Decide the story from results, never the other way round.

| Outcome | Claim that still stands |
|---|---|
| H2, H3, H4 all supported | The full thesis: one anytime, open-world, budget-controlled process |
| H2 fails (Safe-Stop ≈ thresholds) | Per-K calibration makes a single threshold meaningful at every K; the learned safety head is reported as not needed. C3 and the robustness result carry the paper |
| H3's relative criterion fails | Per-K quantile thresholds reject unknowns earlier than a fixed-K gate (if the rejection-K criterion holds); the Mahalanobis and surprise signals are reported as not helping |
| H4's contrast is small | The controller holds the budget under synthetic shifts; real drift on this dataset is too mild to break a static threshold in 18 weeks, which is itself a finding |
| Already established (phase 3) | Prefix-trained PATs lose about a fifth of the accuracy a GRU loses under 10% packet drop; SSL initialisation adds 6–7 points at K ≤ 3; SSL adds no robustness |

## Checkpoints (things to verify before moving on)

| When | Check |
|---|---|
| Before any Kaggle kernel push | Every checkpoint the next session needs, and every one not yet on local disk, is pulled or in the slim checkpoint dataset |
| After every pull | Count the files; a pull can stop early and exit 0 |
| Before quoting a number | It appears in a `results/summaries/*_table.md` produced by `scripts/make_tables.py` |
| Before closing a task | `pytest`, `ruff check src/ tests/ scripts/`, `mypy src/` clean; progress-log line written |
| Before closing a phase | Exit criteria listed in its plan each marked met or explicitly open; `specs/README.md`, `plans/README.md`, `CLAUDE.md` and this file updated |

## Effort estimate

Working days of focused effort for one developer with an AI assistant. They are estimates, not commitments.

| Phase | Days | GPU-hours | Main uncertainty |
|---|---|---|---|
| 4 | 8–12 | 0–2 | CPU speed of the score pass; whether a retrain ablation is needed |
| 5 | 5–7 | 0–1 | how informative the drift signals turn out to be |
| 6 | 8–12 | 0 | streaming inference and the async database on Windows |
| 7 | 10–15 | 6–10 | extra seeds and baselines |

## Long-term vision

Beyond the final-year project, the same machinery points at three directions, each written up in [`docs/future-work.md`](../docs/future-work.md):

1. **Behaviour checks behind an address allow-list** (motivated by the 2026 NBFC payment-API frauds): enrol a site's genuine client from a few hundred flows and reject anything else within the first packets.
2. **Decisions across flows, not only within one:** evidence that adds up over many connections of one host (beaconing, campaigns).
3. **Label-free repair under drift** using the self-supervised objective.

None of these is in scope until phases 4 to 7 are done.

## Decision log

| Date | Decision | Where recorded |
|---|---|---|
| 2026-09-17 | No containerised deployment; SQLite; local processes | spec 019, `CLAUDE.md` |
| 2026-10-03 | B1 XGBoost removed | spec 005, phase-2 plan |
| 2026-11-15 | Tuned recipe (dropout 0, label smoothing 0) is the recommended PAT recipe | `docs/results-phase-3.md` |
| 2026-11-17 | The NBFC fraud story is used as motivation only; no fraud-detection claim; no traffic recording before phase 6 | `docs/future-work.md` FEP-1, `docs/decisions.md` ADR-014 |
| 2026-11-17 | CESNET-CC25 rejected as a dataset; TQH-C2 kept only as an optional demo | `docs/future-work.md` |
| 2026-11-17 | Phase 4 is local-first, replays from compact score files, and keeps `inference/` torch-free | phase-4 plan D1–D3 |
