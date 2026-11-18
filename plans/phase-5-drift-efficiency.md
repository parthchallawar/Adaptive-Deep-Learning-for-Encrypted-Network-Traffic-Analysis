# Plan: Phase 5 — drift monitoring and efficiency benchmarking

- **Specs:** `specs/012-drift-monitoring-and-adaptation.md` (monitoring only), `specs/013-efficiency-and-latency-benchmarking.md`, and the drift part of `specs/004-splits-and-evaluation-protocol.md` (build steps 14, 15)
- **Status:** done (2026-11-18). T1 weekly decay, T2 drift monitor, T3 efficiency and T4 tables, results document and paperwork all landed; adaptation, the canary alert, ONNX and int8 not built (descoped, `docs/future-work.md`). Verdicts in `docs/results-phase-5.md`.
- **Starts when:** phase 4's exit criteria hold. **Exit gate:** phase 6.
- **Roadmap:** [`ROADMAP.md`](ROADMAP.md).

## Purpose

Two questions the report must answer with measurements:

1. How fast does accuracy decay over the 18 drift weeks, and can the system notice it **without labels**?
2. What does each model and policy cost, in packets waited and in compute?

Neither needs new training. Both read phase 4's score files and checkpoints.

## Inputs from phase 4

- `ScoreArrays` for nine runs on `val`, `test_id`, `test_drift` (with a `period` per flow).
- `inference/policy.py`, `inference/unknown.py`, `inference/controller.py` (torch-free) and `evaluation/replay.py`, `evaluation/controller_replay.py`.
- Nine verified checkpoints on local disk.

## Scope decisions

| Item | Decision | Reason |
|---|---|---|
| Label-free adaptation (spec 012 stretch: re-calibration, SSL refresh, clustering) | **Not built** (roadmap DP-5a) | It needs GPU retraining and its own evaluation; monitoring is the required part. Kept in `docs/future-work.md` FEP-3 |
| D2 QUIC cross-protocol drift | Not built (DP-5b) | D2 is not exported |
| ONNX Runtime numbers | Optional, time-boxed to half a day | The PAT writes its attention out by hand for determinism; export may not work first time. Eager PyTorch numbers are the required ones |
| GPU throughput | Reuse the measured Kaggle numbers already recorded in spec 015 (B5 ~13,400, prefix fine-tune ~11,600, SSL ~5,750, GRU ~43,000 flows/s) | No new GPU session for a number we have |
| int8 quantisation | Not built | spec 013 already lists it as appendix-only |

## Tasks

### T1 — Weekly decay curves (spec 004's drift protocol)

`evaluation/drift_curves.py` (NumPy): from a `ScoreArrays`, per period: accuracy and macro-F1 on known flows at K = 5, 8, 30 and at the policy's own decisions; unknown AUROC at K = 8; mean K under a fixed `θ`. Output `drift.json` per run and a plot `plot_weekly_decay` in `plotstyle.py`.

Report: three seeds, tuned SSL PAT, tuned scratch PAT and B3; the decay from `test_id` to the last drift week; whether early-K accuracy decays faster than K = 30 accuracy (this is the "drift × earliness" interaction that spec 000 says nobody has studied).

**Done when:** `results/summaries/phase5_table.md` §1 has the weekly table.

### T2 — Drift monitor (spec 012, required part)

`src/adl_etc/inference/drift.py` (torch-free), tests in `tests/inference/test_drift.py`, config `configs/drift/default.yaml`.

- `Cusum(h=5.0, k=0.5)`, two-sided, on standardised values; `update(x) -> bool`; `reset()`.
- `WindowStats` (from one window of 5,000 decided flows): `mean_k`, `commit_rate`, `reject_rate`, `mean_safe_at_commit`, `mean_pkt_nll` (mean next-packet negative log-likelihood over the packets read), `prior_kl` (KL divergence between the committed-class histogram, smoothed by adding 1e-3 to every class, and the validation histogram), and `theta_commit` when the controller is running.
- `DriftMonitor(reference: dict[str, (mean, std)], warmup=5)`: `update(stats) -> DriftEvent(level, alarms)`; level is `warning` if one signal alarms and `alarm` if two or more alarm within three windows.
- Reference means and standard deviations come from replaying `val` in windows. `h` is raised until `val` gives zero alarms.

Two modes are reported, because the controller hides drift in `mean_k` by design:

| Mode | Primary signals |
|---|---|
| static `θ` | all six |
| controller on | `mean_safe_at_commit`, `mean_pkt_nll`, `prior_kl`, `reject_rate`, and `theta_commit` itself (a drifting threshold is the controller telling us the traffic changed) |

Optional labelled canary (moved here from spec 011): replay a fixed 5,000-flow labelled validation sample every 50 windows and alert if accuracy falls more than 5 points below its validation value.

Validation (offline): Spearman correlation between each signal's weekly mean and that week's macro-F1 from T1; the lead time between the first alarm and the first week whose macro-F1 is 3 points under `test_id`.

Tests: a synthetic step of 2 standard deviations is detected within 5 windows; 500 windows of stationary noise give no alarm; warm-up suppresses the first 5 windows; `prior_kl` is 0 for identical histograms.

**Done when:** the signal-vs-decay table exists for three seeds and at least the required success criterion of spec 012 can be judged.

### T3 — Efficiency benchmark (spec 013)

`src/adl_etc/evaluation/efficiency.py` (torch imported inside functions), `scripts/benchmark.py`, `configs/eval/efficiency.yaml`, test `tests/evaluation/test_efficiency.py` (smoke: every registered model returns every field).

| Measurement | How | Models |
|---|---|---|
| Parameters | `sum(p.numel())` | all |
| FLOPs per flow | `torch.utils.flop_counter.FlopCounterMode`, batch 1 | all |
| CPU latency per flow | 50 warm-up + 500 timed runs, median and p95, 4 threads, batch 1 and 64 | all |
| Streaming step latency | `PATStream.step`, 1,000 flows × 30 steps | PAT |
| Streaming cost of a non-causal model | one full pass per packet | B2 CNN (the argument for causal models) |
| Peak memory | process RSS at batch 64 | all |
| Packets-to-decision | mean, median, p95 of `k` from `replay` | every policy |
| Time-to-decision | sum of the flow's IPTs up to the decision packet, read from the shards by `flow_index` | every policy |

Run each model twice, interleaved, and report the smaller median (laptop thermal throttling). Record the laptop's CPU model, cores and RAM in `results/hardware.json`.

Outputs: `results/<run>/efficiency.json`, `results/summaries/efficiency.md`, and an accuracy-vs-cost scatter (`plot_efficiency_scatter`).

**Done when:** the efficiency table is complete for the PAT, B5, B2, B3 and for Safe-Stop, P-ECHO, P-CAPE, P-RL.

### T4 — Tables, results document, paperwork

`scripts/make_tables.py` gains `phase5_table.{md,tex}`; `docs/results-phase-5.md` gives the verdicts; update `specs/README.md`, `plans/README.md`, `CLAUDE.md`, `ROADMAP.md`. Spec corrections: 012 (canary moved in; controller-on signal set; adaptation descoped), 013 (GPU numbers reused; ONNX optional), 004 (drift metrics added).

## Success criteria

| Spec | Criterion |
|---|---|
| 012 | Signals implemented and tested; correlation with weekly macro-F1 reported; at least one signal alarms before the end of the drift horizon, or the absence is reported with the measured decay next to it |
| 013 | Efficiency table complete; PAT streaming step ≤ 1 ms and full-flow ≤ 5 ms on the laptop CPU, or the measured values reported as a miss; parameters < 2M (already true: 885,821) |
| 004 | Weekly decay curves for three models, three seeds |

## Risks

| Risk | Mitigation |
|---|---|
| Decay over 18 weeks is small, so no signal alarms | Report it. The dataset authors measured 96.3% → 90.5% over 8 weeks with their model, so some decay is expected; if ours is flatter, that is a result |
| Signals correlate with each other, not with accuracy | Report each signal separately; do not claim a combined detector beyond what the table shows |
| Laptop timings vary run to run | Interleave, repeat, report the minimum median and the spread |
| `FlopCounterMode` misses the hand-written attention | Cross-check against an analytic count for one layer; spec 013's consistency test (sum over steps ≈ full pass within 5%) |

## Exit criteria

1. `pytest`, `ruff`, `mypy` clean.
2. `phase5_table.md` and `efficiency.md` produced by `make_tables.py` / `benchmark.py`.
3. `docs/results-phase-5.md` written with verdicts.
4. Spec corrections landed; index files updated.

## Progress log

- **2026-11-17.** Forward draft written. Not started.
- **2026-11-18.** Re-checked against phase 4's real outputs (the 18 drift weeks decay by 6-8 accuracy points, so there is something to detect; every run's score files are local, 45 files each) and built in one session. **T1** `evaluation/drift_curves.py` (every week cut to 18,000 flows: macro-F1 over 150 classes is biased low on a smaller sample and the drift weeks hold a third of a test week's flows; threshold calibrated on validation to mean K 6). **T2** `inference/drift.py` (torch-free: `Cusum`, `window_stats`, `prior_kl`, `DriftMonitor`, `tune_h`) and `evaluation/drift_monitor.py` (two modes, static and controller; references and `h` from validation windows; Spearman against the weekly macro-F1; lead time). **T3** `evaluation/efficiency.py` (torch inside functions), `scripts/benchmark.py`, `scripts/phase5_policy_cost.py`; the FLOP consistency test (stream sum against full pass within 5%) passes; the PAT streaming step is 1.65 ms and does not improve with threads (per-operation overhead), so it misses the 1 ms bar. **T4** `evaluation/phase5_tables.py`, `scripts/make_phase5_tables.py`, `docs/results-phase-5.md`, spec corrections C11-C20. **Findings:** class-prior KL (Spearman about -0.9) and next-packet NLL (-0.89) track the weekly macro-F1; mean K and reject rate do not; no early-vs-late decay interaction; no stationary data to measure false alarms. psutil is optional (not a declared dependency).
