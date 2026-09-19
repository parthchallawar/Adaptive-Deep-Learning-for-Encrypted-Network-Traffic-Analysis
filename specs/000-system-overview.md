# Spec 000: System Overview and Research Thesis

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** reference, never implemented
- **Background:** `docs/research-analysis.md`

## Problem

Encrypted traffic (TLS 1.3, ECH, QUIC) leaves only packet metadata observable. Operators need to classify flows for QoS, capacity planning and security, and they need to do it (a) without decrypting anything, (b) as early in the flow as possible, (c) with a way to say "unknown", (d) within a compute budget, and (e) in a way that survives traffic drift. Existing work addresses these one or two at a time; nothing trains, calibrates and controls a classifier as one anytime, open-world decision process, and nothing studies whether self-supervised pretraining helps early decisions.

## Goals

1. A metadata-only classifier that emits a decision after every packet of the first 30 and can stop early.
2. Self-supervised pretraining (C1) that measurably improves early-prefix accuracy, low-label accuracy and packet-loss robustness.
3. A learned three-way stopping policy (C2): continue / commit / reject-as-unknown.
4. A budget controller (C3) that holds a target packet or compute budget under drift.
5. A unified evaluation protocol (C4): accuracy, earliness, unknown detection, drift, latency, compute, all as functions of prefix length K and of test period.
6. A working system: PCAP and live input, FastAPI inference service, Streamlit dashboard, SQLite storage, MLflow tracking, all run as local processes (containerised deployment deferred, spec 019).
7. A report/paper with ablations and multi-seed statistics.

## Non-goals

- New attention architectures or byte-level foundation models.
- Decrypting, fingerprinting users, or any payload inspection.
- Multi-node or in-switch (P4) deployment.
- Formal conformal guarantees as a headline claim (may be included as an optional module citing Learn-Then-Test).
- Online continual learning of the backbone (drift *monitoring* is in scope; repair is a stretch goal, spec 012).

## System architecture

```
                      offline (Kaggle GPU + local CPU)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  DataZoo (CESNET TLS-Year22 / QUIC22)  ──┐                               │
 │  PCAP (ISCX, USTC) ── flow builder (dpkt)  ┤──► tensor shards (001-003)   │
 │                                            │                              │
 │  SSL pretraining (007) ─► supervised prefix fine-tune (008)               │
 │        └─► stop / unknown heads (009, 010) ─► calibration per K           │
 │  Baselines (005) ──────────────────────────► evaluation harness (004,013) │
 │  MLflow (014)                                                             │
 └──────────────────────────────────────────────────────────────────────────┘
                      online (local processes, CPU)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ capture/pcap ─► flow builder (002) ─► per-packet PPI ─► PAT stream infer  │
 │      ─► policy (009/010) ─► controller (011) ─► decision                  │
 │ FastAPI (016) ◄─► DB (018) ◄─► Streamlit dashboard (017); drift (012)     │
 └──────────────────────────────────────────────────────────────────────────┘
```

## Module map

| Spec | Module | Source location |
|---|---|---|
| 001 | Datasets and acquisition | `src/data/datasets/`, `scripts/download_*.py` |
| 002 | PCAP / live to flows to PPI | `src/data/pcap/`, `src/data/flows.py` |
| 003 | Feature representation and preprocessing | `src/data/features.py`, `src/data/tensors.py` |
| 004 | Splits and evaluation protocol | `src/evaluation/protocol.py`, `src/evaluation/metrics.py` |
| 005 | Baseline models | `src/models/baselines/` |
| 006 | Prefix-Aware Transformer backbone | `src/models/pat.py` |
| 007 | Self-supervised pretraining | `src/training/pretrain.py` |
| 008 | Supervised prefix training and calibration | `src/training/finetune.py`, `src/training/calibrate.py` |
| 009 | Adaptive stopping policy | `src/inference/policy.py` |
| 010 | Unknown / anomaly detection | `src/inference/unknown.py` |
| 011 | Resource budget controller | `src/inference/controller.py` |
| 012 | Drift monitoring (and optional repair) | `src/inference/drift.py` |
| 013 | Efficiency and latency benchmarking | `src/evaluation/efficiency.py` |
| 014 | Experiment tracking and reproducibility | `src/utils/tracking.py`, `configs/` |
| 015 | Kaggle training pipeline | `kernel/`, `scripts/kaggle_sync.sh` |
| 016 | Inference service (FastAPI) | `src/service/` |
| 017 | Dashboard (Streamlit) | `src/dashboard/` |
| 018 | Storage and database | `src/service/db/` |
| 019 | Deployment (Docker) — **deferred**, not in the current plan | `docker/`, `docker-compose.yml` |
| 020 | Testing and quality | `tests/` |

## Key definitions (used across all specs)

- **Flow:** bidirectional 5-tuple connection (src IP, dst IP, src port, dst port, L4 proto), first-seen direction = client to server.
- **PPI (per-packet information):** for the first `K_max = 30` payload-carrying packets, the tuple `(ipt_ms, dir in {+1,-1}, size_bytes, push_flag)`. Matches CESNET DataZoo column order `[IPT, DIR, SIZE, PUSH_FLAGS]`.
- **Prefix K:** the first K packets of a flow's PPI, 1 <= K <= 30.
- **Decision:** one of `COMMIT(class)`, `REJECT(unknown)`, `CONTINUE`. A flow's final outcome is its first non-CONTINUE decision, or a forced decision at K = min(30, flow length).
- **Budget:** a target on E[K] (mean packets read), E[cost] (packets x layers evaluated), or a quantile of K.
- **Period:** a named time slice used for temporal splits: an ISO week of the raw release (`WEEK-2022-31`, the default granularity) or a DataZoo month (`M-2022-8`) when working through the HDF5 path.

## Research hypotheses

| ID | Hypothesis | Primary metric |
|---|---|---|
| H1 | Prefix-predictive SSL improves accuracy at K <= 8 and reduces mean packets-to-decision at equal accuracy vs training from scratch and vs masked-only SSL | accuracy@K curve, AUC of accuracy-vs-K, mean K at 95% of full accuracy |
| H2 | The learned commit-safety head dominates max-prob thresholds (ECHO/CAPE-style) and an RL policy (FastFlow-style) on the accuracy vs mean-K Pareto front | Pareto front area, accuracy at fixed mean K in {4, 6, 8} |
| H3 | Prefix-conditioned unknown scoring rejects unknown apps earlier and with higher AUROC than a fixed-checkpoint energy score | AUROC(K), mean K at rejection, FPR@95TPR |
| H4 | The budget controller holds E[K] within 5% of target across the 18-week drift horizon while static thresholds drift out of budget | budget error over time, accuracy at fixed budget |
| H5 | Drop/jitter-augmented SSL improves robustness to 5% to 20% packet loss and reordering | accuracy drop under perturbation |

## Success criteria (project level)

- All 21 specs implemented or explicitly descoped with a reason; each has passing tests.
- Main table reproduced from configs with three seeds; results and plots in `results/summaries/`.
- End-to-end demo: upload a PCAP or capture live, watch per-flow decisions arrive early, see unknown alerts and the budget knob work.
- Report/paper draft with related work covering the sources in `docs/research-analysis.md`.

## Scope decisions (owner, 2026-09-17)

- **Containerised deployment is out of scope.** Spec 019 is deferred; the service, dashboard, database and MLflow store all run as local processes. The Linux-only `ipfixprobe` exporter is therefore optional too (spec 002 keeps a pure-Python backend as the default path).
- **Kaggle credentials are configured** at `~/.kaggle/kaggle.json` (account `parthrchallawar`); the CLI authenticates successfully.

## Open questions

- Confirm whether the owner wants the optional depth-axis exit (spec 006 section "Depth exits") in the main plan or as a stretch goal.
- Whether Kaggle kernels may use internet (phone-verified account). If not, data reaches Kaggle as a dataset mount and code as an attached utility dataset (spec 015); both paths are specified.
