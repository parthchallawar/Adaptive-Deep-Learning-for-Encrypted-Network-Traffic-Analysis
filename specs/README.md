# specs/

Feature and requirement specifications. Write a spec **before** implementation starts, and keep it updated as the implementation evolves (status: draft → approved → implemented).

One file per feature: `specs/<NNN>-<short-name>.md`. Use [`template.md`](./template.md) as the starting point; the fuller structure used by the existing specs (Problem, Goals, Non-goals, How it works / Design, Inputs and outputs, Edge cases, Performance, Testing, Interactions, Success criteria, Open questions) is preferred for anything non-trivial.

Background and rationale for all specs: [`docs/research-analysis.md`](../docs/research-analysis.md).

**Where the project is and what comes next:** [`plans/ROADMAP.md`](../plans/ROADMAP.md). **How to work a task:** [`docs/playbook.md`](../docs/playbook.md). **Closed decisions:** [`docs/decisions.md`](../docs/decisions.md). **Known gaps and risks:** [`docs/gaps-and-risks.md`](../docs/gaps-and-risks.md).

On 2026-11-17 the specs not yet built (009–013, 016–018) each gained an "Implementation contract" or "Pre-build notes" section that ties them to the code that now exists. Those sections override the older text of the same spec where the two disagree. Plans for phases 5 to 7 were drafted ahead of time at the owner's request (an exception to the "plan just before implementing" rule below); re-check each against real results when its phase starts.

## How to work through these

One spec at a time, in the build order below. For each: read the spec, write its plan in [`plans/`](../plans/) **just before** implementing it, build to the spec's own success criteria and no further, land it with tests, then move on. Do not write all the plans up front, because what you learn in one spec changes the next.

**Spec numbers are identities, not an order.** They were assigned when the specs were written and are referenced throughout the documents and the code, so they never change. The build order is separate and is recorded here and in each spec's `Build step` line. Where the two disagree, the build order wins.

The order below is a verified topological sort of the `Depends on` line of every spec: nothing is built before something it needs.

## Build order

| Step | Spec | Why here | State |
|---|---|---|---|
| 1 | [001](001-datasets-and-acquisition.md) Datasets and acquisition | nothing exists without data | D3/D4 downloaders done; D4 downloaded and exported for real; D1 Path B exporter done, one real day exported; D2 not started; D3 blocked on user registration |
| 2 | [002](002-pcap-flow-pipeline.md) PCAP and live capture to PPI | defines the record everything reads | pcap path done, tested, and run against real D4 captures; live capture and optional backends deferred to phase 5 / indefinitely by design |
| 3 | [003](003-feature-representation-and-preprocessing.md) Features and preprocessing | the on-disk format every module loads | done: schema, shard writer, tokeniser, augmentations and stream tensoriser all built and tested |
| 4 | [004](004-splits-and-evaluation-protocol.md) Splits and evaluation protocol | first pass only; it grows later | first pass done: phase-1 metrics, split loading and leakage assertions built and tested; `load_split` run for real against D4 and, after `protocol.trim_period_boundaries`, against the real 12.5M-flow D1 export too. The `evaluate(model, split) -> Report` entry point (`adl_etc.evaluation.run`) is built and has produced real reports from real trained checkpoints (D4 in T7, D1 in T8). Open-set/drift/budget metrics still extend at steps 11 to 13 |
| 5 | [014](014-experiment-tracking-and-reproducibility.md) Experiment tracking | **moved ahead of 005**: the first trained model must be tracked, or every baseline gets rerun | done for what phase 2 needs: config, seeding, run identity, the run-directory tracker, SQLite MLflow sync and `adl_etc.evaluation.plotstyle` (colour-blind-safe palette, built in T7) all built, tested, and exercised for real -- three real Kaggle-trained runs are in `results/mlflow.db` (T8). `scripts/make_tables.py` (the results-table script) is T9, in progress; a model registry is not built |
| 6 | [015](015-kaggle-training-pipeline.md) Kaggle training pipeline | same reason: the first GPU run must be reproducible | done and run for real: the D1 export kernel (T3, 12,521,630 flows) and the training kernel (T8, B3 on D1, 3 seeds, real measured throughput/epoch-time/GPU-hours recorded in the spec) have both completed successfully on Kaggle |
| 7 | [005](005-baseline-models.md) Baseline models | first real numbers | B2 CNN / B3 GRU-LSTM built and tested (T6; B4 deferred to phase 3); B1 XGBoost removed 2026-10-03 (its real D1 run never completed); P-ECHO and P-CAPE built (T7); real 3-seed D1 numbers on Kaggle GPU: B2 `val_macro_f1` 0.865, B3 0.812. B5 moved to phase 3 (correction 3) |
| 8 | [006](006-prefix-aware-transformer-backbone.md) Prefix-Aware Transformer | the research backbone | **built and tested locally** (`models/pat.py`: PAT + PATStream, 885,821 params; causality, streaming equivalence, strict overfit and real D4/D3 checks pass); B5 trained on `d1_open` (one seed). **run for real on Kaggle 2026-11-13/15** (three sessions, kernel v14-v16, ~14 GPU-hours, one seed): see `docs/results-phase-3.md` and `results/summaries/phase3_table.md` |
| 9 | [007](007-self-supervised-pretraining.md) Self-supervised pretraining | contribution C1 | **code built and tested locally** (`training/ssl.py`, `data/augment_batch.py`, `training/probes.py`; NPP+PFC with twin masking, MPM, probes; bit-exact resume through augmented views). NPP+PFC pretraining **run for real on Kaggle 2026-11-13/15** (three sessions, kernel v14-v16, ~14 GPU-hours, one seed): see `docs/results-phase-3.md` and `results/summaries/phase3_table.md`. H1 partly supported, H5 not supported (provisional, one seed); MPM never fine-tuned |
| 10 | [008](008-supervised-prefix-training-and-calibration.md) Prefix training and calibration | produces the calibrated scores the policy needs | **code built and tested locally** (prefix objective, strict pretrained loading, label regimes with a 3,000-step floor, fine-tuning augmentation, per-K calibration in `evaluation/calibration.py`). **run for real on Kaggle 2026-11-13/17** (four sessions, kernel v14-v17, ~20 GPU-hours): see `docs/results-phase-3.md` and `results/summaries/phase3_table.md`. Recommended recipe is now `ft_pat_{,ssl_}d1_tuned.yaml` (dropout 0, label smoothing 0), three seeds; the other regimes one seed |
| 11 | [010](010-unknown-and-anomaly-detection.md) Unknown and anomaly detection | **before 009**, which consumes its unknown score | not started; **next**. Plan written 2026-11-17 ([phase 4](../plans/phase-4-unknown-stopping-budget.md), T1–T2); the spec has an implementation contract |
| 12 | [009](009-adaptive-stopping-policy.md) Adaptive stopping policy | contribution C2 | not started; phase-4 plan T3–T4; implementation contract in the spec |
| 13 | [011](011-resource-budget-controller.md) Resource budget controller | contribution C3 | not started; phase-4 plan T5; implementation contract in the spec ("compute" budget descoped) |
| 14 | [012](012-drift-monitoring-and-adaptation.md) Drift monitoring | needs the policy and controller to watch | not started; forward-draft plan ([phase 5](../plans/phase-5-drift-efficiency.md)); monitoring only, adaptation not scheduled |
| 15 | [013](013-efficiency-and-latency-benchmarking.md) Efficiency benchmarking | measures everything above | not started; forward-draft plan (phase 5) |
| 16 | [018](018-storage-and-database.md) Storage and database | **before 016**, which writes to it | not started; forward-draft plan ([phase 6](../plans/phase-6-service-dashboard-demo.md)) |
| 17 | [016](016-inference-service-api.md) Inference service | puts the research online | not started; forward-draft plan (phase 6) |
| 18 | [017](017-dashboard.md) Dashboard | reads the service and the results | not started; forward-draft plan (phase 6) |
| 19 | [021](021-report-and-paper.md) Report, paper and presentation | the final artefact; writing may start after step 13 | not started; forward-draft plan ([phase 7](../plans/phase-7-report-paper.md)) |

### Outside the order

| Spec | Why |
|---|---|
| [000](000-system-overview.md) System overview and research thesis | reference. Read it first, never implemented. |
| [019](019-deployment-docker.md) Deployment (Docker) | **deferred** by owner decision on 2026-09-17. Build only if reactivated. |
| [020](020-testing-and-quality.md) Testing, quality and CI | **continuous.** Not a queue position: every spec lands with its own tests. Partly in place already. |

### Three specs are scaffolding, not one-off builds

Specs 004, 014 and 020 grow with everything else. Build what you can test when their step arrives, then extend them as later specs give them real material. Spec 004 in particular cannot be finished at step 4: its open-set, drift and budget metrics only become testable at steps 11 to 13.

## Research contributions

| Contribution | Specs |
|---|---|
| C1 prefix-predictive self-supervised pretraining | 007 |
| C2 learned three-way stopping with prefix-conditioned unknown scoring | 009, 010 |
| C3 budget-tracking resource controller | 011 |
| C4 anytime open-world drift benchmark protocol | 004 |
