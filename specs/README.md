# specs/

Feature and requirement specifications. Write a spec **before** implementation starts, and keep it updated as the implementation evolves (status: draft → approved → implemented).

One file per feature: `specs/<NNN>-<short-name>.md`. Use [`template.md`](./template.md) as the starting point; the fuller structure used by the existing specs (Problem, Goals, Non-goals, How it works / Design, Inputs and outputs, Edge cases, Performance, Testing, Interactions, Success criteria, Open questions) is preferred for anything non-trivial.

Background and rationale for all specs: [`docs/research-analysis.md`](../docs/research-analysis.md).

## How to work through these

One spec at a time, in the build order below. For each: read the spec, write its plan in [`plans/`](../plans/) **just before** implementing it, build to the spec's own success criteria and no further, land it with tests, then move on. Do not write all the plans up front, because what you learn in one spec changes the next.

**Spec numbers are identities, not an order.** They were assigned when the specs were written and are referenced throughout the documents and the code, so they never change. The build order is separate and is recorded here and in each spec's `Build step` line. Where the two disagree, the build order wins.

The order below is a verified topological sort of the `Depends on` line of every spec: nothing is built before something it needs.

## Build order

| Step | Spec | Why here | State |
|---|---|---|---|
| 1 | [001](001-datasets-and-acquisition.md) Datasets and acquisition | nothing exists without data | not started |
| 2 | [002](002-pcap-flow-pipeline.md) PCAP and live capture to PPI | defines the record everything reads | pcap path done and tested |
| 3 | [003](003-feature-representation-and-preprocessing.md) Features and preprocessing | the on-disk format every module loads | schema done, shards and tokeniser pending |
| 4 | [004](004-splits-and-evaluation-protocol.md) Splits and evaluation protocol | first pass only; it grows later | not started |
| 5 | [014](014-experiment-tracking-and-reproducibility.md) Experiment tracking | **moved ahead of 005**: the first trained model must be tracked, or every baseline gets rerun | not started |
| 6 | [015](015-kaggle-training-pipeline.md) Kaggle training pipeline | same reason: the first GPU run must be reproducible | credentials and sync script ready |
| 7 | [005](005-baseline-models.md) Baseline models | first real numbers | not started |
| 8 | [006](006-prefix-aware-transformer-backbone.md) Prefix-Aware Transformer | the research backbone | not started |
| 9 | [007](007-self-supervised-pretraining.md) Self-supervised pretraining | contribution C1 | not started |
| 10 | [008](008-supervised-prefix-training-and-calibration.md) Prefix training and calibration | produces the calibrated scores the policy needs | not started |
| 11 | [010](010-unknown-and-anomaly-detection.md) Unknown and anomaly detection | **before 009**, which consumes its unknown score | not started |
| 12 | [009](009-adaptive-stopping-policy.md) Adaptive stopping policy | contribution C2 | not started |
| 13 | [011](011-resource-budget-controller.md) Resource budget controller | contribution C3 | not started |
| 14 | [012](012-drift-monitoring-and-adaptation.md) Drift monitoring | needs the policy and controller to watch | not started |
| 15 | [013](013-efficiency-and-latency-benchmarking.md) Efficiency benchmarking | measures everything above | not started |
| 16 | [018](018-storage-and-database.md) Storage and database | **before 016**, which writes to it | not started |
| 17 | [016](016-inference-service-api.md) Inference service | puts the research online | not started |
| 18 | [017](017-dashboard.md) Dashboard | reads the service and the results | not started |

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
