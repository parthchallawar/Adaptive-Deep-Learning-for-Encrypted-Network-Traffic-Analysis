# specs/

Feature and requirement specifications. Write a spec **before** implementation starts, and keep it updated as the implementation evolves (status: draft → approved → implemented).

One file per feature: `specs/<NNN>-<short-name>.md`. Use [`template.md`](./template.md) as the starting point; the fuller structure used by the existing specs (Problem, Goals, Non-goals, How it works / Design, Inputs and outputs, Edge cases, Performance, Testing, Interactions, Success criteria, Open questions) is preferred for anything non-trivial.

Background and rationale for all specs: [`docs/research-analysis.md`](../docs/research-analysis.md).

## Index

| # | Spec | Area | Research contribution |
|---|---|---|---|
| [000](000-system-overview.md) | System overview and research thesis | architecture, hypotheses H1 to H5 | all |
| [001](001-datasets-and-acquisition.md) | Datasets and acquisition | CESNET-TLS-Year22, QUIC22, ISCX-VPN, USTC-TFC | |
| [002](002-pcap-flow-pipeline.md) | PCAP and live capture to flows to PPI | ipfixprobe / NFStream / dpkt backends | |
| [003](003-feature-representation-and-preprocessing.md) | Feature representation and preprocessing | shards, tokens, masks, augmentations | |
| [004](004-splits-and-evaluation-protocol.md) | Splits and evaluation protocol | temporal, open-set, metrics vs K and vs period | C4 |
| [005](005-baseline-models.md) | Baseline models | XGBoost, CNN, GRU/LSTM, 30pktTCNET, policy baselines | |
| [006](006-prefix-aware-transformer-backbone.md) | Prefix-Aware Transformer backbone (PAT) | causal encoder, heads, streaming | |
| [007](007-self-supervised-pretraining.md) | Self-supervised prefix-predictive pretraining | NPP + PFC, augmentation | C1 |
| [008](008-supervised-prefix-training-and-calibration.md) | Supervised prefix training and calibration | multi-prefix loss, per-K calibration | |
| [009](009-adaptive-stopping-policy.md) | Adaptive stopping policy | continue / commit / reject | C2 |
| [010](010-unknown-and-anomaly-detection.md) | Unknown-application and anomaly detection | per-K energy + Mahalanobis + surprise | C2 |
| [011](011-resource-budget-controller.md) | Automatic resource budget controller | PI control on thresholds | C3 |
| [012](012-drift-monitoring-and-adaptation.md) | Drift monitoring and optional label-free adaptation | CUSUM on label-free signals | |
| [013](013-efficiency-and-latency-benchmarking.md) | Efficiency and latency benchmarking | FLOPs, latency, time-to-decision | |
| [014](014-experiment-tracking-and-reproducibility.md) | Experiment tracking and reproducibility | MLflow, configs, seeds | |
| [015](015-kaggle-training-pipeline.md) | Kaggle GPU training pipeline | quotas, resumable kernel | |
| [016](016-inference-service-api.md) | Inference service (FastAPI) | streaming decisions, budget API | |
| [017](017-dashboard.md) | Dashboard (Streamlit) | monitor, inspector, budget, drift, experiments | |
| [018](018-storage-and-database.md) | Storage and database | SQLite / PostgreSQL schema | |
| [019](019-deployment-docker.md) | Deployment (Docker) | compose stack | |
| [020](020-testing-and-quality.md) | Testing, quality and CI | invariant tests, tiers | |
