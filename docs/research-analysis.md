# Research Analysis and Development Direction

- **Status:** proposal (v1), awaiting owner review
- **Date:** 2026-09-17
- **Scope:** literature positioning, novelty, dataset strategy, Kaggle training strategy, technology decisions, roadmap

This document is the "why" behind the specs in `specs/`. Each spec references it. Keep the two aligned.

---

## 1. What the project is (restated from the synopsis and guide)

Classify encrypted flows from **packet metadata only** (size, direction, inter-arrival time, TCP flags), never payload, and do it:

1. with a **Transformer + self-supervised pretraining**,
2. **adaptively**: stop reading packets when the prediction is reliable,
3. with **anomaly / unknown-traffic detection**,
4. under an **automatic resource controller** that trades packet budget and compute effort against flow difficulty,
5. evaluated on **accuracy, robustness (drift, packet loss), latency and compute cost**,
6. exposed through a **FastAPI service, a Streamlit dashboard, a database, MLflow** (the synopsis also lists Docker; containerised deployment was descoped on 2026-09-17, see section 7).

The guide PDF adds: XGBoost / CNN / LSTM baselines, distribution-shift evaluation, ablations and efficiency analysis, and a strict leakage discipline.

## 2. What already exists (state of the art, September 2026)

### 2.1 Pretrained / foundation traffic models

| Work | Input | Pretraining | Notes for us |
|---|---|---|---|
| ET-BERT (2022) | payload/header bytes | BERT MLM | Needs raw bytes; not metadata-only; heavy |
| YaTC (2023), NetMamba (2024/2026) | bytes as images / sequences | masked modeling | Same; NetMamba needs Mamba CUDA kernels |
| TrafficFormer (S&P 2025) | header+payload bytes | BERT-style + flow tasks | Same |
| netFound (2024) | header fields, bursts | 4 losses, 53M to 663M params, 5000 A100-hours | Far beyond Kaggle budget |
| Universal embedding / 30pktTCNET (CESNET, 2025) | **first 30 packets: size, dir, IPT** | supervised ArcFace on QUIC SNI domains | Metadata-only, 1.0M params, **pretrained weights public**: strong baseline |
| FlowCLIP (2026) | first 30 packets: size, dir, IPT | CLIP-style contrastive with domain names | Metadata-only; needs SNI labels; no early-flow study |

Takeaway: byte-level foundation models are neither privacy-preserving (they read bytes) nor trainable on Kaggle. Metadata-only pretraining exists but **none of it is designed for, or evaluated on, partial-flow (prefix) classification**.

### 2.2 Early / adaptive classification of flows

| Work | Mechanism | Backbone | Unknown? | Drift? | Budget control? | SSL? |
|---|---|---|---|---|---|---|
| ECHO (2024) | cascade of per-time-window classifiers, global max-prob threshold | logistic regression | no | no | no | no |
| FastFlow (2025) | per-flow stop via Q-learning; synthetic "unknown" | LSTM | yes (synthetic) | no | no | no |
| TINIEE (2025) | depth-wise early exit in-network | small DNN | no | no | no | no |
| P4-SDN collaborative early exit (2025) | confidence early exit at switch (DDoS) | small DNN | no | no | no | no |
| **CAPE-Net (Aug 2026 preprint)** | 18 packet-axis exits; per-class Learn-Then-Test thresholds; energy OOD gate at K>=10; CUSUM drift monitors + bias-only repair | 2-layer GRU/LSTM, 128-d stem, 256 hidden | yes (energy) | yes (monitor + repair) | no (per-class precision targets, not a system budget) | **no** |

CAPE-Net is the closest work and must be treated as a baseline, not ignored. Its own limitations section lists: no drop/reorder robustness, noisy OOD estimates, repair needs labels ("exploring lighter-weight or partially self-supervised label sources" is named as future work), single recurrent backbone. It was trained on an A100 with a 30M-flow training split.

### 2.3 Open-set / unknown traffic

Reject-option 1D-CNN on CESNET-TLS22 (Luxemburk 2022), RoNeTC (TIFS 2025, second-order probability), SepSpace (2026, contrastive + prototype radius), e-FlowPrint (2025, margin/entropy heuristics), evidential DL for TC (FCS 2024), Mahalanobis p-values (2022), TAO-Net (2025, LLM second stage), M3S-UPD (2025, unknown pattern discovery). **All score a completed flow**; none calibrates unknown-ness as a function of how many packets have been seen.

### 2.4 Drift

CESNET-TLS-Year22 (Sci. Data 2024) is the reference year-long corpus: the authors report 96.3% at T+1 week dropping to 90.5% at T+8. Drift-oriented self-evolving classifier (2025), drift-based dataset stability benchmark (2025), UniAlign (2026, domain alignment + checkpoint ensembling). None studies **how drift interacts with early stopping and budgets**.

### 2.5 General early time-series classification (ECTS)

ELECTS (learned stop head + earliness/accuracy loss), Stop&Hop (RL halting for irregular series), DQeND (2026, DQN under non-stationarity, synthetic data only), TMLR 2025 survey/benchmark. These give a principled toolbox; they have not been applied to Transformer packet-prefix models.

## 3. The gap

Nobody has answered, on real traffic:

- **G1.** Does self-supervised pretraining help *early* decisions (accuracy at 3 to 8 packets), or only full-flow accuracy?
- **G2.** Can the stopping decision be *learned* (and calibrated per prefix length) instead of thresholded, and does it beat threshold/RL stopping on the accuracy-earliness frontier?
- **G3.** Can "I need more packets" be separated from "this is an unknown application" *at every prefix length*, so unknown flows are rejected early instead of being deferred to a fixed checkpoint?
- **G4.** Can a system-level **budget** (mean packets per flow, mean compute per flow, or a tail-latency target) be held automatically while traffic drifts, and what does it cost in accuracy?
- **G5.** Does prefix-aware SSL with drop/jitter augmentation give the packet-loss robustness that CAPE-Net and FastFlow leave open?

## 4. Proposed contribution (the novelty, stated defensibly)

**System name:** *AnytimeETC*. **Backbone:** *PAT, the Prefix-Aware Transformer*.

**Thesis.** *An encrypted-traffic classifier should be trained, calibrated and controlled as an anytime, open-world decision process over packet prefixes, and self-supervised prefix-predictive pretraining is what makes such a process data-efficient and drift-robust.*

Concrete, testable contributions:

| # | Contribution | What is new relative to prior work | How it is evaluated |
|---|---|---|---|
| C1 | **Prefix-predictive self-supervised pretraining** for a causal packet Transformer: next-packet prediction (size bin, direction, IAT bin) + prefix-to-flow InfoNCE alignment (CPC-style), with drop/jitter augmentation | Metadata-only SSL that is causal by construction, so every prefix representation is trained; first SSL study targeted at *earliness*; CPC applied to flows | Accuracy-vs-K curves and mean packets-to-decision, SSL vs scratch vs masked-only vs 30pktTCNET; low-label regimes (1%, 10% labels); packet-loss robustness |
| C2 | **Learned three-way stopping (continue / commit / reject)** with a *commit-safety head* (probability the current prediction is correct) and a *prefix-conditioned unknown score*; the SSL next-packet entropy is an input feature to the stop head | CAPE-Net/ECHO threshold max-prob; FastFlow uses RL; none conditions the OOD score on K nor feeds predictive-future uncertainty into stopping | Pareto front (accuracy vs mean K), earliness-of-rejection for unknown apps, AUROC(K) curves vs an energy-at-K=10 baseline |
| C3 | **Budget-tracking resource controller**: a feedback (PI / dual-ascent) loop on the commit and reject thresholds that holds a target mean packet budget, mean compute (packets x layers), or p95 packets, with an optional depth-axis exit | Early-exit budget controllers exist for vision/NLP; no packet-axis controller for traffic; CAPE-Net targets per-class precision, not throughput budgets | Budget tracking error over 8 to 10 drift weeks and under synthetic class-mix shifts; accuracy vs budget curves; comparison with static thresholds |
| C4 | **Unified anytime open-world drift benchmark protocol** on CESNET-TLS-Year22 (+ QUIC22 cross-protocol, + ISCX-VPN via our own PCAP pipeline): metrics reported *as functions of K* and *of test week* | Prior work reports one axis at a time | The evaluation harness itself (spec 004) is a reusable artefact |

What we explicitly *do not* claim: a new attention architecture, conformal guarantees (we may reuse Learn-Then-Test as an optional module and cite it), or a new drift-detection statistic.

Why this is realistic for one developer: the backbone is under 2M parameters; all data fits in RAM as int16 arrays; SSL on 10M flows of 30 tokens is about 300M tokens per epoch, hours on a T4; the controller is about 100 lines; all pieces share one codebase and one evaluation harness.

## 5. Dataset strategy

### 5.1 Primary: CESNET-TLS-Year22 (size XS)

- 180 web-service classes, 10M flows in XS (2.69 GB HDF5; S is 6.7 GB with 25M; full is 136 GB with 508M), captured across all of 2022 on a 100 Gbps ISP backbone; labels from TLS SNI.
- Per flow: **PPI = first 30 packets x [IPT, direction, size, TCP push flag]** (payload-carrying packets only), flow statistics (bytes, packets, duration, 8-bin histograms), TCP flags, ASN. Exactly our feature scheme.
- Two ways in (spec 001): `cesnet-datazoo` downloads the 2.69 GB HDF5 and provides period selection, known/unknown class splitting and DataLoaders, but exposes only monthly periods for this dataset. The raw release, organised by ISO week and day, is mirrored publicly on Kaggle as `pranjalkar99/cesnet-22` (about 30 GB, verified on 2026-09-17 to contain weeks 0 to 52 as `flows-YYYYMMDD.csv.xz` with per-day statistics files).
- **We default to the raw weekly form.** It gives weekly drift curves instead of monthly ones, which is what the research question needs, and it can be mounted directly inside a Kaggle CPU session, so no data is ever downloaded locally or uploaded. Because it is a third-party re-upload, verification against the shipped per-day statistics and against the canonical HDF5 is mandatory before any result depends on it.
- Known artefact: week 10 has an exporter-induced drift; the authors recommend evaluating weeks 1 to 9 and 11 to 52 as separate regimes. We start at week 11.

### 5.2 Secondary: CESNET-QUIC22 (XS, 2.71 GB, 102 apps + 3 background classes, 4 weeks)

Cross-protocol generalisation (TLS to QUIC), an unlabeled SSL corpus, and the corpus on which CESNET's public 30pktTCNET weights were trained (fair baseline).

### 5.3 PCAP-based: ISCX VPN-nonVPN 2016 (about 28 GB pcap, 14 classes) and USTC-TFC2016 (3.7 GB, 10 malware + 10 benign)

Required because the synopsis promises **PCAP and live input**. We run our own PCAP-to-flow-to-PPI pipeline (spec 002) on them, which (a) validates the end-to-end system, (b) gives comparability with the many papers using ISCX, and (c) USTC's malware classes serve as the "unusual traffic" anomaly test. Known weaknesses of ISCX (few sessions per class, per-file labels, leakage risk) are handled by session-grouped splits and by not using it for the main claims.

### 5.4 Live traffic

Scapy with Npcap on Windows feeding the same flow builder; used for the dashboard demo only, never for reported numbers.

### 5.5 Is it enough? Do we need synthetic data?

Yes, it is enough: 10M labeled flows with a full year of drift is more than any comparable student project uses. Synthetic data is used only as **augmentation** (packet drop, reordering, IAT jitter) and for **synthetic unknowns** (FastFlow-style distortions) as a baseline, never as evaluation data.

### 5.6 Splits (details in spec 004)

- Temporal: train weeks 11 to 26, validate weeks 27 to 30, in-distribution test weeks 31 to 34, then drift-test week by week out to week 52 (an 18-week horizon, longer than the 8 to 10 weeks reported by the dataset authors and by CAPE-Net).
- Open-set: 150 known / 30 unknown classes chosen by a fixed seed; unknown classes appear only in test.
- Cross-dataset: TLS-Year22 to QUIC22; TLS-Year22 to ISCX-VPN (category level).
- Preprocessing statistics fitted on train only; no IP, port, SNI or ASN features in the model input (shortcut prevention).

## 6. Kaggle training strategy (details in spec 015)

Credentials are in place: `~/.kaggle/kaggle.json` for account `parthrchallawar`, CLI 2.2.4 installed and authenticating (checked 2026-09-17).

Facts to re-verify in the Kaggle UI, since they change: 30 GPU-hours per week; a session runs at most 12 h; GPU options are a P100 (16 GB) or 2xT4 (2x16 GB); about 29 GB RAM in GPU sessions; 20 GB persisted output; private dataset quota 200 GB; internet in kernels requires a phone-verified account.

Consequences:

- Package data as **pre-tensorised int16/float16 NumPy shards** (PPI [N,30,4], flowstats [N,43], labels); a 10M-flow shard set is about 3 GB and loads fully into RAM. No DataLoader workers needed.
- Build those shards **in a Kaggle CPU session** from the mounted public mirror, and save them as that kernel's output dataset. CPU sessions do not consume the GPU quota, so preparation is free, nothing is uploaded from the laptop, and no kernel needs internet. Locally produced shards (the PCAP datasets, which have no mirror) still go up through the CLI.
- Backbone at most 2M params, batch 4096 to 8192, AMP fp16, OneCycle LR: an SSL epoch over 10M flows is roughly 15 to 25 min on a T4; a supervised fine-tuning epoch roughly 10 min. A full SSL run (10 epochs) plus fine-tune fits in one 12 h session with margin.
- Every job checkpoints each epoch to `/kaggle/working`, supports resume, and logs to a file-based MLflow store that is pulled into `results/`.
- A weekly plan of 30 GPU-hours = 1 SSL run + 6 to 8 fine-tuning/ablation runs. Baselines (XGBoost, CNN, LSTM) run on CPU locally.
- Kernel code is installed from this repository at a pinned commit so runs are reproducible; configs are YAML files versioned in `configs/`.

## 7. Technology decisions

Kept from the synopsis: Python 3.11, PyTorch, scikit-learn, XGBoost, Scapy/PyShark, Pandas/NumPy, Matplotlib/Plotly, FastAPI, Streamlit, SQLite, MLflow.

Descoped by owner decision on 2026-09-17: **Docker and containerised deployment** (spec 019 is deferred), and with it **PostgreSQL** (SQLite only, behind an engine-neutral data layer). Everything runs as local processes: uvicorn, `streamlit run`, an MLflow file store.

Added, each with a reason:

- `cesnet-datazoo` and `cesnet-models`: dataset API and the public pretrained baseline.
- A pure-Python `dpkt` flow builder as the **default** PCAP/live backend (works on Windows, no external dependency), with **NFStream** and CESNET's **ipfixprobe** as optional cross-checks where a Linux environment exists. Parity is enforced against the documented PPI definition rather than against a tool we may not run.
- **ONNX Runtime**: CPU latency benchmarks and the serving path.
- `pytest`, `ruff`, `pre-commit`: quality.
- OmegaConf YAML configs: reproducibility without Hydra's complexity.

Rejected: Mamba/state-space kernels (hardware dependency, no gain at 30 tokens), byte-level foundation models (privacy and compute), Kubernetes (single-machine scope).

## 8. Roadmap (aligns with the guide's four phases)

The spec-by-spec build order, verified against every declared dependency, is in [`specs/README.md`](../specs/README.md).

| Phase | Weeks | Deliverable |
|---|---|---|
| 1 Data | 1 to 3 | DataZoo loading, tensor shards on Kaggle, PCAP pipeline with feature-parity tests, evaluation harness skeleton (specs 001 to 004) |
| 2 Baselines | 4 to 6 | Experiment tracking and the Kaggle pipeline first (014, 015), then XGBoost, CNN, GRU/LSTM, 30pktTCNET and the fixed-K Transformer; accuracy-vs-K curves (005, 006) |
| 3 Core research | 7 to 11 | SSL pretraining (007), prefix supervision (008), unknown detection (010), then the stopping policy (009) that consumes it, drift + open-set evaluation |
| 4 Control + system | 12 to 14 | Budget controller (011), drift monitoring (012), efficiency benchmarks (013), FastAPI + dashboard + SQLite (016 to 018) |
| 5 Paper | 15 to 16 | Ablations, statistics over seeds, report and paper draft |

## 9. Risks

- **Concurrent work (CAPE-Net).** Mitigation: cite it, reproduce its threshold+energy policy as a baseline, keep claims on the four gaps above.
- **Kaggle quota or internet restrictions.** Mitigation: all data pre-packaged; CPU fallbacks for small models; runs sized under 12 h.
- **ISCX label quality.** Mitigation: used for pipeline validation and category-level results only.
- **Over-scoping.** Depth-axis exits and online drift repair are explicitly optional; the project is complete without them.

## 10. Sources

- CAPE-Net: https://www.researchsquare.com/article/rs-10360175/v1
- FastFlow: https://arxiv.org/abs/2504.02174 ; ECHO: https://arxiv.org/abs/2406.01852
- ELECTS: https://arxiv.org/abs/1901.10681 ; Stop&Hop: https://arxiv.org/abs/2208.09795 ; DQeND: https://arxiv.org/abs/2608.20044 ; ECTS survey: https://arxiv.org/abs/2406.18332
- CESNET-TLS-Year22: https://www.nature.com/articles/s41597-024-03927-4 ; DataZoo: https://github.com/CESNET/cesnet-datazoo ; CESNET-QUIC22: https://zenodo.org/records/10728760 ; cesnet-models: https://github.com/CESNET/cesnet-models ; ipfixprobe: https://github.com/CESNET/ipfixprobe
- Universal embedding (QUIC domain pretraining): https://arxiv.org/abs/2502.12930 ; FlowCLIP: https://arxiv.org/abs/2606.17746 ; Reject-option TLS classifier: https://arxiv.org/abs/2202.11984
- netFound: https://arxiv.org/abs/2310.17025 ; NetMamba: https://arxiv.org/abs/2405.11449 ; NetMamba+: https://arxiv.org/abs/2601.21792
- Open-set: RoNeTC https://dl.acm.org/doi/10.1109/TIFS.2025.3544067 ; SepSpace https://link.springer.com/article/10.1186/s42400-026-00635-x ; e-FlowPrint https://www.preprints.org/manuscript/202504.0946/v1 ; Evidential TC https://dl.acm.org/doi/10.1007/s11704-024-3922-6 ; Mahalanobis UQ https://arxiv.org/abs/2205.05628 ; TAO-Net https://arxiv.org/abs/2512.15753 ; M3S-UPD https://arxiv.org/abs/2505.21462
- Drift: self-evolving https://arxiv.org/abs/2501.04246 ; stability benchmark https://arxiv.org/abs/2512.23762 ; UniAlign https://arxiv.org/abs/2605.17575
- Early-exit survey: https://dl.acm.org/doi/full/10.1145/3698767 ; edge budget control: https://arxiv.org/abs/2604.26470 ; P4-SDN early exit: https://arxiv.org/abs/2509.12291
- Datasets: ISCX VPN https://www.unb.ca/cic/datasets/vpn.html ; USTC-TFC2016 https://github.com/davidyslu/USTC-TFC2016 ; NFStream https://github.com/nfstream/nfstream
- Kaggle: https://www.kaggle.com/docs/efficient-gpu-usage ; 12 h sessions https://www.kaggle.com/product-feedback/302908 ; 200 GB quota https://www.kaggle.com/product-announcements/512322
- CPC: van den Oord et al., arXiv:1807.03748
