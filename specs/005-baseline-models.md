# Spec 005: Baseline Models

- **Status:** partially implemented (plan T6: B1 XGBoost, B2 CNN, B3 GRU/LSTM built and tested on real D3/D4, B4 deferred to phase 3; plan T7, 2026-09-22: P-ECHO and P-CAPE built in `evaluation/policies.py` exactly as the table below describes, including the energy gate and the min-support-20 fallback; P-RL deferred to phase 3)
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 7 of 18
- **Depends on:** 003, 004. **Used by:** 004 (comparisons), 013.

## Problem

The research claims are comparative. Without strong, fairly tuned baselines the results are not defensible. Baselines must cover (a) the classical tabular approach, (b) local-pattern and sequential deep models, (c) the public state-of-the-art metadata-only encoder, and (d) the closest adaptive-inference policies so that contributions C1 to C3 are each compared to the right prior.

## Goals

- Implement and tune: XGBoost (flow statistics), 1D-CNN, GRU and LSTM (sequence), the public CESNET 30pktTCNET encoder (frozen and fine-tuned), a fixed-K Transformer (our backbone without any adaptive parts).
- Reproduce three adaptive-policy baselines on top of any backbone: global max-prob threshold with per-K cascade (ECHO-style), per-class threshold + energy gate at fixed K (CAPE-Net-style), and a Q-learning stopper (FastFlow-style, simplified).
- Evaluate all baselines with the spec-004 harness at fixed K and, where applicable, adaptively.

## Non-goals

- Byte-level models (ET-BERT, YaTC, NetMamba, TrafficFormer): they require payload bytes and do not fit the privacy or Kaggle constraints; discussed in related work only.
- Exhaustive hyperparameter search; a bounded random search (20 trials) per baseline on val.

## Models

### B1 XGBoost on flow statistics

- Input: the `FLOWSTATS_DIM` = 46 flow statistics (spec 003; DataZoo's own vector has 43 and is a different vector, never mixed). At every K the statistics come from `adl_etc.data.prefix_stats.prefix_flowstats(ppi, ppi_len, K)`, which recomputes **every** column from the first K PPI entries alone and never reads the stored whole-flow vector, giving an honest "early" tabular baseline. Consequences (tested): `prefix_flowstats(30)` does **not** equal the stored `flowstats` (the PPI holds only payload packets, so `PACKETS`/`BYTES`/`DURATION` count those; only the PPI-derived columns `PPI_LEN`, `PPI_DURATION`, `PPI_ROUNDTRIPS` and the four histograms match exactly), and only `FLAG_PSH` is recoverable from the PPI, so the other five flag columns are 0. B1 is trained once per K in the spec-004 grid. Standardisation is a no-op for trees; a model that needs it must fit its own standardiser on prefix features at the same K, since the shipped `Standardizer` is fit on whole-flow statistics.
- `xgboost.XGBClassifier(tree_method="hist", n_estimators<=2000, early_stopping_rounds=50)`, class weights inverse-frequency-capped.
- CPU, local. Also the model behind the dashboard's "explain" panel (feature importances) if time permits.

### B2 1D-CNN

- **Implemented** (`adl_etc.models.baselines.cnn`, 0.445M params at 150 classes). It is trained on full flows only, so at fixed K it sees zero-padded slots it never met in training and its K=1 accuracy is at chance on real D4 (0.227 vs a 0.222 majority rate, 0.567 at K=3): a per-K evaluation of a full-flow model, and the gap prefix-aware training targets. Batch norm is masked (`MaskedBatchNorm1d`) because real flows average about 7 real packets in 30 slots.
- Input: continuous sequence [30, 4] (spec 003). Three Conv1d blocks (channels 128, 192, 256; kernels 5, 5, 3), BatchNorm, GELU, masked global average + max pooling, MLP head. About 0.4M params.
- Fixed-K evaluation by zero-padding beyond K (mask-aware pooling).

### B3 GRU / LSTM

- Input as B2. Linear stem 4 to 128, two-layer GRU (hidden 256) or LSTM; per-step logits from the hidden state so that fixed-K and adaptive evaluation come from one pass. This mirrors CAPE-Net's backbone, which makes the CAPE-style policy baseline faithful. About 0.6M params.
- Trained with multi-prefix loss (spec 008) so the comparison to our Transformer isolates the architecture, not the training recipe. The minimal form of that loss, `adl_etc.training.losses.multi_prefix_ce` (pooled masked mean of per-position cross-entropy), is built with the shared training loop; spec 008 extends it rather than replacing it.

### B4 CESNET 30pktTCNET (public pretrained encoder)

**Deferred to phase 3 (2026-09-21):** needs the `datazoo`/`cesnet-models` extras and public weights, its DataZoo-batch adapter test needs the Path A HDF5 (not on disk), and its D2 sanity check needs D2 (not acquired).


- `cesnet_models.model_30pktTCNET_256(weights=CESNET_QUIC22_Week46_Domains)`; 1.0M params, 256-d embedding; input PPI [30, 3] (IPT, DIR, SIZE) in DataZoo scaling.
- Two variants: frozen encoder + linear probe; full fine-tune. Non-causal (temporal CNN with global pooling), so fixed-K evaluation needs one pass per K on zero-padded prefixes.
- Serves as the "strong pretrained metadata-only baseline" for H1.

### B5 Fixed-K Transformer

- The spec-006 backbone trained with a single-head CE loss on full flows only (no prefix supervision, no SSL, no policy). Shows what the architecture alone gives.

### Policy baselines (run on B3, B5 and our model)

| Policy | Rule | Tuned on val |
|---|---|---|
| P-ECHO | commit at first K where max-prob >= tau (global) | tau sweep gives the Pareto front |
| P-CAPE | per-class tau_c chosen so that precision of committed flows >= 0.95 (Learn-Then-Test approximated by per-class validation quantiles; documented as an approximation), unknown if energy(K=10) >= theta | tau_c, theta |
| P-RL | DQN with actions {wait, commit} on the per-step embedding, reward −lambda per packet, +1/−1 on commit (FastFlow-style, without the synthetic-unknown branch) | lambda |

## Training recipe (shared)

AdamW, weight decay 1e-4, OneCycle LR (peak 3e-3 for CNN/RNN, 1e-3 for Transformers), batch 4096, label smoothing 0.1, AMP, 20 epochs, early stopping on val macro-F1 (patience 4), class-balanced sampling capped at 10x. Three seeds for the main table.

## Inputs and outputs

- Inputs: shards + stats (spec 003), split config (spec 004), `configs/models/baselines/*.yaml`.
- Outputs: checkpoints (`*.pt`, gitignored), saved logits per K (`.npy`), MLflow runs, reports.

## Edge cases

- 30pktTCNET expects DataZoo's PPI scaling (IPT and size normalisations and clipping); the adapter reproduces DataZoo's `ppi_transform` exactly and is unit-tested against a DataZoo-produced batch.
- B1 prefix statistics for K=1 are degenerate (duration 0, one packet): allowed; the curve starts low by construction.
- P-CAPE per-class thresholds for classes with < 20 val flows fall back to the global threshold (as CAPE-Net's "min support 20").

## Performance considerations

- B1 to B3 train on CPU in under an hour each on 3M flows; B4/B5 on Kaggle GPU in under 2 hours.
- Fixed-K evaluation of non-causal models is limited to 13 K values (spec 004) to bound cost.

## Testing

- Shape/forward tests for each model with a batch of 8.
- Overfit test: each model must be able to fit a tiny training set, run with the shared loop (`adl_etc.training.loop.fit`). **Corrected 2026-09-21:** the original wording, "> 99% train accuracy on a 512-flow subset in 200 steps", is unachievable on raw D4 for any model: real D4 is 90% duplicated (41,992 distinct PPIs in 403,394 flows, label-majority ceiling 0.9126) and even distinct flows contain near-clashes (a 512-flow sample of distinct PPIs: 234 flows with a differently-labelled neighbour within 0.5, so a small MLP reaches 0.955 after 1,600 steps and 0.975 after 3,200). The strict > 99% / 200-step check therefore runs on **synthetic separable flows** (`tests/training/util.make_flows`), where it isolates the model and loop from data ambiguity; the real-data check asserts what real D4 allows (train accuracy > 0.9 and still climbing on 512 distinct-PPI flows after 1,600 steps, `tests/training/test_real_d4.py`).
- Policy tests: P-ECHO with tau=0 commits at K=1; tau=1 never commits before K_max.
- Adapter test for B4 against a DataZoo batch (marked `slow`).

## Interactions

- Provides the comparison rows of the main table (spec 004); its policies are the baselines for spec 009 and 010; efficiency numbers go to spec 013.

## Success criteria

- All baselines evaluated on D1 test-ID and drift periods with three seeds; results logged; acc@K curves plotted together.
- B4 fine-tuned reproduces the public model's reported ballpark on D2 (sanity of the adapter).

## Open questions

- Whether to include a small Mamba-style baseline for the efficiency comparison. Default: no (CUDA kernel dependency, not needed for 30-token inputs).
