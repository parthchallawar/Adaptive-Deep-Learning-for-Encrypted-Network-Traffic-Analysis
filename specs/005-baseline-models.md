# Spec 005: Baseline Models

- **Status:** draft
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

- Input: 43 standardised flowstats (spec 003). For fixed-K evaluation, statistics are recomputed from the first K packets (`prefix_flowstats(K)`), giving an honest "early" tabular baseline.
- `xgboost.XGBClassifier(tree_method="hist", n_estimators<=2000, early_stopping_rounds=50)`, class weights inverse-frequency-capped.
- CPU, local. Also the model behind the dashboard's "explain" panel (feature importances) if time permits.

### B2 1D-CNN

- Input: continuous sequence [30, 4] (spec 003). Three Conv1d blocks (channels 128, 192, 256; kernels 5, 5, 3), BatchNorm, GELU, masked global average + max pooling, MLP head. About 0.4M params.
- Fixed-K evaluation by zero-padding beyond K (mask-aware pooling).

### B3 GRU / LSTM

- Input as B2. Linear stem 4 to 128, two-layer GRU (hidden 256) or LSTM; per-step logits from the hidden state so that fixed-K and adaptive evaluation come from one pass. This mirrors CAPE-Net's backbone, which makes the CAPE-style policy baseline faithful. About 0.6M params.
- Trained with multi-prefix loss (spec 008) so the comparison to our Transformer isolates the architecture, not the training recipe.

### B4 CESNET 30pktTCNET (public pretrained encoder)

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
- Overfit test: each model reaches > 99% train accuracy on a 512-flow subset in 200 steps.
- Policy tests: P-ECHO with tau=0 commits at K=1; tau=1 never commits before K_max.
- Adapter test for B4 against a DataZoo batch (marked `slow`).

## Interactions

- Provides the comparison rows of the main table (spec 004); its policies are the baselines for spec 009 and 010; efficiency numbers go to spec 013.

## Success criteria

- All baselines evaluated on D1 test-ID and drift periods with three seeds; results logged; acc@K curves plotted together.
- B4 fine-tuned reproduces the public model's reported ballpark on D2 (sanity of the adapter).

## Open questions

- Whether to include a small Mamba-style baseline for the efficiency comparison. Default: no (CUDA kernel dependency, not needed for 30-token inputs).
