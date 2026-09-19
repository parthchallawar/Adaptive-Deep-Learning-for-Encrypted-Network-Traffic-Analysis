# Spec 007: Self-Supervised Prefix-Predictive Pretraining

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 9 of 18
- **Depends on:** 003, 006. **Used by:** 008, 009, 010. **Research contribution:** C1 (hypotheses H1, H5).

## Problem

Labels for encrypted traffic come from SNI or from lab captures; both are scarce, noisy and go stale as applications change. Existing metadata-only pretraining (30pktTCNET, FlowCLIP) is supervised by domain names and is evaluated on full flows only. We want a pretraining signal that (a) needs no labels at all, (b) is causal so that every prefix representation is trained, (c) explicitly teaches the model what the *rest* of the flow will look like, because that is exactly the knowledge an early-stopping policy needs, and (d) builds in robustness to packet loss and timing noise.

## Goals

- Two joint objectives on unlabeled PPI: next-packet prediction (NPP) and prefix-to-flow contrastive alignment (PFC).
- Augmentation-based invariance (drop, reorder, jitter) integrated into PFC.
- A pretraining run that fits in one Kaggle session (< 10 h) on D1 train periods (about 3M flows) or D1+D2 (about 4M flows), with an option to scale to D1 size S later.
- Ablation-ready: NPP only, PFC only, masked packet modeling (MPM, non-causal teacher) as the "conventional SSL" comparison, and no pretraining.

## Non-goals

- Pretraining on payload bytes; using SNI/domain names as supervision (that is the FlowCLIP/30pktTCNET baseline, not ours).
- Foundation-scale pretraining (netFound-style); our corpus is at most 25M flows.

## Method

### Objective 1: next-packet prediction (NPP)

At each position k (1 <= k < ppi_len), predict the discretised attributes of packet k+1 from z_k:

`L_NPP = mean_k [ CE(size_bin_{k+1}) + CE(ipt_bin_{k+1}) + CE(dir_{k+1}) ]`

This is the traffic analogue of language modelling: it forces z_k to encode the flow's state (handshake phase, request/response rhythm, streaming vs. bursty). Its predictive entropy at inference is a free "how much do I know about what comes next" signal used by the stop head (spec 009). Targets are the raw next packet (un-augmented view) so that the model learns to denoise.

### Objective 2: prefix-to-flow contrastive alignment (PFC)

For a batch of flows, take two views per flow: `v1 = aug(x)` with a random crop to prefix length K ~ U{3..30} and `v2 = aug'(x)` full length. Let `p_K = proj(z_K(v1))` and `q = proj(z_{len}(v2))`. InfoNCE with in-batch negatives and temperature 0.1:

`L_PFC = −log exp(p_K·q/τ) / Σ_j exp(p_K·q_j/τ)`  (symmetrised)

This is Contrastive Predictive Coding adapted to flows: the representation of a short prefix must already identify the whole flow among thousands of others. It directly targets earliness (H1). Augmentations (`drop` p~U[0,0.2], `reorder`, `ipt_jitter`, `size_jitter`, spec 003) make the representation invariant to loss and timing noise (H5).

### Combined loss

`L = L_NPP + λ_PFC · L_PFC`, λ_PFC = 1.0 (ablated in {0, 0.5, 1, 2}).

### Comparison objective: masked packet modeling (MPM)

A bidirectional copy of the backbone (no causal mask) trained to reconstruct 30% masked packet attributes, then its weights are copied into the causal model for fine-tuning (as YaTC/NetMamba do with images and byte sequences). This isolates "causal, prefix-targeted SSL" from "any SSL".

### Corpus

- Default: D1 train periods, labels discarded (about 3M flows XS). Extended: + D2 all weeks (+ about 2.5M) for cross-protocol pretraining.
- Never any flow from D1 val/test periods (temporal leakage) even though they are unlabeled during pretraining; this is stricter than necessary but keeps the drift study clean.

### Training

AdamW (lr 1e-3 peak, OneCycle, wd 0.05), batch 4096 flows (two views = 8192 forward sequences), AMP fp16, 8 to 10 epochs, gradient clipping 1.0. Checkpoint each epoch; resume support (spec 015). Expected: about 15 to 25 min per epoch on a T4 for 3M flows.

## Inputs and outputs

- Inputs: unlabeled shards (spec 003); `configs/pretrain/*.yaml`.
- Outputs: `pat_ssl_<variant>_<seed>.pt` (backbone + `next` + `proj` heads), pretraining curves, and probe metrics (below) in MLflow.

## Evaluation of the pretraining itself (before fine-tuning)

- Linear probe on frozen z_K at K in {3, 5, 8, 30} with 10% labels; kNN (k=20) accuracy on `proj`.
- Alignment/uniformity statistics of `proj` (to detect collapse; FlowCLIP-style degenerate geometry is a known failure mode).
- NPP perplexity on val as a drift indicator (spec 012 reuses it).

## Edge cases

- Flows with `ppi_len <= 2`: contribute to NPP only where a next packet exists; PFC crop requires K <= ppi_len − 1, otherwise the flow is skipped for PFC in that batch.
- Duplicate flows (identical PPI) in a batch make false negatives in InfoNCE; tolerated (rare at 4096 batch), measured once.
- Drop augmentation shortens flows; PFC crop is applied after augmentation.
- Collapse (uniform `proj`): detected by the uniformity metric; remedy is lower temperature or higher λ_PFC.

## Performance considerations

- Two views double compute; use one augmented view for NPP as well (shared forward pass) to avoid a third pass.
- Data lives in RAM as int16; augmentations run on GPU in integer space to keep the input pipeline from bottlenecking.

## Testing

- Loss unit tests: NPP ignores padded targets; PFC positive on the diagonal; symmetric loss equals mean of two directions.
- Small overfit test (1k flows, 300 steps): NPP loss falls below a threshold; kNN probe > chance.
- Determinism test with fixed seed.

## Interactions

- Produces initial weights for spec 008; `next` head entropy is a feature for spec 009; `proj` is used in spec 010; perplexity feeds spec 012.

## Success criteria (H1/H5 acceptance)

- After fine-tuning (spec 008), SSL-initialised PAT beats scratch PAT by >= 3 points acc@K for K <= 5 on D1 test-ID and by >= 5 points with 10% labels; mean packets-to-decision at equal accuracy reduced by >= 10%.
- Under 10% packet drop at test time, accuracy drop is at least halved relative to scratch training.
- If these fail, the negative result is still reportable; the ablation table must be complete either way.

## Open questions

- Whether to pretrain on D1 size S (25M) for the final run if GPU hours remain. Default: no; keep XS.
- Temperature and crop distribution (uniform vs. biased to short prefixes) to be tuned on val probes.
