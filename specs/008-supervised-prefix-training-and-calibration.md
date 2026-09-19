# Spec 008: Supervised Prefix Training and Calibration

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 10 of 18
- **Depends on:** 003, 004, 006, 007. **Used by:** 009, 010, 011.

## Problem

A model trained only on complete 30-packet flows is poorly calibrated on short prefixes: it has never been asked to commit at K=4, so its confidence there is meaningless and a stopping rule built on it fails. Training must supervise every prefix, keep the outputs comparable across K, handle 180 heavy-tailed classes, and end with probabilities that are calibrated *per prefix length*, because the policy and the controller consume those probabilities as if they were true error rates.

## Goals

- Multi-prefix supervised fine-tuning of the PAT backbone (from SSL or scratch) with all heads trained jointly.
- Per-K temperature calibration and per-K commit-safety calibration on the validation period.
- Class imbalance handling that does not distort test-time behaviour.
- Low-label regimes (1%, 10%, 100%) as first-class configs.

## Non-goals

- Training the unknown detector (spec 010) or tuning thresholds (spec 009/011); this spec produces the calibrated scores they consume.

## Method

### Losses (per flow, averaged over valid positions k <= ppi_len)

1. Prefix classification: `L_cls = Σ_k w_k · CE(cls_k, y)` with `w_k = min(1, k/8)` (ramping so that K=1..7 count less, K>=8 fully), label smoothing 0.1.
2. Self-distillation from the full prefix: `L_sd = Σ_k w_k · KL(softmax(cls_len/T) || softmax(cls_k/T))`, T = 2, weight 0.2 (same device as CAPE-Net's exit distillation; used here for prefix consistency).
3. Commit-safety: `L_safe = Σ_k BCE(safe_k, 1[argmax cls_k == y])` computed on *detached* logits so that the safety head does not push the classifier. Positive/negative reweighting by prefix so that K=1 (mostly wrong) does not dominate.
4. Keep SSL heads alive with small weight (`L_NPP`, 0.1) to preserve the `next` entropy feature and to allow spec 012's perplexity monitor to remain meaningful after fine-tuning.

`L = L_cls + 0.2·L_sd + 1.0·L_safe + 0.1·L_NPP`

### Data and imbalance

- Class-balanced sampler with per-class cap of 10x the smallest class's sampling weight (square-root balancing); test evaluation always uses the natural distribution.
- Prefix crop augmentation is not needed (all positions are supervised in one pass) but `drop`/`ipt_jitter` at low intensity (p=0.05, s=0.1) are on by default for robustness; ablated.

### Optimisation

AdamW, wd 1e-4, OneCycle LR peak 5e-4 (SSL init) or 1e-3 (scratch), batch 4096, AMP, 15 epochs, early stopping on val `AUC_K` (not on acc@30, since earliness is the objective), patience 3, EMA of weights (0.999) for evaluation.

### Layer-wise LR decay for SSL-initialised runs

0.85 per block from the top, to preserve pretrained low-level features under low-label regimes.

### Calibration (on val period only)

- Temperature per K: `T_K = argmin NLL(softmax(cls_K / T_K))`, 30 scalars stored with the checkpoint.
- Commit-safety calibration: isotonic regression per K mapping `sigmoid(safe_K)` to observed correctness rate on val; stored as 30 monotone lookup tables. After this, `p_safe(K) ≈ P(correct | commit at K)`, which is exactly what the controller's budget/accuracy trade-off needs.
- Reliability diagrams and ECE per K are logged (spec 004).

## Inputs and outputs

- Inputs: shards + stats, split config, an SSL checkpoint or none, `configs/finetune/*.yaml` (label fraction, augmentation, loss weights).
- Outputs: `pat_ft_<variant>_<seed>.pt` including calibration tables; saved logits per K on val/test/drift periods; MLflow run.

## Edge cases

- Classes with zero training flows in a low-label regime: removed from the known set for that run and recorded.
- Very short flows (ppi_len <= 2): supervised at their positions only; they dominate "short-flow accuracy", reported separately.
- Isotonic tables with few val samples at K=1..2 (most flows are wrong there): fall back to a Platt (logistic) fit when a bin has < 200 samples.
- Label noise from SNI-based labelling: label smoothing plus the reject option in evaluation; no cleaning.

## Performance considerations

- One forward pass supervises all 30 prefixes: the training cost equals a plain classifier's.
- Saved logits for 1M flows x 30 K x 180 classes in fp16 = 10.8 GB; store top-10 logits plus the energy score and the projection instead (about 1.5 GB) and recompute metrics from those; full logits only for val (needed for calibration).

## Testing

- Loss masking tests; detachment test (gradient of `L_safe` w.r.t. `cls` params is zero).
- Calibration tests: temperature scaling reduces val NLL; isotonic output is monotone and within [0, 1].
- Regression test: fine-tuning from the SSL checkpoint reaches >= the scratch acc@30 on a 100k-flow subset.

## Interactions

- Consumes spec 007 weights; produces the calibrated probabilities used by 009, 010, 011; its saved logits feed spec 004 reports and spec 013's offline policy replay.

## Success criteria

- ECE at K in {3, 8, 30} below 0.03 on val after calibration.
- `AUC_K` on D1 test-ID above every baseline in spec 005 (with the same backbone size class).

## Open questions

- Ramp shape `w_k`; alternatives (uniform, linear) to be ablated once.
- Whether EMA weights or last weights are used for the streaming export (default: EMA).
