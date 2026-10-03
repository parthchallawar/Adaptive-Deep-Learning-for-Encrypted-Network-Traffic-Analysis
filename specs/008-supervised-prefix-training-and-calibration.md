# Spec 008: Supervised Prefix Training and Calibration

- **Status:** draft, revised 2026-10-03 before build (see [Revisions](#revisions)). Not implemented. Plan: [phase 3](../plans/phase-3-pat-ssl-prefix-training.md), tasks T6 and T7.
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 10 of 18
- **Depends on:** 003, 004, 006, 007. **Used by:** 009, 010, 011.

## Problem

A model trained only on complete 30-packet flows is poorly calibrated on short prefixes: it has never been asked to commit at K=4, so its confidence there is meaningless and a stopping rule built on it fails. Training must supervise every prefix, keep the outputs comparable across K, handle 150 heavy-tailed known classes, and end with probabilities that are calibrated *per prefix length* and *for the real class distribution*, because the policy and the controller consume those probabilities as if they were true error rates.

## Goals

- Multi-prefix supervised fine-tuning of the PAT backbone (from SSL or scratch) with all heads trained jointly.
- Per-K temperature calibration and per-K commit-safety calibration on the validation period, after correcting for the training sampler's class prior.
- Class imbalance handling that does not distort test-time behaviour.
- Low-label regimes (1%, 10%, 100%) as first-class configs, each trained long enough to converge.

## Non-goals

- Training the unknown detector (spec 010) or tuning thresholds (spec 009/011); this spec produces the calibrated scores they consume.

## Method

### Losses (per flow, averaged over valid positions k <= ppi_len)

1. Prefix classification: `L_cls = Σ_k w_k · CE(cls_k, y)` with label smoothing `ε`. Defaults: `w_k = min(1, k/8)` and `ε = 0.1`; both are decided on val before the main grid (see [Design choices](#design-choices-decided-on-val-before-the-main-grid)). Measured on the D1 probe day, the ramp gives K=1..5 weights of 0.125 to 0.625 (exactly where H1 is judged) and gives 44% of all supervised positions less than full weight.
2. Self-distillation from the full prefix: `L_sd = Σ_k w_k · KL(softmax(stopgrad(cls_len)/T) || softmax(cls_k/T))`, T = 2, weight 0.2. **The teacher `cls_len` (the last real position) is detached**, so distillation pulls early predictions toward the full-flow one and never the other way.
3. Commit-safety: `L_safe = Σ_k BCE(safe_k, 1[argmax cls_k == y])` computed on *detached* logits so that the safety head does not push the classifier. Positive/negative reweighting by prefix so that K=1 (mostly wrong) does not dominate.
4. Keep SSL heads alive with small weight (`L_NPP` including its `end` term, 0.1) to preserve the `next` entropy and `p_end` features and to keep spec 012's perplexity monitor meaningful after fine-tuning.

`L = L_cls + 0.2·L_sd + 1.0·L_safe + 0.1·L_NPP`

### Data and imbalance

- Class-balanced sampling through the shared loop's `balanced_weights` (each class sampled up to 10x its natural rate), as every baseline uses. Test evaluation always uses the natural distribution.
- **Logit adjustment.** The sampler changes the class prior the model learns, so its raw probabilities describe a world where rare apps are common. Per-K temperature scaling cannot fix that: it only sharpens or flattens scores and never shifts them between classes. Before calibration, logits are corrected post hoc (Menon et al., 2021, "Long-tail learning via logit adjustment"):
  `cls_adj(c) = cls(c) + log π_nat(c) − log π_samp(c)`,
  where `π_samp` is the effective class prior under the sampler, computed exactly from `balanced_weights` and the train counts, and `π_nat` is the train period's natural prior. Calibration, the energy score and the stopping policies consume the adjusted logits. Macro-F1 is reported for both the adjusted and the unadjusted logits, since balanced decisions favour macro-F1 and adjusted ones favour accuracy.
- Prefix crop augmentation is not needed (all positions are supervised in one pass) but `drop`/`ipt_jitter` at low intensity (p=0.05, s=0.1) are on by default for robustness; ablated.

### Optimisation

AdamW, wd 1e-4, OneCycle LR peak 5e-4 (SSL init) or 1e-3 (scratch), AMP, early stopping on val `AUC_K` (not on acc@30, since earliness is the objective), patience 3 validations.

- **EMA of weights (0.999) with warm-up**: decay at step t is `min(0.999, (1 + t) / (10 + t))`, so the average is not dominated by the untrained starting weights in short runs. EMA weights are used for evaluation and for the streaming export.
- **Dropout** 0.1 by default; {0, 0.1} decided on val at 100% labels (Design choices). With about 0.9M parameters and roughly 96M training packets, underfitting is more likely than overfitting at 100% labels. The 10% and 1% regimes keep 0.1.

| Regime | Labelled flows | Batch | Length | Validation |
|---|---|---|---|---|
| 100% | 5,989,515 | 4096 | 15 epochs (~1,460 steps each) | every epoch |
| 10% | ~599k | 1024 | max(15 epochs, 3,000 steps) | every 500 steps or every epoch, whichever is longer |
| 1% | ~60k | 1024 | max(15 epochs, 3,000 steps) | every 500 steps |

Reason: at batch 4096 for 15 epochs, the 1% regime would get only **about 220 optimizer steps**, and an EMA with decay 0.999 would still hold 80% of the untrained starting weights at the end. The step floor and the smaller batch give every regime at least 3,000 updates; the extra compute is small (about 3M samples, half of one 100% epoch).

### Layer-wise LR decay for SSL-initialised runs

0.85 per block from the top, to preserve pretrained low-level features under low-label regimes.

### Low-label regimes

The labelled subset is a seeded uniform draw from the train period, recorded in the run's params. **The label space stays fixed at the 150 known classes in every regime**: a class with no labelled flows is never predicted and its test flows count as errors. Removing it instead would make the 1%, 10% and 100% numbers measure different tasks.

### Calibration (on the validation period only)

Fitted on a seeded 250k-flow sample of val (phase 2's `EvalArrays` cap), in this order:

1. Logit adjustment (above).
2. Temperature per K: `T_K = argmin NLL(softmax(cls_adj_K / T_K))`, 30 scalars.
3. Commit-safety calibration: isotonic regression per K mapping `sigmoid(safe_K)` to the observed correctness rate on val, as 30 monotone lookup tables. After this, `p_safe(K) ≈ P(correct | commit at K)`, which is exactly what the controller's budget/accuracy trade-off needs. Isotonic regression is implemented with pool-adjacent-violators in NumPy (`evaluation/calibration.py`), because `evaluation/` must not import scikit-learn.

Everything is saved as `calibration.json` next to the checkpoint, holding the checkpoint's hash; a mismatched pair raises. Reliability diagrams and ECE per K are reported on **test_id**, the data calibration never saw. Val ECE is logged only as a fit diagnostic.

## Design choices decided on val before the main grid

Four choices change results enough to matter and cannot be settled by argument. Each is tested once, one factor at a time from the default, on the scratch PAT at 100% labels, single seed, before the {scratch, SSL} × {100, 10, 1}% grid is queued:

| Choice | Default | Alternative | Decided by (on val) |
|---|---|---|---|
| Loss ramp `w_k` | `min(1, k/8)` | uniform (`w_k = 1`) | `AUC_K` |
| Label smoothing `ε` | 0.1 | 0 | P-ECHO accuracy at mean K ∈ {4, 6, 8} (spec 000's H2 metric). Published work reports that label smoothing can hurt selective classification, which is what the stopping policy does |
| Dropout (100% labels) | 0.1 | 0 | `AUC_K` |
| Side channel (spec 006) | on | off | `AUC_K` |

A difference smaller than 3x the seed-to-seed standard deviation (measured on B5's three seeds) is a tie, and a tie keeps the default. The chosen settings and every value behind them are recorded in the plan before the grid runs; test_id is not read for any of these choices.

## Inputs and outputs

- Inputs: shards + stats, split config (`d1_open.yaml`, 150 known classes), an SSL checkpoint or none, `configs/finetune/*.yaml` (label fraction, augmentation, loss weights).
- Outputs: `pat_ft_<variant>_<seed>.pt` plus `calibration.json`; saved per-K logits through `EvalArrays` (250k-flow cap per split) on val, test_id and test_drift; an MLflow run.

## Edge cases

- Classes with zero training flows in a low-label regime: stay in the label space and count as errors (see Low-label regimes).
- Very short flows (ppi_len <= 2): supervised at their positions only. They are rare on D1 (0.0% of probe-day flows; 3.0% have ppi_len <= 5) and are reported separately as "short-flow accuracy".
- Isotonic tables with few val samples at a K (fewer than 200): fall back to a Platt (logistic) fit for that K.
- Label noise from SNI-based labelling: label smoothing (if kept by the design choice) plus the reject option in evaluation; no cleaning.

## Performance considerations

- One forward pass supervises all 30 prefixes: the training cost equals a plain classifier's.
- Saved logits use phase 2's 250k-flow `EvalArrays` cap: 250k × 30 × 150 × 2 bytes ≈ 2.25 GB per split per run, which bounds storage without compressing the logits.

## Testing

- Loss masking tests. Detachment tests: the gradient of `L_safe` with respect to `cls` parameters is exactly zero, and `L_sd` sends no gradient into the teacher position's logits.
- EMA warm-up: after t steps from a known start, the EMA equals the closed-form warm-up average.
- Logit adjustment: with a uniform sampler it is a no-op; with a known sampler it shifts each class's logit by exactly `log π_nat − log π_samp`.
- Low-label step rule: a 1% config runs at least 3,000 optimizer steps; the label space is identical across regimes.
- Calibration tests: temperature scaling never raises val NLL; isotonic output is monotone and within [0, 1]; PAV matches scikit-learn's `IsotonicRegression` on random data (test skipped without sklearn); the Platt fallback triggers below 200 samples; a calibration file paired with the wrong checkpoint raises.
- Regression test: fine-tuning from the SSL checkpoint reaches >= the scratch acc@30 on a 100k-flow subset.

## Interactions

- Consumes spec 007 weights; produces the calibrated probabilities used by 009, 010, 011; its saved logits feed spec 004 reports and spec 013's offline policy replay.

## Success criteria

- ECE at K in {3, 8, 30} below 0.03 on **test_id** after calibration.
- `AUC_K` on D1 test_id above every baseline in spec 005 (the CNN and the GRU, same open-set split, same evaluation cap).

## Open questions

- Whether EMA weights or last weights are used for the streaming export (default: EMA).

## Revisions

**2026-10-03, before build** (plan phase 3, findings F7, F8, F9, F11, F13, F14; measured on the D1 probe day and from the real train-split size):

- **Logit adjustment** added before calibration. Class-balanced sampling shifts the learned prior, and temperature scaling cannot shift it back.
- **Low-label training length**: batch 1024 and a 3,000-step floor for the 10% and 1% regimes, plus **EMA warm-up**. As originally written, the 1% regime got about 220 steps and an EMA still 80% made of initial weights.
- **Distillation teacher detached** (the original text did not say).
- **Design choices decided on val before the grid**: loss ramp, label smoothing, dropout and the side channel, one factor at a time, ties keep the default.
- ECE judged on **test_id**, not on the val data calibration is fitted on.
- The label space stays fixed across label regimes; zero-label classes count as errors instead of being removed.
- Saved logits use the 250k `EvalArrays` cap instead of top-10 compression; calibration is fitted on a seeded 250k val sample.
- Isotonic regression in NumPy (PAV), keeping scikit-learn out of `evaluation/`.
- Class balancing stated as the shared loop's `balanced_weights`, so the PAT and the baselines use one sampler.
