# Plan: Phase 3, the Prefix-Aware Transformer, SSL pretraining and prefix training

- **Specs:** [006](../specs/006-prefix-aware-transformer-backbone.md), [007](../specs/007-self-supervised-pretraining.md), [008](../specs/008-supervised-prefix-training-and-calibration.md) (build steps 8 to 10). Also inherits B5 and B4 from [005](../specs/005-baseline-models.md), and phase 2's three open exit criteria.
- **Status:** not started. Plan written 2026-10-03.
- **Entry gate:** phase 2's [exit criteria](phase-2-tracking-kaggle-baselines.md#exit-criteria) are 6 of 9 met. This plan starts anyway and closes the other three itself: criterion 4 in [T0](#t0--close-phase-2s-leftovers), criterion 7 in [T2](#t2--cross-session-resume-on-kaggle-closes-phase-2-criterion-7) and criterion 8 in [T4](#t4--the-pat-backbone-and-b5-spec-006). None of them blocks building the PAT, and all three are cheaper to close alongside phase-3 work than before it.
- **Exit gate:** phase 4 (spec 010, then 009) starts when [Exit criteria](#exit-criteria) are all green.
- **Before starting:** `phase2` is 30+ commits ahead of `main` and has not been merged. Merge it (or branch `phase3` from it) first. This is the user's call.

## Approach

Phase 3 is where the project stops building infrastructure and starts testing its own claims. H1 (prefix-predictive SSL helps early accuracy) and H5 (augmented SSL helps robustness) are decided here. Specs 009 to 011 (phase 4) only consume what this phase produces: calibrated per-K probabilities, a `safe` head, a `next`-entropy signal and a `proj` embedding.

Four rules hold throughout:

1. **One training loop, extended, not forked.** `loop.fit` is the reason a baseline-vs-PAT comparison isolates the architecture and not the recipe. The PAT's extra losses go in through a small objective interface ([T3](#t3--generalise-the-training-loop-without-changing-any-baseline)), and a regression test shows every existing baseline trains bit-identically before and after.
2. **The comparison has to be fair, or the result does not count.** The PAT is compared against the *strongest* baseline at each K (F2), on the same open-set label space (F5), with the same label fractions and the same evaluation cap. Every phase-3 headline number comes from `make_tables.py`, never typed in by hand (phase 2's rule 4).
3. **A negative result is a result.** Spec 007 says this itself: if SSL does not beat scratch by the stated margins, the ablation table still has to be complete. Nothing gets tuned *after* seeing test numbers. Hyperparameters are chosen on val, and test_id is read once per configuration.
4. **Every model is proven locally before it gets a GPU-hour** (phase 2 rule 2, unchanged). The local RTX 3050 handles synthetic overfit tests and small real D3/D4 runs. Kaggle gets only runs that have already passed the smoke test.

## Findings that change the plan

Each of these was checked against the code or the real results rather than taken from the specs. Each changes a task below.

### F1 — `loop.fit` only knows one loss

`fit` calls `_loss(logits, batch, smoothing)`, which is either `cross_entropy_ls` or `multi_prefix_ce` ([loop.py:124-127](../src/adl_etc/training/loop.py#L124-L127)). Early stopping is hard-wired to val macro-F1, and the model must return a logits tensor. The PAT returns four heads, SSL has no labels at all, spec 008 early-stops on val `AUC_K` (not macro-F1), and spec 008 also wants EMA weights and layer-wise LR decay. None of that fits through the current signature. A second loop would break rule 1, so [T3](#t3--generalise-the-training-loop-without-changing-any-baseline) puts an `Objective` between the model and the loop, and makes the baselines' current behaviour the default objective.

### F2 — spec 006's success bar names the wrong baseline

Spec 006 says B5 should be "within 1 point of the **GRU** baseline at K=30". But the real tracked results ([results/summaries/main_table.md](../results/summaries/main_table.md)) put the CNN at **0.8651** val macro-F1 and the GRU at **0.8124**. Under the spec as written, the PAT could be 4 points worse than the best baseline at K=30 and still pass. The bar becomes the strongest baseline at each K: the CNN at full flows, and whichever wins at small K. This also matters for spec 008's own criterion ("`AUC_K` above every baseline"), which is right as written and stays.

**Refined 2026-10-03: compare like with like.** The CNN trains on full flows only; the GRU trains on every prefix, which costs it some full-flow accuracy, so part of the 5-point gap is the recipe, not the architecture. Spec 006's criteria now pair them: B5 (full-flow) against the CNN at K=30, and the prefix-trained PAT against both across all K (`AUC_K`, and accuracy at every K <= 8).

### F3 — the SSL corpus is twice what spec 007 sized, and the GPU is not the bottleneck

Spec 007 assumes "about 3M flows XS" and "15 to 25 min per epoch". The real D1 train split is **5,989,515 flows** (3% sample, weeks 11-26). B3 measured **~43,000 flows/s** on one T4. If the PAT runs near that speed per view, a two-view SSL epoch is about 5 minutes, so 10 epochs take under an hour. That is far below the spec's estimate and far below the 12 h session cap. The weekly 30 GPU-hours is the binding constraint, not session length. The budget table below is still an estimate until T4 measures the PAT for real.

### F4 — the augmentations are per-flow Python functions

`features.drop`, `reorder`, `ipt_jitter` and `size_jitter` each take **one flow** (`ppi_len: int`) ([features.py:386-466](../src/adl_etc/data/features.py#L386-L466)). SSL needs two augmented views of ~6M flows every epoch, which is about 12M Python-level calls per epoch, per augmentation. At even 10 µs each, that is minutes of CPU per epoch and would starve the GPU. [T5](#t5--ssl-pretraining-spec-007) adds batched (vectorised NumPy) versions. The per-flow functions stay as the reference implementation, and a test checks that the batched versions match them statistically. This is the same pattern as the old `logits_at_k`. GPU-side augmentation, which spec 007 suggests, is built only if T4's measurement shows the batched CPU path is still too slow.

### F5 — every real run so far is closed-set, and spec 010 needs open-set checkpoints

T8 trained on all 180 classes by scope decision. Spec 004's 150-known/30-unknown draw exists (`evaluation/unknown_split.py`) but no split file uses it, and training cannot *filter* held-out classes yet (`FlowBatches(train=True)` refuses them rather than dropping them). If the PAT were fine-tuned closed-set here, phase 4's unknown detector (spec 010) would need every checkpoint retrained. [T1](#t1--the-open-set-d1-split-and-training-time-filtering) builds the open-set split first. Every phase-3 headline run uses it, including re-runs of the baselines. B3's three seeds cost 2.3 GPU-hours, which is cheap next to a retrained PAT grid.

### F6 — Kaggle cannot resume across pushes yet

T8 found this for real: each `kaggle kernels push` starts from an empty `/kaggle/working`, so `state.json` from a previous session is invisible and seed 0 was retrained rather than resumed. The pause/resume mechanism works *within* a session, but a run paused by the time guard can never continue, and `run_queue.yaml` has to be hand-edited after every push. Phase 3 queues three to four times as many runs as phase 2, so this has to work. [T2](#t2--cross-session-resume-on-kaggle-closes-phase-2-criterion-7) fixes it.

### F7 — spec 008 grades calibration on the data it was fitted on

Spec 008's criterion is "ECE at K in {3, 8, 30} below 0.03 on **val** after calibration", but temperature and isotonic tables are *fitted* on val. That number is optimistic by construction. ECE is reported on **test_id**, and val ECE is logged only as a fit diagnostic.

### F8 — spec 008's low-label rule makes the regimes incomparable

Spec 008 says classes with zero training flows in a low-label regime are "removed from the known set for that run". At 1% labels (~60k flows) with 150 heavy-tailed classes, that changes the label space between regimes, so the 1%, 10% and 100% numbers would be measured on different tasks. The label space stays fixed at the 150 known classes. A class with no labelled flows is never predicted, and its test flows count as errors, which is the honest cost of having few labels. The label subset is a seeded uniform draw, recorded in the run's params.

### F9 — isotonic regression would pull scikit-learn into `evaluation/`

CLAUDE.md forbids scikit-learn in `evaluation/`, but calibration must be importable wherever reports are built. Pool-adjacent-violators is about 20 lines of NumPy, so `evaluation/calibration.py` implements it directly and is tested against scikit-learn's `IsotonicRegression` in a test that skips when sklearn is missing.

F10 to F14 were found on 2026-10-03 by measuring the D1 probe day (487,081 real flows, `data/processed/cesnet-tls-year22-probe`, same format as the training export; one holiday, so exact values may shift on working weeks, but the pattern will not). Each was applied to specs 006 to 008 directly, since those specs are unbuilt designs (see the corrections table).

### F10 — the size tokens throw away most of the size information the baselines use

D1 flows have **1,460 distinct exact payload sizes**, which the 64 log-spaced size bins collapse into **53**. The bin around 1,400 bytes is **162 bytes wide** (1,338 to 1,500), so a 1,340-byte packet and a full 1,500-byte packet are the same token. Exact TLS record sizes are among the strongest app fingerprints, and the CNN and GRU baselines read them exactly through the continuous view. A token-only PAT would start with less information than the models it must beat. **The side channel is now on by default** (spec 006), carrying the existing standardised continuous view (`log1p(size)·dir`, `log1p(ipt)`, `dir`, `push`); off is an ablation.

Also measured: mean `ppi_len` **16**, median **13** (CLAUDE.md's "~7 real packets" is D4, not D1); 9.5% of flows have <= 7 packets; 16.7% are truncated at 30.

### F11 — the 1% regime would barely train, and its EMA would mostly be the starting weights

1% of 5,989,515 is ~59,895 flows. At spec 008's batch 4096 for 15 epochs that is **about 220 optimizer steps**, and an EMA with decay 0.999 keeps `0.999^220 ≈ 80%` of its initial weights. The "1% labels" result would measure an untrained average, not a low-label model. Spec 008 now uses batch 1024 and a 3,000-step floor for the 10% and 1% regimes, plus EMA warm-up (`min(0.999, (1+t)/(10+t))`).

### F12 — identical flows are common, so contrastive pretraining has false negatives

26.6% of flows share their exact token sequence with another flow. In a random 4,096 batch, **5.6% of anchors have an identical twin**, which InfoNCE treats as a negative and asks the model to separate, an impossible target that adds noise. Spec 007 assumed this was rare. **Twin masking** (hash each flow's tokens, drop equal-hash pairs from the negatives) is now part of PFC.

### F13 — class-balanced sampling shifts the prior, and temperature scaling cannot shift it back

The shared sampler oversamples rare classes up to 10x, so the model's softmax describes a world where rare apps are common, while the policy and controller read the probabilities as real error rates on the natural distribution. Per-K temperature only sharpens or flattens scores. **Post-hoc logit adjustment** (`+ log π_nat − log π_samp`, Menon et al. 2021) is now applied before calibration (spec 008), with macro-F1 reported for both adjusted and unadjusted logits.

### F14 — smaller gaps in the spec text

- No **final LayerNorm** after the last pre-LN block (spec 006): added.
- The self-distillation **teacher was not detached** (spec 008): now `stopgrad(cls_len)`.
- **FFN ratio 2** (256 for `d_model` 128) where 4 is standard: default now 512 (about 0.9M parameters, under the 2M cap); 256 stays in T9's search.
- **An `end` target** ("this is the last packet") added to the `next` head: 83% of D1 flows end before packet 30, and the stopping policy needs to know when waiting cannot bring more evidence. `p_end` feeds `safe`.
- **Four choices that cannot be settled by argument** (loss ramp, label smoothing, dropout, side channel) are now decided on val before the main grid, one factor at a time ([T6](#t6--supervised-prefix-fine-tuning-spec-008)). The ramp gives K=1..5 weights 0.125 to 0.625, exactly where H1 is judged, and 44% of supervised positions get less than full weight; published work reports label smoothing can hurt selective classification.

## Scope decisions

Made once, here, so they are not relitigated mid-task.

| Decision | Choice | Reason |
|---|---|---|
| **Main split** | Open-set D1 (150 known / 30 unknown, spec 004's draw), new file `configs/splits/d1_open.yaml` | F5. `d1_main.yaml` stays as the closed-set record of phase 2's numbers and is not edited. |
| **Phase 2 criterion 4's "on D4" half** | Retired, with the reason recorded in the phase-2 plan | `configs/splits/d4_anomaly.yaml` contributes no train split by design: D4 is the anomaly set (spec 004). 90% of its flows are duplicates (ceiling 0.9126), so a three-seed baseline *trained* on it measures nothing the project uses. Recorded, not silently dropped. The user can overrule this. |
| **Depth exits (spec 006, optional)** | Deferred to phase 4, with spec 011 | Only the budget controller consumes them, and spec 006 itself calls their lazy KV-cache fill "the one piece of real complexity". Building them before their consumer exists repeats phase 2's B5 lesson. The `PATStream` KV-cache *is* built here, because stream-vs-full equivalence is the strongest causality test there is. |
| **Streaming latency measurement** | Deferred to spec 013 (phase 5) | Spec 006's latency target is verified by spec 013. Here only correctness of `PATStream` is tested. |
| **P-RL policy baseline** | Deferred to phase 4, with spec 009 | Its only consumer is spec 009's comparison (phase 2 already deferred it on the same reasoning). |
| **B4 (30pktTCNET)** | Optional, time-boxed to 2 days, last task | It is the "strong pretrained baseline" for H1, so it matters here. But it still needs the `datazoo` extras and public weights, and its D2 sanity check needs D2, which is still not acquired. If it stalls, the H1 write-up says plainly that it was compared against scratch training and MPM only. |
| **Continuous side channel** | **On by default**, using the existing standardised `continuous` view, not spec 006's original `log1p(x)/7.3` constants | One normalisation definition in the project, not two. On by default because of F10; off is one of T6's design ablations. |
| **Hyperparameter tuning** | Equal small budget for B2, B3 and the scratch PAT (T9), or none for anyone | A tuned PAT against default baselines is the classic unfair comparison. If T9 is skipped, all models use their spec defaults and the write-up says so. |
| **Saved logits** | Phase 2's 250k-flow `EvalArrays` cap, not spec 008's top-10 compression | The cap already bounds storage (2.25 GB per run per split). Calibration is fitted on a seeded 250k sample of val, which is ample for 30 temperatures and 30 isotonic tables. |
| **SSL corpus** | D1 train weeks 11-26 only; D2 not used | D2 is not acquired. Spec 007's "+D2 extended corpus" stays an option for later. |

## Tasks

### T0 — Close phase 2's leftovers

**Files:** `kernel/run_queue.yaml`, `plans/phase-2-tracking-kaggle-baselines.md`, `results/summaries/`

- **B1 (XGBoost) is removed** (owner decision, 2026-10-03; done before this plan started, see spec 005). Its real D1 run never completed and was cancelled on Kaggle, so phase 2's criterion 4 no longer waits on it.
- Record the D4 scope decision above in the phase-2 plan's criterion 4.
- **Done when:** `main_table.md` has B2 and B3 at three seeds each (already true), and phase 2's criterion 4 reads "met for B2/B3 on D1; B1 removed; D4 half retired, reasons recorded".

### T1 — The open-set D1 split and training-time filtering

Two parts with different deadlines (owner decision, 2026-10-03: Transformer work goes first, and the baseline retraining waits for spare quota).

**T1a, the code. Deadline: before the first PAT run on Kaggle (T4's B5 seeds).** About a day of local work.

**Files:** `configs/splits/d1_open.yaml`, `src/adl_etc/evaluation/protocol.py`, `src/adl_etc/training/{datasets,labels,run}.py`, `scripts/make_unknown_split.py`, tests

- Run `make_unknown_split.py` once on real D1 class counts (support floor already applied). Commit the drawn partition into `d1_open.yaml` as an `unknown_classes:` list, with the seed and the draw's hash.
- `run_training` builds `LabelSpace.create(known, unknown)` from the split, and **drops** unknown-class flows from the train set before `FlowBatches(train=True)` sees them. Val and test keep them, labelled unknown, so spec 010 has real unknowns to score later. The existing refusal stays as a tripwire behind the filter.
- Add the low-label hook (F8): `data.label_fraction` plus a seeded uniform subsample of *train* only. The label space does not change.
- **Tests:** no unknown class reaches a training batch; val/test keep them; `load_split` still raises on unknowns in train when the YAML is wrong; the label fraction is deterministic per seed and leaves the label space unchanged; `LabelSpace` hash differs between `d1_main` and `d1_open`, so a checkpoint cannot resume across them.
- **Done when:** the tests pass and a local smoke run trains on `d1_open` with 150 known classes.

**T1b, the baseline re-runs. Deadline: before T8 (the comparison), in any Kaggle week with spare quota.**

- Re-run B2 and B3 (three seeds each) on `d1_open.yaml` through the Kaggle queue. No code changes, only the queue entries. About 5 GPU-hours.
- **Done when:** B2 and B3 have three tracked seeds on `d1_open` and the table shows them as a separate group from the closed-set ones.

### T2 — Cross-session resume on Kaggle (closes phase 2 criterion 7)

**Files:** `kernel/kernel.py`, `kernel/kernel-metadata.json`, `scripts/kaggle_sync.sh`, `docs/kaggle-workflow.md`, `tests/test_kernel_smoke.py`

- At startup, before the queue runs, copy the previous session's run directories into `/kaggle/working` when they exist. First choice: the kernel lists *its own* earlier output under `kernel_sources`, which T8 showed mounts the *last successful* run. Whether Kaggle allows a kernel to reference itself must be checked on Kaggle, not assumed. Fallback: `kaggle_sync.sh push-resume` pulls the paused run directories and versions them into a small private dataset that the kernel mounts.
- Once that works, `is_finished` skips completed entries across pushes, and `run_queue.yaml` stops being hand-pruned.
- **Real proof:** push a short run with an artificially short deadline (an env override of the 11.5 h guard, e.g. 10 minutes) so it exits `PAUSED` mid-training. Push again, and check that it resumes from the saved epoch and finishes with the same metrics a never-paused local run gives.
- **Tests:** the copy-in step with a synthetic previous output; a finished entry in it is skipped; a paused one resumes; a corrupt or partial `state.json` from a crashed session refuses loudly rather than training from scratch over it.
- **Done when:** one real Kaggle run has been paused by the guard in one session and completed in the next.

### T3 — Generalise the training loop without changing any baseline

**Files:** `src/adl_etc/training/{loop,objectives,losses,ema}.py`, tests

- `Objective` protocol: `loss(outputs, batch) -> (loss, logs)`, plus `val_score(...)` and `higher_is_better`. The current behaviour becomes `ClassifyObjective` (CE or `multi_prefix_ce`, val macro-F1) and stays the default, so no baseline config changes.
- Models may return a dict of heads. `fit` passes it through untouched.
- Add optional EMA weights (spec 008: 0.999 **with warm-up** `min(0.999, (1+t)/(10+t))`, saved in the checkpoint, used for evaluation), optional layer-wise LR decay (parameter groups from a model-provided depth map), a label-free mode (no `y`, no balanced sampler) for SSL, a **minimum-steps** setting (train for max(epochs, N steps), F11), and validation every N steps for runs whose epochs are short.
- **Regression test (the point of this task):** B2 and B3 trained for 2 epochs on a fixed synthetic set produce identical losses and identical checkpoints before and after the change. The existing bit-exact resume tests stay green unmodified.
- **Done when:** the regression test and every existing `tests/training/` test pass with no edits to their expectations.

### T4 — The PAT backbone and B5 (spec 006)

**Files:** `src/adl_etc/models/pat.py`, `src/adl_etc/models/pat_stream.py`, `configs/models/pat.yaml`, `configs/train/b5_pat_d1.yaml`, `src/adl_etc/models/baselines/predict.py`, tests

- Token embeddings (size, ipt, dir, push, position) plus a learned `[BOS]` at position 0, so there are 31 positions. **Side channel on by default** (F10): `W_side · cont_k` from the standardised continuous view, so the exact packet size reaches the model. Pre-LN blocks, SDPA attention with fp32 softmax, causal mask combined with a key-padding mask from `ppi_len`, **final LayerNorm**. `d_model=128`, `L=4`, 4 heads, **FFN 512** (F14).
- Heads on every position: `cls`, `safe` (input `[z_k, entropy(cls), margin(cls), entropy(next), p_end, k/30]`), `next` (widths taken from `SIZE_VOCAB`, `IPT_VOCAB` and `DIR_VOCAB` in `features.py`, targets are the next token ids, `ignore_index=PAD_INDEX`, plus the binary **`end`** logit, ignored at truncated positions), and `proj` (128-d, L2-normalised).
- `causal: false` switch (no causal mask) for spec 007's MPM comparison. The same class, so weights copy across.
- `PATStream`: per-flow KV cache, `step(packet) -> outputs_k`.
- `predict_dense` support: the PAT is causal, so `DenseLogits(indexing="effective")` with the BOS position dropped. The existing evaluation path then works unchanged.
- **B5** = the PAT with `ClassifyObjective` on full flows only (no prefix supervision, no SSL). Spec 005's definition, finally buildable.
- **Tests:** causality (perturb packet j > k in tokens *and* side channel, outputs at k unchanged); stream output equals full forward at every k within fp32 tolerance, side channel on; parameter count < 2M; padded positions get no gradient through `next`; the `end` target is ignored when `ppi_len` = 30; `causal: false` outputs *do* change when a later packet changes (so the switch is real); the model builds and trains with the side channel off; synthetic overfit > 99% in 200 steps (phase 2's strict check).
- **Kaggle smoke and measurement:** one short B5 run on real D1. Record real flows/s, epoch time and peak memory in spec 015, and rebuild its weekly budget table from them. That closes phase 2 criterion 8. The same run re-measures F10-F12 on the real train split (length distribution, distinct sizes per size bin, identical-token twin rate per 4,096 batch) and records them here. Then three B5 seeds on `d1_open`.
- **Done when:** all tests pass, B5 has three tracked seeds, and spec 015's PAT numbers are measured.

### T5 — SSL pretraining (spec 007)

**Files:** `src/adl_etc/training/{ssl,augment_batch}.py`, `src/adl_etc/evaluation/probes.py`, `configs/pretrain/{npp_pfc,npp,pfc,mpm}.yaml`, tests

- Batched augmentations (F4): vectorised `drop`, `reorder`, `ipt_jitter`, `size_jitter` and `crop` over a whole batch, seeded from `(seed, epoch, batch)` like everything else, so resume stays exact.
- `NPPObjective` (CE on the next packet's tokens plus BCE on `end`, padded and truncated targets ignored, targets from the *un-augmented* flow as spec 007 says) and `PFCObjective` (symmetric InfoNCE, τ = 0.1, prefix crop K ~ U{3..30} after augmentation, flows with `ppi_len <= 2` skipped for PFC only, **twin masking**: flows with identical un-augmented tokens are removed from each other's negatives, F12). The combined loss is `L_NPP + λ·L_PFC`, with λ from config. One augmented view feeds NPP as well, so there is no third forward pass.
- `MPMObjective`: 30% masked token reconstruction on the `causal: false` PAT. Its weights are then loaded into the causal PAT for fine-tuning.
- A pretrain stage in `run.py` (`stage: pretrain`) that reads **only** the train periods (spec 007's temporal rule) and fixed epochs (8 to 10), with no early stopping since there is no label to stop on. The output checkpoint holds the backbone plus the `next` and `proj` heads.
- Probes logged per epoch: kNN (k=20) on `proj`, a linear probe on frozen `z_K` at K ∈ {3, 5, 8, 30} with 10% labels (NumPy logistic regression or torch, not in `evaluation/`), alignment/uniformity of `proj` (collapse detector), and val NPP perplexity.
- **Tests:** NPP ignores padded targets and the truncated `end` position; PFC's positives are on the diagonal and the symmetric loss is the mean of both directions; two identical-token flows in a batch are not each other's negatives; batched augmentations match the per-flow reference in distribution and are deterministic per seed; 1k-flow overfit (NPP loss under a threshold in 300 steps, kNN probe above chance); a deliberately collapsed `proj` trips the uniformity alarm; the pretrain stage refuses a split whose periods include val or test weeks.
- **Done when:** the four SSL variants train locally on a D3 or D4 sample without collapse, then on Kaggle: NPP+PFC at three seeds; NPP-only, PFC-only, MPM, and λ ∈ {0.5, 2} at one seed each.

### T6 — Supervised prefix fine-tuning (spec 008)

**Files:** `src/adl_etc/training/{finetune,losses}.py`, `configs/finetune/{scratch,ssl}_{100,10,1}.yaml`, tests

- `PrefixObjective`: `L_cls` with the ramp `w_k` and label smoothing `ε` (defaults `min(1, k/8)` and 0.1, both decided below); self-distillation `L_sd` (KL from the **detached** last real position to each k, T = 2, weight 0.2, F14); `L_safe` (BCE on *detached* logits, reweighted per K); `L_NPP` (with `end`) at weight 0.1 to keep the `next` head alive. It extends `losses.multi_prefix_ce`; it does not replace it (phase 2 correction 4).
- Initialise from an SSL checkpoint with a strict key check. Refuse a checkpoint built with a different `K_MAX`, vocabulary or tokeniser. The `cls` and `safe` heads start fresh.
- Layer-wise LR decay 0.85 for SSL-initialised runs, EMA 0.999 with warm-up, early stopping on val `AUC_K` (`metrics.auc_k`), patience 3 validations, peak LR 5e-4 (SSL) or 1e-3 (scratch).
- **Label regimes** (F11): 100% at batch 4096 for 15 epochs; 10% and 1% at batch 1024 for max(15 epochs, 3,000 steps), validating every 500 steps when epochs are shorter than that.
- **Design ablations first** (spec 008, "Design choices"). Before the grid, run the scratch PAT at 100% labels, seed 0, once with the defaults (this run is also the grid's scratch-100%-seed-0 entry) and once each with uniform `w_k`, `ε = 0`, dropout 0, and the side channel off. Pick each on val (`AUC_K`; label smoothing on P-ECHO accuracy at mean K ∈ {4, 6, 8}). A difference under 3x B5's seed-to-seed std is a tie and keeps the default. Record the values and the decisions in this plan's progress log **before** the grid is queued. Test_id is not read.
- **Tests:** loss masking; the gradient of `L_safe` with respect to `cls` parameters is exactly zero; `L_sd` sends no gradient into the teacher position's logits; `w_k` ramp values; the 1% config runs at least 3,000 steps with an unchanged label space; a bad SSL checkpoint is refused; fine-tuning from SSL reaches at least scratch acc@30 on a 100k-flow subset (spec 008's regression test).
- **Done when:** the design choices are recorded, and the grid {scratch, SSL-main} × {100%, 10%, 1%} × 3 seeds, plus one 100% run from each SSL ablation, is tracked on `d1_open` with the chosen settings.

### T7 — Per-K calibration (spec 008)

**Files:** `src/adl_etc/evaluation/calibration.py`, `src/adl_etc/evaluation/{report,run}.py`, `src/adl_etc/evaluation/plotstyle.py`, tests

- Pure NumPy, no torch, no sklearn (F9). In order: **logit adjustment** (`+ log π_nat − log π_samp`, with `π_samp` computed exactly from `balanced_weights` and the train counts, F13); per-K temperature on the adjusted logits (30 scalars, a 1-D NLL minimisation); per-K isotonic tables for `sigmoid(safe_K)` via pool-adjacent-violators, falling back to a Platt fit when a K has fewer than 200 val samples.
- The report gives macro-F1 for both adjusted and unadjusted logits; policies, calibration and (later) the energy score use the adjusted ones.
- Fitted on a seeded 250k sample of val, saved as `calibration.json` next to the checkpoint with the checkpoint's hash in it (a mismatched pair raises), and applied by `evaluation.run` before any metric is computed.
- The report gains ECE and a reliability diagram per K ∈ {3, 8, 30}, measured on **test_id** (F7).
- **Tests:** logit adjustment is a no-op under a uniform sampler and an exact per-class shift under a known one; temperature scaling never raises val NLL; isotonic output is monotone and within [0, 1]; the PAV result matches sklearn's on random data (skipped without sklearn); the Platt fallback triggers below 200 samples; calibration paired with the wrong checkpoint raises.
- **Done when:** every T6 checkpoint has its calibration file, and the report shows test_id ECE per K.

### T8 — Robustness evaluation and the H1/H5 verdict

**Files:** `src/adl_etc/evaluation/run.py`, `configs/eval/d1_open_*.yaml`, `scripts/make_tables.py`, `results/summaries/`, `docs/`

- `evaluation.run` gains a `perturb:` block (drop p ∈ {0.05, 0.1, 0.2}, reorder, jitter) applied to test_id with a fixed seed, using T5's batched augmentations.
- `make_tables.py` gains the phase-3 columns: acc@K for K ∈ {1, 3, 5, 8, 30}, `AUC_K`, `k95`, test_id ECE, and accuracy under 10% drop. It also gains a paired bootstrap (SSL vs scratch, same seeds), as spec 004 asks.
- Write `docs/results-phase-3.md`: H1 and H5 judged against spec 007's own thresholds (≥ 3 points acc@K for K ≤ 5, ≥ 5 points at 10% labels, ≥ 10% fewer packets-to-decision at equal accuracy, robustness drop at least halved). Pass or fail, stated plainly, with the ablation table complete either way.
- **Done when:** the tables regenerate from MLflow with zero hand edits and the verdict document cites only numbers that appear in them.

### T9 — Optional: equal-budget tuning and B4

- **Equal-budget search:** `models/search.py` (built in phase 2, never run) with the same budget for B2, B3 and the scratch PAT, for example 8 trials × 3 epochs on a seeded 1M-flow train subsample, chosen on val. The PAT's space includes FFN ∈ {256, 512}, peak LR and weight decay; dropout, the loss ramp, label smoothing and the side channel are left to T6's design ablations, so nothing is tuned twice. Run it before the T6 grid if it is going to run at all, so the grid uses the chosen settings.
- **B4:** install the extras, fetch the weights and build the adapter with a hand-computed `ppi_transform` test. The DataZoo-batch and D2 sanity tests skip with explicit reasons. Time-boxed to 2 days, and recorded honestly if it slips again.

### T10 — Close-out

Spec corrections from the table below, `specs/README.md` build-order state for steps 8 to 10, spec status lines, phase-2 exit criteria updated, this plan's exit criteria assessed item by item.

## Order and parallelism

**Transformer first, baseline retraining later** (owner decision, 2026-10-03). The baseline re-runs (T1b) do not block any Transformer work, so they move out of the critical path and into whichever Kaggle week has quota left over.

```
LOCAL (no GPU quota)
  T3 loop ──► T4 PAT built + tested ──► T6 fine-tune code ──► T1a open-split code
                                         (T2 cross-session resume alongside; T5 SSL code built
                                          while week 1's first Kaggle runs are going)
KAGGLE WEEK 1
  T4 measure + B5 (3 seeds) ──► (T9 search, optional) ──► T6 design ablations ──► T5 SSL runs
                                                          (decisions recorded)
KAGGLE WEEK 2
  T6 fine-tune grid ──► T7 calibration ──► T8 robustness + verdict ──► T10 close-out
                                             ▲
ANY WEEK WITH SPARE QUOTA                    │
  T1b baseline re-runs (B2, B3) ─────────────┘   T0 paperwork: any time
```

The step-by-step order:

1. **T3**: extend the training loop (local).
2. **T4**: build the PAT and test it locally (causality, stream equivalence, overfit).
3. **T6, code only**: `PrefixObjective`, the label-regime rules (local). Scratch fine-tuning needs no SSL checkpoint, so it can be built and run before T5.
4. **T1a**: the open-set split code (local, about a day). Must be done before step 6, so every PAT run on Kaggle trains on the 150 known classes from the start.
5. **T2**: cross-session resume, alongside steps 2 to 4. Must be done before the SSL queue, the first one long enough to need it.
6. **Kaggle week 1**, in this order:
   1. T4's measurement run, which also re-measures F10-F12's numbers on the real train split.
   2. B5 (3 seeds).
   3. T9, if it runs at all.
   4. **T6's design ablations**, with the decisions recorded in the progress log.
   5. T5's SSL runs.

   The design ablations go **before** SSL because one of them (the side channel) changes the model's input, which SSL pretraining must use too. T5's code is built locally while steps 1 to 4 run.
7. **Kaggle week 2**: T6's fine-tune grid with the chosen settings, then T7, then T8's evaluation passes.
8. **T1b**: the B2 and B3 re-runs, in whichever week has spare quota. Must be done before T8 compares anything.
9. **T10**: close-out.

Hard dependencies: T3 blocks T4. T1a blocks every PAT run on Kaggle. T6's design ablations block T5's SSL runs and T6's grid. T2 blocks the T5 and T6 queues on Kaggle. T5 blocks T6's SSL-initialised runs. T7 needs T6's checkpoints. T8 needs T1b, T6 and T7.

## GPU budget (estimate; T4 replaces it with measurements)

Based on B3's measured 43k flows/s, assuming the PAT is similar per view. The quota is 30 GPU-hours **per week** and resets weekly. The ~43 h total is for the whole phase, split across two Kaggle weeks so that neither goes over.

| Job | Runs | Est. GPU-h | Kaggle week |
|---|---|---|---|
| T4 measurement run + B5 | 3 | 2 | 1 |
| T9 search (optional) | 24 | 4 | 1 |
| T6 design ablations (scratch, 100%, seed 0: defaults + 4 alternatives) | 5 | 3.5 | 1 |
| SSL NPP+PFC | 3 | 3 | 1 |
| SSL ablations (NPP, PFC, MPM, λ 0.5, λ 2) | 5 | 5 | 1 |
| Fine-tune 100%: scratch + SSL × 3 seeds, + 5 ablation fine-tunes (the design-ablation default run is reused as scratch seed 0 if every default wins) | 10-11 | 8 | 2 |
| Fine-tune 10% and 1%: scratch + SSL × 3 seeds (batch 1024, ≥ 3,000 steps) | 12 | 2 | 2 (or the local RTX 3050) |
| Robustness and calibration evaluation passes | many | 2 | 2 |
| Baseline re-runs on `d1_open` (B2, B3), T1b | 6 | 5 | whichever week has room |
| Slack for failures | | 8 | 4 per week |
| **Total** | | **~43** | **week 1 ≈ 21.5, week 2 ≈ 16, plus 5 floating** |

**If the budget gets tight**, in order of preference:

1. Run two seeds at once on Kaggle's T4 x2 accelerator. In phase 2 the second T4 sat idle. Check in Kaggle's UI, after one real run, whether a T4 x2 session counts the same quota as a single GPU before relying on it.
2. Move the 10% and 1% fine-tunes to the local RTX 3050.
3. Skip T9 (saves ~4 h); every model then uses spec defaults, stated in the write-up.
4. Push T1b into a third week. It is off the critical path, so a delay only postpones T8's comparison.

## Spec corrections this plan requires

Corrections to **shipped** behaviour (specs 004, 005, 015) land with their task, so those specs always describe what exists. Specs 006, 007 and 008 are unbuilt designs, so their corrections were **applied to the spec text on 2026-10-03** (each spec has a `Revisions` section); the spec is then the build target, and the "Lands with" column names the task that implements it.

| # | Where | Problem | Correction | Lands with |
|---|---|---|---|---|
| 1 | 006 | Success bar is "within 1 point of the GRU at K=30", but the CNN is 5 points stronger (F2). | Compare like with like: B5 vs the CNN at K=30; the prefix-trained PAT vs both across all K. | spec applied 2026-10-03; code T4 |
| 2 | 006 | `next` head widths written as 65/33/2; `dir`'s real vocabulary is 3 (pad, +1, −1). | Widths are `features.py`'s vocabulary constants, with `ignore_index = PAD_INDEX`. | spec applied 2026-10-03; code T4 |
| 3 | 006 | The side channel defines its own `log1p(x)/7.3` scaling. | Use the standardised `continuous` view. One normalisation in the project. | spec applied 2026-10-03; code T4 |
| 4 | 006 | Depth exits and streaming latency are listed as phase-3 deliverables. | Depth exits move to spec 011's phase; latency stays with spec 013. `PATStream` correctness is built here. | spec applied 2026-10-03; code T4 |
| 5 | 007 | Corpus "about 3M XS", "15-25 min/epoch", GPU-side augmentation. | 5,989,515 real flows, measured epoch time, batched CPU augmentation first (F3, F4). | spec applied 2026-10-03; code T5 |
| 6 | 008 | ECE criterion measured on val, the data calibration is fitted on (F7). | ECE reported on test_id. | spec applied 2026-10-03; code T7 |
| 7 | 008 | Low-label regimes remove zero-label classes, changing the task between regimes (F8). | Label space fixed; zero-label classes count as errors. | spec applied 2026-10-03; code T1a, T6 |
| 8 | 008 | Saved logits as top-10 + energy + projection. | Phase 2's 250k `EvalArrays` cap; calibration on a seeded 250k val sample. | spec applied 2026-10-03; code T7 |
| 9 | 008 | Isotonic regression implied via sklearn. | NumPy PAV in `evaluation/calibration.py` (F9). | spec applied 2026-10-03; code T7 |
| 13 | 006 | Token-only input loses the exact packet sizes the baselines use: 1,460 sizes in 53 bins (F10). | Side channel **on by default**; off is a design ablation. | spec applied 2026-10-03; code T4 |
| 14 | 006 | No final LayerNorm; FFN ratio 2; no way to predict that a flow is ending (F14). | Final LayerNorm, FFN 512, an `end` logit in `next`, `p_end` into `safe`. | spec applied 2026-10-03; code T4 |
| 15 | 007 | InfoNCE false negatives assumed rare; measured 5.6% of anchors (F12). | Twin masking in PFC; `end` term in NPP. | spec applied 2026-10-03; code T5 |
| 16 | 008 | The 1% regime gets ~220 steps and an EMA still 80% initial weights (F11). | Batch 1024 and a 3,000-step floor for 10%/1%; EMA warm-up. | spec applied 2026-10-03; code T3, T6 |
| 17 | 008 | The sampler's class prior is never corrected; the distillation teacher is not detached; four impactful choices are fixed by assumption (F13, F14). | Logit adjustment before calibration; `stopgrad` teacher; design ablations decided on val before the grid. | spec applied 2026-10-03; code T6, T7 |
| 10 | 015 | Resume is described as working across sessions; it only works within one (F6). | Document the real cross-session mechanism; budget table measured. | T2, T4 |
| 11 | 005 | P-RL and B4 still listed for phase 3 without status. | P-RL to phase 4 with spec 009; B4's outcome recorded either way. | T9 |
| 12 | 004 | No split file uses the open-set draw. | `d1_open.yaml` named as the main split for phase 3 onward. | T1 |

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **SSL does not beat scratch.** | H1 fails, the headline contribution C1 is weaker. | Rule 3: report it with the full ablation table. The low-label regimes are where SSL usually helps most, so the 10% and 1% results may carry the claim even if 100% does not. Decide nothing after seeing test_id. |
| **The PAT loses to the CNN at K=30.** | "Anytime" story weaker at full length. | The project's claim is earliness (`AUC_K`, acc at small K), not K=30. Report both honestly; F2's bar makes the comparison visible rather than hidden. |
| **PFC collapses.** | Useless `proj`, which also breaks spec 010 later. | Uniformity metric logged every epoch with an alarm; spec 007's remedies (lower τ, higher λ) tried on val probes only. |
| **Kaggle self-reference for resume is not allowed.** | T2's first choice fails. | The `push-resume` dataset fallback needs nothing new from Kaggle. |
| **Augmentation still bottlenecks the GPU.** | SSL epochs several times slower than estimated. | T4 measures first. Move augmentation to the GPU only if the measurement says so. |
| **The baselines were never tuned.** | A reviewer can say the PAT won by attention, not architecture. | T9's equal-budget search, or "all defaults" stated explicitly. |
| **Re-running baselines on the open split costs quota.** | Roughly 6 GPU-hours. | Cheap next to retraining the PAT grid in phase 4 (F5). |
| **Calibrating, early-stopping and threshold-tuning all on the same val.** | Mild optimism in val numbers. | Every reported number is test_id (F7); val is for choices only. |
| **Single-seed design ablations pick noise.** | A default is swapped for an alternative that is not really better. | A difference under 3x B5's seed-to-seed std is a tie and keeps the default (B3's seed std was 0.0007, so real differences of a point or more stand out clearly). |
| **F10-F12's numbers come from one holiday.** | Twin rate, length distribution or size spread could differ on working weeks. | T4's Kaggle measurement run re-measures them on the real train split and records them; a materially different number re-opens the decision it supports. |
| **Logit adjustment lowers macro-F1.** | The headline metric looks worse while accuracy and calibration improve. | Report macro-F1 for both adjusted and unadjusted logits; the policies use adjusted ones because they need real error rates. |

## Exit criteria

Phase 4 (spec 010, then 009) starts when all of these hold.

1. `pytest`, `ruff check src/ tests/ scripts/` and `mypy src/` all clean, with the new causality, stream-equivalence, loop-regression and calibration tests included.
2. `d1_open.yaml` exists and every phase-3 headline run uses it. B2 and B3 have three tracked seeds on it.
3. The PAT passes causality and stream-equivalence tests, has under 2M parameters, and B5 has three tracked seeds.
4. SSL: NPP+PFC at three seeds and every ablation at one seed are tracked, with probe and collapse metrics logged.
5. Fine-tuning: the four design choices (loss ramp, label smoothing, dropout, side channel) were decided on val and recorded with their numbers **before** the grid ran; the {scratch, SSL} × {100, 10, 1}% × 3-seed grid is tracked with those settings, and every checkpoint has a matching calibration file.
6. `docs/results-phase-3.md` gives an H1 and H5 verdict against spec 007's thresholds, citing only `make_tables.py` output, whichever way it falls.
7. A Kaggle run paused in one session has completed in the next (phase 2 criterion 7).
8. Spec 015's PAT throughput and weekly budget table are measured (phase 2 criterion 8).
9. All spec corrections landed; `specs/README.md` build-order state updated for steps 8 to 10.

What phase 4 inherits: calibrated per-K class probabilities, a `safe` head whose output approximates P(correct | commit at K), a `next`-entropy signal, a `proj` embedding for Mahalanobis scoring, real unknown flows in val and test, and a Kaggle queue that survives being interrupted.

## Progress log

- **2026-10-03.** Plan written from specs 005-008 and 015, the phase-2 plan and the real results table. Nine findings checked against code and results, each changing a task: `fit` has one loss (F1); spec 006's bar names the weaker baseline (F2: CNN 0.8651 vs GRU 0.8124); the corpus is 2x spec 007's and the GPU is not the constraint (F3); augmentations are per-flow Python (F4); every real run is closed-set (F5); Kaggle cannot resume across pushes (F6, found for real in T8); spec 008 grades calibration on its own fitting data (F7); spec 008's low-label rule changes the task between regimes (F8); isotonic would pull sklearn into `evaluation/` (F9). Twelve spec corrections scheduled. Not started.
- **2026-10-03 (later).** Two owner decisions folded in. (1) B1 (XGBoost) removed from the baselines (spec 005, phase-2 progress log), so T0 is paperwork only and the baseline re-runs cover B2 and B3. (2) Transformer work goes first: T1 split into T1a (open-split code, before the first PAT Kaggle run) and T1b (B2/B3 re-runs, any week with spare quota, before T8). The order section and the budget table now show which Kaggle week each job falls in, plus four fallbacks if the quota gets tight.
- **2026-10-03 (architecture review).** Reviewed the PAT design and its hyperparameters against real D1 numbers measured on the probe day (487,081 flows). Five new findings (F10-F14), all applied to specs 006-008 directly since they are unbuilt (corrections 1-9 and 13-17 marked applied; 10-12 still land with their tasks): size tokens collapse 1,460 exact sizes into 53 bins, so the side channel is now on by default (F10); the 1% regime would get ~220 steps with an EMA still 80% initial weights, now batch 1024 + a 3,000-step floor + EMA warm-up (F11); 5.6% of InfoNCE anchors have an identical twin, now masked (F12); class-balanced sampling's prior shift is now corrected by logit adjustment before calibration (F13); final LayerNorm, FFN 512, a detached distillation teacher and an `end` target added, and four choices (loss ramp, label smoothing, dropout, side channel) moved to single-seed design ablations decided on val before the grid (F14). F2 refined to compare like with like (B5 vs the CNN; prefix-PAT vs both). Week 1's order is now B5, then T9, then design ablations, then SSL (the side-channel decision must precede SSL); budget ~43 h (week 1 ≈ 21.5, week 2 ≈ 16, 5 floating). Also measured: D1 mean `ppi_len` 16, median 13, not the ~7 CLAUDE.md quotes (that figure is D4's).
