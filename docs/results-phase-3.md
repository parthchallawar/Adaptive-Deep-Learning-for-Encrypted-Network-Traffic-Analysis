# Phase 3 results: the H1 and H5 verdicts

**Status: provisional (one seed, no bootstrap yet).** Written 2026-11-15 from
`results/summaries/phase3_table.md`, which `scripts/make_tables.py` regenerates from
local MLflow. Every number below appears in that table; section references (§1–§4)
are to its sections. A difference quoted here is the subtraction of two numbers
shown there. Nothing in this document is a measurement of its own.

What would make it final: three seeds per group (§1 then shows mean ± std) and the
paired bootstrap in §4. §4 is empty for now: per-flow predictions
(`predictions.npz`) only started being written on 2026-11-15, after these runs were
evaluated.

## Setup in one paragraph

D1 (CESNET-TLS-Year22), open-set split `d1_open`: 150 known classes for training,
30 held out as unknown. The test set is weeks 31–34 (`test_id`), capped at 250,000
flows, the same draw for every model; 25,308 of them are unknown-class (§1).
Accuracy is on known-class flows only. "acc@K" means accuracy after the first K
packets. "AUC_K" is the mean acc@K over K = 1..30, the project's single
early-accuracy number. All runs used one seed, so the numbers are not yet results
in the statistical sense.

Two PAT recipes appear:

- **default**: dropout 0.1, label smoothing 0.1, as specified in spec 008.
- **tuned**: both set to 0. The single-change ablations in §3 favoured this:
  - no dropout: AUC_K 0.803
  - no label smoothing: 0.801
  - the 4-epoch reference: 0.790

  The tuned recipe is now `configs/train/ft_pat_d1_tuned.yaml`.

## H1: prefix-predictive SSL improves early accuracy

Spec 007's acceptance criteria, judged on the tuned pair on `test_id` (§1):

| Criterion (spec 007) | Scratch | SSL init | Difference | Verdict |
|---|---|---|---|---|
| ≥ 3 points acc@K for K ≤ 5, test-ID: K = 1 | 10.8 | 16.9 | +6.1 | **met** |
| … K = 3 | 53.4 | 59.6 | +6.2 | **met** |
| … K = 5 | 75.6 | 78.3 | +2.7 | **not met** (short by 0.3) |
| ≥ 5 points with 10% labels | see below | | | **not met** |
| ≥ 10% fewer packets-to-decision at equal accuracy | k95 8 | k95 8 | 0 | **not measured** (see below) |

- **Early accuracy: met for the first three packets, narrowly missed at K = 5.** The
  advantage also shows in AUC_K (0.814 → 0.830) and in macro-F1 (0.823 → 0.836).
  The default-recipe pair shows the same shape on validation, smaller (§3):
  - acc@1: 7.9 → 12.1
  - acc@3: 41.8 → 45.6
  - acc@5: 71.9 → 72.6
- **10% labels: not met.** Only the default recipe was run at 10% labels, and only
  validation numbers exist (§3). SSL gains:
  - +3.4 at K = 1 (7.4 → 10.8)
  - +2.0 at K = 3 (40.9 → 42.9)
  - +0.6 at K = 5 (70.1 → 70.7)

  None reaches 5 points. At 1% labels the gains are smaller still: AUC_K 0.745 →
  0.752.
- **Packets-to-decision: not measured.** The criterion needs a stopping policy at
  equal accuracy (spec 009, phase 4). The nearest proxy in the table, k95 (the
  first K reaching 95% of full-flow accuracy), is 8 for both models, so there is
  no evidence of a reduction yet.
- **Not tested: the masked-only SSL comparison** that the H1 row of spec 000 also
  names. No MPM-pretrained model was fine-tuned.

**H1 verdict: partly supported.** SSL initialisation clearly helps the earliest
decisions (K ≤ 3) at full labels. It fails the K = 5, low-label and
packets-to-decision criteria, or leaves them unmeasured.

## H5: augmented SSL improves robustness to loss and reordering

Spec 007: "under 10% packet drop at test time, accuracy drop is at least halved
relative to scratch training." Change in AUC_K under 10% drop (§2):

| | Scratch (tuned) | SSL init (tuned) |
|---|---|---|
| Δ AUC_K, drop 10% | -0.028 | -0.028 |

**H5 verdict: not supported.** SSL initialisation does not reduce the damage
at all: the two models lose the same amount at every drop rate (§2: -0.012,
-0.028 and -0.073 for both).

What §2 does show is a different result, worth stating plainly because it is the
strongest one in this phase. **Every prefix-trained PAT is far more robust than
every baseline,** whether it starts from SSL or from scratch:

| Model (§2) | Δ AUC_K, drop 10% | Δ AUC_K, reorder 5% | acc@8 % under drop 10% |
|---|---|---|---|
| PAT tuned, SSL init | -0.028 | -0.025 | 83.6 |
| PAT tuned, scratch | -0.028 | -0.021 | 82.3 |
| PAT default, SSL init | -0.026 | -0.013 | 80.2 |
| B5 PAT (full flow) | -0.049 | -0.016 | 74.5 |
| B2 CNN (per-K) | -0.109 | -0.086 | 62.5 |
| B3 GRU | -0.156 | -0.101 | 65.8 |

The GRU loses 0.156 of AUC_K under 10% drop; the tuned PATs lose 0.028. The likely
source is prefix fine-tuning together with its low-intensity drop/jitter
augmentation (spec 008); SSL is not the source. B5 shares the architecture but
neither of those, and it sits in between. This experiment does not separate the
two causes: the no-augmentation ablation was not robustness-tested.

## The comparison with the baselines (§1)

The headline model is the tuned PAT with SSL init.

- **Against B3 (GRU), the strongest early baseline**, it is ahead at every K shown:
  - acc@1: 16.9 vs 11.3
  - acc@3: 59.6 vs 50.8
  - acc@5: 78.3 vs 73.6
  - acc@8: 86.5 vs 82.8
  - acc@30: 89.8 vs 87.3

  It is also ahead on AUC_K (0.830 vs 0.795), macro-F1 (0.836 vs 0.785) and
  calibration (ECE@30 0.030 vs 0.050). This was not true of the default recipe:
  the default SSL PAT's AUC_K was 0.791, below the GRU's 0.795.
- **Against the full-flow models**, it gives up a little at K = 30 in exchange for
  much better early accuracy:
  - acc@30: 89.8 vs 91.5 (B5) and 90.5 (B2)
  - acc@5: 78.3 vs 60.1 (B5) and 54.3 (B2)
- **Unknown-class detection** (AUROC by energy score) is close to the GRU: 0.863 vs
  0.859 at K = 5, 0.903 vs 0.892 at K = 30. B2's 0.916 at K = 30 is the best
  full-flow value.

B2's AUC_K (0.613) averages its own 13-value K grid. It is not comparable with
the other models' AUC_K, and is left out of the comparisons above.

## Limits and what comes next

1. **One seed.** For a sense of scale, phase 2's three-seed baselines on the
   closed-set split varied by a std of 0.0007 (B3) and 0.0017 (B2) in val macro-F1.
   These are `results/summaries/main_table.md`, the one number here from outside the
   phase-3 table. So gaps of a few hundredths are unlikely to be seed noise. Gaps under a point (SSL vs
   scratch at K = 5, the 10% and 1% label pairs) are not interpretable until
   three seeds exist. Next step: seeds 1 and 2 for the tuned pair and B3 (~6.5
   GPU-hours, not yet run).
2. **No paired bootstrap** (§4 empty). It fills in automatically once those runs
   are evaluated, because evaluation now writes `predictions.npz`.
3. **Missing from `test_id`.** The low-label and default-scratch models were only
   evaluated on validation. Session 1's scratch checkpoint could not be
   retrieved once a later kernel version existed; see `docs/kaggle-workflow.md`.
4. **Validation numbers (§3) are selection-biased.** The best epoch was chosen on
   that split.
