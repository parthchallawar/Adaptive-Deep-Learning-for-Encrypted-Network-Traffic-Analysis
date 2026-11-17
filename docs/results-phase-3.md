# Phase 3 results: the H1 and H5 verdicts

**Status: three seeds for the headline comparison.** Written 2026-11-15 from one
seed, revised 2026-11-17 after phase 3's Session 4 (kernel v17) trained seeds 1 and 2
of the tuned PAT pair and of B3 (GRU). Source: `results/summaries/phase3_table.md`,
which `scripts/make_tables.py` regenerates from local MLflow. Every number below
appears in that table; section references (§1–§4) are to its sections. A difference
quoted here is the subtraction of two numbers shown there. Nothing in this document
is a measurement of its own.

What is final and what is not:

- **Three seeds (0, 1, 2), mean ± std:** the tuned PAT from scratch, the tuned PAT
  with SSL init, and B3. The H1 and H5 verdicts and the comparison with B3 rest on
  these.
- **One seed:** everything else (B5, B2, the default-recipe PATs, the low-label
  pairs, the ablations). Read those as indications.
- **Paired bootstrap (§4):** done for the tuned pair, per seed. The other pairs have
  no per-flow predictions (evaluated before `predictions.npz` existed).

Session 4 also re-evaluated the three seed-0 runs to get their predictions. Their
reports came out with the same content hashes as Session 3's, so the evaluation is
reproducible end to end on Kaggle's GPU.

## Setup in one paragraph

D1 (CESNET-TLS-Year22), open-set split `d1_open`: 150 known classes for training,
30 held out as unknown. The test set is weeks 31–34 (`test_id`), capped at 250,000
flows, the same draw for every model; 25,308 of them are unknown-class (§1).
Accuracy is on known-class flows only. "acc@K" means accuracy after the first K
packets. "AUC_K" is the mean acc@K over K = 1..30, the project's single
early-accuracy number.

Two PAT recipes appear:

- **default**: dropout 0.1, label smoothing 0.1, as specified in spec 008.
- **tuned**: both set to 0. The single-change ablations in §3 (one seed, validation)
  favoured this:
  - no dropout: AUC_K 0.803
  - no label smoothing: 0.801
  - the 4-epoch reference: 0.790

  The tuned recipe is `configs/train/ft_pat_d1_tuned.yaml` (and its `_ssl_` twin).

## H1: prefix-predictive SSL improves early accuracy

Spec 007's acceptance criteria, judged on the tuned pair on `test_id`, three seeds
(§1, mean; §4, per seed):

| Criterion (spec 007) | Scratch | SSL init | Difference | Per seed (§4) | Verdict |
|---|---|---|---|---|---|
| ≥ 3 points acc@K for K ≤ 5, test-ID: K = 1 | 10.5 ± 1.0 | 16.8 ± 1.2 | +6.3 | +6.1, +8.5, +4.2 | **met** |
| … K = 3 | 52.1 ± 1.2 | 59.4 ± 0.5 | +7.3 | +6.2, +8.7, +7.0 | **met** |
| … K = 5 | 75.3 ± 0.6 | 77.7 ± 0.5 | +2.4 | +2.8, +2.9, +1.5 | **not met** |
| ≥ 5 points with 10% labels | see below | | | | **not met** (one seed) |
| ≥ 10% fewer packets-to-decision at equal accuracy | k95 8 | k95 8 | 0 | | **not measured** |

- **Early accuracy: met for K = 1 and K = 3 in every seed, missed at K = 5 in every
  seed.** Each per-seed 95% interval in §4 is narrow (about ±0.1 to 0.2 points) and
  lies above 3 for K = 1 and 3. At K = 5 no seed reaches 3: the closest is seed 1 at
  +2.9 [+2.8, +3.0]. The gain varies more between seeds than the intervals suggest
  (K = 1: +4.2 to +8.5), which is why the verdict uses all three seeds, not one
  interval.
- **Overall:** AUC_K 0.813 ± 0.001 → 0.826 ± 0.004 (+0.013; per seed +0.016, +0.013,
  +0.008, each interval above zero); macro-F1 0.822 → 0.830.
- **10% labels: not met.** Only the default recipe was run at 10% labels, one seed,
  validation only (§3). SSL gains:
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

**H1 verdict: partly supported.** SSL initialisation reliably helps the earliest
decisions (K ≤ 3) at full labels, by 6 to 7 points on average, in all three seeds.
It fails the K = 5 criterion in all three seeds. The low-label criterion fails on the
one seed run, and packets-to-decision is unmeasured.

## H5: augmented SSL improves robustness to loss and reordering

Spec 007: "under 10% packet drop at test time, accuracy drop is at least halved
relative to scratch training." Change in AUC_K under 10% drop (§2, three seeds):

| | Scratch (tuned) | SSL init (tuned) |
|---|---|---|
| Δ AUC_K, drop 10% | -0.028 ± 0.001 | -0.028 ± 0.000 |

**H5 verdict: not supported.** SSL initialisation does not reduce the damage
at all: the two models lose the same amount at every drop rate (§2: -0.012, -0.028,
about -0.07 for both), in every seed.

What §2 does show is a different result, worth stating plainly because it is the
strongest one in this phase. **Every prefix-trained PAT is far more robust than
every baseline,** whether it starts from SSL or from scratch:

| Model (§2) | Seeds | Δ AUC_K, drop 10% | Δ AUC_K, reorder 5% | acc@8 % under drop 10% |
|---|---|---|---|---|
| PAT tuned, SSL init | 3 | -0.028 ± 0.000 | -0.025 ± 0.000 | 83.1 ± 0.5 |
| PAT tuned, scratch | 3 | -0.028 ± 0.001 | -0.021 ± 0.001 | 82.5 ± 0.2 |
| PAT default, SSL init | 1 | -0.026 | -0.013 | 80.2 |
| B5 PAT (full flow) | 1 | -0.049 | -0.016 | 74.5 |
| B2 CNN (per-K) | 1 | -0.109 | -0.086 | 62.5 |
| B3 GRU | 3 | -0.151 ± 0.005 | -0.100 ± 0.001 | 66.2 ± 0.3 |

The GRU loses 0.151 of AUC_K under 10% drop; the tuned PATs lose 0.028, about a
fifth, with seed-to-seed spreads far smaller than the gap. The likely source is
prefix fine-tuning together with its low-intensity drop/jitter augmentation
(spec 008); SSL is not the source. B5 shares the architecture but neither of those,
and it sits in between. This experiment does not separate the two causes: the
no-augmentation ablation was not robustness-tested.

## The comparison with the baselines (§1)

The headline model is the tuned PAT with SSL init.

- **Against B3 (GRU), the strongest early baseline** (both three seeds), it is ahead
  at every K shown:
  - acc@1: 16.8 vs 10.9
  - acc@3: 59.4 vs 50.8
  - acc@5: 77.7 vs 73.5
  - acc@8: 85.9 vs 82.6
  - acc@30: 89.4 vs 87.2

  It is also ahead on AUC_K (0.826 ± 0.004 vs 0.793 ± 0.002), macro-F1 (0.830 vs
  0.782) and calibration (ECE@30 0.034 vs 0.050). The tuned PAT from scratch is
  also ahead of B3 on AUC_K (0.813) but not at K = 1 (10.5 vs 10.9), so the
  earliest-packet lead is SSL's. The default recipe (one seed) was not ahead: its
  SSL PAT's AUC_K was 0.791.
- **Against the full-flow models** (one seed each), it gives up a little at K = 30
  in exchange for much better early accuracy:
  - acc@30: 89.4 vs 91.5 (B5) and 90.5 (B2)
  - acc@5: 77.7 vs 60.1 (B5) and 54.3 (B2)
- **Unknown-class detection** (AUROC by energy score) is slightly ahead of the GRU:
  0.868 vs 0.854 at K = 5, 0.904 vs 0.895 at K = 30. B2's 0.916 at K = 30 (one seed)
  is the best full-flow value.

B2's AUC_K (0.613) averages its own 13-value K grid. It is not comparable with
the other models' AUC_K, and is left out of the comparisons above.

## Limits and what comes next

1. **Three seeds only where it matters most.** The tuned pair and B3 have three;
   B5, B2, the default recipe, the low-label pairs and the ablations have one.
   Across the three-seed groups the AUC_K std is 0.001 to 0.004, so one-seed gaps of
   a hundredth or more are probably real, and gaps under half a point (SSL vs
   scratch at 10% and 1% labels, most ablations) are not interpretable.
2. **Paired bootstrap only for the tuned pair** (§4). The other pairs were
   evaluated before evaluation wrote `predictions.npz`.
3. **Missing from `test_id`.** The low-label and default-scratch models were only
   evaluated on validation. Session 1's scratch checkpoint could not be
   retrieved once a later kernel version existed; see `docs/kaggle-workflow.md`.
4. **Validation numbers (§3) are selection-biased.** The best epoch was chosen on
   that split.
5. **Next:** packets-to-decision (H1's third criterion) needs phase 4's stopping
   policy (spec 009). Whether augmentation or prefix training causes the robustness
   gap needs the no-augmentation ablation evaluated under damage. Its kernel-v14
   checkpoint cannot be pulled, so that means retraining it (~35 GPU-minutes at 4
   epochs) plus one ~5-minute eval entry.
