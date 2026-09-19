# Spec 004: Splits and Evaluation Protocol

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 4 of 18, then extended alongside 010, 009 and 011 (it is scaffolding, not a one-off build)
- **Depends on:** 001, 003. **Used by:** all model and inference specs; 013; 014.

## Problem

The project's claims are about earliness, unknown detection, drift and budgets *jointly*. A single accuracy number on a random split would be misleading (leakage, temporal optimism) and would not test the hypotheses. This spec fixes the splits, metrics and reporting format once so that every model and every ablation is compared on identical terms (contribution C4).

## Goals

- Leakage-free temporal and open-set splits for D1, D2, D3, D4.
- A metric suite that is a function of prefix length K and of test period.
- A single `evaluate(model_or_policy, split) -> Report` entry point producing JSON + plots.
- Multi-seed statistics and paired significance tests for the main table.

## Non-goals

- Random-split results as headline numbers (they are reported once as a sanity baseline only).

## Splits

### D1 CESNET-TLS-Year22 (primary)

| Split | Periods | Purpose |
|---|---|---|
| train | weeks 11 to 26 (months 3 to 6) | supervised training; SSL (labels ignored) |
| val | weeks 27 to 30 | model selection, calibration, threshold fitting, controller tuning |
| test-ID | weeks 31 to 34 | in-distribution (immediately after training) headline numbers |
| test-drift | weeks 35 to 52, evaluated week by week | drift curves out to T+18 weeks |

Granularity note (resolved 2026-09-17): the raw release mirrored on Kaggle is organised by ISO week and day (spec 001, Path B), so **weekly** drift curves are available directly and are the primary presentation; DataZoo's monthly periods (`M-2022-3` and so on) are the equivalent coarse split used when working through Path A. Week numbering follows the directory names `WEEK-2022-NN`.

Rationale: weeks 1 to 10 are avoided because of the documented week-10 exporter change (the dataset authors recommend treating weeks 1 to 9 and 11 to 52 as separate regimes); starting at week 11 keeps the whole usable span contiguous. Four months of training data is about 3M flows in XS, enough for the model size, and leaves an 18-week drift horizon, longer than the 8 to 10 weeks reported by the dataset authors and by CAPE-Net. If GPU time is short, train on weeks 11 to 18 only (documented as `train_small`).

Open-set split: 150 known / 30 unknown classes drawn with seed 42, stratified so that unknowns cover several categories and include both frequent and rare apps. Unknown flows appear only in val (for threshold fitting, at most 50% of unknown classes) and test. A second unknown draw (seed 43) is used to report variance of open-set metrics.

Time-ordered validation: within each period, evaluation is streamed in `ts` order so that the controller (spec 011) sees a realistic sequence.

### D2 CESNET-QUIC22

train W-2022-44, val W-2022-45, test W-2022-46, drift W-2022-47. Also used unlabeled for SSL. Cross-protocol experiment: model pretrained on D1 (SSL) then fine-tuned on D2 with 1%, 10%, 100% labels.

### D3 ISCX VPN-nonVPN

Group-aware split by `session_id` (capture file): no file appears in both train and test. Five-fold grouped CV, category-level labels (7) and VPN/non-VPN condition split (train non-VPN, test VPN) as the guide's "condition shift". Results are secondary.

### D4 USTC-TFC2016

Used as anomaly test only: model trained on D1 (or D3 benign) must flag malware flows as unknown/anomalous. Report AUROC of the unknown score with USTC malware as positives.

## Metrics

All computed by `src/evaluation/metrics.py`, each as a function of K where applicable.

**Classification (known classes):** accuracy, macro-F1, per-class F1, balanced accuracy, confusion matrix (top-30 classes plotted), PR-AUC (macro) and ROC-AUC (macro, one-vs-rest).

**Earliness:**
- `acc@K` for K in 1..30 (fixed-K evaluation of any model).
- `AUC_K` = area under acc@K vs K normalised by 30 (higher is better).
- `K95` = smallest K where acc@K >= 0.95 x acc@30.
- For adaptive policies: mean K, median K, p95 K, accuracy of committed flows, coverage (fraction committed before K_max), and the Pareto front of (mean K, accuracy) obtained by sweeping the policy threshold.
- Harmonic mean of accuracy and earliness (1 − meanK/30), as in the ECTS literature, for single-number comparisons.

**Open-set / anomaly:** AUROC and AUPR of the unknown score (unknowns positive), FPR at 95% TPR, open-set F1 (macro-F1 over known classes plus the unknown class), and *rejection earliness*: mean K at which unknown flows are rejected.

**Drift:** each metric per test period; degradation slope (metric vs months since training); per-class drift table (classes whose F1 drops > 10 points).

**Budget (spec 011):** target vs realised E[K] per window, absolute tracking error, fraction of windows within 5% of target, accuracy at each budget.

**Calibration:** expected calibration error (ECE, 15 bins) at each K, reliability diagrams at K in {3, 8, 30}.

**Efficiency (spec 013):** params, FLOPs per packet step, latency per flow and per packet, throughput, memory.

## Statistics

- Three seeds for every headline configuration (seed controls init, data order, augmentation, unknown draw where applicable).
- Report mean ± std; paired bootstrap (1000 resamples over flows) for differences between our policy and each baseline; 95% CIs in the appendix.

## Reporting format

`Report` = JSON with metric arrays + PNG/HTML plots (accuracy-vs-K, Pareto front, AUROC-vs-K, drift curves, reliability diagrams, budget tracking). Stored under `results/<run_id>/` and logged to MLflow (spec 014). `results/summaries/main_table.md` is regenerated by `scripts/make_tables.py` from MLflow.

## Inputs and outputs

- Inputs: model or policy object exposing `predict_prefix(batch, K)` and, for adaptive policies, `decide_stream(flow)`; shard splits from `configs/splits/*.yaml`.
- Outputs: `Report` objects, tables, figures.

## Edge cases

- Classes absent from a test period: excluded from macro averages for that period, listed in the report.
- Flows shorter than K: `acc@K` uses the flow's full length (a K=10 evaluation on a 6-packet flow uses 6 packets); reported separately as "short-flow accuracy".
- Unknown draw producing a class with < 100 test flows: redraw until each unknown has >= 100 flows in test-ID.
- Controller evaluation must not reuse the same window for tuning and reporting (tuning on val only).

## Performance considerations

- Evaluating 30 prefixes for 1M test flows with a causal Transformer is one forward pass per batch (all prefix outputs come from one pass, spec 006); baselines without causal structure need 30 passes and are evaluated at K in {1,2,3,4,5,6,8,10,12,15,20,25,30}.
- Metrics are computed with NumPy/scikit-learn on saved logits (`.npy` per run) so that policies can be re-evaluated without re-running models.

## Testing

- Metric unit tests with tiny hand-computed examples (acc@K with padding, AUROC with ties, K95 monotonicity).
- Leakage tests: no `session_id` in both train and test (D3); no unknown label in train; stats file hash equals the training period hash.
- Determinism test: same seed, same report hash for a small model.

## Interactions

- Provides the harness used by specs 005 to 012; spec 013 adds efficiency; spec 014 stores results.

## Success criteria

- `python -m src.evaluation.run --config configs/eval/d1_full.yaml --run <id>` produces a complete report for any registered model.
- Main table and figures regenerate from MLflow without manual editing.

## Open questions

- Unknown-class count (30) is a judgement; CAPE-Net's exact split is not public, so we document ours.
- Whether to report drift weekly (52 points, noisier) or in 4-week bins (cleaner) in the paper. Default: weekly curves with a 4-week rolling mean overlaid.
