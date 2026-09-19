# Spec 010: Unknown-Application and Anomalous-Traffic Detection

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 11 of 18 (before 009, which consumes its unknown score)
- **Depends on:** 006, 008. **Used by:** 009, 011, 012, 016, 017. **Research contribution:** C2 (hypothesis H3).

## Problem

A deployed classifier meets applications it was never trained on and traffic that is simply unusual (new services, malware, scanners). Forcing such flows into a known class is worse than saying "unknown". Existing detectors score a *completed* flow (or a fixed checkpoint, K>=10 in CAPE-Net). But the distribution of any unknown score changes with prefix length: at K=2 every flow looks like every other flow, at K=30 the embedding is sharp. A single threshold across K either rejects everything early or nothing. We need a score whose meaning is stable across K so that unknown flows can be rejected as early as they become distinguishable.

## Goals

- Per-prefix unknown score `u_k` from two complementary signals: energy of the class logits and Mahalanobis distance of the projection to class-conditional Gaussians fitted *per K*.
- Per-K threshold `θ_reject(k)` defined as a quantile of `u_k` on known validation flows (a target in-distribution false-positive rate), so that one operating point means the same FPR at every K.
- Evaluation of unknown-application detection (D1 open-set split), cross-protocol unknowns (D2), and anomalous/malicious traffic (D4 malware as anomalies).
- A simple anomaly explanation for the dashboard: which packets / attributes deviate most (from the NPP head's surprise).

## Non-goals

- Discovering and naming new classes (clustering of rejected flows is a stretch goal, spec 012).
- Signature-based malware detection.

## Method

### Signals at prefix k

1. Energy: `E_k = −T · logsumexp(cls_k / T)` with the per-K temperature from spec 008. Higher energy = less evidence for any known class.
2. Relative Mahalanobis: fit per K on training flows class-conditional Gaussians on `proj_k` (128-d, shared covariance per K, shrinkage 0.1) and a background Gaussian; `M_k = min_c d_c(k) − d_bg(k)` (relative Mahalanobis, as in the 2022 UQ-for-traffic work).
3. Surprise: mean NPP negative log-likelihood of packets 2..k (how unlike known traffic the sequence *dynamics* are). Cheap because the head exists anyway.

`u_k = z(E_k) + z(M_k) + β·z(S_k)` where `z(·)` standardises each signal with per-K mean/std from known val flows; β = 0.5, ablated. The combination is a fixed rule, not learned, to keep the score interpretable and robust under drift; a learned combiner is an ablation.

### Per-K thresholds

`θ_reject(k) = Quantile_{1−f}(u_k | known val flows, k)` for a target in-distribution FPR `f` (default 0.02). The controller (spec 011) may move `f` to trade rejection rate against coverage.

### Anomaly mode (D4)

Same score; positives = malware flows, negatives = benign test flows. Additionally the flow-statistics XGBoost baseline provides a tabular anomaly baseline (isolation forest on flowstats) for the report.

### Explanation

For a rejected flow, per-packet surprise `−log p_NPP(packet_j)` highlights which packets were unexpected; the dashboard shows this as a bar over the packet timeline.

## Inputs and outputs

- Inputs: per-K `cls`, `proj`, NPP log-likelihoods (online or saved); Gaussian parameters and thresholds fitted on train/val and stored with the checkpoint.
- Outputs: `u_k`, `θ_reject(k)`, per-packet surprise; open-set metrics via spec 004.

## Edge cases

- Classes with < 50 training flows: no class Gaussian; they contribute to the background only.
- K < 3: scores are near-uninformative; `k_ood = 3` in spec 009 guards; AUROC(K) curves start at K=1 anyway to show this.
- Covariance singularity in fp16: fit in fp64 on CPU from saved projections.
- Drift shifts the score distribution: thresholds are per-K quantiles on val; spec 012 monitors the realised rejection rate and can re-fit quantiles on recent *unlabeled* known-committed flows (label-free re-calibration, unlike CAPE-Net's labelled repair).

## Performance considerations

- Mahalanobis per K needs 30 (128x128) precision matrices and 180x30 means: 15 MB; scoring is one matmul per packet step.
- Energy and surprise are free.

## Testing

- Unit tests: quantile thresholds give the target FPR on val by construction; Mahalanobis of a class mean is zero; standardisation uses val stats only.
- Synthetic test: flows with shuffled packets or inflated sizes get higher `u_k` than clean flows.
- Open-set split test: no unknown class in train shards.

## Interactions

- Spec 009 consumes `u_k` and `θ_reject(k)`; spec 011 moves `f`; spec 012 monitors rejection rate and re-fits thresholds; spec 017 shows alerts and explanations.

## Success criteria (H3 acceptance)

- On D1 open-set test-ID: AUROC of `u_k` at K=8 >= 0.85 and at K=30 >= 0.90 (CAPE-Net reports 0.83 to 0.91 at its fixed checkpoint, different split; we report the full curve).
- Mean rejection K for unknown flows at least 30% lower than the fixed K=10 energy gate at equal FPR.
- D4: malware AUROC >= 0.90 with a D1-trained model or, if the domain gap is too large, with a D3-benign-trained model (both reported).

## Open questions

- Whether the combined score should be learned (small logistic regression on val with synthetic unknowns as positives); default rule-based.
- Rejection-first vs commit-first ordering (spec 009) interacts with FPR; both reported.
