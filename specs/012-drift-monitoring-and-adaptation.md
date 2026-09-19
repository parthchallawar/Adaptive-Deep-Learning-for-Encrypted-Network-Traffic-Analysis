# Spec 012: Drift Monitoring and (Optional) Label-Free Adaptation

- **Status:** draft (monitoring required; adaptation is a stretch goal)
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 14 of 18
- **Depends on:** 004, 007, 008, 010, 011. **Used by:** 016, 017.

## Problem

Traffic changes over weeks (app updates, CDNs, new services). The year-long evaluation (spec 004) *measures* that decay, but a deployed system also needs to *notice* it without labels and, ideally, to slow it cheaply. CAPE-Net monitors per-class exit depth and precision with CUSUM and repairs with labelled flows. We want label-free signals derived from what we already compute, and, if time permits, a label-free repair step that uses the SSL objective.

## Goals (required)

- Online drift signals, all label-free: mean K and commit rate per window (policy behaviour), rejection rate (spec 010), mean commit-safety `p_safe` at decision, NPP perplexity (spec 007), and class-prior shift (KL between the committed-class histogram and the val histogram).
- Change detection per signal with CUSUM/Page-Hinkley and a combined drift level {none, warning, alarm}.
- Dashboard panel and API endpoint exposing the signals and alarms.
- Offline validation that the signals correlate with the measured accuracy decay across the drift weeks.

## Goals (stretch, "adaptation")

- Label-free re-calibration: refit per-K temperatures, isotonic tables and unknown-score quantiles on recent confidently-committed flows (pseudo-labels with `p_safe >= 0.99`), guarded by a holdout check.
- Label-free backbone refresh: a few epochs of the SSL objective (spec 007) on recent unlabeled flows with the classifier heads frozen, then re-calibration; evaluate whether it recovers part of the drift loss. This is exactly the "partially self-supervised label sources for repair" that CAPE-Net names as future work.
- Clustering of rejected flows (HDBSCAN on `proj`) to surface candidate new applications for labelling.

## Non-goals

- Supervised retraining pipelines, active learning UI.

## Method

### Signals per window (W = 5000 flows or 60 s)

| Signal | Source | Expected drift response |
|---|---|---|
| `meanK` | policy | rises (harder traffic) |
| `commit_rate` | policy | falls |
| `reject_rate` | spec 010 | rises (new apps) |
| `mean_safe_at_commit` | spec 008 | falls |
| `npp_ppl` | spec 007 head | rises (unfamiliar dynamics) |
| `prior_kl` | committed class histogram vs val | rises |

### Detection

Each signal standardised by its val-period mean/std; CUSUM with threshold h = 5 and drift allowance 0.5 (tuned on val by requiring zero alarms). Combined level: warning if any one signal alarms, alarm if two or more alarm within 3 windows. Alarms are logged with the contributing signals.

### Validation offline

Replay the drift weeks in time order; plot each signal against the measured weekly macro-F1; report Spearman correlation and the lead time of the first alarm relative to a 3-point F1 drop.

### Adaptation (stretch)

1. Collect the last N = 200k flows (unlabeled).
2. Re-calibration: pseudo-labelled subset → refit temperatures/isotonic/quantiles; accept only if the NLL on a held-back 20% of the pseudo-labelled set improves.
3. SSL refresh: 2 epochs of NPP+PFC with heads frozen and LR 1e-4; accept only if `npp_ppl` on held-back flows improves and the pseudo-label agreement between old and new model exceeds 98%.
4. Evaluate on the following weeks with true labels (offline only) to report the gain.

## Inputs and outputs

- Inputs: the decision stream and scores; val-period statistics; recent shards for adaptation.
- Outputs: signal time series, alarms (DB table, spec 018), updated calibration artefacts (versioned).

## Edge cases

- Budget controller (spec 011) actively changes `meanK`; the monitor uses θ-normalised signals (`p_safe` at commit, `npp_ppl`, `prior_kl`) as primary and treats `meanK` as informational when the controller is active.
- Pseudo-label confirmation bias: acceptance gates above; never adapt the classifier head from pseudo-labels.
- Alarm storms at start-up: warm-up of 5 windows.

## Performance considerations

- Signals are aggregates of quantities already computed; negligible cost. SSL refresh runs offline (Kaggle or local GPU), not in the service.

## Testing

- CUSUM unit tests with synthetic step changes (detects within k windows, no false alarms on stationary noise).
- Replay test on the drift weeks: at least one signal alarms well before the end of the horizon (the dataset authors report a measurable drop by T+8 weeks).
- Adaptation tests: acceptance gates reject a deliberately corrupted refresh.

## Interactions

- Consumes 007/008/010/011 outputs; writes alarms to 018; shown in 017; exposed by 016.

## Success criteria

- Required: drift signals implemented, validated offline with correlation plots, shown in the dashboard.
- Stretch: label-free adaptation recovers >= 25% of the drift-induced F1 loss on at least one drift week, or a negative result is reported.

## Open questions

- Whether the refresh is worth its GPU cost in the final weeks; decide after phase 3 results.
