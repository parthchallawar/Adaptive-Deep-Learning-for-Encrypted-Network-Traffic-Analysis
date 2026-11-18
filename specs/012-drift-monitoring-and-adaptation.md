# Spec 012: Drift Monitoring and (Optional) Label-Free Adaptation

- **Status:** monitoring implemented and run (phase 5, 2026-11-18): `inference/drift.py` (torch-free), `evaluation/{drift_curves,drift_monitor}.py`; three seeds, both threshold modes. Class-prior KL and next-packet NLL track the weekly macro-F1 (Spearman about -0.9); mean K and reject rate do not; every run warns within the horizon, but no signal warns before a 3-point drop without also firing early in the test weeks. Adaptation (stretch) not built; dashboard panel is phase 6. See `docs/results-phase-5.md`.
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

## Pre-build notes (added 2026-11-17)

Build plan (forward draft): [`plans/phase-5-drift-efficiency.md`](../plans/phase-5-drift-efficiency.md), tasks T1 and T2.

- **Only monitoring is planned.** The adaptation goals above are not scheduled (roadmap DP-5a); they are kept in `docs/future-work.md` FEP-3.
- **Module:** `src/adl_etc/inference/drift.py`, torch-free: `Cusum`, `WindowStats`, `DriftMonitor`. Windows of 5,000 decided flows. Reference means and standard deviations come from replaying `val` in windows; `h` is raised until `val` gives zero alarms.
- **Inputs are phase 4's score files** (which carry each flow's period), replayed through `evaluation/replay.py`. No model is run.
- **`npp_ppl` is implemented as the mean next-packet negative log-likelihood** over the packets read (`mean_pkt_nll`); perplexity is its exponential and adds nothing.
- **`prior_kl`** smooths the committed-class histogram by adding 1e-3 to every class before normalising.
- **Two signal sets.** With a static threshold, all six signals are primary. With the controller running, `mean_k` is held by design, so the primary signals are `mean_safe_at_commit`, `mean_pkt_nll`, `prior_kl`, `reject_rate`, and **`theta_commit` itself**: a drifting threshold is the controller reporting that traffic changed.
- **Labelled canary** (moved here from spec 011): a fixed 5,000-flow labelled validation sample replayed every 50 windows; alert if accuracy falls more than 5 points below its validation value. Optional.
- **Weekly decay curves** (accuracy and macro-F1 at K = 5, 8, 30 and at the policy's decisions, per drift week) are computed first, in `evaluation/drift_curves.py`; the signals are validated against them.

## Open questions

- Whether the refresh is worth its GPU cost in the final weeks; decide after phase 3 results.

## Corrections made during phase 5 (2026-11-18)

- **C11.** `npp_ppl` is implemented as the mean next-packet negative log-likelihood over the packets read (`mean_pkt_nll`); perplexity is its exponential and adds nothing. `prior_kl` adds 1e-3 to each class *frequency* (not count) before renormalising, both for the window and for the validation histogram.
- **C12.** Two signal sets. With a static threshold all six signals are watched; with the budget controller running `mean_k` and `commit_rate` are informational and `theta_commit` itself is watched.
- **C13.** The CUSUM threshold `h` is the smallest value from 5.0 (steps of 0.5) at which validation windows give no alarm; the warm-up is 5 windows; a window is 5,000 decided flows.
- **C14.** Adaptation (stretch) is not built (roadmap DP-5a); the canary-accuracy alert (moved here from spec 011) is not built either. Both are in `docs/future-work.md`.
- **C15.** The offline validation cannot measure a false-alarm rate: there is no stationary stretch of labelled data, and the `test_id` weeks already differ slightly from validation. The results document reports sensitivity only.
