# Spec 009: Adaptive Stopping Policy (continue / commit / reject)

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Depends on:** 006, 008, 010. **Used by:** 011, 016. **Research contribution:** C2 (hypothesis H2, part of H3).

## Problem

After each packet the system must decide whether to keep reading, to commit to a class, or to declare the flow unknown. Prior systems use a global or per-class threshold on the maximum class probability (ECHO, CAPE-Net) or a reinforcement-learning policy (FastFlow). Thresholds on max-prob conflate "the classifier is unsure because two classes look alike at this prefix" with "the next packets will resolve it"; RL policies are expensive to train and unstable under drift. We want a decision rule that is learned, calibrated per prefix length, cheap, interpretable, and controllable by external knobs (spec 011).

## Goals

- A three-way decision rule evaluated after every packet, driven by two calibrated scores: commit-safety `p_safe(k)` (spec 008) and unknown score `u(k)` (spec 010).
- Two knobs, `θ_commit` and `θ_reject`, that the controller can move at run time, with monotone, predictable effects.
- An ELECTS-style end-to-end halting objective as an alternative policy for the ablation, and the ECHO / CAPE / RL baselines for comparison.
- Offline replay: any policy can be evaluated on saved per-K scores without re-running the model.

## Non-goals

- Learning the policy jointly with the controller; per-class thresholds as the primary mechanism (kept as a baseline).

## Decision rule (primary: "Safe-Stop")

For a flow at prefix k with class distribution `π_k` (temperature-calibrated), safety `s_k = p_safe(k)`, unknown score `u_k`, and minimum lengths `k_min = 2`, `k_ood = 3`:

```
if k >= k_ood and u_k >= θ_reject(k):            REJECT (unknown)      # spec 010 defines θ_reject(k) as a per-K quantile
elif k >= k_min and s_k >= θ_commit:             COMMIT argmax π_k
elif k == min(K_max, ppi_len):                   COMMIT argmax π_k  if u_k < θ_reject(k) else REJECT
else:                                            CONTINUE
```

Order matters: rejection is checked first so that an unknown flow is not committed by a confidently wrong classifier (CAPE-Net's "confidence-first gate" does the opposite; both orders are evaluated and the difference reported, since it is a genuine design choice).

Why `p_safe` instead of max-prob: `p_safe(k)` is trained to predict correctness at this K and is isotonic-calibrated per K, so a single global `θ_commit` means "commit when the estimated error rate is below 1 − θ_commit" at every K. The same threshold on max-prob would mean different error rates at K=3 and K=20.

### Value-of-waiting feature (part of C2)

`safe` receives the next-packet predictive entropy from the SSL head (spec 006). Intuition: if the model can predict the next packet well, the future carries little new information and committing now loses nothing; if it cannot, waiting is likely to change the decision. This is tested by ablating the feature.

### Alternative policy A: learned halting (ELECTS-style)

A halting head `h_k = σ(·)` defines a stopping distribution `P(stop at k) = h_k Π_{j<k}(1 − h_j)`; loss `Σ_k P(stop at k)·[CE_k + α·k/30]`, α trades earliness for accuracy and must be retrained per operating point. Included as the "end-to-end" comparison; the primary rule is preferred because it needs no retraining to move along the trade-off (the controller can act at run time).

### Alternative policy B: RL stopper (FastFlow-style)

Spec 005's P-RL.

### Baselines: P-ECHO, P-CAPE (spec 005).

## Trade-off curves

Sweeping `θ_commit` in [0.5, 0.999] gives the (mean K, accuracy) Pareto front; sweeping `θ_reject` gives (unknown TPR, FPR) at each prefix. All curves computed offline from saved scores (spec 008 output), which makes policy research cheap after one model run.

## Inputs and outputs

- Inputs: per-flow streams of `(π_k, s_k, u_k, ppi_len)` from the model (online) or saved scores (offline); knobs from the controller; `configs/policy/*.yaml`.
- Outputs: decision, decision K, class, scores at decision time; `PolicyReport` for spec 004.

## Edge cases

- `ppi_len < k_min`: forced decision at the last available packet.
- `θ_commit` > max reachable `s_k`: no early commits; coverage metric shows it; controller guards against this (spec 011).
- Ties in `argmax π_k`: lowest class index (deterministic).
- Score NaNs (numerical issues): treated as `s=0, u=+inf` → the flow continues to K_max then REJECTs; counted and alerted.

## Performance considerations

- The rule is O(1) per packet; offline replay of 1M flows x 30 K is vectorised NumPy (seconds).

## Testing

- Monotonicity tests: raising `θ_commit` never decreases mean K; raising `θ_reject` never increases rejection rate.
- Order-of-checks test with synthetic scores.
- Replay equivalence: online decisions on a fixture batch equal offline replay decisions.
- ELECTS loss unit test: stopping distribution sums to 1 over valid positions.

## Interactions

- Spec 011 moves `θ_commit`/`θ_reject`; spec 010 defines `u_k` and `θ_reject(k)`; spec 016 runs the rule per `PacketEvent`; spec 017 visualises decisions; spec 004 reports.

## Success criteria (H2 acceptance)

- At mean K = 6 on D1 test-ID, Safe-Stop accuracy exceeds P-ECHO by >= 1.5 points and matches or beats P-CAPE and P-RL; Pareto-front area larger than all baselines with the same backbone.
- Ablation shows the next-packet-entropy feature contributes (>= 0.5 point at mean K = 4 to 6) or the negative result is reported.
- Rejection-first ordering reduces confident misclassification of unknowns (open-set F1) without delaying known flows by more than 0.5 packets on average.

## Open questions

- `k_min`/`k_ood` defaults (2/3) to be validated on val; CAPE-Net uses K>=10 for OOD, our per-K calibration is meant to allow earlier rejection.
