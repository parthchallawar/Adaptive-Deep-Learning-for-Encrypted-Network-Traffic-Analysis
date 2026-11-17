# Spec 009: Adaptive Stopping Policy (continue / commit / reject)

- **Status:** draft; not built. Implementation contract added 2026-11-17 (see that section; it overrides older text where they differ)
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 12 of 18 (after 010)
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

## Implementation contract (added 2026-11-17, before the build)

Build plan: [`plans/phase-4-unknown-stopping-budget.md`](../plans/phase-4-unknown-stopping-budget.md), tasks T3 and T4. Where this section and the text above disagree, this section wins (corrections C2, C4, C5 of that plan).

**Modules.**

| File | Holds | Imports torch? |
|---|---|---|
| `src/adl_etc/inference/policy.py` | `Decision`, `decide(...)`: the rule for one flow at one K. The service calls this | no |
| `src/adl_etc/evaluation/replay.py` | `safe_stop`, `echo`, `cape`: the same rules vectorised over a `ScoreArrays` | no |
| `src/adl_etc/evaluation/rl_stop.py` | P-RL: `fit(scores_val, lam)`, `apply(model, scores)` | no |
| `configs/policy/safe_stop.yaml` | `k_min: 2`, `k_ood: 3`, `order: reject_first`, `f: 0.02`, `k_cap: 30` | |

**Inputs.** `p_safe` is `Calibration.safe_probability(safe)` (already built, per-K isotonic). `u_k` and `θ_reject(k)` come from spec 010's `UnknownModel`. All of them are read from the per-run score files, never from a live model, in offline work.

**Decision codes.** `0` commit, `1` reject, `2` forced commit, `3` forced reject. A forced decision happens at `L = min(k_cap, ppi_len)` when nothing fired earlier: reject if `u_L ≥ θ_reject(L)`, else commit. As in the rule above, the forced branch ignores `k_ood`, so it is the only way a flow shorter than `k_ood` packets is rejected. Reference code: `docs/design-phase-4.md` section 2. NaN inputs count as `p_safe = 0`, `u = +inf`, and are counted in the result.

```python
def decide(k, p_safe_k, u_k, pred_k, *, theta_commit, theta_reject_k, L, k_min=2, k_ood=3, order="reject_first") -> Decision | None
def safe_stop(scores, theta_commit, f, unknown_model, *, k_min=2, k_ood=3, order="reject_first", k_cap=30) -> ReplayResult
```

**Baselines.**

- P-ECHO and P-CAPE: score-based forms in `replay.py`, which must equal `evaluation/policies.py`'s `p_echo` / `p_cape` on a fixture built from real logits.
- **P-RL** is finite-horizon optimal stopping by backward induction on `(conf_k, margin_k)`, binned 50 × 50 per K, fitted on known validation flows; `λ` sweeps the trade-off. It is our same-backbone learned stopper, **not** a re-implementation of FastFlow (no LSTM state, no online Q-learning). Say so wherever it is named.
- The ELECTS halting head (alternative policy A) is a stretch item (~3 GPU-hours). If it is not run, it is marked descoped here with that reason.

**Two evaluations**, both on `test_id`, three seeds:

1. *Closed world* (H2's main test): known flows only, rejection off. 200 values of each policy's knob, evenly spaced in logit space for `θ_commit` over [0.5, 0.999]. Report accuracy at mean K = 4, 6, 8 (the best accuracy among sweep settings whose mean K is at most that budget) and `pareto_area` (the mean of that best-within-budget accuracy over 201 budgets from 2 to 12 packets; 0 where a policy cannot operate).
2. *Open world*: all flows, `f = 0.02`. Report open-set macro-F1 ("unknown" as one extra class), the fraction of unknown flows committed, and the mean extra packets for known flows, for both check orders. A rejected known flow counts as an error.

The closed-world sweep also measures spec 007's third H1 criterion: packets-to-decision at equal accuracy for the SSL model against the scratch model.

**Value-of-waiting ablation.** First knock the next-packet-entropy feature out at inference (feed its training mean). If accuracy at mean K 4–6 moves by less than 0.2 points, report "no measurable contribution" and stop. Otherwise retrain only the `safe` head with the backbone frozen, with and without the feature.

**Defaults are fixed before any test number is seen.** `k_min`, `k_ood` and the check order are not selected on `val`'s unknown flows (the same classes are unknown in test); their alternatives are reported as sensitivity.

## Open questions

- `k_min`/`k_ood` defaults (2/3) to be validated on val; CAPE-Net uses K>=10 for OOD, our per-K calibration is meant to allow earlier rejection.
