# Spec 011: Automatic Resource Budget Controller

- **Status:** draft
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Depends on:** 006, 009, 010. **Used by:** 012, 016, 017. **Research contribution:** C3 (hypothesis H4).

## Problem

An operator does not tune thresholds; they state a budget: "read at most 6 packets per flow on average", "keep CPU at 70%", "decide within 200 ms at the 95th percentile". A fixed threshold that meets a budget on the validation week violates it weeks later because drift changes the confidence distribution (harder traffic makes the model wait longer, eating the budget). The synopsis asks for an "automatic resource controller that dynamically adjusts packet budget and inference effort based on flow difficulty". This spec defines that controller as a feedback loop on the policy knobs.

## Goals

- Hold a target on one of: mean packets per flow E[K], mean compute per flow (packets x layers, spec 006 cost model), p95 packets, or a rejection-rate cap; while maximising accuracy.
- Run online over a stream of flows with no labels.
- Be stable (no oscillation), bounded (knobs within safe ranges), and auditable (log every adjustment).
- Support a manual override and a "quality-first" mode (fixed θ, no control) for comparison.

## Non-goals

- Learning a controller with RL; scheduling across multiple models or machines; per-class budgets.

## Method

### Control variable and measurement

- Window: last W = 2000 committed-or-rejected flows (or 10 s, whichever first).
- Measured `B_obs` = mean K in the window (or mean cost, or p95 K, per config).
- Target `B*` from config or the API (spec 016).

### Controller: proportional-integral on the commit threshold

Let `θ` be `θ_commit` (spec 009) mapped through the logit function so that updates are unbounded: `φ = logit(θ)`.

```
e_t = (B_obs − B*) / B*                       # positive = over budget
I_t = clip(I_{t−1} + e_t, −I_max, I_max)      # anti-windup
φ_{t+1} = φ_t − (K_p · e_t + K_i · I_t)       # over budget → lower θ → commit earlier
θ_{t+1} = clip(sigmoid(φ_{t+1}), θ_min, θ_max)
```

Gains `K_p = 0.5, K_i = 0.05` tuned on the val period by minimising tracking error subject to no oscillation (spectral check on the θ trajectory). Equivalent view: dual ascent on the Lagrangian `accuracy − λ·(E[K] − B*)`, which is why moving a single calibrated threshold is the right knob: `p_safe` is the marginal accuracy of committing, so a threshold on it is the Lagrangian-optimal rule for a mean-cost constraint.

### Secondary knobs

- Rejection FPR `f` (spec 010): controlled toward a rejection-rate cap `R*` with the same PI form, lower gains; prevents drift-induced rejection storms.
- Depth exit threshold (optional, spec 006): controlled toward a compute target when `budget_metric = compute`.

### Difficulty-aware admission (per-flow effort)

Beyond the global knob, the policy already allocates effort per flow (easy flows commit at K=3, hard ones read 30). The controller adds a *hard cap* `K_cap` (default 30) lowered when p95 K is the target: flows still undecided at `K_cap` are force-decided. This is the "adjust packet budget based on flow difficulty" behaviour in the synopsis: the cap binds only for the hardest flows.

### Safety rails

- `θ_min = 0.5` (never commit on a coin flip), `θ_max = 0.999`.
- Freeze control if the window has fewer than 200 flows.
- Rate limit: at most one update per window.
- If accuracy on a small labelled canary set (optional, from the val period replayed) drops more than 5 points below the operating curve's expectation, raise an alert (spec 012) but do not change control law.

## Evaluation (H4)

- Offline replay on saved scores (spec 008) over val → test-ID → drift months in time order, with synthetic disturbances: (a) class-mix shift (resample flows so that hard classes double), (b) burst of unknowns (inject D1 unknown-split flows at 10%), (c) step change of `B*` from 8 to 5 to 8.
- Metrics: absolute tracking error, fraction of windows within 5% of `B*`, settling time after a step, accuracy vs budget curve, comparison against static θ tuned on val.
- Online run in the service (spec 016) on a replayed pcap to verify latency of the loop.

## Inputs and outputs

- Inputs: decisions stream `(K, cost, decision)`, config (`budget_metric`, `B*`, gains, rails), optional canary labels.
- Outputs: current knobs, controller state, event log (timestamp, e_t, θ, reason), metrics for spec 004.

## Edge cases

- Target unreachable (e.g. `B* = 1.5` when minimum K is 2): θ saturates at `θ_min`, `I_t` saturates, alert "budget infeasible".
- Empty or tiny windows at start: hold θ at the val-tuned initial value.
- Sudden traffic stop: timer-based window closes; no update if < 200 flows.
- Oscillation detected (sign of e_t alternates > 5 windows with amplitude > 5%): halve gains and log.

## Performance considerations

- O(1) per flow; state is a few floats; the window is a ring buffer.

## Testing

- Simulation tests with a synthetic score generator where the true (θ → E[K]) map is known: controller converges within 10 windows, tracking error < 3%.
- Anti-windup test with an infeasible target.
- Monotonic knob test: lowering θ never increases E[K] in replay.
- Step-response test with plots checked into `tests/fixtures/plots/` (visual regression optional).

## Interactions

- Reads spec 009/010 knobs; writes to them; spec 012 receives alerts; spec 016 exposes `GET/POST /budget`; spec 017 shows the budget dial and trajectories.

## Success criteria

- On the drift months, mean tracking error <= 5% with the controller versus >= 15% for the static threshold (if the static error is smaller, report it honestly and analyse why).
- No oscillation in any replay; settling time after a step change <= 5 windows.

## Open questions

- Window in flows vs. time (default: flows, since evaluation is replayed).
- Whether p95-K control is needed for the report or only E[K] and compute (default: E[K] and compute in the paper; p95 in the demo).
