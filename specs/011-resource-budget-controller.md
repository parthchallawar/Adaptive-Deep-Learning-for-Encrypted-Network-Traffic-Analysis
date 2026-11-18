# Spec 011: Automatic Resource Budget Controller

- **Status:** implemented and run (phase 4, 2026-11-18): `inference/controller.py`, `evaluation/controller_replay.py`; three seeds. Verdict: tracking error within 5% on the drift weeks and fast recovery from a target step, but a static threshold already holds the budget under this mild drift; see `docs/results-phase-4.md`.
- **Owner:** Parth Challawar
- **Created:** 2026-09-17
- **Build step:** step 13 of 18
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

- Offline replay on saved scores (spec 008) over val → test-ID → drift weeks in time order, with synthetic disturbances: (a) class-mix shift (resample flows so that hard classes double), (b) burst of unknowns (inject D1 unknown-split flows at 10%), (c) step change of `B*` from 8 to 5 to 8.
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

- Over the drift weeks, mean tracking error <= 5% with the controller versus >= 15% for the static threshold (if the static error is smaller, report it honestly and analyse why).
- No oscillation in any replay; settling time after a step change <= 5 windows.

## Implementation contract (added 2026-11-17, before the build)

Build plan: [`plans/phase-4-unknown-stopping-budget.md`](../plans/phase-4-unknown-stopping-budget.md), task T5. Where this section and the text above disagree, this section wins (corrections C6, C7 of that plan).

**Scope changes.**

- The "compute" budget metric is **descoped**: depth exits (spec 006, optional) were never built, so compute is proportional to packets read. Metrics are `mean_k` and `p95_k`.
- The labelled-canary alert moves to spec 012; it needs labels and belongs with drift.

**Modules.** `src/adl_etc/inference/controller.py` (pure Python and NumPy, no torch), `src/adl_etc/evaluation/controller_replay.py`, `configs/controller/default.yaml`.

```python
class BudgetController:
    def __init__(self, target, *, metric="mean_k", kp=0.5, ki=0.05, i_max=5.0,
                 theta_min=0.5, theta_max=0.999, window=2000, min_flows=200, theta_init): ...
    theta: float
    def update(self, window_k: np.ndarray) -> ControllerEvent: ...   # one call per closed window
    def set_target(self, target: float) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
```

- A window with fewer than `min_flows` flows produces no update.
- Every update appends a `ControllerEvent(window, observed, error, integral, theta, reason)`.
- Oscillation guard: the sign of the error alternating for more than 5 windows with amplitude above 5% halves both gains and logs it.
- An infeasible target saturates `θ` at a rail with a bounded integral and one "budget infeasible" event.
- The rejection-rate cap is a second instance acting on `logit(f)` with `kp = 0.2`, `ki = 0.02`, `f` in [0.001, 0.2].
- For `p95_k`, the controlled knob is `k_cap`, lowered one packet per window while the window's p95 exceeds the target.

**Replay.** Flows in `(period, flow_index)` order over `val → test_id → test_drift`, read from score files. Per window: `replay.safe_stop` on that slice with the current knobs, then `update`. `theta_init` is found by bisection on `val` so that mean K equals the target. The static baseline holds `theta_init`.

**Tuning.** Gains may be tuned on the `val` period only, over `kp ∈ {0.5, 1.0, 1.5, 2.0}`, `ki ∈ {0, 0.02, 0.05}`, with `i_max = 1.0`.

**Correction C9 (2026-11-17, found while testing the built controller; applied in `inference/controller.py`).** The update is in velocity form, so `kp` already integrates the error and `ki·I` integrates it a second time. With the text's `kp = 0.5, ki = 0.05, i_max = 5` a large first error winds `I` up and keeps pushing `θ` after the error is gone: on a synthetic plant with a known `θ → mean K` map the worst error after window 10 was 37% and a target step took 16 windows to settle. The code's defaults are `kp = 1.0, ki = 0.02, i_max = 1.0` (2% worst error, 3 windows, no oscillation on the same plant). Settling time falls with `kp` (3 and 6 windows at 1.0, 1 and 2 at 2.0 on the replay fixture), so the "≤ 5 windows" criterion is a statement about gains tuned on real validation data, not about any fixed default.

**Metric definitions.**

| Metric | Definition |
|---|---|
| Tracking error | mean of `|B_obs − B*| / B*` over windows, after a 5-window warm-up |
| Within-5% fraction | share of those windows with error ≤ 0.05 |
| Settling time | windows after a target step until the error stays ≤ 0.05 for 3 consecutive windows |
| Accuracy at budget | known-flow accuracy over the same windows |

**Disturbances.** (a) hard classes doubled, where "hard" is the bottom quartile of per-class accuracy at K = 5 on `val`; (b) an unknown burst: the replayed stream is already about 10% unknown-class (25,308 of 250,000 flows in `test_id`), so the share is raised to 30% for 20 windows; (c) target stepped 8 → 5 → 8. Reference replay loop: `docs/design-phase-4.md` section 4.

## Open questions

- Window in flows vs. time (default: flows, since evaluation is replayed).
- Whether p95-K control is needed for the report or only E[K] and compute (default: E[K] and compute in the paper; p95 in the demo).

## Corrections made during phase 4 (2026-11-18)

- **C6.** The "compute" budget metric is descoped (depth exits were never built); the controller targets mean K and p95 K.
- **C7.** The canary-accuracy alert moves to spec 012 (it needs labels and belongs with drift).
- **C9.** Defaults are `kp = 1.0, ki = 0.02, i_max = 1.0` (the spec's 0.5 / 0.05 / 5 wind up in velocity form); on real data the gains are searched on validation only, over `kp` up to 6 (the first search stopped at 2.0, the edge of its grid; every run picked `kp = 3` once it was widened).
- **C10.** Settling time is reported two ways: the strict count (3 consecutive windows each within 5%) and a noise-robust count (mean signed error of 3 windows within 5%). The second was added after the first was seen to measure window noise, not the controller; both are shown.
