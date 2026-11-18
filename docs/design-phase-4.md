# Phase 4 design note: reference algorithms

- **Written:** 2026-11-17. **Goes with:** [`plans/phase-4-unknown-stopping-budget.md`](../plans/phase-4-unknown-stopping-budget.md) (what to build, in what order) and specs 010, 009, 011 (why).
- **What this is:** the four pieces of phase 4 where the arithmetic is easy to get subtly wrong, written out. The code blocks are **reference sketches**. Sections 1 to 3 were run once on random data on 2026-11-17, with these results: the vectorised stopping rule gave the same decisions as a plain per-flow loop over 4,000 flows in 72 settings (both check orders, three `k_min`/`k_ood` pairs, two caps, NaN scores included); mean K never fell when `θ_commit` rose; the all-classes Mahalanobis formula equalled a per-class loop and gave 0 at a class mean; the running sums reproduced the pooled within-class covariance; P-RL's mean K fell steadily as `λ` rose. Section 4 (controller replay) was not run, and nothing here has been run on real data. Turn each of those checks into a real test when the module is built. The tests listed in the plan and in `docs/validation-strategy.md` decide what is correct; if a sketch and a test disagree, work out which one is wrong before changing either.
- **Conventions:** arrays are `[N, 30]` with index `k-1` = "after `k` packets", already filled forward past each flow's `ppi_len`. `K` values in code are 1-based. `labels == -1` marks an unknown-class flow. Every flow has `ppi_len ≥ 1` (assert it).

## 0. Run order

The unknown score needs a pass over training flows before the score passes, so the order of *runs* differs slightly from the order of *tasks*:

```
1. train pass      known-class training sample ──► GaussianBank        (section 1)
2. score passes    val, test_id, test_drift, with the GaussianBank ──► ScoreArrays (incl. mahal)
3. fit thresholds  known val flows of the val ScoreArrays ──► UnknownModel = GaussianBank + z-stats + quantiles
4. everything else replays from ScoreArrays + UnknownModel (no model, no torch)
```

`u` is not stored. It is computed when needed from `energy`, `mahal` and `pkt_nll` with the `UnknownModel` (it depends on `β`, which the sensitivity sweep changes).

## 1. Mahalanobis from running sums

**Why sums.** `proj` is `[N, 30, 128]`; storing it for 250,000 flows is ~1.9 GB in fp16. The Gaussians need only first and second moments.

**Before the model runs**, the class counts are already known from the sample's labels and lengths:

```python
# y: [n] model class index of each sampled known-class training flow; plen: [n]
reached = plen[:, None] >= np.arange(1, 31)[None, :]                 # [n, 30]
n_c = np.zeros((30, C), np.int64)
for k in range(30):
    n_c[k] = np.bincount(y[reached[:, k]], minlength=C)
valid = n_c >= 50                                                     # [30, C] classes that get a Gaussian at this K
```

**During the pass**, per chunk (`x = proj` of the chunk, float64, `[B, 30, 128]`), for each `k`:

```python
m = reached_chunk[:, k]                      # flows that have k+1 packets
xk, yk = x[m, k], y_chunk[m]
np.add.at(sum_c[k], yk, xk)                  # [C, 128]
sum_all[k] += xk.sum(0)
outer_all[k] += xk.T @ xk
v = valid[k, yk]                             # only flows whose class has a Gaussian at this K
outer_valid[k] += xk[v].T @ xk[v]
```

`outer_valid` must exclude rare-class flows, or their scatter leaks into the shared covariance with no mean subtracted for them.

**After the pass**, per `k` (all float64):

```python
cv = valid[k]
mu = sum_c[k][cv] / n_c[k][cv, None]                                  # [Cv, 128]
n_v = n_c[k][cv].sum()
S = (outer_valid[k] - (mu * n_c[k][cv, None]).T @ mu) / (n_v - cv.sum())
S = 0.9 * S + 0.1 * (np.trace(S) / 128) * np.eye(128)                 # shrinkage
P = np.linalg.inv(S)

n = reached[:, k].sum()
mu_bg = sum_all[k] / n
S_bg = outer_all[k] / n - np.outer(mu_bg, mu_bg)
S_bg = 0.9 * S_bg + 0.1 * (np.trace(S_bg) / 128) * np.eye(128)
P_bg = np.linalg.inv(S_bg)
```

`proj` is L2-normalised, so the points lie on a sphere and the raw covariance is close to singular in the radial direction. The shrinkage term is what makes the inverse safe; do not remove it.

**Scoring a batch at `k`** (`x`: `[B, 128]`):

```python
xP = x @ P
d = (xP * x).sum(1)[:, None] - 2.0 * (xP @ mu.T) + ((mu @ P) * mu).sum(1)[None, :]   # [B, Cv]
xb = x - mu_bg
M = d.min(1) - ((xb @ P_bg) * xb).sum(1)                                              # relative Mahalanobis
```

Tests that pin this down: `M` of a class mean has `d.min == 0`; the all-classes formula equals a plain per-class loop on 100 random points; fitting twice from the same sample gives identical arrays.

**z-statistics and quantiles** come from known validation flows (`labels != -1`) that reached `K`, from the validation `ScoreArrays`:

```python
for k in range(30):
    m = (labels != -1) & (ppi_len >= k + 1)
    for j, sig in enumerate((energy, mahal, surprise)):
        z_mean[k, j] = sig[m, k].mean()
        z_std[k, j]  = sig[m, k].std()
z_std = np.where(z_std < 1e-8, np.inf, z_std)      # surprise at K=1 is constant: its z becomes 0
u = ((energy - z_mean[:, 0]) / z_std[:, 0] + (mahal - z_mean[:, 1]) / z_std[:, 1]
     + beta * (surprise - z_mean[:, 2]) / z_std[:, 2])
quantiles[k] = np.quantile(u[m, k], np.linspace(0, 1, 1001))
```

`theta_reject(k, f) = np.interp(1 - f, np.linspace(0, 1, 1001), quantiles[k-1])`.

## 2. The stopping rule, vectorised

The rule for one flow (spec 009), with `L = min(k_cap, ppi_len)`:

```
for k = 1..L:
    reject_now = k >= k_ood and u_k >= theta_reject(k)
    commit_now = k >= k_min and p_safe_k >= theta_commit
    reject_first:  if reject_now → REJECT (1);  elif commit_now → COMMIT (0)
    commit_first:  if commit_now → COMMIT (0);  elif reject_now → REJECT (1)
    if k == L and nothing fired:  FORCED: REJECT (3) if u_L >= theta_reject(L) else COMMIT (2)
```

The forced branch follows the spec literally: it ignores `k_ood`. It is therefore the only way a flow shorter than `k_ood` packets can be rejected. For a flow with `L ≥ k_ood`, a reject at `L` is caught by the normal branch and gets code 1.

```python
def safe_stop(p_safe, u, pred, ppi_len, theta_commit, theta_reject, *,
              k_min=2, k_ood=3, order="reject_first", k_cap=30):
    # p_safe, u, pred: [N, 30]; theta_reject: [30]
    n = len(ppi_len)
    rows = np.arange(n)
    L = np.minimum(ppi_len, k_cap).astype(np.int64)                    # 1-based last usable K
    ks = np.arange(1, 31)[None, :]
    live = ks <= L[:, None]
    p = np.nan_to_num(p_safe, nan=0.0)
    uu = np.where(np.isnan(u), np.inf, u)

    can_reject = live & (ks >= k_ood) & (uu >= theta_reject[None, :])
    can_commit = live & (ks >= k_min) & (p >= theta_commit)

    def first(mask):                                                   # 1-based K of the first True, 31 if none
        return np.where(mask.any(1), mask.argmax(1) + 1, 31)

    k_rej, k_com = first(can_reject), first(can_commit)
    reject_wins = (k_rej <= k_com) if order == "reject_first" else (k_rej < k_com)
    k_early = np.minimum(k_rej, k_com)
    early = k_early <= 30

    k = np.where(early, k_early, L)
    forced_reject = ~early & (uu[rows, L - 1] >= theta_reject[L - 1])
    decision = np.where(early, np.where(reject_wins, 1, 0), np.where(forced_reject, 3, 2))
    prediction = np.where((decision == 1) | (decision == 3), -1, pred[rows, k - 1])
    return k, decision, prediction
```

Checks: `reject_wins` only matters where `early`; where both are 31 it is ignored. With rejection off, pass `theta_reject = np.full(30, np.inf)`. The equality test runs the per-flow loop above over 5,000 fixture flows and compares all three outputs.

**Metrics on the result** (`known = labels != -1`):

| Metric | Formula |
|---|---|
| mean K | `k.mean()` over the flows evaluated |
| known accuracy | `(prediction[known] == labels[known]).mean()` (a rejected known flow is wrong) |
| known FPR | `(prediction[known] == -1).mean()` |
| unknown TPR | `(prediction[~known] == -1).mean()` |
| unknown committed | `1 − unknown TPR` |
| open-set macro-F1 | macro-F1 over `C + 1` classes, mapping `-1` to class `C` in both labels and predictions (check `metrics.open_set_f1` first) |

**Accuracy at a budget, and the front's area.** For a sweep giving points `(mean_k_i, acc_i)`:

```python
def acc_at_budget(mean_k, acc, m):            # best accuracy among settings that read at most m packets on average
    ok = mean_k <= m
    return acc[ok].max() if ok.any() else 0.0

def pareto_area(mean_k, acc, lo=2.0, hi=12.0, n=201):
    return float(np.mean([acc_at_budget(mean_k, acc, m) for m in np.linspace(lo, hi, n)]))
```

"Accuracy at mean K = 4, 6, 8" in the plan means `acc_at_budget` at 4, 6, 8. A policy that cannot operate at a budget scores 0 there; report each policy's smallest reachable mean K next to its area so this is visible.

## 3. P-RL: stopping by backward induction

The learned-stopper baseline on the same backbone. State at `k`: the pair `(conf_k, margin_k)`, each cut into 50 equal bins on [0, 1], giving `b ∈ [0, 2500)`. Fitted on known validation flows; `λ` sets the price of a packet.

```python
def fit_rl(conf, margin, pred, labels, ppi_len, lam, n_bins=50, min_count=20):
    N = len(labels)
    L = np.minimum(ppi_len, 30)
    b = (np.minimum((conf * n_bins).astype(int), n_bins - 1) * n_bins
         + np.minimum((margin * n_bins).astype(int), n_bins - 1))            # [N, 30]
    correct = (pred == labels[:, None]).astype(np.float64)                   # [N, 30]
    S = n_bins * n_bins
    q_commit = np.zeros((30, S)); q_cont = np.full((30, S), -np.inf)
    v_next = None                                                            # V_{k+1} per flow
    for k in range(29, -1, -1):                                              # K = k + 1
        alive = L >= k + 1
        cnt = np.bincount(b[alive, k], minlength=S)
        qc = np.bincount(b[alive, k], weights=correct[alive, k], minlength=S) / np.maximum(cnt, 1)
        qc = np.where(cnt >= min_count, qc, correct[alive, k].mean()) - lam * (k + 1) / 30
        q_commit[k] = qc
        go_on = L > k + 1                                                    # flows that have a next packet
        if k < 29 and go_on.any():
            c2 = np.bincount(b[go_on, k], minlength=S)
            qn = np.bincount(b[go_on, k], weights=v_next[go_on], minlength=S) / np.maximum(c2, 1)
            q_cont[k] = np.where(c2 >= min_count, qn, v_next[go_on].mean())
        v_here = np.maximum(q_commit[k][b[:, k]], q_cont[k][b[:, k]])
        v_here = np.where(L == k + 1, q_commit[k][b[:, k]], v_here)          # last packet: must commit
        v_next = v_here
    return q_commit, q_cont

def apply_rl(q_commit, q_cont, conf, margin, pred, ppi_len, n_bins=50):
    n = len(ppi_len)
    L = np.minimum(ppi_len, 30)
    b = (np.minimum((conf * n_bins).astype(int), n_bins - 1) * n_bins
         + np.minimum((margin * n_bins).astype(int), n_bins - 1))            # same binning as in fit
    kk = np.arange(30)[None, :]
    stop = q_commit[kk, b] >= q_cont[kk, b]                                  # [N, 30]
    stop |= kk == (L[:, None] - 1)                                           # the last packet always stops
    stop &= kk < L[:, None]
    k = stop.argmax(1) + 1                                                   # first True; one always exists at L
    return k, pred[np.arange(n), k - 1]
```

Put the binning in one helper used by both functions. At `K = 30`, `q_cont` stays `-inf`, so the flow always stops there. A larger `λ` must never raise mean K (property test); a two-step toy problem with hand-computed values is the unit test.

This is **not** FastFlow: there is no recurrent state in the stopper and no online Q-learning. It answers one question, "does an optimal stopper on confidence and margin beat the safety head on the same backbone?", and the paper must describe it in those terms.

## 4. Controller replay

```python
def replay(scores, unknown_model, controller, order, *, window=2000, f=0.02, k_cap=30, **policy):
    # order: flow indices sorted by (period, flow_index); the stream holds every sampled flow, known and unknown
    trace = []
    for w, start in enumerate(range(0, len(order), window)):
        sl = order[start:start + window]
        theta_rej = unknown_model.thresholds(f)                               # [30]
        k, decision, prediction = safe_stop(scores.p_safe[sl], u[sl], scores.pred[sl], scores.ppi_len[sl],
                                            controller.theta, theta_rej, k_cap=k_cap, **policy)
        trace.append(window_stats(w, k, decision, prediction, scores.labels[sl], controller))
        controller.update(k)                                                  # acts on the NEXT window
    return trace
```

Points that are easy to get wrong:

- **Knobs change only between windows.** Every flow in a window is decided with the knob values in force when the window opened.
- **The stream contains unknown flows.** About 10% of `test_id` is unknown-class (25,308 of 250,000). Mean K is over all decided flows; accuracy is over known flows.
- **The initial threshold** comes from bisection on `val`: mean K is non-decreasing in `θ_commit`, so bisect on `logit(θ)` until `|mean K − B*| < 0.01`. If `B*` is below the mean K at `θ_min` or above it at `θ_max`, clamp and record "target unreachable on val".
- **The static baseline** runs the same loop with a controller whose gains are zero.
- **Disturbance (a), hard classes doubled:** in the chosen stretch of windows, redraw the stretch's flows with replacement, weight 2 for flows of hard classes and 1 for the rest, same stretch length, seeded.
- **Disturbance (b), unknown burst:** the stream is already ~10% unknown, so the burst **raises** the share to 30% for 20 windows by replacing known flows in that stretch with unknown flows drawn with replacement from the same period.
- **Disturbance (c), target step:** call `controller.set_target` at fixed window indices (8 → 5 → 8).
- **Tracking error** is computed after a 5-window warm-up and, for (c), separately per segment.

## 5. What to check by eye before trusting a number

| Number | Sanity check |
|---|---|
| AUROC from `energy_raw` in the score files | equals the phase-3 report's AUROC for that run (0.868 at K = 5, 0.904 at K = 30 on average for the tuned SSL PAT) |
| Accuracy from `pred` at K = 5, 8, 30 | equals the phase-3 report |
| Safe-Stop with `θ_commit = 0.999`, rejection off | mean K close to the mean of `min(30, ppi_len)`; accuracy close to full-flow accuracy |
| Safe-Stop with `θ_commit = 0.5` | mean K a little above `k_min` |
| Known FPR at `f = 0.02` on `val` | 0.02 at every K by construction; on `test_id` close to it, drifting upward in later weeks |
| P-RL with `λ = 0` | waiting is free, so it gives the largest mean K and the highest accuracy of its sweep (within noise of full-flow accuracy) |
| Controller with an easy target | error settles within a few windows and stays; `θ` does not saw-tooth |
