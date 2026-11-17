# Plan: Phase 4 — unknown detection, stopping policy, budget controller

- **Specs:** `specs/010-unknown-and-anomaly-detection.md`, `specs/009-adaptive-stopping-policy.md`, `specs/011-resource-budget-controller.md` (build steps 11, 12, 13)
- **Status:** not-started (plan written 2026-11-17)
- **Decides:** H2 (learned stopping beats thresholds), H3 (per-K unknown score), H4 (budget held under drift), and H1's third criterion (packets-to-decision), which phase 3 could not measure.
- **Exit gate:** phase 5 (specs 012, 013) starts when the [Exit criteria](#exit-criteria) hold.
- **Master roadmap:** [`ROADMAP.md`](ROADMAP.md). **How to work a task:** [`docs/playbook.md`](../docs/playbook.md). **Worked reference code for the four tricky algorithms** (Mahalanobis from running sums, the vectorised stopping rule, the P-RL baseline, controller replay): [`docs/design-phase-4.md`](../docs/design-phase-4.md).

Phase 4 is where contributions C2 and C3 are built and tested. Almost all of it is offline arithmetic on saved per-packet scores, so it needs very little GPU time. The one thing that can still be lost is the trained checkpoints, so that comes first.

## Where phase 4 starts (checked on disk, 2026-11-17)

| Fact | Evidence |
|---|---|
| Phase 3 is merged to `main`; tree clean | commit `c47649f` |
| Three seeds exist for tuned scratch, tuned SSL and B3 on `d1_open` | `results/kaggle-p3-s4/`, `docs/results-phase-3.md` |
| Headline model: tuned PAT with SSL init, test AUC_K 0.826 ± 0.004 | `results/summaries/phase3_table.md` §1 |
| Unknown AUROC by plain energy is already 0.868 at K=5 and 0.904 at K=30 | same table |
| Each run directory holds `calibration.json` and `eval/val/{safe.npy, labels.npy, ppi_len.npy, flow_index.npy, report.json}` | `results/kaggle-p3-s4/results/ft-pat-ssl_tuned-d1open-s1/` |
| **Of the nine headline checkpoints only B3 seed 0 is on local disk.** Six are in kernel v17's output; the tuned pair's seed 0 is only in the superseded v16 output (see T0) | file search of `results/kaggle-p3-s{1,2,3,4}/` |
| A checkpoint's identity is `file_sha256(best.pt)`, stored as `checkpoint_hash` in `calibration.json` | `evaluation/run.py`, `training/run.py` |
| Unknown-class flows have label `-1` after `LabelSpace.to_model` | `training/labels.py` |
| `metrics.py` already has `auroc`, `aupr`, `fpr_at_tpr`, `open_set_f1`, `rejection_earliness`, `mean_k`, `p95_k`, `coverage`, `committed_accuracy`, `pareto_front` | `evaluation/metrics.py` |
| `predict_heads(model, flows, label_space, standardizer, device=, heads=(...))` returns any named PAT head as NumPy | `models/baselines/predict.py` |
| The `safe` head reads five detached features: entropy of `cls`, top-2 margin, mean entropy of the three `next` heads, `sigmoid(next_end)`, `k/30` | `models/pat.py` |
| `tests/test_import_boundary.py` already proves `adl_etc.evaluation` imports without torch or mlflow | `evaluation/run.py` docstring |
| Phase 3's evaluations all used the same 250,000-flow draw (`subsample_for_eval`, seed 0) on `test_id` | `configs/eval/d1_open_robust.yaml` |
| Saved evaluation has no `proj` embedding and no next-packet outputs | file listing above |
| The PAT returns `cls`, `safe`, `next_size`, `next_ipt`, `next_dir`, `next_end`, `proj` per position, plus `cls0`/`z0` | `src/adl_etc/models/pat.py` |
| Fine-tuning keeps the next-packet heads trained (`objective.npp_weight: 0.1`) | `configs/train/ft_pat_d1.yaml` |
| `d1_open` has four periods: `train`, `val`, `test_id`, `test_drift` | `configs/splits/d1_open.yaml` |
| The full D1 export is on local disk | `results/kaggle-full/shards/` |

## Findings that change the specs

| # | Finding | Consequence |
|---|---|---|
| P4-F1 | Eight of the nine headline checkpoints are not on local disk: six are in Kaggle kernel v17's output and two (the tuned pair's seed 0) only in v16's, which v17 superseded. A later kernel version makes an earlier version's output unreachable from the CLI (this already lost one checkpoint in phase 3). | **T0 pulls them before any kernel push.** Nothing else in this phase may push the training kernel first. |
| P4-F2 | Spec 010 needs `proj` at every K. Saving it is ~1.9 GB per run per split (250k × 30 × 128, fp16). | Never save `proj`. Fit the Gaussians from running sums inside the model pass and save only scalar scores (T1, the `ScoreArrays` artefact). |
| P4-F3 | Plain energy already reaches spec 010's absolute AUROC bars, or nearly. | Spec 010 gains a relative criterion: the combined score must beat energy alone (correction C1). |
| P4-F4 | The 30 unknown classes are the same in `val` and in `test_id`. | Nothing may be fitted or chosen using unknown-class validation flows. Defaults (`β`, `k_min`, `k_ood`, `f`, check order) are fixed in advance; their alternatives are reported as ablations, not used for selection (correction C2). |
| P4-F5 | `p_echo` and `p_cape` take a `DenseLogits` (2 GB per run, not pulled). | Score-based versions that need only `pred`, `conf`, `energy_raw`, with an equality test against the originals on a fixture (T3). |
| P4-F6 | Spec 009 requires a P-RL baseline, deferred in phases 2 and 3. | Build it as offline fitted-value stopping on saved scores, CPU only (T3). Stated honestly as "same-backbone optimal stopping", not a FastFlow re-implementation (correction C4). |
| P4-F7 | Depth exits (spec 006, optional) were never built. | Spec 011's "compute" budget metric is descoped; the controller targets mean K and p95 K (correction C6). |
| P4-F8 | H4 needs scores in time order across `test_drift`'s 18 weeks. The evaluation cap is 250k flows for a whole split. | The score pass takes a per-period cap and saves each flow's period (T1). |
| P4-F9 | Spec 016 says online and offline must share one implementation. `evaluation/` may not import torch. | The decision rule, the unknown scorer and the controller live in `src/adl_etc/inference/` as **torch-free NumPy**; `evaluation/` replays import them (decision D2). |

## Design decisions

- **D1. One compact score artefact per (run, split).** Every policy, threshold and controller experiment replays from it; the model is run once. See T1 for the exact fields.
- **D2. `inference/` is the single source of truth and is torch-free.** `inference/unknown.py`, `inference/policy.py`, `inference/controller.py` hold the per-flow logic. `evaluation/replay.py` holds the vectorised offline forms. A test asserts the two give identical decisions on a fixture (spec 009's replay-equivalence test). Extend `tests/test_import_boundary.py` (which already walks `adl_etc.evaluation`) to walk `adl_etc.inference` too, so that importing any module there with torch, scikit-learn and mlflow blocked succeeds.
- **D3. Local CPU first, Kaggle second.** The PAT has 885,821 parameters; T0 measures CPU throughput. If it is at least 500 flows/s, every score pass runs locally and phase 4 needs no GPU except optional retraining ablations.
- **D4. Defaults are fixed before any test number is seen:** `β = 0.5`, `k_min = 2`, `k_ood = 3`, `f = 0.02`, rejection checked first, `W = 2000`, `K_p = 0.5`, `K_i = 0.05`. Gains may be re-tuned on the **val period only**.
- **D5. Every reported number comes from `test_id` or `test_drift`.** `val` is for fitting and choosing only.

## Order

```
T0 secure checkpoints ──► T1 score pass + ScoreArrays ──► T2 unknown score (010) ──► T3 stopping policy (009) ──► T5 controller (011) ──► T6 report/tables ──► T7 three seeds + results doc ──► T8 paperwork
                                                                                  └─► T4 ablations (after T3, needs a decision at DP-4b)
```

Each task ends with `pytest`, `ruff check src/ tests/ scripts/` and `mypy src/` clean, and a progress-log line here.

## Tasks

### T0 — Secure the checkpoints, measure CPU speed (no code; do this first)

**Where each `best.pt` is (checked 2026-11-17 from the pulled run folders; each file is ~9–14 MB):**

| Run | Trained in | On local disk? | How to get it |
|---|---|---|---|
| `bl-gru-main-d1open-s0` | Session 2 (v15) | **yes**, `results/kaggle-p3-s2/results/` | also in the slim checkpoint dataset |
| `pt-pat_pretrain-npp_pfc-d1open-s0` | Session 1 (v14) | **yes**, `results/kaggle-p3-s1/results/` | — |
| `ft-pat-tuned-d1open-s{1,2}`, `ft-pat-ssl_tuned-d1open-s{1,2}`, `bl-gru-main-d1open-s{1,2}` | Session 4 (v17, the latest version) | no | `kaggle kernels output parthchallawar/adl-encrypted-traffic-train -p results/kaggle-p3-s4 --file-pattern 'best\.pt$' --page-size 200` |
| **`ft-pat-tuned-d1open-s0`, `ft-pat-ssl_tuned-d1open-s0`** | Session 3 (**v16, superseded by v17**) | no | **at risk.** Session 4 only read them from a mount for evaluation; they are not in v17's output. See "Seed 0 of the tuned pair" below |

**Seed 0 of the tuned pair.** Try in this order:

1. The Kaggle website: open the notebook `adl-encrypted-traffic-train`, pick **version 16** in the version history, open its Output, and download `results/ft-pat-tuned-d1open-s0/best.pt` and `results/ft-pat-ssl_tuned-d1open-s0/best.pt` by hand.
2. If the website no longer serves version 16's output: retrain both (`ft_pat_d1_tuned.yaml` and `ft_pat_ssl_d1_tuned.yaml`, `seed=0`, ~1 GPU-hour each; the SSL one needs `init_from` pointing at the pretraining checkpoint, which is local). Then run the `d1_open_robust.yaml` evaluation on each and compare with `results/kaggle-p3-s4/results/robust/<run>/eval/test_id/report.json`: the same report hash means the retrained model is the same model. If the hash differs, the seed-0 rows of `phase3_table.md` must be regenerated from the retrained runs and the difference recorded in the progress log (AUC_K should agree within about 0.002, the seed-to-seed spread).
3. Whichever way they are obtained, push all nine run folders to the slim checkpoint dataset (`./scripts/kaggle_sync.sh push-checkpoints <run-dir> ...`) so no later kernel version can hide them again.

1. Pull `best.pt` for all nine headline runs from kernel v17's output into `results/kaggle-p3-s4/results/<run>/best.pt`, following `docs/kaggle-workflow.md` (`--file-pattern`; kernel id is in `kernel/kernel-metadata.json`). Runs: `ft-pat-tuned-d1open-s{0,1,2}`, `ft-pat-ssl_tuned-d1open-s{0,1,2}`, `bl-gru-main-d1open-s{0,1,2}`. Use the table above: six come from v17's output, B3 seed 0 is already local, and the tuned pair's seed 0 needs the recovery steps above.
2. Count the files. A pull can stop early and still exit 0 (phase-3 lesson): expect exactly nine.
3. For each, confirm `adl_etc.utils.provenance.file_sha256(<run>/best.pt)` equals `checkpoint_hash` in that run's `calibration.json` (this is the check `Calibration.load(..., checkpoint_hash=...)` makes), then load it once with `training.checkpoint.load_checkpoint`.
4. Copy the nine files to a second location outside `results/` (they are gitignored and not reproducible without ~7 GPU-hours).
5. Measure CPU throughput: run the tuned SSL seed-0 model over 20,000 validation flows at batch 1024 with `torch.set_num_threads(4)`; record flows/s here.

**Done when:** nine verified checkpoints on disk and backed up; CPU flows/s recorded. **If a checkpoint cannot be recovered:** retrain that run (~1 GPU-hour for a PAT, ~40 minutes for the GRU) before anything else, with the same config and seed, and note that its numbers must be re-checked against `phase3_table.md`.

### T1 — Score pass and the `ScoreArrays` artefact

New files: `src/adl_etc/evaluation/scores.py` (NumPy only), `src/adl_etc/training/score_pass.py` (torch), `configs/eval/d1_open_scores.yaml`, tests under `tests/evaluation/test_scores.py` and `tests/training/test_score_pass.py`.

`ScoreArrays` (a directory of `.npy` files plus `scores.json`, written with `utils/atomic.py`). `N` flows, position `k-1` = after `k` packets. Positions at or beyond a flow's `ppi_len` repeat its last valid value, so a scan never needs a mask.

| Field | Shape, dtype | Meaning |
|---|---|---|
| `pred` | `[N,30]` int16 | argmax of prior-adjusted logits (`Calibration.adjusted`), lowest index on ties |
| `conf` | `[N,30]` float16 | largest calibrated class probability (`Calibration.probabilities`) |
| `margin` | `[N,30]` float16 | top-1 minus top-2 calibrated probability |
| `p_safe` | `[N,30]` float16 | `Calibration.safe_probability(safe)` |
| `energy` | `[N,30]` float32 | `−T_k · logsumexp(adjusted / T_k)` with the per-K temperature |
| `energy_raw` | `[N,30]` float32 | `policies.energy_score` on raw logits (what P-CAPE's gate uses) |
| `mahal` | `[N,30]` float32 | relative Mahalanobis `M_k`, computed in the pass from T2's Gaussian bank; so the training-sample pass that fits the bank runs before the score passes (run order in `docs/design-phase-4.md` section 0) |
| `pkt_nll` | `[N,30]` float32 | index `j`: next-packet negative log-likelihood of packet `j+1` given the first `j` (sum of size, IPT and direction terms); index 0 is 0 |
| `next_entropy` | `[N,30]` float16 | entropy of the next-packet prediction after `k` packets |
| `labels`, `is_unknown`, `ppi_len`, `flow_index`, `period` | `[N]` | labels are model indices from `LabelSpace.to_model` (`-1` = unknown class), as in `EvalArrays`; `is_unknown` (`labels == -1`) is stored explicitly; `flow_index` is the index the sampler drew; `period` is an index into `scores.json["periods"]` |

`scores.json`: schema version, `checkpoint_hash`, calibration hash, unknown-model hash (or null), split, period list, caps, seed, git commit.

Functions:

- `scores.ScoreArrays.save/load`, `scores.surprise(pkt_nll) -> [N,30]` (running mean over packets `2..k`; `S_1 = 0`), `scores.at_len(arr, ppi_len)`.
- `score_pass.run_score_pass(model, calibration, split, period_names, cap_per_period, seed, unknown_model=None) -> ScoreArrays`. Uses `subsample_for_eval` per period so the class mix stays natural. For the GRU (no `safe`/`proj`/next heads) it fills `p_safe` with `conf`, and `mahal`/`pkt_nll`/`next_entropy` with zeros, and records `"heads": "cls-only"` in `scores.json`.
- `score_pass.accumulate_gaussian_sums(model, split, ...)` for T2 (below).

CLI: a new entry point, `python -m adl_etc.evaluation.score_run --config configs/eval/d1_open_scores.yaml --override run_dir=<run> out_dir=<run>`, modelled on `evaluation/run.py` (`run_dir` rebuilds the model from `config_resolved.yaml`; every torch-touching import is inside `main()`, because `tests/test_import_boundary.py` walks this package). It writes `<out_dir>/eval/<split>/scores/`. `evaluation/run.py` itself is not changed: it is already long, and the Kaggle queue's skip rule is keyed on its output files.

How the pass works, so that nothing large is ever held:

- Take the flows in chunks of 8,192. For each chunk call `predict_heads(..., heads=("cls", "safe", "proj", "next_size", "next_ipt", "next_dir", "next_end"))`, reduce it to the scalar fields below, and discard the heads.
- `pkt_nll` reuses the per-token cross-entropy and the target alignment of the next-packet loss in `training/pat_losses.py` (with no reduction). Do not re-derive which token is the target of which position.
- `next_entropy` is exactly the third `safe` feature: the mean normalised entropy of the three `next` heads.
- **Sampling.** For `val` and `test_id`, use the same whole-split draw as phase 3 (`subsample_for_eval(len(data), cap=250000, seed=0)`), so every score lines up flow for flow with phase 3's reports and `predictions.npz`; derive each flow's period from the cumulative period lengths. For `test_drift`, draw 50,000 per week (`subsample_for_eval` on each period's `ArrayData`, built with `ArrayData.from_shards([shard_set], [mask])` in the split file's period order).

Tests: round-trip save/load; fill-forward beyond `ppi_len`; `pred`/`conf` equal a direct computation from logits on a 200-flow fixture; `surprise` hand-checked on a 3-flow fixture; a second pass with the same seed is byte-identical.

**Done when:** `scores/` exists for all nine runs on `val` (250k), `test_id` (250k) and `test_drift` (50k per week); a re-run reproduces `scores.json`'s content hash; and **the phase-3 numbers come back out of the score files**: for each run, `metrics.auroc(energy_raw at K, labels == -1)` at K = 5 and 30 on `test_id` and accuracy at K = 5, 8, 30 from `pred` match that run's phase-3 `report.json` to the fourth decimal. A mismatch means the pass, the sampling or the checkpoint is wrong; stop and find it before T2.

### T2 — Unknown score (spec 010, hypothesis H3)

New files: `src/adl_etc/inference/unknown.py` (NumPy), tests `tests/inference/test_unknown.py`, config block `unknown:` in `configs/eval/d1_open_scores.yaml`.

Fitting, per K, on **known-class training flows** that reached K (`calibration.reached`), at most 1,500 per class, sampled with `seeded_generator(seed, "unknown_fit")`:

1. In one model pass accumulate in float64: `sum_c[30,C,128]`, `n_c[30,C]`, `outer[30,128,128]` (sum of `x xᵀ`), `sum_all[30,128]`.
2. Class means `μ_{k,c}` for classes with `n_c ≥ 50`; others contribute to the background only.
3. Shared covariance `Σ_k = (outer − Σ_c n_c μ_c μ_cᵀ) / (n − C_k)`, then shrink: `Σ ← 0.9 Σ + 0.1 · (tr Σ / 128) · I`.
4. Background Gaussian `(μ_bg,k, Σ_bg,k)` from `sum_all` and `outer`, same shrinkage.
5. `M_k(x) = min_c (x−μ_c)ᵀ Σ_k⁻¹ (x−μ_c) − (x−μ_bg)ᵀ Σ_bg,k⁻¹ (x−μ_bg)`. Compute all classes at once: `xᵀPx − 2 xᵀPμ_c + μ_cᵀPμ_c`.

`UnknownModel` (saved as `unknown.npz` + `unknown.json` beside `calibration.json`, with `checkpoint_hash`; loading under another checkpoint raises):

- `means`, `precision`, `bg_mean`, `bg_precision`, `class_mask` per K;
- `z_mean[30,3]`, `z_std[30,3]` for energy, Mahalanobis and surprise, from **known val flows that reached K**;
- `quantiles[30,1001]`: the 0.000…1.000 quantiles of `u_k` over the same flows.

Scoring: `u_k = z(E_k) + z(M_k) + β·z(S_k)`, `β = 0.5`. `theta_reject(k, f)` interpolates `quantiles[k-1]` at `1 − f`.

Evaluation (pure NumPy). **Reuse what `evaluation/metrics.py` already has** (`auroc`, `aupr`, `fpr_at_tpr`, `rejection_earliness`, `open_set_f1`); read each docstring before use and add a function only if none fits:

- AUROC at K: `metrics.auroc(score[:, k-1], labels == -1)` for `u`, and for each single signal (the score files are already filled forward past `ppi_len`).
- Rejection K: first K at or after `k_ood` where `u_k ≥ θ_reject(k)`; use `metrics.rejection_earliness` if its definition matches, otherwise add `rejection_k(u, theta, k_ood)`.
- Baseline "fixed gate": `energy_raw` at K=10 with a threshold giving the same known-flow FPR.

Report on `test_id`, three seeds: AUROC(K) for K=1..30 for `u`, energy only, Mahalanobis only, surprise only; mean rejection K of unknown flows for `u` vs the fixed K=10 gate at equal FPR (0.02 and 0.05).

Tests (spec 010's list, made concrete): the per-K threshold gives FPR `f` on the fitting flows within 1/n; the Mahalanobis distance of a class mean to itself is 0; z-statistics change if and only if val known flows change (unknown val flows have no effect — this is the P4-F4 guard); flows with shuffled packets score higher on average than the originals.

D4 anomaly mode (spec 010) is **optional in this phase**: run the D1-trained model's score pass over D4 and report malware AUROC; expect a domain-gap caveat. Skip if T0–T3 are behind.

**Done when:** H3's numbers exist for three seeds. **Decision point DP-4a** (see the roadmap): if the combined `u_k` does not beat energy alone by at least 1 AUROC point at some K ≤ 8, keep per-K quantile thresholds on energy alone as the shipped score, record the negative result, and continue. The policy and the controller do not depend on which signal wins.

### T3 — Stopping policy (spec 009, hypothesis H2)

New files: `src/adl_etc/inference/policy.py` (per-flow rule, NumPy), `src/adl_etc/evaluation/replay.py` (vectorised), `src/adl_etc/evaluation/rl_stop.py` (P-RL), `configs/policy/safe_stop.yaml`, tests under `tests/inference/` and `tests/evaluation/`.

Rule, for a flow with `L = min(30, ppi_len)`:

```
for k in 1..L:
    if k >= k_ood and u_k >= theta_reject(k):   REJECT at k
    elif k >= k_min and p_safe_k >= theta_commit: COMMIT pred_k at k
    elif k == L: FORCED: REJECT if u_k >= theta_reject(k) else COMMIT pred_k
    # the forced branch follows spec 009 literally and ignores k_ood: it is the only way a flow
    # shorter than k_ood packets is rejected. Worked through in docs/design-phase-4.md section 2.
```

`inference.policy.decide(k, p_safe_k, u_k, theta_commit, theta_reject_k, L, k_min, k_ood, order) -> Decision`. NaN inputs are treated as `p_safe = 0`, `u = +inf` and counted.

`evaluation.replay.safe_stop(scores, theta_commit, f, unknown_model, *, k_min=2, k_ood=3, order="reject_first", k_cap=30) -> ReplayResult` with `k[N]`, `decision[N]` (0 commit, 1 reject, 2 forced commit, 3 forced reject), `prediction[N]` (−1 for reject). Vectorised: first index where each condition holds, then `min`.

Baselines on scores: `replay.echo(scores, tau)`, `replay.cape(scores, tau_c, gate_k=10, gate_threshold)` (must equal `policies.p_echo` / `p_cape` on a fixture built from real logits), and `rl_stop.fit(scores_val, lam)` / `rl_stop.apply`.

P-RL definition: finite-horizon optimal stopping by backward induction on the state `(conf_k, margin_k)` binned 50 × 50 per K. `Q_commit = P(correct | bin, k) − λ·k/30`; `V_k = max(Q_commit, E[V_{k+1} | bin, k])`, both estimated on `val` known flows; bins with fewer than 20 flows fall back to the K-wide mean. `λ` sweeps the trade-off.

Metrics (`evaluation/metrics.py`; `mean_k`, `p95_k`, `coverage`, `committed_accuracy`, `pareto_front` and `open_set_f1` exist already, only `pareto_area` is new): `mean_k`, accuracy on known flows (a rejected known flow counts as wrong), coverage, unknown TPR, known FPR, open-set macro-F1 with "unknown" as one extra class, the fraction of unknown flows committed, and `pareto_area(mean_k, acc, lo=2, hi=12)` (the mean, over 201 budgets from 2 to 12 packets, of the best accuracy reachable within each budget; 0 where a policy cannot operate).

Two evaluations, both on `test_id`, three seeds:

1. **Closed world (H2's main test):** known flows only, rejection off. Sweep 200 values of `θ_commit` evenly in logit space over [0.5, 0.999]; the same for ECHO's `τ`, CAPE's precision target and P-RL's `λ`. Report accuracy at mean K = 4, 6, 8 (the best accuracy among sweep settings whose mean K is at most that budget) and `pareto_area`. This also gives **H1's third criterion**: packets-to-decision at equal accuracy for the SSL model against the scratch model.
2. **Open world:** all flows, `f = 0.02`. Report open-set macro-F1, unknown flows committed, and the mean extra packets for known flows, for rejection-first and for commit-first.

Tests: raising `θ_commit` never lowers mean K; raising `f` never lowers the rejection rate; order-of-checks on hand-built scores; per-flow `decide` loop equals `replay.safe_stop` on 5,000 fixture flows; forced decision at `ppi_len < k_min`.

**Done when:** the H2 table exists for three seeds.

### T4 — Ablations (after T3; scope set at DP-4b)

| Ablation | How | Cost |
|---|---|---|
| Next-packet-entropy feature in `safe` (feature index 2 of 5) | First a knock-out at inference. Move the feature construction in `PAT._outputs` into a module-level function `safe_features(...)` with no change in behaviour (the loop-regression and stream-equivalence tests guard this). In the score pass, recompute `safe` as `model.head_safe(cat([z, features']))` with feature 2 replaced by its mean over validation flows, then **re-fit that K's safety map on `val`** (`calibration.isotonic_fit`) so both variants are calibrated the same way. If the knock-out moves accuracy at mean K 4–6 by less than 0.2 points, report "no measurable contribution" and stop. Otherwise retrain only the `safe` head with the backbone frozen, with and without the feature (new `objective: safe_only`). | 0, or ~30 GPU-min |
| Rejection-first vs commit-first | replay flag | 0 |
| `β` in {0, 0.5, 1}, `k_ood` in {3, 5, 10}, `k_min` in {1, 2, 3} | replay sweeps, reported as sensitivity, defaults unchanged (D4) | 0 |
| Learned combiner for `u_k` | logistic regression on (E, M, S) with **synthetic** unknowns (packet-shuffled and size-inflated known val flows) as positives | 0 |
| ELECTS-style halting head (spec 009, policy A) | **stretch**: three retrains at three `α`; skip unless H2 is otherwise decided and quota is spare | ~3 GPU-h |

### T5 — Budget controller (spec 011, hypothesis H4)

New files: `src/adl_etc/inference/controller.py` (pure Python/NumPy), `src/adl_etc/evaluation/controller_replay.py`, `configs/controller/default.yaml`, tests under `tests/inference/test_controller.py`.

`BudgetController(target, metric="mean_k"|"p95_k", kp=0.5, ki=0.05, i_max=5.0, theta_min=0.5, theta_max=0.999, window=2000, min_flows=200)`:

```
e = (B_obs − B*) / B*
I = clip(I + e, −i_max, i_max)
phi = logit(theta) − (kp·e + ki·I)
theta = clip(sigmoid(phi), theta_min, theta_max)
```

`update(window_k: np.ndarray) -> ControllerEvent` (one call per window; no update below `min_flows`); `state_dict()/load_state_dict()`; an event log entry for every update (`window`, `B_obs`, `e`, `I`, `theta`, `reason`). Oscillation guard as in the spec (sign of `e` alternating for more than 5 windows above 5% amplitude halves the gains and logs). A second instance of the same class on `logit(f)` (`kp=0.2`, `ki=0.02`, `f` in [0.001, 0.2]) holds a rejection-rate cap. For `p95_k` the controller lowers `k_cap` instead of `θ`.

`controller_replay.run(scores, order, controller, disturbances) -> ReplayTrace`: flows in `(period, flow_index)` order; for each window, `replay.safe_stop` on that slice with the current knobs, then `controller.update`. Initial `θ` from bisection on `val` to hit `B*`. Static baseline: that initial `θ` held fixed.

Disturbances (spec 011): (a) hard classes doubled (hard = bottom quartile of per-class accuracy at K=5 on `val`); (b) an unknown burst: the stream is already about 10% unknown-class, so the share is raised to 30% for 20 windows; (c) `B*` stepped 8 → 5 → 8.

Report over `val → test_id → test_drift`, three seeds, `B* ∈ {4, 6, 8}`: mean absolute tracking error, fraction of windows within 5%, settling time after the step, accuracy at budget, controller vs static.

Tests: on a synthetic score generator with a known `θ → E[K]` map the controller converges within 10 windows to under 3% error; an infeasible target saturates `θ` at `θ_min` with a bounded integral and one "budget infeasible" event; lowering `θ` never raises mean K; `state_dict` round-trips.

**Done when:** the H4 table exists for three seeds.

### T6 — Report and tables

- `evaluation/report.py`: `Report` gains optional sections `unknown` (AUROC(K) per signal, rejection K), `policy` (the four fronts, accuracy at mean K 4/6/8, `pareto_area`), `budget` (tracking metrics). The content hash covers them.
- `evaluation/plotstyle.py`: `plot_auroc_vs_k`, `plot_pareto_fronts`, `plot_budget_trace` (matplotlib imported inside each function).
- `scripts/make_tables.py`: `results/summaries/phase4_table.{md,tex}` with sections for H3, H2, H4 and H1's third criterion; same dirty-tree and config-drift refusals.
- `scripts/mlflow_import.py`: attach `scores.json`, the policy and budget JSON files to their runs.

### T7 — Results document

`docs/results-phase-4.md`, in the style of `docs/results-phase-3.md`: every number from `phase4_table.md`; a verdict for H2, H3, H4 and H1's third criterion against the criteria below, whichever way each falls; a limits section.

### T8 — Paperwork

Land the spec corrections below; update `specs/README.md` (steps 11–13), `plans/README.md`, `CLAUDE.md` (current state, new modules, the `inference/` torch-free rule) and the roadmap's status table.

## Spec corrections scheduled

| # | Spec | Correction | Lands with |
|---|---|---|---|
| C1 | 010 | Add: combined `u_k` must beat energy alone by ≥ 1 AUROC point at some K ≤ 8, or the negative result is reported and energy with per-K quantiles ships. | T2 |
| C2 | 010, 009 | Nothing fitted or selected on unknown-class val flows; the "learned combiner" open question uses synthetic unknowns only. | T2 |
| C3 | 010 | Inputs: `proj` is never stored; Gaussians come from running sums; the stored artefact is `unknown.npz`. | T2 |
| C4 | 009, 005 | P-RL is offline fitted-value stopping on `(conf, margin)`; say so where FastFlow is named. | T3 |
| C5 | 009 | ELECTS halting head is a stretch item; if not run, mark it descoped with the reason. | T4 |
| C6 | 011 | "compute" budget metric descoped (no depth exits); metrics are mean K and p95 K. | T5 |
| C7 | 011 | Canary-accuracy alert moves to spec 012 (it needs labels and belongs with drift). | T5 |
| C8 | 004 | Add the open-set, policy and budget metrics defined in T2/T3/T5 to the metric list. | T6 |

## Acceptance criteria (from the specs, with C1 applied)

| Hypothesis | Criterion | Where |
|---|---|---|
| H3 | AUROC of `u_k` ≥ 0.85 at K=8 and ≥ 0.90 at K=30 on `test_id`; beats energy alone by ≥ 1 point at some K ≤ 8; mean rejection K of unknown flows ≥ 30% lower than the fixed K=10 gate at equal FPR | T2 |
| H2 | At mean K=6, Safe-Stop accuracy ≥ P-ECHO + 1.5 points and ≥ P-CAPE and P-RL; largest `pareto_area`; rejection-first lowers committed unknowns without delaying known flows by more than 0.5 packets | T3 |
| H1 (3rd) | SSL model needs ≥ 10% fewer packets than scratch at equal accuracy | T3 |
| H4 | Mean tracking error ≤ 5% with the controller over `test_drift`, against ≥ 15% static; no oscillation; settling ≤ 5 windows | T5 |

A criterion that fails is reported as failed. Do not move a threshold after seeing the number.

## Kaggle budget

| Item | GPU-hours | Needed? |
|---|---|---|
| Score passes, if CPU is too slow (T0 decides) | ~1 | only if under 500 flows/s locally |
| `safe`-head retrain ablation | ~0.5 | only if the knock-out shows an effect |
| ELECTS halting head | ~3 | stretch |
| The tuned pair's seed 0, if version 16's output cannot be downloaded from the website | ~2 + two 5-minute evaluations | only then (T0) |
| Any other lost checkpoint | ~1 each | only if T0 fails |

Expected total: 0 to 2 GPU-hours.

## Risks

| Risk | Effect | Mitigation |
|---|---|---|
| A checkpoint is lost before T0 | ~1 GPU-hour each to retrain; numbers must be re-verified | T0 is first; no kernel push before it |
| CPU score pass too slow | Phase waits on Kaggle sessions | One Kaggle session computes all scores; fp16 everywhere to keep the pull small |
| Combined unknown score no better than energy | H3's relative criterion fails | DP-4a: ship energy with per-K quantiles; the per-K threshold idea still stands |
| `p_safe` is no better than max-prob | H2 fails | Report it; the paper's claim moves to per-K calibration, robustness and the controller (roadmap, fallback table) |
| Static threshold already holds the budget under drift | H4's contrast is small | Spec 011 already says to report it; the synthetic disturbances still show the controller's value |
| Tuning on test by accident | Invalid results | D4/D5; a test asserts unknown val flows cannot affect fitted artefacts; all sweeps on test are labelled sensitivity |
| Score files drift from checkpoints | Wrong numbers silently | Every artefact carries `checkpoint_hash`; loaders raise on mismatch |

## Exit criteria

1. `pytest`, `ruff`, `mypy` clean, including the replay-equivalence, monotonicity, controller-convergence and no-torch-in-`inference/` tests.
2. Nine verified checkpoints on disk and backed up.
3. `scores/` for nine runs × three splits, reproducible by hash.
4. H2, H3, H4 and H1's third criterion each have a three-seed verdict in `docs/results-phase-4.md`, citing only `make_tables.py` output.
5. Corrections C1–C8 landed; `specs/README.md`, `plans/README.md`, `CLAUDE.md` and `plans/ROADMAP.md` updated.

What phase 5 inherits: a torch-free decision rule, unknown scorer and controller in `inference/`; time-ordered scores for the drift weeks; a replay harness that can feed any monitor.

## Progress log

- **2026-11-17.** Plan written from specs 009–011, the phase-3 plan and results, and a check of what is on disk. Nine findings (P4-F1 to F9), five design decisions, eight spec corrections scheduled. Not started. First action: T0.
