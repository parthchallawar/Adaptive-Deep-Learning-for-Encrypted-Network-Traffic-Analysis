# Validation and testing strategy, phases 4 to 7

Written 2026-11-17. Spec 020 holds the general testing rules; this page says, for the work still ahead, **what must be proven and by which kind of check**. A task is not done until its row here is green.

## Five kinds of check

| Kind | What it proves | Example already in the repository |
|---|---|---|
| **Unit** | One function does what its definition says, on hand-computable input | `tests/conftest.py`'s synthetic capture with a hand-computed PPI |
| **Property** | A rule holds for every input (monotonicity, bounds, sums to one) | calibration tests |
| **Equivalence** | Two implementations of the same thing agree exactly | stream vs full forward for the PAT; the loop's golden-file regression |
| **Real data** | The code survives the real corpus and gives plausible numbers | `tests/training/test_real_d4.py` (marked `slow`) |
| **Reproducibility** | The same inputs give the same bytes | report content hashes reproduced across Kaggle sessions |

Add to every new module: **one planted-bug check** (break the code on purpose; a test must fail).

## Phase 4

| Component | Unit | Property | Equivalence | Real data | Reproducibility |
|---|---|---|---|---|---|
| `ScoreArrays` (T1) | save/load round trip; `surprise` on a 3-flow fixture | values beyond `ppi_len` repeat the last valid one | `pred`/`conf` equal a direct computation from logits on 200 flows | one run's `val` scores; AUROC of `energy_raw` matches the phase-3 report | second pass, same seed, same `scores.json` hash |
| Unknown score (T2) | Mahalanobis of a class mean is 0; threshold hits FPR `f` on its fitting flows | raising `f` never lowers the rejection rate; **unknown val flows cannot change the fitted artefact** | all-classes distance formula equals a per-class loop | AUROC(K) on `test_id`, three seeds | `unknown.npz` hash stable |
| Safe-Stop (T3) | order of checks on hand-built scores; forced decision when `ppi_len < k_min`; NaN handling | raising `θ_commit` never lowers mean K | per-flow `decide` loop equals `replay.safe_stop` on 5,000 flows; score-based ECHO/CAPE equal `policies.p_echo`/`p_cape` on a fixture | Pareto fronts on `test_id` | same sweep, same table |
| P-RL (T3) | value recursion on a 2-step toy problem | larger `λ` never raises mean K | — | front on `test_id` | deterministic fit |
| Controller (T5) | one update by hand | integral bounded; `θ` within rails | — | replay over `val → test_id → test_drift` | `state_dict` round trip gives the same trajectory |
| Controller behaviour (T5) | infeasible target → saturation + one event | converges within 10 windows to < 3% on a synthetic generator with a known `θ → E[K]` map; no oscillation | — | disturbances (a), (b), (c) | — |
| Import hygiene | — | importing `adl_etc.inference.*` with torch, scikit-learn and mlflow blocked succeeds | — | — | — |

**Leak guards specific to phase 4.** (1) A test fits the unknown model twice, once with the unknown-class validation flows replaced by random ones, and asserts identical artefacts. (2) No function that fits or selects anything accepts `test_id` or `test_drift` data; replays over them only apply fixed knobs or the controller's own online updates.

## Phase 5

| Component | Unit | Property | Real data |
|---|---|---|---|
| `Cusum` | detects a 2-σ step within 5 windows | 500 windows of stationary noise give no alarm | zero alarms on `val` after choosing `h` |
| `WindowStats` | `prior_kl` is 0 for identical histograms; each field by hand on a 10-flow window | — | computed for every drift week |
| `DriftMonitor` | warm-up suppresses the first 5 windows; level rule (one signal → warning, two within three windows → alarm) | — | signal vs weekly macro-F1 table, three seeds |
| Efficiency | every registered model returns every field | FLOPs per full flow ≈ sum over packet steps within 5% (causal model) | table on the laptop, two interleaved repeats |

## Phase 6

| Component | Check |
|---|---|
| Database | create from empty; 50,000 rows through the writer queue in under 10 s; a reader sees rows while the writer runs; schema-version mismatch raises |
| Service | for a fixture PCAP, decisions equal offline `replay.safe_stop` on the same flows; lowering the budget lowers `θ_commit` after replayed traffic; a model reload affects only new flows; a flow with no payload packets is `undecidable` |
| Dashboard | each page renders in `streamlit.testing.v1.AppTest` against a stub client; plot functions run on fixture reports |
| Demo | the script in `docs/demo.md` run start to finish twice |

## Phase 7

| Check | How |
|---|---|
| Every number in the text is in a generated table | one final pass, text against `results/summaries/*.md` |
| Every table regenerates | `scripts/make_tables.py` on a clean tree |
| Every main-table row has three seeds | the table script prints the seed count per row |
| Every failed or untested criterion is in the limitations section | checklist against specs 007–013's success criteria |

## Standing checks (every task, every phase)

1. `pytest`, `ruff check src/ tests/ scripts/`, `mypy src/` clean.
2. Tests that need torch or mlflow skip with a reason when the extra is missing.
3. Code a Kaggle kernel imports is re-tested in the venv pinned to Kaggle's versions before a push.
4. A number that looks too good is treated as a bug first: check for duplicate flows across the split, for label leakage, and for a metric computed on the wrong subset.
