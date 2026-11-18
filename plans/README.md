# plans/

Implementation plans, derived from the specs in [`specs/`](../specs/). Written before the code, updated as work progresses. A spec says *what* a component is and why; a plan says *in what order* it gets built and how far along it is.

Plans are not a mirror of the specs. There are two kinds, and the filename says which:

- **Phase plans:** `plans/phase-<N>-<short-name>.md`. One unit of work spanning several specs, used when those specs form a single dependency chain and splitting them would produce documents that only say "wait for the others". This is the common case and the normal entry point for work.
- **Per-spec plans:** `plans/<NNN>-<short-name>.md`, where `NNN` matches the spec number. Used only when one spec is large enough to be sequenced on its own.

Never number a plan `NNN` unless it really is the plan for spec `NNN`.

Use [`template.md`](./template.md) as the starting point.

## Current

**Start with [`ROADMAP.md`](ROADMAP.md):** the master roadmap (status, phase flow, decision points with defaults, what the paper claims under each outcome).

| Plan | Covers | Status |
|---|---|---|
| [phase-4-unknown-stopping-budget.md](phase-4-unknown-stopping-budget.md) | specs 010, 009, 011 (build steps 11 to 13) | **done 2026-11-18** (T4 ablations and the ELECTS comparison descoped). Three seeds scored on Kaggle (kernel v18, 1 h 20 min); H3 AUROC bars met but not the early-rejection bar, H2 not met against P-ECHO / P-RL (Safe-Stop wins only the open-world order), H4 holds the budget but static is also fine under mild drift. See [results](../docs/results-phase-4.md). |
| [phase-5-drift-efficiency.md](phase-5-drift-efficiency.md) | specs 012 (monitoring), 013, drift part of 004 (steps 14, 15) | **done 2026-11-18** (adaptation, canary, ONNX not built). Steady decay measured for three models; class-prior KL and next-packet NLL track it; PAT full-flow 2.49 ms meets the bar, streaming step 1.65 ms does not. See [results](../docs/results-phase-5.md). |
| [phase-6-service-dashboard-demo.md](phase-6-service-dashboard-demo.md) | specs 018, 016, 017 (steps 16 to 18) | forward draft, 2026-11-17; re-check at phase start |
| [phase-7-report-paper.md](phase-7-report-paper.md) | spec 021 (step 19) and the experiments still owed | forward draft, 2026-11-17; re-check at phase start |
| [phase-1-data-pipeline.md](phase-1-data-pipeline.md) | specs 001 to 004 (build steps 1 to 4) | tasks T1 to T8 done; exit criteria 2/5 blocked on a Kaggle-kernel run (D1), which phase 2's T3 closes; D3's registration gate cleared 2026-09-21 |
| [phase-2-tracking-kaggle-baselines.md](phase-2-tracking-kaggle-baselines.md) | specs 014, 015, 005 (build steps 5 to 7) | all nine tasks (T1-T9) done, 2026-09-23; `scripts/make_tables.py` regenerates `results/summaries/` from the real 3-seed B3-on-D1 MLflow run (`val_macro_f1` 0.812-0.813). All 12 spec corrections landed. **Phase 2's own exit criteria are not all green** (3 of 9 open: no tracked B3-on-D4 three-seed run (B2 now done on D1; B1 XGBoost removed 2026-10-03); no genuine mid-flight Kaggle resume proven; spec 015's PAT-specific budget table still an estimate) -- left open for phase 3 or a follow-up, recorded rather than glossed over |
| [phase-3-pat-ssl-prefix-training.md](phase-3-pat-ssl-prefix-training.md) | specs 006, 007, 008 (build steps 8 to 10), plus B4/B5 from 005 and phase 2's three open exit criteria | **code built; Kaggle runs done** (2026-11-13/17, ~20 GPU-hours, 24 models + robustness on nine; three seeds for the tuned pair and B3, one for the rest); H1/H5 verdict and paired bootstrap in `docs/results-phase-3.md`. Open: three seeds for the other groups; exit criteria not all green (see the plan's progress log) |
