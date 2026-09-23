# plans/

Implementation plans, derived from the specs in [`specs/`](../specs/). Written before the code, updated as work progresses. A spec says *what* a component is and why; a plan says *in what order* it gets built and how far along it is.

Plans are not a mirror of the specs. There are two kinds, and the filename says which:

- **Phase plans:** `plans/phase-<N>-<short-name>.md`. One unit of work spanning several specs, used when those specs form a single dependency chain and splitting them would produce documents that only say "wait for the others". This is the common case and the normal entry point for work.
- **Per-spec plans:** `plans/<NNN>-<short-name>.md`, where `NNN` matches the spec number. Used only when one spec is large enough to be sequenced on its own.

Never number a plan `NNN` unless it really is the plan for spec `NNN`.

Use [`template.md`](./template.md) as the starting point.

## Current

| Plan | Covers | Status |
|---|---|---|
| [phase-1-data-pipeline.md](phase-1-data-pipeline.md) | specs 001 to 004 (build steps 1 to 4) | tasks T1 to T8 done; exit criteria 2/5 blocked on a Kaggle-kernel run (D1), which phase 2's T3 closes; D3's registration gate cleared 2026-09-21 |
| [phase-2-tracking-kaggle-baselines.md](phase-2-tracking-kaggle-baselines.md) | specs 014, 015, 005 (build steps 5 to 7) | all nine tasks (T1-T9) done, 2026-09-23; `scripts/make_tables.py` regenerates `results/summaries/` from the real 3-seed B3-on-D1 MLflow run (`val_macro_f1` 0.812-0.813). All 12 spec corrections landed. **Phase 2's own exit criteria are not all green** (3 of 9 open: no tracked B1/B2/B3-on-D4 three-seed runs; no genuine mid-flight Kaggle resume proven; spec 015's PAT-specific budget table still an estimate) -- left open for phase 3 or a follow-up, recorded rather than glossed over |
