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
| [phase-2-tracking-kaggle-baselines.md](phase-2-tracking-kaggle-baselines.md) | specs 014, 015, 005 (build steps 5 to 7) | in progress: T1 to T7 done (B4 deferred; D1 export is 12.5M real flows on Kaggle, `load_split` passes for real; the evaluation entry point produced a real report from a real D4 checkpoint); T8's training kernel and entry point are built and tested locally, real Kaggle GPU run not yet pushed; T9 not started; 12 spec corrections scheduled, 8 landed |
