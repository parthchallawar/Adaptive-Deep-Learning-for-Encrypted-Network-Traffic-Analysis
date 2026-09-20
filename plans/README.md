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
| [phase-1-data-pipeline.md](phase-1-data-pipeline.md) | specs 001 to 004 (build steps 1 to 4) | in-progress: tasks T1 to T8, 11 of 14 items done |
