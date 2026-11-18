# Working playbook

How to carry this project forward, one task at a time, whether the work is done by the owner, by a new developer, or by an AI assistant of any size. Written 2026-11-17.

## 1. What to read, in order

1. `CLAUDE.md` — commands, architecture, the traps.
2. `plans/ROADMAP.md` — where the project is and what comes next.
3. The current phase plan (now `plans/phase-4-unknown-stopping-budget.md`) — find the first task that is not done.
4. The one spec that task implements.
5. Only the source files that task names.

Do not read everything. A task section is written to be enough on its own.

## 2. The task loop

1. **Pick one task** from the current plan (T0, T1, …). Do not start two.
2. **Check its inputs exist** (files, checkpoints, earlier tasks). If one is missing, that is the task.
3. **Write the tests first**, from the task's "Tests" list. They must fail for the right reason.
4. **Write the code** in the files the task names. Keep to the import rules (section 3).
5. **Plant a bug** in the new code (flip a comparison, drop a term) and confirm a test catches it. Remove the bug. If no test caught it, add the missing test.
6. **Run the three checks**: `pytest`, `ruff check src/ tests/ scripts/`, `mypy src/`. All clean.
7. **Run it on real data** when the task says so, and look at the numbers. A number that looks too good is a bug until proven otherwise (see the D4 and D3 traps in `CLAUDE.md`).
8. **Write one progress-log line** in the plan: date, what was built, what was measured, what surprised you.
9. **Stop.** Commit only when the owner asks.

## 3. Rules that are never broken

1. **No torch, scikit-learn or mlflow imports in `data/`, `evaluation/`, `inference/` or `utils/tracking.py`.** In `evaluation/run.py`-style entry points, torch is imported inside `main()`.
2. **Token 0 means "no packet".** Never recompute bin edges at import.
3. **Leakage rules are enforced by `load_split`; do not work around a `LeakageError`.**
4. **Nothing is fitted or chosen on `test_id` or `test_drift`.** Nothing is fitted or chosen on unknown-class validation flows either.
5. **A number is quoted only if it appears in a `results/summaries/*_table.md` made by `scripts/make_tables.py`.** One-seed numbers are labelled.
6. **A failed criterion is reported as failed.** Thresholds are not moved after the number is seen.
7. **Never edit `configs/train/ft_pat_d1.yaml`** or reuse an existing run name for a different configuration.
8. **Before any Kaggle kernel push**, every checkpoint that exists only on Kaggle is pulled or pushed to the slim checkpoint dataset.
9. **No Docker artefacts** in the repository.
10. **Checkpoints and `results/` are never committed.**
11. **No claim about detecting fraud, malware or attacks.** The system classifies services and flags unknown ones.

## 4. Commands

```
.venv\Scripts\activate
pytest                                   # whole suite
pytest -m "not integration"              # skip loopback-socket tests
pytest tests/path/test_file.py::test_name
ruff check src/ tests/ scripts/
mypy src/
python -m adl_etc.training.run   --config configs/train/<name>.yaml
python -m adl_etc.evaluation.run --config configs/eval/<name>.yaml
python scripts/mlflow_import.py --src <pulled-dir> [...]
python scripts/make_tables.py
./scripts/kaggle_sync.sh check|push-code|push-kernel|pull-results|push-checkpoints
```

Real-data checks need no Kaggle: `load_split('configs/splits/d1_open.yaml', root='results/kaggle-full/shards')`.

## 5. Kaggle session checklist

**Before**

- [ ] Every checkpoint the session needs is mounted (the kernel's own latest output, or the slim checkpoint dataset).
- [ ] Every checkpoint that exists only in the current kernel version's output is pulled or pushed to the checkpoint dataset. A new version hides the old one's files.
- [ ] The suite passes in the venv pinned to Kaggle's versions if kernel-imported code changed.
- [ ] `git status` clean; `push-code` run from that commit (it writes `GIT_COMMIT` / `GIT_DIRTY`).
- [ ] `kernel/run_queue.yaml` is ordered seed by seed, every entry has a unique run name, and the total estimate is under 11 hours.
- [ ] Quota checked (30 GPU-hours per week per account).

**During**

- [ ] `kaggle kernels status`; logs need `-f` while running. A session can sit queued for hours.

**After**

- [ ] Pull with `--file-pattern`; **count the files** (a pull can stop early and exit 0); pull again if short.
- [ ] Every run's `state.json` says `finished` or `early_stopped`.
- [ ] `python scripts/mlflow_import.py --src <dir>`, then `python scripts/make_tables.py`.
- [ ] Copy `results/mlflow.db`, `results/summaries/` and new `best.pt` files to the backup location.
- [ ] One progress-log entry: kernel version, duration, what finished, the headline numbers, any lesson.

## 6. Writing a results document

Model: `docs/results-phase-3.md`.

- First lines: what is final (how many seeds) and what is not.
- Every number is in the generated table; a difference is a subtraction of two numbers shown there.
- One verdict per hypothesis, against the spec's own criterion, in the words "supported", "partly supported" or "not supported".
- A "Limits and what comes next" section.

## 7. Briefing an AI assistant

Give it one task and the files that task names. A brief that works:

```
Project: metadata-only early classification of encrypted flows (see CLAUDE.md).
Task: <plan file>, task <T-number>. Read that section and <spec file> first.
Files you may create or change: <list from the task>.
Rules: docs/playbook.md section 3.
Do: write the tests listed in the task first, then the code, then run
pytest, ruff and mypy. Plant one bug and show a test catches it.
Stop when the task's "Done when" line is true. Do not start the next task.
Report: what you built, test results, any number you measured, anything
in the spec that turned out wrong.
```

For a smaller model, also:

- Paste the task section into the brief instead of asking it to find it.
- Ask for one file at a time.
- Ask it to state, before coding, the shapes and dtypes of every array it will touch.
- Have it run the three checks itself and paste the output.
- Check its work against the task's tests, not against its own summary.

## 8. Definition of done

A task is done when its "Done when" line is true, the three checks are clean, a planted bug was caught, and the plan has a progress-log line. A phase is done when each exit criterion in its plan is marked met or explicitly left open with a reason, and `specs/README.md`, `plans/README.md`, `plans/ROADMAP.md` and `CLAUDE.md` say the same thing.

## 9. When something is unclear

1. Look for a decision point in `plans/ROADMAP.md`; each has a default.
2. Look in `docs/decisions.md`; the question may already be closed.
3. If the spec and the code disagree, the code that passed tests on real data is the evidence. Record a spec correction in the plan's table and fix the spec.
4. If a real choice remains, pick the option that keeps results comparable with what is already measured, write the reason in the progress log, and continue.

## 10. Backups

`results/` is gitignored and exists once. After every session that produces something:

- `results/mlflow.db` and `results/summaries/` → the backup location.
- Every `best.pt` of a run that appears in a results table → the backup location **and** the slim Kaggle checkpoint dataset (`./scripts/kaggle_sync.sh push-checkpoints`).

The D1 export can be rebuilt from the public Kaggle mirror by the export kernel; checkpoints cannot be rebuilt without GPU hours.
