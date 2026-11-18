# Gaps, debt, limits and failure points

- **Written:** 2026-11-17, at the end of phase 3, from a review of every spec, plan, results document and the files on disk.
- **Use:** check this list at the start of each phase. Each item names the action and the phase that owns it. Strike an item through, with a date, when it is closed.

## A. Gaps in the current plan

| # | Gap | Why it matters | Action | Owner |
|---|---|---|---|---|
| ~~A1~~ | ~~The nine headline checkpoints (tuned scratch, tuned SSL, B3; seeds 0–2) were mostly not on local disk; the tuned pair's seed 0 was only in the superseded v16 output~~ **Closed 2026-11-17:** all nine are local, hash-verified against their calibration files, and in the Kaggle checkpoint dataset (11 runs incl. B5 and the pretraining run). Seed 0 came from `scripts/kaggle_pull_version.py` (version 16), no retrain | | | phase-4 T0, done |
| A2 | H1's low-label criterion was judged on one seed of the default recipe | The paper recommends the tuned recipe; the criterion should be judged on it | Tuned recipe at 10% labels, three seeds (~1.5 GPU-h) | phase-7 A1 |
| A3 | The masked-only SSL comparison named in H1 was never fine-tuned | A reader will ask whether next-packet prediction beats plain masking | MPM pretrain + tuned fine-tune (~4 GPU-h), or state it as untested | phase-7 A4 |
| A4 | The robustness result does not separate augmentation from prefix training | It is the strongest phase-3 finding and its cause is unknown | Retrain `abl_noaug`, robustness-evaluate (~0.7 GPU-h) | phase-7 A3 |
| A5 | B5 and B2 have one seed | They are rows in the main table | Seeds 1–2 (~4 GPU-h) | phase-7 A2 |
| A6 | Packets-to-decision (H1's third criterion) is unmeasured | It needs a stopping policy | Falls out of phase-4 T3 | phase 4 |
| A7 | The 30 unknown classes are the same in `val` and `test_id` | Anything tuned on unknown val flows contaminates the test | Fixed defaults; a test that unknown val flows cannot change a fitted artefact | phase-4 D4, T2 |
| A8 | Spec 000 promises cross-protocol (D2 QUIC) and cross-dataset (D3 ISCX) evaluation; D2 was never exported and D3 was only used for pipeline checks | Contribution C4 is narrower than written | Default: state it as a limitation and future work (roadmap DP-5b). To close it instead: one export-kernel run for D2 and a `d2_quic` score pass | phase 5 or 7 |
| A9 | CAPE-Net, FastFlow and 30pktTCNET are not reproduced | "Beats prior work" cannot be claimed directly | Compare against P-CAPE and P-RL, our versions of their decision rules on our backbone, and say so; try 30pktTCNET's public weights if cheap | phase 4, phase-7 A6 |
| A10 | Two phase-2 exit items are still open: a three-seed B3-on-D4 run, and a genuine mid-flight Kaggle pause-and-resume | Recorded as open, never closed | B3-on-D4: close as "will not do" (D4 is 90% duplicate flows and is used only for pipeline and anomaly checks). Resume: either prove it once with a deliberate short session (~20 GPU-min) or close as accepted risk, since every real session so far has finished inside the limit | phase-4 T8 |
| A11 | `agents/` holds only a README and a template; the project's own multi-agent pipeline was never defined | A convention with no content confuses a new reader | Leave empty and say so in `agents/README.md`, or delete the convention. `docs/playbook.md` now covers how work is handed to an assistant | phase-4 T8 |
| A12 | No spec covered the final report | The last phase would be improvised | Spec 021 written | done 2026-11-17 |

## B. Missing or thin specifications

| # | Missing | Where it is now defined | Action |
|---|---|---|---|
| B1 | The compact score artefact that every phase-4 experiment reads | phase-4 plan T1 | Fold into spec 004 when T1 lands (correction C8) |
| B2 | Open-set, policy and budget metrics (open-set macro-F1, `pareto_area`, tracking error, settling time) | phase-4 plan T2/T3/T5 | Same |
| B3 | P-RL baseline (deferred twice, never specified beyond a name) | phase-4 plan T3 | Spec 005/009 correction C4 |
| B4 | Demo data source: D1 has no PCAPs, so a PCAP demo cannot show in-distribution traffic | phase-6 plan (shard replay) | Spec 016 gains `POST /ingest/shards` |
| B5 | Backup policy for `results/` (gitignored, one copy, on a laptop) | section C below | One paragraph in `docs/playbook.md`; done there |
| B6 | Report and paper | spec 021 | done |

## C. Technical debt

| # | Debt | Risk | Action |
|---|---|---|---|
| C1 | `results/` is the only copy of checkpoints, `mlflow.db`, pulled runs and summaries | A disk failure loses ~30 GPU-hours and every result | After each Kaggle session copy `results/mlflow.db`, `results/summaries/` and every `best.pt` to a second location; keep headline checkpoints in the slim Kaggle checkpoint dataset too |
| C2 | Kaggle's image is older than the laptop (numpy 2.0.2 / torch 2.10 vs numpy 2.5.3 / torch 2.14) | Code that passes locally fails on Kaggle | Re-run the suite in the pinned venv before every push that changes kernel-imported code (`CLAUDE.md` has the command) |
| C3 | `configs/train/ft_pat_d1.yaml` is kept only as the ablation reference; the recommended recipe is the `_tuned` pair | Editing the old file reuses run names the queue treats as finished | Never edit it; new variants get new config files and new `variant` names |
| C4 | `kernel/kernel.py` is one 492-line file | Hard to change safely | It must stay one file (Kaggle ships only `code_file`); keep `tests/test_kernel_smoke.py` green and add a queue test for every new entry type |
| C5 | `data/prefix_stats.py` has no consumer since B1 was removed | Dead code | Keep (tested, small) or delete in phase 7; do not extend |
| C6 | `evaluation/metrics.logits_at_k` survives only for a regression test | Someone may call it | Leave the docstring warning; do not use in new code |
| C7 | Phase 2's B3 seeds on `d1_main` were trained before the seeding fix and are not reproducible by seed alone | Only matters if they are quoted | The paper quotes `d1_open` runs only |
| C8 | Calibration, early stopping and threshold choice all use `val` | Mild optimism in val numbers | Report `test_id` and `test_drift` only (already the rule) |
| C9 | `CLAUDE.md`'s "Current state" is one very long paragraph | Easy to misread, hard to update | Rewrite as a short status table pointing at `plans/ROADMAP.md` when phase 4 closes (T8) |
| C10 | Training-run evaluation is saved under `<run>/eval/val/`, test evaluation under `robust/<run>/eval/test_id/` | Two places to look | Phase-4 score files go under `<run>/eval/<split>/scores/` for every split; document the layout once in spec 004 |

## D. Scalability limits

| Limit | Today | When it bites | What to do |
|---|---|---|---|
| Evaluation cap | 250,000 flows per split | Rare classes have few test flows; per-week drift windows | Per-period cap in the score pass (phase-4 T1) |
| Full logits | ~2 GB per run per split, slow to pull (~0.6 MB/s from Kaggle) | Any offline policy work | Score files (~150 MB) replace them for phase 4 onward |
| Training data in memory | 12.5M flows as int16 fits in Kaggle's ~29 GB | A larger D1 sample than 3% | Not needed; if ever, shard-wise loading in `FlowBatches` |
| GPU quota | 30 GPU-hours per week per account, 12-hour sessions, queue waits of hours | Phase 7's extra seeds | Order experiments by value; seed-by-seed queues |
| Service | One process, SQLite, target 5,000 packet events/s | Never, for a demo | PostgreSQL is a URL change; horizontal scaling is out of scope |
| Controller window | 2,000 flows | Low-traffic demos | `min_flows = 200` freeze already specified |

## E. Failure points to watch

| Where | Failure | Guard |
|---|---|---|
| Kaggle outputs | Older kernel versions' files become unreachable | Pull or `push-checkpoints` before every push |
| Kaggle pulls | Stops early, exit code 0 | Count files after every pull, and run `python scripts/check_shards.py --root results/kaggle-full/shards` for the shard export (it checks every array's exact size against `meta.json`) |
| Local D1 shards | Five weeks (48-52) were silently incomplete until 2026-11-17: `load_split` reads only labels and timestamps, so the damage surfaced only when a model first read `ppi.npy` | `scripts/check_shards.py` after every pull; repaired by re-pulling `parthchallawar/adl-export-d1` for those weeks |
| `kaggle kernels output <slug>/<version>` | The CLI parses the version suffix and ignores it, so it always returns the latest version's files; the API's *listing* endpoints reject a version (404), so an older version's files cannot be listed | `python scripts/kaggle_pull_version.py <owner>/<slug> <version> <file path> ...` downloads named files from a specific version (the download request does take one). Still pull what you need before the next push; and keep every headline checkpoint in the Kaggle checkpoint dataset |
| Score passes on CPU | The PAT scores ~320 flows/s here (313 at 4 threads, 326 at 10), so a full set (~1.3M flows per run) takes about 70 minutes per PAT | Run full passes in one Kaggle session; use small samples for development |
| Kaggle start-up | A kernel can fail before Python starts, with an empty log (v12) | Re-push; check quota; keep sessions self-contained |
| Score files | Computed from a different checkpoint than the calibration | `checkpoint_hash` in every artefact; loaders raise |
| Mahalanobis | Singular covariance at small K or for rare classes | float64, shrinkage 0.1, classes under 50 flows excluded |
| Policy inputs | NaN scores | Treated as `p_safe = 0`, `u = +inf`, counted and reported |
| Controller | Integral wind-up on an infeasible target; oscillation | Clipped integral; gain halving; tests for both |
| Evaluation | Tuning on test by accident | Defaults fixed in advance; sweeps on test labelled "sensitivity" |
| Windows | Killed `uvicorn`/`mlflow` workers keep the port; SQLite files stay locked | Rehearsal checklist; one database file per run |
| Datasets | D4: 90% duplicate flows, shard order by capture file. D3: 81% one class | Already documented in `CLAUDE.md`; never report a flow-level-split score |

## F. Opportunities

| Idea | Gain | Status |
|---|---|---|
| Run phase-4 score passes on the laptop CPU | No GPU quota, no Kaggle round-trips | phase-4 T0 measures it |
| Replay every policy, threshold and controller experiment from score files | Seconds per experiment instead of a model run | phase-4 D1 |
| Use the session's second, idle T4 for a second queue entry | Could halve wall-clock time | Unverified whether Kaggle counts quota per session-hour; test with two short entries before relying on it |
| Screen ablations at a reduced budget (4 epochs) | ~3× cheaper | Already done in phase 3; keep |
| Seed-by-seed queue order | A cut session still leaves complete seeds | Already done in Session 4; keep |
| One shared plotting module for report and dashboard | No duplicated figure code | phase-6 S7 |
