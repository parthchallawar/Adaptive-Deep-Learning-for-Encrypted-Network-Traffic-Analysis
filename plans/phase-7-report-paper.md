# Plan: Phase 7 — remaining experiments, report and paper

- **Spec:** `specs/021-report-and-paper.md`
- **Status:** not-started. **Forward draft**, written 2026-11-17.
- **Starts when:** phase 4 closes (writing can begin then; phases 5 and 6 add sections).
- **Roadmap:** [`ROADMAP.md`](ROADMAP.md).

## Purpose

Close the experimental gaps that a reader would ask about, then produce the final-year report, a paper draft and the presentation. Every number in them comes from `scripts/make_tables.py` output.

## Part A — experiments still owed

Ordered by how much the argument needs them. GPU-hours are estimates from measured throughput (`specs/015`, phase-3 log).

| # | Experiment | Why | GPU-h | Decision |
|---|---|---|---|---|
| A1 | Tuned recipe at 10% labels, scratch and SSL, seeds 0–2 | H1's second criterion was judged on one seed of the *default* recipe; it needs the recipe the paper recommends | ~1.5 | run |
| A2 | B5 (full-flow PAT) and B2 (CNN) at seeds 1–2 on `d1_open`, with robustness evaluation | Both appear in the main table with one seed | ~4 | run (roadmap DP-7a) |
| A3 | `abl_noaug` retrained (4 epochs) and robustness-evaluated | Separates "augmentation" from "prefix training" as the cause of the robustness result, the strongest finding of phase 3 | ~0.7 | run |
| A4 | MPM (masked-only) pretraining, then tuned fine-tune, seed 0 | Spec 000's H1 names the comparison with masked-only SSL; it was never fine-tuned | ~4 | run if quota allows; otherwise state it as untested |
| A5 | Low-label pairs at 1% with the tuned recipe | Completes the label-efficiency curve | ~0.5 | optional |
| A6 | 30pktTCNET public weights as a baseline | Closest metadata-only pretrained model | 0–1 | only if the weights load within a day (DP-7b) |

Before pushing any of these: secure every checkpoint that is only on Kaggle (roadmap checkpoint list). Queue them seed by seed so a cut session leaves complete seeds. After each session: pull, count files, `scripts/mlflow_import.py`, `scripts/make_tables.py`.

Then re-run phase 4's score pass and replays for any new headline run, so every model in the main table has policy and unknown numbers.

## Part B — figures and tables

All produced by scripts; none drawn by hand.

| Item | Source | Script |
|---|---|---|
| Main table: accuracy at K = 1, 3, 5, 8, 30, AUC_K, macro-F1, ECE, unknown AUROC, three seeds | `phase3_table.md` §1 + A2 | `make_tables.py` |
| Accuracy-vs-K curves, all models | reports | `plotstyle` |
| Robustness table and plot (drop, reorder) | `phase3_table.md` §2 + A3 | `make_tables.py` |
| Label-efficiency plot (100%, 10%, 1%) | A1, A5 | `make_tables.py` |
| Pareto fronts: Safe-Stop vs ECHO, CAPE, RL | `phase4_table.md` | `plotstyle.plot_pareto_fronts` |
| AUROC(K) for the unknown score and its parts | `phase4_table.md` | `plotstyle.plot_auroc_vs_k` |
| Budget tracking under drift and disturbances | `phase4_table.md` | `plotstyle.plot_budget_trace` |
| Weekly decay and drift signals | `phase5_table.md` | `plotstyle.plot_weekly_decay` |
| Efficiency table and accuracy-vs-cost scatter | `efficiency.md` | `scripts/benchmark.py` |
| Ablation table (five design ablations + phase-4 ablations) | `phase3_table.md` §3, `phase4_table.md` | `make_tables.py` |
| System diagram | spec 000 | redraw once, by hand, in the report tool |

## Part C — writing

Outline, with where each section's material already exists:

| Section | Source |
|---|---|
| 1 Introduction: encrypted traffic, why early, why unknown, why a budget. One motivating case: the 2026 payment-API frauds where a trusted cloud address was abused (motivation only) | `docs/research-analysis.md` §1, `docs/future-work.md` FEP-1 |
| 2 Related work | `docs/research-analysis.md` §2 and §10 (re-check for papers published since September 2026) |
| 3 Problem setting and definitions | spec 000 "Key definitions", spec 004 |
| 4 Data and protocol | `docs/datasets/cesnet-tls-year22.md`, spec 004, `configs/splits/d1_open.yaml` |
| 5 Method: PAT, SSL, prefix training, calibration | specs 006–008 |
| 6 Method: unknown score, Safe-Stop, controller | specs 010, 009, 011 |
| 7 Results: early accuracy and H1 | `docs/results-phase-3.md` |
| 8 Results: robustness and H5 | `docs/results-phase-3.md` |
| 9 Results: H2, H3, H4 | `docs/results-phase-4.md` |
| 10 Results: drift and efficiency | `docs/results-phase-5.md` |
| 11 System and demo | specs 016–018, `docs/demo.md` |
| 12 Limitations | each results document's "Limits" section, `docs/gaps-and-risks.md` |
| 13 Conclusion and future work | `docs/future-work.md` |
| Appendix: reproducibility | configs, commit hashes, hardware, Kaggle sessions |

Writing rules:

- A hypothesis that failed is reported as failed (H5 is "not supported"; say so in the abstract-level summary too).
- One-seed numbers are labelled "one seed" every time they appear.
- No claim about detecting fraud, malware or attacks. The project classifies services and flags unknown ones.
- Validation numbers are never presented as results.
- The comparison with CAPE-Net and FastFlow is with **our re-implementations of their decision rules on our backbone** (P-CAPE, P-RL), stated in those words.

## Part D — presentation and viva

- Slides: problem (with the news headline as slide 1), idea, data, method in three pictures, four result slides (early accuracy, robustness, stopping and budget, drift), demo, limits, future work.
- A list of likely questions with answers, kept in `docs/viva-notes.md`: why metadata only; why 30 packets; why not just a threshold on confidence; what "unknown" means and does not mean; why H5 failed; what would be needed to catch a payment fraud (see FEP-1: an enrolment step, genuine traffic, and a test, none of which exist).
- The five-minute demo from `docs/demo.md`, rehearsed twice.

## Success criteria

1. Every table and figure regenerates from one command on a clean tree.
2. The main table has three seeds for every row it contains.
3. Report and paper draft complete, with a limitations section that names every failed or untested criterion.
4. Presentation and demo rehearsed.

## Risks

| Risk | Mitigation |
|---|---|
| Kaggle quota runs short | A1–A3 first (~6 GPU-h); A4–A6 are optional and their absence is stated |
| New related work appeared | Re-search section 2's topics before writing; update the comparison table |
| Results change the story late | The roadmap's outcome table already says what is claimed under each outcome |
| Numbers copied by hand drift from the tables | Tables are included from generated `.tex`; prose numbers are checked against them in one final pass |

## Progress log

- **2026-11-17.** Forward draft written. Not started.
