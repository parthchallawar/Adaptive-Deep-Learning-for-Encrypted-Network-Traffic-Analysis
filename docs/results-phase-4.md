# Phase 4 results: the H2, H3 and H4 verdicts

**Status: three seeds for the headline comparison, written 2026-11-18.** Source:
`results/summaries/phase4_table.md` and `phase4_criteria.json`, which
`scripts/make_phase4_tables.py` regenerates from each run's `eval/phase4.json`. Every
number below appears in that table; section references (§1 to §5) are to its sections.
A difference quoted here is the subtraction of two numbers shown there. Verdicts are
copied from §5 of that table, never restated by hand, and no bar was moved after the
numbers were seen. Where a metric was *added* after seeing a failure, this document says so.

## What was run

Nine runs on `d1_open` (150 known classes, 30 held out as unknown): the tuned PAT with SSL
init (`ft-pat-ssl_tuned-d1open`, the headline group), the tuned PAT from scratch
(`ft-pat-tuned`), and B3 GRU (`bl-gru-main`), three seeds each. One Kaggle session (kernel
v18) scored 250,000 `val` flows, 250,000 `test_id` flows and 360,000 `test_drift` flows
(20,000 per week over 18 weeks) per run and built each run's report. Everything fitted
(Gaussians, per-K thresholds, controller gains, the first threshold) used validation scores
only; the 30 unknown classes are the same in `val` and `test_id`, so no unknown-class
validation flow selected anything. The B3 GRU has no projection head: it has the
closed-world part of H2 only, no H3, no H4.

## Verdict table (headline group, three seeds)

| Hypothesis | Verdict | One line |
| --- | --- | --- |
| H3, unknown score AUROC bars | **met** | u: 0.910 at K=8 (bar 0.85), 0.903 at K=30 (bar 0.90) |
| H3, beats energy alone at some K ≤ 8 | **met** | gain 0.113, all of it at K = 2 and 3 |
| H3, rejects unknowns ≥ 30% earlier than the fixed K=10 gate | **not met** | 4.0% at f = 0.02, 28.3% at f = 0.05 |
| H2, Safe-Stop beats P-CAPE at mean K = 6 | **met** | +9.0 points |
| H2, Safe-Stop beats P-ECHO by ≥ 1.5 points at mean K = 6 | **not met** | −0.7 points |
| H2, Safe-Stop beats P-RL | **not met** | −1.1 points; largest Pareto area is P-RL (0.858), Safe-Stop 0.785 |
| H2, reject-first commits fewer unknowns, no extra packets | **met** | 44.8% against 46.5%; 0.0 extra packets |
| H4, tracking error on the drift weeks ≤ 5% | **met** | 2.8 / 3.6 / 4.5% at budgets 4 / 6 / 8 |
| H4, a static threshold drifts ≥ 15% (the contrast) | **not met** | static: 3.1 / 3.5 / 6.2% |
| H4, no oscillation | **not met** | 2 events in total, one each in two runs |
| H4, settles within 5 windows after a target step | strict: **not met** (up to 19); noise-robust: **met** (at most 2) | see below |
| H1, third criterion (packets to equal accuracy, SSL vs scratch) | 80%: **met** (12.4% fewer); 85%: **not met** (8.6%, bar 10%) | |

## H3: the unknown score

**What held.** The combined score `u` reaches both absolute bars (§1). At packets 2 and 3 it
is much better than plain energy (0.766 against 0.653 at K=2, 0.783 against 0.748 at K=3),
which is where an early reject matters most.

**What did not.** From K = 5 on, plain energy is as good or better than `u` (K=8: 0.919
against 0.910; K=30: 0.915 against 0.903), so the 0.113 "gain over energy" comes entirely
from the first three packets. The surprise term alone is weak (0.50 at K=1, 0.79 at best)
and the Mahalanobis term falls to 0.80 at K=30. In practice the rule is: use `u` early, and
plain energy with per-K quantiles later is no worse.

**The earliness claim fails.** At the matched 5% known-rejection rate the rule rejects
unknowns 28.3% earlier than the fixed K=10 gate (bar 30%); at 2% only 4.0% earlier (§2).
The tuned scratch model does worse (18.5% and 2.8%): the SSL model helps here too.

**Fallback story.** The unknown score is usable from the second packet and clearly better
than energy there, and at a 5% false-alarm target an unknown flow is rejected about
three packets earlier than by a fixed K=10 gate on the SSL model (mean rejection K 7.2
against 10), but at a 2% target the gain is under half a packet. The "30% earlier" claim is not supported and the thesis must not
make it. Ideas that were *not* tried (they would have to be chosen on held-out known
classes treated as fake unknowns, never on test): allowing a reject from packet 2, a
non-uniform spread of the false-alarm budget over K, and per-K choice of score. They are in
`docs/future-work.md`.

## H2: the stopping rule

**Closed world (§3).** At a mean of 6 packets: Safe-Stop 88.1%, P-ECHO 88.8%, P-RL 89.2%,
P-CAPE 79.1% (SSL model, three seeds). P-RL has the largest Pareto area (0.858), then
P-ECHO (0.854), then P-CAPE (0.786) and Safe-Stop (0.785), the last two within one standard
deviation of each other. Safe-Stop cannot stop before about packet 3 (smallest
mean K 2.98) while P-ECHO and P-RL reach 1.0, which costs it Pareto area by construction.
Safe-Stop beats P-CAPE clearly. It does not beat P-ECHO or P-RL.

**Open world (§4).** Checking for unknowns *before* committing sends fewer unknown flows to
a class (44.8% against 46.5% with commit-first), at the same mean K for known flows
(5.44), and costs 0.2 points of known accuracy (83.7 against 83.9). Still, almost half of
the unknown flows are committed to a class.

**Fallback story.** The claim that survives is about safety, not accuracy: a reject-first
stopping rule is the safer order in the open world at no cost in packets. The thesis must
not claim that Safe-Stop is the most accurate or the most economical stopping rule.

## H4: the budget controller

**The gains.** The first run's gain search stopped at kp = 2.0 and every run picked that
edge, with 17-18 windows to settle. A wider search on validation scores (kp up to 6) put the
best pair at kp = 3 in all six PAT runs; kp ≥ 6 oscillates on validation. All numbers above
are from the wider search; the first-pass reports are kept in `results/phase4_v1/`.

**What held.** Over the 18 drift weeks the controller keeps the mean packets-per-flow within
5% of the target at all three budgets (§4, §5). Under the target-step disturbance
(8 → 5 → 8) its mean error is about 6% against 26% for the static threshold, which is where
it earns its place.

**What did not.** The contrast bar fails: a static threshold calibrated once on validation
already holds the budget to 3.1 / 3.5 / 6.2% on these drift weeks (§4), so the data's drift
is too mild to show a large benefit (plan decision DP-4c: report this honestly, and lead the
H4 section with the disturbance results, not the drift weeks).

**Settling time: two metrics, and why.** The first metric counts windows until three
consecutive windows are each within 5% of the target. A window's mean K is noisy (about 5%
for 2,000 flows), so that count is hit or missed by chance even when the controller sits on
the target: after an 8 → 5 step the controller is on target in 2-3 windows (errors 6.7%, 2.6%,
1.5%), yet the strict count reads 19. **After seeing this** a second metric was added: the
mean signed error of three windows within 5%. It is tested to ignore noise and still flag a
slow or biased controller (`tests/evaluation/test_controller_replay.py`). Both are reported.
Strict: up to 19 windows, not met. Noise-robust: at most 2, met. A reader who prefers the
strict metric should read H4's settling bar as not met.

**Oscillation.** The controller halves its gains when six consecutive windows alternate in
sign with an error above 5%. With 5% window noise that can fire by chance, and it did twice
in the whole grid (one run, one budget, one data span each). Bar: zero events: not met.

## The decisions this phase took

- **DP-4a** (does `u` beat energy by ≥ 1 point at some K ≤ 8?): yes, at K = 2 and 3, so `u`
  stays in, and later K could use energy with per-K quantiles.
- **DP-4b** (ELECTS, the next-packet-entropy knock-out): not run; descoped (spec correction
  C5) and listed in `docs/future-work.md`.
- **DP-4c** (is the static threshold already within 5%?): yes on these drift weeks; reported.

## What phase 4 does and does not support

Supported: the PAT-based rules beat the B3 GRU baseline by about 7 points of closed-world
Pareto area (0.854 against 0.779, §3); `u` is a strong unknown score from the second packet;
reject-first is the safer open-world order at no cost in packets; the controller holds a
packet budget and recovers quickly from a target change. Not supported: earlier unknown
rejection than a fixed-K gate by 30%; Safe-Stop as the best closed-world stopping rule;
a large controller advantage under this drift. Next phases (drift monitoring, efficiency)
read the score files and the replay harness this phase built.

## Limits

- Three seeds for the headline group; every number is mean ± std over seeds (n shown).
- One drift stream (the 18 weeks after the test weeks); the drift is mild.
- The unknown classes are the same in `val` and `test_id`: nothing was selected with them,
  but the unknown AUROC is for these 30 classes only.
- P-RL is same-backbone offline fitted-value stopping on `(conf, margin)`, not a
  re-implementation of FastFlow (spec correction C4).
- The settling metric was added after seeing the first one fail; both are shown.
