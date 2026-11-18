# Phase 5 results: drift decay, label-free drift signals, and cost

**Status: three seeds for every group, written 2026-11-18.** Source:
`results/summaries/phase5_table.md` and `phase5_criteria.json`, which
`scripts/make_phase5_tables.py` regenerates from each run's `drift.json`,
`drift_monitor.json`, `policy_cost.json` and from `efficiency.json`. Every number below appears
in that table (sections §1 to §5). A difference quoted here is the subtraction of two numbers
shown there. No bar was moved after the numbers were seen. Nothing in this phase trains a model:
it replays the score files of phase 4 (nine runs) and times the architectures on this laptop.

## What was run

* **Weekly decay (T1).** For each of the 4 `test_id` weeks and 18 `test_drift` weeks: accuracy
  and macro-F1 on known flows at K = 5, 8, 30, the same under Safe-Stop at a threshold fixed on
  validation (mean K about 6), the mean K that fixed threshold needs, and (PAT only) the unknown
  AUROC at K = 8. Every week is cut to 18,000 flows, because macro-F1 over 150 classes is lower
  on a smaller sample and the drift weeks hold about a third of a test week's flows.
* **Drift monitor (T2).** Six label-free signals over windows of 5,000 decided flows, a CUSUM
  per signal, a combined level. References and the CUSUM threshold `h` come from validation
  windows only (`h` is the smallest value that gives no alarm on validation). Two modes: a
  static threshold, and the budget controller running (then `mean_k` and `commit_rate` are
  informational and the controller's own `theta_commit` is watched).
* **Efficiency (T3).** Parameters, FLOPs, CPU latency and the streaming step of the three
  architectures; packets and flow-time to decision of each policy.

## H-level verdicts (specs 004, 012, 013)

| Spec | Criterion | Verdict |
| --- | --- | --- |
| 004 | weekly decay curves for three models, three seeds | **met** |
| 012 | the monitor warns or alarms within the drift horizon (3 of 3 seeds, both modes) | **met** |
| 012 | correlation with the weekly macro-F1 reported | reported: class-prior KL −0.90 (best) |
| 013 | PAT full-flow latency ≤ 5 ms | **met** (2.49 ms) |
| 013 | PAT parameters < 2 M | **met** (885,821) |
| 013 | PAT streaming step ≤ 1 ms | **not met** (1.65 ms) |

## Decay over 18 weeks (§1)

All three models decay steadily and almost linearly. From the test weeks to the last four drift
weeks (mean of three seeds), accuracy under Safe-Stop falls 6.4 points for the SSL PAT
(0.886 → 0.823), 6.6 for the scratch PAT and 7.7 for the GRU (0.841 → 0.763); macro-F1 falls about
12 to 13 points for all three (0.851 → 0.723 for the SSL PAT). The weekly accuracy slope is about
−0.0027 for the PAT and −0.0041 for the GRU. So the PAT loses less *accuracy* than the GRU but not
less *macro-F1*: the classes that decay are the rarer ones, and the PAT's advantage does not
reach them. SSL init and scratch decay alike (−6.4 against −6.6). The unknown AUROC at K = 8 also
falls, 0.897 → 0.858. A fixed commit threshold needs only 0.06 to 0.34 more packets at the end
(6.23 → 6.42 for the SSL PAT), so the decay shows up in accuracy, not in packets.

**Drift × earliness.** Spec 000 hoped that early prefixes would decay faster than late ones. They
do not: accuracy at K = 5 falls 6.3 points and at K = 30 6.0 for the SSL PAT; the difference
(−0.3 ± 0.3 points; −0.7 for the scratch PAT) is within noise of zero, and for the GRU the late
prefix falls *more* (+1.2 points in the other direction). **That interaction is not supported at
this scale**; the thesis must not claim it.

## The label-free drift signals (§2)

Spearman correlation of each signal's weekly mean with the week's macro-F1 under Safe-Stop, over
the 22 weeks (the test weeks and the drift weeks), mean over three seeds:

* **Strong:** class-prior KL −0.90 (SSL PAT), −0.91 (scratch PAT), −0.85 (GRU); next-packet NLL
  −0.89 / −0.90 (PAT only, it needs that head).
* **Moderate:** mean `p_safe` at commit +0.68 / +0.60 (PAT), only +0.32 (GRU); commit rate
  +0.61 / +0.56 / +0.65.
* **Weak or absent:** mean K −0.13 / +0.07 (PAT; −0.55 for the GRU), reject rate −0.23 / −0.24,
  and the controller's `theta_commit` +0.22 / +0.03 (one SSL seed never alarmed on it).

So the signals one would reach for first (how many packets the policy reads, how often it
rejects) do not follow the decay; **what the model commits to** (class-prior shift) and **how
surprised its next-packet head is** do. Both need no label.

**Caveat on those correlations.** There are 22 weeks and the signals and the macro-F1 both trend
over time, so any signal that moves steadily will correlate highly; the Spearman values rank the
signals, they do not prove a cause.

**When do they alarm?** The first week in which macro-F1 under Safe-Stop is 3 points below the
test weeks' mean is week 37 or 38. With a static threshold, class-prior KL first alarms in
week 38 in nine of nine runs (week 38 or 39 with the controller on): **in the same week as the
drop or one week after it (0 to −1 weeks of lead), not before it.** The next-packet NLL, commit
rate and reject rate (for the GRU also mean K and `theta_commit`) alarm in weeks 32 to 34, four to
six weeks earlier, but those weeks are still in the `test_id` period right after
validation. The macro-F1 at K = 8 does slip from 0.852 (week 31) to 0.823 (week 33) in the
headline seed-0 run, so that may be early detection of a small real change; it may also be a
validation-to-test shift that the signals read as drift. **With no stationary stretch of data,
the false-alarm rate cannot be measured**, so this phase can claim sensitivity (warnings in every
run, both modes) but not specificity.

The combined monitor (a warning when one signal alarms, an alarm when two do within three
windows) warns in weeks 32 to 33 and alarms in weeks 33 to 34 in static mode for every seed; with
the controller on, the SSL PAT's alarm moves to weeks 33, 39 and 34 (§2, last table), because the
controller hides some of the movement.

## Cost (§3, §4)

* **Size and compute.** The PAT has 885,821 parameters and 55.3 MFLOP per flow, against 730,390
  and 43.6 for the GRU and 444,502 and 16.7 for the CNN. B5, the SSL PAT and the scratch PAT share
  one architecture, so one row stands for all three.
* **Full flow, CPU, batch 1, 4 threads:** PAT 2.49 ms (p95 3.30), GRU 2.03 ms, CNN 0.62 ms. At
  batch 64 the PAT needs 0.345 ms per flow.
* **Streaming.** The PAT's KV-cache step takes **1.65 ms**, above spec 013's 1 ms bar. Thirty
  steps cost 53.8 ms, 22 times a full pass (2.49 ms), although the FLOP totals agree (54.4 against
  55.3 MFLOP, within the 5% consistency test): the step is dominated by per-operation overhead of
  eager PyTorch on tiny tensors, not by arithmetic. Using 1, 2 or 4 threads changed nothing
  (1.68, 1.80, 1.74 ms). The GRU's recurrent step is 0.20 ms. The CNN cannot stream: deciding at
  every packet costs one full pass per packet, 501 MFLOP over a flow, nine times the PAT's. A
  deployed PAT would batch several flows per step or compile or export the step; neither was done.
* **Observation cost.** At about 6 packets on average, a decision comes after about 1.3 s of the
  flow's own clock (SSL PAT: 1,274 ms for Safe-Stop, 1,301 for P-RL, 1,423 for P-ECHO), 16 to 18% of
  the flow's duration; the 95th-percentile flow needs 13 to 15 packets. At the same mean packets
  Safe-Stop is about 10% earlier in flow time than P-ECHO. P-CAPE could not be matched to 6
  packets on every model (7.4 packets for the SSL PAT, 5.5 for the scratch PAT), so its row is
  not a like-for-like comparison.

## Limits

* Timings are from one laptop (Intel, 10 cores, 23.7 GB) with other work stopped; they vary by
  about ten percent between rounds and will differ on another machine. The batch-1 latency is of
  random inputs of full length.
* The drift stream is one 18-week stretch of one dataset, with a mild decay. The signals were
  validated on it only.
* The CUSUM threshold `h` was raised until validation gave no alarm; it was not tuned on the
  weekly macro-F1.
* The canary check (spec 012's optional labelled alert) and label-free adaptation were not built.

## What phase 5 supports

Supported: the decay is steady and measurable, with the PAT losing less accuracy than the GRU
(not less macro-F1); two label-free signals (class-prior KL and next-packet NLL) follow the weekly
macro-F1 closely and a combined monitor raises warnings in every run; the PAT is small, with a
2.5 ms full-flow CPU latency. Not supported: that early prefixes decay faster; that any signal
warns *before* a 3-point drop without also firing early in the test weeks; a PAT streaming step
under 1 ms in eager PyTorch. The service (phase 6) therefore starts from batched or compiled
streaming steps, and its drift panel from class-prior KL and next-packet NLL.
