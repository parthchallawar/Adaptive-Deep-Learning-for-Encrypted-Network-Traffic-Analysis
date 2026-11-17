"""Phase 4's summary tables from the per-run ``phase4.json`` files (plan phase 4, T6/T7).

Input: a directory of ``<run>/eval/phase4.json`` (what ``kernel.run_score_entry`` writes and
``scripts/phase4_report.py`` writes), run names like ``ft-pat-ssl_tuned-d1open-s0``. Runs that
differ only in the trailing ``-s<seed>`` form a group; every number below is the mean and standard
deviation over the group's seeds, with the seed count shown, so a one-seed row can never pass for
a three-seed one.

:func:`criteria` turns the specs' acceptance rules (spec 010 for H3, spec 009 for H2 and H1's
third criterion, spec 011 for H4) into explicit comparisons, each returning the measured value,
the bar, and ``met``. A verdict in a results document must quote these, never override them.
No torch, no model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

SEED_RE = re.compile(r"^(?P<group>.+)-s(?P<seed>\d+)$")
POLICIES = ("safe_stop", "p_echo", "p_cape", "p_rl")
POLICY_LABEL = {"safe_stop": "Safe-Stop", "p_echo": "P-ECHO", "p_cape": "P-CAPE", "p_rl": "P-RL"}
BUDGETS = ("4.0", "6.0", "8.0")
AUROC_KS = ("1", "2", "3", "5", "8", "12", "20", "30")

Groups = dict[str, list[tuple[int, dict[str, Any]]]]


def load_reports(root: Path) -> Groups:
    """``{group: [(seed, report), ...]}`` from ``<root>/<run>/eval/phase4.json`` (or a flat
    ``<root>/<run>/phase4.json``)."""
    groups: Groups = {}
    for path in sorted(root.glob("*/eval/phase4.json")) + sorted(root.glob("*/phase4.json")):
        run = path.parent.parent.name if path.parent.name == "eval" else path.parent.name
        m = SEED_RE.match(run)
        if not m:
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        groups.setdefault(m["group"], []).append((int(m["seed"]), report))
    for members in groups.values():
        members.sort(key=lambda t: t[0])
    return groups


def mean_std(values: list[float | None]) -> tuple[float, float, int] | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def fmt(stat: tuple[float, float, int] | None, *, pct: bool = False, digits: int = 3) -> str:
    if stat is None:
        return "-"
    mean, std, n = stat
    if pct:
        mean, std = 100 * mean, 100 * std
        digits = 1
    body = f"{mean:.{digits}f}" if n == 1 else f"{mean:.{digits}f} ± {std:.{digits}f}"
    return f"{body} (n={n})"


def dig(report: dict[str, Any], *path: str) -> Any:
    """``report[path[0]][path[1]]...``; a list is indexed by a digit string. ``None`` if absent."""
    cur: Any = report
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        elif isinstance(cur, list) and key.isdigit() and int(key) < len(cur):
            cur = cur[int(key)]
        else:
            return None
    return cur


def column(members: list[tuple[int, dict[str, Any]]], *path: str) -> list[float | None]:
    return [dig(r, *path) for _, r in members]


def min_mean_k_at_accuracy(front: dict[str, list[float]] | None, target: float) -> float | None:
    """The fewest mean packets in a sweep (``fronts[policy]``) that reach ``target`` accuracy."""
    if not front:
        return None
    mk, acc = np.asarray(front["mean_k"]), np.asarray(front["accuracy"])
    ok = acc >= target
    return float(mk[ok].min()) if ok.any() else None


# --- the acceptance rules ---------------------------------------------------------------------


def _check(name: str, value: float | None, bar: str, met: bool | None) -> dict[str, Any]:
    return {"criterion": name, "value": value, "bar": bar, "met": met}


def criteria(groups: Groups, *, headline: str = "ft-pat-ssl_tuned-d1open") -> dict[str, Any]:
    """Each acceptance rule, evaluated on the seed-mean of ``headline``'s group (a rule that cannot
    be evaluated, e.g. no drift scores, reports ``met: None`` rather than guessing)."""
    out: dict[str, Any] = {"headline": headline, "H3": [], "H2": [], "H4": [], "H1_packets": []}
    members = groups.get(headline, [])
    if not members:
        return out
    n = len(members)
    out["seeds"] = n

    def mean_of(*path: str) -> float | None:
        s = mean_std(column(members, *path))
        return None if s is None else s[0]

    # H3 (spec 010): AUROC bars, beating energy alone, earliness against the fixed gate
    a8, a30 = mean_of("h3", "auroc", "u", "8"), mean_of("h3", "auroc", "u", "30")
    out["H3"].append(_check("AUROC of u at K=8", a8, ">= 0.85", None if a8 is None else a8 >= 0.85))
    out["H3"].append(
        _check("AUROC of u at K=30", a30, ">= 0.90", None if a30 is None else a30 >= 0.90)
    )
    gains = []
    for k in ("1", "2", "3", "5", "8"):
        u, e = mean_of("h3", "auroc", "u", k), mean_of("h3", "auroc", "energy_raw", k)
        if u is not None and e is not None:
            gains.append(u - e)
    best = max(gains) if gains else None
    out["H3"].append(
        _check(
            "best AUROC gain of u over raw energy, K <= 8",
            best,
            ">= 0.01 (else ship energy with per-K quantiles, plan DP-4a)",
            None if best is None else best >= 0.01,
        )
    )
    for i, fpr in enumerate((0.02, 0.05)):
        kr = mean_of("h3", "earliness", str(i), "k_reduction")
        out["H3"].append(
            _check(
                f"rejection-K reduction vs the K=10 gate at matched known rejection (f={fpr})",
                kr,
                ">= 0.30",
                None if kr is None else kr >= 0.30,
            )
        )

    # H2 (spec 009): Safe-Stop against every baseline at a mean of 6 packets
    acc = {
        p: v
        for p in POLICIES
        if (v := mean_of("h2", "closed_world", p, "acc_at_budget", "6.0")) is not None
    }
    area = {
        p: v for p in POLICIES if (v := mean_of("h2", "closed_world", p, "pareto_area")) is not None
    }
    if len(acc) == len(POLICIES):
        lead = acc["safe_stop"] - acc["p_echo"]
        out["H2"].append(
            _check("Safe-Stop minus P-ECHO accuracy at mean K=6", lead, ">= 0.015", lead >= 0.015)
        )
        for p in ("p_cape", "p_rl"):
            d = acc["safe_stop"] - acc[p]
            out["H2"].append(
                _check(f"Safe-Stop minus {POLICY_LABEL[p]} accuracy at mean K=6", d, ">= 0", d >= 0)
            )
    if len(area) == len(POLICIES):
        top = max(area, key=lambda p: area[p])
        out["H2"].append(
            _check(
                "largest Pareto area",
                area["safe_stop"],
                f"Safe-Stop must be largest (largest is {POLICY_LABEL[top]})",
                top == "safe_stop",
            )
        )
    rf = mean_of("h2", "open_world", "reject_first", "unknown_committed")
    cf = mean_of("h2", "open_world", "commit_first", "unknown_committed")
    if rf is not None and cf is not None:
        out["H2"].append(
            _check(
                "unknown flows committed: reject-first minus commit-first", rf - cf, "< 0", rf < cf
            )
        )
    dk = mean_of("h2", "open_world", "reject_first", "mean_k_known")
    dk0 = mean_of("h2", "open_world", "commit_first", "mean_k_known")
    if dk is not None and dk0 is not None:
        out["H2"].append(
            _check(
                "extra packets for known flows, reject-first vs commit-first",
                dk - dk0,
                "<= 0.5",
                dk - dk0 <= 0.5,
            )
        )

    # H4 (spec 011): the controller over the drift weeks, at each budget
    for b in BUDGETS:
        c = mean_of("h4", "budgets", b, "controller", "test_drift", "mean_abs_error")
        s = mean_of("h4", "budgets", b, "static", "test_drift", "mean_abs_error")
        if c is not None and s is not None:
            out["H4"].append(
                _check(
                    f"tracking error over the drift weeks, budget {b}: controller",
                    c,
                    "<= 0.05",
                    c <= 0.05,
                )
            )
            out["H4"].append(
                _check(
                    f"tracking error over the drift weeks, budget {b}: static",
                    s,
                    ">= 0.15 (the contrast)",
                    s >= 0.15,
                )
            )
    osc = sum(
        (dig(r, "h4", "budgets", b, "controller", sp, "oscillation_events") or 0)
        for _, r in members
        for b in BUDGETS
        for sp in ("val", "test_id", "test_drift")
    )
    if any(dig(r, "h4") for _, r in members):
        out["H4"].append(
            _check(
                "oscillation events, all budgets, all spans, all seeds",
                float(osc),
                "== 0",
                osc == 0,
            )
        )
        step = [
            w
            for _, r in members
            for w in (
                dig(r, "h4", "disturbances", "target_step_8_5_8", "controller", "settling_windows")
                or []
            )
        ]
        if step:
            out["H4"].append(
                _check(
                    "windows to settle after a target step (worst)",
                    float(max(step)),
                    "<= 5",
                    max(step) <= 5 and min(step) >= 0,
                )
            )

    # H1's third criterion (spec 007): packets to reach equal accuracy, SSL init against scratch
    scratch = groups.get(headline.replace("ft-pat-ssl_tuned", "ft-pat-tuned"))
    if scratch:
        for target in (0.80, 0.85):
            with_ssl = mean_std(
                [
                    min_mean_k_at_accuracy(dig(r, "h2", "fronts", "safe_stop"), target)
                    for _, r in members
                ]
            )
            from_scratch = mean_std(
                [
                    min_mean_k_at_accuracy(dig(r, "h2", "fronts", "safe_stop"), target)
                    for _, r in scratch
                ]
            )
            if with_ssl and from_scratch and from_scratch[0] > 0:
                red = 1.0 - with_ssl[0] / from_scratch[0]
                out["H1_packets"].append(
                    _check(
                        f"packets to reach {target:.0%} accuracy, SSL vs scratch (Safe-Stop)",
                        red,
                        ">= 0.10 fewer",
                        red >= 0.10,
                    )
                )
    return out


# --- rendering --------------------------------------------------------------------------------


def _table(header: list[str], rows: list[list[str]]) -> str:
    line = lambda cells: "| " + " | ".join(cells) + " |"  # noqa: E731
    return "\n".join([line(header), line(["---"] * len(header)), *(line(r) for r in rows)])


def render_markdown(groups: Groups) -> str:
    out = ["# Phase 4 tables", ""]
    out.append(
        "Generated by `scripts/make_phase4_tables.py` from the per-run `phase4.json` files. "
        "Every cell is mean ± std over the group's seeds (n shown). Numbers are test-split "
        "numbers; every fit behind them used validation scores only. A criterion is quoted "
        "from section 5, never restated."
    )
    names = sorted(groups)
    out += ["", "## 1. H3: unknown score AUROC by prefix length K (unknown = positive)", ""]
    rows = []
    for g in names:
        if dig(groups[g][0][1], "h3") is None:
            continue
        for sig in ("energy_raw", "u", "mahal", "surprise"):
            rows.append(
                [
                    g,
                    sig,
                    *(fmt(mean_std(column(groups[g], "h3", "auroc", sig, k))) for k in AUROC_KS),
                ]
            )
    out.append(_table(["run group", "signal", *(f"K={k}" for k in AUROC_KS)], rows))
    out += [
        "",
        "## 2. H3: how early unknown flows are rejected, against the fixed K=10 energy gate",
        "",
    ]
    rows = []
    for g in names:
        for i, fpr in enumerate((0.02, 0.05)):
            if dig(groups[g][0][1], "h3", "earliness", str(i)) is None:
                continue
            rows.append(
                [
                    g,
                    f"{fpr:g}",
                    fmt(
                        mean_std(
                            column(
                                groups[g], "h3", "earliness", str(i), "gate", "known_reject_rate"
                            )
                        ),
                        pct=True,
                    ),
                    fmt(
                        mean_std(
                            column(
                                groups[g], "h3", "earliness", str(i), "ours", "known_reject_rate"
                            )
                        ),
                        pct=True,
                    ),
                    fmt(
                        mean_std(
                            column(
                                groups[g], "h3", "earliness", str(i), "gate", "unknown_reject_rate"
                            )
                        ),
                        pct=True,
                    ),
                    fmt(
                        mean_std(
                            column(
                                groups[g], "h3", "earliness", str(i), "ours", "unknown_reject_rate"
                            )
                        ),
                        pct=True,
                    ),
                    fmt(
                        mean_std(
                            column(groups[g], "h3", "earliness", str(i), "ours", "mean_rejection_k")
                        ),
                        digits=2,
                    ),
                    fmt(
                        mean_std(column(groups[g], "h3", "earliness", str(i), "k_reduction")),
                        pct=True,
                    ),
                ]
            )
    out.append(
        _table(
            [
                "run group",
                "gate f",
                "gate: known rejected %",
                "ours: known rejected %",
                "gate: unknown rejected %",
                "ours: unknown rejected %",
                "ours: mean rejection K (gate = 10)",
                "K reduction %",
            ],
            rows,
        )
    )
    out += [
        "",
        "## 3. H2: closed-world accuracy (%) at a mean of 4 / 6 / 8 packets, and Pareto area",
        "",
    ]
    rows = []
    for g in names:
        for p in POLICIES:
            cells = [
                fmt(
                    mean_std(column(groups[g], "h2", "closed_world", p, "acc_at_budget", b)),
                    pct=True,
                )
                for b in BUDGETS
            ]
            rows.append(
                [
                    g,
                    POLICY_LABEL[p],
                    *cells,
                    fmt(mean_std(column(groups[g], "h2", "closed_world", p, "pareto_area"))),
                    fmt(
                        mean_std(column(groups[g], "h2", "closed_world", p, "smallest_mean_k")),
                        digits=2,
                    ),
                ]
            )
    out.append(
        _table(
            [
                "run group",
                "policy",
                "mean K 4",
                "mean K 6",
                "mean K 8",
                "Pareto area",
                "smallest mean K",
            ],
            rows,
        )
    )
    out += ["", "## 4. H2 open world and H4 controller", ""]
    rows = []
    for g in names:
        for o in ("reject_first", "commit_first"):
            if dig(groups[g][0][1], "h2", "open_world", o) is None:
                continue
            rows.append(
                [
                    g,
                    o,
                    fmt(
                        mean_std(column(groups[g], "h2", "open_world", o, "known_accuracy")),
                        pct=True,
                    ),
                    fmt(
                        mean_std(column(groups[g], "h2", "open_world", o, "unknown_committed")),
                        pct=True,
                    ),
                    fmt(
                        mean_std(column(groups[g], "h2", "open_world", o, "open_set_macro_f1")),
                        pct=True,
                    ),
                    fmt(
                        mean_std(column(groups[g], "h2", "open_world", o, "mean_k_known")), digits=2
                    ),
                ]
            )
    out.append(
        _table(
            [
                "run group",
                "check order",
                "known accuracy %",
                "unknown committed %",
                "open-set macro-F1 %",
                "mean K (known)",
            ],
            rows,
        )
    )
    out.append("")
    rows = []
    for g in names:
        for b in BUDGETS:
            if dig(groups[g][0][1], "h4", "budgets", b) is None:
                continue
            for label in ("controller", "static"):
                rows.append(
                    [
                        g,
                        b,
                        label,
                        *(
                            fmt(
                                mean_std(
                                    column(
                                        groups[g], "h4", "budgets", b, label, sp, "mean_abs_error"
                                    )
                                ),
                                pct=True,
                            )
                            for sp in ("val", "test_id", "test_drift")
                        ),
                        fmt(
                            mean_std(
                                column(
                                    groups[g], "h4", "budgets", b, label, "test_drift", "accuracy"
                                )
                            ),
                            pct=True,
                        ),
                    ]
                )
    out.append(
        _table(
            [
                "run group",
                "target K",
                "policy",
                "error % (val)",
                "error % (test_id)",
                "error % (drift weeks)",
                "accuracy % (drift weeks)",
            ],
            rows,
        )
    )
    out += ["", "## 5. Acceptance criteria (specs 007, 009, 010, 011), on the headline group", ""]
    crit = criteria(groups)
    if not crit.get("seeds"):
        out.append("_No headline group (`ft-pat-ssl_tuned-d1open`) found._")
    else:
        out.append(f"Headline group `{crit['headline']}`, {crit['seeds']} seed(s).")
        out.append("")
        rows = []
        for hyp in ("H3", "H2", "H4", "H1_packets"):
            for c in crit[hyp]:
                v = c["value"]
                met = {True: "**met**", False: "**not met**", None: "not evaluable"}[c["met"]]
                rows.append([hyp, c["criterion"], "-" if v is None else f"{v:.3f}", c["bar"], met])
        out.append(_table(["hypothesis", "criterion", "measured", "bar", "result"], rows))
    out.append("")
    return "\n".join(out)
