"""Phase 5's summary tables and acceptance criteria (plan phase 5, T4).

Reads, per run, ``<root>/<run>/{drift.json, drift_monitor.json, policy_cost.json}`` (written by
``scripts/phase5_drift_curves.py``, ``phase5_drift_monitor.py`` and ``phase5_policy_cost.py``) and,
once, ``<root>/efficiency.json`` (``scripts/benchmark.py``). Every cell is mean ± std over the seeds
of a run group with the seed count shown. :func:`criteria` writes spec 012's, 013's and 004's
acceptance rules as explicit comparisons; a rule that cannot be evaluated is left out, never
guessed.

NumPy and JSON only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from adl_etc.evaluation.phase4_tables import dig, fmt, mean_std

SEED_RE = re.compile(r"^(?P<group>.+)-s(?P<seed>\d+)$")
Groups = dict[str, list[tuple[int, dict[str, Any]]]]

DECAY_METRICS = (
    ("acc_k5", "accuracy at K=5"),
    ("acc_k8", "accuracy at K=8"),
    ("acc_k30", "accuracy at K=30"),
    ("macro_f1_k8", "macro-F1 at K=8"),
    ("policy_accuracy", "accuracy under Safe-Stop (mean K≈6)"),
    ("policy_macro_f1", "macro-F1 under Safe-Stop (mean K≈6)"),
    ("policy_mean_k", "mean K of Safe-Stop at a fixed threshold"),
    ("unknown_auroc_k8", "unknown AUROC at K=8"),
)
POLICY_LABEL = {"safe_stop": "Safe-Stop", "p_echo": "P-ECHO", "p_cape": "P-CAPE", "p_rl": "P-RL"}
SIGNAL_NOTE = {
    "mean_k": "mean K",
    "commit_rate": "commit rate",
    "reject_rate": "reject rate",
    "mean_safe_at_commit": "mean p_safe at commit",
    "mean_pkt_nll": "next-packet NLL",
    "prior_kl": "class-prior KL",
    "theta_commit": "controller θ",
}


def load_group(root: Path, filename: str) -> Groups:
    """``{group: [(seed, json), ...]}`` from ``<root>/<run>/<filename>``."""
    groups: Groups = {}
    for path in sorted(root.glob(f"*/{filename}")):
        m = SEED_RE.match(path.parent.name)
        if not m:
            continue
        groups.setdefault(m["group"], []).append(
            (int(m["seed"]), json.loads(path.read_text(encoding="utf-8")))
        )
    for members in groups.values():
        members.sort(key=lambda t: t[0])
    return groups


def load_efficiency(root: Path) -> dict[str, Any] | None:
    path = root / "efficiency.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _col(members: list[tuple[int, dict[str, Any]]], *path: str) -> list[float | None]:
    return [dig(r, *path) for _, r in members]


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _check(name: str, value: float | None, bar: str, met: bool | None) -> dict[str, Any]:
    return {"criterion": name, "value": value, "bar": bar, "met": met}


def _first_alarm_week(report: dict[str, Any], mode: str, key: str = "first_alarm") -> str | None:
    hit = dig(report, mode, "lead", key)
    return None if not hit else str(hit["week"]).split(":")[-1].replace("WEEK-2022-", "wk")


def criteria(
    decay: Groups,
    monitor: Groups,
    efficiency: dict[str, Any] | None,
    *,
    headline: str = "ft-pat-ssl_tuned-d1open",
) -> dict[str, Any]:
    out: dict[str, Any] = {"headline": headline, "004": [], "012": [], "013": []}

    # spec 004: weekly decay curves for three models, three seeds
    if decay:
        out["004"].append(
            _check(
                "weekly decay curves, models with three seeds",
                float(sum(1 for m in decay.values() if len(m) >= 3)),
                ">= 3 models",
                sum(1 for m in decay.values() if len(m) >= 3) >= 3,
            )
        )

    members = monitor.get(headline, [])
    if members:
        # spec 012: at least one signal alarms before the end of the drift horizon
        for mode in ("static", "controller"):
            hits = [
                dig(r, mode, "summary", "first_alarm") or dig(r, mode, "summary", "first_warning")
                for _, r in members
            ]
            out["012"].append(
                _check(
                    f"{mode}: the combined monitor raises a warning or alarm within the horizon "
                    "(seeds that do)",
                    float(sum(1 for h in hits if h)),
                    f"== {len(members)}",
                    all(hits),
                )
            )
        # the correlation with the weekly macro-F1 is *reported*, with the strongest signal named
        sigs = dig(members[0][1], "static", "signals") or []
        best = None
        for s in sigs:
            rho = mean_std(_col(members, "static", "correlation", s, "spearman_vs_policy_macro_f1"))
            if rho and (best is None or abs(rho[0]) > abs(best[1])):
                best = (s, rho[0])
        if best:
            out["012"].append(
                _check(
                    f"strongest Spearman correlation with weekly macro-F1 ({best[0]}), reported",
                    best[1],
                    "reported, no bar",
                    None,
                )
            )
        # lead time of the best signal's first alarm against the 3-point macro-F1 drop
        leads = _col(members, "static", "per_signal_lead_weeks", "prior_kl")
        if any(v is not None for v in leads):
            vals = [v for v in leads if v is not None]
            out["012"].append(
                _check(
                    "class-prior KL: weeks between its first alarm and the first week 3 macro-F1 "
                    "points below the test weeks (positive = alarm first)",
                    float(sum(vals) / len(vals)),
                    "reported, no bar",
                    None,
                )
            )

    # spec 013
    if efficiency:
        pat = next((v for k, v in efficiency["models"].items() if k.startswith("PAT")), None)
        if pat:
            out["013"] += [
                _check(
                    "PAT streaming step, median, batch 1, CPU, ms",
                    pat["stream_step"]["median_ms"],
                    "<= 1.0",
                    pat["stream_step"]["median_ms"] <= 1.0,
                ),
                _check(
                    "PAT full-flow latency, median, batch 1, CPU, ms",
                    pat["latency_batch1"]["median_ms"],
                    "<= 5.0",
                    pat["latency_batch1"]["median_ms"] <= 5.0,
                ),
                _check("PAT parameters", float(pat["params"]), "< 2,000,000", pat["params"] < 2e6),
            ]
    return out


def render_markdown(
    decay: Groups,
    monitor: Groups,
    cost: Groups,
    efficiency: dict[str, Any] | None,
    crit: dict[str, Any] | None = None,
) -> str:
    parts = [
        "# Phase 5 tables\n",
        "Generated by `scripts/make_phase5_tables.py`. Every cell is mean ± std over the group's "
        "seeds (n shown). Weekly numbers are known-class flows; each week is cut to the same "
        "number of flows so macro-F1 is comparable across weeks. A criterion is quoted from "
        "section 5, never restated.\n",
    ]

    # 1. decay
    rows = []
    for g, m in sorted(decay.items()):
        for key, label in DECAY_METRICS:
            if all(dig(r, "summary", key) is None for _, r in m):
                continue
            base = mean_std(_col(m, "summary", key, "test_id"))
            tail = mean_std(_col(m, "summary", key, "last_four_drift"))
            chg = mean_std(_col(m, "summary", key, "change"))
            slope = mean_std(_col(m, "summary", key, "slope_per_week"))
            rows.append([g, label, fmt(base), fmt(tail), fmt(chg, digits=3), fmt(slope, digits=4)])
    parts += [
        "## 1. Weekly decay: the test weeks against the last four drift weeks\n",
        _table(
            ["run group", "metric", "test weeks", "last 4 drift weeks", "change", "slope / week"],
            rows,
        ),
        "",
    ]
    rows = []
    for g, m in sorted(decay.items()):
        e = mean_std(_col(m, "summary", "early_minus_late"))
        if e:
            rows.append([g, fmt(e)])
    parts += [
        "Does an early prefix lose more than a late one? (change in accuracy at K=5 minus "
        "change at "
        "K=30; negative = the early prefix loses more)\n",
        _table(["run group", "early minus late"], rows),
        "",
    ]

    # 2. monitor
    parts.append("## 2. Drift signals against the measured macro-F1\n")
    for mode in ("static", "controller"):
        rows = []
        for g, m in sorted(monitor.items()):
            sigs = dig(m[0][1], mode, "signals") or []
            for s in sigs:
                rho = mean_std(_col(m, mode, "correlation", s, "spearman_vs_policy_macro_f1"))
                firsts = [dig(r, mode, "per_signal_first_alarm", s) for _, r in m]
                weeks = ", ".join(
                    "-" if not f else str(f["week"]).split(":")[-1].replace("WEEK-2022-", "wk")
                    for f in firsts
                )
                leads = [dig(r, mode, "per_signal_lead_weeks", s) for _, r in m]
                lead = ", ".join("-" if v is None else f"{v:+d}" for v in leads)
                rows.append([g, SIGNAL_NOTE.get(s, s), fmt(rho, digits=2), weeks, lead])
        parts += [
            f"### {mode} threshold" + (" (budget controller on)" if mode == "controller" else ""),
            "",
            _table(
                [
                    "run group",
                    "signal",
                    "Spearman vs weekly macro-F1",
                    "first alarm (week, per seed)",
                    "weeks before the 3-point F1 drop (per seed)",
                ],
                rows,
            ),
            "",
        ]
    rows = []
    for g, m in sorted(monitor.items()):
        for mode in ("static", "controller"):
            rows.append(
                [
                    g,
                    mode,
                    ", ".join(f"{dig(r, mode, 'h'):.1f}" for _, r in m),
                    ", ".join(str(_first_alarm_week(r, mode, "first_warning")) for _, r in m),
                    ", ".join(str(_first_alarm_week(r, mode, "first_alarm")) for _, r in m),
                    ", ".join(
                        str(dig(r, mode, "lead", "first_week_below_floor") or "-")
                        .split(":")[-1]
                        .replace("WEEK-2022-", "wk")
                        for _, r in m
                    ),
                ]
            )
    parts += [
        "### The combined monitor\n",
        _table(
            [
                "run group",
                "mode",
                "CUSUM h (per seed)",
                "first warning",
                "first alarm",
                "first week 3 macro-F1 points down",
            ],
            rows,
        ),
        "",
    ]

    # 3. efficiency
    parts.append("## 3. Model cost (one CPU, 4 threads, batch 1 unless stated)\n")
    if efficiency:
        hw = efficiency["hardware"]
        parts.append(
            f"Machine: {hw.get('processor', '?')}, {hw.get('cpu_physical', '?')} cores / "
            f"{hw.get('cpu_logical', '?')} threads, {hw.get('ram_gb', '?')} GB, torch "
            f"{hw.get('torch', '?')}. Each model timed twice, interleaved; the smaller median is "
            "kept.\n"
        )
        rows = []
        for name, r in efficiency["models"].items():
            rows.append(
                [
                    name,
                    f"{r['params']:,}",
                    f"{r['flops_full_flow'] / 1e6:.1f}",
                    f"{r['latency_batch1']['median_ms']:.2f}",
                    f"{r['latency_batch1']['p95_ms']:.2f}",
                    f"{r['latency_batch64']['median_ms'] / 64:.3f}",
                    f"{r['stream_step']['median_ms']:.3f}",
                    f"{r['stream_total_30_ms']:.1f}",
                    f"{r['flops_stream_total_30'] / 1e6:.1f}",
                ]
            )
        parts += [
            _table(
                [
                    "model",
                    "params",
                    "MFLOP / flow",
                    "full flow ms (median)",
                    "full flow ms (p95)",
                    "ms / flow at batch 64",
                    "streaming step ms",
                    "30 steps ms",
                    "MFLOP over 30 steps",
                ],
                rows,
            ),
            "",
            "The CNN is not causal: deciding after every packet costs one full pass per packet, "
            'so its "streaming step" is its full-flow latency. The GRU\'s step is one recurrent '
            "cell step. B5, the SSL PAT and the scratch PAT share one architecture, so one row "
            "stands for all three.\n",
        ]
    else:
        parts.append("(no efficiency.json yet)\n")

    # 4. policy cost
    rows = []
    for g, m in sorted(cost.items()):
        for pol, label in POLICY_LABEL.items():
            if dig(m[0][1], pol) is None:
                continue
            rows.append(
                [
                    g,
                    label,
                    fmt(mean_std(_col(m, pol, "packets", "mean")), digits=2),
                    fmt(mean_std(_col(m, pol, "packets", "p95")), digits=1),
                    fmt(mean_std(_col(m, pol, "time_ms", "mean")), digits=0),
                    fmt(mean_std(_col(m, pol, "time_share_of_flow")), pct=True),
                    fmt(mean_std(_col(m, pol, "accuracy")), pct=True),
                ]
            )
    parts += [
        "## 4. Observation cost of each policy at a matched mean of about 6 packets\n",
        "Known-class flows of `test_id`. The knob of each policy is the one whose mean K on "
        "validation is closest to 6. Time is the sum of the flow's own inter-packet times up "
        "to the "
        "decision packet.\n",
        _table(
            [
                "run group",
                "policy",
                "packets, mean",
                "packets, p95",
                "time to decision, mean ms",
                "share of the flow's duration",
                "accuracy %",
            ],
            rows,
        ),
        "",
    ]

    # 5. criteria
    if crit is not None:
        rows = []
        for spec in ("004", "012", "013"):
            for c in crit[spec]:
                met = "reported" if c["met"] is None else ("**met**" if c["met"] else "**not met**")
                val = "-" if c["value"] is None else f"{c['value']:.3f}"
                rows.append([spec, c["criterion"], val, c["bar"], met])
        parts += [
            "## 5. Acceptance criteria (specs 004, 012, 013)\n",
            f"Headline group `{crit['headline']}`.\n",
            _table(["spec", "criterion", "measured", "bar", "result"], rows),
            "",
        ]
    return "\n".join(parts)
