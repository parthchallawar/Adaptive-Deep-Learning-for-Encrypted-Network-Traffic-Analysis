#!/usr/bin/env python
"""Regenerate results/summaries/ from local MLflow (plan T9, spec 004 & 014).

    python scripts/make_tables.py
    python scripts/make_tables.py --allow-dirty

Groups every finished MLflow run by its run name with the seed stripped
(``adl_etc.utils.runinfo.parse_run_name``'s stage/model/variant/split parts),
reports each group's ``best_val_macro_f1`` as mean +/- std across seeds plus
a 1000-resample bootstrap 95% CI (spec 004: "paired bootstrap ... 95% CIs" --
here over the group's seed values, the granularity MLflow's own summary
metrics give; a bootstrap over individual flow predictions needs a second
model or policy to pair against on the same flows, which does not exist yet),
and writes ``results/summaries/main_table.md`` and ``main_table.tex``.

Refuses (spec 014's edge cases):

- any candidate run with ``git_dirty=true``, unless ``--allow-dirty`` (then
  included, and flagged in the output).
- two runs sharing a run name but different ``config_hash`` values (config
  drift: the same name must always mean the same config).

Excluded, always: runs whose real status (the ``adl.status`` tag
:mod:`adl_etc.utils.mlflow_sync` records -- :class:`adl_etc.utils.tracking.Tracker`'s
own ``RUNNING``/``FINISHED``/``KILLED``/``PAUSED`` vocabulary, distinct from
:mod:`adl_etc.training.loop`'s ``finished``/``early_stopped``/``paused``
result strings, which a run's config/params never carry through to MLflow)
never reached ``FINISHED``. MLflow itself has no PAUSED, and a run resumed on
Kaggle but re-pulled mid-resume, or genuinely killed, is not a result (spec
014).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np

from adl_etc.utils import mlflow_sync as S
from adl_etc.utils.runinfo import parse_run_name

# Tracker's own status vocabulary (adl_etc.utils.tracking), not
# training.loop's "finished"/"early_stopped"/"paused" result strings -- a
# training run's own "early_stopped" is still Tracker-FINISHED (run_training
# only distinguishes paused from everything else when it calls tracker.close).
DONE_STATUSES = {"FINISHED"}
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0
HEADLINE_METRIC = "best_val_macro_f1"


class TableError(RuntimeError):
    """A table cannot be produced from what MLflow currently holds."""


def fetch_runs(tracking_uri: str, experiment: str) -> list[Any]:
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return []
    runs: list[Any] = []
    token: str | None = None
    while True:
        page = client.search_runs([exp.experiment_id], max_results=1000, page_token=token)
        runs.extend(page)
        token = page.token
        if not token:
            break
    return runs


def _group_key(run_name: str) -> tuple[str, str, str, str] | None:
    try:
        stage, model, variant, split, _seed = parse_run_name(run_name)
    except ValueError:
        return None  # not one of ours; skip rather than crash on a stray run
    return stage, model, variant, split


def _bootstrap_ci(values: np.ndarray, *, n: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    resamples = rng.choice(values, size=(n, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(resamples, [2.5, 97.5])
    return float(lo), float(hi)


def build_rows(runs: list[Any], *, allow_dirty: bool) -> tuple[list[dict[str, Any]], list[str]]:
    dirty = sorted({r.info.run_name for r in runs if r.data.tags.get("git_dirty") == "true"})
    if dirty and not allow_dirty:
        raise TableError(
            f"{len(dirty)} run(s) were trained on a dirty tree, refusing to include them in the "
            f"table (pass --allow-dirty to include anyway): {dirty}"
        )

    kept = [r for r in runs if r.data.tags.get("adl.status") in DONE_STATUSES]

    by_name: dict[str, set[str]] = defaultdict(set)
    for r in kept:
        by_name[r.info.run_name].add(r.data.tags.get("config_hash", ""))
    drift = {name: sorted(hashes) for name, hashes in by_name.items() if len(hashes) > 1}
    if drift:
        detail = "; ".join(f"{n}: {h}" for n, h in drift.items())
        raise TableError(f"config drift -- the same run name has different config hashes: {detail}")

    groups: dict[tuple[str, str, str, str], list[Any]] = defaultdict(list)
    for r in kept:
        key = _group_key(r.info.run_name)
        if key is not None:
            groups[key].append(r)

    rows: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        vals = np.array(
            [m.data.metrics[HEADLINE_METRIC] for m in members if HEADLINE_METRIC in m.data.metrics],
            dtype=float,
        )
        if vals.size == 0:
            continue
        lo, hi = _bootstrap_ci(vals) if vals.size > 1 else (float(vals[0]), float(vals[0]))
        rows.append(
            {
                "run_group": "-".join(key),
                "seeds": sorted(int(m.data.tags.get("seed", -1)) for m in members),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "ci_lo": lo,
                "ci_hi": hi,
                "commits": sorted({m.data.tags.get("git_commit", "unknown")[:12] for m in members}),
                "config_hashes": sorted(
                    {m.data.tags.get("config_hash", "unknown")[:12] for m in members}
                ),
                "dirty": any(m.info.run_name in dirty for m in members),
            }
        )
    return rows, dirty


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Baseline results",
        "",
        "Auto-generated by `scripts/make_tables.py` from local MLflow "
        f"(`{HEADLINE_METRIC}`, mean ± std over seeds, "
        f"{N_BOOTSTRAP}-resample bootstrap 95% CI). Do not hand-edit; "
        "every number here traces to a run name, a config hash and a commit (spec 014).",
        "",
        "| Run group | seeds | " + HEADLINE_METRIC + " (mean ± std) | 95% CI | commit "
        "| config hash |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        flag = " ⚠️ dirty" if row["dirty"] else ""
        lines.append(
            f"| `{row['run_group']}` | {len(row['seeds'])} ({','.join(map(str, row['seeds']))}) "
            f"| {row['mean']:.4f} ± {row['std']:.4f} "
            f"| [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] "
            f"| {', '.join(row['commits'])} | {', '.join(row['config_hashes'])}{flag} |"
        )
    lines.append("")
    return "\n".join(lines)


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def render_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated by scripts/make_tables.py -- do not hand-edit.",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Run group & seeds & " + _tex_escape(HEADLINE_METRIC) + r" & 95\% CI \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex_escape(row['run_group'])} & {len(row['seeds'])} & "
            f"{row['mean']:.4f} $\\pm$ {row['std']:.4f} & "
            f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--experiment", default=S.DEFAULT_EXPERIMENT)
    parser.add_argument("--out", type=Path, default=None, help="default: <results>/summaries")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="include runs trained on an uncommitted tree"
    )
    args = parser.parse_args(argv)

    uri, _artifact_root = S.local_store(args.results)
    runs = fetch_runs(uri, args.experiment)
    if not runs:
        print(f"no runs found in experiment {args.experiment!r} at {uri}", file=sys.stderr)
        return 1

    try:
        rows, dirty = build_rows(runs, allow_dirty=args.allow_dirty)
    except TableError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not rows:
        print("no finished runs to report", file=sys.stderr)
        return 1

    out_dir = args.out or (args.results / "summaries")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "main_table.md").write_text(render_markdown(rows), encoding="utf-8")
    (out_dir / "main_table.tex").write_text(render_latex(rows), encoding="utf-8")

    print(f"wrote {out_dir}/main_table.md and main_table.tex ({len(rows)} run group(s))")
    if dirty:
        print(f"included {len(dirty)} dirty run(s) (--allow-dirty): {dirty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
