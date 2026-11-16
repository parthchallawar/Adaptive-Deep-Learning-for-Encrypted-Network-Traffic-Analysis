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


def checked_groups(
    runs: list[Any], *, allow_dirty: bool
) -> tuple[dict[tuple[str, str, str, str], list[Any]], list[str]]:
    """Finished runs grouped by run name without the seed, after spec 014's
    refusals (dirty trees unless allowed, config drift always). Every table
    goes through this, so none of them can include what another refuses."""
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
    return groups, dirty


def build_rows(runs: list[Any], *, allow_dirty: bool) -> tuple[list[dict[str, Any]], list[str]]:
    groups, dirty = checked_groups(runs, allow_dirty=allow_dirty)

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


# --- phase 3 (plan phase 3, T8): per-K accuracy, robustness, paired bootstrap ----------------
#
# Built from the evaluation metrics `scripts/mlflow_import.py` attaches to each run
# (`<split>.acc_k5`, `test_id.drop_10.auc_k`, ...; adl_etc.utils.mlflow_sync.attach_eval)
# and, for the bootstrap, the `eval/<split>/predictions.npz` artifacts.

PHASE3_SPLIT = "d1open"
PHASE3_STAGES = ("ft", "bl")  # fine-tuned PATs and the baselines; pretraining has no accuracy
ACC_KS = (1, 3, 5, 8, 30)
CONDITIONS = (
    ("drop_05", "drop 5%"),
    ("drop_10", "drop 10%"),
    ("drop_20", "drop 20%"),
    ("reorder_05", "reorder 5%"),
    ("ipt_jitter_20", "IPT jitter 0.2"),
)
#: (scratch variant, SSL variant): spec 007's H1 comparisons, same recipe either side
BOOTSTRAP_PAIRS = (
    ("tuned", "ssl_tuned"),
    ("scratch", "ssl"),
    ("scratch_l10", "ssl_l10"),
    ("scratch_l01", "ssl_l01"),
)
LABELS = {
    ("ft", "pat", "ssl_tuned"): "PAT tuned, SSL init",
    ("ft", "pat", "tuned"): "PAT tuned, scratch",
    ("ft", "pat", "ssl"): "PAT default, SSL init",
    ("ft", "pat", "scratch"): "PAT default, scratch",
    ("bl", "gru", "main"): "B3 GRU",
    ("bl", "pat", "main"): "B5 PAT (full flow)",
    ("bl", "cnn", "main"): "B2 CNN (per-K)",
    ("ft", "pat", "ssl_l10"): "PAT default, SSL init, 10% labels",
    ("ft", "pat", "scratch_l10"): "PAT default, scratch, 10% labels",
    ("ft", "pat", "ssl_l01"): "PAT default, SSL init, 1% labels",
    ("ft", "pat", "scratch_l01"): "PAT default, scratch, 1% labels",
    ("ft", "pat", "abl_ref4"): "Ablation reference (4 epochs)",
    ("ft", "pat", "abl_drop0"): "Ablation: no dropout",
    ("ft", "pat", "abl_ls0"): "Ablation: no label smoothing",
    ("ft", "pat", "abl_noaug"): "Ablation: no augmentation",
    ("ft", "pat", "abl_noside"): "Ablation: no size side channel",
    ("ft", "pat", "abl_ramp1"): "Ablation: uniform loss ramp",
}
ORDER = list(LABELS)


def _seed_values(members: list[Any], key: str) -> list[float]:
    return [float(m.data.metrics[key]) for m in members if key in m.data.metrics]


def _mean_std(values: list[float]) -> tuple[float, float, int] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), (float(arr.std(ddof=1)) if arr.size > 1 else 0.0), int(arr.size)


def _fmt(stat: tuple[float, float, int] | None, *, pct: bool = False, signed: bool = False) -> str:
    if stat is None:
        return "–"
    mean, std, n = stat
    scale, digits = (100.0, 1) if pct else (1.0, 3)
    sign = "+" if signed else ""
    text = f"{mean * scale:{sign}.{digits}f}"
    return text if n == 1 else f"{text} ± {std * scale:.{digits}f}"


def _delta_values(members: list[Any], split: str, cond: str, metric: str) -> list[float]:
    out = []
    for m in members:
        a = m.data.metrics.get(f"{split}.{metric}")
        b = m.data.metrics.get(f"{split}.{cond}.{metric}")
        if a is not None and b is not None:
            out.append(float(b) - float(a))
    return out


def _seed_of(run: Any) -> int:
    return int(run.data.tags.get("seed", -1))


def phase3_bootstrap(groups: dict, client: Any, tmp: Path) -> list[dict[str, Any]]:
    """Paired bootstrap (``adl_etc.evaluation.bootstrap``) for each scratch/SSL pair,
    per seed and split, wherever both runs carry ``predictions.npz``. A pair with no
    such artifacts is reported as unavailable rather than left out."""
    from adl_etc.evaluation.bootstrap import PREDICTIONS_FILE, Predictions, paired_bootstrap

    rows: list[dict[str, Any]] = []
    for scratch, ssl in BOOTSTRAP_PAIRS:
        a_runs = groups.get(("ft", "pat", scratch, PHASE3_SPLIT), [])
        b_runs = groups.get(("ft", "pat", ssl, PHASE3_SPLIT), [])
        if not a_runs or not b_runs:
            continue
        found = False
        for split in ("test_id", "val"):
            for a in sorted(a_runs, key=_seed_of):
                b = next((r for r in b_runs if _seed_of(r) == _seed_of(a)), None)
                if b is None:
                    continue
                paths = []
                for run in (a, b):
                    dst = tmp / run.info.run_id
                    dst.mkdir(parents=True, exist_ok=True)
                    try:
                        local = client.download_artifacts(
                            run.info.run_id, f"eval/{split}/{PREDICTIONS_FILE}", str(dst)
                        )
                    except Exception:  # MLflow raises a plain exception when absent
                        break
                    paths.append(local)
                if len(paths) != 2:
                    continue
                result = paired_bootstrap(Predictions.load(paths[0]), Predictions.load(paths[1]))
                rows.append(
                    {"pair": (scratch, ssl), "split": split, "seed": _seed_of(a), **result}
                )
                found = True
        if not found:
            rows.append({"pair": (scratch, ssl), "split": None, "seed": None})
    return rows


def build_phase3(groups: dict, client: Any | None = None, tmp: Path | None = None) -> dict:
    keys = [k for k in groups if k[3] == PHASE3_SPLIT and k[0] in PHASE3_STAGES]
    keys.sort(key=lambda k: (ORDER.index(k[:3]) if k[:3] in ORDER else len(ORDER), k))

    def row(key: tuple[str, str, str, str], split: str) -> dict[str, Any] | None:
        members = groups[key]
        if not _seed_values(members, f"{split}.auc_k"):
            return None
        ms = lambda name: _mean_std(_seed_values(members, f"{split}.{name}"))  # noqa: E731
        return {
            "label": LABELS.get(key[:3], "-".join(key)),
            "group": "-".join(key),
            "seeds": sorted(_seed_of(m) for m in members if f"{split}.auc_k" in m.data.metrics),
            "acc": {k: ms(f"acc_k{k}") for k in ACC_KS},
            "auc_k": ms("auc_k"),
            "macro_f1": ms("macro_f1"),
            "k95": ms("k95"),
            "auroc5": ms("auroc_k5"),
            "auroc30": ms("auroc_k30"),
            "ece30": ms("ece_k30"),
            "n_flows": ms("n_flows"),
            "n_unknown": ms("n_unknown"),
            "robust": {
                cond: _mean_std(_delta_values(members, split, cond, "auc_k"))
                for cond, _ in CONDITIONS
            },
            "acc8_drop10": ms("drop_10.acc_k8"),
            "commits": sorted({m.data.tags.get("git_commit", "unknown")[:8] for m in members}),
        }

    test = [r for k in keys if (r := row(k, "test_id")) is not None]
    val = [r for k in keys if (r := row(k, "val")) is not None]
    boot = phase3_bootstrap(groups, client, tmp) if client is not None and tmp is not None else []
    return {"test": test, "val": val, "bootstrap": boot}


def render_phase3_markdown(p3: dict) -> str:
    acc_head = " | ".join(f"acc@{k} %" for k in ACC_KS)
    lines = [
        "# Phase 3 results (D1, open-set split `d1_open`)",
        "",
        "Auto-generated by `scripts/make_tables.py` from local MLflow (evaluation metrics "
        "attached by `scripts/mlflow_import.py`). Do not hand-edit. Mean ± std over seeds "
        "where a group has more than one; a single value is one seed.",
        "",
        "## 1. Test (`test_id`, weeks 31–34), clean",
        "",
        f"| Model | run group | seeds | {acc_head} | AUC_K | macro-F1 | k95 | AUROC@5 "
        "| AUROC@30 | ECE@30 |",
        "|---|---|---|" + "---|" * (len(ACC_KS) + 6),
    ]
    for r in p3["test"]:
        accs = " | ".join(_fmt(r["acc"][k], pct=True) for k in ACC_KS)
        k95 = "–" if r["k95"] is None else f"{r['k95'][0]:.0f}"
        lines.append(
            f"| {r['label']} | `{r['group']}` | {','.join(map(str, r['seeds']))} | {accs} "
            f"| {_fmt(r['auc_k'])} | {_fmt(r['macro_f1'])} | {k95} | {_fmt(r['auroc5'])} "
            f"| {_fmt(r['auroc30'])} | {_fmt(r['ece30'])} |"
        )
    if p3["test"] and p3["test"][0]["n_flows"]:
        n, unk = p3["test"][0]["n_flows"][0], p3["test"][0]["n_unknown"][0]
        lines += [
            "",
            f"{n:,.0f} flows (the evaluation cap, the same draw for every model), {unk:,.0f} of "
            "them from the 30 held-out unknown classes. Accuracy, macro-F1, k95, AUC_K and ECE "
            "are over known-class flows only; AUROC is known vs unknown by energy score. "
            "B2 CNN is per-K: its AUC_K averages its own 13-value K grid, not K = 1..30, so it "
            "is not comparable with the causal models' AUC_K.",
        ]
    lines += [
        "",
        "## 2. Robustness (`test_id`): change in AUC_K under damage",
        "",
        "Every model sees the same damaged flows (each condition's draw depends only on the "
        "seed and its name). Negative = worse than clean.",
        "",
        "| Model | AUC_K clean | " + " | ".join(f"Δ {label}" for _, label in CONDITIONS)
        + " | acc@8 % under drop 10% |",
        "|---|---|" + "---|" * (len(CONDITIONS) + 1),
    ]
    for r in p3["test"]:
        if not any(r["robust"].values()):
            continue
        deltas = " | ".join(_fmt(r["robust"][c], signed=True) for c, _ in CONDITIONS)
        lines.append(
            f"| {r['label']} | {_fmt(r['auc_k'])} | {deltas} | {_fmt(r['acc8_drop10'], pct=True)} |"
        )
    lines += [
        "",
        "## 3. Validation (weeks 27–30): every phase-3 run",
        "",
        "From each run's post-training pass. The best epoch was chosen on this split, so "
        "these numbers are selection-biased; section 1 is the unbiased comparison.",
        "",
        f"| Model | run group | seeds | {acc_head} | AUC_K | macro-F1 | k95 | AUROC@30 | commit |",
        "|---|---|---|" + "---|" * (len(ACC_KS) + 5),
    ]
    for r in p3["val"]:
        accs = " | ".join(_fmt(r["acc"][k], pct=True) for k in ACC_KS)
        k95 = "–" if r["k95"] is None else f"{r['k95'][0]:.0f}"
        lines.append(
            f"| {r['label']} | `{r['group']}` | {','.join(map(str, r['seeds']))} | {accs} "
            f"| {_fmt(r['auc_k'])} | {_fmt(r['macro_f1'])} | {k95} | {_fmt(r['auroc30'])} "
            f"| {', '.join(r['commits'])} |"
        )
    lines += [
        "",
        "## 4. Paired bootstrap, SSL init minus scratch",
        "",
        f"{N_BOOTSTRAP} resamples of flows (known classes), both models on each resample; "
        "95% percentile interval. This is flow-sampling uncertainty for one seed, not "
        "seed-to-seed variance.",
        "",
        "| Pair (scratch → SSL) | split | seed | Δ AUC_K [95% CI] | Δ acc@1 | Δ acc@3 | Δ acc@5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for b in p3["bootstrap"]:
        pair = f"`{b['pair'][0]}` → `{b['pair'][1]}`"
        if b["split"] is None:
            lines.append(
                f"| {pair} | – | – | no per-flow predictions in MLflow for this pair "
                "(`predictions.npz` is written by evaluations from 2026-11-15 on) | | | |"
            )
            continue
        auc = b["auc_k"]

        def pp(k: int, b: dict = b) -> str:
            row = b["acc_at_k"].get(str(k))
            if row is None:
                return "–"
            lo, hi = 100 * row["ci_lo"], 100 * row["ci_hi"]
            return f"{100 * row['diff']:+.1f} [{lo:+.1f}, {hi:+.1f}]"

        lines.append(
            f"| {pair} | {b['split']} | {b['seed']} | {auc['diff']:+.3f} "
            f"[{auc['ci_lo']:+.3f}, {auc['ci_hi']:+.3f}] | {pp(1)} | {pp(3)} | {pp(5)} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_phase3_latex(p3: dict) -> str:
    lines = [
        "% Auto-generated by scripts/make_tables.py -- do not hand-edit.",
        r"\begin{tabular}{l" + "c" * (len(ACC_KS) + 3) + "}",
        r"\toprule",
        "Model & " + " & ".join(f"acc@{k}" for k in ACC_KS) + r" & AUC$_K$ & macro-F1 & "
        r"$\Delta$AUC$_K$ (drop 10\%) \\",
        r"\midrule",
    ]
    for r in p3["test"]:
        accs = " & ".join(_fmt(r["acc"][k], pct=True) for k in ACC_KS)
        lines.append(
            f"{_tex_escape(r['label'])} & {accs} & {_fmt(r['auc_k'])} & {_fmt(r['macro_f1'])} & "
            f"{_fmt(r['robust']['drop_10'], signed=True)} \\\\".replace("±", r"$\pm$")
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
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

    import tempfile

    from mlflow import MlflowClient

    groups, _ = checked_groups(runs, allow_dirty=args.allow_dirty)
    with tempfile.TemporaryDirectory() as tmp:
        p3 = build_phase3(groups, MlflowClient(tracking_uri=uri), Path(tmp))
    if p3["test"] or p3["val"]:
        (out_dir / "phase3_table.md").write_text(render_phase3_markdown(p3), encoding="utf-8")
        (out_dir / "phase3_table.tex").write_text(render_phase3_latex(p3), encoding="utf-8")
        print(
            f"wrote {out_dir}/phase3_table.md and phase3_table.tex ({len(p3['test'])} test, "
            f"{len(p3['val'])} val group(s), {len(p3['bootstrap'])} bootstrap row(s))"
        )
    if dirty:
        print(f"included {len(dirty)} dirty run(s) (--allow-dirty): {dirty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
