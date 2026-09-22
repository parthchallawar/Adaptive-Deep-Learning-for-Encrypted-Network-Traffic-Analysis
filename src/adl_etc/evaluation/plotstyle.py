"""One shared plotting style, so every figure in the paper matches without
per-script styling (plan T7). Imported by anything that draws a figure from
a :class:`~adl_etc.evaluation.report.Report`.

``matplotlib`` is imported lazily inside each function, not at module level:
this file lives under ``adl_etc.evaluation``, which ``tests/test_import_boundary.py``
walks to prove torch/mlflow-freedom, and while matplotlib is neither, keeping
the *module* itself free of any plotting-library import means it stays
importable (for ``PALETTE``/``AXIS_LABELS`` alone) in a minimal environment
that never plots anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Okabe & Ito (2008), the standard colour-blind-safe 8-colour palette.
#: Index 0 (black) is reserved for reference/baseline lines.
PALETTE: tuple[str, ...] = (
    "#000000",  # black -- reference/baseline
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)

AXIS_LABELS: dict[str, str] = {
    "k": "packets read (K)",
    "acc": "accuracy",
    "mean_k": "mean packets to commit",
    "coverage": "coverage",
    "auroc": "AUROC (unknown vs known)",
    "confidence": "predicted confidence",
    "ece": "expected calibration error",
}


def _mpl() -> Any:
    import matplotlib

    matplotlib.use("Agg")  # headless: every consumer here saves to a file
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(color=PALETTE[1:]),  # reserve black for the caller
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
        }
    )
    return plt


def plot_accuracy_vs_k(
    curves: dict[str, dict[int, float]], out_path: str | Path, *, title: str = ""
) -> None:
    """``curves`` is ``{series_name: {k: accuracy}}`` -- more than one series
    (e.g. several baselines, or known-only vs some ablation) on one axis."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, curve in curves.items():
        ks = sorted(curve)
        ax.plot(ks, [curve[k] for k in ks], marker="o", markersize=3, label=name)
    ax.set_xlabel(AXIS_LABELS["k"])
    ax.set_ylabel(AXIS_LABELS["acc"])
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)
    if len(curves) > 1:
        ax.legend(fontsize=8)
    fig.savefig(out_path)
    plt.close(fig)


def plot_pareto_front(
    fronts: dict[str, list[tuple[float, float]]], out_path: str | Path, *, title: str = ""
) -> None:
    """``fronts`` is ``{policy_name: [(mean_k, accuracy), ...]}``, each
    already Pareto-filtered (:func:`~adl_etc.evaluation.metrics.pareto_front`)."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, points in fronts.items():
        if not points:
            continue
        xs, ys = zip(*points, strict=True)
        ax.plot(xs, ys, marker="o", markersize=4, label=name)
    ax.set_xlabel(AXIS_LABELS["mean_k"])
    ax.set_ylabel(AXIS_LABELS["acc"])
    if title:
        ax.set_title(title)
    if len(fronts) > 1:
        ax.legend(fontsize=8)
    fig.savefig(out_path)
    plt.close(fig)


def plot_reliability_diagram(reliability: dict[str, Any], out_path: str | Path) -> None:
    """``reliability`` is a `Report.reliability` dict: bin centers, per-bin
    accuracy and confidence (``None`` for an empty bin)."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], color=PALETTE[0], linestyle="--", linewidth=1, label="perfect")
    centers = reliability["bin_centers"]
    acc = [v if v is not None else float("nan") for v in reliability["bin_accuracy"]]
    ax.bar(centers, acc, width=1.0 / reliability["n_bins"], alpha=0.7, edgecolor=PALETTE[0])
    ax.set_xlabel(AXIS_LABELS["confidence"])
    ax.set_ylabel(AXIS_LABELS["acc"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"ECE = {reliability['ece']:.3f}")
    fig.savefig(out_path)
    plt.close(fig)


def plot_auroc_vs_k(curve: dict[int, float], out_path: str | Path, *, title: str = "") -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ks = sorted(curve)
    ax.plot(ks, [curve[k] for k in ks], marker="o", markersize=3, color=PALETTE[1])
    ax.axhline(0.5, color=PALETTE[0], linestyle="--", linewidth=1)  # chance
    ax.set_xlabel(AXIS_LABELS["k"])
    ax.set_ylabel(AXIS_LABELS["auroc"])
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)
    fig.savefig(out_path)
    plt.close(fig)
