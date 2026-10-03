"""plotstyle.py: shared palette/labels, and that the plotting functions
actually produce a file (plan T7)."""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")

from adl_etc.evaluation import plotstyle as S  # noqa: E402


def test_palette_is_colour_blind_safe_okabe_ito_and_reserves_black():
    assert len(S.PALETTE) == 8
    assert S.PALETTE[0] == "#000000"
    assert len(set(S.PALETTE)) == 8  # no accidental duplicate


def test_axis_labels_cover_what_the_plot_functions_need():
    for key in ("k", "acc", "mean_k", "auroc", "confidence"):
        assert key in S.AXIS_LABELS
        assert S.AXIS_LABELS[key]  # non-empty


def test_plot_accuracy_vs_k_writes_a_file(tmp_path):
    out = tmp_path / "acc.png"
    S.plot_accuracy_vs_k({"gru": {1: 0.3, 5: 0.6, 30: 0.9}}, out, title="D4")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_accuracy_vs_k_handles_several_series(tmp_path):
    out = tmp_path / "acc_multi.png"
    S.plot_accuracy_vs_k(
        {"gru": {1: 0.3, 30: 0.9}, "cnn": {1: 0.2, 30: 0.8}}, out
    )
    assert out.exists()


def test_plot_pareto_front_writes_a_file(tmp_path):
    out = tmp_path / "pareto.png"
    S.plot_pareto_front({"echo": [(1.0, 0.5), (5.0, 0.8), (10.0, 0.95)]}, out)
    assert out.exists()


def test_plot_pareto_front_handles_an_empty_front(tmp_path):
    out = tmp_path / "pareto_empty.png"
    S.plot_pareto_front({"echo": []}, out)
    assert out.exists()


def test_plot_reliability_diagram_writes_a_file(tmp_path):
    out = tmp_path / "reliability.png"
    reliability = {
        "n_bins": 3,
        "bin_centers": [0.166, 0.5, 0.833],
        "bin_accuracy": [0.1, None, 0.9],
        "bin_confidence": [0.2, None, 0.85],
        "ece": 0.12,
    }
    S.plot_reliability_diagram(reliability, out)
    assert out.exists()


def test_plot_auroc_vs_k_writes_a_file(tmp_path):
    out = tmp_path / "auroc.png"
    S.plot_auroc_vs_k({1: 0.5, 10: 0.7, 30: 0.9}, out)
    assert out.exists()


def test_the_module_itself_never_imports_matplotlib_eagerly():
    """PALETTE/AXIS_LABELS must be usable with no plotting library installed
    -- matplotlib is imported lazily, inside each plot function."""
    import subprocess
    import sys

    probe = (
        "import sys; "
        "sys.modules['matplotlib'] = None; "  # poison it: any import raises TypeError
        "from adl_etc.evaluation import plotstyle as S; "
        "assert S.PALETTE and S.AXIS_LABELS; "
        "print('OK')"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout
    assert "OK" in out
