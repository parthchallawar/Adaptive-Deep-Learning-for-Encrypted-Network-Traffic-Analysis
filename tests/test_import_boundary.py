"""Phase 1's torch-free core stays torch-free (plan phase 2, "What this phase
inherits").

Nothing under ``adl_etc.data``, ``adl_etc.evaluation`` or ``adl_etc.utils`` may
import torch or mlflow just by being imported: the exporters and the metric
suite must keep installing and running in seconds, and a Kaggle kernel has no
mlflow. Run in a fresh interpreter so an earlier test that imported either one
cannot mask a regression.
"""

from __future__ import annotations

import subprocess
import sys

PROBE = """
import importlib, pkgutil, sys
import adl_etc

heavy = ("torch", "mlflow")
checked, leaks = 0, {}
for pkg_name in ("adl_etc.data", "adl_etc.evaluation", "adl_etc.utils"):
    pkg = importlib.import_module(pkg_name)
    for info in pkgutil.walk_packages(pkg.__path__, prefix=pkg_name + "."):
        importlib.import_module(info.name)
        checked += 1
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in heavy)
        if loaded:
            leaks[info.name] = loaded[:3]
            break
    if leaks:
        break
print("CHECKED", checked)
print("LEAK" if leaks else "CLEAN", leaks)
"""


def test_core_packages_import_without_torch_or_mlflow() -> None:
    out = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, check=True
    ).stdout

    assert "CLEAN" in out, out
    checked = int(next(line for line in out.splitlines() if line.startswith("CHECKED")).split()[1])
    assert checked >= 20, f"only imported {checked} modules; the walk is not covering the packages"
