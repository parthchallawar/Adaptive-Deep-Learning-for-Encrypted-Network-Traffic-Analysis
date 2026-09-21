"""tests/training needs torch (the `train` extra). Where it is absent, as on a
Kaggle-shaped environment or a phase-1-only install, skip the directory and say so
in the run header instead of failing at collection."""

from __future__ import annotations

import importlib.util

_HAS_TORCH = importlib.util.find_spec("torch") is not None

collect_ignore_glob = [] if _HAS_TORCH else ["test_*.py"]


def pytest_report_header(config: object) -> str | None:
    if _HAS_TORCH:
        return None
    return "tests/training: SKIPPED, torch is not installed (pip install -e '.[train]')"
