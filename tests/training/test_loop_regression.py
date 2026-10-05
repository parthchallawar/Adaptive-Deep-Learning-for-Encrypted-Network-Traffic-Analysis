"""The baselines train exactly as they did before the loop was generalised (plan
phase 3, T3).

``golden_loop.json`` was recorded from the phase-2 loop (commit 2a8a001, before
the ``Objective`` interface existed): B2 (CNN) and B3 (GRU), two epochs on fixed
synthetic flows, CPU, no AMP. The generalised loop must reproduce every history
value and every parameter tensor. Float results can differ across torch builds,
so the comparison is skipped (with the reason) under a different torch
major.minor than the one that recorded it; on the recording build it is exact
up to a tolerance far below any real change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from adl_etc.data.features import Standardizer
from adl_etc.models.baselines import build_model
from adl_etc.training.datasets import FlowBatches
from adl_etc.training.labels import LabelSpace
from adl_etc.training.loop import TrainSettings, fit
from adl_etc.utils.seeding import seed_everything
from tests.training.util import make_cfg, make_flows, make_info, write_shards

GOLDEN = Path(__file__).with_name("golden_loop.json")
N_CLASSES = 4
TIMING = {"epoch_seconds", "flows_per_sec", "train_seconds", "train_flows_per_sec"}


def _torch_minor() -> str:
    return ".".join(torch.__version__.split("+")[0].split(".")[:2])


def train_baseline(name: str, tmp: Path) -> dict[str, Any]:
    """Two epochs of one baseline; returns its history and per-tensor sums."""
    space = LabelSpace.create(known=range(N_CLASSES))
    train_data = make_flows(400, N_CLASSES, seed=11)
    val_data = make_flows(120, N_CLASSES, seed=12)
    shards = write_shards(tmp / "shards", train_data)
    std = Standardizer.fit(shards)
    shards.close()
    train = FlowBatches(train_data, space, standardizer=std, train=True)
    val = FlowBatches(val_data, space, standardizer=std)
    settings = TrainSettings(epochs=2, batch_size=64, lr=3e-3, patience=10, amp=False)
    info = make_info(tmp, make_cfg(tmp), name=f"bl-{name}-v-syn-s0", seed=0)

    seed_everything(0)
    model = build_model({"name": name}, N_CLASSES)
    result = fit(model, train, val, settings=settings, run_dir=tmp / "run", run_info=info)

    state = model.state_dict()
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return {
        "history": [{k: v for k, v in row.items() if k not in TIMING} for row in result.history],
        "tensor_sums": {k: float(v.double().sum()) for k, v in sorted(state.items())},
        "sha256": digest.hexdigest(),
    }


def record(tmp: Path) -> None:
    """Regenerate ``golden_loop.json``. Only ever run against the *old* loop."""
    out = {"torch": _torch_minor(), "models": {}}
    for name in ("cnn", "gru"):
        out["models"][name] = train_baseline(name, tmp / name)
    GOLDEN.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize("name", ["cnn", "gru"])
def test_baselines_train_exactly_as_before_the_loop_was_generalised(
    name: str, tmp_path: Path
) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if golden["torch"] != _torch_minor():
        pytest.skip(f"golden values recorded under torch {golden['torch']}, not {_torch_minor()}")
    want = golden["models"][name]

    got = train_baseline(name, tmp_path)

    assert len(got["history"]) == len(want["history"])
    for g_row, w_row in zip(got["history"], want["history"], strict=True):
        assert g_row.keys() == w_row.keys()
        for key in w_row:
            np.testing.assert_allclose(g_row[key], w_row[key], rtol=1e-6, atol=1e-9, err_msg=key)
    assert got["tensor_sums"].keys() == want["tensor_sums"].keys()
    for key, value in want["tensor_sums"].items():
        got_sum = got["tensor_sums"][key]
        np.testing.assert_allclose(got_sum, value, rtol=1e-6, atol=1e-7, err_msg=key)
    assert got["sha256"] == want["sha256"]  # bit-identical parameters on the recording build
