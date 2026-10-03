"""The training loop on real traffic (plan phase 2 T5's done-criterion).

Uses the real D4 shard set when it is on this machine and skips, with a reason,
when it is not. Every expectation below comes from measuring what real D4
allows, not from the spec's hoped-for numbers; see the module-level notes.

**What real D4 is like** (measured 2026-09-21):

* It is heavily duplicated: 41,992 distinct PPIs in 403,394 flows (10.4%). Where
  identical inputs carry different labels no model can do better than the
  label-majority ceiling, which is 0.9126 for the whole corpus.
* Even among *distinct* PPIs there are near-clashes: in a 512-flow sample, 234
  flows have a differently-labelled neighbour within 0.5 in standardised
  feature space (minimum distance 0.000). A small MLP therefore memorises such a
  set slowly: 0.955 after 1,600 steps, 0.975 after 3,200, not 0.99 after 200.
* Its shards are ordered by capture file, so **a contiguous train/val slice
  gives disjoint classes** (a first probe of this module shared 1 class of 20
  between train and val and reported a flat F1 of 0.12). Always shuffle first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from adl_etc.data.features import Standardizer
from adl_etc.data.tensors import ShardSet
from adl_etc.training.checkpoint import load_checkpoint
from adl_etc.training.datasets import ArrayData, FlowBatches
from adl_etc.training.labels import LabelSpace
from adl_etc.training.loop import TrainSettings, fit
from adl_etc.utils.seeding import seed_everything
from tests.training.util import TinyMLP, make_cfg, make_info

D4 = Path(__file__).resolve().parents[2] / "data" / "processed" / "ustc-tfc2016" / "all"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (D4 / "meta.json").exists(), reason="D4 shards not exported here"),
]


@pytest.fixture(scope="module")
def d4() -> tuple[ArrayData, Standardizer, LabelSpace]:
    shards = ShardSet.open(D4)
    try:
        std = Standardizer.fit(shards)
        space = LabelSpace.from_label_map(shards.meta["label_map"])
        data = ArrayData.from_shards([shards])
    finally:
        shards.close()
    return data, std, space


def take(data: ArrayData, idx: np.ndarray) -> ArrayData:
    return ArrayData(data.ppi[idx], data.ppi_len[idx], data.label[idx])


def shuffled_split(data: ArrayData, n: int, seed: int) -> tuple[ArrayData, ArrayData]:
    """A flow-level 80/20 split of ``n`` random flows. Mechanics only: flows from
    one capture file (and exact duplicates) land on both sides, so the score it
    gives is optimistic and is never a result."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data))[:n]
    cut = int(0.8 * n)
    return take(data, perm[:cut]), take(data, perm[cut:])


def settings(**kw) -> TrainSettings:
    base = dict(epochs=2, batch_size=256, lr=5e-3, patience=99, amp=False)
    return TrainSettings(**{**base, **kw})


def test_the_split_helper_covers_every_class_on_both_sides(d4) -> None:
    """The guard against the contiguous-slice mistake described above."""
    data, _, space = d4

    tr, va = shuffled_split(data, 30_000, seed=0)

    assert len(set(tr.label)) == len(set(va.label)) == space.n_classes == 20


def test_the_loop_learns_real_d4_traffic(d4, tmp_path: Path) -> None:
    data, std, space = d4
    tr, va = shuffled_split(data, 30_000, seed=0)
    train = FlowBatches(tr, space, standardizer=std, train=True)
    val = FlowBatches(va, space, standardizer=std)
    info = make_info(tmp_path, make_cfg(tmp_path))
    seed_everything(0)

    result = fit(
        TinyMLP(space.n_classes, 128), train, val,
        settings=settings(epochs=6), run_dir=tmp_path / "run", run_info=info,
    )  # fmt: skip

    f1 = [h["val_macro_f1"] for h in result.history]
    assert f1[-1] > 0.5, f"val macro-F1 {f1[-1]:.3f}, chance is 0.05"
    assert f1[-1] > f1[0] + 0.1, f"no learning across epochs: {f1}"
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]


def test_a_real_run_pauses_resumes_and_lands_exactly_where_an_uninterrupted_one_does(
    d4, tmp_path: Path
) -> None:
    """T5's done-criterion on real data: complete, write a checkpoint, resume."""
    data, std, space = d4
    tr, va = shuffled_split(data, 20_000, seed=1)
    train = FlowBatches(tr, space, standardizer=std, train=True)
    val = FlowBatches(va, space, standardizer=std)
    info = make_info(tmp_path, make_cfg(tmp_path))

    def run(run_dir: Path, **kw):
        seed_everything(0)
        model = TinyMLP(space.n_classes, 128)
        return fit(
            model, train, val, settings=settings(), run_dir=run_dir, run_info=info, **kw
        ), model

    whole, m_whole = run(tmp_path / "whole")
    first, _ = run(tmp_path / "split", pause_check=lambda done, s: done == 1)
    second, m_split = run(tmp_path / "split")

    assert first.status == "paused" and (tmp_path / "split" / "ckpt_epoch1.pt").is_file()
    assert second.status == "finished" and second.epochs_run == 1
    assert [h["val_macro_f1"] for h in second.history] == [h["val_macro_f1"] for h in whole.history]
    assert all(
        torch.equal(a, b)
        for a, b in zip(m_whole.state_dict().values(), m_split.state_dict().values(), strict=True)
    )
    assert load_checkpoint(second.best_checkpoint)["label_space_hash"] == space.hash


def test_the_loop_drives_a_512_flow_real_set_toward_memorisation(d4, tmp_path: Path) -> None:
    """Spec 005 asks for > 99% in 200 steps. Real D4 does not allow that (see the
    module notes), so this checks what it does allow: on 512 flows with *distinct*
    PPIs a small MLP keeps climbing with steps (0.955 at 1,600 steps here)."""
    data, std, space = d4
    rng = np.random.default_rng(0)
    first_seen: dict[bytes, int] = {}
    for i in rng.permutation(len(data)):
        first_seen.setdefault(data.ppi[i].tobytes(), int(i))
        if len(first_seen) == 4000:
            break
    pick = np.sort(rng.choice(np.array(sorted(first_seen.values())), 512, replace=False))
    fair = FlowBatches(take(data, pick), space, standardizer=std, train=True)
    info = make_info(tmp_path, make_cfg(tmp_path))
    seed_everything(0)

    result = fit(
        TinyMLP(space.n_classes, 512), fair, fair,
        settings=settings(epochs=200, batch_size=64, lr=1e-2, weight_decay=0.0,
                          label_smoothing=0.0, balanced_cap=None),
        run_dir=tmp_path / "run", run_info=info,
    )  # fmt: skip

    acc = [h["val_acc"] for h in result.history]  # val is the training set here
    assert acc[-1] > 0.9, f"train accuracy {acc[-1]:.3f} after {200 * 8} steps"
    assert acc[-1] > acc[24] + 0.1  # still climbing: it is a step budget, not a bug
