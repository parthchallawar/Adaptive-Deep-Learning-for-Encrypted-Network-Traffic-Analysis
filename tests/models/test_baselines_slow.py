"""The baselines end to end: strict overfit on synthetic flows, then real D4 and D3
(plan phase 2 T6's done-criteria). Marked ``slow``.

Real-data splits here are flow-level random 80/20 splits for *mechanics only*
(D4 is 90% duplicated and both corpora share capture files across the split), so
the scores are optimistic and are never results. What is asserted is shape, not
level: curves rise with K, early K is above chance for models trained on prefixes,
and no early-K leak.

Curves are **balanced accuracy** against a chance level of 1/n_classes, not plain
accuracy: D3's largest class is 81% of its flows (and its smallest has 130), and
class-balanced training deliberately trades majority-class accuracy for rare-class
recall (the GRU's plain accuracy on D3 is 0.18 while its balanced accuracy is 0.62).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from adl_etc.data.features import Standardizer
from adl_etc.data.tensors import ShardSet
from adl_etc.evaluation.dense_logits import DenseLogits
from adl_etc.evaluation.metrics import balanced_accuracy
from adl_etc.models.baselines import build_model
from adl_etc.models.baselines.predict import predict_dense
from adl_etc.training.datasets import ArrayData, FlowBatches
from adl_etc.training.labels import LabelSpace
from adl_etc.training.loop import TrainSettings, fit
from adl_etc.utils.config import load_config
from adl_etc.utils.runinfo import RunInfo
from adl_etc.utils.seeding import seed_everything
from adl_etc.utils.tracking import Tracker, read_run_dir
from tests.training.util import make_cfg, make_flows, make_info, write_shards

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "models" / "baselines"
PROCESSED = ROOT / "data" / "processed"
DATASETS = {"d4": "ustc-tfc2016", "d3": "iscx-vpn-2016"}


# --- strict overfit, synthetic (spec 005's test, on data where it is well-posed) -----------------


@pytest.mark.parametrize("name", ["cnn", "gru", "lstm"])
def test_each_model_overfits_a_512_flow_set_above_99_percent_in_200_steps(
    name: str, tmp_path: Path
) -> None:
    """Spec 005: > 99% train accuracy on a 512-flow subset in 200 steps. Run on
    synthetic separable flows, where it isolates model and loop from data ambiguity
    (raw D4 makes it unachievable for any model, see spec 005)."""
    classes = 8
    data = make_flows(512, classes, seed=1)
    shards = write_shards(tmp_path, data, n_classes=classes)
    std = Standardizer.fit(shards)
    shards.close()
    fb = FlowBatches(data, LabelSpace.create(range(classes)), standardizer=std, train=True)
    info = make_info(tmp_path, make_cfg(tmp_path), name=f"bl-{name}-v-syn-s0")
    seed_everything(0)
    settings = TrainSettings(  # spec 005's CNN/RNN peak LR, no regularisation, 25 x 8 = 200 steps
        epochs=25, batch_size=64, lr=3e-3, patience=99, amp=False,
        label_smoothing=0.0, balanced_cap=None,
    )  # fmt: skip

    result = fit(
        build_model({"name": name}, classes), fb, fb,
        settings=settings, run_dir=tmp_path / "run", run_info=info,
    )  # fmt: skip

    assert result.history[-1]["val_acc"] > 0.99, f"{name}: {result.history[-1]['val_acc']:.4f}"


# --- real data -----------------------------------------------------------------------------------


@pytest.fixture(scope="module", params=list(DATASETS))
def real(request: pytest.FixtureRequest) -> tuple[str, ArrayData, Standardizer, LabelSpace]:
    root = PROCESSED / DATASETS[request.param] / "all"
    if not (root / "meta.json").exists():
        pytest.skip(f"{request.param} shards not exported on this machine")
    shards = ShardSet.open(root)
    try:
        std = Standardizer.fit(shards)
        space = LabelSpace.from_label_map(shards.meta["label_map"])
        data = ArrayData.from_shards([shards])
    finally:
        shards.close()
    return request.param, data, std, space


def split(data: ArrayData, n: int, seed: int) -> tuple[ArrayData, ArrayData]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data))[:n]
    cut = int(0.8 * n)

    def take(idx: np.ndarray) -> ArrayData:
        return ArrayData(data.ppi[idx], data.ppi_len[idx], data.label[idx])

    return take(perm[:cut]), take(perm[cut:])


def bal_curve(dense: DenseLogits, val: ArrayData, space: LabelSpace, ks: list[int]) -> list[float]:
    y = space.to_model(val.label)
    return [balanced_accuracy(dense.predictions_at(k, val.ppi_len), y) for k in ks]


def chance_rate(val: ArrayData, space: LabelSpace) -> float:
    """Balanced accuracy of a random guess: 1 / (classes present in val)."""
    y = space.to_model(val.label)
    return 1.0 / len(np.unique(y[y >= 0]))


def check_curve(
    curve: list[float], chance: float, label: str, *, trained_on_prefixes: bool = True
) -> None:
    """The shape a sane early-classification curve has, at any score level.

    ``trained_on_prefixes=False`` is for B2, the CNN: it is trained on full flows
    only (spec 005) and evaluated at K by zero-padding, so at K=1 it sees 29 padded
    positions it never met in training and does little better than chance
    (measured: balanced accuracy 0.16 vs 0.125 chance on D3, plain accuracy 0.227 vs
    a 0.222 majority rate on D4, then large gains by K=3). That is the gap
    prefix-aware training (C1) targets, not a defect, so K=1 is only required to
    beat chance for models that were trained on prefixes.
    """
    if trained_on_prefixes:
        assert curve[0] > chance + 0.03, f"{label}: K=1 {curve[0]:.3f} vs chance {chance:.3f}"
    assert curve[-1] >= curve[0] - 0.02, f"{label}: accuracy fell with K: {curve}"
    assert curve[-1] > chance + 0.2, f"{label}: K=30 barely above chance {chance:.3f}: {curve}"


@pytest.mark.parametrize("model_name", ["cnn", "gru"])
def test_b2_and_b3_train_pause_resume_and_have_sane_curves_on_real_traffic(
    real, model_name: str, tmp_path: Path
) -> None:
    name, data, std, space = real
    tr, va = split(data, 10_000, seed=1)
    train = FlowBatches(tr, space, standardizer=std, train=True)
    val = FlowBatches(va, space, standardizer=std)
    info = make_info(tmp_path, make_cfg(tmp_path), name=f"bl-{model_name}-v-{name}-s0")
    settings = TrainSettings(epochs=4, batch_size=256, lr=3e-3, patience=99, amp=False)

    def run(run_dir: Path, **kw):
        seed_everything(0)
        model = build_model({"name": model_name}, space.n_classes)
        return fit(
            model, train, val, settings=settings, run_dir=run_dir, run_info=info, **kw
        ), model

    whole, m_whole = run(tmp_path / "whole")
    first, _ = run(tmp_path / "split", pause_check=lambda done, s: done == 2)
    second, m_split = run(tmp_path / "split")

    assert first.status == "paused" and second.status == "finished" and second.epochs_run == 2
    assert [h["val_macro_f1"] for h in second.history] == [h["val_macro_f1"] for h in whole.history]
    assert all(
        torch.equal(a, b)
        for a, b in zip(m_whole.state_dict().values(), m_split.state_dict().values(), strict=True)
    )

    dense = predict_dense(m_split, va, space, std, batch_size=1024)
    ks = [1, 3, 6, 10, 30]
    check_curve(
        bal_curve(dense, va, space, ks),
        chance_rate(va, space),
        f"{model_name}/{name}",
        trained_on_prefixes=model_name != "cnn",  # only the GRU sees prefixes in training
    )


def test_a_shipped_config_runs_through_the_tracker_and_lands_in_mlflow(
    real, tmp_path: Path
) -> None:
    """The real pipeline a Kaggle run will use, minus Kaggle: config file -> model and
    settings -> RunInfo -> fit with a Tracker -> run directory -> local MLflow store."""
    pytest.importorskip("mlflow", reason="needs the `train` extra")
    from adl_etc.utils.mlflow_sync import local_store, sync_run_dir

    name, data, std, space = real
    if name != "d4":
        pytest.skip("one dataset is enough for the pipeline check")
    tr, va = split(data, 6_000, seed=2)
    cfg = load_config(
        CONFIGS / "cnn.yaml", ["train.epochs=2", "train.batch_size=256", "train.amp=false"]
    )
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "GIT_COMMIT").write_text("abc1234\n", encoding="utf-8")
    (tmp_path / "code" / "GIT_DIRTY").write_text("0\n", encoding="utf-8")
    info = RunInfo.collect("bl-cnn-v-d4-s0", 0, cfg, code_root=tmp_path / "code")
    train = FlowBatches(tr, space, standardizer=std, train=True)
    val = FlowBatches(va, space, standardizer=std)

    seed_everything(0)
    with Tracker.start(info, cfg, tmp_path / "runs" / info.run_name) as tracker:
        result = fit(
            build_model(cfg.model, space.n_classes), train, val,
            settings=TrainSettings.from_cfg(cfg.train), run_dir=tmp_path / "ckpt",
            run_info=info, tracker=tracker,
        )  # fmt: skip
    uri, art = local_store(tmp_path / "results")

    synced = sync_run_dir(tmp_path / "runs" / info.run_name, tracking_uri=uri, artifact_root=art)

    record = read_run_dir(tmp_path / "runs" / info.run_name)
    assert result.status == "finished" and record.status == "FINISHED"
    assert record.params["config_hash"] == info.config_hash and record.params["model.name"] == "cnn"
    assert synced.created and synced.mlflow_status == "FINISHED"
    assert synced.metrics_logged >= 2 * 7  # two epochs of the per-epoch metrics
