"""predict_dense and B1 XGBoost (plan phase 2 T6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from adl_etc.data.features import Standardizer
from adl_etc.evaluation.dense_logits import K_GRID, NotEvaluatedError
from adl_etc.models.baselines import CNNBaseline, RNNBaseline
from adl_etc.models.baselines.predict import predict_dense
from adl_etc.models.baselines.xgb import XGBBaseline, XGBError, class_weights
from adl_etc.training.datasets import ArrayData
from adl_etc.training.labels import LabelSpace
from adl_etc.utils.tracking import Tracker, read_run_dir
from tests.training.util import make_cfg, make_flows, make_info, write_shards

C = 4
SPACE = LabelSpace.create(range(C))


@pytest.fixture(scope="module")
def std(tmp_path_factory: pytest.TempPathFactory) -> Standardizer:
    root = tmp_path_factory.mktemp("std")
    shards = write_shards(root, make_flows(600, C, seed=1))
    out = Standardizer.fit(shards)
    shards.close()
    return out


@pytest.fixture(scope="module")
def flows() -> ArrayData:
    return make_flows(120, C, seed=7, min_len=3, max_len=12)


# --- predict_dense -------------------------------------------------------------------------------


def test_a_causal_model_is_run_once_and_holds_every_k(flows: ArrayData, std: Standardizer) -> None:
    torch.manual_seed(0)
    model = RNNBaseline(C, hidden=32, stem=16)

    dense = predict_dense(model, flows, SPACE, std, batch_size=32)

    assert dense.indexing == "effective" and dense.evaluated_k == tuple(range(1, 31))
    assert dense.logits.shape == (120, 30, C)
    # it agrees with running the whole thing through the model directly
    from adl_etc.training.datasets import FlowBatches, to_torch

    fb = FlowBatches(flows, SPACE, standardizer=std)
    with torch.no_grad():
        direct = model.eval()(to_torch(fb.batch(np.arange(120)))).numpy()
    np.testing.assert_allclose(dense.logits, direct, atol=1e-5)


def test_a_non_causal_model_is_run_once_per_grid_k(flows: ArrayData, std: Standardizer) -> None:
    torch.manual_seed(0)

    dense = predict_dense(CNNBaseline(C, channels=(8, 8, 8)), flows, SPACE, std)

    assert dense.indexing == "nominal" and dense.evaluated_k == K_GRID
    with pytest.raises(NotEvaluatedError):
        dense.at(7, flows.ppi_len)  # 7 is not on the grid


def test_a_custom_grid_is_respected(flows: ArrayData, std: Standardizer) -> None:
    dense = predict_dense(CNNBaseline(C, channels=(8, 8, 8)), flows, SPACE, std, ks=[2, 9])

    assert dense.evaluated_k == (2, 9)


def test_a_per_k_prediction_depends_only_on_the_first_k_packets(
    flows: ArrayData, std: Standardizer
) -> None:
    """Truncation is real: scrambling every packet after K leaves K's logits unchanged."""
    torch.manual_seed(0)
    model = CNNBaseline(C, channels=(8, 8, 8))
    k = 5
    scrambled = ArrayData(flows.ppi.copy(), flows.ppi_len, flows.label)
    rng = np.random.default_rng(0)
    scrambled.ppi[:, k:, 2] = rng.integers(1, 1500, scrambled.ppi[:, k:, 2].shape)  # sizes

    a = predict_dense(model, flows, SPACE, std, ks=[k])
    b = predict_dense(model, scrambled, SPACE, std, ks=[k])

    np.testing.assert_allclose(a.at(k, flows.ppi_len), b.at(k, flows.ppi_len), atol=1e-5)


def test_prediction_includes_unknown_class_flows_and_restores_the_model_mode(
    std: Standardizer,
) -> None:
    space = LabelSpace.create(range(C), unknown=[9])
    mixed = make_flows(40, 10, seed=3, class_ids=[0, 1, 9])
    model = CNNBaseline(C, channels=(8, 8, 8)).train()

    dense = predict_dense(model, mixed, space, std, ks=[3])

    assert len(dense) == 40  # an open-set evaluation needs the unknown flows too
    assert model.training is True  # left as it was found


def test_a_model_emitting_the_wrong_shape_is_rejected(flows: ArrayData, std: Standardizer) -> None:
    class BadCausal(RNNBaseline):
        def forward(self, batch):  # type: ignore[no-untyped-def]
            return super().forward(batch)[:, :10]

    class BadPerK(CNNBaseline):
        def forward(self, batch):  # type: ignore[no-untyped-def]
            return super().forward(batch)[:, None, :]

    with pytest.raises(ValueError, match="causal model must emit"):
        predict_dense(BadCausal(C, hidden=8, stem=8), flows, SPACE, std)
    with pytest.raises(ValueError, match="per-K model must emit"):
        predict_dense(BadPerK(C, channels=(8, 8, 8)), flows, SPACE, std, ks=[3])


# --- XGBoost -------------------------------------------------------------------------------------


def small_xgb(**kw) -> XGBBaseline:
    base = dict(ks=(1, 3, 12), n_estimators=60, early_stopping_rounds=10, max_depth=4, n_jobs=2)
    return XGBBaseline(SPACE, **{**base, **kw})


@pytest.fixture(scope="module")
def train_val() -> tuple[ArrayData, ArrayData]:
    return make_flows(800, C, seed=1), make_flows(300, C, seed=2)


@pytest.fixture(scope="module")
def fitted(train_val) -> XGBBaseline:
    model = small_xgb()
    model.fit(*train_val)
    return model


def test_xgb_fits_one_booster_per_k_and_reports_val_scores(fitted: XGBBaseline) -> None:
    assert sorted(fitted.boosters) == [1, 3, 12]
    assert all(0 <= fitted.best_iteration[k] < 60 for k in fitted.ks)


def test_device_defaults_to_cpu_and_is_threaded_through() -> None:
    assert small_xgb().params["device"] == "cpu"
    assert small_xgb(device="cuda").params["device"] == "cuda"


@pytest.mark.gpu
def test_xgb_trains_on_gpu_when_device_is_cuda(train_val) -> None:
    """Real GPU fit, not just the param threading above -- the thing that
    actually matters (plan T8-follow-up: B1's CPU-only run ran past 10 hours
    without finishing one seed; ``device="cuda"`` is the fix). Checked via
    ``nvidia-smi``, not ``torch.cuda.is_available()``: xgboost's wheel bundles
    its own CUDA runtime independent of torch's, and a CPU-only torch build
    (this project's own local venv, in fact) would otherwise wrongly skip a
    real, usable GPU."""
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        pytest.skip("no CUDA device available")
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, timeout=10, check=True)
    except (subprocess.CalledProcessError, OSError):
        pytest.skip("no CUDA device available")

    model = small_xgb(device="cuda")
    model.fit(*train_val)
    assert sorted(model.boosters) == [1, 3, 12]


def test_xgb_predicts_nominal_logits_and_learns(fitted: XGBBaseline, train_val) -> None:
    _, val = train_val

    dense = fitted.predict_dense(val)

    assert dense.indexing == "nominal" and dense.evaluated_k == (1, 3, 12)
    assert dense.logits.shape == (300, 30, C)
    y = SPACE.to_model(val.label)
    acc = {k: float((dense.predictions_at(k, val.ppi_len) == y).mean()) for k in (1, 3, 12)}
    assert acc[12] > 0.9, acc
    with pytest.raises(NotEvaluatedError):
        dense.at(2, val.ppi_len)


def test_early_k_is_not_better_than_late_k_the_leakage_tripwire(fitted, train_val) -> None:
    """Plan T6: B1 at K=1 must not beat B1 at K=30 by more than noise. If it did,
    the future would be leaking into the early features (spec 005, F2)."""
    _, val = train_val
    dense = fitted.predict_dense(val)
    y = SPACE.to_model(val.label)

    acc1 = float((dense.predictions_at(1, val.ppi_len) == y).mean())
    acc12 = float((dense.predictions_at(12, val.ppi_len) == y).mean())

    assert acc1 <= acc12 + 0.02


def test_xgb_logits_are_log_probabilities(fitted: XGBBaseline, train_val) -> None:
    dense = fitted.predict_dense(train_val[1])

    probs = np.exp(dense.logits[:, 2, :])  # K=3
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_save_and_load_reproduce_predictions_exactly(
    fitted: XGBBaseline, train_val, tmp_path: Path
) -> None:
    _, val = train_val
    fitted.save(tmp_path / "xgb")

    loaded = XGBBaseline.load(tmp_path / "xgb", SPACE)

    assert loaded.ks == fitted.ks and loaded.best_iteration == fitted.best_iteration
    assert loaded.params["device"] == fitted.params["device"]
    np.testing.assert_array_equal(
        loaded.predict_dense(val).logits, fitted.predict_dense(val).logits
    )


def test_loading_under_a_different_label_space_or_from_nothing_is_refused(
    fitted: XGBBaseline, tmp_path: Path
) -> None:
    fitted.save(tmp_path / "xgb")

    with pytest.raises(XGBError, match="different label space"):
        XGBBaseline.load(tmp_path / "xgb", LabelSpace.create(range(C), unknown=[9]))
    with pytest.raises(XGBError, match="unreadable"):
        XGBBaseline.load(tmp_path / "missing", SPACE)


def test_an_unfitted_model_cannot_predict_or_save(tmp_path: Path) -> None:
    model = small_xgb()

    with pytest.raises(XGBError, match="not been fitted"):
        model.predict_dense(make_flows(5, C, seed=1))
    with pytest.raises(XGBError, match="nothing to save"):
        model.save(tmp_path / "x")


def test_a_class_absent_from_training_gets_a_floor_logit_and_is_never_predicted() -> None:
    train = make_flows(600, 6, seed=1, class_ids=[0, 1, 3])  # classes 2 never appears
    val = make_flows(200, 6, seed=2, class_ids=[0, 1, 3])
    space = LabelSpace.create(range(4))
    model = XGBBaseline(space, ks=(3,), n_estimators=30, early_stopping_rounds=5, n_jobs=2)

    model.fit(train, val)
    dense = model.predict_dense(val)

    assert model.present is not None and model.present.tolist() == [0, 1, 3]
    assert (dense.logits[:, 2, 2] < -20).all()  # column 2 is the absent class
    assert (dense.predictions_at(3, val.ppi_len) != 2).all()


def test_fit_rejects_unknown_class_flows_and_a_single_class(train_val) -> None:
    space = LabelSpace.create(range(C), unknown=[9])
    mixed = make_flows(100, 10, seed=3, class_ids=[0, 1, 9])
    with pytest.raises(ValueError, match="unknown-class"):
        XGBBaseline(space, ks=(3,), n_estimators=5).fit(mixed, train_val[1])

    one_class = make_flows(50, C, seed=4, class_ids=[1])
    with pytest.raises(XGBError, match="at least 2 classes"):
        small_xgb().fit(one_class, train_val[1])

    only_unseen_val = make_flows(50, C, seed=5, class_ids=[3])
    two = make_flows(100, C, seed=6, class_ids=[0, 1])
    with pytest.raises(ValueError, match="seen in training"):
        small_xgb().fit(two, only_unseen_val)


def test_class_weights_follow_the_capped_inverse_frequency_rule() -> None:
    y = np.array([0] * 100 + [1] * 10 + [2])  # the hand example: factors .37, 3.7, 10

    w = class_weights(y, 3, cap=10)

    # factors: mean_count = 111/3 = 37, so min(10, 37/100), min(10, 37/10), min(10, 37/1)
    #        = 0.37, 3.7, 10; relative to class 0 that is 1 : 10 : 27.027
    ratio = np.array([w[0], w[100], w[110]]) / w[0]
    np.testing.assert_allclose(ratio, [1, 10, 10 / 0.37], rtol=1e-6)
    assert w.mean() == pytest.approx(1.0)


def test_max_train_flows_subsamples_the_training_set_reproducibly(train_val) -> None:
    a = small_xgb(ks=(3,), max_train_flows=200)
    b = small_xgb(ks=(3,), max_train_flows=200)

    a.fit(*train_val)
    b.fit(*train_val)

    np.testing.assert_array_equal(
        a.predict_dense(train_val[1]).logits, b.predict_dense(train_val[1]).logits
    )


def test_from_config_builds_the_model_and_rejects_typos() -> None:
    m = XGBBaseline.from_config({"name": "xgb", "max_depth": 3, "n_estimators": 7}, SPACE, seed=5)

    assert m.params["max_depth"] == 3 and m.params["seed"] == 5
    with pytest.raises(ValueError, match="expected model name"):
        XGBBaseline.from_config({"name": "cnn"}, SPACE)
    with pytest.raises(ValueError, match="bad arguments"):
        XGBBaseline.from_config({"name": "xgb", "max_dept": 3}, SPACE)


def test_fit_logs_each_k_to_a_tracker_with_k_as_the_step(train_val, tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    info = make_info(tmp_path, cfg)
    tracker = Tracker.start(info, cfg, tmp_path / "track")

    small_xgb().fit(*train_val, tracker=tracker)

    points = [m for m in read_run_dir(tmp_path / "track").metrics if m.key == "val_macro_f1"]
    assert [m.step for m in points] == [1, 3, 12]


# --- gaps found by mutation-checking ------------------------------------------------------


def third_packet_flows(n: int, seed: int) -> ArrayData:
    """Flows where ONLY the third packet's size carries the class. Packets 1 and 2 are
    identical noise for every class, so a model that has seen only K <= 2 packets has
    no information, and one that has seen K >= 3 has all of it."""
    from adl_etc.data import ppi as P

    rng = np.random.default_rng(seed)
    label = rng.integers(0, C, n).astype(np.int64)
    ppi = np.zeros((n, P.K_MAX, P.PPI_CHANNELS), dtype=P.PPI_DTYPE)
    ppi_len = np.full(n, 8, dtype=np.int8)
    ppi[:, :8, P.IPT_POS] = 5
    ppi[:, 0, P.IPT_POS] = 0
    ppi[:, :8, P.DIR_POS] = P.DIR_FWD
    ppi[:, :8, P.SIZE_POS] = rng.integers(90, 110, (n, 8))  # same distribution for all classes
    ppi[:, 2, P.SIZE_POS] = 300 + 250 * label + rng.integers(0, 20, n)  # the only signal
    return ArrayData(ppi, ppi_len, label)


def test_xgb_cannot_use_packets_it_has_not_seen() -> None:
    """The real leakage test for B1 (spec 005, F2). The K=1 <= K=30 tripwire alone
    passes even if K were ignored; this cannot: with the class in packet 3 only, K=2
    must be at chance and K=3 near perfect."""
    train, val = third_packet_flows(1500, 1), third_packet_flows(600, 2)
    model = XGBBaseline(
        SPACE, ks=(1, 2, 3, 8), n_estimators=80, early_stopping_rounds=10, max_depth=4, n_jobs=2
    )
    model.fit(train, val)

    dense = model.predict_dense(val)
    y = SPACE.to_model(val.label)
    acc = {k: float((dense.predictions_at(k, val.ppi_len) == y).mean()) for k in (1, 2, 3, 8)}

    assert acc[1] < 0.45 and acc[2] < 0.45, f"early K is reading the future: {acc}"
    assert acc[3] > 0.95 and acc[8] > 0.95, acc


def test_predictions_use_the_early_stopped_best_iteration_not_every_tree() -> None:
    import xgboost as xgb

    from adl_etc.data.prefix_stats import prefix_flowstats

    # 35% of the training labels are flipped and the trees are deep, so validation
    # loss bottoms out early and then rises: a genuinely early-stopped booster.
    train = make_flows(400, C, seed=1)
    rng = np.random.default_rng(0)
    flip = rng.random(len(train)) < 0.35
    noisy = ArrayData(
        train.ppi, train.ppi_len, np.where(flip, rng.integers(0, C, len(train)), train.label)
    )
    val = make_flows(300, C, seed=2)
    model = XGBBaseline(
        SPACE, ks=(12,), n_estimators=300, early_stopping_rounds=5, max_depth=8, n_jobs=2
    )
    model.fit(noisy, val)
    k = 12
    best = model.best_iteration[k]
    total = model.boosters[k].num_boosted_rounds()
    assert best + 1 < total, f"fixture never early-stopped (best {best} of {total})"

    features = prefix_flowstats(val.ppi, val.ppi_len, k)
    used = model._proba(k, features)

    booster = model.boosters[k]
    up_to_best = booster.predict(xgb.DMatrix(features), iteration_range=(0, best + 1))
    all_trees = booster.predict(xgb.DMatrix(features), iteration_range=(0, total))
    np.testing.assert_array_equal(used, up_to_best)
    assert not np.array_equal(used, all_trees)


def test_class_weights_actually_reach_the_booster(
    train_val, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xgboost as xgb

    seen: list[np.ndarray] = []
    real_fit = xgb.XGBClassifier.fit

    def spy(self, x, y, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(np.asarray(kwargs["sample_weight"]))
        return real_fit(self, x, y, *args, **kwargs)

    monkeypatch.setattr(xgb.XGBClassifier, "fit", spy)
    train, val = train_val
    skewed = ArrayData(
        train.ppi, train.ppi_len, np.where(np.arange(len(train)) < 700, 0, train.label)
    )
    XGBBaseline(SPACE, ks=(3,), n_estimators=5, class_weight_cap=10.0).fit(skewed, val)

    (w,) = seen
    y = SPACE.to_model(skewed.label)
    assert w.mean() == pytest.approx(1.0) and w.std() > 0
    # the over-represented class 0 is down-weighted relative to the rare ones
    assert w[y == 0].mean() < w[y == 1].mean()
