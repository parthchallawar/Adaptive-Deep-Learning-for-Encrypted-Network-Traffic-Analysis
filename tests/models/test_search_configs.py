"""The random-search sampler and the shipped baseline configs (plan phase 2 T6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adl_etc.models.baselines import CNNBaseline, RNNBaseline, build_model
from adl_etc.models.baselines.xgb import XGBBaseline
from adl_etc.models.search import DEFAULT_TRIALS, SearchSpaceError, sample_trials, to_overrides
from adl_etc.training.labels import LabelSpace
from adl_etc.training.loop import TrainSettings
from adl_etc.utils.config import ConfigError, load_config, to_container

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "models" / "baselines"
NAMES = ("cnn", "gru", "lstm", "xgb")

SPACE = {
    "train.lr": {"loguniform": [0.0005, 0.01]},
    "model.dropout": {"choice": [0.0, 0.1, 0.2]},
    "model.hidden": {"int": [128, 512]},
    "train.label_smoothing": {"uniform": [0.0, 0.2]},
}


# --- the sampler ----------------------------------------------------------------------------------


def test_default_is_twenty_trials_and_draws_are_reproducible() -> None:
    a = sample_trials(SPACE, seed=3)
    b = sample_trials(SPACE, seed=3)

    assert len(a) == DEFAULT_TRIALS == 20
    assert a == b
    assert a != sample_trials(SPACE, seed=4)


def test_every_draw_respects_its_distribution() -> None:
    for trial in sample_trials(SPACE, n_trials=200, seed=0):
        assert 0.0005 <= trial["train.lr"] <= 0.01
        assert trial["model.dropout"] in (0.0, 0.1, 0.2)
        assert 128 <= trial["model.hidden"] <= 512 and isinstance(trial["model.hidden"], int)
        assert 0.0 <= trial["train.label_smoothing"] <= 0.2


def test_loguniform_is_log_spread_not_linear() -> None:
    lrs = [t["train.lr"] for t in sample_trials(SPACE, n_trials=2000, seed=1)]

    below_geometric_mean = sum(lr < (0.0005 * 0.01) ** 0.5 for lr in lrs) / len(lrs)

    assert below_geometric_mean == pytest.approx(0.5, abs=0.05)  # linear would give ~0.05


def test_int_bounds_are_inclusive() -> None:
    draws = {t["n"] for t in sample_trials({"n": {"int": [1, 3]}}, n_trials=200, seed=0)}

    assert draws == {1, 2, 3}


def test_trial_i_is_the_same_whether_the_search_runs_5_or_20_trials() -> None:
    assert sample_trials(SPACE, n_trials=5, seed=9) == sample_trials(SPACE, n_trials=20, seed=9)[:5]


def test_adding_a_parameter_does_not_change_the_others_draws_in_a_trial() -> None:
    """Draws come in sorted-key order from one stream, so a parameter that sorts
    after the others leaves theirs untouched."""
    base = sample_trials({"a": {"uniform": [0, 1]}}, n_trials=3, seed=2)
    more = sample_trials({"a": {"uniform": [0, 1]}, "z": {"choice": [1, 2]}}, n_trials=3, seed=2)

    assert [t["a"] for t in base] == [t["a"] for t in more]


def test_dict_order_of_the_space_does_not_matter() -> None:
    forward = sample_trials(SPACE, n_trials=4, seed=5)
    reverse = sample_trials(dict(reversed(list(SPACE.items()))), n_trials=4, seed=5)

    assert forward == reverse


@pytest.mark.parametrize(
    "space, message",
    [
        ({}, "empty"),
        ({"x": {}}, "exactly one"),
        ({"x": {"uniform": [0, 1], "int": [1, 2]}}, "exactly one"),
        ({"x": {"gaussian": [0, 1]}}, "exactly one"),
        ({"x": {"uniform": [1, 0]}}, "low < high"),
        ({"x": {"uniform": [1]}}, "low < high"),
        ({"x": {"loguniform": [0, 1]}}, "low > 0"),
        ({"x": {"int": [1.5, 3]}}, "integers"),
        ({"x": {"choice": []}}, "non-empty"),
        ({"x": {"choice": "abc"}}, "non-empty"),
    ],
)
def test_invalid_spaces_are_rejected(space, message) -> None:
    with pytest.raises(SearchSpaceError, match=message):
        sample_trials(space, n_trials=1)


def test_zero_trials_is_rejected() -> None:
    with pytest.raises(SearchSpaceError, match="n_trials"):
        sample_trials(SPACE, n_trials=0)


def test_overrides_keep_floats_exact() -> None:
    (trial,) = sample_trials(SPACE, n_trials=1, seed=0)

    overrides = dict(o.split("=", 1) for o in to_overrides(trial))

    assert float(overrides["train.lr"]) == trial["train.lr"]  # not rounded
    assert int(overrides["model.hidden"]) == trial["model.hidden"]


# --- the shipped configs -------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_every_baseline_config_loads_and_has_a_model_section_and_a_search_space(name: str) -> None:
    cfg = load_config(CONFIGS / f"{name}.yaml")

    assert cfg.model.name == name and len(cfg.search) >= 3


@pytest.mark.parametrize("name", ["cnn", "gru", "lstm"])
def test_torch_configs_build_their_model_and_valid_train_settings(name: str) -> None:
    cfg = load_config(CONFIGS / f"{name}.yaml")

    model = build_model(cfg.model, n_classes=20)
    settings = TrainSettings.from_cfg(cfg.train)

    assert isinstance(model, CNNBaseline if name == "cnn" else RNNBaseline)
    assert (settings.epochs, settings.batch_size, settings.patience) == (20, 4096, 4)
    assert settings.lr == 0.003 and settings.balanced_cap == 10.0  # spec 005's CNN/RNN recipe


def test_the_xgb_config_builds_the_model() -> None:
    cfg = load_config(CONFIGS / "xgb.yaml")

    model = XGBBaseline.from_config(cfg.model, LabelSpace.create(range(5)), seed=1)

    assert model.params["n_estimators"] == 2000 and model.params["early_stopping_rounds"] == 50
    assert model.params["max_train_flows"] is None


@pytest.mark.parametrize("name", NAMES)
def test_every_search_key_exists_in_its_config_and_yields_a_buildable_model(name: str) -> None:
    """A search over a key the config does not have would silently do nothing (or
    raise mid-search). load_config raises on unknown override keys, so applying
    every sampled trial proves each key is real."""
    path = CONFIGS / f"{name}.yaml"
    space = to_container(load_config(path))["search"]

    for trial in sample_trials(space, n_trials=8, seed=0):
        cfg = load_config(path, to_overrides(trial))
        if name == "xgb":
            XGBBaseline.from_config(cfg.model, LabelSpace.create(range(5)))
        else:
            build_model(cfg.model, n_classes=5)
            TrainSettings.from_cfg(cfg.train)


def test_a_typo_in_a_search_key_is_caught_by_the_config_layer() -> None:
    path = CONFIGS / "cnn.yaml"

    with pytest.raises(ConfigError, match="not in the config"):
        load_config(path, to_overrides({"train.learning_rate": 0.1}))


def test_lstm_and_gru_configs_differ_only_in_the_model_name() -> None:
    gru = to_container(load_config(CONFIGS / "gru.yaml"))
    lstm = to_container(load_config(CONFIGS / "lstm.yaml"))

    assert gru["model"]["name"] == "gru" and lstm["model"]["name"] == "lstm"
    assert {**gru, "model": {**gru["model"], "name": "x"}} == {
        **lstm,
        "model": {**lstm["model"], "name": "x"},
    }
