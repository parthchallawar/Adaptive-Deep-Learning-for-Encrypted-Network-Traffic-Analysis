"""The shipped phase-3 training configs (plan phase 3, T4 and T6): they load,
build, name distinct runs, give every label regime at least spec 008's 3,000
optimizer steps, and a pretrained run cannot start without its checkpoint."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from adl_etc.models.pat import PAT
from adl_etc.models.registry import build_model
from adl_etc.training.loop import ClassifyObjective, TrainSettings
from adl_etc.training.objectives import PrefixObjective, build_objective
from adl_etc.utils.config import load_config
from adl_etc.utils.runinfo import make_run_name

TRAIN = Path(__file__).resolve().parents[2] / "configs" / "train"
N_TRAIN_D1_OPEN = 5_420_243  # real, plan phase 3 T1a: d1_open's train split after filtering
FT = ["ft_pat_d1", "ft_pat_d1_l10", "ft_pat_d1_l01"]
FT_SSL = ["ft_pat_ssl_d1", "ft_pat_ssl_d1_l10", "ft_pat_ssl_d1_l01"]
DESIGN_ABLATIONS = {
    "abl_ramp1": ["objective.ramp=1.0"],
    "abl_ls0": ["train.label_smoothing=0.0"],
    "abl_drop0": ["model.dropout=0.0"],
    "abl_noside": ["model.side_channel=false"],
    "abl_noaug": ["finetune_augment.drop_p=0.0", "finetune_augment.ipt_s=0.0"],
}


def load(name: str, *overrides: str):
    extra = ["init_from=some/ssl.pt"] if name.startswith("ft_pat_ssl") else []
    return load_config(TRAIN / f"{name}.yaml", [*extra, *overrides])


def optimizer_steps(cfg) -> int:
    s = TrainSettings.from_cfg(cfg.train)
    per_epoch = s.samples_per_epoch or round(N_TRAIN_D1_OPEN * float(cfg.label_fraction))
    return math.ceil(per_epoch / s.batch_size) * s.epochs


@pytest.mark.parametrize("name", ["b5_pat_d1", *FT, *FT_SSL])
def test_every_phase3_config_builds_its_model_and_objective(name: str) -> None:
    cfg = load(name)
    model = build_model(cfg.model, 150)
    settings = TrainSettings.from_cfg(cfg.train)
    objective = build_objective(cfg.get("objective"), model=model, n_classes=150,
                                settings=settings)  # fmt: skip

    assert isinstance(model, PAT) and model.side_channel
    assert cfg.split == "configs/splits/d1_open.yaml" and cfg.split_name == "d1open"
    if name == "b5_pat_d1":
        assert isinstance(objective, ClassifyObjective) and objective.final_only
        assert objective.key == "cls" and settings.ema_decay is None  # the shared recipe
    else:
        assert isinstance(objective, PrefixObjective) and objective.monitor == "val_auc_k"
        assert settings.ema_decay == 0.999 and not settings.decay_norm_and_bias


@pytest.mark.parametrize("name", [*FT, *FT_SSL])
def test_every_label_regime_gets_at_least_3000_optimizer_steps(name: str) -> None:
    assert optimizer_steps(load(name)) >= 3000


def test_the_1_percent_regime_validates_every_500_steps() -> None:
    s = TrainSettings.from_cfg(load("ft_pat_d1_l01").train)
    assert s.samples_per_epoch is not None
    assert math.ceil(s.samples_per_epoch / s.batch_size) == 500


def test_ssl_configs_differ_from_scratch_only_in_init_lr_and_layer_decay() -> None:
    for scratch, ssl in zip(FT, FT_SSL, strict=True):
        a = OmegaConf.to_container(load(scratch))
        b = OmegaConf.to_container(load(ssl))
        assert isinstance(a, dict) and isinstance(b, dict)
        assert b.pop("init_from") == "some/ssl.pt"
        assert (b["train"].pop("lr"), b["train"].pop("layer_decay")) == (0.0005, 0.85)
        a["train"].pop("lr")
        assert b.pop("variant").startswith("ssl") and a.pop("variant").startswith("scratch")
        assert a == b


def test_every_run_name_is_distinct() -> None:
    names = set()
    for name in ["b5_pat_d1", *FT, *FT_SSL]:
        cfg = load(name)
        names.add(make_run_name(cfg.stage, cfg.model.name, cfg.variant, cfg.split_name, 0))
    for variant, overrides in DESIGN_ABLATIONS.items():
        cfg = load("ft_pat_d1", f"variant={variant}", *overrides)
        names.add(make_run_name(cfg.stage, cfg.model.name, cfg.variant, cfg.split_name, 0))
    assert len(names) == 7 + len(DESIGN_ABLATIONS)


@pytest.mark.parametrize("variant", list(DESIGN_ABLATIONS))
def test_each_design_ablation_changes_exactly_its_one_setting(variant: str) -> None:
    """``load_config`` raises on an override of a key the config lacks, so this
    also proves every ablation names a real setting."""
    base = OmegaConf.to_container(load("ft_pat_d1"))
    cfg = load("ft_pat_d1", f"variant={variant}", *DESIGN_ABLATIONS[variant])
    changed = OmegaConf.to_container(cfg)
    assert isinstance(base, dict) and isinstance(changed, dict)

    diffs = [
        f"{section}.{key}" for section in ("train", "model", "objective", "finetune_augment")
        for key in base[section] if base[section][key] != changed[section][key]
    ]  # fmt: skip
    assert diffs == [o.split("=")[0] for o in DESIGN_ABLATIONS[variant]]


def test_a_pretrained_config_refuses_to_start_without_its_checkpoint(tmp_path: Path) -> None:
    from adl_etc.training import run as R
    from tests.training.test_run import PAT_SMALL, build_cfg

    cfg = load_config(build_cfg(tmp_path, model=PAT_SMALL,
                                extra="objective: {name: prefix}\ninit_from: ???"))  # fmt: skip
    with pytest.raises(ValueError, match="needs `init_from`"):
        R.run_training(cfg, seed=0)


PT = ["pt_pat_npp_pfc_d1", "pt_pat_npp_d1", "pt_pat_pfc_d1", "pt_pat_mpm_d1"]


@pytest.mark.parametrize("name", PT)
def test_every_pretraining_config_builds_a_label_free_run(name: str) -> None:
    from adl_etc.training.ssl import MPMObjective, PretrainPAT, SSLObjective

    cfg = load(name)
    model = build_model(cfg.model, 150)
    settings = TrainSettings.from_cfg(cfg.train)
    objective = build_objective(cfg.objective, model=model, n_classes=150, settings=settings)

    assert isinstance(model, PretrainPAT) and not objective.uses_labels
    assert cfg.split == "configs/splits/d1_open.yaml" and cfg.stage == "pt"
    assert (settings.lr, settings.weight_decay, settings.grad_clip) == (0.001, 0.05, 1.0)
    if name == "pt_pat_mpm_d1":
        assert isinstance(objective, MPMObjective) and not model.causal
    else:
        assert isinstance(objective, SSLObjective) and model.causal
        weights = (objective.npp_weight, objective.pfc_weight)
        assert weights == {"pt_pat_npp_pfc_d1": (1.0, 1.0), "pt_pat_npp_d1": (1.0, 0.0),
                           "pt_pat_pfc_d1": (0.0, 1.0)}[name]  # fmt: skip


def test_the_lambda_ablations_are_one_override_each() -> None:
    for variant, weight in (("npp_pfc_l05", 0.5), ("npp_pfc_l2", 2.0)):
        cfg = load("pt_pat_npp_pfc_d1", f"variant={variant}", f"objective.pfc_weight={weight}")
        assert cfg.objective.pfc_weight == weight and cfg.variant == variant


def test_pretraining_run_names_are_distinct_from_each_other_and_fine_tuning() -> None:
    names = {
        make_run_name(c.stage, c.model.name, c.variant, c.split_name, 0)
        for c in [load(n) for n in [*PT, "b5_pat_d1", *FT, *FT_SSL]]
    }
    assert len(names) == len(PT) + 1 + len(FT) + len(FT_SSL)


def test_the_committed_kaggle_queue_is_valid_open_set_and_distinct() -> None:
    """Every entry of kernel/run_queue.yaml resolves (an override of a key the
    config lacks would raise), trains on the open-set split, and names a run of
    its own, so no entry is silently skipped as "already finished"."""
    repo = TRAIN.parents[1]
    queue = OmegaConf.to_container(OmegaConf.load(repo / "kernel" / "run_queue.yaml"))
    assert isinstance(queue, list) and queue

    names = []
    for entry in queue:
        cfg = load_config(repo / entry["config"], list(entry.get("overrides", [])))
        assert cfg.split == "configs/splits/d1_open.yaml" and cfg.device == "cuda"
        names.append(make_run_name(cfg.stage, cfg.model.name, cfg.variant, cfg.split_name,
                                   int(cfg.seed)))  # fmt: skip
    assert len(set(names)) == len(names)
