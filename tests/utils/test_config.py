"""Config composition, overrides and hashing (spec 014, plan phase 2 T1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from adl_etc.utils.config import (
    ConfigError,
    config_hash,
    flatten,
    load_config,
    save_resolved,
    to_container,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- composition ---------------------------------------------------------------


def test_child_overrides_its_defaults_and_defaults_key_is_dropped(tmp_path: Path) -> None:
    write(tmp_path / "base.yaml", "optim:\n  lr: 0.001\n  wd: 0.0001\nseed: 0\n")
    child = write(
        tmp_path / "child.yaml",
        "defaults: [base.yaml]\noptim:\n  lr: 0.003\nmodel: cnn\n",
    )

    cfg = load_config(child)

    assert cfg.optim.lr == 0.003  # child wins
    assert cfg.optim.wd == 0.0001  # inherited, not clobbered by the child's partial `optim`
    assert cfg.seed == 0
    assert cfg.model == "cnn"
    assert "defaults" not in cfg


def test_later_default_wins_over_earlier(tmp_path: Path) -> None:
    write(tmp_path / "a.yaml", "x: 1\ny: 1\n")
    write(tmp_path / "b.yaml", "x: 2\n")
    child = write(tmp_path / "child.yaml", "defaults: [a.yaml, b.yaml]\n")

    cfg = load_config(child)

    assert (cfg.x, cfg.y) == (2, 1)


def test_default_paths_are_relative_to_the_file_that_names_them(tmp_path: Path) -> None:
    write(tmp_path / "shared" / "leaf.yaml", "depth: leaf\n")
    write(tmp_path / "shared" / "mid.yaml", "defaults: [leaf.yaml]\nmid: true\n")
    top = write(tmp_path / "runs" / "top.yaml", "defaults: [../shared/mid.yaml]\ntop: true\n")

    cfg = load_config(top)

    assert (cfg.depth, cfg.mid, cfg.top) == ("leaf", True, True)


def test_cli_override_beats_child_and_defaults(tmp_path: Path) -> None:
    write(tmp_path / "base.yaml", "seed: 0\nlr: 0.1\n")
    child = write(tmp_path / "child.yaml", "defaults: [base.yaml]\nlr: 0.2\n")

    cfg = load_config(child, ["lr=0.3", "seed=7"])

    assert cfg.lr == 0.3
    assert cfg.seed == 7 and isinstance(cfg.seed, int)  # parsed as YAML, not left a string


def test_circular_defaults_raise(tmp_path: Path) -> None:
    write(tmp_path / "a.yaml", "defaults: [b.yaml]\n")
    write(tmp_path / "b.yaml", "defaults: [a.yaml]\n")

    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_missing_default_file_raises(tmp_path: Path) -> None:
    child = write(tmp_path / "child.yaml", "defaults: [nope.yaml]\n")

    with pytest.raises(ConfigError, match="not found"):
        load_config(child)


def test_defaults_as_a_bare_string_raises_rather_than_iterating_characters(tmp_path: Path) -> None:
    child = write(tmp_path / "child.yaml", "defaults: base.yaml\n")

    with pytest.raises(ConfigError, match="list of paths"):
        load_config(child)


# --- overrides -------------------------------------------------------------------


def test_override_of_an_unknown_key_raises(tmp_path: Path) -> None:
    cfg_path = write(tmp_path / "c.yaml", "optim:\n  lr: 0.1\n")

    with pytest.raises(ConfigError, match="not in the config"):
        load_config(cfg_path, ["optim.learning_rate=0.5"])  # typo for optim.lr

    with pytest.raises(ConfigError, match="not in the config"):
        load_config(cfg_path, ["lr=0.5"])


def test_config_is_read_only(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path / "c.yaml", "seed: 0\n"))

    with pytest.raises(Exception, match="read-only|readonly"):
        cfg.seed = 5


def test_interpolations_are_resolved(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path / "c.yaml", "a: 4\nb: ${a}\n"))

    assert cfg.b == 4
    assert to_container(cfg) == {"a": 4, "b": 4}


# --- hashing -----------------------------------------------------------------------


def test_hash_ignores_key_order_comments_and_whitespace(tmp_path: Path) -> None:
    one = load_config(write(tmp_path / "1.yaml", "a: 1\nb:\n  c: 2\n  d: 3\n"))
    two = load_config(
        write(tmp_path / "2.yaml", "# a comment\nb:\n    d: 3\n    c: 2   # trailing\n\na: 1\n")
    )

    assert config_hash(one) == config_hash(two)


def test_hash_changes_when_any_value_changes(tmp_path: Path) -> None:
    path = write(tmp_path / "c.yaml", "a: 1\nb:\n  c: 2\n")

    base = config_hash(load_config(path))

    assert config_hash(load_config(path, ["a=2"])) != base
    assert config_hash(load_config(path, ["b.c=3"])) != base


def test_hash_of_interpolated_config_equals_hash_of_its_literal_form(tmp_path: Path) -> None:
    interpolated = load_config(write(tmp_path / "i.yaml", "a: 4\nb: ${a}\n"))
    literal = load_config(write(tmp_path / "l.yaml", "a: 4\nb: 4\n"))

    assert config_hash(interpolated) == config_hash(literal)


def test_hash_distinguishes_type_not_just_text(tmp_path: Path) -> None:
    as_int = load_config(write(tmp_path / "i.yaml", "x: 1\n"))
    as_str = load_config(write(tmp_path / "s.yaml", "x: '1'\n"))

    assert config_hash(as_int) != config_hash(as_str)


# --- flatten / save -------------------------------------------------------------


def test_flatten_gives_dotted_keys(tmp_path: Path) -> None:
    text = "seed: 0\noptim:\n  lr: 0.1\n  sched:\n    enabled: true\n"
    cfg = load_config(write(tmp_path / "c.yaml", text))

    assert flatten(cfg) == {"seed": 0, "optim.lr": 0.1, "optim.sched.enabled": True}


def test_save_resolved_stands_alone_and_keeps_the_hash(tmp_path: Path) -> None:
    write(tmp_path / "base.yaml", "a: 4\n")
    cfg = load_config(write(tmp_path / "c.yaml", "defaults: [base.yaml]\nb: ${a}\n"), ["a=9"])

    dest = tmp_path / "out" / "config_resolved.yaml"
    save_resolved(cfg, dest)

    text = dest.read_text(encoding="utf-8")
    assert "${" not in text and "defaults" not in text
    assert config_hash(OmegaConf.load(dest)) == config_hash(cfg)  # type: ignore[arg-type]


# --- the repo's own configs ---------------------------------------------------------


@pytest.mark.parametrize(
    "rel",
    [
        "configs/data/pcap.yaml",
        "configs/splits/d1_main.yaml",
        "configs/splits/d2_quic.yaml",
        "configs/splits/d3_grouped.yaml",
        "configs/splits/d4_anomaly.yaml",
    ],
)
def test_existing_repo_configs_load_unchanged(rel: str) -> None:
    """The composer must accept every config phase 1 already shipped."""
    cfg = load_config(REPO_ROOT / rel)

    assert len(config_hash(cfg)) == 64
