"""``git_dirty`` and its interaction with ``git_commit`` (spec 014)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from adl_etc.utils.provenance import git_commit, git_dirty

GIT_IDENTITY = ["-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", *GIT_IDENTITY, *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    (r / "a.txt").write_text("one\n", encoding="utf-8")
    git(r, "add", "a.txt")
    git(r, "commit", "-q", "-m", "init")
    return r


def test_clean_checkout_is_not_dirty(repo: Path) -> None:
    assert git_dirty(repo) is False
    assert git_commit(repo) == git(repo, "rev-parse", "HEAD")


def test_modified_tracked_file_is_dirty(repo: Path) -> None:
    (repo / "a.txt").write_text("two\n", encoding="utf-8")

    assert git_dirty(repo) is True


def test_untracked_file_is_dirty(repo: Path) -> None:
    (repo / "new.py").write_text("x = 1\n", encoding="utf-8")

    assert git_dirty(repo) is True


def test_committing_the_change_makes_it_clean_again(repo: Path) -> None:
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", "edit")

    assert git_dirty(repo) is False


def test_outside_a_checkout_is_unknown_not_clean(tmp_path: Path) -> None:
    """``None``, never ``False``: "could not check" must not read as "clean"."""
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()

    assert git_dirty(not_a_repo) is None
    assert git_commit(not_a_repo) == "unknown"
