import subprocess
from pathlib import Path

import pytest


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
            "HOME": str(cwd),
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        },
    )


def _make_commit(repo: Path, msg: str) -> None:
    _git(["add", "-A"], repo)
    _git(["commit", "-m", msg], repo)


@pytest.fixture
def flat_repo(tmp_path: Path) -> Path:
    """Flat layout: modules at repo root."""
    repo = tmp_path / "flat"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    (repo / "app.py").write_text('"""App module."""\nfrom helpers import do_thing\n')
    (repo / "helpers.py").write_text('"""Helper module."""\ndef do_thing():\n    return 42\n')
    (repo / "tests").mkdir()
    (repo / "tests/test_helpers.py").write_text(
        "from helpers import do_thing\n\ndef test_it():\n    assert do_thing() == 42\n"
    )
    _make_commit(repo, "initial")
    return repo


@pytest.fixture
def src_repo(tmp_path: Path) -> Path:
    """Src-layout: package under src/."""
    repo = tmp_path / "srclayout"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    pkg = repo / "src" / "mypkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "server.py").write_text(
        '"""Server module."""\nfrom .payload import build\n\ndef run():\n    return build()\n'
    )
    (pkg / "payload.py").write_text(
        '"""Payload module."""\ndef build():\n    return "ok"\n'
    )
    (repo / "tests").mkdir()
    (repo / "tests/test_payload.py").write_text(
        "from mypkg.payload import build\n\ndef test_build():\n    assert build() == 'ok'\n"
    )
    _make_commit(repo, "initial")
    return repo
