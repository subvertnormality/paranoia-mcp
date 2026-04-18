from pathlib import Path

import pytest

from paranoia.payload import (
    _imported_modules,
    _rel_path_to_module,
    build_payload,
    build_test_import_index,
    count_tokens,
    find_tests_for,
    find_tests_importing,
    is_safe_rel_path,
    parse_imports,
)


class TestPathSafety:
    def test_rejects_parent_traversal(self, tmp_path: Path) -> None:
        assert is_safe_rel_path(tmp_path, "../../etc/passwd") is False

    def test_rejects_absolute(self, tmp_path: Path) -> None:
        assert is_safe_rel_path(tmp_path, "/etc/passwd") is False

    def test_rejects_empty(self, tmp_path: Path) -> None:
        assert is_safe_rel_path(tmp_path, "") is False

    def test_accepts_in_repo(self, tmp_path: Path) -> None:
        assert is_safe_rel_path(tmp_path, "src/foo.py") is True

    def test_accepts_normalised_inner_traversal(self, tmp_path: Path) -> None:
        assert is_safe_rel_path(tmp_path, "src/../src/foo.py") is True


class TestModulePathMapping:
    def test_flat_layout(self) -> None:
        assert _rel_path_to_module("app.py") == "app"
        assert _rel_path_to_module("pkg/sub.py") == "pkg.sub"

    def test_src_layout_stripped(self) -> None:
        assert _rel_path_to_module("src/mypkg/server.py") == "mypkg.server"
        assert _rel_path_to_module("src/mypkg/__init__.py") == "mypkg"

    def test_non_python_returns_none(self) -> None:
        assert _rel_path_to_module("README.md") is None


class TestParseImports:
    def test_absolute_import_flat(self, flat_repo: Path) -> None:
        src = (flat_repo / "app.py").read_text()
        result = parse_imports(src, flat_repo, importing_rel_path="app.py")
        assert "helpers.py" in result

    def test_relative_import_src_layout(self, src_repo: Path) -> None:
        src = (src_repo / "src/mypkg/server.py").read_text()
        result = parse_imports(
            src, src_repo, importing_rel_path="src/mypkg/server.py"
        )
        assert "src/mypkg/payload.py" in result

    def test_syntax_error_returns_empty(self, tmp_path: Path) -> None:
        assert parse_imports("def (((", tmp_path) == []


class TestTestMatching:
    def test_filename_match(self, flat_repo: Path) -> None:
        all_files = ["app.py", "helpers.py", "tests/test_helpers.py"]
        result = find_tests_for(flat_repo, "helpers.py", all_files)
        assert "tests/test_helpers.py" in result

    def test_import_match_src_layout(self, src_repo: Path) -> None:
        all_files = [
            "src/mypkg/payload.py",
            "src/mypkg/server.py",
            "tests/test_payload.py",
        ]
        index = build_test_import_index(src_repo, all_files)
        result = find_tests_importing("src/mypkg/payload.py", index)
        assert "tests/test_payload.py" in result


class TestBudgetEnforcement:
    def test_over_budget_drops_priority_zero_sections(self, flat_repo: Path) -> None:
        """Touched files are priority 0; before the fix they bypassed budget.
        Now a touched file too large to fit must be dropped with a note."""
        import subprocess

        big = flat_repo / "big.py"
        big.write_text('"""big"""\n' + ("x = 1  # filler\n" * 500))
        env = {
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": str(flat_repo),
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        }
        subprocess.run(["git", "add", "-A"], cwd=flat_repo, check=True, capture_output=True, env=env)
        subprocess.run(["git", "commit", "-m", "big"], cwd=flat_repo, check=True, capture_output=True, env=env)

        payload = build_payload(
            repo_path=str(flat_repo),
            base_ref="main~1",
            head_ref="main",
            extra_files=[],
            token_budget=1500,
        )
        assert "DROPPED" in payload
        assert "big.py" in payload.split("=== DROPPED")[-1]

    def test_under_budget_no_drops(self, flat_repo: Path) -> None:
        payload = build_payload(
            repo_path=str(flat_repo),
            base_ref="main~0",
            head_ref="main",
            extra_files=[],
            token_budget=50000,
        )
        assert "DROPPED" not in payload
        assert count_tokens(payload) < 50000


class TestRelativeImportResolution:
    def test_single_dot(self, src_repo: Path) -> None:
        src = '"""x"""\nfrom .payload import build\n'
        mods = _imported_modules(src, importing_rel_path="src/mypkg/server.py")
        assert "mypkg.payload" in mods
        assert "mypkg.payload.build" in mods

    def test_absolute_import(self) -> None:
        mods = _imported_modules("import json\nfrom os import path\n")
        assert "json" in mods
        assert "os" in mods
        assert "os.path" in mods
