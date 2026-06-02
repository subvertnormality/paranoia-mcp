import pytest

from paranoia.server import _parse_scout_response, _validate_token_budget


class TestScoutParser:
    def test_plain_json_array(self) -> None:
        assert _parse_scout_response('["a.py", "b.py"]') == ["a.py", "b.py"]

    def test_handles_prose_around_array(self) -> None:
        assert _parse_scout_response(
            'Here you go: ["foo.py", "bar.py"] end'
        ) == ["foo.py", "bar.py"]

    def test_handles_code_fences(self) -> None:
        raw = '```json\n["x.py"]\n```'
        assert _parse_scout_response(raw) == ["x.py"]

    def test_non_json_returns_empty(self) -> None:
        assert _parse_scout_response("not json at all") == []

    def test_caps_at_15(self) -> None:
        raw = "[" + ", ".join(f'"f{i}.py"' for i in range(30)) + "]"
        assert len(_parse_scout_response(raw)) == 15

    def test_filters_non_strings(self) -> None:
        assert _parse_scout_response('["a.py", 42, null, "b.py"]') == ["a.py", "b.py"]

    def test_empty_array(self) -> None:
        assert _parse_scout_response("[]") == []


class TestTokenBudgetValidation:
    def test_valid_budget(self) -> None:
        assert _validate_token_budget(50_000) == 50_000

    def test_rejects_too_small(self) -> None:
        with pytest.raises(ValueError, match=">= 1000"):
            _validate_token_budget(500)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            _validate_token_budget(-1)

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            _validate_token_budget(0)

    def test_rejects_over_model_context(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            _validate_token_budget(500_000)

    def test_rejects_non_int(self) -> None:
        with pytest.raises(ValueError, match="integer"):
            _validate_token_budget("50000")

    def test_rejects_bool(self) -> None:
        with pytest.raises(ValueError, match="integer"):
            _validate_token_budget(True)


class TestCritiquePlanValidation:
    @pytest.mark.asyncio
    async def test_rejects_both_args(self) -> None:
        from paranoia.server import call_tool
        with pytest.raises(ValueError, match="not both"):
            await call_tool("critique_plan", {
                "plan_text": "x",
                "plan_path": "/tmp/y.md",
            })

    @pytest.mark.asyncio
    async def test_rejects_neither_arg(self) -> None:
        from paranoia.server import call_tool
        with pytest.raises(ValueError, match="requires plan_text or plan_path"):
            await call_tool("critique_plan", {})


class TestCritiquePlanRepoGrounding:
    @pytest.mark.asyncio
    async def test_no_repo_path_sends_plan_only(self, monkeypatch) -> None:
        """Backward compat: without repo_path the critic still gets only the
        plan (+context), no REPOSITORY CONTEXT section."""
        from paranoia import server

        captured: dict[str, str] = {}

        def fake_gpt(system: str, user: str) -> str:
            captured["user"] = user
            return "ok"

        monkeypatch.setattr(server, "_gpt", fake_gpt)
        await server.call_tool("critique_plan", {"plan_text": "do the thing"})
        assert "=== PLAN ===" in captured["user"]
        assert "REPOSITORY CONTEXT" not in captured["user"]

    @pytest.mark.asyncio
    async def test_repo_path_sends_actual_code(self, flat_repo, monkeypatch) -> None:
        """The core fix: with repo_path + files, the plan critic receives the
        real source, not just the plan text."""
        from paranoia import server

        captured: dict[str, str] = {}

        def fake_gpt(system: str, user: str) -> str:
            captured["user"] = user
            return "ok"

        monkeypatch.setattr(server, "_gpt", fake_gpt)
        await server.call_tool("critique_plan", {
            "plan_text": "Change do_thing to return 0",
            "repo_path": str(flat_repo),
            "files": [{"path": "helpers.py", "reason": "the function being changed"}],
        })
        assert "=== REPOSITORY CONTEXT ===" in captured["user"]
        assert "=== TREE ===" in captured["user"]
        assert "def do_thing" in captured["user"]  # actual code reached the critic

    @pytest.mark.asyncio
    async def test_deep_scout_pulls_in_files(self, flat_repo, monkeypatch) -> None:
        """deep=True runs a scouting pass; files the scout names are included
        even when the author flagged none."""
        from paranoia import server

        calls: list[str] = []
        captured: dict[str, str] = {}

        def fake_gpt(system: str, user: str) -> str:
            calls.append(system)
            if "scouting pass" in system or "NOT reviewing yet" in system:
                return '["helpers.py"]'
            captured["user"] = user
            return "ok"

        monkeypatch.setattr(server, "_gpt", fake_gpt)
        await server.call_tool("critique_plan", {
            "plan_text": "Change do_thing",
            "repo_path": str(flat_repo),
            "deep": True,
        })
        # Two model calls: the scout, then the review.
        assert len(calls) == 2
        assert "def do_thing" in captured["user"]

    @pytest.mark.asyncio
    async def test_bad_repo_path_returns_error_not_raises(self, monkeypatch) -> None:
        from paranoia import server

        monkeypatch.setattr(server, "_gpt", lambda s, u: "ok")
        result = await server.call_tool("critique_plan", {
            "plan_text": "x",
            "repo_path": "/nonexistent/repo/path",
        })
        assert "[paranoia error]" in result[0].text
