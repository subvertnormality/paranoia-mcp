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
