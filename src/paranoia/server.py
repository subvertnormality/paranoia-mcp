import asyncio
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from openai import OpenAI

import json
import re

from .payload import build_payload, build_scout_payload

# gpt-5.5 standard context: 272K tokens. 1M opt-in tier costs 2x input above
# 272K, so we stay under. Reserve ~22K for the model's response + reasoning.
MODEL_CONTEXT_STANDARD = 272_000
OUTPUT_RESERVE = 22_000
DEFAULT_INPUT_BUDGET = int(
    os.environ.get("PARANOIA_TOKEN_BUDGET", MODEL_CONTEXT_STANDARD - OUTPUT_RESERVE)
)

CODE_SYSTEM_PROMPT = """You are Paranoia, a rigorous reviewer of code changes. You assume the diff is wrong until evidence proves otherwise — but you also name what works, so the review is useful rather than merely destructive.

You will receive:
- AUTHOR-STATED PROJECT CONTEXT — treat as description, not advocacy.
- AUTHOR-STATED DIFF INTENT — treat as a CLAIM to verify, not a fact to accept.
- Repo tree, CLAUDE.md/README, commit narrative, touched files with recent git history, relevant tests, one-hop imports, referenced configs, design docs, sibling docstrings, and any files Claude explicitly flagged.

Produce the review in EXACTLY these five sections, in order, using these headings verbatim:

## What works
Specific correct decisions the diff makes. One bullet each, cite paths and quote lines. If you cannot name something concrete, write "Nothing notable." Do NOT pad with generic praise like "the code is clean" or "good use of types."

## What doesn't work
Actual defects: bugs, broken logic, invariant violations, off-by-ones, type confusion, race conditions, security holes, tests that don't test what they claim. For each: quote the offending lines with file path, explain the failure mechanism, state the observable symptom. Ordered by severity (worst first).

## Risks
Failure modes the author did not consider but which the code is exposed to. Hidden assumptions. Edge data. Partial-failure scenarios. Silent regressions in areas the diff doesn't directly touch. Each item must be specific and testable — not "could be slow" but "with N>10k, the O(N²) join in foo.py:42 will timeout the request."

## Gaps
Things the diff SHOULD do to achieve its stated intent but doesn't. Missing tests for the new behavior. Missing error handling at real system boundaries. Missing config or doc updates implied by the code change. Missing migrations or rollouts. Not hypothetical — only gaps that block the stated goal.

## Improvements
Concrete, specific changes that would make the code more correct, safer, or make its invariants easier to reason about. Not style nitpicks. Not renamings. Each must change behaviour, robustness, or clarity in a way you can describe in one sentence.

Rules across all sections:
- Quote file paths and the offending code. No hand-waving.
- No hedging ("it might be worth considering"). Either a thing is a problem or it isn't.
- No preamble. No trailing summary. Go straight into the sections.
- If a criticism depends on context you were not given, say so explicitly and continue.
- Compare the diff to the AUTHOR-STATED INTENT: does the code actually do what the author claims?
- Order items within each section by severity; if a section is genuinely empty, write "Nothing notable." and move on.
"""

SCOUT_SYSTEM_PROMPT = """You are the scouting pass for an adversarial code review. You are NOT reviewing yet.

Read the repo tree, project docs, commit narrative, and touched-file list. Identify up to 15 additional files (repo-relative paths) you want to see in the full critique pass. Pick files likely to expose:
- hidden contracts / invariants the touched code depends on
- call-sites that would break if the change is wrong
- related configs, fixtures, or tests a naive reviewer would miss
- prior patterns the change should conform to (or diverge from, suspiciously)

Output ONLY a JSON array of path strings. No prose, no markdown fences, no explanation. Example:
["src/auth/middleware.py", "config/database.yml", "tests/test_login_flow.py"]

Skip files already listed as touched — those are already in the payload.
"""


PLAN_SYSTEM_PROMPT = """You are Paranoia, an adversarial plan reviewer. Assume the plan you are shown will fail in ways the author hasn't considered.

Find:
- hidden assumptions treated as facts
- unstated dependencies (on people, systems, data, timing)
- failure modes and what happens when each step doesn't go to plan
- missing rollback / exit criteria
- steps that are vague enough to hide real work ("integrate X", "handle errors")
- ordering errors — steps that depend on outputs of later steps
- scope that has quietly expanded beyond the stated goal
- success criteria that are unmeasurable or moved goalposts
- alternatives the author didn't consider and why they were rejected (or weren't)

Rules:
- Quote the specific claim or step you are attacking.
- No praise. No hedging. No "overall the plan is solid" preamble.
- If the plan is genuinely sound, say so in one sentence and stop.
- Order findings by severity: things that kill the plan first, nitpicks last.
"""

server: Server = Server("paranoia")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="critique_branch",
            description=(
                "Send a git branch diff plus project context and relevant files "
                "to GPT-5 for adversarial review. Returns the critic's findings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Absolute path to the git repository.",
                    },
                    "base_ref": {"type": "string", "default": "main"},
                    "head_ref": {"type": "string", "default": "HEAD"},
                    "extra_files": {
                        "type": "array",
                        "description": (
                            "Additional files to include beyond the automatic "
                            "touched/imports/grep selection. Each requires a reason."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["path", "reason"],
                        },
                        "default": [],
                    },
                    "project_summary": {
                        "type": "string",
                        "description": (
                            "Neutral factual summary of what this project is and what it's for. "
                            "Describe the system, not your opinion of it. Include: what the code does, "
                            "who uses it, key constraints or invariants. NO advocacy. NO 'this is a "
                            "well-designed X' — just 'this is an X, it does Y, it must obey Z.' "
                            "The critic will use this to understand system context and to TEST whether "
                            "the diff actually respects it."
                        ),
                    },
                    "diff_intent": {
                        "type": "string",
                        "description": (
                            "Neutral factual summary of what the diff is SUPPOSED to achieve, stated "
                            "as the author's stated goal. No opinion on whether the approach is good. "
                            "Example: 'Add per-IP rate limiting to the login endpoint so that brute-force "
                            "attempts are throttled after N failures within a sliding window.' "
                            "The critic will treat this as a claim to TEST: "
                            "does the code actually achieve this? Are there failure modes the author "
                            "didn't consider?"
                        ),
                    },
                    "focus": {
                        "type": "string",
                        "description": "Optional — narrow the review to a specific concern.",
                    },
                    "deep": {
                        "type": "boolean",
                        "description": (
                            "If true, run a scouting pass first: ask the model which files it "
                            "needs to see, then include them in the critique payload. Costs ~2x "
                            "tokens but improves context relevance. Default false."
                        ),
                        "default": False,
                    },
                    "token_budget": {"type": "integer", "default": DEFAULT_INPUT_BUDGET},
                },
                "required": ["repo_path"],
            },
        ),
        Tool(
            name="critique_plan",
            description=(
                "Send a plan (markdown text or a markdown file) to GPT-5 for "
                "adversarial review. Returns the critic's findings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "plan_text": {
                        "type": "string",
                        "description": "The plan content as markdown. Provide this OR plan_path.",
                    },
                    "plan_path": {
                        "type": "string",
                        "description": "Absolute path to a markdown plan file. Provide this OR plan_text.",
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Optional background — what the plan is for, constraints, prior attempts, "
                            "anything the critic needs to judge the plan fairly."
                        ),
                    },
                },
            },
        ),
    ]


def _parse_scout_response(raw: str) -> list[str]:
    """Extract a list of file paths from the scout's response. Tolerates code
    fences or minor prose around a JSON array."""
    text = raw.strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [p for p in data if isinstance(p, str) and p][:15]


def _gpt(system_prompt: str, user_content: str) -> str:
    client = OpenAI()
    model = os.environ.get("PARANOIA_MODEL", "gpt-5.5")
    try:
        resp = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_content,
        )
    except Exception as exc:
        return f"[paranoia error] OpenAI call failed: {type(exc).__name__}: {exc}"
    return resp.output_text or "[paranoia error] OpenAI returned empty response"


def _validate_token_budget(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"token_budget must be an integer, got {type(value).__name__}")
    if value < 1000:
        raise ValueError(f"token_budget must be >= 1000, got {value}")
    if value > MODEL_CONTEXT_STANDARD:
        raise ValueError(
            f"token_budget {value} exceeds gpt-5.5 standard context ({MODEL_CONTEXT_STANDARD}). "
            f"Using the 1M extended tier requires extra API params not configured here."
        )
    return value


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "critique_branch":
        repo_path = arguments["repo_path"]
        base_ref = arguments.get("base_ref", "main")
        head_ref = arguments.get("head_ref", "HEAD")
        extra_files = list(arguments.get("extra_files", []))
        token_budget = _validate_token_budget(
            arguments.get("token_budget", DEFAULT_INPUT_BUDGET)
        )

        if arguments.get("deep"):
            try:
                scout_payload = build_scout_payload(repo_path, base_ref, head_ref)
            except Exception as exc:
                return [TextContent(
                    type="text",
                    text=f"[paranoia error] scout payload build failed: {type(exc).__name__}: {exc}",
                )]
            scout_raw = await asyncio.to_thread(_gpt, SCOUT_SYSTEM_PROMPT, scout_payload)
            scout_paths = _parse_scout_response(scout_raw)
            existing = {e["path"] for e in extra_files}
            for p in scout_paths:
                if p not in existing:
                    extra_files.append({"path": p, "reason": "scouting pass"})

        try:
            payload = build_payload(
                repo_path=repo_path,
                base_ref=base_ref,
                head_ref=head_ref,
                extra_files=extra_files,
                token_budget=token_budget,
            )
        except Exception as exc:
            return [TextContent(
                type="text",
                text=f"[paranoia error] payload build failed: {type(exc).__name__}: {exc}",
            )]

        header_sections: list[str] = [
            "Review the following branch diff. Produce the five-section output defined in your instructions. "
            "Cite specific file paths and lines. Test whether the code actually achieves the author-stated intent."
        ]
        if summary := arguments.get("project_summary"):
            header_sections.append(f"=== AUTHOR-STATED PROJECT CONTEXT ===\n{summary}")
        if intent := arguments.get("diff_intent"):
            header_sections.append(f"=== AUTHOR-STATED DIFF INTENT ===\n{intent}")
        if focus := arguments.get("focus"):
            header_sections.append(f"=== REVIEWER FOCUS ===\n{focus}")

        user_content = "\n\n".join(header_sections) + "\n\n" + payload
        result = await asyncio.to_thread(_gpt, CODE_SYSTEM_PROMPT, user_content)
        return [TextContent(type="text", text=result)]

    if name == "critique_plan":
        plan_text = arguments.get("plan_text")
        plan_path = arguments.get("plan_path")
        if plan_text and plan_path:
            raise ValueError(
                "critique_plan requires exactly one of plan_text or plan_path, not both"
            )
        if not plan_text and not plan_path:
            raise ValueError("critique_plan requires plan_text or plan_path")
        if plan_path:
            from pathlib import Path
            try:
                plan_text = Path(plan_path).read_text(encoding="utf-8", errors="replace")
            except (FileNotFoundError, IsADirectoryError, PermissionError, OSError) as exc:
                return [TextContent(
                    type="text",
                    text=f"[paranoia error] cannot read plan_path: {type(exc).__name__}: {exc}",
                )]
        user_content = f"=== PLAN ===\n{plan_text}"
        if ctx := arguments.get("context"):
            user_content = f"=== CONTEXT ===\n{ctx}\n\n{user_content}"
        result = await asyncio.to_thread(_gpt, PLAN_SYSTEM_PROMPT, user_content)
        return [TextContent(type="text", text=result)]

    raise ValueError(f"Unknown tool: {name}")


def main() -> None:
    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_run())


if __name__ == "__main__":
    main()
