# Paranoia

An MCP server that gets a second, adversarial opinion on your code changes and plans from ChatGPT (GPT-5.4).

## What it does

Paranoia exposes two tools to Claude Code (or any MCP client):

- **`critique_branch`** — review a git branch diff plus relevant context. Returns a structured critique: what works, what doesn't, risks, gaps, improvements.
- **`critique_plan`** — review a markdown plan. Returns the same five-section critique applied to the plan's logic, assumptions, and ordering.

The review is deliberately adversarial. The critic assumes the change is wrong until evidence proves otherwise, cites specific file paths and lines, and tests whether the code actually achieves its stated intent.

## Why

Claude is already in your session and can review its own work, but it reviews with the same biases it wrote with. A different model with a fresh context is a genuine second opinion. Paranoia packages that into one MCP call so you don't have to copy-paste code between tools.

## How it builds context

For `critique_branch`, the server assembles a payload in layers. Each layer is labelled so the critic knows *why* each file is in the payload.

**Priority 0 (always sent):**
- Repo tree (`git ls-files`)
- `CLAUDE.md` and `README.md` if present
- Per-commit narrative (`git log --reverse --patch --stat base..head`) — shows the author's commit messages alongside the patches, so the critic sees intent as well as outcome
- Touched files in full, each prefixed with their last 5 commits
- Tests for touched files — matched by filename (word-level stem match with parent-dir signal for generic stems like `strategy.py`) AND by AST-parsed imports, capped at 8 tests per touched file
- Config files referenced by touched code (regex for `configs/*.yml`, etc.)
- Module docstrings of sibling files in touched directories
- Any `extra_files` the client explicitly flagged, with the reason each was picked

**Priority 1 (dropped only if over budget):**
- One-hop imports of touched Python files
- Design docs — markdown under the repo that references touched files by path or dotted module name

**Priority 2 (dropped first if over budget):**
- Files containing references to new/changed symbols, via `git grep -w`

**Optional deep mode** (`deep: true`) — runs a scouting pass first: sends a lightweight payload (tree, docs, narrative, touched-file summary) to GPT-5.4 and asks it to list up to 15 additional files it wants to see. Those files are merged into the critique payload. Costs roughly 2x tokens; improves context relevance for non-trivial diffs.

The critic receives author-stated context (project summary and diff intent) framed as claims to verify, not facts to accept. Anti-padding rules in the system prompt prevent the critic from opening with preamble or generic praise.

## Prerequisites

- Python 3.11+
- An OpenAI API key with access to GPT-5.4 (either `Chat completions` or `Responses` endpoint scope — the server uses `/v1/responses`)
- `git` on `PATH`

## Install

```bash
git clone <this repo>
cd paranoia-mcp
pip install -e .
```

This puts a `paranoia` executable on your PATH.

## Configure

Set the API key in your shell's persistent env file so the MCP subprocess inherits it:

```bash
echo 'export OPENAI_API_KEY=sk-...' >> ~/.zshenv
source ~/.zshenv
```

Optional env vars:

| Variable | Default | Purpose |
|---|---|---|
| `PARANOIA_MODEL` | `gpt-5.4` | Which OpenAI model to call |
| `PARANOIA_TOKEN_BUDGET` | `250000` | Input token budget (gpt-5.4 standard context is 272K; leave ~22K for response + reasoning) |

## Wire into Claude Code

```bash
claude mcp add paranoia -- paranoia
```

To ensure Claude only uses it when you explicitly ask, add this line to `~/.claude/CLAUDE.md`:

```
Never call the `paranoia` MCP server unless I explicitly ask for adversarial review, critique, or a second opinion.
```

## Tools

### `critique_branch`

| Arg | Type | Required | Purpose |
|---|---|---|---|
| `repo_path` | string | yes | Absolute path to the git repo |
| `base_ref` | string | no (default `main`) | Base ref for the diff |
| `head_ref` | string | no (default `HEAD`) | Head ref for the diff |
| `project_summary` | string | no | Neutral factual description of what the project is and does. Not advocacy. The critic uses this to check whether the diff respects system constraints |
| `diff_intent` | string | no | Neutral factual statement of what the diff is *supposed* to achieve. The critic treats this as a claim to test |
| `focus` | string | no | Narrow the review to a specific concern |
| `extra_files` | array | no | `[{path, reason}]` — files you want the critic to see that the heuristic might miss. Reasons are passed to the critic so it can weight them |
| `deep` | boolean | no (default `false`) | Run a scouting pass first to let the model pick additional files |
| `token_budget` | integer | no | Override the input token budget |

Returns the critic's five-section review: `## What works`, `## What doesn't work`, `## Risks`, `## Gaps`, `## Improvements`.

### `critique_plan`

| Arg | Type | Required | Purpose |
|---|---|---|---|
| `plan_text` | string | one of these | Plan content as markdown |
| `plan_path` | string | one of these | Absolute path to a markdown plan file |
| `context` | string | no | Background the critic needs to judge the plan fairly |

Returns a plan-focused critique: hidden assumptions, unstated dependencies, failure modes, missing rollback, ordering errors, scope creep.

## Example

In Claude Code:

> "Run paranoia on this branch. Project: REST API backend for a booking system. Intent: add per-IP rate limiting to the login endpoint. Use deep mode."

Claude calls:

```json
{
  "name": "critique_branch",
  "arguments": {
    "repo_path": "/Users/you/Work/my-project",
    "base_ref": "main",
    "head_ref": "HEAD",
    "project_summary": "REST API backend for a booking system. Python/FastAPI, Postgres, Redis for session state. Authentication via short-lived JWTs.",
    "diff_intent": "Add per-IP rate limiting to the login endpoint so brute-force attempts are throttled after N failures within a sliding window.",
    "deep": true
  }
}
```

## Known limits

- **Python-centric static analysis** — import parsing uses `ast`. TS/JS repos get the diff, tests, docs, and grep-refs but no import graph.
- **Generic-named files** — `strategy.py`, `config.py` etc. only match tests that reference their parent directory in the filename. Real coverage often comes through imports instead, which the import-based test matcher does catch.
- **No memory across calls** — each critique starts fresh. No accumulated history of prior reviews.
- **Budget is a heuristic** — grep-neighbours are dropped first if the payload exceeds budget. For very large diffs, consider lowering expectations or raising `token_budget` (and paying OpenAI's extended-context pricing if you push past 272K input tokens).

## Security

Paranoia reads your repo and sends its contents to OpenAI. Don't point it at repos containing secrets you wouldn't want OpenAI to retain. The server has no telemetry, no logging, no state; nothing is written back to your repo.
