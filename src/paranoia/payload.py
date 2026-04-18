import ast
import re
import subprocess
from pathlib import Path

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(s: str) -> int:
        return len(_enc.encode(s, disallowed_special=()))
except ImportError:
    def count_tokens(s: str) -> int:
        return len(s) // 4


SYMBOL_PATTERN = re.compile(
    r"^\+\s*(?:async\s+)?(?:def|class)\s+(\w+)", re.MULTILINE
)

SKIP_SYMBOLS = {"__init__", "main", "setUp", "tearDown", "test"}


def _run(cmd: list[str], cwd: Path) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {r.stderr.strip()}")
    return r.stdout


def get_diff(repo: Path, base: str, head: str) -> str:
    return _run(["git", "diff", f"{base}...{head}"], repo)


def get_commit_narrative(repo: Path, base: str, head: str) -> str:
    """Per-commit story: each commit's message + its patch, oldest first."""
    try:
        return _run(
            ["git", "log", "--reverse", "--patch", "--stat", f"{base}..{head}"],
            repo,
        )
    except RuntimeError:
        return ""


def get_touched_files(repo: Path, base: str, head: str) -> list[str]:
    out = _run(["git", "diff", "--name-only", f"{base}...{head}"], repo)
    return [l for l in out.splitlines() if l.strip()]


def extract_changed_symbols(diff_text: str) -> list[str]:
    found = set(SYMBOL_PATTERN.findall(diff_text)) - SKIP_SYMBOLS
    return sorted(s for s in found if not s.startswith("_"))


_PACKAGE_ROOT_CANDIDATES = ("", "src")


def parse_imports(
    py_source: str, repo_root: Path, importing_rel_path: str | None = None
) -> list[str]:
    try:
        tree = ast.parse(py_source)
    except SyntaxError:
        return []
    # Resolve the importing file's dotted package for relative import handling.
    importing_pkg_parts: list[str] = []
    if importing_rel_path:
        mod = _rel_path_to_module(importing_rel_path)
        if mod:
            importing_pkg_parts = mod.split(".")[:-1]

    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    mods.add(node.module)
            elif importing_pkg_parts:
                # Relative import: go up `level-1` from the importing package.
                drop = node.level - 1
                if drop < len(importing_pkg_parts) or drop == 0:
                    anchor = (
                        importing_pkg_parts
                        if drop == 0
                        else importing_pkg_parts[:-drop]
                    )
                    parts = list(anchor)
                    if node.module:
                        parts.extend(node.module.split("."))
                    if parts:
                        mods.add(".".join(parts))

    paths: list[str] = []
    for mod in mods:
        rel = mod.replace(".", "/")
        for pkg_root in _PACKAGE_ROOT_CANDIDATES:
            base = repo_root / pkg_root if pkg_root else repo_root
            candidate = base / f"{rel}.py"
            if candidate.is_file():
                paths.append(candidate.relative_to(repo_root).as_posix())
                break
            pkg_init = base / rel / "__init__.py"
            if pkg_init.is_file():
                paths.append(pkg_init.relative_to(repo_root).as_posix())
                break
    return paths


def grep_refs(repo: Path, symbol: str) -> list[str]:
    try:
        out = _run(["git", "grep", "-l", "-w", symbol], repo)
    except RuntimeError:
        return []
    return [l for l in out.splitlines() if l.strip()]


def is_safe_rel_path(repo: Path, rel_path: str) -> bool:
    """Reject absolute paths and paths that escape the repo via `..`."""
    if not rel_path or rel_path.startswith("/"):
        return False
    try:
        resolved = (repo / rel_path).resolve()
        repo_resolved = repo.resolve()
    except (ValueError, OSError):
        return False
    try:
        resolved.relative_to(repo_resolved)
        return True
    except ValueError:
        return False


def read_file(repo: Path, rel_path: str) -> str | None:
    if not is_safe_rel_path(repo, rel_path):
        return None
    try:
        return (repo / rel_path).read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, IsADirectoryError, PermissionError, OSError):
        return None


def file_history(repo: Path, rel_path: str, n: int = 5) -> str:
    try:
        out = _run(
            ["git", "log", f"-n{n}", "--oneline", "--", rel_path], repo
        )
    except RuntimeError:
        return ""
    return out.strip()


def _rel_path_to_module(rel_path: str) -> str | None:
    if not rel_path.endswith(".py"):
        return None
    stripped = rel_path
    for pkg_root in _PACKAGE_ROOT_CANDIDATES:
        if not pkg_root:
            continue
        prefix = pkg_root + "/"
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    mod = stripped[:-3].replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    return mod or None


def _imported_modules(py_source: str, importing_rel_path: str | None = None) -> set[str]:
    try:
        tree = ast.parse(py_source)
    except SyntaxError:
        return set()
    importing_pkg_parts: list[str] = []
    if importing_rel_path:
        mod = _rel_path_to_module(importing_rel_path)
        if mod:
            importing_pkg_parts = mod.split(".")[:-1]
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                mods.add(node.module)
                for alias in node.names:
                    mods.add(f"{node.module}.{alias.name}")
            elif node.level > 0 and importing_pkg_parts:
                drop = node.level - 1
                if drop < len(importing_pkg_parts) or drop == 0:
                    anchor = (
                        importing_pkg_parts if drop == 0 else importing_pkg_parts[:-drop]
                    )
                    parts = list(anchor)
                    if node.module:
                        parts.extend(node.module.split("."))
                    if parts:
                        base = ".".join(parts)
                        mods.add(base)
                        for alias in node.names:
                            mods.add(f"{base}.{alias.name}")
    return mods


def _is_test_file(rel_path: str) -> bool:
    if not rel_path.endswith(".py"):
        return False
    name = Path(rel_path).stem
    return name.startswith("test_") or name.endswith("_test")


def build_test_import_index(
    repo: Path, all_files: list[str]
) -> dict[str, set[str]]:
    """Map test file path -> set of modules it imports."""
    index: dict[str, set[str]] = {}
    for f in all_files:
        if not _is_test_file(f):
            continue
        src = read_file(repo, f)
        if src is None:
            continue
        index[f] = _imported_modules(src, importing_rel_path=f)
    return index


def find_tests_importing(
    rel_path: str, test_import_index: dict[str, set[str]]
) -> list[str]:
    module = _rel_path_to_module(rel_path)
    if not module:
        return []
    hits: list[str] = []
    for test_path, imports in test_import_index.items():
        if test_path == rel_path:
            continue
        if module in imports or any(
            imp == module or imp.startswith(module + ".") for imp in imports
        ):
            hits.append(test_path)
    return hits


GENERIC_STEMS = {
    "strategy", "config", "utils", "helpers", "main", "base", "common",
    "core", "client", "server", "api", "models", "types", "constants",
    "interface", "interfaces", "manager", "handler", "service",
}


def extract_module_docstring(py_source: str) -> str:
    try:
        tree = ast.parse(py_source)
    except SyntaxError:
        return ""
    return (ast.get_docstring(tree) or "").strip()


def find_sibling_docstrings(
    repo: Path, touched: list[str]
) -> dict[str, list[tuple[str, str]]]:
    touched_set = set(touched)
    out: dict[str, list[tuple[str, str]]] = {}
    dirs = {Path(p).parent.as_posix() for p in touched if p.endswith(".py")}
    for d in dirs:
        dir_path = repo / d if d and d != "." else repo
        if not dir_path.is_dir():
            continue
        siblings: list[tuple[str, str]] = []
        for f in sorted(dir_path.glob("*.py")):
            rel = f.relative_to(repo).as_posix()
            if rel in touched_set or f.name == "__init__.py":
                continue
            src = read_file(repo, rel)
            if src is None:
                continue
            doc = extract_module_docstring(src)
            if doc:
                first_line = doc.splitlines()[0][:200]
                siblings.append((rel, first_line))
        if siblings:
            out[d] = siblings
    return out


SKIP_MARKDOWN = {"README.md", "CLAUDE.md", "CHANGELOG.md", "LICENSE.md", "CONTRIBUTING.md"}


def find_design_docs(
    repo: Path, touched: list[str], all_files: list[str]
) -> list[str]:
    """Match docs that reference touched files by full path or dotted module.

    Bare stems like 'config' or 'handler' are too common in prose to be a
    signal. We require either the file path, the dotted module path, or the
    immediate parent directory path to appear.
    """
    targets: set[str] = set()
    for t in touched:
        targets.add(t)
        parent = Path(t).parent.as_posix()
        if parent and parent != ".":
            targets.add(parent)
        if t.endswith(".py"):
            mod = _rel_path_to_module(t)
            if mod and "." in mod:
                targets.add(mod)
                parent_mod = mod.rsplit(".", 1)[0]
                if "." in parent_mod:
                    targets.add(parent_mod)
    if not targets:
        return []
    patterns = [re.compile(re.escape(t)) for t in targets]
    hits: list[str] = []
    for f in all_files:
        if not f.endswith(".md") or Path(f).name in SKIP_MARKDOWN:
            continue
        content = read_file(repo, f)
        if content is None:
            continue
        if any(p.search(content) for p in patterns):
            hits.append(f)
    return hits


CONFIG_PATH_PATTERN = re.compile(
    r'["\']((?:configs?|conf|settings)/[a-zA-Z0-9_/\-.]+\.(?:ya?ml|toml|json|ini))["\']'
    r'|["\']([a-zA-Z0-9_/\-.]+\.(?:ya?ml|toml|ini))["\']'
)


def find_referenced_configs(repo: Path, touched: list[str]) -> list[str]:
    referenced: set[str] = set()
    touched_set = set(touched)
    for path in touched:
        content = read_file(repo, path)
        if content is None:
            continue
        for match in CONFIG_PATH_PATTERN.finditer(content):
            candidate = match.group(1) or match.group(2)
            if candidate and candidate not in touched_set and (repo / candidate).exists():
                referenced.add(candidate)
    return sorted(referenced)


def find_tests_for(repo: Path, rel_path: str, all_files: list[str]) -> list[str]:
    stem = Path(rel_path).stem
    if not stem or stem.startswith("test_") or stem.endswith("_test") or stem == "__init__":
        return []
    # For generic stems, require a second signal: a parent dir name in the test filename.
    parent_parts = {p for p in Path(rel_path).parent.parts if p not in {".", ""}}
    require_parent_match = stem in GENERIC_STEMS
    hits: list[str] = []
    for f in all_files:
        if f == rel_path or not f.endswith(".py"):
            continue
        name = Path(f).stem
        if not (name.startswith("test_") or name.endswith("_test")):
            continue
        name_parts = set(name.split("_"))
        stem_parts = set(stem.split("_"))
        # Match if whole stem appears OR (for compound stems) all sub-parts appear.
        if stem not in name_parts and not (len(stem_parts) > 1 and stem_parts <= name_parts):
            continue
        if require_parent_match and not (parent_parts & name_parts):
            continue
        hits.append(f)
    return hits


def build_scout_payload(repo_path: str, base_ref: str, head_ref: str) -> str:
    """Lightweight payload for the scouting pass. Skips full touched-file
    contents but includes tree, project docs, narrative, and touched-file list
    with module docstrings so the critic can reason about what else to read."""
    repo = Path(repo_path).resolve()
    narrative = get_commit_narrative(repo, base_ref, head_ref)
    diff = get_diff(repo, base_ref, head_ref) if not narrative else ""
    touched = get_touched_files(repo, base_ref, head_ref)
    all_files = [l for l in _run(["git", "ls-files"], repo).splitlines() if l.strip()]

    touched_summary_lines: list[str] = []
    for path in touched:
        src = read_file(repo, path)
        doc = extract_module_docstring(src or "") if path.endswith(".py") else ""
        first = doc.splitlines()[0][:200] if doc else ""
        touched_summary_lines.append(f"- {path}" + (f"  ({first})" if first else ""))

    parts = [
        f"=== REPO: {repo.name} ===",
        f"=== TREE ===\n" + "\n".join(all_files),
    ]
    for doc in ("CLAUDE.md", "README.md"):
        if (content := read_file(repo, doc)):
            parts.append(f"=== {doc} ===\n{content}")
    if narrative:
        parts.append(
            f"=== COMMIT NARRATIVE ({base_ref}..{head_ref}, oldest first) ===\n{narrative}"
        )
    else:
        parts.append(f"=== DIFF ({base_ref}...{head_ref}) ===\n{diff}")
    parts.append("=== TOUCHED FILES ===\n" + "\n".join(touched_summary_lines))
    return "\n\n".join(parts)


def build_payload(
    repo_path: str,
    base_ref: str,
    head_ref: str,
    extra_files: list[dict],
    token_budget: int,
) -> str:
    repo = Path(repo_path).resolve()
    diff = get_diff(repo, base_ref, head_ref)
    narrative = get_commit_narrative(repo, base_ref, head_ref)
    touched = get_touched_files(repo, base_ref, head_ref)
    all_files = [l for l in _run(["git", "ls-files"], repo).splitlines() if l.strip()]

    # (priority, label, path, content). priority 0 = must keep.
    sections: list[tuple[int, str, str, str]] = []
    included: set[str] = set()

    def add(priority: int, label: str, path: str, content: str | None) -> None:
        if path in included or content is None:
            return
        included.add(path)
        sections.append((priority, label, path, content))

    for path in touched:
        content = read_file(repo, path)
        if content is None:
            continue
        history = file_history(repo, path)
        header = f"[recent commits]\n{history}\n\n" if history else ""
        add(0, "TOUCHED", path, header + content)

    test_import_index = build_test_import_index(repo, all_files)
    max_tests_per_touched = 8
    for path in touched:
        filename_hits = find_tests_for(repo, path, all_files)
        import_hits = [t for t in find_tests_importing(path, test_import_index) if t not in filename_hits]
        ordered = filename_hits + import_hits
        for test_path in ordered[:max_tests_per_touched]:
            add(0, f"TEST FOR {path}", test_path, read_file(repo, test_path))

    for path in touched:
        if not path.endswith(".py"):
            continue
        src = read_file(repo, path)
        if src is None:
            continue
        for imp_path in parse_imports(src, repo, importing_rel_path=path):
            add(1, "IMPORT OF TOUCHED", imp_path, read_file(repo, imp_path))

    for sym in extract_changed_symbols(diff):
        for ref_path in grep_refs(repo, sym):
            if ref_path in touched:
                continue
            add(2, f"REFERENCES `{sym}`", ref_path, read_file(repo, ref_path))

    for cfg_path in find_referenced_configs(repo, touched):
        add(0, "CONFIG REFERENCED BY TOUCHED", cfg_path, read_file(repo, cfg_path))

    for doc_path in find_design_docs(repo, touched, all_files):
        add(1, "DESIGN DOC (mentions touched modules)", doc_path, read_file(repo, doc_path))

    sibling_map = find_sibling_docstrings(repo, touched)
    for d, siblings in sibling_map.items():
        block = "\n".join(f"- {path}: {doc}" for path, doc in siblings)
        sentinel = f"__sibling_docstrings__::{d or '.'}"
        add(0, f"SIBLINGS IN {d or '.'}", sentinel, block)

    for entry in extra_files:
        path = entry["path"]
        reason = entry.get("reason", "unspecified")
        add(0, f"CLAUDE-FLAGGED (reason: {reason})", path, read_file(repo, path))

    # Essential header: repo name + the thing being reviewed. Never dropped.
    essential_parts = [f"=== REPO: {repo.name} ==="]
    if narrative:
        essential_parts.append(
            f"=== COMMIT NARRATIVE ({base_ref}..{head_ref}, oldest first) ===\n{narrative}"
        )
    else:
        essential_parts.append(f"=== DIFF ({base_ref}...{head_ref}) ===\n{diff}")

    # Optional header items, priority-ordered. CLAUDE.md before README before tree.
    optional_header: list[tuple[str, str]] = []
    for doc in ("CLAUDE.md", "README.md"):
        if (content := read_file(repo, doc)):
            optional_header.append((doc, f"=== {doc} ===\n{content}"))
    optional_header.append(("TREE", f"=== TREE ===\n" + "\n".join(all_files)))

    used = sum(count_tokens(p) for p in essential_parts)
    dropped: list[tuple[str, str, int]] = []
    header_parts = list(essential_parts)

    for label, block in optional_header:
        cost = count_tokens(block)
        if used + cost <= token_budget:
            header_parts.append(block)
            used += cost
        else:
            dropped.append(("HEADER", label, cost))

    sections.sort(key=lambda s: s[0])
    body_parts: list[str] = []
    for prio, label, path, content in sections:
        block = f"=== {label}: {path} ===\n{content}"
        cost = count_tokens(block)
        if used + cost <= token_budget:
            body_parts.append(block)
            used += cost
        else:
            dropped.append((label, path, cost))

    if dropped:
        note = "=== DROPPED (over budget) ===\n" + "\n".join(
            f"- [{label}] {path} ({cost} tokens)" for label, path, cost in dropped
        )
        body_parts.append(note)

    header = "\n\n".join(header_parts)
    return header + "\n\n" + "\n\n".join(body_parts)
