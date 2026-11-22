"""The command line of the docs server, for coding agents without MCP.

    python -m qagents.docs_mcp                       serve MCP over stdio (what MCP clients run)
    python -m qagents.docs_mcp search "<query>"      ranked search: docs, examples, API
    python -m qagents.docs_mcp read <page> [section] a page or one section
    python -m qagents.docs_mcp pages                 the table of contents
    python -m qagents.docs_mcp api <symbol>          signature, docstring, members, source
    python -m qagents.docs_mcp examples              the examples gallery
    python -m qagents.docs_mcp example <name>        one example's code
    python -m qagents.docs_mcp check <file.py | ->   static check against the installed qagents
    python -m qagents.docs_mcp llms [--full]         llms.txt (or llms-full.txt)
    python -m qagents.docs_mcp skill [--to DIR]      the Claude Code skill (SKILL.md)

Each command prints the same Markdown the MCP tool returns. ``skill`` prints the skill file
of the installed version as it is, and ``skill --to .claude/skills/qagents`` writes it there
as UTF-8 (shell redirection on Windows PowerShell 5.1 would write UTF-16). Exit codes: 0 on
success, 1 when something was not found, ``check`` found problems or ``skill --to`` would
replace a different file, 2 for a mistake in the command line.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from .sources import DocsSource

COMMANDS = (
    "serve", "search", "read", "pages", "api", "examples", "example", "check", "llms", "skill",
)  # fmt: skip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m qagents.docs_mcp",
        description=(
            "The qagents documentation for coding agents: an MCP server over stdio (no "
            "command) or the same answers on the command line."
        ),
    )
    parser.add_argument(
        "--docs-dir",
        default=None,
        metavar="DIR",
        help="read the docs from this qagents checkout or docs folder",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    commands.add_parser("serve", help="serve MCP over stdio (the default)")

    search = commands.add_parser("search", help="search the docs, examples and API")
    search.add_argument("query", nargs="+", help="words, a question or a qagents name")
    search.add_argument("--limit", type=int, default=8, help="how many results (default 8)")
    search.add_argument(
        "--kind", choices=["docs", "example", "api"], default=None, help="only one kind"
    )

    read = commands.add_parser("read", help="read a page or one section of it")
    read.add_argument("path", help="for example concepts/permissions.md")
    read.add_argument("section", nargs="?", default=None, help="an anchor or a heading")

    commands.add_parser("pages", help="the table of contents")

    api = commands.add_parser("api", help="signature, docstring and members of a name")
    api.add_argument("symbol", help="for example Agent.run or qagents.toolkits.files")

    commands.add_parser("examples", help="the examples gallery")

    example = commands.add_parser("example", help="the code of one example")
    example.add_argument("name", help="for example supervisor")

    check = commands.add_parser("check", help="check Python code against the installed qagents")
    check.add_argument("files", nargs="+", help="Python files, or - for stdin")

    llms = commands.add_parser("llms", help="print llms.txt")
    llms.add_argument("--full", action="store_true", help="print llms-full.txt instead")

    skill = commands.add_parser(
        "skill", help="print the Claude Code skill (SKILL.md), or write it with --to"
    )
    skill.add_argument(
        "--to",
        default=None,
        metavar="DIR",
        help="write SKILL.md into this folder, for example .claude/skills/qagents",
    )
    skill.add_argument(
        "--force", action="store_true", help="replace a SKILL.md there that is different"
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Run the docs command line; returns the exit code."""
    stdin = stdin if stdin is not None else sys.stdin
    out = stdout if stdout is not None else _tolerant(sys.stdout)
    err = stderr if stderr is not None else _tolerant(sys.stderr)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else 2

    from .service import DocsService
    from .sources import find_source

    source = find_source(args.docs_dir)
    command = args.command or "serve"
    if command == "skill":
        return _skill(source, args.to, force=args.force, out=out, err=err)
    if command == "serve":
        from .server import serve_stdio

        serve_stdio(DocsService(source=source))
        return 0

    service = DocsService(source=source)
    if command == "search":
        text = service.search(" ".join(args.query), args.limit, args.kind)
        return _emit(out, shell_hints(text), failed=text.startswith("No results"))
    if command == "read":
        text = service.read_page(args.path, args.section)
        failed = text.startswith(("There is no page", "Page "))
        return _emit(out, shell_hints(text) if failed else text, failed=failed)
    if command == "pages":
        return _emit(out, shell_hints(service.list_pages()))
    if command == "api":
        text = service.get_api(args.symbol)
        return _emit(out, shell_hints(text), failed=not text.startswith("# "))
    if command == "examples":
        return _emit(out, shell_hints(service.list_examples()))
    if command == "example":
        text = service.read_example(args.name)
        failed = text.startswith("There is no example")
        return _emit(out, shell_hints(text) if failed else text, failed=failed)
    if command == "check":
        failed = False
        for name in args.files:
            if name == "-":
                code, label = stdin.read(), "<stdin>"
            else:
                try:
                    code, label = Path(name).read_text(encoding="utf-8-sig"), name
                except (OSError, UnicodeDecodeError) as error:
                    err.write(f"check: cannot read {name}: {error}\n")
                    failed = True
                    continue
            report = service.check(code)
            if len(args.files) > 1:
                out.write(f"<!-- {label} -->\n")
            out.write(report.render(service.version))
            failed = failed or bool(report.errors)
        return 1 if failed else 0
    if command == "llms":
        if args.full:
            from .llms import build_llms_full

            text = build_llms_full(service.corpus, render_api=service._render_directive)
        else:
            text = service.llms_txt()
        return _emit(out, text)
    parser.error(f"unknown command {command!r}")  # pragma: no cover - argparse rejects it
    return 2


def _skill(source: DocsSource, to: str | None, *, force: bool, out: IO[str], err: IO[str]) -> int:
    """Print the skill file of the installed version, or write it into the folder ``to``."""
    from . import prompts
    from .sources import SKILL_URL

    file = source.skill_md
    if file is None:
        err.write(prompts.SKILL_MISSING.format(url=SKILL_URL) + "\n")
        return 1
    try:
        text = file.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        err.write(f"skill: cannot read {file}: {error}\n")
        return 1
    if to is None:
        return _emit(out, text)
    target = Path(to).expanduser()
    if target.name.lower() != "skill.md":
        target = target / "SKILL.md"
    version = source.version
    try:
        if target.is_file():
            if target.read_text(encoding="utf-8-sig", errors="replace") == text:
                return _emit(out, prompts.SKILL_UNCHANGED.format(path=target, version=version))
            if not force:
                err.write(prompts.SKILL_EXISTS.format(path=target) + "\n")
                return 1
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    except OSError as error:
        err.write(f"skill: cannot write {target}: {error}\n")
        return 1
    return _emit(out, prompts.SKILL_WRITTEN.format(path=target, version=version))


_COMMAND = "python -m qagents.docs_mcp"
_HINTS = [
    (re.compile(r'read_page\("([^"]+)", section="([^"]+)"\)'), _COMMAND + r" read \1 \2"),
    (re.compile(r'read_page\("([^"]+)"\)'), _COMMAND + r" read \1"),
    (re.compile(r"read_page\(path\) or read_page\(path, section=\.\.\.\)"),
     _COMMAND + " read <path> [section]"),
    (re.compile(r'read_example\("([^"]+)"\)'), _COMMAND + r" example \1"),
    (re.compile(r"read_example\(name\)"), _COMMAND + " example <name>"),
    (re.compile(r'get_api\("([^"]+)"\)'), _COMMAND + r" api \1"),
    (re.compile(r"list_examples\(\)"), _COMMAND + " examples"),
    (re.compile(r"list_pages\(\)"), _COMMAND + " pages"),
    (re.compile(r"search_docs\(\)"), _COMMAND + " search"),
]  # fmt: skip


def shell_hints(text: str) -> str:
    """The answer with its hints (``read_page("x.md")``) written as shell commands.

    Used for the server's own lines (results, lists, not-found messages), never for the
    content of a page or an example.
    """
    for pattern, replacement in _HINTS:
        text = pattern.sub(replacement, text)
    return text


def _emit(out: IO[str], text: str, *, failed: bool = False) -> int:
    out.write(text if text.endswith("\n") else text + "\n")
    out.flush()
    return 1 if failed else 0


def _tolerant(stream: IO[str]) -> IO[str]:
    """The stream, set to replace characters its encoding cannot show instead of failing."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):  # pragma: no cover - a closed or odd stream
            pass
    return stream
