"""File tools confined to one directory.

Every path the model gives is resolved (following symlinks and Windows
junctions) and must end up inside ``root``; anything else is refused. Listings
and searches skip symlinks and junctions, and never enter or read anything that
resolves outside ``root``. Reading tools are read-only and may run in parallel.
Writing tools exist only with ``writable=True`` and ask for approval unless a
permission rule allows them.

Reads are bounded: ``read_file`` streams the file and stops after the lines it
returns, and ``search_files`` skips files larger than ``max_file_bytes``.
``search_files`` runs the model's regular expression in a separate Python
process (``sys.executable -I``) that is stopped after ``search_timeout``
seconds: Python's ``re`` cannot be interrupted, so a pattern with catastrophic
backtracking would otherwise freeze every run in the agent's process. Each
search therefore starts an interpreter, which takes a few tens of milliseconds.
In a frozen application (PyInstaller, cx_Freeze) or an embedded interpreter,
``sys.executable`` is not a Python interpreter, so ``search_files`` refuses to
run unless ``search_python`` names one.
"""

from __future__ import annotations

import fnmatch
import json
import re
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from ..errors import ToolError
from ..tools import Tool, tool

_SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv"}
_SEARCH_LINE_CHARS = 10_000
"""``search_files`` matches the first this many characters of each line."""

# Runs in a separate interpreter (``python -I -c``), so it imports nothing from qagents.
# ``lines`` is the same function as `_lines` below.
_SEARCH_PROGRAM = r"""
import json, re, sys

def lines(handle, limit):
    while piece := handle.readline(limit):
        if len(piece) == limit and not piece.endswith("\n"):
            while (rest := handle.readline(limit)) and not rest.endswith("\n"):
                pass
        yield from piece.splitlines()

job = json.loads(sys.stdin.buffer.read())
expression = re.compile(job["pattern"])
hits = []
for shown, path in job["files"]:
    found = []
    try:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(lines(handle, job["line_chars"]), start=1):
                if expression.search(line):
                    found.append(f"{shown}:{number}: {line.strip()[:300]}")
                    if len(hits) + len(found) >= job["max_results"]:
                        break
    except (UnicodeDecodeError, OSError):
        continue
    hits.extend(found)
    if len(hits) >= job["max_results"]:
        break
sys.stdout.buffer.write(json.dumps(hits).encode("ascii"))
"""


def file_tools(
    root: str | Path,
    *,
    writable: bool = False,
    max_read_chars: int = 100_000,
    max_results: int = 200,
    max_file_bytes: int = 10_000_000,
    search_timeout: float = 30.0,
    search_python: str | Path | None = None,
) -> list[Tool]:
    """Tools that list, read, search and (with ``writable=True``) change files under ``root``.

    Args:
        root: The workspace directory. The model's paths are relative to it.
        writable: Also give ``write_file`` and ``edit_file``, which ask for approval.
        max_read_chars: The most characters ``read_file`` returns, and the longest
            line it reads.
        max_results: The most entries ``list_files`` and ``search_files`` return.
        max_file_bytes: ``search_files`` skips larger files and says how many it skipped.
        search_timeout: Seconds before a ``search_files`` call is stopped.
        search_python: The Python interpreter ``search_files`` runs its search in.
            Default: ``sys.executable``, unless the application is frozen or
            ``sys.executable`` is not named like a Python interpreter; then
            ``search_files`` refuses to run.
    """
    base = Path(root).resolve()
    if not base.is_dir():
        raise ValueError(f"{base} is not a directory")
    python = str(search_python) if search_python is not None else _default_python()

    def inside(path: Path) -> bool:
        return path == base or path.is_relative_to(base)

    def locate(path: str) -> Path:
        candidate = (base / path).resolve()
        if not inside(candidate):
            raise ToolError(f"{path!r} is outside the workspace; use a path relative to its root.")
        return candidate

    def shown(path: Path) -> str:
        return path.relative_to(base).as_posix() or "."

    def walk(start: Path) -> Iterator[Path]:
        for entry in sorted(start.iterdir()):
            if _is_link(entry):
                continue  # a symlink or junction may point outside the workspace
            if entry.is_dir():
                # resolve() catches any other kind of link that leads outside.
                if entry.name not in _SKIP_DIRS and inside(entry.resolve()):
                    yield from walk(entry)
            else:
                yield entry

    @tool(read_only=True)
    def list_files(path: str = ".", pattern: str = "*") -> str:
        """List files under a directory of the workspace, recursively.

        Args:
            path: Directory, relative to the workspace root.
            pattern: Shell-style pattern for the file's name or relative path, for example "*.py".
        """
        start = locate(path)
        if not start.is_dir():
            raise ToolError(f"{path!r} is not a directory.")
        matches = [
            shown(entry)
            for entry in walk(start)
            if fnmatch.fnmatch(shown(entry), pattern) or fnmatch.fnmatch(entry.name, pattern)
        ]
        if not matches:
            return "No files match."
        listing = "\n".join(matches[:max_results])
        extra = len(matches) - max_results
        return listing if extra <= 0 else f"{listing}\n... and {extra} more"

    @tool(read_only=True)
    def read_file(path: str, start_line: int = 1, max_lines: int = 2000) -> str:
        """Read a text file from the workspace, with line numbers.

        Args:
            path: File path, relative to the workspace root.
            start_line: First line to return (1-based).
            max_lines: How many lines to return at most.
        """
        target = locate(path)
        if not target.is_file():
            raise ToolError(f"{path!r} is not a file. Use list_files to see what exists.")
        first = max(start_line, 1)
        wanted = max(max_lines, 1)
        chosen: list[str] = []
        size = 0
        more = False
        try:
            with target.open(encoding="utf-8") as handle:
                for number, line in enumerate(_lines(handle, max(max_read_chars, 1)), start=1):
                    if number < first:
                        continue
                    if len(chosen) == wanted or size > max_read_chars:
                        more = True
                        break
                    chosen.append(f"{number}\t{line}")
                    size += len(chosen[-1]) + 1
        except UnicodeDecodeError as error:
            raise ToolError(f"{path!r} is not a UTF-8 text file.") from error
        text = "\n".join(chosen)
        if len(text) > max_read_chars:
            note = f"[cut at {max_read_chars} characters; read a smaller range]"
            text = f"{text[:max_read_chars]}\n{note}"
        if more:
            text += f"\n[more lines follow; continue with start_line={first + len(chosen)}]"
        return text or "(the file is empty)"

    @tool(read_only=True)
    def search_files(pattern: str, path: str = ".", glob: str = "*") -> str:
        """Search file contents in the workspace with a regular expression.

        Args:
            pattern: Regular expression to look for.
            path: Directory to search, relative to the workspace root.
            glob: Only search files whose name matches this shell-style pattern.
        """
        if python is None:
            raise ToolError(
                "search_files cannot run here: it has no Python interpreter to search in. "
                "Use list_files and read_file instead."
            )
        try:
            re.compile(pattern)
        except re.error as error:
            raise ToolError(f"Invalid regular expression: {error}") from error
        start = locate(path)
        if not start.is_dir():
            raise ToolError(f"{path!r} is not a directory.")
        files: list[tuple[str, str]] = []
        too_large = 0
        for entry in walk(start):
            if not fnmatch.fnmatch(entry.name, glob):
                continue
            try:
                real = entry.resolve()
                if not inside(real) or not real.is_file():
                    continue
                if real.stat().st_size > max_file_bytes:
                    too_large += 1
                    continue
            except OSError:
                continue
            files.append((shown(entry), str(real)))
        hits = _search(python, pattern, files, max_results, search_timeout) if files else []
        text = "\n".join(hits) if hits else "No matches."
        if len(hits) >= max_results:
            text += f"\n[stopped at {max_results} matches]"
        if too_large:
            text += f"\n[{too_large} files larger than {max_file_bytes} bytes were not searched]"
        return text

    tools = [list_files, read_file, search_files]
    if not writable:
        return tools

    @tool(requires_approval=True, idempotent=True)
    def write_file(path: str, content: str) -> str:
        """Create a file in the workspace, or replace its whole content.

        Args:
            path: File path, relative to the workspace root.
            content: The complete new content of the file.
        """
        target = locate(path)
        if target.is_dir():
            raise ToolError(f"{path!r} is a directory.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {shown(target)}."

    @tool(requires_approval=True)
    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Replace one exact piece of text in a workspace file.

        Args:
            path: File path, relative to the workspace root.
            old_text: Text to replace. It must appear exactly once in the file.
            new_text: Text to put in its place.
        """
        target = locate(path)
        if not target.is_file():
            raise ToolError(f"{path!r} is not a file.")
        content = target.read_text(encoding="utf-8")
        occurrences = content.count(old_text)
        if occurrences == 0:
            raise ToolError("old_text was not found. Read the file and copy the text exactly.")
        if occurrences > 1:
            raise ToolError(
                f"old_text appears {occurrences} times. "
                "Include more surrounding text so it is unique."
            )
        target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {shown(target)}."

    return [*tools, write_file, edit_file]


def _is_link(path: Path) -> bool:
    """A symlink or a Windows junction (``Path.is_junction`` needs Python 3.12).

    A path that does not exist is not a link; one that cannot be inspected counts as one.
    """
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    junction = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", None)  # defined on Windows only
    return stat.S_ISLNK(info.st_mode) or (
        junction is not None and getattr(info, "st_reparse_tag", 0) == junction
    )


def _lines(handle: TextIO, limit: int) -> Iterator[str]:
    """The lines of a text file, split like ``str.splitlines``, in bounded memory.

    A line longer than ``limit`` characters is cut there; the rest of it is skipped.
    """
    while piece := handle.readline(limit):
        if len(piece) == limit and not piece.endswith("\n"):
            while (rest := handle.readline(limit)) and not rest.endswith("\n"):
                pass
        yield from piece.splitlines()


def _default_python() -> str | None:
    """``sys.executable`` when it is a Python interpreter, else ``None``.

    In a frozen or embedded application it is the host program, and starting
    it would start another copy of the application.
    """
    if getattr(sys, "frozen", False) or not sys.executable:
        return None
    name = Path(sys.executable).name.lower()
    return sys.executable if name.startswith(("python", "pypy", "graalpy")) else None


def _search(
    python: str, pattern: str, files: list[tuple[str, str]], max_results: int, timeout: float
) -> list[str]:
    """Run the search in a child interpreter that is killed after ``timeout`` seconds."""
    job = {
        "pattern": pattern,
        "files": files,
        "max_results": max_results,
        "line_chars": _SEARCH_LINE_CHARS,
    }
    try:
        done = subprocess.run(
            [python, "-I", "-c", _SEARCH_PROGRAM],
            input=json.dumps(job).encode("ascii"),
            capture_output=True,
            timeout=timeout,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        raise ToolError(
            f"The search took longer than {timeout:g} seconds and was stopped. Use a simpler "
            "pattern, or search fewer files with path= or glob=."
        ) from None
    except OSError as error:
        raise ToolError(f"The search could not start: {error}") from error
    if done.returncode != 0:
        reason = done.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise ToolError(f"The search failed: {reason[-1][:300] if reason else 'no reason given'}")
    return json.loads(done.stdout)
