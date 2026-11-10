"""File tools confined to one directory.

Every path the model gives is resolved (following symlinks) and must end up
inside ``root``; anything else is refused. Reading tools are read-only and
may run in parallel. Writing tools exist only with ``writable=True`` and ask
for approval unless a permission rule allows them.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from ..errors import ToolError
from ..tools import Tool, tool

_SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv"}


def file_tools(
    root: str | Path,
    *,
    writable: bool = False,
    max_read_chars: int = 100_000,
    max_results: int = 200,
) -> list[Tool]:
    base = Path(root).resolve()
    if not base.is_dir():
        raise ValueError(f"{base} is not a directory")

    def locate(path: str) -> Path:
        candidate = (base / path).resolve()
        if candidate != base and not candidate.is_relative_to(base):
            raise ToolError(f"{path!r} is outside the workspace; use a path relative to its root.")
        return candidate

    def shown(path: Path) -> str:
        return path.relative_to(base).as_posix() or "."

    def walk(start: Path):
        for entry in sorted(start.iterdir()):
            if entry.is_symlink():
                continue  # a link may point outside the workspace
            if entry.is_dir():
                if entry.name not in _SKIP_DIRS:
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
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise ToolError(f"{path!r} is not a UTF-8 text file.") from error
        first = max(start_line, 1)
        chosen = lines[first - 1 : first - 1 + max(max_lines, 1)]
        text = "\n".join(f"{number}\t{line}" for number, line in enumerate(chosen, start=first))
        if len(text) > max_read_chars:
            note = f"[cut at {max_read_chars} characters; read a smaller range]"
            text = f"{text[:max_read_chars]}\n{note}"
        remaining = len(lines) - (first - 1 + len(chosen))
        if remaining > 0:
            text += f"\n[{remaining} more lines; continue with start_line={first + len(chosen)}]"
        return text or "(the file is empty)"

    @tool(read_only=True)
    def search_files(pattern: str, path: str = ".", glob: str = "*") -> str:
        """Search file contents in the workspace with a regular expression.

        Args:
            pattern: Regular expression to look for.
            path: Directory to search, relative to the workspace root.
            glob: Only search files whose name matches this shell-style pattern.
        """
        try:
            expression = re.compile(pattern)
        except re.error as error:
            raise ToolError(f"Invalid regular expression: {error}") from error
        start = locate(path)
        if not start.is_dir():
            raise ToolError(f"{path!r} is not a directory.")
        hits: list[str] = []
        for entry in walk(start):
            if not fnmatch.fnmatch(entry.name, glob):
                continue
            try:
                content = entry.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(content.splitlines(), start=1):
                if expression.search(line):
                    hits.append(f"{shown(entry)}:{number}: {line.strip()[:300]}")
                    if len(hits) >= max_results:
                        return "\n".join(hits) + f"\n[stopped at {max_results} matches]"
        return "\n".join(hits) if hits else "No matches."

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
