"""Memory as plain files.

Each note is a Markdown file in one directory. The agent decides what is
worth writing down and reads it back in a later run, which keeps memory
inspectable: you can open the folder and see exactly what the agent knows.

Only regular files directly in that directory are notes. A note that is a
symlink or a Windows junction is refused, so the tools never read or write
outside the directory through one. Saving writes a new file and moves it into
place, which replaces a link or a hard link instead of writing through it.
On Linux and macOS the new file is readable by its owner only (mode 0600,
whatever the umask), also when it replaces an existing note, so another
account (a person reviewing what a service agent wrote) needs to read the
notes as that owner or through an ACL.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
from pathlib import Path

from ..errors import ToolError
from ..tools import Tool, tool
from .files import _is_link

_NOTE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
_SUMMARY_CHARS = 4096
"""``list_notes`` reads this much of each note to find its first line."""


def notes_tools(directory: str | Path, *, max_note_chars: int = 20_000) -> list[Tool]:
    """Tools that list, read, save and delete Markdown notes in ``directory``.

    Args:
        directory: The notes folder; it is created when missing.
        max_note_chars: The longest note ``save_note`` accepts and ``read_note`` returns.
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    folder = folder.resolve()

    def note_path(name: str) -> Path:
        if not _NOTE_NAME.match(name):
            raise ToolError(
                "A note name is 1-80 letters, digits, '-' or '_', "
                "for example 'customer-preferences'."
            )
        path = folder / f"{name}.md"
        if _is_link(path) or path.resolve().parent != folder:
            raise ToolError(f"The note {name!r} is a link to another place, so it is not used.")
        return path

    def read(path: Path, limit: int) -> str:
        with open(path, encoding="utf-8", errors="replace", opener=_open_no_follow) as handle:
            return handle.read(limit)

    @tool(read_only=True)
    def list_notes() -> str:
        """List the notes saved in earlier work, with the first line of each."""
        entries = []
        for path in sorted(folder.glob("*.md")):
            if _is_link(path) or not path.is_file():
                continue
            try:
                lines = read(path, _SUMMARY_CHARS).strip().splitlines()
            except OSError:
                continue
            entries.append(f"{path.stem}: {lines[0][:120] if lines else '(empty)'}")
        return "\n".join(entries) if entries else "There are no notes yet."

    @tool(read_only=True)
    def read_note(name: str) -> str:
        """Read one saved note.

        Args:
            name: The note's name, as shown by list_notes.
        """
        path = note_path(name)
        if not path.is_file():
            raise ToolError(f"There is no note named {name!r}. Use list_notes to see the names.")
        content = read(path, max_note_chars + 1)
        if len(content) > max_note_chars:
            content = f"{content[:max_note_chars]}\n[cut at {max_note_chars} characters]"
        return content

    @tool(idempotent=True)
    def save_note(name: str, content: str) -> str:
        """Save a note for future work, replacing a note with the same name.

        Args:
            name: Short name made of letters, digits, '-' or '_'.
            content: The note, in Markdown. Start with a one-line summary.
        """
        if len(content) > max_note_chars:
            raise ToolError(f"A note can hold at most {max_note_chars} characters; shorten it.")
        path = note_path(name)
        handle, temporary = tempfile.mkstemp(dir=folder, prefix=f".{name}.", suffix=".tmp")
        try:
            with open(handle, "w", encoding="utf-8") as file:
                file.write(content)
            os.replace(temporary, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
        return f"Saved note {name!r}."

    @tool(idempotent=True)
    def delete_note(name: str) -> str:
        """Delete a saved note that is wrong or no longer useful.

        Args:
            name: The note's name, as shown by list_notes.
        """
        path = note_path(name)
        if not path.is_file():
            raise ToolError(f"There is no note named {name!r}.")
        path.unlink()
        return f"Deleted note {name!r}."

    return [list_notes, read_note, save_note, delete_note]


def _open_no_follow(path: str, flags: int) -> int:
    """``open`` that refuses a symlink where the platform can (POSIX ``O_NOFOLLOW``)."""
    return os.open(path, flags | getattr(os, "O_NOFOLLOW", 0))
