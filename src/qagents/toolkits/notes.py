"""Memory as plain files.

Each note is a Markdown file in one directory. The agent decides what is
worth writing down and reads it back in a later run, which keeps memory
inspectable: you can open the folder and see exactly what the agent knows.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..errors import ToolError
from ..tools import Tool, tool

_NOTE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def notes_tools(directory: str | Path, *, max_note_chars: int = 20_000) -> list[Tool]:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)

    def note_path(name: str) -> Path:
        if not _NOTE_NAME.match(name):
            raise ToolError(
                "A note name is 1-80 letters, digits, '-' or '_', "
                "for example 'customer-preferences'."
            )
        return folder / f"{name}.md"

    @tool(read_only=True)
    def list_notes() -> str:
        """List the notes saved in earlier work, with the first line of each."""
        entries = []
        for path in sorted(folder.glob("*.md")):
            lines = path.read_text(encoding="utf-8").strip().splitlines()
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
        return path.read_text(encoding="utf-8")

    @tool(idempotent=True)
    def save_note(name: str, content: str) -> str:
        """Save a note for future work, replacing a note with the same name.

        Args:
            name: Short name made of letters, digits, '-' or '_'.
            content: The note, in Markdown. Start with a one-line summary.
        """
        if len(content) > max_note_chars:
            raise ToolError(f"A note can hold at most {max_note_chars} characters; shorten it.")
        note_path(name).write_text(content, encoding="utf-8")
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
