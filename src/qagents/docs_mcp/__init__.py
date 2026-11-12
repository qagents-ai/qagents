"""The qagents documentation for coding agents, over MCP or the command line.

Coding agents (Claude Code, Codex CLI, Cursor, VS Code, any MCP client) can query the
documentation of the qagents version that is installed, so what they read matches the code
they write against:

    python -m qagents.docs_mcp                       # MCP over stdio (needs qagents[mcp])
    python -m qagents.docs_mcp search "approval"     # the same answers, on the command line
    python -m qagents.docs_mcp api Agent.resume
    python -m qagents.docs_mcp check my_agent.py
    python -m qagents.docs_mcp skill --to .claude/skills/qagents   # the Claude Code skill

The MCP tools are ``search_docs``, ``read_page``, ``list_pages``, ``get_api``,
``list_examples``, ``read_example`` and ``check_code``; see `qagents.docs_mcp.server`. The
answers come from `DocsService`, which needs no MCP SDK and is what the command line uses.

Where the docs are read from: the copy bundled in the wheel, else the repository checkout of
an editable install, else ``QAGENTS_DOCS_DIR`` (see `qagents.docs_mcp.sources`). The API
reference always comes from the installed package itself. Every string the server shows a
model is in `qagents.docs_mcp.prompts`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .service import DocsService as DocsService

__all__ = ["DocsService"]


def __getattr__(name: str) -> Any:
    # Resolved on first access, so importing the package does not load the corpus code.
    if name == "DocsService":
        from .service import DocsService

        return DocsService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
