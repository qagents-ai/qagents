"""Session stores beyond the built-in `InMemoryStore` and `SQLiteStore`.

| Store | For |
|---|---|
| `qagents.stores.postgres.PostgresStore` | workers on several hosts sharing one PostgreSQL |

Each store keeps the same append-only JSON journal and implements the
optional run status and lease methods of `qagents.session.SessionStore`, so it
works with `qagents.runtime.AgentRuntime`. Importing this package does not
import any database driver; a store checks for its driver when you create it.
"""

from __future__ import annotations

from .postgres import PostgresStore

__all__ = ["PostgresStore"]
