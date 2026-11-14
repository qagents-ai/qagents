"""Transactional inbox operations over a local SQLite database."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING

from . import prompts
from .errors import ConfigurationError, ToolError
from .session import _enable_wal

if TYPE_CHECKING:
    from pathlib import Path


class SQLiteMailbox:
    def __init__(self, path: Path, members: list[str], capacity: int, max_text_chars: int):
        self.path = path
        self.capacity = capacity
        config = json.dumps([sorted(members), capacity, max_text_chars])
        with self._connection(wal=True) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS team_config "
                "(id INTEGER PRIMARY KEY CHECK (id = 1), config TEXT NOT NULL)"
            )
            conn.execute("INSERT OR IGNORE INTO team_config VALUES (1, ?)", (config,))
            stored = conn.execute("SELECT config FROM team_config WHERE id = 1").fetchone()[0]
            if stored != config:
                raise ConfigurationError(
                    "The team mailbox configuration must match the database's members and limits."
                )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS team_messages (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    body TEXT NOT NULL
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS team_inbox ON team_messages(recipient, sequence)"
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS team_tool_calls (
                    member TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (member, operation_id)
                )"""
            )

    @contextmanager
    def _connection(self, *, wal: bool = False):
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=5)
        try:
            if wal:
                _enable_wal(conn)
            conn.execute("PRAGMA synchronous=FULL")
            yield conn
            conn.commit()
        finally:
            # Closing rolls back any transaction that could not commit.
            conn.close()

    @staticmethod
    def _replay(conn, member, operation_id, kind):
        if operation_id is None:
            return None
        row = conn.execute(
            "SELECT kind, payload FROM team_tool_calls WHERE member = ? AND operation_id = ?",
            (member, operation_id),
        ).fetchone()
        if row is None:
            return None
        if row[0] != kind:
            raise ToolError(prompts.TEAM_REPLAY_CONFLICT)
        return json.loads(row[1])

    @staticmethod
    def _record(conn, member, operation_id, kind, payload):
        if operation_id is not None:
            conn.execute(
                "INSERT INTO team_tool_calls VALUES (?, ?, ?, ?)",
                (member, operation_id, kind, json.dumps(payload)),
            )

    def send(self, sender: str, recipient: str, body: str, operation_id: str | None):
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if (saved := self._replay(conn, sender, operation_id, "send")) is not None:
                if saved[:3] != [sender, recipient, body]:
                    raise ToolError(prompts.TEAM_REPLAY_CONFLICT)
                return saved
            count = conn.execute(
                "SELECT COUNT(*) FROM team_messages WHERE recipient = ?", (recipient,)
            ).fetchone()[0]
            if count >= self.capacity:
                raise ToolError(prompts.TEAM_INBOX_FULL.format(member=recipient))
            write = conn.execute(
                "INSERT INTO team_messages(sender, recipient, body) VALUES (?, ?, ?)",
                (sender, recipient, body),
            )
            message = (sender, recipient, body, write.lastrowid)
            self._record(conn, sender, operation_id, "send", message)
            return message

    def receive(self, member: str, limit: int, operation_id: str | None, *, final: bool):
        # None keeps polling. A recorded empty result ends replay immediately.
        with self._connection() as conn:
            if (saved := self._replay(conn, member, operation_id, "receive")) is not None:
                return saved
            if not (final and operation_id is not None):
                available = conn.execute(
                    "SELECT 1 FROM team_messages WHERE recipient = ? LIMIT 1", (member,)
                ).fetchone()
                if available is None:
                    return [] if final else None
            conn.execute("BEGIN IMMEDIATE")
            # A competing reader could empty the inbox or finish the same call.
            if (saved := self._replay(conn, member, operation_id, "receive")) is not None:
                return saved
            messages = conn.execute(
                """SELECT sender, recipient, body, sequence FROM team_messages
                   WHERE recipient = ? ORDER BY sequence LIMIT ?""",
                (member, limit),
            ).fetchall()
            conn.executemany(
                "DELETE FROM team_messages WHERE sequence = ?", ((row[3],) for row in messages)
            )
            if not messages and not final:
                return None  # another reader took the batch
            self._record(conn, member, operation_id, "receive", messages)
            return messages
