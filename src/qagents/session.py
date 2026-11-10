"""Sessions: a conversation that outlives a process.

A session is an append-only journal of JSON entries. The agent loop writes an
entry for every message, every tool start and every tool result *as it
happens*, so the journal always says exactly how far a run got. Conversation
state is derived by reading the journal; nothing is pickled and nothing
depends on the shape of your code, so a run can be resumed by a different
process, a different deployment, or after the agent's tools changed.

Entry kinds:

| Kind | Meaning (data) |
|---|---|
| `run_started` | a run began (`agent`) |
| `message` | a message joined the conversation (`message`) |
| `tool_started` | a tool call began executing (`tool_call_id`, `name`) |
| `tool_result` | a tool call finished (`result`) |
| `agent_switched` | a handoff changed the active agent (`agent`) |
| `run_paused` | the run is waiting for approvals (`pending`) |
| `run_finished` | the run ended (`status`, `stop_reason`) |
| `run_failed` | the run raised and stays open for a resume (`error`) |
| `compaction` | the history was replaced by a shorter one (`messages`) |
| `workflow_step` | a workflow step completed (`name`, `result`) |

Readers must ignore kinds they do not know: new ones may be added.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .messages import Message, ToolResultPart


class Entry(BaseModel):
    kind: str
    data: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    ts: float = Field(default_factory=time.time)
    seq: int = 0
    """Position in the session, assigned by the store."""


class PendingApproval(BaseModel):
    """A tool call that is waiting for a human decision."""

    tool_call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class SessionState(BaseModel):
    """What the journal says about a session right now."""

    messages: list[Message] = Field(default_factory=list)
    open_run_id: str | None = None
    """Id of a run that started and has not finished (running, paused or crashed)."""
    agent: str | None = None
    """Agent that was active in the open run."""
    tool_results: dict[str, ToolResultPart] = Field(default_factory=dict)
    """Results already recorded for tool calls of the last assistant message."""
    started_tools: list[str] = Field(default_factory=list)
    """Tool calls that began executing and have no recorded result."""
    pending_approvals: list[PendingApproval] = Field(default_factory=list)
    run_messages: int = 0
    """How many messages the latest run has written to the journal."""

    @property
    def unresolved_tool_calls(self) -> bool:
        """True when the last message is an assistant message whose tool calls lack results."""
        return bool(
            self.messages and self.messages[-1].role == "assistant" and self.messages[-1].tool_calls
        )


def derive_state(entries: list[Entry]) -> SessionState:
    """Replay a journal into the current `SessionState`."""
    state = SessionState()
    started: dict[str, None] = {}
    for entry in entries:
        kind, data = entry.kind, entry.data
        if kind == "message":
            state.messages.append(Message.model_validate(data["message"]))
            state.run_messages += 1
            state.tool_results = {}
            state.pending_approvals = []
            started = {}
        elif kind == "tool_started":
            started[data["tool_call_id"]] = None
        elif kind == "tool_result":
            result = ToolResultPart.model_validate(data["result"])
            state.tool_results[result.tool_call_id] = result
            started.pop(result.tool_call_id, None)
        elif kind == "run_started":
            state.open_run_id = entry.run_id
            state.agent = data.get("agent")
            state.run_messages = 0
        elif kind == "agent_switched":
            state.agent = data.get("agent")
        elif kind == "run_paused":
            state.pending_approvals = [
                PendingApproval.model_validate(item) for item in data.get("pending", [])
            ]
        elif kind == "run_finished":
            if entry.run_id == state.open_run_id:
                state.open_run_id = None
                state.pending_approvals = []
        elif kind == "compaction":
            state.messages = [Message.model_validate(item) for item in data["messages"]]
    state.started_tools = list(started)
    return state


class SessionStore(ABC):
    """Where journals live. Implement three methods to add a backend."""

    @abstractmethod
    async def append(self, session_id: str, entries: list[Entry]) -> None:
        """Durably add ``entries`` to the end of the session, in order."""

    @abstractmethod
    async def load(self, session_id: str) -> list[Entry]:
        """Return every entry of the session, oldest first."""

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Remove the session."""

    async def aclose(self) -> None:  # noqa: B027 - optional to override
        """Release what the store holds."""


class InMemoryStore(SessionStore):
    """Keeps journals in a dict. For tests and for sessions that need not survive a restart."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[Entry]] = {}

    async def append(self, session_id: str, entries: list[Entry]) -> None:
        journal = self._sessions.setdefault(session_id, [])
        for entry in entries:
            journal.append(entry.model_copy(update={"seq": len(journal) + 1}))

    async def load(self, session_id: str) -> list[Entry]:
        return list(self._sessions.get(session_id, []))

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class SQLiteStore(SessionStore):
    """Journals in one SQLite file (WAL mode). Good for a single host.

    Calls run on worker threads, so the event loop is never blocked on disk.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._connection: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            connection = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS entries ("
                " session_id TEXT NOT NULL, seq INTEGER NOT NULL, ts REAL NOT NULL,"
                " run_id TEXT, kind TEXT NOT NULL, data TEXT NOT NULL,"
                " PRIMARY KEY (session_id, seq))"
            )
            self._connection = connection
        return self._connection

    def _append(self, session_id: str, entries: list[Entry]) -> None:
        with self._lock:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            try:
                (last,) = connection.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM entries WHERE session_id = ?", (session_id,)
                ).fetchone()
                connection.executemany(
                    "INSERT INTO entries (session_id, seq, ts, run_id, kind, data)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            session_id,
                            last + offset,
                            entry.ts,
                            entry.run_id,
                            entry.kind,
                            json.dumps(entry.data, ensure_ascii=False),
                        )
                        for offset, entry in enumerate(entries, start=1)
                    ],
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def _load(self, session_id: str) -> list[Entry]:
        with self._lock:
            rows = (
                self._connect()
                .execute(
                    "SELECT seq, ts, run_id, kind, data FROM entries"
                    " WHERE session_id = ? ORDER BY seq",
                    (session_id,),
                )
                .fetchall()
            )
        return [
            Entry(seq=seq, ts=ts, run_id=run_id, kind=kind, data=json.loads(data))
            for seq, ts, run_id, kind, data in rows
        ]

    def _delete(self, session_id: str) -> None:
        with self._lock:
            self._connect().execute("DELETE FROM entries WHERE session_id = ?", (session_id,))

    def _close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    async def append(self, session_id: str, entries: list[Entry]) -> None:
        if entries:
            await asyncio.to_thread(self._append, session_id, entries)

    async def load(self, session_id: str) -> list[Entry]:
        return await asyncio.to_thread(self._load, session_id)

    async def delete(self, session_id: str) -> None:
        await asyncio.to_thread(self._delete, session_id)

    async def aclose(self) -> None:
        await asyncio.to_thread(self._close)


class Session:
    """One conversation in a store.

    ``Session()`` alone keeps the conversation in memory. Pass a store and an
    id to make it durable:

        store = SQLiteStore("agents.db")
        session = Session(store, id=f"user-{user_id}")
        await agent.run("Hello", session=session)
    """

    def __init__(self, store: SessionStore | None = None, id: str | None = None) -> None:
        self.store = store or InMemoryStore()
        self.id = id or uuid.uuid4().hex

    async def entries(self) -> list[Entry]:
        return await self.store.load(self.id)

    async def state(self) -> SessionState:
        return derive_state(await self.entries())

    async def messages(self) -> list[Message]:
        return (await self.state()).messages

    async def append(self, kind: str, data: dict[str, Any], *, run_id: str | None = None) -> None:
        await self.store.append(self.id, [Entry(kind=kind, data=data, run_id=run_id)])

    async def clear(self) -> None:
        await self.store.delete(self.id)

    def __repr__(self) -> str:
        return f"Session(id={self.id!r}, store={type(self.store).__name__})"
