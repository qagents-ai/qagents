"""Bounded messaging for collaborating agents."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from . import prompts
from ._team_sqlite import SQLiteMailbox
from .errors import ConfigurationError, ToolError
from .tools import RunContext, tool

if TYPE_CHECKING:
    from .tools import Tool


@dataclass(frozen=True)
class TeamMessage:
    """A message with a sender assigned by the host, not by the model."""

    sender: str
    recipient: str
    text: str
    sequence: int


class TeamMailbox:
    """Named inboxes shared by one team run, including its coordinator.

    Assign identities using `endpoint()` in application code. Give each
    participant its own endpoint through a linked child run or bound tools.
    Pass ``db_path`` for a durable mailbox shared by processes on one host.
    """

    def __init__(
        self,
        members: list[str],
        *,
        capacity: int = 64,
        max_text_chars: int = 8000,
        db_path: str | PathLike[str] | None = None,
    ):
        if not members or any(not isinstance(name, str) or not name.strip() for name in members):
            raise ConfigurationError("Team member names must be nonempty strings.")
        if len(set(members)) != len(members):
            raise ConfigurationError("Team member names must be unique.")
        if (
            type(capacity) is not int
            or capacity < 1
            or type(max_text_chars) is not int
            or max_text_chars < 1
        ):
            raise ConfigurationError(
                "Team mailbox capacity and text limit must be positive integers."
            )
        self._members = frozenset(members)
        self._sqlite = (
            SQLiteMailbox(Path(db_path).resolve(), members, capacity, max_text_chars)
            if db_path is not None
            else None
        )
        self._queues: dict[str, deque[TeamMessage]] = {}
        self._ready: dict[str, asyncio.Event] = {}
        if self._sqlite is None:
            self._queues = {name: deque() for name in members}
            self._ready = {name: asyncio.Event() for name in members}
        self._capacity = capacity
        self._text_limit = max_text_chars
        self._sequence = 0
        self._loop = None

    def endpoint(self, member: str) -> TeamEndpoint:
        """Bind a trusted participant identity to its mailbox and tools."""
        if member not in self._members:
            raise ConfigurationError(f"Unknown team member {member!r}.")
        return TeamEndpoint(self, member)

    def _assert_loop(self):
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise ConfigurationError("A team mailbox belongs to one running event loop.")


@dataclass(frozen=True)
class TeamEndpoint:
    """Send as one registered member and read only that member's inbox."""

    _mailbox: TeamMailbox = field(repr=False)
    member: str

    async def send(self, recipient: str, text: str) -> TeamMessage:
        """Queue a message addressed to another registered team member."""
        return await self._send(recipient, text, operation_id=None)

    async def _send(self, recipient: str, text: str, operation_id: str | None) -> TeamMessage:
        box = self._mailbox
        if not isinstance(recipient, str) or recipient not in box._members:
            raise ToolError(prompts.TEAM_UNKNOWN_MEMBER.format(member=recipient))
        if not isinstance(text, str) or not text.strip() or len(text) > box._text_limit:
            raise ToolError(prompts.TEAM_TEXT_LIMIT.format(limit=box._text_limit))
        if box._sqlite is not None:
            row = await asyncio.to_thread(
                box._sqlite.send, self.member, recipient, text, operation_id
            )
            return TeamMessage(*row)
        box._assert_loop()
        queue = box._queues[recipient]
        if len(queue) >= box._capacity:
            raise ToolError(prompts.TEAM_INBOX_FULL.format(member=recipient))
        box._sequence += 1
        item = TeamMessage(self.member, recipient, text, box._sequence)
        queue.append(item)
        box._ready[recipient].set()
        return item

    async def receive(self, *, wait_s: float = 0, limit: int = 16) -> list[TeamMessage]:
        """Read up to limit messages in order, optionally waiting for an arrival."""
        return await self._receive(wait_s, limit, operation_id=None)

    async def _receive(
        self, wait_s: float, limit: int, operation_id: str | None
    ) -> list[TeamMessage]:
        box = self._mailbox
        if (
            type(wait_s) not in (float, int)
            or not 0 <= wait_s <= 30
            or type(limit) is not int
            or not 1 <= limit <= 64
        ):
            raise ToolError(prompts.TEAM_RECEIVE_LIMIT)
        if box._sqlite is not None:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + wait_s
            while True:
                final = loop.time() >= deadline
                rows = await asyncio.to_thread(
                    box._sqlite.receive, self.member, limit, operation_id, final=final
                )
                if rows is not None:
                    return [TeamMessage(*row) for row in rows]
                await asyncio.sleep(min(0.05, max(0, deadline - loop.time())))
        box._assert_loop()
        queue, ready = box._queues[self.member], box._ready[self.member]
        try:
            async with asyncio.timeout(wait_s):
                while not queue and wait_s:
                    await ready.wait()
        except TimeoutError:
            return []
        received = [queue.popleft() for _ in range(min(len(queue), limit))]
        if not queue:
            ready.clear()
        return received

    def tools(self) -> list[Tool]:
        """Expose send_message and receive_messages with a host-bound sender."""

        durable = self._mailbox._sqlite is not None

        @tool(idempotent=durable, description=prompts.TEAM_SEND_DESCRIPTION)
        async def send_message(ctx: RunContext, recipient: str, text: str) -> TeamMessage:
            return await self._send(recipient, text, ctx.operation_id)

        @tool(idempotent=durable, timeout=35, description=prompts.TEAM_RECEIVE_DESCRIPTION)
        async def receive_messages(ctx: RunContext, wait_s: float = 0) -> list[TeamMessage]:
            return await self._receive(wait_s, 16, ctx.operation_id)

        return [send_message, receive_messages]
