"""In-memory streams for JSON-RPC tests: what one side writes, the other reads."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from qagents.integrations._jsonrpc import JsonRpcPeer


class MemoryWriter:
    """A writer that feeds an `asyncio.StreamReader`, like a pipe.

    ``gate``: while it is set to an unset `asyncio.Event`, ``drain()`` waits
    for it, which is how a test plays a peer that stops reading.
    """

    def __init__(self, target: asyncio.StreamReader) -> None:
        self.target = target
        self.closed = False
        self.gate: asyncio.Event | None = None
        self.drains = 0
        self.fail_with: BaseException | None = None

    def write(self, data: bytes) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        if self.closed:
            raise RuntimeError("the writer is closed")
        self.target.feed_data(data)

    async def drain(self) -> None:
        self.drains += 1
        if self.gate is not None:
            await self.gate.wait()
        await asyncio.sleep(0)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.target.feed_eof()

    def is_closing(self) -> bool:
        return self.closed


def pipe() -> tuple[asyncio.StreamReader, MemoryWriter]:
    """A reader and the writer that feeds it."""
    reader = asyncio.StreamReader(limit=2**31)
    return reader, MemoryWriter(reader)


def peer_pair(**options: Any) -> tuple[JsonRpcPeer, JsonRpcPeer]:
    """Two connected peers, ``a`` and ``b`` (not started)."""
    a_in, b_out = pipe()
    b_in, a_out = pipe()
    a = JsonRpcPeer(a_in, a_out, name="a", **options)
    b = JsonRpcPeer(b_in, b_out, name="b", **options)
    return a, b


class RawSide:
    """The other end of one peer, driven by hand: send raw lines, read what the peer wrote."""

    def __init__(self, **options: Any) -> None:
        peer_in, self.writer = pipe()
        self.reader, peer_out = pipe()
        self.peer = JsonRpcPeer(peer_in, peer_out, name="peer", **options)

    def send_raw(self, data: bytes) -> None:
        self.writer.write(data)

    def send(self, message: Any) -> None:
        self.send_raw(json.dumps(message).encode("utf-8") + b"\n")

    def eof(self) -> None:
        self.writer.close()

    async def receive(self, timeout: float = 5.0) -> Any:
        line = await asyncio.wait_for(self.reader.readline(), timeout)
        if not line:
            raise EOFError("the peer closed its output")
        return json.loads(line)
