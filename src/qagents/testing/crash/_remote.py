"""Authenticated local world RPC and verified child-process control for internal tests.

Only children started by the harness receive the random authentication key. The
pickle-based multiprocessing connection is an internal trusted-peer transport,
never a network-facing API. SQLite worker threads read cached clock values only.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import inspect
import os
import pickle
import secrets
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path
from typing import Any

from qagents.effects._ports import ReachEvent
from qagents.errors import ConfigurationError, JournalOutcomeUnknown, LeaseError

from ._clock import ManualClock, _finite
from ._loop import VirtualLoop


def _disable_nagle(connection: Connection) -> None:
    # Each request waits on a reply. Avoid the delayed-ACK/Nagle interaction
    # on the short frames sent over this exclusively loopback TCP socket.
    wire = socket.socket(fileno=connection.fileno())
    try:
        wire.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    finally:
        wire.detach()


class RemoteProtocolError(RuntimeError):
    """The child/parent exchange could not establish a valid world observation."""


class RemoteCallError(RuntimeError):
    """An ordinary remote handler exception, retained without inventing a reply."""


@dataclass(frozen=True)
class KillWithoutReply:
    """The selected store boundary was reached; terminate the verified child now."""

    reason: str


@dataclass(frozen=True)
class CancelDriver:
    """Deliver a destination result along with an injected worker cancellation."""

    result: Any


class RemoteClock:
    """Cached clocks refreshed by every reply; explicit jumps ask the parent."""

    def __init__(self, request: Callable[[str, Any], Any]) -> None:
        self._request = request
        self._store, self._monotonic = 0.0, 0.0
        self._lock = threading.Lock()

    def refresh(self, values: Any) -> None:
        if not isinstance(values, dict) or set(values) != {"store", "monotonic"}:
            raise RemoteProtocolError("reply needs both clock domains")
        store = _finite(values["store"], "remote store time")
        monotonic = _finite(values["monotonic"], "remote monotonic time")
        with self._lock:
            if monotonic < self._monotonic:
                raise RemoteProtocolError("remote monotonic time moved backwards")
            self._store, self._monotonic = store, monotonic

    def store_time(self) -> float:
        with self._lock:
            return self._store

    def monotonic(self) -> float:
        with self._lock:
            return self._monotonic

    def advance(self, delta: float) -> None:
        duration = _finite(delta, "remote advance")
        if duration < 0:
            raise ValueError("remote advance cannot be negative")
        self._request("clock.advance", duration)

    def step_store(self, delta: float) -> None:
        self._request("clock.step_store", _finite(delta, "remote store step"))


class RemoteClient:
    """One synchronous RPC stream; async proxy calls use a counted worker thread."""

    def __init__(self, connection: Connection, *, timeout_s: float = 30) -> None:
        self.connection = connection
        self.timeout_s = _finite(timeout_s, "remote timeout")
        if self.timeout_s <= 0:
            raise ValueError("remote timeout must be positive")
        self.clock = RemoteClock(self.request)
        self._lock = threading.Lock()
        self._sequence = 0
        self._closed = False
        self.driver: asyncio.Task | None = None

    @classmethod
    def connect(cls, address: tuple[str, int], authkey: bytes, *, timeout_s: float = 30):
        if address[0] != "127.0.0.1":
            raise ValueError("remote crash worlds are loopback only")
        made = cls(Client(address, family="AF_INET", authkey=authkey), timeout_s=timeout_s)
        _disable_nagle(made.connection)
        try:
            made.request("hello", {"pid": os.getpid(), "protocol": 1})
        except BaseException:
            made.close()
            raise
        return made

    def request(self, method: str, value: Any = None) -> Any:
        with self._lock:
            if self._closed:
                raise RemoteProtocolError("remote connection is closed")
            self._sequence += 1
            sequence = self._sequence
            self.connection.send({"sequence": sequence, "method": method, "value": value})
            if not self.connection.poll(self.timeout_s):
                self.close()
                raise RemoteProtocolError("remote world did not reply before its real-time bound")
            try:
                reply = self.connection.recv()
            except EOFError as error:
                raise RemoteProtocolError("remote world closed without a reply") from error
            if (
                not isinstance(reply, dict)
                or type(reply.get("sequence")) is not int
                or reply.get("sequence") != sequence
                or type(reply.get("ok")) is not bool
            ):
                self.close()
                raise RemoteProtocolError("malformed or out-of-order remote reply")
            self.clock.refresh(reply.get("clock"))
            if not reply["ok"]:
                error = reply.get("error")
                if isinstance(error, Exception):
                    raise error
                raise RemoteCallError(str(error))
            return reply.get("value")

    async def call(self, method: str, value: Any = None) -> Any:
        result = await asyncio.to_thread(self.request, method, value)
        if isinstance(result, CancelDriver):
            if self.driver is None:
                raise RemoteProtocolError("A cancellation requires an active worker task.")
            self.driver.cancel()
            return result.result
        return result

    def reached(self, event: ReachEvent) -> None:
        self.request("reach", asdict(event))

    def close(self) -> None:
        # Deliberately does not take _lock: timeout handling already owns it.
        if not self._closed:
            self._closed = True
            self.connection.close()


class OwnedLegProcess:
    """Own a Popen handle, then verify the interpreter PID before counting a kill."""

    def __init__(
        self, script: Path, args: Sequence[str], *, directory: Path, bootstrap: Any
    ) -> None:
        script = script.resolve(strict=True)
        directory.mkdir(parents=True, exist_ok=True)
        self.stdout_path, self.stderr_path = (
            directory / "child.stdout.log",
            directory / "child.stderr.log",
        )
        self._stdout = self.stdout_path.open("xb")
        try:
            self._stderr = self.stderr_path.open("xb")
        except BaseException:
            self._stdout.close()
            raise
        self.verified = False
        self.killed = False
        self.reported_pid: int | None = None
        try:
            self.process = subprocess.Popen(
                [
                    getattr(sys, "_base_executable", sys.executable),
                    "-I",
                    "-S",
                    "-B",
                    "-u",
                    str(script),
                    *args,
                ],
                stdin=subprocess.PIPE,
                stdout=self._stdout,
                stderr=self._stderr,
                cwd=Path.cwd(),
                close_fds=True,
                env={
                    name: os.environ[name]
                    for name in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "LANG", "LC_ALL")
                    if name in os.environ
                },
            )
            with self.process.stdin as stdin:
                pickle.dump(bootstrap.pop("paths"), stdin)
                pickle.dump(bootstrap, stdin)
        except BaseException:
            if hasattr(self, "process") and self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=10)
            self._stdout.close()
            self._stderr.close()
            raise

    def verify(self, pid: Any) -> None:
        if type(pid) is not int or pid <= 0 or pid != self.process.pid:
            raise RemoteProtocolError("reported interpreter PID differs from the owned Popen PID")
        if self.process.poll() is not None:
            raise RemoteProtocolError("child exited before PID verification")
        self.reported_pid, self.verified = pid, True

    def kill(self) -> int:
        if not self.verified:
            raise RemoteProtocolError(
                "refusing a trial kill before actual interpreter PID verification"
            )
        if self.process.poll() is not None:
            raise RemoteProtocolError("child already exited; the requested kill did not happen")
        self.process.kill()
        code = self.process.wait(timeout=10)
        if code == 0:
            raise RemoteProtocolError("child exited normally; no crash can be claimed")
        self.killed = True
        return code

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                # Cleanup is not a successful trial kill. A Popen handle, never an
                # enumerated PID, identifies the process this object started.
                self.process.kill()
                self.process.wait(timeout=10)
        finally:
            self._stdout.close()
            self._stderr.close()


class WorldServer:
    """One authenticated child connection served by an owned, accounted thread."""

    def __init__(
        self,
        loop: VirtualLoop,
        clock: ManualClock,
        handlers: Mapping[str, Callable],
        *,
        reach: Callable[[ReachEvent], None] | None = None,
        verify_pid: Callable[[int], None] | None = None,
        kill: Callable[[], Any] | None = None,
    ) -> None:
        reserved = {"hello", "clock.advance", "clock.step_store", "reach"}
        if reserved & handlers.keys() or not all(
            isinstance(name, str) and callable(fn) for name, fn in handlers.items()
        ):
            raise ValueError("remote handlers must be named callables outside reserved operations")
        self.loop, self.clock, self.handlers = loop, clock, dict(handlers)
        self.reach, self.verify_pid, self.kill = reach, verify_pid, kill
        self.authkey = secrets.token_bytes(32)
        self.listener = Listener(("127.0.0.1", 0), family="AF_INET", authkey=self.authkey)
        self.address = self.listener.address
        self.connection: Connection | None = None
        self.future: concurrent.futures.Future = concurrent.futures.Future()
        self.wrapped: asyncio.Future | None = None
        self.thread: threading.Thread | None = None
        self.errors: list[str] = []
        self.requests: list[tuple[int, str]] = []
        self.reported_pid: int | None = None
        self.kill_reason: str | None = None
        self._watcher: threading.Thread | None = None
        self._closed = threading.Event()
        self._pending_task: asyncio.Task | None = None

    def start(self) -> None:
        if self.thread is not None:
            raise ValueError("world server already started")
        self.future.set_running_or_notify_cancel()
        self.wrapped = asyncio.wrap_future(self.future, loop=self.loop)
        self.loop.observe_thread(self.future)
        self.thread = threading.Thread(target=self._serve, name="qagents-world-rpc", daemon=True)
        self.thread.start()

    async def _dispatch(self, request: Any) -> Any:
        self._pending_task = asyncio.current_task()
        try:
            return await self._handle(request)
        finally:
            self._pending_task = None

    async def _handle(self, request: Any) -> Any:
        if not isinstance(request, dict) or set(request) != {"sequence", "method", "value"}:
            raise RemoteProtocolError("malformed world request")
        sequence, method, value = request["sequence"], request["method"], request["value"]
        if (
            type(sequence) is not int
            or sequence != len(self.requests) + 1
            or not isinstance(method, str)
        ):
            raise RemoteProtocolError("world request is out of order")
        self.requests.append((sequence, method))
        if self.reported_pid is None and method != "hello":
            raise RemoteProtocolError("child must identify its interpreter before world operations")
        if method == "hello":
            if (
                self.reported_pid is not None
                or not isinstance(value, dict)
                or set(value) != {"pid", "protocol"}
            ):
                raise RemoteProtocolError("invalid child handshake")
            pid = value["pid"]
            if (
                type(pid) is not int
                or pid <= 0
                or type(value["protocol"]) is not int
                or value["protocol"] != 1
            ):
                raise RemoteProtocolError("invalid child PID or protocol")
            if self.verify_pid is not None:
                self.verify_pid(pid)
            self.reported_pid = pid
            return {"pid": pid, "protocol": 1}
        if method == "clock.advance":
            self.clock.advance(value)
            return None
        if method == "clock.step_store":
            self.clock.step_store(value)
            return None
        if method == "reach":
            fields = {"session_id", "record_id", "effect", "rule", "point", "detail"}
            if not isinstance(value, dict) or set(value) != fields:
                raise RemoteProtocolError("invalid reach event")
            if any(
                not isinstance(value[name], str) for name in fields - {"point", "detail"}
            ) or any(
                value[name] is not None and not isinstance(value[name], str)
                for name in {"point", "detail"}
            ):
                raise RemoteProtocolError("invalid reach event fields")
            if self.reach is not None:
                self.reach(ReachEvent(**value))
            return None
        handler = self.handlers.get(method)
        if handler is None:
            raise RemoteProtocolError("unknown world operation")
        result = handler(value)
        return await result if inspect.isawaitable(result) else result

    def _serve(self) -> None:
        self.future._qagents_thread_id = threading.get_ident()
        try:
            self.connection = self.listener.accept()
            _disable_nagle(self.connection)
            while not self._closed.is_set():
                if not self.connection.poll(0.05):
                    continue
                try:
                    request = self.connection.recv()
                except EOFError:
                    break
                sequence = request.get("sequence") if isinstance(request, dict) else None
                try:
                    result = self.loop.call_from_thread(self._dispatch(request))
                    if isinstance(result, KillWithoutReply):
                        if self.kill is None:
                            raise ConfigurationError("remote kill boundary has no owned process")
                        self.kill()
                        self.kill_reason = result.reason
                        break
                    reply = {"sequence": sequence, "ok": True, "value": result}
                except Exception as error:
                    if isinstance(error, (LeaseError, JournalOutcomeUnknown)):
                        error = type(error)(str(error))
                    try:
                        # Validate round-trip because some exception constructors are keyword-only.
                        pickle.loads(pickle.dumps(error))
                    except Exception:
                        error = RemoteCallError(f"{type(error).__name__}: {error}")
                    reply = {"sequence": sequence, "ok": False, "error": error}
                reply["clock"] = {
                    "store": self.clock.store_time(),
                    "monotonic": self.clock.monotonic(),
                }
                self.connection.send(reply)
        except (EOFError, OSError) as error:
            if not self._closed.is_set():
                self.errors.append(type(error).__name__)
        except BaseException as error:
            self.errors.append(f"{type(error).__name__}: {error}")
        finally:
            if self.connection is not None:
                self.connection.close()
            self.listener.close()
            self.future.set_result(None)

    async def wait(self, worker: OwnedLegProcess) -> None:
        # The loop need not wait for the process-exit watcher before advancing
        # virtual time inside a parent-owned fake destination.
        exited = self.loop.create_future()

        def watch() -> None:
            code = worker.process.wait()
            if not self.loop.is_closed():
                with contextlib.suppress(RuntimeError):
                    self.loop.call_soon_threadsafe(exited.set_result, code)

        self._watcher = threading.Thread(target=watch, daemon=True)
        self._watcher.start()
        done, _ = await asyncio.wait({exited, self.wrapped}, return_when=asyncio.FIRST_COMPLETED)
        if exited in done and not worker.verified:
            raise RemoteProtocolError(f"Worker exited before handshake (exit {exited.result()}).")
        await self.wrapped

    async def aclose(self) -> None:
        self._closed.set()
        if self._pending_task is not None:
            self._pending_task.cancel()
        if self.thread is None:
            self.listener.close()
            return
        if self.connection is None and self.thread.is_alive():
            # Wake accept with an authenticated local peer; do not depend on closing
            # a listening socket from another thread interrupting accept on Windows.
            def wake():
                with contextlib.suppress(OSError, EOFError):
                    Client(self.address, family="AF_INET", authkey=self.authkey).close()

            await asyncio.to_thread(wake)
        elif self.connection is not None:
            self.connection.close()
        assert self.wrapped is not None
        await self.wrapped
        await self.loop.threads_drained()
        self.thread.join(timeout=1)
        if self._watcher is not None:
            self._watcher.join(timeout=1)
        if self.thread.is_alive():
            raise RemoteProtocolError("remote world thread did not exit")
