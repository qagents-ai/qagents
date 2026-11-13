"""A JSON-RPC 2.0 peer over newline-delimited JSON, for both directions of a stdio link.

ACP agents and the Devvi SDK speak JSON-RPC 2.0 with one JSON message per
line on the process's stdin and stdout. Both sides send requests: qagents asks
the agent to run a prompt, and the agent asks qagents for permission. Both
sides send notifications too (session updates, cancels).

    peer = JsonRpcPeer.for_process(process, cancel_method="$/cancel_request")
    peer.on_notification("session/update", on_update)            # sync or async
    peer.on_request("session/request_permission", on_permission)  # async, returns the result
    peer.start()
    info = await peer.request("initialize", {"protocolVersion": 1}, timeout=30)
    await peer.notify("session/cancel", {"sessionId": session_id})
    await peer.aclose()

What it guarantees:

* **Order.** Messages are handled in the order they arrive. Notification
  handlers run inline, one after the other, before the next line is read, so
  a ``tool_call`` update is always handled before a permission request that
  follows it. Keep them quick: a slow one holds up everything behind it.
  Request handlers run in their own tasks, so a permission prompt that waits
  for a human does not stop the stream.
* **Answers to the peer.** A handler's return value is the result. Raise
  `JsonRpcError` to answer with that error; any other exception answers
  -32603 (logged here, its text not sent). A method without a handler gets
  -32601. A ``cancel_method`` notification for a request still being handled
  cancels its handler and answers -32800.
* **Closing.** End of stream, a broken pipe, a line over ``max_line_bytes``
  or ``max_malformed`` malformed lines in a row close the peer. Every pending
  request then fails at once with `ExternalAgentConnectionError` (or its
  subclass `ExternalAgentProtocolError`) that names the request and, for a
  process, its exit code and stderr tail. Nothing waits forever.
* **Size.** Lines up to ``max_line_bytes`` (32 MiB by default) are read in
  chunks, so a 20 MiB message is accepted without stalling the event loop on
  a quadratic scan. A longer line closes the connection with a clear error:
  its request id cannot be known, so failing everything is the only answer
  that never leaves a caller waiting.
* **Malformed lines** (not JSON, not JSON-RPC 2.0) are logged and skipped,
  as Devvi does with its input; ``max_malformed`` of them in a row (3) close
  the connection. Blank lines are ignored. A batch (a JSON array) is handled
  item by item, and each answer is sent on its own line.
* **Backpressure.** Writes are serialised and wait for the transport to drain,
  so a peer that stops reading slows the sender down instead of filling memory.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Protocol, Self

from .. import prompts
from ..external import (
    ExternalAgentConnectionError,
    ExternalAgentError,
    ExternalAgentProtocolError,
    ExternalAgentTimeoutError,
)

logger = logging.getLogger("qagents.integrations.jsonrpc")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
REQUEST_CANCELLED = -32800
"""Error codes of JSON-RPC 2.0, plus -32800 (request cancelled, as LSP and ACP use it)."""

MAX_LINE_BYTES = 32 * 1024 * 1024
"""Default largest line accepted: 32 MiB. Devvi's SDK sends large lines (tool schemas,
timed streams), so this is far above asyncio's 64 KiB default."""

MAX_MALFORMED = 3
"""Default number of malformed lines in a row that closes the connection."""

READ_CHUNK = 1024 * 1024

RequestHandler = Callable[[Any], Awaitable[Any]]
"""``async (params) -> result``. Raise `JsonRpcError` to answer with an error."""

NotificationHandler = Callable[[Any], Awaitable[None] | None]
"""``(params) -> None``, sync or async. Runs inline: keep it quick."""


class JsonRpcError(ExternalAgentError):
    """A JSON-RPC error object: the peer answered a request with an error, or a handler
    raises it to answer the peer with one.

    Args:
        code: The JSON-RPC error code, for example -32602 (invalid params).
        message: The error message.
        data: Optional extra data of the error object.
        method: The method of the request it answers, when known.
    """

    def __init__(
        self, code: int, message: str, data: Any = None, *, method: str | None = None
    ) -> None:
        where = f" ({method})" if method else ""
        super().__init__(f"JSON-RPC error {code}{where}: {message}")
        self.code = code
        self.message = message
        self.data = data
        self.method = method

    def to_object(self) -> dict[str, Any]:
        """The error object to send: ``{"code", "message"}`` and ``data`` when set."""
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error


class _Writer(Protocol):
    def write(self, data: bytes) -> object: ...

    async def drain(self) -> None: ...

    def close(self) -> object: ...

    def is_closing(self) -> bool: ...


class _Reader(Protocol):
    async def read(self, n: int = -1) -> bytes: ...


class JsonRpcPeer:
    """One end of a JSON-RPC 2.0 connection over newline-delimited JSON.

    Args:
        reader: Where messages come from (a process's stdout).
        writer: Where messages go (a process's stdin). It must have ``write``,
            ``drain``, ``close`` and ``is_closing``, like ``asyncio.StreamWriter``.
        name: Name used in log lines and errors, for example ``"devvi"``.
        max_line_bytes: Largest line accepted; a longer one closes the connection.
        max_malformed: Malformed lines in a row that close the connection.
        cancel_method: Notification method for cancelling a request, such as
            ACP's ``"$/cancel_request"`` (params ``{"requestId": id}``). When
            set, a request the caller gives up (cancelled or timed out) is
            announced to the peer, and the same notification from the peer
            cancels the handler of that request. ``None``: neither.
        process: A `ManagedProcess` (or anything with ``explain_exit()``,
            ``exit_summary()``, ``returncode`` and ``stderr_tail()``). When the
            stream ends, errors then say how the process ended.
    """

    def __init__(
        self,
        reader: _Reader,
        writer: _Writer,
        *,
        name: str = "peer",
        max_line_bytes: int = MAX_LINE_BYTES,
        max_malformed: int = MAX_MALFORMED,
        cancel_method: str | None = None,
        process: Any = None,
    ) -> None:
        self.name = name
        self.max_line_bytes = max_line_bytes
        self.max_malformed = max(1, max_malformed)
        self.cancel_method = cancel_method
        self._reader = reader
        self._writer = writer
        self._process = process
        self._requests: dict[str, RequestHandler] = {}
        self._notifications: dict[str, NotificationHandler] = {}
        self._fallback: Callable[[str, Any], Awaitable[None] | None] | None = None
        self._pending: dict[Any, tuple[str, asyncio.Future[Any]]] = {}
        self._handling: dict[Any, asyncio.Task[None]] = {}
        self._cancelled_by_peer: set[Any] = set()
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._closed = False
        self._closed_event = asyncio.Event()
        self._close_error: ExternalAgentConnectionError | None = None
        self._close_callbacks: list[Callable[[ExternalAgentConnectionError], None]] = []
        self._malformed_run = 0
        self.malformed = 0
        """Malformed lines received in all."""
        self.received = 0
        """Messages received (a batch counts its items)."""
        self.sent = 0
        """Messages sent."""

    @classmethod
    def for_process(cls, process: Any, **options: Any) -> Self:
        """A peer on a `ManagedProcess`'s stdout and stdin, named after it."""
        options.setdefault("name", getattr(process, "name", "peer"))
        return cls(process.stdout, process.stdin, process=process, **options)

    # -- set up -------------------------------------------------------------------------

    def on_request(self, method: str, handler: RequestHandler) -> None:
        """Answer the peer's requests for ``method`` with ``handler(params)``."""
        self._requests[method] = handler

    def on_notification(self, method: str, handler: NotificationHandler) -> None:
        """Call ``handler(params)`` for each notification ``method`` from the peer."""
        self._notifications[method] = handler

    def on_other_notification(self, handler: Callable[[str, Any], Awaitable[None] | None]) -> None:
        """Call ``handler(method, params)`` for notifications no other handler takes."""
        self._fallback = handler

    def on_close(self, callback: Callable[[ExternalAgentConnectionError], None]) -> None:
        """Call ``callback(error)`` once when the connection closes, for whatever reason."""
        if self._closed and self._close_error is not None:
            callback(self._close_error)
        else:
            self._close_callbacks.append(callback)

    def start(self) -> Self:
        """Start reading. Call it once the handlers are registered; safe to call again."""
        if self._reader_task is None and not self._closed:
            self._reader_task = asyncio.create_task(self._read_loop(), name=f"jsonrpc:{self.name}")
        return self

    # -- state ----------------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def close_error(self) -> ExternalAgentConnectionError | None:
        """Why the connection closed; ``None`` while it is open."""
        return self._close_error

    async def wait_closed(self) -> ExternalAgentConnectionError | None:
        """Wait until the connection closes; returns why."""
        await self._closed_event.wait()
        return self._close_error

    # -- sending ---------------------------------------------------------------------------

    async def request(
        self, method: str, params: Any = None, *, timeout: float | None = None
    ) -> Any:
        """Send a request and wait for its result.

        Args:
            method: The method name.
            params: An object or an array; left out when ``None``.
            timeout: Seconds to wait, writing included; ``None`` waits until
                the answer comes or the connection closes.

        Raises:
            JsonRpcError: The peer answered with an error.
            ExternalAgentConnectionError: The connection closed before the answer.
            ExternalAgentTimeoutError: No answer within ``timeout`` (announced
                to the peer with ``cancel_method``, when set).
        """
        self._raise_if_closed(method)
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = (method, future)
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        try:
            async with asyncio.timeout(timeout) as scope:
                await self._send(message)
                return await future
        except TimeoutError:
            if not scope.expired():
                raise
            self._give_up(request_id)
            raise ExternalAgentTimeoutError(
                prompts.EXTERNAL_RPC_TIMEOUT.format(name=self.name, method=method, seconds=timeout)
            ) from None
        except asyncio.CancelledError:
            self._give_up(request_id)
            raise
        finally:
            self._pending.pop(request_id, None)
            if future.done() and not future.cancelled():
                future.exception()  # retrieved: a failure set while we raised is not reported again
            else:
                future.cancel()

    async def notify(self, method: str, params: Any = None) -> None:
        """Send a notification (no answer).

        Raises:
            ExternalAgentConnectionError: The connection is closed or writing failed.
        """
        self._raise_if_closed(method)
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        await self._send(message)

    async def aclose(self) -> None:
        """Close the connection: fail pending requests, stop reading, close the writer."""
        self._shutdown(
            ExternalAgentConnectionError(
                f"The connection to {self.name} was closed by qagents.",
                reason=prompts.EXTERNAL_REASON_CLOSED,
            )
        )
        task = self._reader_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        handlers = list(self._handling.values())
        for handler in handlers:
            handler.cancel()
        if handlers:
            await asyncio.gather(*handlers, return_exceptions=True)
        with contextlib.suppress(Exception):
            if not self._writer.is_closing():
                self._writer.close()

    async def _send(self, message: dict[str, Any]) -> None:
        data = _encode(message)
        async with self._write_lock:
            self._raise_if_closed(message.get("method"))
            try:
                self._writer.write(data)
                await self._writer.drain()
            except (OSError, RuntimeError, ValueError) as error:  # broken pipe, closed transport
                failure = ExternalAgentConnectionError(
                    f"Writing to {self.name} failed: {error}.", reason=prompts.EXTERNAL_REASON_LOST
                )
                self._shutdown(failure)
                raise failure from error
        self.sent += 1

    def _give_up(self, request_id: int) -> None:
        """The caller stopped waiting for ``request_id``: tell the peer, if it can be told."""
        if self._pending.pop(request_id, None) is None or self.cancel_method is None:
            return
        if self._closed:
            return
        params = {"requestId": request_id}
        cancel = {"jsonrpc": "2.0", "method": self.cancel_method, "params": params}
        task = asyncio.get_running_loop().create_task(self._send_quietly(cancel))
        task.add_done_callback(_ignore_result)

    async def _send_quietly(self, message: dict[str, Any]) -> None:
        with contextlib.suppress(ExternalAgentConnectionError):
            await self._send(message)

    def _raise_if_closed(self, method: Any) -> None:
        if not self._closed:
            return
        error = self._close_error
        what = f" Cannot send {method!r}." if isinstance(method, str) else ""
        raise ExternalAgentConnectionError(
            f"{error}{what}" if error else f"The connection to {self.name} is closed.{what}",
            reason=error.reason if error else prompts.EXTERNAL_REASON_CLOSED,
            returncode=error.returncode if error else None,
            stderr_tail=error.stderr_tail if error else None,
        )

    # -- receiving -------------------------------------------------------------------------

    async def _read_loop(self) -> None:
        failure: ExternalAgentConnectionError | None = None
        try:
            async for line in self._lines():
                await self._handle_line(line)
                if self._closed:
                    return
            failure = await self._eof_error()
        except ExternalAgentConnectionError as error:
            failure = error
        except asyncio.CancelledError:
            raise
        except Exception as error:  # a bug in a handler must not leave callers waiting
            logger.exception("Reading from %s failed", self.name)
            failure = ExternalAgentConnectionError(
                f"Reading from {self.name} failed: {type(error).__name__}: {error}",
                reason=prompts.EXTERNAL_REASON_FAILED,
            )
        finally:
            if failure is None and not self._closed:
                failure = ExternalAgentConnectionError(
                    f"Reading from {self.name} stopped.", reason=prompts.EXTERNAL_REASON_CLOSED
                )
            if failure is not None:
                self._shutdown(failure)

    async def _lines(self) -> AsyncIterator[bytes]:
        buffer = bytearray()
        scanned = 0
        while True:
            newline = buffer.find(b"\n", scanned)
            if newline > self.max_line_bytes or (newline < 0 and len(buffer) > self.max_line_bytes):
                # Checked whether or not the newline came in the same read: a long line is
                # refused even when it arrives in one chunk.
                raise self._too_large()
            if newline >= 0:
                line = bytes(buffer[:newline])
                del buffer[: newline + 1]
                scanned = 0
                yield line
                continue
            scanned = len(buffer)
            try:
                chunk = await self._reader.read(READ_CHUNK)
            except (OSError, ValueError) as error:
                raise ExternalAgentConnectionError(
                    f"Reading from {self.name} failed: {error}.",
                    reason=prompts.EXTERNAL_REASON_LOST,
                ) from error
            if not chunk:
                if buffer.strip():
                    yield bytes(buffer)  # a last line without its newline
                return
            buffer += chunk

    def _too_large(self) -> ExternalAgentProtocolError:
        return ExternalAgentProtocolError(
            f"{self.name} sent a line longer than {self.max_line_bytes} bytes "
            f"({_mib(self.max_line_bytes)} MiB). The connection was closed, because "
            "the request it answers cannot be known. Raise max_line_bytes if the "
            "agent legitimately sends messages this large.",
            reason=prompts.EXTERNAL_REASON_TOO_LARGE,
        )

    async def _eof_error(self) -> ExternalAgentConnectionError:
        process = self._process
        if process is None:
            return ExternalAgentConnectionError(
                f"{self.name} closed the connection.", reason=prompts.EXTERNAL_REASON_CLOSED
            )
        details = await process.explain_exit()
        return ExternalAgentConnectionError(
            f"{self.name} closed the connection: {details}",
            reason=process.exit_summary(),
            returncode=process.returncode,
            stderr_tail=process.stderr_tail() or None,
        )

    async def _handle_line(self, line: bytes) -> None:
        text = line.strip()
        if not text:
            return
        try:
            message = json.loads(text)
        except ValueError:  # includes invalid UTF-8
            self._malformed_line(line, "it is not JSON")
            return
        if isinstance(message, list):
            if not message:
                self._malformed_line(line, "it is an empty batch")
                return
            for item in message:
                await self._dispatch(item, line)
            return
        await self._dispatch(message, line)

    async def _dispatch(self, message: Any, line: bytes) -> None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            self._malformed_line(line, "it is not a JSON-RPC 2.0 message")
            return
        method = message.get("method")
        if "method" in message:
            if not isinstance(method, str):
                if "id" in message:
                    await self._answer_error(
                        message["id"], INVALID_REQUEST, prompts.EXTERNAL_RPC_INVALID_REQUEST
                    )
                self._malformed_line(line, "its method is not a string")
                return
            self._malformed_run = 0
            self.received += 1
            if "id" in message:
                self._start_handler(message["id"], method, message.get("params"))
            else:
                await self._notification(method, message.get("params"))
            return
        if "id" in message and ("result" in message) != ("error" in message):
            if not _is_response_id(message["id"]):  # an array or an object: no request has it
                self._malformed_line(line, "its id is not a string or a number")
                return
            self._malformed_run = 0
            self.received += 1
            self._resolve(message)
            return
        self._malformed_line(line, "it is neither a request, a notification nor a response")

    def _malformed_line(self, line: bytes, why: str) -> None:
        self.malformed += 1
        self._malformed_run += 1
        preview = line[:200].decode("utf-8", errors="replace")
        logger.warning("Skipped a line from %s because %s: %r", self.name, why, preview)
        if self._malformed_run >= self.max_malformed:
            raise ExternalAgentProtocolError(
                f"{self.name} sent {self._malformed_run} malformed lines in a row (the last "
                f"one: {why}). The connection was closed.",
                reason=prompts.EXTERNAL_REASON_MALFORMED,
            )

    def _resolve(self, message: dict[str, Any]) -> None:
        request_id = message["id"]
        entry = self._pending.pop(request_id, None)
        if entry is None and isinstance(request_id, str) and request_id.isdigit():
            entry = self._pending.pop(int(request_id), None)
        if entry is None:
            logger.debug("%s answered request %r, which no one waits for", self.name, request_id)
            return
        method, future = entry
        if future.done():
            return
        if "error" in message:
            error = message["error"]
            if isinstance(error, dict):
                code = error.get("code")
                future.set_exception(
                    JsonRpcError(
                        code if isinstance(code, int) else INTERNAL_ERROR,
                        str(error.get("message", "")),
                        error.get("data"),
                        method=method,
                    )
                )
            else:
                future.set_exception(JsonRpcError(INTERNAL_ERROR, str(error), method=method))
        else:
            future.set_result(message.get("result"))

    async def _notification(self, method: str, params: Any) -> None:
        if method == self.cancel_method and self.cancel_method is not None:
            request_id = params.get("requestId") if isinstance(params, Mapping) else None
            task = self._handling.get(_key(request_id))
            if task is not None and not task.done():
                self._cancelled_by_peer.add(_key(request_id))
                task.cancel()
            return
        handler = self._notifications.get(method)
        try:
            if handler is not None:
                outcome = handler(params)
            elif self._fallback is not None:
                outcome = self._fallback(method, params)
            else:
                logger.debug("No handler for notification %r from %s", method, self.name)
                return
            if inspect.isawaitable(outcome):
                await outcome
        except Exception:  # a broken handler is a bug: log it, keep the connection
            logger.exception("Handler of notification %r from %s raised", method, self.name)

    def _start_handler(self, request_id: Any, method: str, params: Any) -> None:
        key = _key(request_id)
        task = asyncio.get_running_loop().create_task(
            self._handle_request(request_id, method, params), name=f"jsonrpc:{self.name}:{method}"
        )
        self._handling[key] = task
        task.add_done_callback(lambda _: self._handling.pop(key, None))

    async def _handle_request(self, request_id: Any, method: str, params: Any) -> None:
        handler = self._requests.get(method)
        try:
            if handler is None:
                missing = prompts.EXTERNAL_RPC_METHOD_NOT_FOUND.format(method=method)
                await self._answer_error(request_id, METHOD_NOT_FOUND, missing)
                return
            try:
                result = await handler(params)
            except JsonRpcError as error:
                await self._answer(request_id, error=error.to_object())
                return
            except asyncio.CancelledError:
                if _key(request_id) not in self._cancelled_by_peer or self._closed:
                    raise
                self._cancelled_by_peer.discard(_key(request_id))
                await self._answer_error(
                    request_id, REQUEST_CANCELLED, prompts.EXTERNAL_RPC_REQUEST_CANCELLED
                )
                return
            except Exception as error:
                logger.exception("Handler of request %r from %s raised", method, self.name)
                await self._answer_error(
                    request_id,
                    INTERNAL_ERROR,
                    prompts.EXTERNAL_RPC_INTERNAL_ERROR.format(type=type(error).__name__),
                )
                return
            try:
                await self._answer(request_id, result=result)
            except (TypeError, ValueError) as error:  # the result is not JSON
                logger.error("Result of %r for %s is not JSON: %s", method, self.name, error)
                await self._answer_error(
                    request_id, INTERNAL_ERROR, prompts.EXTERNAL_RPC_BAD_RESULT
                )
        except ExternalAgentConnectionError:
            return  # closed while answering: nobody is left to answer

    async def _answer(
        self, request_id: Any, *, result: Any = None, error: dict[str, Any] | None = None
    ) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
        if error is not None:
            message["error"] = error
        else:
            message["result"] = result
        await self._send(message)

    async def _answer_error(self, request_id: Any, code: int, message: str) -> None:
        with contextlib.suppress(ExternalAgentConnectionError):
            await self._answer(request_id, error={"code": code, "message": message})

    # -- closing ---------------------------------------------------------------------------

    def _shutdown(self, error: ExternalAgentConnectionError) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_error = error
        pending, self._pending = self._pending, {}
        for request_id, (method, future) in pending.items():
            if not future.done():
                future.set_exception(error.for_request(method, request_id))
        current = asyncio.current_task()
        for task in list(self._handling.values()):
            if task is not current:
                task.cancel()
        self._closed_event.set()
        level = logging.DEBUG if error.reason == prompts.EXTERNAL_REASON_CLOSED else logging.INFO
        logger.log(level, "Connection to %s closed: %s", self.name, error)
        callbacks, self._close_callbacks = self._close_callbacks, []
        for callback in callbacks:
            try:
                callback(error)
            except Exception:
                logger.exception("A close callback of %s raised", self.name)


def _encode(message: dict[str, Any]) -> bytes:
    try:
        text = json.dumps(message, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return (text + "\n").encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate: send it escaped
        text = json.dumps(message, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        return (text + "\n").encode("ascii")


def _is_response_id(value: Any) -> bool:
    """Whether a response's ``id`` can answer a request: a string, a number or ``null``.

    ``true`` and ``false`` are refused too: as ``int`` they would match requests 1 and 0.
    """
    if isinstance(value, bool):
        return False
    return value is None or isinstance(value, int | float | str)


def _key(request_id: Any) -> Any:
    if request_id is None or isinstance(request_id, int | str):
        return request_id
    return repr(request_id)


def _mib(size: int) -> str:
    return f"{size / (1024 * 1024):g}"


def _ignore_result(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        task.exception()
