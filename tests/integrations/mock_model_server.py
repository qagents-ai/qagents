"""A local mock of the Anthropic Messages API, so the real Devvi CLI runs with no key and no cost.

Ported from Devvi's own smoke test (``scripts/smoke-python-runtime.py`` in devvi-harness:
``MockModelHandler``, ``text_chunks``, ``tool_call_chunks``, ``message_start``). Devvi's
``deepseek-official`` route (``packages/llm/llm-deepseek``) posts Anthropic Messages
requests with ``stream: true`` to ``$DEEPSEEK_BASE_URL/v1/messages`` and reads Server-Sent
Events back. This server answers them from a script:

    with MockModel() as model:
        write = shell_arguments(write_file_command("out.txt", "hello"), "Write out.txt")
        model.on_prompt("WRITE-FILE", ToolUse(SHELL, write, id="w1"))
        model.after_tool("w1", Text("Wrote out.txt."))
        agent = DevviAgent(workspace=ws, env=model.env(), env_passthrough=(),
                           model=MOCK_ROUTE, patches=[mock_models_row()], ...)

How a reply is chosen for a request, in order:

1. The text of the latest message contains a `MockModel.on_prompt` marker: that reply
   (a new prompt wins, even when Devvi puts it in one message with an earlier tool result
   or an unanswered earlier prompt; of several markers, the one that occurs last wins).
2. The latest message holds a ``tool_result``: the reply registered with
   `MockModel.after_tool` for its ``tool_use_id`` (else the default).
3. An earlier user prompt of the conversation contains a marker: that reply.
4. Otherwise ``MockModel.default`` (``Text("MOCK OK")``).

A reply is a `Text`, a `ToolUse`, a `Hold` (wait until released, then answer), a `Fail`
(an HTTP error), or a function of the `MockRequest` that returns one of these. Every
request is kept in `MockModel.requests`, so a test can check what Devvi sent its model
(the conversation, the advertised tools).

``GET /models`` (Devvi's model listing) is answered 404, so Devvi uses the models declared
in its profile (`mock_models_row`). The server listens on ``127.0.0.1`` only and runs in
a thread; it needs the standard library only.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MOCK_PROVIDER = "deepseek-official"
"""Devvi's route that speaks Anthropic Messages to ``DEEPSEEK_BASE_URL``."""

MOCK_MODEL = "smoke-model"
"""The model id the mock answers as (the name Devvi's smoke test uses)."""

MOCK_ROUTE = (MOCK_PROVIDER, MOCK_MODEL)
"""``DevviAgent(model=MOCK_ROUTE)``."""

MOCK_API_KEY = "qagents-mock-key"
"""A dummy credential: Devvi's route wants a key, the mock ignores it. Not a secret."""

SHELL = "$shell"
"""`ToolUse` name meaning Devvi's shell tool: ``pwsh`` on Windows, ``bash`` elsewhere
(``packages/bundle/base`` mounts one of them), whichever the request advertises."""

IS_WINDOWS = sys.platform == "win32"


def mock_models_row(*models: str | Mapping[str, Any]) -> dict[str, Any]:
    """Patch row declaring the mock's model(s) on the ``deepseek-official`` route.

    The row replaces ``llm-deepseek``'s whole config (``models``), as Devvi's smoke test does
    with ``{"id": "llm-deepseek", "config": {"models": [{"id": "smoke-model"}]}}``.
    """
    listed = [
        {"id": item} if isinstance(item, str) else dict(item) for item in (models or (MOCK_MODEL,))
    ]
    return {"id": "llm-deepseek", "config": {"models": listed}}


# --- replies -----------------------------------------------------------------------------------


@dataclass
class Usage:
    """Token counts the mock reports for one response (Anthropic wire names)."""

    input_tokens: int = 3
    output_tokens: int = 3
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None

    def start(self) -> dict[str, int]:
        usage = {"input_tokens": self.input_tokens, "output_tokens": 0}
        if self.cache_read_tokens is not None:
            usage["cache_read_input_tokens"] = self.cache_read_tokens
        if self.cache_write_tokens is not None:
            usage["cache_creation_input_tokens"] = self.cache_write_tokens
        return usage


@dataclass
class Text:
    """A text answer that ends the model's turn, optionally after a thinking block."""

    text: str
    thinking: str | None = None
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end_turn"


@dataclass
class ToolUse:
    """One tool call (``stop_reason: tool_use``), optionally after some text.

    ``name`` is the tool's name as Devvi advertises it, or `SHELL` for its shell tool.
    """

    name: str
    arguments: dict[str, Any]
    id: str | None = None
    text: str | None = None
    usage: Usage = field(default_factory=Usage)


@dataclass
class Hold:
    """Wait before answering: until `release` is called (or ``timeout`` seconds pass).

    While a reply is held, Devvi's turn is open: its model request is in flight.
    """

    then: Any
    timeout: float = 60.0
    started: threading.Event = field(default_factory=threading.Event)
    """Set when a request reached this reply."""
    _gate: threading.Event = field(default_factory=threading.Event)

    def release(self) -> None:
        self._gate.set()

    def wait(self) -> None:
        self.started.set()
        self._gate.wait(self.timeout)


@dataclass
class Fail:
    """An HTTP error answer (``{"type": "error", "error": {...}}``)."""

    status: int = 400
    message: str = "mock model refused the request"
    error_type: str = "invalid_request_error"


Reply = Text | ToolUse | Hold | Fail
Responder = Reply | Callable[["MockRequest"], Any]


# --- requests ----------------------------------------------------------------------------------


@dataclass
class MockRequest:
    """One ``POST /v1/messages`` body, with helpers for choosing a reply."""

    body: dict[str, Any]
    index: int

    @property
    def messages(self) -> list[dict[str, Any]]:
        messages = self.body.get("messages")
        return [m for m in messages if isinstance(m, dict)] if isinstance(messages, list) else []

    @property
    def latest(self) -> dict[str, Any]:
        """The last message that is not a system message (a system update may follow)."""
        for message in reversed(self.messages):
            if message.get("role") != "system":
                return message
        return {}

    @property
    def tools(self) -> set[str]:
        """Names of the tools the request advertises."""
        tools = self.body.get("tools")
        if not isinstance(tools, list):
            return set()
        return {t["name"] for t in tools if isinstance(t, dict) and isinstance(t.get("name"), str)}

    @property
    def shell(self) -> str:
        """Devvi's shell tool in this request: ``bash`` or ``pwsh``."""
        preferred = ("pwsh", "bash") if IS_WINDOWS else ("bash", "pwsh")
        for name in preferred:
            if name in self.tools:
                return name
        return preferred[0]

    @property
    def tool_results(self) -> list[dict[str, Any]]:
        """``tool_result`` blocks of the latest message."""
        return [
            block
            for block in _blocks(self.latest.get("content"))
            if block.get("type") == "tool_result"
        ]

    @property
    def tool_result_text(self) -> str:
        results = self.tool_results
        return message_text(results[-1].get("content")) if results else ""

    @property
    def prompts(self) -> list[str]:
        """User text blocks, latest message first."""
        found: list[str] = []
        for message in reversed(self.messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                found.append(content)
                continue
            found.extend(
                block["text"]
                for block in _blocks(content)
                if block.get("type") == "text" and isinstance(block.get("text"), str)
            )
        return found

    @property
    def prompt(self) -> str:
        """The latest user text (empty when the latest message is a tool result)."""
        if self.tool_results:
            return ""
        return message_text(self.latest.get("content"))

    @property
    def system(self) -> str:
        return message_text(self.body.get("system"))

    def transcript(self) -> str:
        """Every text of the conversation, joined (to check what history Devvi resent)."""
        parts = [message_text(m.get("content")) for m in self.messages]
        for message in self.messages:
            for block in _blocks(message.get("content")):
                if block.get("type") == "tool_result":
                    parts.append(message_text(block.get("content")))
        return "\n".join(parts)


def _blocks(content: Any) -> list[dict[str, Any]]:
    return (
        [block for block in content if isinstance(block, dict)] if isinstance(content, list) else []
    )


def message_text(content: Any) -> str:
    """Text of Messages content in string or block-list form."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in _blocks(content) if isinstance(block.get("text"), str)
    )


# --- SSE chunks (the shapes of Devvi's smoke test) --------------------------------------------------


def message_start(usage: Usage, message_id: str) -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": MOCK_MODEL,
            "content": [],
            "usage": usage.start(),
        },
    }


def text_chunks(reply: Text, message_id: str = "msg_mock") -> list[dict[str, Any]]:
    """A complete Messages text response (thinking first, when given)."""
    chunks = [message_start(reply.usage, message_id)]
    index = 0
    if reply.thinking is not None:
        chunks += [
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "thinking_delta", "thinking": reply.thinking},
            },
            {"type": "content_block_stop", "index": index},
        ]
        index += 1
    chunks += [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": reply.text},
        },
        {"type": "content_block_stop", "index": index},
        {
            "type": "message_delta",
            "delta": {"stop_reason": reply.stop_reason},
            "usage": {"output_tokens": reply.usage.output_tokens},
        },
        {"type": "message_stop"},
    ]
    return chunks


def tool_call_chunks(
    reply: ToolUse, name: str, call_id: str, message_id: str = "msg_mock"
) -> list[dict[str, Any]]:
    """A complete Messages tool-use response."""
    chunks = [message_start(reply.usage, message_id)]
    index = 0
    if reply.text:
        chunks += [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": reply.text}},
            {"type": "content_block_stop", "index": 0},
        ]  # fmt: skip
        index = 1
    chunks += [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(reply.arguments)},
        },
        {"type": "content_block_stop", "index": index},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": reply.usage.output_tokens},
        },
        {"type": "message_stop"},
    ]
    return chunks


# --- the server -------------------------------------------------------------------------------------


class MockModel:
    """The mock Messages endpoint: a scripted `ThreadingHTTPServer` on ``127.0.0.1``.

    Use it as a context manager (it starts and stops the server). `url` is the value for
    ``DEEPSEEK_BASE_URL``; `env` gives both variables Devvi's route reads.
    """

    def __init__(self, default: Responder | None = None) -> None:
        self.default: Responder = default if default is not None else Text("MOCK OK")
        self.requests: list[MockRequest] = []
        self.errors: list[str] = []
        """Requests the mock could not answer (a bug in the test's script)."""
        self._prompts: list[tuple[str, Responder]] = []
        self._followups: dict[str, Responder] = {}
        self._lock = threading.Lock()
        self._calls = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url = ""

    # -- the script ------------------------------------------------------------------

    def on_prompt(self, marker: str, reply: Responder) -> None:
        """Answer a conversation whose user text contains ``marker`` (latest prompt first)."""
        self._prompts.append((marker, reply))

    def after_tool(self, call_id: str, reply: Responder) -> None:
        """Answer the request that carries the result of tool call ``call_id``."""
        self._followups[call_id] = reply

    def choose(self, request: MockRequest) -> Responder:
        """The reply for ``request`` (see the module docstring for the order)."""
        latest = message_text(request.latest.get("content"))
        found = [(latest.rfind(marker), reply) for marker, reply in self._prompts]
        found = [(where, reply) for where, reply in found if where >= 0]
        if found:  # a new prompt, even when it also carries a tool result or an older prompt
            return max(found, key=lambda item: item[0])[1]  # the marker said last wins
        results = request.tool_results
        if results:
            call_id = results[-1].get("tool_use_id")
            if isinstance(call_id, str) and call_id in self._followups:
                return self._followups[call_id]
            return self.default
        for prompt in request.prompts[1:]:
            for marker, reply in self._prompts:
                if marker in prompt:
                    return reply
        return self.default

    # -- what Devvi sent -------------------------------------------------------------------

    def requests_with(self, marker: str) -> list[MockRequest]:
        """Requests whose conversation mentions ``marker``."""
        with self._lock:
            requests = list(self.requests)
        return [r for r in requests if marker in r.transcript()]

    def wait_for_requests(self, count: int, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.requests) >= count:
                    return
            time.sleep(0.02)
        raise TimeoutError(f"the mock model got {len(self.requests)} requests, not {count}")

    def env(self) -> dict[str, str]:
        """``DEEPSEEK_BASE_URL`` and a dummy ``DEEPSEEK_API_KEY`` for Devvi's route."""
        return {"DEEPSEEK_BASE_URL": self.url, "DEEPSEEK_API_KEY": MOCK_API_KEY}

    # -- lifecycle -----------------------------------------------------------------------------

    def start(self) -> MockModel:
        handler = type("Handler", (_Handler,), {"model": self})
        self._server = _Server(("127.0.0.1", 0), handler)
        self._server.model = self
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address[:2]
        self.url = f"http://{host}:{port}"
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = self._thread = None

    def __enter__(self) -> MockModel:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # -- answering ------------------------------------------------------------------------------

    def _record(self, body: dict[str, Any]) -> MockRequest:
        with self._lock:
            request = MockRequest(body, len(self.requests))
            self.requests.append(request)
            return request

    def _next_call_id(self) -> str:
        with self._lock:
            self._calls += 1
            return f"mock-call-{self._calls}"

    def answer(self, request: MockRequest) -> Iterator[dict[str, Any]] | Fail:
        """The SSE chunks (or the HTTP error) for ``request``."""
        reply: Any = self.choose(request)
        for _ in range(8):  # resolve functions and holds
            if isinstance(reply, Hold):
                reply.wait()
                reply = reply.then
            elif callable(reply) and not isinstance(reply, Text | ToolUse | Fail):
                reply = reply(request)
            else:
                break
        message_id = f"msg_mock_{request.index}"
        if isinstance(reply, Fail):
            return reply
        if isinstance(reply, Text):
            return iter(text_chunks(reply, message_id))
        if isinstance(reply, ToolUse):
            name = request.shell if reply.name == SHELL else reply.name
            call_id = reply.id or self._next_call_id()
            return iter(tool_call_chunks(reply, name, call_id, message_id))
        raise TypeError(f"mock model reply {reply!r} is not a Text, ToolUse, Hold or Fail")


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    model: MockModel | None = None

    def handle_error(self, request: Any, client_address: Any) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, ConnectionError):
            return  # Devvi closed a connection (a cancel, a killed process): not a failure
        if self.model is not None:
            self.model.errors.append(f"server error: {error!r}")


class _Handler(BaseHTTPRequestHandler):
    model: MockModel
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - http.server's name
        if self.path.split("?")[0].rstrip("/") != "/v1/messages":
            self._plain(404, "not found")
            return
        length = int(self.headers.get("content-length", "0"))
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            self._plain(400, "body is not JSON")
            return
        request = self.model._record(body)
        try:
            answer = self.model.answer(request)
        except Exception as error:  # a bug in the test's script: tell Devvi and the test
            self.model.errors.append(f"request {request.index}: {error!r}")
            self._json(
                500, {"type": "error", "error": {"type": "api_error", "message": str(error)}}
            )
            return
        if isinstance(answer, Fail):
            error = {"type": answer.error_type, "message": answer.message}
            self._json(answer.status, {"type": "error", "error": error})
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        try:
            for chunk in answer:
                data = json.dumps(chunk)
                self.wfile.write(f"event: {chunk['type']}\ndata: {data}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # Devvi gave up on the request (a cancel): nothing to answer
        self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        self._plain(404, "not found")  # /models: Devvi falls back to its declared models

    def _plain(self, status: int, text: str) -> None:
        data = text.encode()
        self.send_response(status)
        self.send_header("content-type", "text/plain")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, body: Mapping[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


# --- shell commands for Devvi's shell tool ------------------------------------------------------------


def shell_arguments(
    command: str, description: str = "Run a command", **extra: Any
) -> dict[str, Any]:
    """Arguments of Devvi's ``bash``/``pwsh`` tool (both need ``description``)."""
    return {"description": description, "command": command, **extra}


def write_file_command(path: str, text: str) -> str:
    """A shell command that writes ``text`` to ``path`` (relative to the workspace)."""
    if IS_WINDOWS:
        quoted = text.replace("'", "''")
        return f"Set-Content -LiteralPath '{path}' -Value '{quoted}' -NoNewline -Encoding utf8"
    quoted = text.replace("'", "'\\''")
    return f"printf '%s' '{quoted}' > '{path}'"


def sleep_command(seconds: int) -> str:
    return f"Start-Sleep -Seconds {seconds}" if IS_WINDOWS else f"sleep {seconds}"


def escalated_write_command(path: str, text: str) -> str:
    """Write a file outside the workspace (needs ``sandbox_permissions``), then print it."""
    if IS_WINDOWS:
        quoted = text.replace("'", "''")
        return (
            f"Set-Content -LiteralPath '{path}' -Value '{quoted}' -NoNewline -Encoding utf8; "
            f"Get-Content -LiteralPath '{path}'"
        )
    quoted = text.replace("'", "'\\''")
    return f"printf '%s' '{quoted}' > '{path}' && cat '{path}'"


__all__ = [
    "IS_WINDOWS",
    "MOCK_API_KEY",
    "MOCK_MODEL",
    "MOCK_PROVIDER",
    "MOCK_ROUTE",
    "SHELL",
    "Fail",
    "Hold",
    "MockModel",
    "MockRequest",
    "Text",
    "ToolUse",
    "Usage",
    "escalated_write_command",
    "message_text",
    "mock_models_row",
    "shell_arguments",
    "sleep_command",
    "text_chunks",
    "tool_call_chunks",
    "write_file_command",
]
