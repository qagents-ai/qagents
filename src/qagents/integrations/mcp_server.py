"""Serve qagents tools over MCP, so that another agent can call them.

A coding agent such as Devvi runs its own loop with its own tools. To let it use
qagents tools as well (a customer lookup, a ticket system), hand them to it as an
MCP server. MCP gives an agent tools; it does not drive the agent. qagents drives
agents over ACP or the Devvi SDK (`qagents.external`).

There are two ways to serve tools.

**In process, over Streamable HTTP** (`tool_server`). The server listens on
``127.0.0.1`` only, on a random port, and refuses every request that does not
carry its random bearer token. Give `ToolServer.url` and `ToolServer.headers` to
the agent; `ToolServer.acp_config` is the entry for ACP ``session/new``
``mcpServers``. This is the default for tools that live in your process: their
``deps``, the permission rules and the journal stay where they are.

Bound to a running qagents run (``tool_server(tools, ctx=ctx)`` inside a tool),
every call is an ordinary tool call of that run, made with ``RunContext.call_tool``:

* the run's `Permissions` decide it, with the run's ``on_ask``; "ask" without an
  ``on_ask`` is refused, because a call over MCP cannot pause the run;
* it is journaled in the run's session before and after it runs
  (``approval_decided``, ``tool_started``, ``tool_result``; the entries carry
  ``source: "mcp"``, the server name and the calling tool call's id), and it
  counts toward ``Limits.max_tool_calls`` (from before it is decided, so calls
  that arrive together never pass the limit, and a refused call does not count);
* ``Limits.parallel_tools`` keeps it apart from the other calls served through
  the same calling tool call (not from calls served through other calling calls,
  nor from the model's own calls); a call that runs alone also waits for the
  thread of an earlier sync call (served or the run's own) that timed out and
  still runs;
* the run's ``before_tool`` and ``after_tool`` hooks see it (a hook that raises
  fails the run, as for the model's calls), and its ``tool_call_started`` and
  ``tool_call_finished`` events reach the run's stream;
* the tool gets the run's ``deps`` and ``ctx.state``, and its usage is added to
  the run;
* it runs one level below the calling tool call (``ctx.depth + 1``), because it
  comes from the agent that the tool call runs. A served agent or external agent
  therefore starts at ``ctx.depth + 2``, and ``Limits.max_depth`` bounds chains
  through served tools as it bounds sub-agents.

So a qagents tool never runs without a qagents decision.

    @tool
    async def code(ctx: RunContext, task: str) -> str:
        '''Let the coding agent do the task.'''
        async with tool_server([lookup_customer, create_ticket], ctx=ctx) as server:
            return await run_coding_agent(task, mcp_servers=[server.acp_config()])

**In a separate process, over stdio**
(``python -I -m qagents.integrations.mcp_server --spec FILE``). The spec file is
JSON (`ToolServerSpec`): it lists importable tools as ``"package.module:attr"``
and the permission rules. There is no parent run here: the spec's rules decide,
"ask" is refused, and nothing is journaled. `ToolServerSpec.write`,
`stdio_command` and `acp_stdio_config` build the file and the command line.

What the calling agent sees: each tool's name, its description and the JSON
Schema of its arguments, exactly as a qagents model would, plus MCP annotations
from the tool's flags (``readOnlyHint`` for ``read_only``, ``idempotentHint`` for
``idempotent``). A result is one text block, the text a qagents model would
read. Errors (invalid arguments, a `ToolError`, a timeout, a refusal) are MCP
results with ``isError`` set, so the calling model can read them and adapt.
Refusals are worded by ``TOOL_SERVER_REFUSED`` and the other ``TOOL_SERVER_*``
strings (in `qagents.prompts`); every other text is the agent loop's own wording.

Shutting down (leaving ``async with``, `ToolServer.aclose`) refuses new calls,
cancels running ones (a bound call then records ``TOOL_INTERRUPTED``, or
``TOOL_CANCELLED`` for a read-only tool: only that is harmless when it is cut
short), ends the MCP sessions
and closes the port. A cancelled `ToolServer.aclose` still does all of that, in
the background. A bound server also refuses calls once its tool call has ended,
and writes nothing to the journal for them; a call that was already journaled as
started then records ``TOOL_SERVER_RUN_ENDED`` and does not run.

It uses the official MCP Python SDK (``qagents[mcp]``); the HTTP mode runs it on
uvicorn, which the SDK installs.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import contextlib
import dataclasses
import hmac
import importlib
import inspect
import json
import logging
import os
import re
import secrets
import socket
import sys
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

try:
    import anyio
    import mcp_types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.server.transport_security import TransportSecuritySettings
except ImportError as error:  # pragma: no cover - depends on the environment
    raise ImportError(
        "qagents.integrations.mcp_server needs the MCP Python SDK 2.x: pip install "
        f"'qagents[mcp]' (or pip install 'mcp>=2.3,<3'). Import failed with: {error}"
    ) from error

from .. import prompts
from ..agent import Agent, Limits
from ..errors import ConfigurationError, DeferToolCall, ToolArgumentError, ToolError
from ..messages import ToolCallPart, ToolResultPart
from ..permissions import Mode, Permissions
from ..tools import (
    RunContext,
    Tool,
    Toolkit,
    ToolTimeout,
    _abandoned_threads_to,
    render_tool_output,
)
from ..tools import tool as make_tool

logger = logging.getLogger("qagents.mcp_server")

SOURCE = "mcp"
"""The ``source`` of the journal entries written for calls that arrive over MCP."""

METADATA_KEY = "qagents.tool_server"
"""``ToolResultPart.metadata`` key of a served call: ``{"server", "caller_tool_call_id"}``."""

MCP_PATH = "/mcp"
"""Path of the Streamable HTTP endpoint."""

_HOST = "127.0.0.1"
_MAX_SESSIONS = 16
_SETTLE_S = 10.0
"""How long a cancelled call may take to record its result."""
_DOTTED = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_REFERENCE = re.compile(rf"^{_DOTTED}:{_DOTTED}$")
"""``package.module:attr``, where both sides may be dotted."""

# --- the tools ----------------------------------------------------------------------


async def collect_tools(items: Iterable[Any]) -> list[Tool]:
    """Turn what ``Agent(tools=...)`` accepts into a list of tools to serve.

    Accepts `Tool` objects, functions (wrapped with ``tool()``), agents (their
    ``as_tool()``), `Toolkit` objects (their tools, listed now) and lists of
    these. The toolkits stay open; closing them is up to the caller.

    Raises:
        ConfigurationError: An item is not a tool, two tools have the same name,
            or a tool is ``deferred`` (its result would come from ``agent.resume``,
            which a caller over MCP cannot wait for).
    """
    found: list[Tool] = []
    for item in _flatten(items):
        if isinstance(item, Toolkit):
            found.extend(await item.tools())
        else:
            found.append(item)
    names: dict[str, Tool] = {}
    for tool in found:
        if tool.name in names:
            raise ConfigurationError(prompts.TOOL_SERVER_DUPLICATE_NAME.format(name=tool.name))
        if tool.effects:
            raise ConfigurationError(f"Effects tool {tool.name!r} needs a fenced agent run.")
        if tool.deferred:
            raise ConfigurationError(prompts.TOOL_SERVER_DEFERRED_TOOL.format(name=tool.name))
        names[tool.name] = tool
    return found


def _flatten(items: Iterable[Any]) -> list[Tool | Toolkit]:
    flat: list[Tool | Toolkit] = []
    for item in items:
        if isinstance(item, Tool | Toolkit):
            flat.append(item)
        elif isinstance(item, Agent):
            flat.append(item.as_tool())
        elif callable(item):
            flat.append(make_tool(item))
        elif isinstance(item, Iterable) and not isinstance(item, str | bytes):
            flat.extend(_flatten(item))
        else:
            raise ConfigurationError(prompts.TOOL_SERVER_NOT_A_TOOL.format(item=item))
    return flat


def _definition(tool: Tool) -> mcp_types.Tool:
    """The MCP definition of ``tool``: what a qagents model would see, plus annotations."""
    schema = dict(tool.parameters) if isinstance(tool.parameters, Mapping) else {}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    annotations = mcp_types.ToolAnnotations(
        read_only_hint=tool.read_only,
        idempotent_hint=None if tool.read_only else tool.idempotent,
    )
    return mcp_types.Tool(
        name=tool.name,
        description=tool.description or None,
        input_schema=schema,
        annotations=annotations,
    )


# --- one call ----------------------------------------------------------------------


@dataclasses.dataclass
class _Verdict:
    call: ToolCallPart | None
    """The call to run (with an approver's edited arguments), or ``None`` when refused."""
    refusal: str | None = None
    """The result text of a refused call."""
    approval: dict[str, Any] | None = None
    """The ``approval_decided`` record, when ``on_ask`` answered."""
    edited: bool = False


async def _authorize(
    permissions: Permissions,
    ctx: RunContext[Any],
    tool: Tool,
    call: ToolCallPart,
    approval_timeout_s: float | None,
) -> _Verdict:
    """Decide one call the way the agent loop does, without pausing: "ask" needs ``on_ask``."""

    def refused(reason: str | None, approval: dict[str, Any] | None = None) -> _Verdict:
        text = prompts.TOOL_SERVER_REFUSED.format(
            name=call.name, reason=reason or prompts.PERMISSION_REASON_DEFAULT
        )
        return _Verdict(None, text, approval)

    try:
        decision = await permissions.decide(ctx, tool, call)
    except Exception:  # decide() already refuses when can_use_tool raises; this is the rest
        logger.exception("The permission check for %s raised; refusing the call", call.name)
        return refused(None)
    if decision.behavior == "deny":
        return refused(decision.reason)
    if decision.behavior == "allow":
        return _Verdict(call)
    if permissions.on_ask is None:
        return refused(prompts.TOOL_SERVER_NO_APPROVER)
    try:
        async with asyncio.timeout(approval_timeout_s):
            approval = await permissions.request_approval(ctx, tool, call, decision.reason)
    except TimeoutError:
        seconds = approval_timeout_s or 0
        return refused(prompts.TOOL_SERVER_APPROVAL_TIMEOUT.format(seconds=seconds))
    edited = approval.approved and approval.arguments not in (None, call.arguments)
    record: dict[str, Any] = {
        "tool_call_id": call.id,
        "name": call.name,
        "approved": approval.approved,
        "reason": approval.reason,
        "arguments_changed": edited,
    }
    if edited:
        record["arguments"] = approval.arguments
    if not approval.approved:
        return refused(approval.reason or prompts.APPROVAL_DECLINED, record)
    if not edited:
        return _Verdict(call, approval=record)
    assert approval.arguments is not None
    changed = call.model_copy(update={"arguments": approval.arguments, "raw_arguments": None})
    if (invalid := _invalid_arguments(tool, changed)) is not None:
        return _Verdict(None, invalid, record)
    recheck = await permissions.decide(ctx, tool, changed)
    if recheck.behavior == "deny":  # an approval answers "ask", never a deny
        return refused(recheck.reason, record)
    return _Verdict(changed, approval=record, edited=True)


def _invalid_arguments(tool: Tool, call: ToolCallPart) -> str | None:
    try:
        tool.validate(call.arguments)
    except ToolArgumentError as error:
        return prompts.INVALID_ARGUMENTS.format(name=call.name, error=error)
    return None


async def _invoke(
    ctx: RunContext[Any], tool: Tool, call: ToolCallPart, limits: Limits, gate: _Gate
) -> ToolResultPart:
    """Run the tool and turn what happens into a result, worded as the agent loop words it.

    A sync call that times out or is cancelled and still runs in its thread is handed
    to ``gate`` (before the error leaves the tool, so before the call gives its place
    back), so that later calls that run alone wait for it. A cancellation passes
    through: the caller records the call as interrupted.
    """
    metadata: dict[str, Any] = {}
    is_error = True
    try:
        with _abandoned_threads_to(lambda thread: gate.abandoned(tool, thread)):
            value = await tool.run(ctx, call.arguments, timeout=limits.tool_timeout_s)
        if isinstance(value, ToolResultPart):
            content, is_error = value.content, value.is_error
            metadata.update(value.metadata)
        else:
            content = render_tool_output(value)
            is_error = False
    except DeferToolCall:
        content = prompts.TOOL_SERVER_CANNOT_DEFER.format(name=call.name)
    except ToolArgumentError as error:
        content = prompts.INVALID_ARGUMENTS.format(name=call.name, error=error)
    except ToolError as error:
        content = str(error)
        metadata.update(getattr(error, "metadata", None) or {})
    except ToolTimeout as timeout:
        if not timeout.started:
            template = prompts.TOOL_NOT_STARTED
        elif timeout.running:
            template = prompts.TOOL_TIMEOUT_ABANDONED  # its thread went to the gate
        else:
            template = prompts.TOOL_TIMEOUT
        content = template.format(name=call.name, seconds=timeout.seconds)
        metadata["timeout"] = timeout.seconds
    except Exception as error:
        logger.exception("Served tool %s raised", call.name)
        content = prompts.TOOL_FAILED.format(
            name=call.name,
            error=prompts.TOOL_FAILED_EXCEPTION.format(type=type(error).__name__, error=error),
        )
        metadata["exception"] = type(error).__name__
    return ToolResultPart(
        tool_call_id=call.id,
        name=call.name,
        content=_shorten(content, limits.max_tool_result_chars),
        is_error=is_error,
        metadata=metadata,
    )


def _shorten(text: str, limit: int | None) -> str:
    """Keep the start and the end of an over-long text, as the agent loop does."""
    if limit is None or len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    marker = prompts.TRUNCATED.format(omitted=len(text) - limit)
    return f"{text[:head]}{marker}{text[len(text) - tail :]}"


def _error(call: ToolCallPart, content: str) -> ToolResultPart:
    return ToolResultPart(tool_call_id=call.id, name=call.name, content=content, is_error=True)


def _interrupted(tool: Tool, call: ToolCallPart) -> ToolResultPart:
    """The result of a call that was cancelled while it ran: only a read-only call is
    harmless when it is cut short; any other call's outcome is unknown (as in the loop)."""
    template = prompts.TOOL_CANCELLED if tool.read_only else prompts.TOOL_INTERRUPTED
    return _error(call, template.format(name=call.name))


def _new_call(tool: Tool, arguments: Mapping[str, Any]) -> ToolCallPart:
    return ToolCallPart(id=f"mcp-{uuid.uuid4().hex}", name=tool.name, arguments=dict(arguments))


def _waited(call: ToolCallPart, seconds: float) -> ToolResultPart:
    """The result of a call that waited its whole timeout for an abandoned thread."""
    result = _error(call, prompts.TOOL_WAITED_FOR_EARLIER.format(name=call.name, seconds=seconds))
    result.metadata["timeout"] = seconds
    return result


async def _threads_finished(
    threads: list[concurrent.futures.Future[Any]], timeout: float | None
) -> bool:
    """Wait up to ``timeout`` seconds (``None``: no limit) for ``threads`` to return."""
    waiters = [asyncio.wrap_future(thread) for thread in threads]
    try:
        _, pending = await asyncio.wait(waiters, timeout=timeout)
    finally:
        for waiter in waiters:
            if waiter.done() and not waiter.cancelled():
                waiter.exception()  # retrieved: the earlier call's error was reported already
            else:
                waiter.cancel()  # the thread already runs: cancelling its waiter cannot stop it
    return not pending


class _Gate:
    """Keeps calls apart as the agent loop does (``Limits.parallel_tools``).

    Calls that run alone (by default: tools that are not read-only) wait for
    every other call; the others run together, at most ``max_parallel_tools``
    at a time. A sync call that runs alone, timed out or was cancelled, and still
    runs in its thread is remembered: the next call that runs alone also waits for that
    thread, within its own timeout, and is not run if the thread is still busy
    then (``TOOL_WAITED_FOR_EARLIER``). A server bound to a run has no gate of its
    own: the run keeps its served calls apart (``RunContext.call_tool``), and shares
    the abandoned threads with its own calls.
    """

    def __init__(self, limits: Limits) -> None:
        self._mode = limits.parallel_tools
        self._timeout = limits.tool_timeout_s
        self._slots = asyncio.Semaphore(max(1, limits.max_parallel_tools))
        self._readers = 0
        self._writer = False
        self._changed = asyncio.Event()
        self._threads: list[concurrent.futures.Future[Any]] = []

    def _alone(self, tool: Tool) -> bool:
        return self._mode == "none" or (self._mode == "read_only" and not tool.read_only)

    def abandoned(self, tool: Tool, thread: concurrent.futures.Future[Any]) -> None:
        """Remember the thread of a call of ``tool`` that timed out or was cancelled and
        still runs (handed over by `Tool.run`, see ``_invoke``)."""
        if not thread.done() and self._alone(tool):
            self._threads.append(thread)

    async def _wait_for_stuck_threads(self, tool: Tool) -> float | None:
        """Wait for the threads of earlier calls that timed out and still run.

        Returns ``None`` when the call may start, else the seconds it waited in vain.
        """
        threads = self._threads
        if not threads:
            return None
        limit = tool.timeout if tool.timeout is not None else self._timeout
        finished = await _threads_finished(list(threads), limit or None)
        threads[:] = [thread for thread in threads if not thread.done()]  # in place: shared
        return None if finished else float(limit or 0)

    @contextlib.asynccontextmanager
    async def hold(self, tool: Tool) -> AsyncIterator[float | None]:
        """Wait for the call's turn. Yields ``None`` when the call may run; else the
        seconds it waited for an abandoned thread, and the call must not run."""
        alone = self._alone(tool)
        async with self._slots:
            while True:
                if alone and not self._writer and self._readers == 0:
                    self._writer = True
                    break
                if not alone and not self._writer:
                    self._readers += 1
                    break
                self._changed.clear()
                await self._changed.wait()
            try:
                yield await self._wait_for_stuck_threads(tool) if alone else None
            finally:
                if alone:
                    self._writer = False
                else:
                    self._readers -= 1
                self._changed.set()


class _Calls(ABC):
    """Runs the calls that arrive over MCP: on their own, or as tool calls of a run."""

    def __init__(self, server: str, approval_timeout_s: float | None) -> None:
        self.server = server
        self.approval_timeout_s = approval_timeout_s
        self.closing = False

    @abstractmethod
    async def call(self, tool: Tool, arguments: Mapping[str, Any]) -> ToolResultPart:
        """Decide, run and record one call; return its result."""


class _LocalCalls(_Calls):
    """Calls with no parent run: the server's own permission rules, deps and limits."""

    def __init__(
        self,
        server: str,
        *,
        permissions: Permissions,
        deps: Any,
        limits: Limits,
        approval_timeout_s: float | None,
    ) -> None:
        super().__init__(server, approval_timeout_s)
        self.permissions = permissions
        self.deps = deps
        self.limits = limits
        self.run_id = f"mcp-server-{uuid.uuid4().hex}"
        self.state: dict[str, Any] = {}
        self.counted = 0
        """Calls that count toward ``Limits.max_tool_calls``, with those being decided."""
        self.gate = _Gate(limits)

    async def call(self, tool: Tool, arguments: Mapping[str, Any]) -> ToolResultPart:
        call = _new_call(tool, arguments)
        ctx: RunContext[Any] = RunContext(
            deps=self.deps,
            run_id=self.run_id,
            agent_name=self.server,
            tool_call_id=call.id,
            operation_id=f"{self.run_id}:{call.id}",
            state=self.state,
        )
        if self.closing:
            return _error(call, _refusal(call, prompts.TOOL_SERVER_CLOSING))
        if (invalid := _invalid_arguments(tool, call)) is not None:
            return _error(call, invalid)
        budget = self.limits.max_tool_calls
        if budget is not None and self.counted >= budget:
            return _error(call, prompts.TOOL_CALL_LIMIT.format(name=call.name, limit=budget))
        # Take the slot before the call is decided, with no await in between: calls that
        # arrive together cannot all pass the check, and nobody is asked to approve a call
        # that could not run. A refused call gives its slot back.
        self.counted += 1
        verdict = _Verdict(None)
        try:
            verdict = await _authorize(self.permissions, ctx, tool, call, self.approval_timeout_s)
        finally:
            if verdict.call is None:
                self.counted -= 1
        if verdict.call is None:
            return _error(call, verdict.refusal or prompts.PERMISSION_REASON_DEFAULT)
        async with self.gate.hold(tool) as waited:
            if waited is not None:  # an earlier call's thread still runs: this one did not
                return _waited(call, waited)
            if self.closing:  # it waited for an earlier call, and the server began to close
                return _error(call, _refusal(call, prompts.TOOL_SERVER_CLOSING))
            result = await _invoke(ctx, tool, verdict.call, self.limits, self.gate)
        return _note_edit(result, verdict)


def _refusal(call: ToolCallPart, reason: str) -> str:
    return prompts.TOOL_SERVER_REFUSED.format(name=call.name, reason=reason)


def _note_edit(result: ToolResultPart, verdict: _Verdict) -> ToolResultPart:
    """Tell the caller which arguments ran when an approver changed them."""
    if not verdict.edited or verdict.call is None:
        return result
    note = prompts.APPROVAL_ARGUMENTS_CHANGED.format(
        arguments=json.dumps(verdict.call.arguments, ensure_ascii=False, default=str)
    )
    content = prompts.APPROVAL_ARGUMENTS_CHANGED_RESULT.format(note=note, content=result.content)
    return result.model_copy(update={"content": content})


class _BoundCalls(_Calls):
    """Calls as ordinary tool calls of the run that ``origin`` (a running tool call) belongs
    to, through ``RunContext.call_tool``: the run decides, journals, counts, hooks and
    runs them as it does the model's calls (see the module docstring)."""

    def __init__(
        self, origin: RunContext[Any], server: str, approval_timeout_s: float | None
    ) -> None:
        super().__init__(server, approval_timeout_s)
        if origin.tool_call_id is None or origin.limits is None:
            raise ConfigurationError(
                "tool_server(ctx=...) needs the RunContext of a running tool call: the one a "
                "tool receives as its first parameter."
            )
        if origin.permissions is None:
            raise ConfigurationError(
                "tool_server(ctx=...) could not find the permission rules of the run, so it "
                "would not be able to decide calls. Pass the RunContext a tool received."
            )
        self.origin = origin

    async def call(self, tool: Tool, arguments: Mapping[str, Any]) -> ToolResultPart:
        if self.closing:  # refused before the run sees it: nothing is journaled
            call = _new_call(tool, arguments)
            return _error(call, _refusal(call, prompts.TOOL_SERVER_CLOSING))
        return await self.origin.call_tool(
            tool,
            arguments,
            source=SOURCE,
            approval_timeout_s=self.approval_timeout_s,
            tags={"server": self.server},
            metadata={
                METADATA_KEY: {
                    "server": self.server,
                    "caller_tool_call_id": self.origin.tool_call_id,
                }
            },
        )


# --- the MCP server ------------------------------------------------------------------


class _Dispatcher:
    """The MCP side: lists the tools, runs each call in a task of its own, maps results."""

    def __init__(self, name: str, tools: Sequence[Tool], calls: _Calls) -> None:
        self.name = name
        self.tools = {tool.name: tool for tool in tools}
        self.calls = calls
        self.inflight: set[asyncio.Task[ToolResultPart]] = set()
        self.definitions = [_definition(tool) for tool in tools]

    def server(self) -> Server[Any]:
        async def list_tools(
            ctx: Any, params: mcp_types.PaginatedRequestParams | None
        ) -> mcp_types.ListToolsResult:
            return mcp_types.ListToolsResult(tools=list(self.definitions))

        async def call_tool(
            ctx: Any, params: mcp_types.CallToolRequestParams
        ) -> mcp_types.CallToolResult:
            return await self.handle(params.name, params.arguments or {})

        def input_schema(name: str) -> Mapping[str, Any] | None:
            tool = self.tools.get(name)
            return None if tool is None else _definition(tool).input_schema

        return Server(
            self.name,
            version=_version(),
            on_list_tools=list_tools,
            on_call_tool=call_tool,
            get_tool_input_schema=input_schema,
        )

    async def handle(self, name: str, arguments: Mapping[str, Any]) -> mcp_types.CallToolResult:
        tool = self.tools.get(name)
        if tool is None:
            available = prompts.UNKNOWN_TOOL_SEPARATOR.join(self.tools) or prompts.UNKNOWN_TOOL_NONE
            return _mcp_result(prompts.UNKNOWN_TOOL.format(name=name, available=available), True)
        # The call runs in a plain asyncio task, outside the SDK's cancel scopes, so that a
        # cancelled request can still journal its result.
        task = asyncio.get_running_loop().create_task(self.calls.call(tool, arguments))
        self.inflight.add(task)
        task.add_done_callback(self.inflight.discard)
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if task.cancelled() and current is not None and current.cancelling() == 0:
                # The server cancelled the call (it is closing); the request itself goes on,
                # so it gets an answer. The call recorded its own result when it stopped.
                stopped = _interrupted(tool, ToolCallPart(id="", name=name))
                return _mcp_result(stopped.content, True)
            task.cancel()  # the client cancelled the request, or the transport stops
            with anyio.CancelScope(shield=True):
                await asyncio.wait({task}, timeout=_SETTLE_S)
            raise
        except Exception as error:
            logger.exception("The served call to %s failed", name)
            text = prompts.TOOL_FAILED.format(name=name, error=type(error).__name__)
            return _mcp_result(text, True)
        return _mcp_result(result.content, result.is_error)

    async def cancel_inflight(self, timeout: float) -> None:
        """Cancel the calls that still run and wait (up to ``timeout``) for them to record."""
        tasks = set(self.inflight)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)


def _mcp_result(text: str, is_error: bool) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=text)], is_error=is_error
    )


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("qagents")
    except Exception:  # pragma: no cover - running from a source tree
        return "0"


# --- Streamable HTTP, in process ---------------------------------------------------------


class ToolServer:
    """qagents tools served over MCP Streamable HTTP on ``127.0.0.1``. Build one with `tool_server`.

    Use it as ``async with``; `start` and `aclose` do the same by hand. While it
    runs, `url` and `headers` (the bearer token) are what the calling agent needs.
    """

    def __init__(
        self,
        tools: Iterable[Any],
        *,
        ctx: RunContext[Any] | None = None,
        permissions: Permissions | None = None,
        deps: Any = None,
        limits: Limits | None = None,
        name: str = "qagents",
        approval_timeout_s: float | None = 60.0,
        shutdown_timeout_s: float = 10.0,
    ) -> None:
        """Configure the server; nothing listens until `start`.

        Args:
            tools: What ``Agent(tools=...)`` accepts: tools, functions, agents,
                toolkits and lists of them.
            ctx: The `RunContext` of the running tool call to bind to. Every call
                then is a tool call of that run (see the module docstring), at
                ``ctx.depth + 1``, and ``permissions``, ``deps`` and ``limits`` come
                from the run.
            permissions: Without ``ctx``: the rules that decide calls. Default
                ``Permissions()``: tools declared ``requires_approval`` are refused
                unless an ``allow`` rule or an ``on_ask`` lets them run.
            deps: Without ``ctx``: ``ctx.deps`` for the tools.
            limits: Without ``ctx``: ``tool_timeout_s``, ``max_tool_result_chars``,
                ``max_tool_calls``, ``parallel_tools`` and ``max_parallel_tools``
                apply. Default ``Limits()``.
            name: The MCP server name the calling agent sees.
            approval_timeout_s: How long ``on_ask`` may take before the call is
                refused; ``None`` waits for as long as the caller does.
            shutdown_timeout_s: How long `aclose` waits for cancelled calls to
                record their results, and then for the HTTP server to stop.

        Raises:
            ConfigurationError: ``ctx`` together with ``permissions``, ``deps`` or
                ``limits``.
        """
        if ctx is not None and (permissions is not None or deps is not None or limits is not None):
            raise ConfigurationError(
                "tool_server(ctx=...) uses the permissions, deps and limits of the run it is "
                "bound to; do not pass permissions=, deps= or limits= as well."
            )
        self.name = name
        self._sources = list(tools)
        self._ctx = ctx
        self._permissions = permissions if permissions is not None else Permissions()
        self._deps = deps
        self._limits = limits or Limits()
        self._approval_timeout_s = approval_timeout_s
        self._shutdown_timeout_s = shutdown_timeout_s
        self._token = secrets.token_urlsafe(32)
        self._port: int | None = None
        self._dispatcher: _Dispatcher | None = None
        self._owner: asyncio.Task[None] | None = None
        self._uvicorn: Any = None
        self._socket: socket.socket | None = None
        self._ready: asyncio.Event | None = None
        self._stop: asyncio.Event | None = None
        self._shutdown: asyncio.Task[None] | None = None
        """The shutdown, in a task of its own: a cancelled `aclose` does not stop it."""
        self._closing = False
        self._closed = False

    @property
    def bound(self) -> bool:
        """True when calls are tool calls of a run (``ctx=`` was given)."""
        return self._ctx is not None

    @property
    def tools(self) -> list[Tool]:
        """The served tools (known once started)."""
        return list(self._dispatcher.tools.values()) if self._dispatcher is not None else []

    @property
    def running(self) -> bool:
        """True between `start` and `aclose`."""
        return self._dispatcher is not None and not self._closing

    @property
    def port(self) -> int:
        """The TCP port on ``127.0.0.1``."""
        if self._port is None:
            raise RuntimeError("The tool server is not started: use `async with tool_server(...)`.")
        return self._port

    @property
    def url(self) -> str:
        """The MCP endpoint, ``http://127.0.0.1:<port>/mcp``."""
        return f"http://{_HOST}:{self.port}{MCP_PATH}"

    @property
    def headers(self) -> dict[str, str]:
        """Headers every request must carry: ``{"Authorization": "Bearer <token>"}``.

        The token is random and new for every server. Pass it only to the agent
        that should call the tools; never log it.
        """
        return {"Authorization": f"Bearer {self._token}"}

    def acp_config(self, name: str | None = None) -> dict[str, Any]:
        """This server as an entry of ACP ``session/new`` ``mcpServers``.

        ``{"type": "http", "name", "url", "headers": [{"name", "value"}]}``; the
        agent's tools are then named ``mcp__<name>__<tool>`` (Devvi).
        """
        return {
            "type": "http",
            "name": name or self.name,
            "url": self.url,
            "headers": [{"name": key, "value": value} for key, value in self.headers.items()],
        }

    async def start(self) -> Self:
        """List the tools, bind ``127.0.0.1`` on a random port and start serving.

        Raises:
            ConfigurationError: A tool cannot be served, or ``ctx`` is not the
                context of a running tool call.
            RuntimeError: The server was started before.
        """
        if self._dispatcher is not None or self._closed:
            raise RuntimeError("A ToolServer can be started once; build a new one.")
        import uvicorn

        tools = await collect_tools(self._sources)
        calls: _Calls
        if self._ctx is not None:
            calls = _BoundCalls(self._ctx, self.name, self._approval_timeout_s)
        else:
            calls = _LocalCalls(
                self.name,
                permissions=self._permissions,
                deps=self._deps,
                limits=self._limits,
                approval_timeout_s=self._approval_timeout_s,
            )
        dispatcher = _Dispatcher(self.name, tools, calls)
        sock = _loopback_socket()
        port = sock.getsockname()[1]
        server = dispatcher.server()
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[f"{_HOST}:{port}"],
            allowed_origins=[f"http://{_HOST}:{port}"],
        )
        app = server.streamable_http_app(
            streamable_http_path=MCP_PATH,
            host=_HOST,
            transport_security=security,
            max_sessions=_MAX_SESSIONS,
        )
        config = uvicorn.Config(
            _BearerGate(app, self._token, lambda: self._closing),
            lifespan="off",  # the session manager is run by _own, below
            log_config=None,  # leave the application's logging alone
            log_level=None,
            http="h11",
            ws="none",
            proxy_headers=False,
            server_header=False,
            date_header=False,
            timeout_graceful_shutdown=2,
        )
        self._uvicorn = _quiet_uvicorn(config)
        self._socket, self._port, self._dispatcher = sock, port, dispatcher
        self._ready, self._stop = asyncio.Event(), asyncio.Event()
        self._owner = asyncio.get_running_loop().create_task(
            self._own(server, sock), name=f"qagents-tool-server:{self.name}"
        )
        ready = asyncio.ensure_future(self._ready.wait())
        try:
            await asyncio.wait({ready, self._owner}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:  # cancelled while starting: leave nothing running
            await asyncio.shield(self.aclose())
            raise
        finally:
            ready.cancel()
        if not self._ready.is_set():
            await self.aclose()
            error = self._owner.exception() if not self._owner.cancelled() else None
            raise RuntimeError(f"The tool server did not start: {error!r}") from error
        logger.info("Serving %d qagents tool(s) over MCP on %s", len(tools), self.url)
        return self

    async def _own(self, server: Server[Any], sock: socket.socket) -> None:
        """Run the session manager and uvicorn; the session manager's task group lives here."""
        assert self._ready is not None and self._stop is not None
        uv = self._uvicorn
        serve: asyncio.Task[None] | None = None
        try:
            async with server.session_manager.run():
                serve = asyncio.get_running_loop().create_task(uv.serve(sockets=[sock]))
                stop = asyncio.ensure_future(self._stop.wait())
                started = asyncio.ensure_future(uv.started_event.wait())
                try:
                    await asyncio.wait({serve, started}, return_when=asyncio.FIRST_COMPLETED)
                    if started.done():
                        self._ready.set()
                        await asyncio.wait({serve, stop}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    stop.cancel()
                    started.cancel()
                # Leaving the session manager ends every MCP session, so open streams close.
        finally:
            if serve is not None:
                uv.should_exit = True
                done, _ = await asyncio.wait({serve}, timeout=self._shutdown_timeout_s)
                if not done:
                    uv.force_exit = True
                    serve.cancel()
                    await asyncio.wait({serve}, timeout=self._shutdown_timeout_s)
                if serve.done() and not serve.cancelled() and serve.exception() is not None:
                    raise serve.exception()  # type: ignore[misc]

    def refuse_new_calls(self) -> None:
        """Refuse every call that arrives from now on (``TOOL_SERVER_CLOSING``; nothing is
        journaled for it), and keep serving everything else until `aclose`: the MCP
        sessions stay up and calls that already run go on.

        For a caller that must stop the agent's work before it can close the server, such
        as an abandoned turn whose agent is still being told to close its session. Does
        nothing before `start`; `aclose` refuses new calls too.
        """
        if self._dispatcher is not None:
            self._dispatcher.calls.closing = True

    async def aclose(self) -> None:
        """Stop serving: refuse new calls, cancel running ones (they record their results),
        end the MCP sessions and close the port.

        Safe to call twice: a second call waits for the same shutdown. Safe to cancel
        (a deadline around ``async with``): the shutdown goes on in a task of its own,
        and the HTTP server begins to stop at once, without waiting any longer for
        the cancelled calls to record their results.
        """
        task = self._shutdown
        if task is None:
            self._closed = self._closing = True  # from now on, new calls are refused
            if self._dispatcher is not None:
                self._dispatcher.calls.closing = True
            task = self._shutdown = asyncio.get_running_loop().create_task(
                self._shut_down(), name=f"qagents-tool-server-close:{self.name}"
            )
        if task.done():
            return
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if self._stop is not None:
                self._stop.set()  # stop serving now; the shutdown task closes the port
            raise

    async def _shut_down(self) -> None:
        """Cancel the running calls, then stop the HTTP server and close the port."""
        owner = self._owner
        try:
            if self._dispatcher is not None:
                await self._dispatcher.cancel_inflight(self._shutdown_timeout_s)
        finally:
            if self._stop is not None:
                self._stop.set()  # also when this task is cancelled: the loop is closing
        if owner is not None:
            done, _ = await asyncio.wait({owner}, timeout=self._shutdown_timeout_s * 2 + 5)
            if not done:
                owner.cancel()
                await asyncio.wait({owner}, timeout=self._shutdown_timeout_s)
            if owner.done() and not owner.cancelled() and owner.exception() is not None:
                logger.warning("The tool server stopped with an error: %r", owner.exception())
        if self._socket is not None:
            self._socket.close()
        logger.info("The qagents tool server %r stopped", self.name)

    async def __aenter__(self) -> Self:
        return await self.start()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        where = f"port={self._port}" if self._port is not None else "not started"
        return f"ToolServer(name={self.name!r}, {where}, bound={self.bound})"


def tool_server(
    tools: Iterable[Any],
    *,
    ctx: RunContext[Any] | None = None,
    permissions: Permissions | None = None,
    deps: Any = None,
    limits: Limits | None = None,
    name: str = "qagents",
    approval_timeout_s: float | None = 60.0,
    shutdown_timeout_s: float = 10.0,
) -> ToolServer:
    """Serve ``tools`` over MCP Streamable HTTP on ``127.0.0.1``, with a random bearer token.

    ``async with tool_server(tools, ctx=ctx) as server`` inside a tool binds the
    server to that tool call's run: every call goes through the run's permission
    rules and is journaled as one of its tool calls. Without ``ctx`` the server
    decides calls with ``permissions`` and journals nothing. The arguments are
    those of `ToolServer`.

    Returns:
        A `ToolServer`, not started yet: use it with ``async with``.
    """
    return ToolServer(
        tools,
        ctx=ctx,
        permissions=permissions,
        deps=deps,
        limits=limits,
        name=name,
        approval_timeout_s=approval_timeout_s,
        shutdown_timeout_s=shutdown_timeout_s,
    )


class _BearerGate:
    """ASGI wrapper: only requests with the server's bearer token reach the MCP app."""

    def __init__(self, app: Any, token: str, closing: Callable[[], bool]) -> None:
        self._app = app
        self._expected = f"bearer {token}".encode("ascii")
        self._closing = closing

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            return  # no websockets, and the lifespan is not run through the app
        given = b""
        for key, value in scope.get("headers") or ():
            if key == b"authorization":
                given = value
                break
        scheme, _, token = given.partition(b" ")
        candidate = scheme.lower() + b" " + token.strip()
        if not hmac.compare_digest(candidate, self._expected):
            await _respond(send, 401, b"Unauthorized", [(b"www-authenticate", b"Bearer")])
            return
        if self._closing():
            await _respond(send, 503, b"Shutting down", [])
            return
        await self._app(scope, receive, send)


async def _respond(send: Any, status: int, body: bytes, headers: list[tuple[bytes, bytes]]) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"text/plain; charset=utf-8"), *headers],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _loopback_socket() -> socket.socket:
    """A TCP socket bound to 127.0.0.1 on a free port, which no other process can take over."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)  # Windows: no port sharing
        if exclusive is not None:
            sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        sock.bind((_HOST, 0))
    except BaseException:
        sock.close()
        raise
    return sock


_QUIET_UVICORN: type | None = None


def _quiet_uvicorn(config: Any) -> Any:
    """A uvicorn server that leaves the process's signal handlers alone and reports start-up."""
    global _QUIET_UVICORN
    if _QUIET_UVICORN is None:
        import uvicorn

        class QuietServer(uvicorn.Server):
            def __init__(self, config: uvicorn.Config) -> None:
                super().__init__(config)
                self.started_event = asyncio.Event()

            @contextlib.contextmanager
            def capture_signals(self) -> Any:
                yield  # Ctrl+C stays the application's: the server is stopped by aclose()

            async def startup(self, sockets: list[socket.socket] | None = None) -> None:
                await super().startup(sockets=sockets)
                self.started_event.set()

        _QUIET_UVICORN = QuietServer
    return _QUIET_UVICORN(config)


# --- stdio, in a separate process ------------------------------------------------------------


class ToolRules(BaseModel):
    """The permission rules of a `ToolServerSpec`: those of `Permissions`, without callbacks."""

    model_config = ConfigDict(extra="forbid")

    mode: Mode = "default"
    allow: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)
    """Patterns that ask. With no one to answer over stdio, such calls are refused."""
    deny: list[str] = Field(default_factory=list)

    def build(self) -> Permissions:
        return Permissions(mode=self.mode, allow=self.allow, ask=self.ask, deny=self.deny)


class ToolServerSpec(BaseModel):
    """What ``python -I -m qagents.integrations.mcp_server --spec FILE`` serves (JSON).

    ```json
    {
      "name": "qagents",
      "tools": ["my_app.tools:lookup_customer", "my_app.tools:TICKET_TOOLS"],
      "permissions": {"allow": ["create_ticket"], "deny": ["delete_*"]},
      "deps": "my_app.tools:make_deps"
    }
    ```

    The process runs with ``-I`` (no current directory, no ``PYTHONPATH``, no
    user site-packages on ``sys.path``), so the tools must be importable from the
    environment, or from a directory listed in ``paths``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "qagents"
    """The MCP server name the calling agent sees."""
    tools: list[str]
    """``"package.module:attr"`` references: a tool, a function, an agent, a toolkit or a
    list of them (see `collect_tools`)."""
    permissions: ToolRules = Field(default_factory=ToolRules)
    paths: list[str] = Field(default_factory=list)
    """Absolute directories put first on ``sys.path`` before the tools are imported."""
    deps: str | None = None
    """``"package.module:attr"`` of ``ctx.deps``: a value, or a function (sync or async)
    called once, with no arguments, when the server starts."""
    tool_timeout_s: float | None = 120.0
    max_tool_result_chars: int | None = 60_000
    max_tool_calls: int | None = None
    """Calls this server runs before it refuses more (for the process's lifetime)."""

    @field_validator("tools")
    @classmethod
    def _check_tools(cls, value: list[str]) -> list[str]:
        for reference in value:
            if not _REFERENCE.match(reference):
                raise ValueError(f"{reference!r} is not a 'package.module:attr' reference")
        return value

    @field_validator("deps")
    @classmethod
    def _check_deps(cls, value: str | None) -> str | None:
        if value is not None and not _REFERENCE.match(value):
            raise ValueError(f"{value!r} is not a 'package.module:attr' reference")
        return value

    @field_validator("paths")
    @classmethod
    def _check_paths(cls, value: list[str]) -> list[str]:
        for path in value:
            if not Path(path).is_absolute():
                raise ValueError(f"{path!r} is not an absolute path")
        return value

    @classmethod
    def load(cls, path: str | Path) -> ToolServerSpec:
        """Read a spec file.

        Raises:
            ConfigurationError: The file cannot be read or is not a valid spec.
        """
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as error:
            raise ConfigurationError(f"Cannot read the tool server spec {path}: {error}") from error
        try:
            return cls.model_validate(json.loads(text))
        except (ValueError, ValidationError) as error:
            raise ConfigurationError(
                f"The tool server spec {path} is not valid: {error}"
            ) from error

    def write(self, path: str | Path) -> Path:
        """Write the spec as JSON; return the absolute path."""
        target = Path(path).resolve()
        target.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return target

    async def resolve(self) -> tuple[list[Tool], list[Toolkit], Any]:
        """Import the tools and the deps: ``(tools, toolkits to close, deps)``.

        Raises:
            ConfigurationError: A path is missing, a reference cannot be imported,
                or what it names cannot be served.
        """
        _add_import_paths(self.paths)
        sources = _flatten(load_object(reference) for reference in self.tools)
        toolkits = [item for item in sources if isinstance(item, Toolkit)]
        tools = await collect_tools(sources)
        deps: Any = None
        if self.deps is not None:
            deps = load_object(self.deps)
            if callable(deps):
                deps = deps()
            if inspect.isawaitable(deps):
                deps = await deps
        return tools, toolkits, deps

    def limits(self) -> Limits:
        return Limits(
            tool_timeout_s=self.tool_timeout_s,
            max_tool_result_chars=self.max_tool_result_chars,
            max_tool_calls=self.max_tool_calls,
        )


def _add_import_paths(paths: Sequence[str]) -> None:
    """Put ``paths`` first on ``sys.path``, in their order."""
    for entry in reversed(paths):
        if not Path(entry).is_dir():
            raise ConfigurationError(
                f"The spec lists the path {entry!r}, which is not a directory."
            )
        if entry not in sys.path:
            sys.path.insert(0, entry)


def load_object(reference: str) -> Any:
    """Import ``"package.module:attr"`` (``attr`` may be dotted) and return the object.

    Raises:
        ConfigurationError: The reference is malformed or cannot be imported.
    """
    if not _REFERENCE.match(reference):
        raise ConfigurationError(f"{reference!r} is not a 'package.module:attr' reference.")
    module_name, _, attribute = reference.partition(":")
    try:
        value: Any = importlib.import_module(module_name)
    except Exception as error:
        raise ConfigurationError(
            f"Cannot import {module_name!r} for {reference!r}: {type(error).__name__}: {error}"
        ) from error
    for part in attribute.split("."):
        try:
            value = getattr(value, part)
        except AttributeError as error:
            raise ConfigurationError(f"{reference!r}: {part!r} not found.") from error
    return value


def tool_reference(tool: Tool) -> str:
    """The ``"package.module:attr"`` that names ``tool`` in its module, for a spec file.

    Raises:
        ConfigurationError: The tool is not a module-level name of an importable
            module (it was built inside a function, or lives in ``__main__``).
    """
    function = getattr(tool, "function", None)
    module_name = getattr(function, "__module__", None)
    if not module_name or module_name == "__main__":
        raise ConfigurationError(
            f"Tool {tool.name!r} is not importable by another process: define it at module "
            "level in an importable module, or serve it in process with tool_server()."
        )
    module = sys.modules.get(module_name) or importlib.import_module(module_name)
    candidates = [getattr(function, "__name__", None), tool.name]
    for name in candidates:
        if name and getattr(module, name, None) is tool:
            return f"{module_name}:{name}"
    for name, value in vars(module).items():
        if value is tool:
            return f"{module_name}:{name}"
    raise ConfigurationError(
        f"Tool {tool.name!r} is not a module-level name of {module_name!r}, so another process "
        "cannot import it. Assign it to a module-level name, or serve it with tool_server()."
    )


def stdio_command(spec_file: str | Path) -> list[str]:
    """The command line of the stdio server for ``spec_file``, with this interpreter.

    The interpreter path is made absolute but not resolved: a virtual environment's
    ``python`` is often a link to the base interpreter, which would lose the
    environment's packages.
    """
    return [
        os.path.abspath(sys.executable),
        "-I",
        "-m",
        "qagents.integrations.mcp_server",
        "--spec",
        str(Path(spec_file).resolve()),
    ]


def acp_stdio_config(
    spec_file: str | Path, *, name: str = "qagents", env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """The stdio server as an entry of ACP ``session/new`` ``mcpServers``.

    ``{"name", "command" (absolute), "args", "env": [{"name", "value"}]}``. Put no
    secret in ``env``: the agent may log its configuration, and an agent may drop
    credential-named variables before it starts the server (Devvi drops names that
    match ``KEY|PASSWORD|SECRET|TOKEN``). Tools that need a credential load it in a
    ``deps`` factory that reads a file or a credential store.
    """
    command = stdio_command(spec_file)
    return {
        "name": name,
        "command": command[0],
        "args": command[1:],
        "env": [{"name": key, "value": value} for key, value in (env or {}).items()],
    }


async def serve_stdio(
    tools: Iterable[Any],
    *,
    permissions: Permissions | None = None,
    deps: Any = None,
    limits: Limits | None = None,
    name: str = "qagents",
    approval_timeout_s: float | None = 60.0,
) -> None:
    """Serve ``tools`` over MCP on this process's stdin and stdout until stdin closes.

    While it serves, stray output of the tools (``print``) goes to stderr, not to
    the protocol stream. Calls still running when stdin closes are cancelled.
    """
    collected = await collect_tools(tools)
    calls = _LocalCalls(
        name,
        permissions=permissions if permissions is not None else Permissions(),
        deps=deps,
        limits=limits or Limits(),
        approval_timeout_s=approval_timeout_s,
    )
    dispatcher = _Dispatcher(name, collected, calls)
    server = dispatcher.server()
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        calls.closing = True
        await dispatcher.cancel_inflight(_SETTLE_S)


async def _serve_spec(spec: ToolServerSpec) -> None:
    tools, toolkits, deps = await spec.resolve()
    try:
        await serve_stdio(
            tools,
            permissions=spec.permissions.build(),
            deps=deps,
            limits=spec.limits(),
            name=spec.name,
        )
    finally:
        for toolkit in toolkits:
            with contextlib.suppress(Exception):
                await toolkit.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    """``python -I -m qagents.integrations.mcp_server --spec FILE``: serve over stdio.

    Exit codes: 0 when stdin closed, 2 for a bad spec or a tool that cannot be
    served, 1 for any other failure. Messages go to stderr.
    """
    parser = argparse.ArgumentParser(
        prog="python -I -m qagents.integrations.mcp_server",
        description="Serve qagents tools over MCP on stdin and stdout.",
    )
    parser.add_argument("--spec", required=True, help="JSON file listing the tools to serve.")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level on stderr (default WARNING).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr, level=args.log_level, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        spec = ToolServerSpec.load(args.spec)
        asyncio.run(_serve_spec(spec))
    except ConfigurationError as error:
        print(f"qagents MCP server: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        logger.exception("The qagents MCP server failed")
        print(f"qagents MCP server failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "MCP_PATH",
    "METADATA_KEY",
    "SOURCE",
    "ToolRules",
    "ToolServer",
    "ToolServerSpec",
    "acp_stdio_config",
    "collect_tools",
    "load_object",
    "main",
    "serve_stdio",
    "stdio_command",
    "tool_reference",
    "tool_server",
]


if __name__ == "__main__":
    # Run through the module's real name, so that tools importing it share one copy.
    from qagents.integrations.mcp_server import main as _main

    sys.exit(_main())
