"""OpenTelemetry tracing, as a hook.


    from qagents.tracing import OTelHooks

    agent = Agent("qcloud:<model-id>", hooks=[OTelHooks()])

It uses whatever tracer provider your application configured, so the spans go
wherever your other spans go (any OTLP backend: Langfuse, Phoenix, LangSmith,
Logfire, Grafana, Datadog, or an MLflow server's ``/v1/traces`` endpoint). One
run produces:

    invoke_agent <agent>
      chat <model>            one per model request
      execute_tool <tool>     one per tool call

`OTelHooks` propagates to child runs. A sub-agent's run (an agent used as a
tool, or any run started with ``parent=ctx``) nests its ``invoke_agent`` span
under the ``execute_tool`` span of the call that started it, so a multi-agent
run is one trace:

    invoke_agent lead
      chat <model>
      execute_tool researcher
        invoke_agent researcher
          chat <model>
          execute_tool search

Attach the hook to the top agent only, or to every agent: a hook object is
used once per run, so the same instance on the parent and the child does not
produce two sets of spans. A top-level run nests under the span that is
current when it starts, as any instrumented library would.

Attribute names follow the OpenTelemetry GenAI semantic conventions. Those
are still marked "Development" upstream and may be renamed; the names used
here are kept in one place (``ATTR``) so they can be adjusted.

Message and tool content is left out of spans unless ``record_content=True``,
because it often contains personal data. That includes error text: a failed
tool call gets the status description `TOOL_FAILED`, and a failed run records
only the exception's type (no message, no stack trace), because exception
messages often quote the model's answer or a tool's output. Approval tokens
are never recorded.

A compactor's own model request (the summary of `SummarizingCompactor`) goes
through the run's hooks, so it has a ``chat`` span of its own.

An external agent used as a tool (`ExternalAgent.as_tool`: Devvi, an ACP agent)
runs its own loop, so its model requests and tools get no spans of their own.
Its permission decisions and the end of its turn are span events on the
``execute_tool`` span of the call that ran it (on the run span when the event
names no such call), so a trace shows who approved what and how the turn ended:

    execute_tool coder
      event qagents.external.permission_decided   coder.bash: allow, by on_ask
      event qagents.external.turn_finished        completed, end_turn, tokens

A permission event has the tool's name, the answer (``outcome``, ``approved``),
what the rules said and who decided; the arguments and the reason only with
``record_content``. A turn event has the status, the session, the backend, the
token usage and the number of tool calls and changed files. Its stop reason is
recorded when it is a known word: one of `qagents.external.STOP_REASONS`, a reason
qagents gives a cancelled turn (``permission``, ``closed``) or a word an adapter passes
through (ACP's ``max_turn_requests``, Devvi's ``forked``, ``blocked`` and
``interrupted``). Any other stop reason, such as an error turn's (the agent's message),
and the changed paths are recorded only with ``record_content``. A field of the wrong
type (a custom adapter's own ``turn_finished`` data) is left out; it never fails the run.

A tool span stays open until its ``tool_call_finished`` event, so the events of the
turn that are still on their way reach it. It ends at the time its tool returned, and
the events that arrive after that time are stamped with it, so they fall inside the span.
A calling run that is cancelled or times out during the turn ends the span before the
turn reports its end: that span has no ``turn_finished`` event.

`qagents.integrations.mlflow.MLflowHooks` does not record these events; export the
spans of `OTelHooks` to MLflow's ``/v1/traces`` endpoint to see them there.

Every span a run opened is ended when the run ends, however it ends. A tool
call that paused the run for input (``DeferToolCall``) ends its span without
an error and with ``qagents.tool.deferred`` set; the run span of a paused run
has ``qagents.status`` ``"paused"`` and lists the waiting calls.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .errors import ConfigurationError
from .events import ExternalAgentEvent, InputRequired, SubagentEvent, ToolCallFinished
from .hooks import Hooks
from .messages import ToolCallPart, ToolResultPart
from .models.base import ModelRequest, ModelResponse
from .results import RunResult
from .tools import RunContext

ATTR = {
    "operation": "gen_ai.operation.name",
    "agent": "gen_ai.agent.name",
    "conversation": "gen_ai.conversation.id",
    "provider": "gen_ai.provider.name",
    "response_model": "gen_ai.response.model",
    "response_id": "gen_ai.response.id",
    "finish_reasons": "gen_ai.response.finish_reasons",
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "tool_name": "gen_ai.tool.name",
    "tool_call_id": "gen_ai.tool.call.id",
    "input_messages": "gen_ai.input.messages",
    "output_messages": "gen_ai.output.messages",
    "tool_arguments": "gen_ai.tool.call.arguments",
    "tool_result": "gen_ai.tool.call.result",
    "run_id": "qagents.run_id",
    "parent_run_id": "qagents.parent_run_id",
    "parent_tool_call_id": "qagents.parent_tool_call_id",
    "depth": "qagents.depth",
    "status": "qagents.status",
    "stop_reason": "qagents.stop_reason",
    "cost": "qagents.cost_usd",
    "cached_input_tokens": "qagents.usage.cached_input_tokens",
    "deferred": "qagents.tool.deferred",
    "pending_approvals": "qagents.pending_approvals",
    "pending_inputs": "qagents.pending_inputs",
    "error_type": "error.type",
    "external_tool": "qagents.external.permission.tool",
    "external_tool_call_id": "qagents.external.permission.tool_call_id",
    "external_outcome": "qagents.external.permission.outcome",
    "external_approved": "qagents.external.permission.approved",
    "external_decision": "qagents.external.permission.decision",
    "external_decided_by": "qagents.external.permission.decided_by",
    "external_arguments": "qagents.external.permission.arguments",
    "external_reason": "qagents.external.permission.reason",
    "external_status": "qagents.external.status",
    "external_stop_reason": "qagents.external.stop_reason",
    "external_session": "qagents.external.session_id",
    "external_backend": "qagents.external.backend",
    "external_tool_calls": "qagents.external.tool_calls",
    "external_changed_files": "qagents.external.changed_files",
    "external_changed_files_count": "qagents.external.changed_files_count",
}
"""Span attribute names, in one place so a renamed convention is a one-line change."""

EVENTS = {
    "permission_decided": "qagents.external.permission_decided",
    "turn_finished": "qagents.external.turn_finished",
}
"""Span event names, by the ``kind`` of the external agent's event they record."""

_EXTERNAL_STOP_WORDS = frozenset(
    {"permission", "closed", "max_turn_requests", "forked", "blocked", "interrupted"}
)
"""Stop reasons besides ``STOP_REASONS`` that are words, not content, so a span records them
without ``record_content``. qagents' reasons for a cancelled turn: ``permission`` (the turn
paused the calling run for a permission), ``closed`` (the agent was closed). The words an
adapter passes through: ACP's ``max_turn_requests``
(`qagents.integrations.acp.mapping.STOP_REASONS`) and Devvi's ``forked``, ``blocked`` and
``interrupted`` (`qagents.integrations.devvi.events.TURN_END_KINDS`)."""

UNFINISHED = "the run ended before this finished"
"""Status description of a span that its run left open (a failure or a cancellation)."""

TOOL_FAILED = "the tool reported an error"
"""Status description of a failed tool call when content is not recorded."""


def _number(value: Any) -> float | None:
    """``value`` as a finite number, or None when it is not one (a bool, a string, NaN)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _count(value: Any) -> int | None:
    """``value`` as a count of tokens, calls or files, or None when it is not a number."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    number = _number(value)
    return int(number) if number is not None else None


@dataclass
class _OpenRun:
    """The spans one run has open."""

    span: Any
    parent_run_id: str | None
    chat: Any = None
    tools: dict[str, Any] = field(default_factory=dict)
    """Open ``execute_tool`` spans by tool call id."""
    finishing: dict[str, tuple[Any, int]] = field(default_factory=dict)
    """``execute_tool`` spans whose tool returned, with the time it did (nanoseconds), by
    tool call id. Each ends at its ``tool_call_finished`` event: the events an external
    agent's turn sent before that event are still on their way to `OTelHooks.on_event`."""


class OTelHooks(Hooks):
    """Emit OpenTelemetry spans for runs, model requests and tool calls.

    Args:
        tracer: The tracer to use; defaults to ``trace.get_tracer("qagents")``
            from the globally configured tracer provider.
        record_content: Put messages, tool arguments and tool results on the
            spans, and the text of errors (a failed tool's result, the message
            and stack trace of the exception that ended a run). For an external
            agent: the arguments and reason of each permission decision, the
            paths its turn changed and a stop reason that is not a known word
            (an error turn's message). Off by default because they often contain
            personal data.

    Raises:
        ConfigurationError: OpenTelemetry is not installed.
    """

    propagate: ClassVar[bool] = True
    """Child runs are traced too, nested under the tool call that started them."""

    def __init__(self, tracer: Any = None, *, record_content: bool = False) -> None:
        try:
            from opentelemetry import trace
        except ImportError as error:
            raise ConfigurationError(
                'Tracing needs OpenTelemetry: pip install "qagents[otel]"'
            ) from error
        self._trace = trace
        self._tracer = tracer or trace.get_tracer("qagents")
        self._record_content = record_content
        self._runs: dict[str, _OpenRun] = {}

    def _child(self, ctx: RunContext[Any], name: str, attributes: dict[str, Any]) -> Any:
        run = self._runs.get(ctx.run_id)
        context = self._trace.set_span_in_context(run.span) if run is not None else None
        return self._tracer.start_span(name, context=context, attributes=attributes)

    async def on_run_start(self, ctx: RunContext[Any]) -> None:
        attributes: dict[str, Any] = {
            ATTR["operation"]: "invoke_agent",
            ATTR["agent"]: ctx.agent_name,
            ATTR["conversation"]: ctx.session_id or "",
            ATTR["run_id"]: ctx.run_id,
            ATTR["depth"]: ctx.depth,
        }
        if ctx.parent_run_id is not None:
            attributes[ATTR["parent_run_id"]] = ctx.parent_run_id
        if ctx.parent_tool_call_id is not None:
            attributes[ATTR["parent_tool_call_id"]] = ctx.parent_tool_call_id
        context = None  # the current context: a top-level run nests under the caller's span
        parent = self._runs.get(ctx.parent_run_id) if ctx.parent_run_id else None
        if parent is not None:
            anchor = parent.tools.get(ctx.parent_tool_call_id or "", parent.span)
            context = self._trace.set_span_in_context(anchor)
        span = self._tracer.start_span(
            f"invoke_agent {ctx.agent_name}", context=context, attributes=attributes
        )
        self._runs[ctx.run_id] = _OpenRun(
            span=span, parent_run_id=ctx.parent_run_id if parent is not None else None
        )

    async def before_model(self, ctx: RunContext[Any], request: ModelRequest) -> None:
        run = self._runs.get(ctx.run_id)
        if run is None:
            return
        attributes: dict[str, Any] = {ATTR["operation"]: "chat", ATTR["agent"]: ctx.agent_name}
        if self._record_content:
            attributes[ATTR["input_messages"]] = json.dumps(
                [message.model_dump(mode="json") for message in request.messages],
                ensure_ascii=False,
            )
        if run.chat is not None:  # never left open by the loop; end it rather than leak it
            self._fail(run.chat)
        run.chat = self._child(ctx, "chat", attributes)

    async def after_model(self, ctx: RunContext[Any], response: ModelResponse) -> None:
        run = self._runs.get(ctx.run_id)
        if run is None or run.chat is None:
            return
        span, run.chat = run.chat, None
        if response.model:
            span.update_name(f"chat {response.model}")
            span.set_attribute(ATTR["response_model"], response.model)
        if response.provider:
            span.set_attribute(ATTR["provider"], response.provider)
        if response.response_id:
            span.set_attribute(ATTR["response_id"], response.response_id)
        span.set_attribute(ATTR["finish_reasons"], [response.finish_reason])
        span.set_attribute(ATTR["input_tokens"], response.usage.input_tokens)
        span.set_attribute(ATTR["output_tokens"], response.usage.output_tokens)
        if response.usage.cached_input_tokens:
            span.set_attribute(ATTR["cached_input_tokens"], response.usage.cached_input_tokens)
        if response.usage.cost_usd is not None:
            span.set_attribute(ATTR["cost"], response.usage.cost_usd)
        if self._record_content:
            span.set_attribute(
                ATTR["output_messages"],
                json.dumps([response.message.model_dump(mode="json")], ensure_ascii=False),
            )
        span.end()

    async def before_tool(self, ctx: RunContext[Any], call: ToolCallPart) -> None:
        run = self._runs.get(ctx.run_id)
        if run is None:
            return
        attributes: dict[str, Any] = {
            ATTR["operation"]: "execute_tool",
            ATTR["tool_name"]: call.name,
            ATTR["tool_call_id"]: call.id,
        }
        if self._record_content:
            attributes[ATTR["tool_arguments"]] = json.dumps(call.arguments, ensure_ascii=False)
        run.tools[call.id] = self._child(ctx, f"execute_tool {call.name}", attributes)

    async def after_tool(
        self, ctx: RunContext[Any], call: ToolCallPart, result: ToolResultPart
    ) -> None:
        run = self._runs.get(ctx.run_id)
        span = run.tools.pop(call.id, None) if run is not None else None
        if span is None:
            return
        returned = time.time_ns()
        if result.is_error:
            text = result.content[:200] if self._record_content else TOOL_FAILED
            span.set_status(self._trace.StatusCode.ERROR, text)
            if result.metadata.get("exception"):  # the exception's class name, not its message
                span.set_attribute(ATTR["error_type"], str(result.metadata["exception"]))
        if self._record_content:
            span.set_attribute(ATTR["tool_result"], result.content)
        assert run is not None
        run.finishing[call.id] = (span, returned)  # ended by its tool_call_finished

    async def on_event(self, ctx: RunContext[Any], event: Any) -> None:
        if isinstance(event, ToolCallFinished):
            run = self._runs.get(ctx.run_id)
            finished = run.finishing.pop(event.call.id, None) if run is not None else None
            if finished is not None:
                span, returned = finished
                span.end(end_time=returned)
        elif isinstance(event, SubagentEvent):
            inner = event.event
            if isinstance(inner, ExternalAgentEvent) and inner.kind in EVENTS:
                self._external_event(ctx, event.tool_call_id, inner)
        elif isinstance(event, InputRequired):
            # A tool that raised DeferToolCall ran before_tool but gets no after_tool.
            run = self._runs.get(ctx.run_id)
            span = run.tools.pop(event.input.tool_call_id, None) if run is not None else None
            if span is not None:
                span.set_attribute(ATTR["deferred"], True)
                span.end()

    def _external_event(
        self, ctx: RunContext[Any], tool_call_id: str | None, event: ExternalAgentEvent
    ) -> None:
        """Add an external agent's ``permission_decided`` or ``turn_finished`` event to the
        span of the tool call that ran the agent, or to the run span.

        Only what says who decided what and how the turn ended: the arguments, the reason,
        a stop reason that is not a known word (an error turn's message) and the changed
        paths are content, recorded only with ``record_content``. The data comes from the
        adapter, which may be a custom one: a field of the wrong type is left out, so
        tracing never fails a run that would succeed without it.
        """
        run = self._runs.get(ctx.run_id)
        if run is None:
            return
        span = run.tools.get(tool_call_id or "")
        when: int | None = None  # now
        if span is None and (finishing := run.finishing.get(tool_call_id or "")) is not None:
            span, when = finishing  # it happened before the tool returned: inside the span
        if span is None:
            span = run.span
        data = event.data
        attributes: dict[str, Any] = {ATTR["agent"]: event.agent}
        if event.kind == "permission_decided":
            for source, target in (
                ("tool", "external_tool"),
                ("tool_call_id", "external_tool_call_id"),
                ("outcome", "external_outcome"),
                ("decision", "external_decision"),
                ("rule", "external_decided_by"),
            ):
                value = data.get(source)
                if isinstance(value, str) and value:
                    attributes[ATTR[target]] = value
            attributes[ATTR["external_approved"]] = data.get("approved") is True
            if self._record_content:
                raw_arguments = data.get("arguments", {})
                if isinstance(raw_arguments, Mapping):
                    try:
                        arguments = json.dumps(raw_arguments, ensure_ascii=False, default=str)
                    except (TypeError, ValueError, RecursionError):  # a bad key, a cycle, too deep
                        pass
                    else:
                        attributes[ATTR["external_arguments"]] = arguments
                reason = data.get("reason")
                if isinstance(reason, str) and reason:
                    attributes[ATTR["external_reason"]] = reason
        else:
            from .external.contract import STOP_REASONS  # loaded: an external agent ran

            status = data.get("status")
            if isinstance(status, str) and status:
                attributes[ATTR["external_status"]] = status
            stop_reason = data.get("stop_reason")
            if isinstance(stop_reason, str) and stop_reason:
                known = stop_reason in STOP_REASONS or stop_reason in _EXTERNAL_STOP_WORDS
                if self._record_content or known:
                    attributes[ATTR["external_stop_reason"]] = stop_reason
            session = event.session_id or data.get("session_id")
            if isinstance(session, str) and session:
                attributes[ATTR["external_session"]] = session
            backend = data.get("backend")
            if isinstance(backend, str) and backend:
                attributes[ATTR["external_backend"]] = backend
            usage = data.get("usage")
            usage = usage if isinstance(usage, Mapping) else {}
            attributes[ATTR["input_tokens"]] = _count(usage.get("input_tokens")) or 0
            attributes[ATTR["output_tokens"]] = _count(usage.get("output_tokens")) or 0
            if cached := _count(usage.get("cached_input_tokens")):
                attributes[ATTR["cached_input_tokens"]] = cached
            if (cost := _number(usage.get("cost_usd"))) is not None:
                attributes[ATTR["cost"]] = cost
            calls = data.get("tool_calls")
            counts = calls.values() if isinstance(calls, Mapping) else ()
            attributes[ATTR["external_tool_calls"]] = sum(_count(n) or 0 for n in counts)
            if (count := _count(data.get("changed_files_count"))) is not None:
                attributes[ATTR["external_changed_files_count"]] = count
            files = data.get("changed_files")
            if self._record_content and isinstance(files, list | tuple) and files:
                paths = [path for path in files if isinstance(path, str)]
                if paths:
                    attributes[ATTR["external_changed_files"]] = paths
        span.add_event(EVENTS[event.kind], attributes, timestamp=when)

    async def on_run_end(
        self, ctx: RunContext[Any], result: RunResult | None, error: BaseException | None
    ) -> None:
        run = self._runs.pop(ctx.run_id, None)
        if run is None:
            return
        # Child runs that never reported their end (a cancelled task) are closed with this one.
        for run_id in self._descendants(ctx.run_id):
            orphan = self._runs.pop(run_id)
            self._close_open_spans(orphan)
            self._fail(orphan.span)
        self._close_open_spans(run)
        span = run.span
        if result is not None:
            span.set_attribute(ATTR["status"], result.status)
            if result.stop_reason:
                span.set_attribute(ATTR["stop_reason"], result.stop_reason)
            span.set_attribute(ATTR["input_tokens"], result.usage.input_tokens)
            span.set_attribute(ATTR["output_tokens"], result.usage.output_tokens)
            if result.usage.cost_usd is not None:
                span.set_attribute(ATTR["cost"], result.usage.cost_usd)
            if result.pending_approvals:
                span.set_attribute(
                    ATTR["pending_approvals"],
                    [item.tool_call_id for item in result.pending_approvals],
                )
            if result.pending_inputs:
                span.set_attribute(
                    ATTR["pending_inputs"], [item.tool_call_id for item in result.pending_inputs]
                )
        if error is not None:
            self._record_error(span, error)
        span.end()

    def _record_error(self, span: Any, error: BaseException) -> None:
        """Mark ``span`` as failed by ``error``.

        The message and the stack trace (which repeats the messages of chained exceptions)
        are recorded only with ``record_content``: they often quote content, such as the
        answer that failed validation. Otherwise the exception event has its type only.
        """
        kind = type(error)
        name = kind.__qualname__
        if kind.__module__ not in ("builtins", ""):
            name = f"{kind.__module__}.{name}"
        span.set_attribute(ATTR["error_type"], name)
        if self._record_content:
            span.record_exception(error)
            span.set_status(self._trace.StatusCode.ERROR, str(error)[:200])
        else:
            span.add_event("exception", {"exception.type": name})
            span.set_status(self._trace.StatusCode.ERROR, kind.__name__)

    def _descendants(self, run_id: str) -> list[str]:
        """Open runs below ``run_id``, deepest first."""
        found: list[str] = []
        frontier = [run_id]
        while frontier:
            children = [key for key, run in self._runs.items() if run.parent_run_id in frontier]
            found.extend(children)
            frontier = children
        return found[::-1]

    def _close_open_spans(self, run: _OpenRun) -> None:
        """End the model and tool spans a failure or a cancellation left open."""
        if run.chat is not None:
            self._fail(run.chat)
            run.chat = None
        for span, returned in run.finishing.values():  # their tools returned: no failure
            span.end(end_time=returned)
        run.finishing.clear()
        for span in run.tools.values():
            self._fail(span)
        run.tools.clear()

    def _fail(self, span: Any) -> None:
        span.set_status(self._trace.StatusCode.ERROR, UNFINISHED)
        span.end()
