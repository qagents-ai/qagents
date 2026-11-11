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
because it often contains personal data. Approval tokens are never recorded.

Every span a run opened is ended when the run ends, however it ends. A tool
call that paused the run for input (``DeferToolCall``) ends its span without
an error and with ``qagents.tool.deferred`` set; the run span of a paused run
has ``qagents.status`` ``"paused"`` and lists the waiting calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .errors import ConfigurationError
from .events import InputRequired
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
}
"""Span attribute names, in one place so a renamed convention is a one-line change."""

UNFINISHED = "the run ended before this finished"
"""Status description of a span that its run left open (a failure or a cancellation)."""


@dataclass
class _OpenRun:
    """The spans one run has open."""

    span: Any
    parent_run_id: str | None
    chat: Any = None
    tools: dict[str, Any] = field(default_factory=dict)
    """Open ``execute_tool`` spans by tool call id."""


class OTelHooks(Hooks):
    """Emit OpenTelemetry spans for runs, model requests and tool calls.

    Args:
        tracer: The tracer to use; defaults to ``trace.get_tracer("qagents")``
            from the globally configured tracer provider.
        record_content: Put messages, tool arguments and tool results on the
            spans. Off by default because they often contain personal data.

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
        if result.is_error:
            span.set_status(self._trace.StatusCode.ERROR, result.content[:200])
        if self._record_content:
            span.set_attribute(ATTR["tool_result"], result.content)
        span.end()

    async def on_event(self, ctx: RunContext[Any], event: Any) -> None:
        # A tool that raised DeferToolCall ran before_tool but gets no after_tool.
        if isinstance(event, InputRequired):
            run = self._runs.get(ctx.run_id)
            span = run.tools.pop(event.input.tool_call_id, None) if run is not None else None
            if span is not None:
                span.set_attribute(ATTR["deferred"], True)
                span.end()

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
            span.record_exception(error)
            span.set_status(self._trace.StatusCode.ERROR, str(error)[:200])
        span.end()

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
        for span in run.tools.values():
            self._fail(span)
        run.tools.clear()

    def _fail(self, span: Any) -> None:
        span.set_status(self._trace.StatusCode.ERROR, UNFINISHED)
        span.end()
