"""OpenTelemetry tracing, as a hook.


    from qagents.tracing import OTelHooks

    agent = Agent("qcloud:<model-id>", hooks=[OTelHooks()])

It uses whatever tracer provider your application configured, so the spans go
wherever your other spans go (any OTLP backend: Langfuse, Phoenix, LangSmith,
Logfire, Grafana, Datadog). One run produces:

    invoke_agent <agent>
      chat <model>            one per model request
      execute_tool <tool>     one per tool call

Attribute names follow the OpenTelemetry GenAI semantic conventions. Those
are still marked "Development" upstream and may be renamed; the names used
here are kept in one place (``ATTR``) so they can be adjusted.

Message and tool content is left out of spans unless ``record_content=True``,
because it often contains personal data.
"""

from __future__ import annotations

import json
from typing import Any

from .errors import ConfigurationError
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
}


class OTelHooks(Hooks):
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
        self._runs: dict[str, Any] = {}
        self._chats: dict[str, Any] = {}
        self._tools: dict[tuple[str, str], Any] = {}

    def _child(self, ctx: RunContext[Any], name: str, attributes: dict[str, Any]) -> Any:
        parent = self._runs.get(ctx.run_id)
        context = self._trace.set_span_in_context(parent) if parent is not None else None
        return self._tracer.start_span(name, context=context, attributes=attributes)

    async def on_run_start(self, ctx: RunContext[Any]) -> None:
        self._runs[ctx.run_id] = self._tracer.start_span(
            f"invoke_agent {ctx.agent_name}",
            attributes={
                ATTR["operation"]: "invoke_agent",
                ATTR["agent"]: ctx.agent_name,
                ATTR["conversation"]: ctx.session_id or "",
                "qagents.run_id": ctx.run_id,
            },
        )

    async def before_model(self, ctx: RunContext[Any], request: ModelRequest) -> None:
        attributes: dict[str, Any] = {ATTR["operation"]: "chat", ATTR["agent"]: ctx.agent_name}
        if self._record_content:
            attributes[ATTR["input_messages"]] = json.dumps(
                [message.model_dump(mode="json") for message in request.messages],
                ensure_ascii=False,
            )
        self._chats[ctx.run_id] = self._child(ctx, "chat", attributes)

    async def after_model(self, ctx: RunContext[Any], response: ModelResponse) -> None:
        span = self._chats.pop(ctx.run_id, None)
        if span is None:
            return
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
        if response.usage.cost_usd is not None:
            span.set_attribute("qagents.cost_usd", response.usage.cost_usd)
        if self._record_content:
            span.set_attribute(
                ATTR["output_messages"],
                json.dumps([response.message.model_dump(mode="json")], ensure_ascii=False),
            )
        span.end()

    async def before_tool(self, ctx: RunContext[Any], call: ToolCallPart) -> None:
        attributes: dict[str, Any] = {
            ATTR["operation"]: "execute_tool",
            ATTR["tool_name"]: call.name,
            ATTR["tool_call_id"]: call.id,
        }
        if self._record_content:
            attributes[ATTR["tool_arguments"]] = json.dumps(call.arguments, ensure_ascii=False)
        self._tools[(ctx.run_id, call.id)] = self._child(
            ctx, f"execute_tool {call.name}", attributes
        )

    async def after_tool(
        self, ctx: RunContext[Any], call: ToolCallPart, result: ToolResultPart
    ) -> None:
        span = self._tools.pop((ctx.run_id, call.id), None)
        if span is None:
            return
        if result.is_error:
            span.set_status(self._trace.StatusCode.ERROR, result.content[:200])
        if self._record_content:
            span.set_attribute(ATTR["tool_result"], result.content)
        span.end()

    async def on_run_end(
        self, ctx: RunContext[Any], result: RunResult | None, error: BaseException | None
    ) -> None:
        # Close spans a failure left open, so nothing leaks.
        leftovers = [self._chats.pop(ctx.run_id, None)]
        leftovers += [
            self._tools.pop(key) for key in [k for k in self._tools if k[0] == ctx.run_id]
        ]
        for span in leftovers:
            if span is not None:
                span.set_status(self._trace.StatusCode.ERROR, "the run ended before this finished")
                span.end()
        span = self._runs.pop(ctx.run_id, None)
        if span is None:
            return
        if result is not None:
            span.set_attribute("qagents.status", result.status)
            if result.stop_reason:
                span.set_attribute("qagents.stop_reason", result.stop_reason)
            span.set_attribute(ATTR["input_tokens"], result.usage.input_tokens)
            span.set_attribute(ATTR["output_tokens"], result.usage.output_tokens)
            if result.usage.cost_usd is not None:
                span.set_attribute("qagents.cost_usd", result.usage.cost_usd)
        if error is not None:
            span.record_exception(error)
            span.set_status(self._trace.StatusCode.ERROR, str(error)[:200])
        span.end()
