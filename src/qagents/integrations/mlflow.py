"""MLflow tracing, as a hook.


    import mlflow
    from qagents.integrations.mlflow import MLflowHooks

    mlflow.set_tracking_uri("http://127.0.0.1:5000")   # or MLFLOW_TRACKING_URI
    mlflow.set_experiment("support-agent")
    agent = Agent("qcloud:<model-id>", hooks=[MLflowHooks()])

The hook writes to whatever tracking server and experiment your application
configured; it never changes that configuration. Pass ``experiment_id=`` to
log the traces of this hook to another experiment, or use `enable`, which
also sets the tracking URI and experiment for the process when you ask it to.

One top-level run is one MLflow trace:

    invoke_agent lead            AGENT       one per run
      chat <model>               CHAT_MODEL  one per model request
      lookup                     TOOL        one per tool call
      researcher                 TOOL        a sub-agent used as a tool...
        invoke_agent researcher  AGENT       ...nests its run under that call
          chat <model>           CHAT_MODEL

`MLflowHooks` propagates to child runs (``Hooks.propagate``), so attaching it
to the top agent traces the whole tree; the same instance on several agents
records each run once. A top-level run that starts while an MLflow span is
active (inside ``@mlflow.trace`` or ``mlflow.start_span``) nests under it, as
MLflow nests any span, so it joins your application's trace. That trace is
your application's: set its session and user with
``mlflow.update_current_trace``; the hook adds neither, nor its tags.

What is recorded:

- Model spans: token usage in MLflow's ``mlflow.chat.tokenUsage`` format
  (input, output, total and cache-read tokens), so the trace list shows the
  tokens of the whole tree; the cost when the model reports one; the model,
  provider, finish reason and response id.
- Tool spans: the call id, and an error status when the tool failed.
- Run spans: run id, session id, agent name, depth, parent ids, status,
  stop reason, usage and cost of the run, guardrail and pending calls.
- The trace: the session id as MLflow's session (``mlflow.trace.session``),
  the optional ``user_id`` (``mlflow.trace.user``), your ``tags``, and the tags
  ``qagents.status`` and ``qagents.stop_reason`` when the run ends. A paused
  run ends its trace with ``qagents.status = "paused"``; resuming it starts a
  new trace in the same session.

Message content stays out of traces unless ``record_content=True``. Then model
spans carry their messages, tool definitions and response in the OpenAI chat
format, which the MLflow UI renders as a conversation; tool spans carry the
arguments and the result; run spans carry the prompt and the final answer.
Inline images (``ImagePart.data``, or a ``data:`` URL) are replaced by a short
placeholder. Error text is content too: without ``record_content`` a failed
tool call has the status description `TOOL_FAILED`, and a span that an
exception ended records only the exception's type, without its message or
stack trace, because exception messages often quote the model's answer.
Approval tokens and ``ctx.state`` values are never recorded.

A compactor's own model request (the summary of `SummarizingCompactor`) goes
through the run's hooks, so it has a ``chat`` span and counts in the trace's
token usage.

Every span a run opened is ended when the run ends: on an answer, a pause, a
cancellation, a limit, a guardrail block or an exception. Spans left open by a
failure end with an error status; a tool call that paused the run for input
ends without one, with ``qagents.tool.status = "deferred"``.

MLflow exports traces in a background thread. A short script should call
``mlflow.flush_trace_async_logging()`` (or `MLflowHooks.flush`) before it
exits, so the last traces are written.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic_core import to_jsonable_python

from ..context import _compaction_model
from ..errors import ConfigurationError
from ..events import AgentSwitched, ApprovalRequired, GuardrailTriggered, InputRequired
from ..hooks import Hooks
from ..messages import ImagePart, Message, ReasoningPart, TextPart, ToolCallPart, ToolResultPart
from ..models.base import ModelRequest, ModelResponse, ToolSpec
from ..results import RunResult
from ..tools import RunContext

logger = logging.getLogger(__name__)

ATTR = {
    "run_id": "qagents.run_id",
    "session_id": "qagents.session_id",
    "agent": "qagents.agent",
    "depth": "qagents.depth",
    "parent_run_id": "qagents.parent_run_id",
    "parent_tool_call_id": "qagents.parent_tool_call_id",
    "resumed": "qagents.resumed",
    "status": "qagents.status",
    "stop_reason": "qagents.stop_reason",
    "turns": "qagents.turns",
    "usage": "qagents.usage",
    "cost": "qagents.cost_usd",
    "finish_reason": "qagents.finish_reason",
    "response_id": "qagents.response_id",
    "reasoning_tokens": "qagents.usage.reasoning_tokens",
    "tool_call_id": "qagents.tool_call_id",
    "tool_status": "qagents.tool.status",
    "exception": "qagents.tool.exception",
    "guardrail": "qagents.guardrail",
    "pending_approvals": "qagents.pending_approvals",
    "pending_inputs": "qagents.pending_inputs",
}
"""Names of the span attributes, trace tags and trace metadata qagents adds.

MLflow's own keys (token usage, model, cost, chat tools, message format,
session and user) come from ``mlflow.tracing.constant``."""

UNFINISHED = "the run ended before this finished"
"""Status description of a span that its run left open."""

TOOL_FAILED = "the tool reported an error"
"""Status description of a failed tool call when content is not recorded."""

_MIN_VERSION_HINT = 'pip install -U "mlflow>=3.5"'


@dataclass
class _OpenRun:
    """The spans one run has open."""

    span: Any
    trace_id: str | None
    owns_trace: bool
    """This run started the trace (so it sets the trace's tags when it ends)."""
    parent_run_id: str | None
    model: str | None = None
    provider: str | None = None
    chat: Any = None
    tools: dict[str, Any] = field(default_factory=dict)
    """Open TOOL spans by tool call id."""


class MLflowHooks(Hooks):
    """Record runs as MLflow traces.

    Traces go to the tracking server MLflow is configured with for the
    process (``mlflow.set_tracking_uri``, ``MLFLOW_TRACKING_URI`` or `enable`).
    MLflow's tracing exports through one process-wide provider, so there is
    no per-hook ``tracking_uri``; ``experiment_id`` picks the experiment.

    Args:
        record_content: Record messages, tool definitions, tool arguments and
            results, the prompt and the answer, and the text of errors (a
            failed tool's result, the message and stack trace of an
            exception). Off by default because they often contain personal
            data.
        experiment_id: Log the traces this hook starts to this experiment
            instead of the active one. Ignored for runs that nest in a span of
            your application (that trace is already in an experiment).
        user_id: The end user the runs serve, stored as MLflow's trace user
            (``mlflow.trace.user``).
        tags: Extra tags for every trace this hook starts.

    Raises:
        ConfigurationError: MLflow is not installed, or too old.
    """

    propagate: ClassVar[bool] = True
    """Child runs are traced too, nested under the tool call that started them."""

    def __init__(
        self,
        *,
        record_content: bool = False,
        experiment_id: str | None = None,
        user_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        mlflow = _import_mlflow()
        from mlflow.entities import SpanEvent, SpanStatus, SpanStatusCode, SpanType
        from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey

        self._mlflow = mlflow
        self._SpanEvent = SpanEvent
        self._SpanStatus = SpanStatus
        self._ERROR = SpanStatusCode.ERROR
        self._SpanType = SpanType
        self._key = SpanAttributeKey
        self._meta = TraceMetadataKey
        self.record_content = record_content
        self.experiment_id = experiment_id
        self.user_id = user_id
        self.tags = {str(key): str(value) for key, value in (tags or {}).items()}
        self.last_trace_id: str | None = None
        """Id of the trace the last top-level run wrote, once it has ended."""
        self._runs: dict[str, _OpenRun] = {}

    def flush(self) -> None:
        """Wait until MLflow has exported every finished trace."""
        self._mlflow.flush_trace_async_logging()

    # -- hook points ------------------------------------------------------------

    async def on_run_start(self, ctx: RunContext[Any]) -> None:
        parent_span: Any = None
        parent = self._runs.get(ctx.parent_run_id) if ctx.parent_run_id else None
        if parent is not None:
            parent_span = parent.tools.get(ctx.parent_tool_call_id or "", parent.span)
        # Without a parent, MLflow nests the span under the active one of the application.
        owns_trace = parent_span is None and self._mlflow.get_current_active_span() is None

        attributes: dict[str, Any] = {
            ATTR["run_id"]: ctx.run_id,
            ATTR["agent"]: ctx.agent_name,
            ATTR["depth"]: ctx.depth,
        }
        if ctx.session_id:
            attributes[ATTR["session_id"]] = ctx.session_id
        if ctx.parent_run_id is not None:
            attributes[ATTR["parent_run_id"]] = ctx.parent_run_id
        if ctx.parent_tool_call_id is not None:
            attributes[ATTR["parent_tool_call_id"]] = ctx.parent_tool_call_id
        run = getattr(ctx, "_run", None)
        if getattr(run, "resume", False):
            attributes[ATTR["resumed"]] = True

        options: dict[str, Any] = {}
        if owns_trace:
            metadata = {ATTR["run_id"]: ctx.run_id}
            if ctx.session_id:
                metadata[self._meta.TRACE_SESSION] = ctx.session_id
            if self.user_id is not None:
                metadata[self._meta.TRACE_USER] = self.user_id
            options = {"metadata": metadata, "tags": dict(self.tags)}
            if self.experiment_id is not None:
                options["experiment_id"] = self.experiment_id
        span = self._mlflow.start_span_no_context(
            f"invoke_agent {ctx.agent_name}",
            span_type=self._SpanType.AGENT,
            parent_span=parent_span,
            inputs=self._run_inputs(ctx) if self.record_content else None,
            attributes=attributes,
            **options,
        )
        model, provider = _agent_model(ctx)
        self._runs[ctx.run_id] = _OpenRun(
            span=span,
            trace_id=_trace_id(span),
            owns_trace=owns_trace,
            parent_run_id=ctx.parent_run_id if parent is not None else None,
            model=model,
            provider=provider,
        )

    async def before_model(self, ctx: RunContext[Any], request: ModelRequest) -> None:
        run = self._runs.get(ctx.run_id)
        if run is None:
            return
        if run.chat is not None:  # never left open by the loop; end it rather than leak it
            self._fail(run.chat)
        model, provider = _agent_model(ctx)  # a handoff changes the model
        run.model, run.provider = model or run.model, provider or run.provider
        model, provider = run.model, run.provider
        summarizer = _compaction_model.get()
        if summarizer is not None:  # a compactor's summary request goes to its own model
            model, provider = summarizer.name, summarizer.provider
        attributes: dict[str, Any] = {}
        if model:
            attributes[self._key.MODEL] = model
        if provider:
            attributes[self._key.MODEL_PROVIDER] = provider
        inputs = None
        if self.record_content:
            attributes[self._key.MESSAGE_FORMAT] = "openai"
            inputs = {"messages": chat_messages(request.messages)}
            if model:
                inputs["model"] = model
        run.chat = self._mlflow.start_span_no_context(
            f"chat {model}" if model else "chat",
            span_type=self._SpanType.CHAT_MODEL,
            parent_span=run.span,
            inputs=inputs,
            attributes=attributes,
        )
        if self.record_content and request.tools:
            self._set_tools(run.chat, request.tools)

    async def after_model(self, ctx: RunContext[Any], response: ModelResponse) -> None:
        run = self._runs.get(ctx.run_id)
        if run is None or run.chat is None:
            return
        span, run.chat = run.chat, None
        usage = response.usage
        attributes: dict[str, Any] = {
            self._key.CHAT_USAGE: _token_usage(response),
            ATTR["finish_reason"]: response.finish_reason,
        }
        if response.model:
            attributes[self._key.MODEL] = response.model
        if response.provider:
            attributes[self._key.MODEL_PROVIDER] = response.provider
        if response.response_id:
            attributes[ATTR["response_id"]] = response.response_id
        if usage.reasoning_tokens:
            attributes[ATTR["reasoning_tokens"]] = usage.reasoning_tokens
        if usage.cost_usd is not None:
            attributes[self._key.LLM_COST] = {"total_cost": usage.cost_usd}
            attributes[ATTR["cost"]] = usage.cost_usd
        outputs = None
        if self.record_content:
            outputs = {
                "id": response.response_id,
                "object": "chat.completion",
                "model": response.model or run.model,
                "choices": [
                    {
                        "index": 0,
                        "message": _assistant_message(response.message),
                        "finish_reason": response.finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                },
            }
        span.end(outputs=outputs, attributes=attributes)

    async def before_tool(self, ctx: RunContext[Any], call: ToolCallPart) -> None:
        run = self._runs.get(ctx.run_id)
        if run is None:
            return
        run.tools[call.id] = self._mlflow.start_span_no_context(
            call.name,
            span_type=self._SpanType.TOOL,
            parent_span=run.span,
            inputs=dict(call.arguments) if self.record_content else None,
            attributes={ATTR["tool_call_id"]: call.id},
        )

    async def after_tool(
        self, ctx: RunContext[Any], call: ToolCallPart, result: ToolResultPart
    ) -> None:
        run = self._runs.get(ctx.run_id)
        span = run.tools.pop(call.id, None) if run is not None else None
        if span is None:
            return
        attributes: dict[str, Any] = {ATTR["tool_status"]: "error" if result.is_error else "ok"}
        if result.metadata.get("exception"):
            attributes[ATTR["exception"]] = str(result.metadata["exception"])
        status = None
        if result.is_error:
            text = result.content[:500] if self.record_content else TOOL_FAILED
            status = self._SpanStatus(self._ERROR, text)
        outputs = result.content if self.record_content else None
        span.end(outputs=outputs, attributes=attributes, status=status)

    async def on_event(self, ctx: RunContext[Any], event: Any) -> None:
        run = self._runs.get(ctx.run_id)
        if run is None:
            return
        if isinstance(event, InputRequired):
            # A tool that raised DeferToolCall ran before_tool but gets no after_tool.
            span = run.tools.pop(event.input.tool_call_id, None)
            if span is not None:
                span.end(attributes={ATTR["tool_status"]: "deferred"})
            self._event(run.span, "input_required", event.input.tool_call_id, event.input.name)
        elif isinstance(event, ApprovalRequired):
            item = event.approval
            self._event(run.span, "approval_required", item.tool_call_id, item.name)
        elif isinstance(event, GuardrailTriggered):
            details = {"name": event.name, "stage": event.stage}
            if self.record_content:
                details["reason"] = event.reason
            run.span.add_event(self._SpanEvent("guardrail_tripped", attributes=details))
        elif isinstance(event, AgentSwitched):
            details = {"from_agent": event.from_agent, "to_agent": event.to_agent}
            run.span.add_event(self._SpanEvent("agent_switched", attributes=details))

    async def on_run_end(
        self, ctx: RunContext[Any], result: RunResult | None, error: BaseException | None
    ) -> None:
        run = self._runs.pop(ctx.run_id, None)
        if run is None:
            return
        # Child runs that never reported their end (a cancelled task) are closed with this one.
        for run_id in self._descendants(ctx.run_id):
            orphan = self._runs.pop(run_id)
            self._close_open_spans(orphan, None)
            self._fail(orphan.span)
        self._close_open_spans(run, error)

        attributes: dict[str, Any] = {}
        status: Any = None
        outputs: Any = None
        if result is not None:
            attributes.update(self._result_attributes(result))
            if result.status == "limit_reached":
                status = self._SpanStatus(self._ERROR, f"limit_reached: {result.stop_reason}")
            if self.record_content:
                outputs = _run_outputs(result)
        if error is not None:
            recorded = self._record_error(run.span, error)
            if recorded is not None:
                status = recorded
            attributes[ATTR["status"]] = "failed"
        if run.owns_trace and run.trace_id is not None:
            self._tag_trace(run.trace_id, attributes)
        run.span.end(outputs=outputs, attributes=attributes, status=status)
        if run.owns_trace and run.trace_id is not None:
            self.last_trace_id = run.trace_id

    # -- helpers ----------------------------------------------------------------

    def _run_inputs(self, ctx: RunContext[Any]) -> dict[str, Any]:
        """The new prompt of a run; none when it resumes or continues a conversation."""
        last = ctx.messages[-1] if ctx.messages else None
        if last is not None and last.role == "user" and not getattr(ctx._run, "resume", False):
            return {"messages": chat_messages([last])}
        return {"messages": []}

    def _result_attributes(self, result: RunResult) -> dict[str, Any]:
        usage = result.usage
        attributes: dict[str, Any] = {
            ATTR["status"]: result.status,
            ATTR["turns"]: result.turns,
            ATTR["usage"]: {
                "requests": usage.requests,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
            },
        }
        if result.stop_reason:
            attributes[ATTR["stop_reason"]] = result.stop_reason
        if usage.cost_usd is not None:
            attributes[ATTR["cost"]] = usage.cost_usd
        if result.guardrail is not None:
            guardrail = {"name": result.guardrail.name, "stage": result.guardrail.stage}
            if self.record_content:
                guardrail["reason"] = result.guardrail.reason
            attributes[ATTR["guardrail"]] = guardrail
        if result.pending_approvals:
            attributes[ATTR["pending_approvals"]] = [
                {"tool_call_id": item.tool_call_id, "name": item.name}
                for item in result.pending_approvals
            ]
        if result.pending_inputs:
            attributes[ATTR["pending_inputs"]] = [
                {"tool_call_id": item.tool_call_id, "name": item.name}
                for item in result.pending_inputs
            ]
        return attributes

    def _tag_trace(self, trace_id: str, attributes: dict[str, Any]) -> None:
        for name in ("status", "stop_reason"):
            value = attributes.get(ATTR[name])
            if value is not None:
                self._mlflow.set_trace_tag(trace_id, ATTR[name], str(value))

    def _set_tools(self, span: Any, tools: list[ToolSpec]) -> None:
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in tools
        ]
        try:
            self._mlflow.tracing.set_span_chat_tools(span, definitions)
        except Exception:  # a schema MLflow's ChatTool model does not accept; keep it as is
            span.set_attribute(self._key.CHAT_TOOLS, definitions)

    def _event(self, span: Any, name: str, tool_call_id: str, tool: str) -> None:
        span.add_event(
            self._SpanEvent(name, attributes={"tool_call_id": tool_call_id, "name": tool})
        )

    def _descendants(self, run_id: str) -> list[str]:
        """Open runs below ``run_id``, deepest first."""
        found: list[str] = []
        frontier = [run_id]
        while frontier:
            children = [key for key, run in self._runs.items() if run.parent_run_id in frontier]
            found.extend(children)
            frontier = children
        return found[::-1]

    def _close_open_spans(self, run: _OpenRun, error: BaseException | None) -> None:
        """End the model and tool spans a failure or a cancellation left open."""
        if run.chat is not None:
            if isinstance(error, Exception):  # most likely the model request failed
                run.chat.end(status=self._record_error(run.chat, error))
            else:
                self._fail(run.chat)
            run.chat = None
        for span in run.tools.values():
            self._fail(span)
        run.tools.clear()

    def _record_error(self, span: Any, error: BaseException) -> Any:
        """Add the exception event of ``error`` to ``span``; return the status to end it with.

        The message and the stack trace (which repeats the messages of chained exceptions)
        are recorded only with ``record_content``: they often quote content, such as the
        answer that failed validation. Otherwise the event has the exception's type only.
        Returns ``None`` when MLflow's ``record_exception`` already set the status.
        """
        if self.record_content:
            span.record_exception(error if isinstance(error, Exception) else str(error))
            return None
        name = type(error).__name__
        span.add_event(self._SpanEvent("exception", attributes={"exception.type": name}))
        return self._SpanStatus(self._ERROR, name)

    def _fail(self, span: Any) -> None:
        span.end(status=self._SpanStatus(self._ERROR, UNFINISHED))


def enable(
    *,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    experiment_id: str | None = None,
    record_content: bool = False,
    user_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> MLflowHooks:
    """Point MLflow at a server and experiment, and return hooks that trace into it.


        hooks = enable(tracking_uri="http://127.0.0.1:5000", experiment_name="agents")
        agent = Agent("qcloud:<model-id>", hooks=[hooks])

    Unlike `MLflowHooks`, this changes MLflow's process-wide configuration, and
    only what you pass: ``tracking_uri`` calls ``mlflow.set_tracking_uri``, and
    ``experiment_name`` or ``experiment_id`` call ``mlflow.set_experiment``
    (creating the named experiment if it does not exist).

    Args:
        tracking_uri: The tracking server or database, for example
            ``"http://127.0.0.1:5000"`` or ``"sqlite:///mlflow.db"``.
        experiment_name: The experiment to log to, created if needed.
        experiment_id: The experiment to log to, by id.
        record_content: See `MLflowHooks`.
        user_id: See `MLflowHooks`.
        tags: See `MLflowHooks`.

    Returns:
        A `MLflowHooks` to pass to ``Agent(hooks=[...])``.

    Raises:
        ConfigurationError: Both ``experiment_name`` and ``experiment_id`` were
            given, or MLflow is not installed.
    """
    if experiment_name is not None and experiment_id is not None:
        raise ConfigurationError(
            "Pass either experiment_name or experiment_id to enable(), not both."
        )
    mlflow = _import_mlflow()
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment_name is not None or experiment_id is not None:
        mlflow.set_experiment(experiment_name=experiment_name, experiment_id=experiment_id)
    return MLflowHooks(record_content=record_content, user_id=user_id, tags=tags)


# -- the OpenAI chat format MLflow renders ------------------------------------------


def chat_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Messages in the OpenAI chat format that the MLflow UI renders as a conversation.

    A tool message becomes one ``role: "tool"`` message per result. Inline
    images (``data``, or a ``data:`` URL) are replaced by a text placeholder
    with their media type and size; other image URLs are kept.
    """
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            converted.extend(
                {"role": "tool", "tool_call_id": result.tool_call_id, "content": result.content}
                for result in message.tool_results
            )
        elif message.role == "assistant":
            converted.append(_assistant_message(message))
        else:
            converted.append({"role": message.role, "content": _content(message)})
    return converted


def _assistant_message(message: Message) -> dict[str, Any]:
    reasoning = "".join(p.text for p in message.parts if isinstance(p, ReasoningPart) and p.text)
    content: Any = message.text or None
    if reasoning:
        content = [{"type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]}]
        if message.text:
            content.append({"type": "text", "text": message.text})
    converted: dict[str, Any] = {"role": "assistant", "content": content}
    if message.tool_calls:
        converted["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments
                    if call.raw_arguments is not None
                    else json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    return converted


def _content(message: Message) -> str | list[dict[str, Any]]:
    if all(isinstance(part, TextPart) for part in message.parts):
        return message.text
    parts: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            if part.url and not part.url.lower().startswith("data:"):
                parts.append({"type": "image_url", "image_url": {"url": part.url}})
            else:
                parts.append({"type": "text", "text": _image_placeholder(part)})
    return parts


def _image_placeholder(part: ImagePart) -> str:
    """A short stand-in for an inline image: its media type and size, not its data."""
    if part.url:  # a data: URL, "data:<media type>[;base64],<payload>"
        header, _, payload = part.url.partition(",")
        kind = header[len("data:") :].split(";")[0] or part.media_type or "image"
        encoding = "base64 " if ";base64" in header.lower() else ""
        return f"[{kind}, {len(payload)} {encoding}characters]"
    return f"[{part.media_type or 'image'}, {len(part.data or '')} base64 characters]"


def _run_outputs(result: RunResult) -> dict[str, Any]:
    # A text answer given with the final-answer tool is in the output, not in the last message.
    text = result.output if isinstance(result.output, str) else result.text
    outputs: dict[str, Any] = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}]
    }
    if result.output is not None and not isinstance(result.output, str):
        outputs["output"] = to_jsonable_python(result.output, fallback=repr)
    return outputs


def _token_usage(response: ModelResponse) -> dict[str, int]:
    usage = response.usage
    tokens = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }
    if usage.cached_input_tokens:
        tokens["cache_read_input_tokens"] = usage.cached_input_tokens
    written = response.message.metadata.get("cache_creation_input_tokens")
    if isinstance(written, int) and written:
        tokens["cache_creation_input_tokens"] = written
    return tokens


def _agent_model(ctx: RunContext[Any]) -> tuple[str | None, str | None]:
    """The model name and provider of the agent a run is using right now."""
    model = getattr(getattr(getattr(ctx, "_run", None), "agent", None), "model", None)
    return getattr(model, "name", None), getattr(model, "provider", None)


def _trace_id(span: Any) -> str | None:
    from mlflow.entities import LiveSpan

    return span.trace_id if isinstance(span, LiveSpan) else None  # not a NoOpSpan


def _import_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as error:
        raise ConfigurationError(
            'MLflow tracing needs MLflow: pip install "qagents[mlflow]"'
        ) from error
    try:
        import mlflow.tracing  # noqa: F401
        from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey

        _ = (SpanAttributeKey.CHAT_USAGE, SpanAttributeKey.MESSAGE_FORMAT)
        _ = (TraceMetadataKey.TRACE_SESSION, TraceMetadataKey.TRACE_USER)
        _ = (mlflow.start_span_no_context, mlflow.flush_trace_async_logging)
    except (ImportError, AttributeError) as error:
        raise ConfigurationError(
            f"MLflow {getattr(mlflow, '__version__', '?')} lacks the tracing API qagents uses "
            f"({error}). Upgrade it: {_MIN_VERSION_HINT}"
        ) from error
    return mlflow
