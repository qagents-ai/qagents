"""Keeping a long conversation inside the model's context window.

Before each model request, the agent asks its `Compactor` whether the history should be
replaced by a shorter one. Three are included:

| Compactor | What it does | Cost |
|---|---|---|
| `ToolResultClearing` | puts a placeholder in place of old tool results | no model request |
| `SummarizingCompactor` | replaces older messages with a summary | one model request |
| `CompactorChain` | tries compactors in order; the first result wins | what the winner costs |

Clearing keeps every message, so the conversation keeps its shape; a summary is shorter but
loses detail.

A common setup clears tool results first and summarises only when that is no longer enough:

    compactor = CompactorChain([
        ToolResultClearing(trigger_fraction=0.6, keep_last=4),
        SummarizingCompactor(trigger_fraction=0.8, keep_last=8),
    ])

**When they act.** `ToolResultClearing` and `SummarizingCompactor` compare the size of the next
request (`next_request_tokens`: the tokens the provider reported for the last request, plus an
estimate of the messages added since) with a threshold: ``trigger_tokens``, a
``trigger_fraction`` of the model's context window (``model.capabilities().context_length``), or
the lower of the two when both are given. The first request of a run or of a resume has no
previous response; the compactor then receives an empty response (no usage) and the whole
history is estimated.

**The journal keeps everything.** Compaction changes what is sent, not what happened. The
replacement history is written to the session journal as one ``compaction`` entry; the earlier
``message`` and ``tool_result`` entries stay as they were, so the journal still shows every
original message and tool result for audit. `Session.messages` and ``agent.resume`` continue
from the replacement. Messages a compactor produced are marked in their metadata
(``{"compaction": "summary"}`` on a summary message, ``{"compaction": "cleared"}`` on a cleared
tool result); metadata is not sent to providers.

**Caches and replayed reasoning.** Editing the history has a price with providers that bind
something to an exact prefix of it:

- Prompt caches (explicit cache breakpoints, automatic prefix caching in hosted APIs, prefix
  caching in vLLM or SGLang) reuse only the longest unchanged prefix of a request. Clearing a
  tool result changes the history from that message on, so the next request pays the full input
  price from the earliest edited message onwards; a summary rewrites nearly all of it. Act
  rarely and remove a lot each time: a high trigger, and ``clear_at_least`` for
  `ToolResultClearing`, keep the number of cache misses low.
- Some providers return reasoning for replay (thinking blocks with signatures, encrypted
  reasoning items, thought signatures) and expect it back unchanged, together with the turn it
  came from. The compactors here never edit a `ReasoningPart`. `ToolResultClearing` changes
  only tool results (and, with ``clear_arguments=True``, the arguments of the matching calls in
  assistant messages); `SummarizingCompactor` drops the summarised turns whole, reasoning
  included. A provider may still treat an edited history as new and not reuse reasoning or
  cache from it, so measure on the provider you use.

To write your own strategy, subclass `Compactor`, and check your replacement histories with
`unpaired_tool_calls` in your tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from . import prompts
from .errors import ConfigurationError
from .messages import (
    ImagePart,
    Message,
    Part,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from .models.base import (
    Model,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    ToolSpec,
    resolve_model,
)
from .tools import RunContext, Tool

# The token-estimate heuristic; `estimate_tokens` documents what each number stands for.
_BYTES_PER_TOKEN = 4
_MESSAGE_TOKENS = 4
_TOOL_PART_TOKENS = 8
_IMAGE_TOKENS = 1_000
_TOOL_DEFINITION_TOKENS = 10


# --- token estimates -----------------------------------------------------------------


def estimate_tokens(
    messages: Iterable[Message], tools: Iterable[ToolSpec | Tool] | None = None
) -> int:
    """Estimate how many tokens ``messages`` (and ``tools``) take up in a request.

    A fast heuristic that needs no tokenizer: one token per four bytes of UTF-8 text, plus a
    fixed amount for structure. It counts

    - text and reasoning text, tool-call names and argument JSON, and tool-result content, at
      four bytes per token, which comes to about four characters per token for English and code,
      two for Greek or Cyrillic and one and a third for Chinese or Japanese;
    - 4 tokens per message, for the role and separators;
    - 8 tokens per tool call and per tool result, for ids and framing;
    - 1,000 tokens per image, whatever its size (an image's base64 data is not counted as text);
    - for each tool definition in ``tools``: its name, description and parameter schema as
      compact JSON, plus 10 tokens.

    Opaque provider data in a `ReasoningPart` is not counted. The estimate never decreases when
    messages are added or text grows. Real counts depend on the model's tokenizer and on the
    language, so treat this as a rough figure; use the provider's reported usage, or its
    token-counting endpoint, when you need exact numbers.

    Args:
        messages: The messages to estimate.
        tools: Tool definitions sent with the messages, as `ToolSpec` or `Tool`.

    Returns:
        The estimated number of tokens.
    """
    size = 0
    fixed = 0
    for message in messages:
        fixed += _MESSAGE_TOKENS
        for part in message.parts:
            if isinstance(part, TextPart | ReasoningPart):
                size += _utf8_size(part.text)
            elif isinstance(part, ToolCallPart):
                fixed += _TOOL_PART_TOKENS
                size += _utf8_size(part.name) + _utf8_size(_arguments_text(part))
            elif isinstance(part, ToolResultPart):
                fixed += _TOOL_PART_TOKENS
                size += _utf8_size(part.content)
            elif isinstance(part, ImagePart):
                fixed += _IMAGE_TOKENS
    for item in tools or ():
        spec = item.spec if isinstance(item, Tool) else item
        fixed += _TOOL_DEFINITION_TOKENS
        size += (
            _utf8_size(spec.name)
            + _utf8_size(spec.description)
            + _utf8_size(_compact_json(spec.parameters))
        )
    return fixed + -(-size // _BYTES_PER_TOKEN)


def estimate_request_tokens(request: ModelRequest) -> int:
    """Estimate the input tokens of a complete request, as `estimate_tokens` does.

    Counts the messages (the system message included), the tool definitions and the output
    schema. Useful in a ``before_model`` hook, which receives the exact request.

    Args:
        request: The request to estimate.

    Returns:
        The estimated number of input tokens.
    """
    tokens = estimate_tokens(request.messages, request.tools)
    if request.output is not None:
        schema = _compact_json(request.output.json_schema)
        tokens += -(-_utf8_size(schema) // _BYTES_PER_TOKEN)
    return tokens


def next_request_tokens(messages: Sequence[Message], last_response: ModelResponse | None) -> int:
    """Estimate the input tokens of the next request.

    Starts from the tokens the provider reported for the last request, input plus output (the
    output is part of the history now), and adds `estimate_tokens` for the messages that joined
    the history after that response, typically tool results. This keeps a single large tool
    result from slipping past a threshold that the last request was still under.

    When there is no response, or the provider reported no tokens, the whole history is
    estimated instead. That estimate leaves out the system message and the tool definitions,
    which a compactor does not see.

    Args:
        messages: The history the next request will send, without the system message.
        last_response: The previous model response of the run, if any.

    Returns:
        The estimated number of input tokens.
    """
    reported = 0
    if last_response is not None:
        reported = last_response.usage.input_tokens + last_response.usage.output_tokens
    if reported <= 0:
        return estimate_tokens(messages)
    boundary = _last_assistant_index(messages)
    return reported + estimate_tokens(messages[boundary + 1 :])


def unpaired_tool_calls(messages: Sequence[Message]) -> list[str]:
    """Ids of tool calls and tool results that lack their counterpart.

    A conversation is well-formed when each assistant message with tool calls is followed by a
    tool message holding a result for each call, and each tool result answers a call of the
    assistant message right before it. Providers reject requests that break this rule. A
    history a `Compactor` returns must be well-formed; call this in the tests of your own
    compactor.

    Args:
        messages: A conversation, with or without the system message.

    Returns:
        The ids of calls without a result and of results without a call, in order of
        appearance. Empty when the conversation is well-formed.
    """
    problems: list[str] = []
    for index, message in enumerate(messages):
        previous = messages[index - 1] if index > 0 else None
        following = messages[index + 1] if index + 1 < len(messages) else None
        if message.role == "assistant" and message.tool_calls:
            answered = (
                {result.tool_call_id for result in following.tool_results}
                if following is not None and following.role == "tool"
                else set()
            )
            problems.extend(call.id for call in message.tool_calls if call.id not in answered)
        if message.tool_results:
            asked = (
                {call.id for call in previous.tool_calls}
                if previous is not None and previous.role == "assistant" and message.role == "tool"
                else set()
            )
            problems.extend(
                result.tool_call_id
                for result in message.tool_results
                if result.tool_call_id not in asked
            )
    return problems


# --- compactors ------------------------------------------------------------------------


class Compactor:
    """Decides, before a model request, whether to send a shorter history.

    Subclass it and implement `compact`. One compactor instance serves every run of the agent
    it belongs to, concurrent runs included, so keep per-run state out of it.
    """

    async def compact(
        self,
        ctx: RunContext[Any],
        messages: list[Message],
        last_response: ModelResponse,
        model: Model,
    ) -> list[Message] | None:
        """Return a shorter history to use from now on, or ``None`` to keep it.

        Args:
            ctx: The run context. Count model requests you make with ``ctx.add_usage``.
            messages: The history the next request would send, without the system message.
            last_response: The previous model response of the run. Its usage says how large
                the last request was. Before the first request of a run or a resume it is an
                empty assistant response with no usage.
            model: The model the next request goes to.

        Returns:
            The replacement history, or ``None``. A replacement must keep every tool call
            followed by its result (`unpaired_tool_calls` returns nothing for it). The agent
            writes it to the journal as a ``compaction`` entry.
        """
        raise NotImplementedError


class _ThresholdCompactor(Compactor):
    """Shared trigger: a token count, a fraction of the context window, or both."""

    def __init__(self, *, trigger_tokens: int | None, trigger_fraction: float | None) -> None:
        owner = type(self).__name__
        if trigger_tokens is None and trigger_fraction is None:
            raise ValueError(
                f"{owner} needs trigger_tokens=, trigger_fraction= or both, "
                f"for example {owner}(trigger_fraction=0.8)."
            )
        if trigger_tokens is not None and trigger_tokens < 0:
            raise ValueError("trigger_tokens must be 0 or more.")
        if trigger_fraction is not None and not 0 < trigger_fraction <= 1:
            raise ValueError(
                "trigger_fraction is a fraction of the context window: more than 0 and at "
                "most 1, for example 0.8."
            )
        self.trigger_tokens = trigger_tokens
        """Act when the next request is estimated at this many tokens or more."""
        self.trigger_fraction = trigger_fraction
        """Act when the next request fills this fraction of the model's context window."""

    async def threshold(self, model: Model) -> int:
        """The size of the next request, in tokens, at which this compactor acts.

        Args:
            model: The model the next request goes to.

        Returns:
            ``trigger_tokens``, ``trigger_fraction`` of the model's context length, or the
            lower of the two when both are set. When the model does not report a context
            length, ``trigger_tokens``.

        Raises:
            ConfigurationError: Only ``trigger_fraction`` is set and the model does not report
                a context length.
        """
        thresholds: list[int] = []
        if self.trigger_tokens is not None:
            thresholds.append(self.trigger_tokens)
        if self.trigger_fraction is not None:
            context_length = (await model.capabilities()).context_length
            if context_length:
                thresholds.append(int(context_length * self.trigger_fraction))
            elif not thresholds:
                raise ConfigurationError(
                    f"{type(self).__name__}(trigger_fraction={self.trigger_fraction}) needs "
                    f"the context length of model {model.provider}:{model.name}, which its "
                    "capabilities() do not report. Pass trigger_tokens= as well, or a model "
                    "that reports context_length."
                )
        return min(thresholds)

    async def triggered(
        self, messages: Sequence[Message], last_response: ModelResponse | None, model: Model
    ) -> bool:
        """Whether the next request reaches the threshold (see `next_request_tokens`)."""
        return next_request_tokens(messages, last_response) >= await self.threshold(model)


class ToolResultClearing(_ThresholdCompactor):
    """Replaces the content of old tool results with a short placeholder.

    Once the next request reaches the threshold, the content of every tool result older than
    the most recent ``keep_last`` results becomes ``qagents.prompts.TOOL_RESULT_CLEARED``.
    Messages are neither removed nor reordered, so every tool call keeps its result and the
    conversation keeps its shape. It costs no model request, which makes it the cheap first
    step before `SummarizingCompactor`.

    Never cleared: results the model has not seen yet (those after the last assistant
    message), results of tools named in ``exclude``, results already cleared, and results no
    longer than the placeholder. ``keep_last`` counts the most recent results of all tools,
    excluded ones included. When there is nothing to clear, or less than ``clear_at_least``
    tokens to gain, it returns ``None``, so a `CompactorChain` moves on to its next compactor.

    Clearing edits messages in the middle of the history, which invalidates a provider's prompt
    cache from the first edited message on (see the module notes on caches).

        agent = Agent(model, tools=[...], compactor=ToolResultClearing(trigger_fraction=0.6))
    """

    def __init__(
        self,
        *,
        trigger_tokens: int | None = None,
        trigger_fraction: float | None = None,
        keep_last: int = 3,
        exclude: Iterable[str] = (),
        clear_at_least: int = 0,
        clear_arguments: bool = False,
    ) -> None:
        """Configure when and what to clear.

        Args:
            trigger_tokens: Act when the next request is estimated at this many tokens or more.
            trigger_fraction: Act when the next request fills this fraction of the model's
                context window. Give one or both; with both, the lower threshold applies.
            keep_last: How many of the most recent tool results keep their content.
            exclude: Names of tools whose results are never cleared.
            clear_at_least: Clear only when that frees at least this many estimated tokens,
                so that breaking the prompt cache is worth it.
            clear_arguments: Also empty the arguments of the calls whose results are cleared,
                for tools that receive large inputs (file contents, documents). This edits
                assistant messages.

        Raises:
            ValueError: An argument is out of range.
        """
        super().__init__(trigger_tokens=trigger_tokens, trigger_fraction=trigger_fraction)
        if keep_last < 0:
            raise ValueError("keep_last must be 0 or more.")
        if isinstance(exclude, str):
            raise ValueError(f"exclude takes a list of tool names; write exclude=[{exclude!r}].")
        if clear_at_least < 0:
            raise ValueError("clear_at_least must be 0 or more.")
        self.keep_last = keep_last
        """How many of the most recent tool results keep their content."""
        self.exclude = frozenset(exclude)
        """Names of tools whose results are never cleared."""
        self.clear_at_least = clear_at_least
        """The fewest estimated tokens a clearing must free to happen."""
        self.clear_arguments = clear_arguments
        """Also empty the arguments of the calls whose results are cleared."""

    async def compact(
        self,
        ctx: RunContext[Any],
        messages: list[Message],
        last_response: ModelResponse | None,
        model: Model,
    ) -> list[Message] | None:
        if not await self.triggered(messages, last_response, model):
            return None
        boundary = _last_assistant_index(messages)
        positions = [
            (index, slot)
            for index, message in enumerate(messages)
            for slot, part in enumerate(message.parts)
            if isinstance(part, ToolResultPart)
        ]
        recent = set(positions[max(0, len(positions) - self.keep_last) :])

        edits: dict[int, list[Part]] = {}
        freed = 0
        for index, slot in positions:
            result = messages[index].parts[slot]
            assert isinstance(result, ToolResultPart)
            if (
                index > boundary
                or (index, slot) in recent
                or result.name in self.exclude
                or result.metadata.get("compaction") == "cleared"
            ):
                continue
            placeholder = prompts.TOOL_RESULT_CLEARED.format(name=result.name)
            gain = _utf8_size(result.content) - _utf8_size(placeholder)
            if gain <= 0:
                continue
            freed += gain
            parts = edits.setdefault(index, list(messages[index].parts))
            parts[slot] = result.model_copy(
                update={
                    "content": placeholder,
                    "metadata": {
                        **result.metadata,
                        "compaction": "cleared",
                        "original_chars": len(result.content),
                    },
                }
            )
            if self.clear_arguments and index > 0 and messages[index - 1].role == "assistant":
                # The call is in the assistant message right before its result; ids alone are
                # not enough, because some providers reuse them across turns.
                parts = edits.setdefault(index - 1, list(messages[index - 1].parts))
                for call_slot, call in enumerate(parts):
                    if not isinstance(call, ToolCallPart) or call.id != result.tool_call_id:
                        continue
                    arguments_gain = _utf8_size(_arguments_text(call)) - len("{}")
                    if arguments_gain > 0:
                        freed += arguments_gain
                        parts[call_slot] = call.model_copy(
                            update={"arguments": {}, "raw_arguments": None}
                        )

        if not edits or freed // _BYTES_PER_TOKEN < self.clear_at_least:
            return None
        return [
            message.model_copy(update={"parts": edits[index]}) if index in edits else message
            for index, message in enumerate(messages)
        ]


class SummarizingCompactor(_ThresholdCompactor):
    """Replaces the older part of the conversation with a model-written summary.

    Once the next request reaches the threshold, everything except the most recent
    ``keep_last`` messages is rendered as a plain transcript (`render_transcript`) and
    summarised by ``model`` when given, otherwise by the agent's own model; the summary request
    is counted in the run's usage. The summary, prefixed with ``prompts.SUMMARY_PREFIX``, takes
    the place of those messages as one user message. The kept tail never starts with a tool
    result, so a tool call is never separated from its result. System messages at the start of
    the history are kept as they are. When the older part is only an earlier summary, nothing
    is summarised again.

        agent = Agent(model, compactor=SummarizingCompactor(trigger_fraction=0.8, keep_last=8))

    The summary rewrites almost the whole history, so a provider's prompt cache starts over
    after it, and reasoning from the summarised turns is dropped with them.
    """

    def __init__(
        self,
        *,
        trigger_tokens: int | None = None,
        trigger_fraction: float | None = None,
        keep_last: int = 6,
        model: Model | str | None = None,
        max_summary_tokens: int = 2000,
    ) -> None:
        """Configure when to summarise and with which model.

        Args:
            trigger_tokens: Act when the next request is estimated at this many tokens or more.
            trigger_fraction: Act when the next request fills this fraction of the model's
                context window. Give one or both; with both, the lower threshold applies.
            keep_last: How many of the most recent messages are kept verbatim.
            model: The model that writes the summary, for example a cheaper one. ``None``
                uses the agent's model.
            max_summary_tokens: ``max_tokens`` of the summary request.

        Raises:
            ValueError: An argument is out of range.
        """
        if keep_last < 1:
            raise ValueError("keep_last must be at least 1")
        super().__init__(trigger_tokens=trigger_tokens, trigger_fraction=trigger_fraction)
        self.keep_last = keep_last
        """How many of the most recent messages are kept verbatim."""
        self.model = resolve_model(model) if model is not None else None
        """The model that writes the summary; ``None`` uses the agent's model."""
        self.max_summary_tokens = max_summary_tokens
        """``max_tokens`` of the summary request."""

    async def compact(
        self,
        ctx: RunContext[Any],
        messages: list[Message],
        last_response: ModelResponse | None,
        model: Model,
    ) -> list[Message] | None:
        if not await self.triggered(messages, last_response, model):
            return None
        lead = 0
        while lead < len(messages) and messages[lead].role == "system":
            lead += 1
        cut = _safe_cut(messages, self.keep_last)
        if cut <= lead:
            return None
        older, recent = messages[lead:cut], messages[cut:]
        if all(_is_summary(message) for message in older):
            return None
        request = ModelRequest(
            messages=[
                Message.user(f"{render_transcript(older)}\n\n---\n\n{prompts.SUMMARY_REQUEST}")
            ],
            settings=ModelSettings(max_tokens=self.max_summary_tokens),
        )
        response = await (self.model or model).request(request)
        ctx.add_usage(response.usage)
        summary = response.message.text.strip()
        if not summary:
            return None
        replacement = Message(
            role="user",
            parts=[TextPart(text=prompts.SUMMARY_PREFIX + summary)],
            metadata={"compaction": "summary"},
        )
        return [*messages[:lead], replacement, *recent]


class CompactorChain(Compactor):
    """Tries several compactors in order; the first that returns a history wins.

    The rest are not asked for that request. Put the cheapest first:

        CompactorChain([
            ToolResultClearing(trigger_fraction=0.6),
            SummarizingCompactor(trigger_fraction=0.8),
        ])

    Here old tool results are cleared first; once there is nothing left to clear and the
    conversation still fills 80% of the window, it is summarised.
    """

    def __init__(self, compactors: Iterable[Compactor]) -> None:
        """Build the chain.

        Args:
            compactors: The compactors to try, in order.

        Raises:
            ValueError: ``compactors`` is empty.
            ConfigurationError: An item is not a `Compactor`.
        """
        self.compactors = list(compactors)
        """The compactors, in the order they are tried."""
        if not self.compactors:
            raise ValueError("CompactorChain needs at least one compactor.")
        for item in self.compactors:
            if not isinstance(item, Compactor):
                raise ConfigurationError(
                    f"{item!r} is not a Compactor. Pass instances such as "
                    "ToolResultClearing(...) or SummarizingCompactor(...)."
                )

    async def compact(
        self,
        ctx: RunContext[Any],
        messages: list[Message],
        last_response: ModelResponse,
        model: Model,
    ) -> list[Message] | None:
        for compactor in self.compactors:
            replacement = await compactor.compact(ctx, list(messages), last_response, model)
            if replacement:
                return replacement
        return None


# --- helpers ---------------------------------------------------------------------------


def _safe_cut(messages: list[Message], keep_last: int) -> int:
    """Index where the kept tail starts, moved back so it never starts with a tool result."""
    cut = len(messages) - keep_last
    while cut > 0 and messages[cut].role == "tool":
        cut -= 1
    return cut


def _is_summary(message: Message) -> bool:
    if message.metadata.get("compaction") == "summary":
        return True
    # Summaries written before they were marked in metadata.
    return message.role == "user" and message.text.startswith(prompts.SUMMARY_PREFIX)


def _last_assistant_index(messages: Sequence[Message]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "assistant":
            return index
    return -1


def _arguments_text(call: ToolCallPart) -> str:
    """The argument text a provider adapter sends for ``call``."""
    if call.raw_arguments is not None and call.parse_error is None:
        return call.raw_arguments
    return json.dumps(call.arguments, ensure_ascii=False)


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _utf8_size(text: str) -> int:
    return len(text) if text.isascii() else len(text.encode("utf-8", errors="replace"))


def render_transcript(messages: list[Message]) -> str:
    """A plain-text rendering of messages, safe to send to any model without tool definitions."""
    lines: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, TextPart) and part.text:
                lines.append(f"{message.role.capitalize()}: {part.text}")
            elif isinstance(part, ToolCallPart):
                arguments = part.raw_arguments if part.raw_arguments is not None else part.arguments
                lines.append(f"Assistant called tool {part.name}: {arguments}")
            elif isinstance(part, ToolResultPart):
                label = "Tool error" if part.is_error else "Tool result"
                lines.append(f"{label} ({part.name}): {part.content}")
            elif isinstance(part, ReasoningPart):
                continue
    return "\n".join(lines)
