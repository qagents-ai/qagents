from __future__ import annotations

from typing import Any

import pytest

from qagents import Agent, Session, Usage, prompts, tool
from qagents.context import (
    Compactor,
    CompactorChain,
    SummarizingCompactor,
    ToolResultClearing,
    estimate_request_tokens,
    estimate_tokens,
    next_request_tokens,
    unpaired_tool_calls,
)
from qagents.errors import ConfigurationError
from qagents.messages import ImagePart, Message, TextPart, ToolCallPart, ToolResultPart
from qagents.models.base import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    OutputSpec,
    ToolSpec,
)
from qagents.models.testing import ScriptedModel, call
from qagents.prompts import SUMMARY_PREFIX
from qagents.tools import RunContext


class WindowedModel(ScriptedModel):
    """A scripted model whose catalog reports a context window."""

    def __init__(self, replies: list[Any], *, context_length: int | None, **options: Any) -> None:
        super().__init__(replies, **options)
        self.context_length = context_length

    async def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(context_length=self.context_length)


def cleared(name: str = "fetch") -> str:
    return prompts.TOOL_RESULT_CLEARED.format(name=name)


def conversation(*contents: str, names: list[str] | None = None) -> list[Message]:
    """A user message, then one tool call and its result per content. The last result is unseen."""
    messages = [Message.user("Read the pages.")]
    for index, content in enumerate(contents):
        name = names[index] if names else "fetch"
        messages.append(
            Message.assistant(
                tool_calls=[ToolCallPart(id=f"c{index}", name=name, arguments={"page": index})]
            )
        )
        messages.append(
            Message(
                role="tool",
                parts=[ToolResultPart(tool_call_id=f"c{index}", name=name, content=content)],
            )
        )
    return messages


def response(input_tokens: int = 0, output_tokens: int = 0) -> ModelResponse:
    usage = Usage(requests=1, input_tokens=input_tokens, output_tokens=output_tokens)
    return ModelResponse(message=Message.assistant("..."), usage=usage)


def contents(messages: list[Message]) -> list[str]:
    return [result.content for message in messages for result in message.tool_results]


# --- estimation ----------------------------------------------------------------------


def test_estimate_is_roughly_proportional_to_text_length():
    short = estimate_tokens([Message.user("a" * 4_000)])
    long = estimate_tokens([Message.user("a" * 8_000)])
    assert 950 <= short <= 1_050  # about four characters per token for ASCII
    assert 1.9 <= long / short <= 2.1
    prose = "The agent reads the file, then writes a short summary of it. " * 100
    assert abs(estimate_tokens([Message.user(prose)]) - len(prose) / 4) <= 0.05 * len(prose) / 4
    assert estimate_tokens([]) == 0


def test_estimate_never_decreases_as_the_conversation_grows():
    history: list[Message] = []
    previous = 0
    for index in range(30):
        if index % 3 == 0:
            history.append(Message.user(f"question {index} " * index))
        elif index % 3 == 1:
            history.append(Message.assistant(tool_calls=[call("fetch", page=index)]))
        else:
            history.append(
                Message(
                    role="tool",
                    parts=[ToolResultPart(tool_call_id="x", name="fetch", content="r" * index)],
                )
            )
        current = estimate_tokens(history)
        assert current > previous
        previous = current
    # Growing one text never lowers the estimate either.
    sizes = [estimate_tokens([Message.user("b" * length)]) for length in range(0, 200, 7)]
    assert sizes == sorted(sizes)


def test_estimate_counts_non_ascii_text_by_its_utf8_size():
    english = estimate_tokens([Message.user("a" * 3_000)])
    greek = estimate_tokens([Message.user("α" * 3_000)])  # two bytes per character
    japanese = estimate_tokens([Message.user("日" * 3_000)])  # three bytes per character
    assert english < greek < japanese
    assert 1.9 <= (greek - 4) / (english - 4) <= 2.1


def test_estimate_counts_tool_calls_results_images_and_definitions():
    base = [Message.user("Look at this.")]
    with_call = [
        *base,
        Message.assistant(tool_calls=[call("write_file", content="z" * 4_000)]),
    ]
    assert estimate_tokens(with_call) - estimate_tokens(base) >= 1_000

    image = Message.user([TextPart(text="What is it?"), ImagePart(data="A" * 1_000_000)])
    assert 1_000 <= estimate_tokens([image]) <= 1_100  # a fixed cost, not the base64 size

    @tool(read_only=True)
    def search(query: str, limit: int = 10) -> str:
        """Search the catalogue for products that match a query."""
        return query

    spec = ToolSpec(name="lookup", description="d" * 400, parameters={"type": "object"})
    with_tools = estimate_tokens(base, [spec, search])
    assert with_tools >= estimate_tokens(base) + 100 + 2 * 10
    assert estimate_tokens(base, [search]) == estimate_tokens(base, [search.spec])


def test_estimate_request_counts_the_system_message_tools_and_output_schema():
    messages = [Message.system("s" * 400), Message.user("Hello")]
    spec = ToolSpec(name="lookup", description="Look up a key.")
    plain = estimate_request_tokens(ModelRequest(messages=messages))
    assert plain == estimate_tokens(messages)
    with_tools = estimate_request_tokens(ModelRequest(messages=messages, tools=[spec]))
    assert with_tools == estimate_tokens(messages, [spec])
    schema = {"type": "object", "properties": {"total": {"type": "number"}}}
    with_output = estimate_request_tokens(
        ModelRequest(messages=messages, tools=[spec], output=OutputSpec(json_schema=schema))
    )
    assert with_output > with_tools


def test_next_request_adds_new_messages_to_the_reported_usage():
    messages = conversation("x" * 4_000)
    new_results = estimate_tokens(messages[-1:])
    assert new_results >= 1_000
    assert next_request_tokens(messages, response(1_000, 100)) == 1_100 + new_results
    # Without reported usage the whole history is estimated.
    assert next_request_tokens(messages, response()) == estimate_tokens(messages)
    assert next_request_tokens(messages, None) == estimate_tokens(messages)


def test_unpaired_tool_calls_finds_calls_and_results_without_a_counterpart():
    assert unpaired_tool_calls(conversation("a", "b")) == []
    open_call = [Message.user("hi"), Message.assistant(tool_calls=[call("fetch")])]
    assert unpaired_tool_calls(open_call) == [open_call[1].tool_calls[0].id]
    orphan = [
        Message.user("hi"),
        Message(role="tool", parts=[ToolResultPart(tool_call_id="lost", name="x", content="")]),
    ]
    assert unpaired_tool_calls(orphan) == ["lost"]


# --- thresholds ------------------------------------------------------------------------


async def test_trigger_fraction_uses_the_context_window_the_model_reports():
    model = WindowedModel([], context_length=10_000)
    compactor = ToolResultClearing(trigger_fraction=0.5, keep_last=0)
    assert await compactor.threshold(model) == 5_000

    messages = conversation("x" * 2_000, "y" * 2_000)  # the unseen result is ~512 tokens
    assert await compactor.compact(RunContext(), messages, response(3_000), model) is None
    replaced = await compactor.compact(RunContext(), messages, response(6_000), model)
    assert replaced is not None and contents(replaced) == [cleared(), "y" * 2_000]

    # A large unseen tool result counts: the last request was under the threshold.
    assert await compactor.compact(RunContext(), messages, response(4_600), model) is not None


async def test_with_both_triggers_the_lower_threshold_applies():
    both = ToolResultClearing(trigger_tokens=1_000, trigger_fraction=0.5)
    assert await both.threshold(WindowedModel([], context_length=100_000)) == 1_000
    assert await both.threshold(WindowedModel([], context_length=1_000)) == 500
    # An unknown window falls back to trigger_tokens.
    assert await both.threshold(WindowedModel([], context_length=None)) == 1_000


async def test_trigger_fraction_alone_fails_when_the_window_is_unknown():
    compactor = SummarizingCompactor(trigger_fraction=0.8)
    with pytest.raises(ConfigurationError, match="trigger_tokens="):
        await compactor.threshold(ScriptedModel([]))

    @tool(read_only=True)
    def lookup(key: str) -> str:
        """Look something up."""
        return key

    agent = Agent(
        ScriptedModel([call("lookup", key="a"), "done"]), tools=[lookup], compactor=compactor
    )
    with pytest.raises(ConfigurationError, match="context length"):
        await agent.run("go")


def test_compactor_arguments_are_checked_when_built():
    with pytest.raises(ValueError, match="trigger_tokens=, trigger_fraction="):
        SummarizingCompactor()
    with pytest.raises(ValueError, match="trigger_tokens=, trigger_fraction="):
        ToolResultClearing(keep_last=2)
    with pytest.raises(ValueError, match="trigger_fraction"):
        SummarizingCompactor(trigger_fraction=1.5)
    with pytest.raises(ValueError, match="trigger_fraction"):
        ToolResultClearing(trigger_fraction=0)
    with pytest.raises(ValueError, match="trigger_tokens"):
        ToolResultClearing(trigger_tokens=-1)
    with pytest.raises(ValueError, match="keep_last"):
        SummarizingCompactor(trigger_tokens=10, keep_last=0)
    with pytest.raises(ValueError, match="keep_last"):
        ToolResultClearing(trigger_tokens=10, keep_last=-1)
    with pytest.raises(ValueError, match=r"exclude=\['read_plan'\]"):
        ToolResultClearing(trigger_tokens=10, exclude="read_plan")
    with pytest.raises(ValueError, match="clear_at_least"):
        ToolResultClearing(trigger_tokens=10, clear_at_least=-5)


# --- tool result clearing --------------------------------------------------------------


async def test_clearing_keeps_pairing_and_the_most_recent_results():
    pages = [f"page {index} " + "x" * 400 for index in range(5)]
    messages = conversation(*pages)
    compactor = ToolResultClearing(trigger_tokens=0, keep_last=2)

    replaced = await compactor.compact(RunContext(), messages, response(), ScriptedModel([]))

    assert replaced is not None
    assert [m.role for m in replaced] == [m.role for m in messages]
    assert unpaired_tool_calls(replaced) == []
    assert contents(replaced) == [cleared()] * 3 + pages[3:]
    first = replaced[2].tool_results[0]
    assert first.metadata == {"compaction": "cleared", "original_chars": len(pages[0])}
    assert first.tool_call_id == "c0" and first.name == "fetch" and not first.is_error
    assert [m.tool_calls for m in replaced] == [m.tool_calls for m in messages]
    assert contents(messages) == pages  # the input list is not modified
    # Nothing left to clear: the history is kept, so a chain moves on to its next compactor.
    assert await compactor.compact(RunContext(), replaced, response(), ScriptedModel([])) is None


async def test_keep_last_counts_results_not_messages():
    parallel = [
        ToolCallPart(id="p0", name="fetch", arguments={"page": 0}),
        ToolCallPart(id="p1", name="fetch", arguments={"page": 1}),
    ]
    messages = [
        Message.user("Read the pages."),
        Message.assistant(tool_calls=parallel),
        Message(
            role="tool",
            parts=[
                ToolResultPart(tool_call_id="p0", name="fetch", content="a" * 400),
                ToolResultPart(tool_call_id="p1", name="fetch", content="b" * 400),
            ],
        ),
        *conversation("c" * 400, "d" * 400)[1:],
    ]
    compactor = ToolResultClearing(trigger_tokens=0, keep_last=3)
    replaced = await compactor.compact(RunContext(), messages, response(), ScriptedModel([]))
    assert replaced is not None
    assert contents(replaced) == [cleared(), "b" * 400, "c" * 400, "d" * 400]
    assert unpaired_tool_calls(replaced) == []


async def test_results_the_model_has_not_seen_are_never_cleared():
    messages = conversation("a" * 400, "b" * 400)
    compactor = ToolResultClearing(trigger_tokens=0, keep_last=0)
    replaced = await compactor.compact(RunContext(), messages, response(), ScriptedModel([]))
    assert replaced is not None and contents(replaced) == [cleared(), "b" * 400]

    only_unseen = conversation("a" * 400)
    assert await compactor.compact(RunContext(), only_unseen, response(), ScriptedModel([])) is None


async def test_clearing_skips_excluded_tools_and_short_results():
    messages = conversation(
        "plan " * 100,
        "ok",
        "page " * 100,
        "last " * 100,
        names=["read_plan", "fetch", "fetch", "fetch"],
    )
    compactor = ToolResultClearing(trigger_tokens=0, keep_last=0, exclude=["read_plan"])
    replaced = await compactor.compact(RunContext(), messages, response(), ScriptedModel([]))
    assert replaced is not None
    assert contents(replaced) == ["plan " * 100, "ok", cleared(), "last " * 100]
    assert compactor.exclude == frozenset({"read_plan"})


async def test_clear_at_least_waits_until_clearing_is_worth_a_cache_miss():
    messages = conversation("a" * 400, "b" * 400, "c" * 400)  # two clearable, ~80 tokens each
    model = ScriptedModel([])
    small = ToolResultClearing(trigger_tokens=0, keep_last=0, clear_at_least=1_000)
    assert await small.compact(RunContext(), messages, response(), model) is None
    enough = ToolResultClearing(trigger_tokens=0, keep_last=0, clear_at_least=100)
    replaced = await enough.compact(RunContext(), messages, response(), model)
    assert replaced is not None and contents(replaced)[:2] == [cleared(), cleared()]


async def test_clear_arguments_empties_the_inputs_of_cleared_calls_only():
    big = {"path": "notes.md", "content": "z" * 2_000}
    messages = [
        Message.user("Write the files."),
        Message.assistant(tool_calls=[ToolCallPart(id="w0", name="write_file", arguments=big)]),
        Message(role="tool", parts=[ToolResultPart(tool_call_id="w0", name="write_file", content="Saved notes.md. " * 10)]),
        Message.assistant(tool_calls=[ToolCallPart(id="w1", name="write_file", arguments=big)]),
        Message(role="tool", parts=[ToolResultPart(tool_call_id="w1", name="write_file", content="Saved notes.md again. " * 10)]),
    ]  # fmt: skip
    compactor = ToolResultClearing(trigger_tokens=0, keep_last=0, clear_arguments=True)
    replaced = await compactor.compact(RunContext(), messages, response(), ScriptedModel([]))
    assert replaced is not None
    first, second = replaced[1].tool_calls[0], replaced[3].tool_calls[0]
    assert (first.id, first.name, first.arguments, first.raw_arguments) == (
        "w0", "write_file", {}, None,
    )  # fmt: skip
    assert second.arguments == big
    assert contents(replaced)[0] == cleared("write_file")
    assert estimate_tokens(replaced) < estimate_tokens(messages) - 400
    assert unpaired_tool_calls(replaced) == []
    assert messages[1].tool_calls[0].arguments == big  # the input list is not modified


async def test_clear_arguments_pairs_by_position_when_a_provider_reuses_call_ids():
    big = {"content": "z" * 1_000}
    messages = [Message.user("Write the file three times.")]
    for _ in range(3):
        messages.append(
            Message.assistant(tool_calls=[ToolCallPart(id="0", name="write_file", arguments=big)])
        )
        messages.append(
            Message(
                role="tool",
                parts=[ToolResultPart(tool_call_id="0", name="write_file", content="ok " * 50)],
            )
        )
    compactor = ToolResultClearing(trigger_tokens=0, keep_last=1, clear_arguments=True)
    replaced = await compactor.compact(RunContext(), messages, response(), ScriptedModel([]))
    assert replaced is not None
    assert [m.tool_calls[0].arguments for m in replaced if m.role == "assistant"] == [{}, {}, big]
    assert contents(replaced) == [cleared("write_file")] * 2 + ["ok " * 50]


async def test_clearing_through_an_agent_run_is_journaled_and_the_originals_kept():
    pages = {1: "first " * 400, 2: "second " * 400, 3: "third " * 400}

    @tool(read_only=True)
    def fetch(page: int) -> str:
        """Fetch a page."""
        return pages[page]

    model = ScriptedModel(
        [call("fetch", page=1), call("fetch", page=2), call("fetch", page=3), "Done."]
    )
    session = Session()
    compactor = ToolResultClearing(trigger_tokens=800, keep_last=1)
    agent = Agent(model, tools=[fetch], compactor=compactor)

    result = await agent.run("Read three pages.", session=session)

    assert result.output == "Done."
    sent = [contents(request.messages) for request in model.requests]
    assert sent == [
        [],
        [pages[1]],  # under the threshold: nothing cleared yet
        [cleared(), pages[2]],
        [cleared(), cleared(), pages[3]],
    ]
    assert all(unpaired_tool_calls(request.messages) == [] for request in model.requests)

    entries = await session.entries()
    assert [entry.kind for entry in entries].count("compaction") == 2
    # The journal keeps every original result for audit.
    recorded = [entry.data["result"]["content"] for entry in entries if entry.kind == "tool_result"]
    assert recorded == [pages[1], pages[2], pages[3]]
    journaled = [
        Message.model_validate(entry.data["message"])
        for entry in entries
        if entry.kind == "message"
    ]
    assert contents(journaled) == [pages[1], pages[2], pages[3]]
    # What the session continues from is the compacted history.
    restored = await session.messages()
    assert [m.model_dump() for m in restored] == [m.model_dump() for m in result.messages]
    assert contents(restored) == [cleared(), cleared(), pages[3]]
    assert restored[2].tool_results[0].metadata["compaction"] == "cleared"


# --- summarising -----------------------------------------------------------------------


async def test_summarizing_by_fraction_through_an_agent_run():
    @tool(read_only=True)
    def lookup(key: str) -> str:
        """Look something up."""
        return f"value of {key}"

    big = Usage(requests=1, input_tokens=900, output_tokens=200)
    model = WindowedModel(
        [call("lookup", key="a"), "Final answer."], context_length=2_000, usage=big
    )
    summarizer = ScriptedModel(["The user asked about a."])
    compactor = SummarizingCompactor(trigger_fraction=0.5, keep_last=2, model=summarizer)
    session = Session()

    result = await Agent(model, tools=[lookup], compactor=compactor).run(
        "Look up a", session=session
    )

    assert result.output == "Final answer."
    after = model.requests[1].messages
    assert after[0].text == SUMMARY_PREFIX + "The user asked about a."
    assert after[0].metadata == {"compaction": "summary"}
    assert [m.role for m in after] == ["user", "assistant", "tool"]
    assert result.usage.requests == 3
    assert [entry.kind for entry in await session.entries()].count("compaction") == 1


async def test_summarizing_does_not_summarise_a_summary_again():
    summarizer = ScriptedModel([])
    compactor = SummarizingCompactor(trigger_tokens=0, keep_last=2, model=summarizer)
    marked = Message(
        role="user",
        parts=[TextPart(text=SUMMARY_PREFIX + "Earlier work.")],
        metadata={"compaction": "summary"},
    )
    unmarked = Message.user(SUMMARY_PREFIX + "Written by qagents 0.1.")  # no metadata yet
    for summary in (marked, unmarked):
        messages = [summary, *conversation("a" * 400)[1:]]
        assert (
            await compactor.compact(RunContext(), messages, response(), ScriptedModel([])) is None
        )
    assert summarizer.requests == []


async def test_summarizing_keeps_leading_system_messages():
    summarizer = ScriptedModel(["Short summary."])
    compactor = SummarizingCompactor(trigger_tokens=0, keep_last=2, model=summarizer)
    messages = [
        Message.system("You are terse."),
        Message.user("First question"),
        Message.assistant("First answer"),
        Message.user("Second question"),
        Message.assistant("Second answer"),
    ]
    replaced = await compactor.compact(RunContext(), messages, response(), ScriptedModel([]))
    assert replaced is not None
    assert [m.text for m in replaced] == [
        "You are terse.",
        SUMMARY_PREFIX + "Short summary.",
        "Second question",
        "Second answer",
    ]
    transcript = summarizer.requests[0].messages[0].text
    assert "You are terse." not in transcript and "First question" in transcript


# --- chains ----------------------------------------------------------------------------


class Recording(Compactor):
    def __init__(self, name: str, log: list[str], replacement: list[Message] | None) -> None:
        self.name = name
        self.log = log
        self.replacement = replacement

    async def compact(self, ctx, messages, last_response, model):
        self.log.append(self.name)
        return self.replacement


async def test_chain_tries_compactors_in_order_and_the_first_history_wins():
    log: list[str] = []
    shorter = [Message.user("short")]
    chain = CompactorChain(
        [
            Recording("a", log, None),
            Recording("b", log, shorter),
            Recording("c", log, [Message.user("other")]),
        ]
    )
    assert await chain.compact(RunContext(), [], response(), ScriptedModel([])) == shorter
    assert log == ["a", "b"]

    log.clear()
    nothing = CompactorChain([Recording("a", log, None), Recording("b", log, None)])
    assert await nothing.compact(RunContext(), [], response(), ScriptedModel([])) is None
    assert log == ["a", "b"]

    with pytest.raises(ValueError, match="at least one"):
        CompactorChain([])
    with pytest.raises(ConfigurationError, match="is not a Compactor"):
        CompactorChain([Recording("a", log, None), "summarise"])  # type: ignore[list-item]


async def test_chain_clears_first_and_summarises_when_nothing_is_left_to_clear():
    summarizer = ScriptedModel(["Summary of the pages."])
    chain = CompactorChain(
        [
            ToolResultClearing(trigger_tokens=0, keep_last=0),
            SummarizingCompactor(trigger_tokens=0, keep_last=2, model=summarizer),
        ]
    )
    messages = conversation("a" * 400, "b" * 400)
    first = await chain.compact(RunContext(), messages, response(), ScriptedModel([]))
    assert first is not None and contents(first) == [cleared(), "b" * 400]
    assert summarizer.requests == []

    second = await chain.compact(RunContext(), first, response(), ScriptedModel([]))
    assert second is not None
    assert second[0].text == SUMMARY_PREFIX + "Summary of the pages."
    assert contents(second) == ["b" * 400] and unpaired_tool_calls(second) == []
    assert len(summarizer.requests) == 1
