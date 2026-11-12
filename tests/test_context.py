from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from qagents import Agent, Hooks, Session, Usage, prompts, tool
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
from qagents.errors import ConfigurationError, ModelResponseError
from qagents.messages import (
    ImagePart,
    Message,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from qagents.models.base import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    OutputSpec,
    ToolSpec,
)
from qagents.models.testing import FunctionModel, ScriptedModel, call
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
    def __init__(self, name: str, log: list[Any], replacement: list[Message] | None) -> None:
        self.name = name
        self.log = log
        self.replacement = replacement

    async def compact(self, ctx, messages, last_response, model):
        self.log.append((self.name, [message.text for message in messages]))
        return self.replacement


async def test_chain_asks_every_compactor_in_order_each_on_the_previous_result():
    log: list[Any] = []
    shorter = [Message.user("short")]
    other = [Message.user("other")]
    chain = CompactorChain(
        [Recording("a", log, None), Recording("b", log, shorter), Recording("c", log, other)]
    )
    original = [Message.user("long")]
    assert await chain.compact(RunContext(), original, response(), ScriptedModel([])) == other
    assert log == [("a", ["long"]), ("b", ["long"]), ("c", ["short"])]

    log.clear()
    nothing = CompactorChain([Recording("a", log, None), Recording("b", log, None)])
    assert await nothing.compact(RunContext(), original, response(), ScriptedModel([])) is None
    assert [name for name, _ in log] == ["a", "b"]

    with pytest.raises(ValueError, match="at least one"):
        CompactorChain([])
    with pytest.raises(ConfigurationError, match="is not a Compactor"):
        CompactorChain([Recording("a", log, None), "summarise"])  # type: ignore[list-item]


async def test_chain_summarises_only_when_clearing_was_not_enough():
    messages = conversation("a" * 4_000, "b" * 4_000, "c" * 400)
    cleared_size = estimate_tokens(
        await ToolResultClearing(trigger_tokens=0, keep_last=0).compact(
            RunContext(), messages, response(), ScriptedModel([])
        )
        or []
    )
    assert cleared_size < estimate_tokens(messages) - 1_500

    # Clearing brings the request under the summarizer's trigger: no summary request.
    summarizer = ScriptedModel(["Summary of the pages."])
    chain = CompactorChain(
        [
            ToolResultClearing(trigger_tokens=0, keep_last=0),
            SummarizingCompactor(trigger_tokens=cleared_size + 50, keep_last=2, model=summarizer),
        ]
    )
    first = await chain.compact(RunContext(), messages, response(), ScriptedModel([]))
    assert first is not None and contents(first) == [cleared(), cleared(), "c" * 400]
    assert summarizer.requests == []

    # Still over it after clearing: the cleared history is summarised in the same step.
    chain.compactors[1] = SummarizingCompactor(
        trigger_tokens=cleared_size - 50, keep_last=2, model=summarizer
    )
    second = await chain.compact(RunContext(), messages, response(), ScriptedModel([]))
    assert second is not None
    assert second[0].text == SUMMARY_PREFIX + "Summary of the pages."
    assert contents(second) == ["c" * 400] and unpaired_tool_calls(second) == []
    (request,) = summarizer.requests
    assert cleared() in request.messages[0].text  # it summarised what clearing left


async def test_chain_gives_later_compactors_the_size_of_the_shortened_request():
    messages = conversation("a" * 4_000, "b" * 4_000, "c" * 400)
    last = response(input_tokens=5_000, output_tokens=100)
    seen: list[int] = []

    class Probe(Compactor):
        async def compact(self, ctx, messages, last_response, model):
            seen.append(next_request_tokens(messages, last_response))
            return None

    clearing = ToolResultClearing(trigger_tokens=0, keep_last=0)
    replaced = await CompactorChain([clearing, Probe()]).compact(
        RunContext(), messages, last, ScriptedModel([])
    )
    assert replaced is not None
    saved = estimate_tokens(messages) - estimate_tokens(replaced)
    # The reported size (system message and tool definitions included) less what was saved.
    assert seen == [next_request_tokens(messages, last) - saved]


async def test_the_documented_chain_keeps_a_tool_heavy_run_inside_the_window():
    @tool(read_only=True)
    def fetch(page: int) -> str:
        """Fetch a page."""
        return "r" * 4_000  # about 1,000 tokens

    replies: list[Any] = [
        Message.assistant("x" * 2_000, tool_calls=[call("fetch", page=page)]) for page in range(20)
    ]
    model = WindowedModel([*replies, "done"], context_length=10_000)
    summarizer = ScriptedModel(["Summary of the pages so far."] * 20)
    chain = CompactorChain(
        [
            ToolResultClearing(trigger_fraction=0.6, keep_last=2),
            SummarizingCompactor(trigger_fraction=0.8, keep_last=4, model=summarizer),
        ]
    )
    result = await Agent(model, tools=[fetch], compactor=chain).run("Read every page.")

    assert result.output == "done"
    sizes = [estimate_request_tokens(request) for request in model.requests]
    assert max(sizes) < 10_000  # without a summary it grows past the window
    assert summarizer.requests  # once clearing was not enough, it summarised
    assert all(unpaired_tool_calls(request.messages) == [] for request in model.requests)


# --- summary requests --------------------------------------------------------------------


def history_to_summarise() -> list[Message]:
    return [
        Message.user("My e-mail is jane@example.com; migrate the billing DB."),
        Message.assistant("Decided: use logical replication."),
        Message.user("And then?"),
        Message.assistant("Then switch over."),
    ]


def summary_response(text: str, finish_reason: str) -> ModelResponse:
    return ModelResponse(
        message=Message.assistant(text),
        finish_reason=finish_reason,  # type: ignore[arg-type]
        usage=Usage(requests=1, input_tokens=500, output_tokens=2_000),
    )


@pytest.mark.parametrize(
    ("reply", "match"),
    [
        (summary_response("The user wants to migrate the billing DB. Decided: use", "length"),
         "cut off at the token limit"),
        (summary_response("I cannot help with that.", "content_filter"), "refused"),
        (summary_response("  ", "stop"), "was empty"),
    ],
)  # fmt: skip
async def test_a_summary_that_was_cut_off_or_refused_never_replaces_the_history(reply, match):
    summarizer = ScriptedModel([reply])
    compactor = SummarizingCompactor(trigger_tokens=0, keep_last=1, model=summarizer)
    usage: list[Usage] = []
    ctx = RunContext(_add_usage=usage.append)
    with pytest.raises(ModelResponseError, match=match) as raised:
        await compactor.compact(ctx, history_to_summarise(), response(), ScriptedModel([]))
    assert "The history was not replaced" in str(raised.value)
    assert usage == [reply.usage]  # the request was billed all the same

    # Through a run: the run fails, and the journal records no compaction.
    session = Session()
    agent = Agent(
        ScriptedModel(["unused"]),
        compactor=SummarizingCompactor(trigger_tokens=0, keep_last=1, model=ScriptedModel([reply])),
    )
    with pytest.raises(ModelResponseError, match=match):
        await agent.run("Next?", history=history_to_summarise(), session=session)
    assert "compaction" not in [entry.kind for entry in await session.entries()]


async def test_the_summary_request_leaves_max_tokens_to_the_model_by_default():
    summarizer = ScriptedModel(["Summary.", "Summary."])
    for compactor, expected in (
        (SummarizingCompactor(trigger_tokens=0, keep_last=1, model=summarizer), None),
        (SummarizingCompactor(trigger_tokens=0, keep_last=1, model=summarizer,
                              max_summary_tokens=8_000), 8_000),
    ):  # fmt: skip
        await compactor.compact(RunContext(), history_to_summarise(), response(), summarizer)
        assert summarizer.requests[-1].settings.max_tokens == expected
    with pytest.raises(ValueError, match="max_summary_tokens"):
        SummarizingCompactor(trigger_tokens=0, max_summary_tokens=0)


async def test_the_summary_request_goes_through_the_run_s_hooks():
    seen: list[str] = []

    class Redact(Hooks):
        async def before_model(self, ctx, request):
            seen.append(f"before {len(request.messages)}")
            redacted = [
                message.model_copy(
                    update={
                        "parts": [
                            TextPart(text=message.text.replace("jane@example.com", "<email>"))
                        ]
                    }
                )
                if message.role == "user"
                else message
                for message in request.messages
            ]
            return request.model_copy(update={"messages": redacted})

        async def after_model(self, ctx, response):
            seen.append(f"after {response.model}")

    summarizer = ScriptedModel(["The user wants a billing DB migration."], name="summarizer")
    model = ScriptedModel(["Switch over at night."])
    agent = Agent(
        model,
        compactor=SummarizingCompactor(trigger_tokens=0, keep_last=1, model=summarizer),
        hooks=[Redact()],
    )
    result = await agent.run("Next?", history=history_to_summarise())

    assert result.output == "Switch over at night."
    assert seen == ["before 1", "after summarizer", "before 2", "after scripted"]
    (sent,) = summarizer.requests
    assert "jane@example.com" not in sent.messages[0].text and "<email>" in sent.messages[0].text
    assert result.usage.requests == 2  # the summary request is counted in the run


# --- reasoning that is not sent back -----------------------------------------------------


def test_reasoning_counts_only_when_it_is_sent_back():
    thought = "t" * 8_000
    plain = [Message.user("q"), Message(role="assistant", parts=[TextPart(text="ok")])]
    dropped = [
        Message.user("q"),
        Message(role="assistant", parts=[ReasoningPart(text=thought), TextPart(text="ok")]),
    ]
    replayed = [
        Message.user("q"),
        Message(
            role="assistant",
            parts=[
                ReasoningPart(text=thought, provider="anthropic", opaque={"provider": "anthropic"}),
                TextPart(text="ok"),
            ],
        ),
    ]
    assert estimate_tokens(dropped) == estimate_tokens(plain)
    assert estimate_tokens(replayed) == estimate_tokens(plain) + 2_000

    # The reasoning tokens of the last response count only when its reasoning is replayed.
    usage = Usage(requests=1, input_tokens=1_000, output_tokens=2_500, reasoning_tokens=2_000)
    last = ModelResponse(message=dropped[1], usage=usage)
    assert next_request_tokens(dropped, last) == 1_500
    last = ModelResponse(message=replayed[1], usage=usage)
    assert next_request_tokens(replayed, last) == 3_500


async def test_reasoning_that_is_never_sent_back_does_not_trigger_a_summary():
    def reply(request: ModelRequest) -> Message:
        return Message(
            role="assistant", parts=[ReasoningPart(text="t" * 8_000), TextPart(text="ok")]
        )

    summarizer = ScriptedModel([])
    agent = Agent(
        FunctionModel(reply),
        compactor=SummarizingCompactor(trigger_tokens=6_000, keep_last=2, model=summarizer),
    )
    session = Session()
    for prompt in ("one", "two", "three", "four"):
        await agent.run(prompt, session=session)
    assert summarizer.requests == []  # the history actually sent stays tiny


# --- the first request of a run --------------------------------------------------------


def wide_tool(index: int) -> Any:
    """A tool with a long description: about 360 tokens of definition."""

    @tool(name=f"tool_{index}", description="Look up a key. " + "detail " * 200, read_only=True)
    def lookup(key: str) -> str:
        return key

    return lookup


@pytest.mark.parametrize("heavy", ["instructions", "tools"])
async def test_the_first_request_of_a_run_counts_the_system_prompt_and_tools(heavy):
    # About 6,000 tokens of system prompt or tool definitions, and a short history: before
    # the first request there is no reported usage, and the whole request must count.
    options: dict[str, Any] = (
        {"instructions": "rule " * 5_000}
        if heavy == "instructions"
        else {"tools": [wide_tool(index) for index in range(16)]}
    )
    summarizer = ScriptedModel(["Earlier: two questions."])
    model = ScriptedModel(["Third answer."])
    agent = Agent(
        model,
        compactor=SummarizingCompactor(trigger_tokens=3_000, keep_last=1, model=summarizer),
        **options,
    )
    history = [Message.user("q1"), Message.assistant("a1"), Message.user("q2")]
    history.append(Message.assistant("a2"))
    await agent.run("q3", history=history)
    assert len(summarizer.requests) == 1
    assert model.requests[0].messages[-2].text == SUMMARY_PREFIX + "Earlier: two questions."


async def test_the_first_request_of_a_run_counts_the_output_schema_in_a_chain():
    class Report(BaseModel):
        summary: str = Field(description="detail " * 3_000)

    clearing = ToolResultClearing(trigger_tokens=4_000, keep_last=0)
    summarizer = ScriptedModel(["Earlier: a lookup."])
    chain = CompactorChain(
        [clearing, SummarizingCompactor(trigger_tokens=4_000, keep_last=1, model=summarizer)]
    )
    model = ScriptedModel(['{"summary": "done"}'])
    agent = Agent(model, output_type=Report, compactor=chain)
    history = conversation("short result", "another short result")
    history.append(Message.assistant("Read both."))
    await agent.run("Report.", history=history)
    assert estimate_tokens(history) < 100  # the schema alone reaches the triggers
    assert len(summarizer.requests) == 1  # clearing freed too little; the summary was needed


async def test_a_reported_usage_is_not_estimated_again():
    # After the first request the reported usage already includes the system prompt and the
    # tools, so the request is not rebuilt to estimate them.
    summarizer = ScriptedModel([])
    rendered: list[int] = []

    def instructions(ctx: RunContext[Any]) -> str:
        rendered.append(ctx.turn)
        return "rule " * 5_000

    model = ScriptedModel([call("tool_0", key="a"), "Done."], usage=Usage(input_tokens=100))
    agent = Agent(
        model,
        instructions=instructions,
        tools=[wide_tool(0)],
        compactor=SummarizingCompactor(trigger_tokens=3_000, keep_last=1, model=summarizer),
    )
    await agent.run("Look up a")
    assert summarizer.requests == []  # the provider said the requests were small
    assert len(rendered) == 3  # once to size the first request, then once per request


async def test_a_system_message_in_the_history_is_counted_once_before_the_first_request():
    summarizer = ScriptedModel(["Earlier: q1."] * 2)
    model = ScriptedModel(["ok"])
    agent = Agent(
        model,
        instructions="rule " * 2_000,
        compactor=SummarizingCompactor(trigger_tokens=8_000, keep_last=1, model=summarizer),
    )
    history = [Message.system("policy " * 2_000), Message.user("q1"), Message.assistant("a1")]
    await agent.run("q2", history=history)
    assert estimate_request_tokens(model.requests[0]) < 8_000
    assert summarizer.requests == []  # under the trigger: no billed summary
