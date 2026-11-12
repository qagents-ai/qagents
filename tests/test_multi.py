"""fan_out (orchestrator and workers) and route (classifier, then specialist)."""

from __future__ import annotations

import asyncio
import enum
import json
import logging
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from qagents import (
    Agent,
    Hooks,
    InMemoryStore,
    Limits,
    Message,
    RunContext,
    Session,
    Usage,
    prompts,
    tool,
)
from qagents.errors import (
    ConfigurationError,
    GuardrailTripped,
    ModelServerError,
    OutputValidationError,
)
from qagents.models.base import ModelRequest, ModelResponse
from qagents.models.testing import FunctionModel, ScriptedModel, call
from qagents.multi import FAN_OUT_TIMEOUT_S, RouteResult, fan_out, route

ONE = Usage(requests=1, input_tokens=10, output_tokens=2)


def task_of(request: ModelRequest) -> str:
    """The task a worker was given: the first user message of its conversation."""
    return next(message.text for message in request.messages if message.role == "user")


def fan_out_result(result) -> list[dict[str, Any]]:
    tool_message = next(message for message in result.messages if message.role == "tool")
    return json.loads(tool_message.tool_results[0].content)


class Tracer(Hooks):
    propagate = True

    def __init__(self) -> None:
        self.runs: list[tuple[str, str | None, str | None, int]] = []

    async def on_run_start(self, ctx: RunContext[Any]) -> None:
        self.runs.append((ctx.agent_name, ctx.parent_run_id, ctx.parent_tool_call_id, ctx.depth))


# -- fan_out ------------------------------------------------------------------------------


async def test_fan_out_answers_every_task_in_task_order_as_linked_child_runs():
    async def work(request: ModelRequest) -> str:
        task = task_of(request)
        await asyncio.sleep(0.03 if task == "Spain" else 0.0)  # the first task finishes last
        return f"facts about {task}"

    researcher = Agent(
        FunctionModel(work, usage=ONE), name="researcher", description="Finds market facts."
    )
    parallel = fan_out(researcher)
    lead_model = ScriptedModel(
        [call("researcher_parallel", tasks=["Spain", "Greece", "Italy"]), "Report written."],
        usage=ONE,
    )
    tracer = Tracer()
    lead = Agent(lead_model, name="lead", tools=[parallel], hooks=[tracer])

    stream = lead.stream("Compare the solar markets.")
    events = [event async for event in stream]
    result = stream.result

    assert result.output == "Report written."
    assert fan_out_result(result) == [
        {"task": "Spain", "answer": "facts about Spain"},
        {"task": "Greece", "answer": "facts about Greece"},
        {"task": "Italy", "answer": "facts about Italy"},
    ]
    spec = lead_model.requests[0].tools[0]
    assert spec.description.startswith("Finds market facts.\n\nGive researcher several")
    assert spec.parameters["properties"]["tasks"]["maxItems"] == 8
    # Each worker saw only its own task.
    assert sorted(task_of(request) for request in researcher.model.requests) == [
        "Greece",
        "Italy",
        "Spain",
    ]
    # Children are linked to the fan_out call and their usage counts in the lead's run.
    call_id = result.messages[1].tool_calls[0].id
    assert tracer.runs[0] == ("lead", None, None, 0)
    assert sorted(tracer.runs[1:]) == [("researcher", result.run_id, call_id, 1)] * 3
    assert result.usage.requests == 5 and result.usage.input_tokens == 50
    labels = {event.agent for event in events if event.type == "subagent_event"}
    assert labels == {"researcher[0]", "researcher[1]", "researcher[2]"}
    assert all(event.tool_call_id == call_id for event in events if event.type == "subagent_event")


async def test_fan_out_reports_failed_tasks_without_stopping_the_others():
    @tool(read_only=True)
    def search(query: str) -> str:
        """Search."""
        return "more"

    def work(request: ModelRequest) -> Any:
        task = task_of(request)
        if task == "explode":
            raise RuntimeError("the provider is down")
        if task == "loop":
            return call("search", query="again")
        return f"done: {task}"

    worker = Agent(FunctionModel(work), name="worker", tools=[search], limits=Limits(max_turns=2))
    lead = Agent(
        ScriptedModel([call("worker_parallel", tasks=["one", "explode", "loop", "two"]), "ok"]),
        tools=[fan_out(worker)],
    )
    result = await lead.run("go")

    one, explode, loop, two = fan_out_result(result)
    assert one == {"task": "one", "answer": "done: one"}
    assert explode == {
        "task": "explode",
        "error": prompts.FAN_OUT_TASK_FAILED.format(
            name="worker", error="RuntimeError: the provider is down"
        ),
    }
    assert loop == {
        "task": "loop",
        "error": prompts.SUBAGENT_STOPPED.format(name="worker", reason="max_turns"),
    }
    assert two == {"task": "two", "answer": "done: two"}
    assert result.output == "ok"


async def test_fan_out_bounds_how_many_children_work_at_once():
    running = 0
    peak = 0

    async def work(request: ModelRequest) -> str:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return "done"

    worker = Agent(FunctionModel(work), name="worker")
    tasks = [f"task {number}" for number in range(5)]
    lead = Agent(
        ScriptedModel([call("batch", tasks=tasks), "ok"]),
        tools=[fan_out(worker, name="batch", max_parallel=2)],
    )
    result = await lead.run("go")
    assert [entry["task"] for entry in fan_out_result(result)] == tasks
    assert peak == 2


async def test_fan_out_refuses_too_many_or_no_tasks_before_starting_any():
    worker_model = ScriptedModel([])
    worker = Agent(worker_model, name="worker")
    lead = Agent(
        ScriptedModel(
            [
                call("worker_parallel", tasks=["a", "b", "c"]),
                call("worker_parallel", tasks=[]),
                "ok",
            ]
        ),
        tools=[fan_out(worker, max_tasks=2)],
    )
    result = await lead.run("go")
    too_many, empty = [
        message.tool_results[0] for message in result.messages if message.role == "tool"
    ]
    assert too_many.is_error and "at most 2 items" in too_many.content
    assert empty.is_error and "at least 1 item" in empty.content
    assert worker_model.requests == []
    with pytest.raises(ConfigurationError, match="max_parallel"):
        fan_out(worker, max_parallel=0)
    with pytest.raises(ConfigurationError, match="max_tasks"):
        fan_out(worker, max_tasks=0)


async def test_fan_out_children_stop_at_the_parents_remaining_budget():
    worker_model = FunctionModel(
        lambda request: call("search", query="more"),
        usage=Usage(requests=1, input_tokens=20, output_tokens=0),
    )

    @tool(read_only=True)
    def search(query: str) -> str:
        """Search."""
        return "more"

    worker = Agent(worker_model, name="worker", tools=[search])
    lead = Agent(
        ScriptedModel([call("worker_parallel", tasks=["a"]), "ok"], usage=ONE),
        tools=[fan_out(worker)],
        limits=Limits(max_total_tokens=100),
    )
    result = await lead.run("go")
    (entry,) = fan_out_result(result)
    assert entry["error"] == prompts.SUBAGENT_STOPPED.format(
        name="worker", reason="max_total_tokens"
    )
    assert len(worker_model.requests) == 5  # 88 tokens were left: five requests of 20


async def test_cancelling_the_parent_cancels_the_children():
    started = 0
    stopped: list[str] = []

    async def work(request: ModelRequest) -> str:
        nonlocal started
        started += 1
        try:
            await asyncio.sleep(10)
        finally:
            stopped.append(task_of(request))
        return "never"

    worker = Agent(FunctionModel(work), name="worker")
    lead = Agent(
        ScriptedModel([call("worker_parallel", tasks=["a", "b", "c"])]),
        tools=[fan_out(worker, max_parallel=2)],
    )
    stream = lead.stream("go")
    async for event in stream:
        if event.type == "subagent_event" and event.event.type == "turn_started":
            if started == 2:
                stream.cancel()
    result = stream.result

    assert (result.status, result.stop_reason) == ("cancelled", "cancelled")
    assert sorted(stopped) == ["a", "b"]  # the third task never started
    # Not read-only by default: its outcome is unknown, so it is not repeated.
    assert result.messages[-1].tool_results[0].content == prompts.TOOL_INTERRUPTED.format(
        name="worker_parallel"
    )


# -- route --------------------------------------------------------------------------------


def specialists() -> dict[str, Agent]:
    return {
        "billing": Agent(ScriptedModel(["Your invoice is 40 EUR."], usage=ONE), name="billing"),
        "technical": Agent(ScriptedModel(["Restart the router."], usage=ONE), name="technical"),
    }


@pytest.mark.parametrize(
    ("mode", "reply"),
    [
        ("native", '"billing"'),
        ("prompted", '"billing"'),
        ("tool", call("final_answer", value="billing")),
    ],
)
async def test_route_runs_the_agent_a_literal_classifier_picks(mode, reply):
    classifier_model = ScriptedModel([reply], usage=ONE)
    triage = Agent(
        classifier_model,
        name="triage",
        instructions="Pick the team.",
        output_type=Literal["billing", "technical"],
        output_mode=mode,
    )
    routes = specialists()
    session = Session()
    routed = await route(triage, routes, "How much is my invoice?", session=session)

    assert isinstance(routed, RouteResult)
    assert routed.route == "billing" and routed.result.output == "Your invoice is 40 EUR."
    assert routed.classification.agent == "triage"
    assert routed.usage.requests == 2
    # The classifier ran on its own; the specialist continued the given session.
    assert routed.classification.session is not session
    assert [message.text for message in await session.messages()] == [
        "How much is my invoice?",
        "Your invoice is 40 EUR.",
    ]
    assert routes["technical"].model.requests == []


async def test_route_reads_plain_text_and_enum_answers():
    class Team(enum.Enum):
        BILLING = "billing"
        TECHNICAL = "technical"

    triage = Agent(ScriptedModel(['"technical"']), name="triage", output_type=Team)
    assert (await route(triage, specialists(), "My router is down")).route == "technical"

    for answer, expected in [
        ("Technical.", "technical"),
        ("  billing\n", "billing"),
        ("`Billing`", "billing"),
        ('"technical"', "technical"),
    ]:
        triage = Agent(ScriptedModel([answer]), name="triage")
        assert (await route(triage, specialists(), "hi")).route == expected

    triage = Agent(ScriptedModel(["I am not sure."]), name="triage")
    routed = await route(triage, specialists(), "hi", default="technical")
    assert routed.route == "technical" and routed.result.output == "Restart the router."

    triage = Agent(ScriptedModel(["I am not sure."]), name="triage")
    with pytest.raises(OutputValidationError, match="names none of the routes"):
        await route(triage, specialists(), "hi")


async def test_route_takes_no_route_when_the_classifier_does_not_finish():
    def no_invoices(ctx: RunContext[Any], message: Any) -> None:
        raise GuardrailTripped("not allowed")

    triage = Agent(
        ScriptedModel([]),
        name="triage",
        output_type=Literal["billing", "technical"],
        input_guardrails=[no_invoices],
    )
    routes = specialists()
    routed = await route(triage, routes, "hi")
    assert routed.route is None and routed.result is routed.classification
    assert routed.result.status == "blocked"
    assert all(agent.model.requests == [] for agent in routes.values())


async def test_route_checks_its_configuration_before_any_request():
    model = ScriptedModel([])
    with pytest.raises(ConfigurationError, match="at least one route"):
        await route(Agent(model), {}, "hi")
    with pytest.raises(ConfigurationError, match="must be an Agent"):
        await route(Agent(model), {"billing": "billing-agent"}, "hi")  # type: ignore[dict-item]
    with pytest.raises(ConfigurationError, match="default='sales'"):
        await route(Agent(model), specialists(), "hi", default="sales")
    with pytest.raises(ConfigurationError, match=r"can answer \['sales'\]"):
        await route(Agent(model, output_type=Literal["billing", "sales"]), specialists(), "hi")
    with pytest.raises(ConfigurationError, match="output_type"):
        await route(Agent(model, output_type=int), specialists(), "hi")
    assert model.requests == []


async def test_route_from_a_tool_links_both_runs_and_counts_their_usage():
    tracer = Tracer()
    triage = Agent(
        ScriptedModel(['"billing"'], usage=ONE),
        name="triage",
        output_type=Literal["billing", "technical"],
    )
    routes = specialists()

    @tool(read_only=True)
    async def ask_team(ctx: RunContext[Any], question: str) -> str:
        """Ask the right team."""
        routed = await route(triage, routes, question, parent=ctx)
        return f"{routed.route}: {routed.result.output}"

    lead = Agent(
        ScriptedModel([call("ask_team", question="Invoice?"), "Done."], usage=ONE),
        name="lead",
        tools=[ask_team],
        hooks=[tracer],
    )
    result = await lead.run("go")
    assert result.messages[2].tool_results[0].content == "billing: Your invoice is 40 EUR."
    assert [(name, depth) for name, _, _, depth in tracer.runs] == [
        ("lead", 0),
        ("triage", 1),
        ("billing", 1),
    ]
    assert result.usage.requests == 4


# -- fixes ----------------------------------------------------------------------------------


async def test_the_usage_of_a_worker_that_raises_or_times_out_reaches_the_lead():
    @tool(read_only=True)
    def search(query: str) -> str:
        """Search."""
        return "found"

    def bills_then_fails(request: ModelRequest) -> Any:
        if request.messages[-1].role == "tool":
            raise ModelServerError("the provider is down")
        return call("search", query="q")  # a billed response

    worker = Agent(FunctionModel(bills_then_fails, usage=ONE), name="worker", tools=[search])
    lead = Agent(
        ScriptedModel([call("worker_parallel", tasks=["a", "b"]), "ok"], usage=ONE),
        tools=[fan_out(worker)],
    )
    result = await lead.run("go")
    assert all("error" in entry for entry in fan_out_result(result))
    assert result.usage.requests == 2 + 2  # the lead's two, and one billed response per worker

    @tool(read_only=True)
    async def wait_forever(query: str) -> str:
        """Search slowly."""
        await asyncio.Event().wait()
        return "never"

    stuck = Agent(
        FunctionModel(lambda request: call("wait_forever", query="q"), usage=ONE),
        name="w",
        tools=[wait_forever],
    )
    lead = Agent(
        ScriptedModel([call("w_parallel", tasks=["a"]), "ok"], usage=ONE),
        tools=[fan_out(stuck, timeout=0.1)],
    )
    result = await lead.run("go")
    assert result.messages[2].tool_results[0].is_error  # the fan-out timed out
    assert result.usage.requests == 3  # the stuck worker's request still counts


async def test_route_counts_the_usage_of_a_route_that_raises():
    triage = Agent(
        ScriptedModel(['"billing"'], usage=ONE),
        name="triage",
        output_type=Literal["billing", "technical"],
    )
    broken = Agent(
        ScriptedModel([call("missing_tool"), ModelServerError("down")], usage=ONE), name="billing"
    )

    @tool(read_only=True)
    async def ask_team(ctx: RunContext[Any], question: str) -> str:
        """Ask the right team."""
        routed = await route(triage, {"billing": broken, "technical": broken}, question, parent=ctx)
        return routed.result.text

    lead = Agent(
        ScriptedModel([call("ask_team", question="Invoice?"), "Done."], usage=ONE),
        tools=[ask_team],
    )
    result = await lead.run("go")
    assert result.messages[2].tool_results[0].is_error
    assert result.usage.requests == 2 + 1 + 1  # lead, classifier, the route's billed request


async def test_fan_out_is_not_read_only_by_default_so_a_resume_does_not_repeat_it():
    sent: list[str] = []
    started = asyncio.Event()

    @tool
    async def send_email(to: str) -> str:
        """Send an email."""
        sent.append(to)
        started.set()
        await asyncio.Event().wait()
        return "sent"

    mailer = Agent(ScriptedModel([call("send_email", to="a@x")]), name="mailer", tools=[send_email])
    parallel = fan_out(mailer)
    assert parallel.read_only is False and parallel.timeout == FAN_OUT_TIMEOUT_S
    lead_model = ScriptedModel([call("mailer_parallel", tasks=["mail a@x"]), "I am not sure."])
    lead = Agent(lead_model, name="lead", tools=[parallel])
    session = Session()
    running = asyncio.create_task(lead.run("go", session=session))
    await asyncio.wait_for(started.wait(), 5)
    running.cancel()  # the process stops while the worker sends the email
    with pytest.raises(asyncio.CancelledError):
        await running

    resumed = await lead.resume(session)
    assert resumed.status == "completed" and sent == ["a@x"]
    content = resumed.messages[2].tool_results[0].content
    assert content == prompts.TOOL_INTERRUPTED.format(name="mailer_parallel")


async def test_a_refusing_worker_is_an_error_not_an_answer():
    refusal = ModelResponse(
        message=Message.assistant("I cannot help with that."), finish_reason="content_filter"
    )

    class Facts(BaseModel):
        facts: list[str]

    plain = Agent(ScriptedModel([refusal]), name="plain")
    typed = Agent(ScriptedModel([refusal]), name="typed", output_type=Facts)
    lead = Agent(
        ScriptedModel(
            [[call("plain_parallel", tasks=["a"]), call("typed_parallel", tasks=["b"])], "ok"]
        ),
        tools=[fan_out(plain), fan_out(typed)],
    )
    result = await lead.run("go")
    entries = [json.loads(part.content) for part in result.messages[2].tool_results]
    assert entries == [
        [
            {
                "task": "a",
                "error": prompts.SUBAGENT_STOPPED.format(name="plain", reason="content_filter"),
            }
        ],
        [
            {
                "task": "b",
                "error": prompts.SUBAGENT_STOPPED.format(name="typed", reason="content_filter"),
            }
        ],
    ]


async def test_a_worker_too_deep_is_reported_without_a_traceback(caplog):
    worker_model = ScriptedModel([])
    worker = Agent(worker_model, name="worker")
    lead = Agent(
        ScriptedModel([call("worker_parallel", tasks=["a"]), "ok"]),
        tools=[fan_out(worker)],
        limits=Limits(max_depth=0),
    )
    with caplog.at_level(logging.ERROR, logger="qagents"):
        result = await lead.run("go")
    (entry,) = fan_out_result(result)
    assert entry["error"] == prompts.SUBAGENT_TOO_DEEP.format(name="worker", depth=1, limit=0)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert worker_model.requests == []


async def test_route_reads_quotes_and_markdown_around_a_final_full_stop():
    for answer in ['"billing".', "'billing'.", "**Billing**.", "`billing`."]:
        triage = Agent(ScriptedModel([answer]), name="triage")
        routed = await route(triage, specialists(), "hi")
        assert routed.route == "billing" and routed.used_default is False


async def test_route_says_when_it_took_the_default_and_can_journal_the_classifier():
    triage = Agent(ScriptedModel(["I am not sure."]), name="triage")
    routed = await route(triage, specialists(), "hi", default="technical")
    assert routed.route == "technical" and routed.used_default is True

    store = InMemoryStore()
    triage = Agent(ScriptedModel(["billing"]), name="triage")
    classifier_session = Session(store, id="triage-1")
    routed = await route(triage, specialists(), "Invoice?", classifier_session=classifier_session)
    assert routed.classification.session is classifier_session
    assert [m.text for m in await classifier_session.messages()] == ["Invoice?", "billing"]
