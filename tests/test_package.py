"""The package as a whole: exports, optional extras, prompts and the 0.2 integration fixes."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from typing import Any

import pytest

import qagents
from qagents import (
    Agent,
    Hooks,
    Message,
    Session,
    ToolCallPart,
    ToolResultPart,
    Usage,
    prompts,
    tool,
)
from qagents.context import Compactor
from qagents.models.base import ModelResponse, resolve_model
from qagents.models.testing import ScriptedModel, call

OPTIONAL = ("anthropic", "mlflow", "mcp", "mcp_types", "psycopg", "psycopg_pool", "httpx2")


def run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_version_is_0_2_0():
    assert qagents.__version__ == "0.2.0"


def test_import_qagents_loads_no_optional_dependency():
    done = run_python(
        f"""
        import sys
        import qagents, qagents.models, qagents.toolkits, qagents.integrations, qagents.stores
        from qagents import *
        loaded = sorted({{name.split(".")[0] for name in sys.modules}} & set({OPTIONAL!r}))
        print(loaded)
        """
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_qagents_works_when_the_optional_packages_are_missing():
    done = run_python(
        f"""
        import sys
        for name in {OPTIONAL!r}:
            sys.modules[name] = None  # every import of it now fails
        import inspect, pydoc
        import qagents, qagents.toolkits
        from qagents import *
        from qagents import AnthropicModel, MLflowHooks, PostgresStore
        try:
            from qagents import MCPToolkit
        except ImportError as error:
            print("from-import refused")
        for module in (qagents, qagents.toolkits):
            try:
                module.MCPToolkit
            except AttributeError as error:
                print("mcp:", "qagents[mcp]" in str(error))
            print(hasattr(module, "MCPToolkit"), "MCPToolkit" in dir(module))
            pydoc.render_doc(module)
            inspect.getmembers(module)
        """
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split("\n")[0] == "from-import refused"
    assert done.stdout.strip().split("\n")[1:] == ["mcp: True", "False False"] * 2


def test_lazy_names_resolve_to_their_modules():
    from qagents.integrations.agui import to_agui_events
    from qagents.integrations.mlflow import MLflowHooks
    from qagents.models.anthropic import AnthropicModel, AnthropicPricing
    from qagents.stores.postgres import PostgresStore
    from qagents.toolkits.mcp import MCPToolkit

    assert qagents.AnthropicModel is AnthropicModel
    assert qagents.models.AnthropicModel is AnthropicModel
    assert qagents.models.AnthropicPricing is AnthropicPricing
    assert qagents.MCPToolkit is MCPToolkit
    assert qagents.toolkits.MCPToolkit is MCPToolkit
    assert qagents.MLflowHooks is MLflowHooks
    assert qagents.integrations.MLflowHooks is MLflowHooks
    assert qagents.integrations.to_agui_events is to_agui_events
    assert qagents.PostgresStore is PostgresStore
    assert {"AnthropicModel", "MCPToolkit", "MLflowHooks", "PostgresStore"} <= set(dir(qagents))
    assert "MCPToolkit" not in qagents.__all__  # star imports never need an extra
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        _ = qagents.Nope


def test_every_name_in_all_is_importable():
    for module in (qagents, qagents.models, qagents.toolkits):
        for name in module.__all__:
            assert getattr(module, name) is not None, (module.__name__, name)


def test_the_anthropic_prefix_is_registered_without_importing_the_sdk():
    done = run_python(
        """
        import sys
        from qagents.models import resolve_model
        model = resolve_model("anthropic:claude-opus-5-5")
        print(type(model).__name__, model.name, "anthropic" in sys.modules)
        """
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["AnthropicModel", "claude-opus-5-5", "False"]
    assert type(resolve_model("anthropic:claude-haiku-5-5")).__name__ == "AnthropicModel"


def test_error_classes_live_in_errors_and_keep_their_old_paths():
    from qagents import errors
    from qagents.models.recording import CassetteMissError
    from qagents.runtime import RunInterrupted
    from qagents.session import LeaseError
    from qagents.toolkits.mcp import MCPConnectionError, MCPToolChangedError
    from qagents.workflow import WorkflowSuspended

    assert LeaseError is errors.LeaseError and issubclass(LeaseError, errors.SessionError)
    assert RunInterrupted is errors.RunInterrupted
    assert WorkflowSuspended is errors.WorkflowSuspended
    assert CassetteMissError is errors.CassetteMissError
    assert issubclass(CassetteMissError, errors.ModelError)
    assert MCPConnectionError is errors.MCPConnectionError
    assert MCPToolChangedError is errors.MCPToolChangedError
    assert issubclass(MCPToolChangedError, errors.ConfigurationError)


def test_model_facing_strings_live_only_in_prompts():
    import qagents.context
    import qagents.evals
    import qagents.integrations.agui
    import qagents.multi
    import qagents.toolkits.mcp

    moved = {
        qagents.multi: ["FAN_OUT_DESCRIPTION", "FAN_OUT_TASKS_DESCRIPTION", "FAN_OUT_TASK_FAILED"],
        qagents.context: ["TOOL_RESULT_CLEARED"],
        qagents.evals: ["JUDGE_INSTRUCTIONS", "JUDGE_PROMPT", "JUDGE_TASK", "JUDGE_REFERENCE"],
        qagents.integrations.agui: ["INPUT_CANCELLED"],
        qagents.toolkits.mcp: ["MCP_TOOL_TIMEOUT", "MCP_IMAGE_PLACEHOLDER", "MCP_TOOL_CHANGED"],
    }
    for module, names in moved.items():
        for name in names:
            assert isinstance(getattr(prompts, name), str)
            assert not hasattr(module, name), f"{module.__name__}.{name} should be in prompts"


async def test_as_tool_describes_its_task_parameter_with_a_prompt_constant(monkeypatch):
    monkeypatch.setattr(prompts, "SUBAGENT_TASK_DESCRIPTION", "Say everything.")
    helper = Agent(ScriptedModel(["done"]), name="helper")
    built = helper.as_tool()
    assert built.parameters["properties"]["task"]["description"] == "Say everything."


class Ends(Hooks):
    def __init__(self) -> None:
        self.ends: list[tuple[Any, type[BaseException] | None]] = []

    async def on_run_end(self, ctx, result, error):
        self.ends.append((result, type(error) if error is not None else None))


async def test_on_run_end_runs_when_the_stream_is_closed_early():
    hooks = Ends()
    agent = Agent(ScriptedModel(["one", "two"]), hooks=[hooks])
    stream = agent.stream("go")
    async with stream:
        async for event in stream:
            if event.type == "turn_started":
                break
    assert hooks.ends == [(None, GeneratorExit)]


async def test_on_run_end_runs_when_the_run_task_is_cancelled():
    hooks = Ends()
    started = asyncio.Event()

    @tool
    async def wait_forever() -> str:
        """Wait."""
        started.set()
        await asyncio.sleep(60)
        return "never"

    agent = Agent(ScriptedModel([call("wait_forever")]), tools=[wait_forever], hooks=[hooks])
    task = asyncio.create_task(agent.run("go"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert hooks.ends == [(None, asyncio.CancelledError)]


class Recorder(Compactor):
    def __init__(self) -> None:
        self.seen: list[ModelResponse] = []

    async def compact(self, ctx, messages, last_response, model):
        self.seen.append(last_response)
        return None


async def test_the_compactor_is_asked_before_the_first_request_of_a_run():
    recorder = Recorder()
    agent = Agent(ScriptedModel(["first", "second"]), compactor=recorder)
    session = Session()
    await agent.run("hello", session=session)
    await agent.run("again", session=session)
    assert len(recorder.seen) == 2  # one request per run, both asked
    assert all(response.usage == Usage() for response in recorder.seen)
    assert all(response.message == Message(role="assistant") for response in recorder.seen)


async def test_a_long_history_is_compacted_on_the_first_request():
    from qagents.context import ToolResultClearing

    clearing = ToolResultClearing(trigger_tokens=50, keep_last=0)
    agent = Agent(ScriptedModel(["ok"]), compactor=clearing)
    history = [
        Message.user("look it up"),
        Message.assistant(tool_calls=[ToolCallPart(id="c1", name="lookup")]),
        Message(
            role="tool",
            parts=[ToolResultPart(tool_call_id="c1", name="lookup", content="x" * 2000)],
        ),
        Message.assistant("found it"),
    ]
    session = Session()
    result = await agent.run("and now?", session=session, history=history)
    assert result.status == "completed"
    cleared = [part for message in result.messages for part in message.tool_results]
    assert cleared and cleared[0].content == prompts.TOOL_RESULT_CLEARED.format(name="lookup")


async def test_signed_pauses_resume_through_agui():
    from qagents import Permissions
    from qagents.integrations.agui import agui_resume, to_agui_events

    refunds: list[str] = []

    @tool(requires_approval=True)
    def refund(order: str) -> str:
        """Refund an order."""
        refunds.append(order)
        return "refunded"

    @tool(deferred=True)
    def fetch_report(report_id: str) -> str:
        """Fetch a report another system produces."""
        raise AssertionError("never runs")

    agent = Agent(
        ScriptedModel([[call("refund", order="A-1"), call("fetch_report", report_id="r")], "ok"]),
        tools=[refund, fetch_report],
        permissions=Permissions(approval_secret="s3cret-for-tests-only-0123456789"),
    )
    session = Session()
    events = [event async for event in to_agui_events(agent.stream("go", session=session))]
    outcome = events[-1]["outcome"]
    assert outcome["type"] == "interrupt"
    answers = []
    for interrupt in outcome["interrupts"]:
        token = interrupt["metadata"]["token"]
        if interrupt["id"].startswith("approval:"):
            payload: Any = {"approved": True, "token": token}
        else:
            payload = {"token": token, "value": "the report"}
        answers.append({"interruptId": interrupt["id"], "status": "resolved", "payload": payload})
    result = await agent.resume(session, **agui_resume(answers))
    assert (result.status, refunds) == ("completed", ["A-1"])


async def test_agui_native_mode_keeps_parallel_children_of_one_call_apart():
    from qagents.integrations.agui import to_agui_events
    from qagents.models.testing import FunctionModel
    from qagents.multi import fan_out

    def work(request) -> str:
        task = next(message.text for message in request.messages if message.role == "user")
        return f"facts about {task}"

    researcher = Agent(FunctionModel(work), name="researcher")
    lead = Agent(
        ScriptedModel([call("researcher_parallel", tasks=["Spain", "Greece"]), "Done."]),
        name="lead",
        tools=[fan_out(researcher)],
    )
    events = [e async for e in to_agui_events(lead.stream("go"), run_id="r", subagents="native")]
    started = [e for e in events if e["type"] == "SUBAGENT_STARTED"]
    assert sorted(e["name"] for e in started) == ["researcher[0]", "researcher[1]"]
    ids = {e["subagentRunId"] for e in started}
    assert len(ids) == 2
    finished = [e for e in events if e["type"] == "SUBAGENT_FINISHED"]
    assert {e["subagentRunId"] for e in finished} == ids
    texts: dict[str, str] = {}
    for event in events:
        if event["type"] == "TEXT_MESSAGE_CONTENT" and "subagentRunId" in event:
            child = event["subagentRunId"]
            texts[child] = texts.get(child, "") + event["delta"]
    assert sorted(texts.values()) == ["facts about Greece", "facts about Spain"]
