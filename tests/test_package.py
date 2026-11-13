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
    RunContext,
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


# --- external agents, ACP, Devvi and the MCP tool server ----------------------

NEW_MODULES = (
    "qagents.external",
    "qagents.integrations.acp",
    "qagents.integrations.devvi",
    "qagents.integrations.mcp_server",
    "qagents.docs_mcp",
    "mcp",
    "mcp_types",
)
EXTERNAL_NAMES = (
    "ExternalAgent",
    "ExternalApproval",
    "ExternalCapabilities",
    "ExternalResult",
    "ScriptedExternalAgent",
)
INTEGRATION_NAMES = {
    "ACPAgent": "qagents.integrations.acp",
    "MCPServerHTTP": "qagents.integrations.acp",
    "MCPServerStdio": "qagents.integrations.acp",
    "DevviAgent": "qagents.integrations.devvi",
    "TokenPricing": "qagents.integrations.devvi",
    "ToolServer": "qagents.integrations.mcp_server",
    "ToolServerSpec": "qagents.integrations.mcp_server",
    "tool_server": "qagents.integrations.mcp_server",
}


def test_import_qagents_loads_no_external_agent_module():
    done = run_python(
        f"""
        import sys
        import qagents, qagents.integrations
        from qagents import *
        from qagents.integrations import *
        names = {NEW_MODULES!r}
        print(sorted(m for m in sys.modules if m in names or m.startswith(tuple(n + "." for n in names))))
        """
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_the_external_agent_names_resolve_lazily():
    import importlib

    import qagents.integrations

    external = importlib.import_module("qagents.external")
    for name in EXTERNAL_NAMES:
        assert getattr(qagents, name) is getattr(external, name)
        assert name in dir(qagents) and name not in qagents.__all__
    for name, module in INTEGRATION_NAMES.items():
        assert getattr(qagents.integrations, name) is getattr(importlib.import_module(module), name)
        assert name in dir(qagents.integrations)


def test_the_tool_server_names_say_which_extra_is_missing():
    done = run_python(
        """
        import sys
        for name in ("mcp", "mcp_types"):
            sys.modules[name] = None  # every import of it now fails
        import inspect, pydoc
        import qagents.integrations as integrations
        try:
            from qagents.integrations import tool_server
        except ImportError:
            print("from-import refused")
        try:
            integrations.ToolServer
        except AttributeError as error:
            print("mcp:", "qagents[mcp]" in str(error), type(error.__cause__).__name__)
        print(hasattr(integrations, "tool_server"), "tool_server" in dir(integrations))
        pydoc.render_doc(integrations)
        inspect.getmembers(integrations)
        print(integrations.DevviAgent.__name__, integrations.ACPAgent.__name__)
        """
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split("\n")[:4] == [
        "from-import refused",
        "mcp: True ImportError",
        "False False",
        "DevviAgent ACPAgent",
    ]


def test_external_agent_event_is_one_of_the_run_events():
    import typing

    from qagents import events, external

    assert events.ExternalAgentEvent in typing.get_args(events.Event)
    assert external.ExternalAgentEvent is events.ExternalAgentEvent
    assert "ExternalAgentEvent" in events.__all__ and "ExternalAgentEvent" in external.__all__
    event = events.ExternalAgentEvent(agent="coder", kind="turn_finished")
    assert event.model_dump() == {
        "type": "external_event",
        "agent": "coder",
        "kind": "turn_finished",
        "session_id": None,
        "data": {},
    }


def test_external_agent_strings_live_in_prompts():
    import ast
    import importlib
    import importlib.util
    import inspect
    import types

    for gone in (
        "qagents.integrations._external_prompts",
        "qagents.integrations.acp._prompts",
        "qagents.integrations.devvi._prompts",
    ):
        assert importlib.util.find_spec(gone) is None, gone
    body = ast.parse(inspect.getsource(prompts)).body
    documented = {
        node.targets[0].id
        for node, after in zip(body, body[1:], strict=False)
        if isinstance(node, ast.Assign)
        and isinstance(after, ast.Expr)
        and isinstance(after.value, ast.Constant)
        and isinstance(after.value.value, str)
    }
    prefixes = ("EXTERNAL_", "TOOL_SERVER_", "ACP_", "DEVVI_")
    names = [name for name in vars(prompts) if name.startswith(prefixes)]
    assert len(names) == 191
    assert all(isinstance(getattr(prompts, name), str) for name in names)
    assert set(names) <= documented
    readers = (
        "qagents.external",
        "qagents.integrations._jsonrpc",
        "qagents.integrations._process",
        "qagents.integrations.mcp_server",
        "qagents.integrations.acp.client",
        "qagents.integrations.acp.mapping",
        "qagents.integrations.devvi",
        "qagents.integrations.devvi.sdk",
        "qagents.integrations.devvi.acp",
        "qagents.integrations.devvi.events",
        "qagents.integrations.devvi.launcher",
        "qagents.integrations.devvi.patches",
        "qagents.integrations._process",
        "qagents.integrations._jsonrpc",
    )
    for reader in readers:
        module = importlib.import_module(reader)
        used = [
            value
            for value in vars(module).values()
            if isinstance(value, types.ModuleType) and value.__name__.endswith("prompts")
        ]
        assert used and all(value is prompts for value in used), reader


async def test_a_new_prompt_value_reaches_the_external_agent_tool(monkeypatch):
    from qagents.external import ScriptedExternalAgent

    monkeypatch.setattr(prompts, "EXTERNAL_AGENT_DESCRIPTION", "Ask {name}.")
    built = ScriptedExternalAgent([["done"]], name="coder").as_tool()
    assert built.description.startswith("Ask coder.")


async def test_new_values_of_the_external_agent_joiners_reach_their_readers(monkeypatch, tmp_path):
    """One constant (or more) from every module that builds external-agent text: a new value
    reaches the calling model, the external agent, or the MCP client's model."""
    from qagents.errors import ToolError
    from qagents.external import (
        STATE_SESSIONS,
        ExternalApproval,
        PermissionStep,
        ScriptedExternalAgent,
        ToolStep,
    )
    from qagents.external.permissions import ExternalTool
    from qagents.integrations._jsonrpc import JsonRpcError
    from qagents.integrations.acp import mapping
    from qagents.integrations.devvi import DevviAgent
    from qagents.integrations.devvi import events as devvi_events
    from qagents.integrations.devvi import sdk as devvi_sdk
    from qagents.integrations.mcp_server import _note_edit, _Verdict
    from qagents.messages import TextPart

    for name, value in {
        "EXTERNAL_AGENT_FOOTER_TOOL_COUNT": "{tool} x{count}",  # contract.py
        "EXTERNAL_AGENT_FOOTER_LIST_SEPARATOR": " | ",
        "EXTERNAL_AGENT_NO_ARGUMENTS": "(none)",  # permissions.py
        "EXTERNAL_AGENT_DECLINED_REASON": " because {reason}",
        "EXTERNAL_TOOL_DESCRIPTION": "{agent} runs {tool}.",
        "EXTERNAL_TOOL_NEVER_RUN": "Not here.",
        "EXTERNAL_AGENT_TASK_SEPARATOR": " / ",  # turn.py
        "EXTERNAL_AGENT_FORGOTTEN_SESSION": "forgot {session} ({limit})",  # tool.py
        "EXTERNAL_AGENT_UNKNOWN_SESSION": "never saw {session}",
        "ACP_DIFF_PLACEHOLDER": "<diff {path}>",  # acp/mapping.py
        "ACP_TOOL_CONTENT_SEPARATOR": " + ",
        "ACP_JSONRPC_ERROR": "code {code}",
        "EXTERNAL_AGENT_DESCRIPTION_WITH_HINT": "{hint} {description}",  # devvi/__init__.py
        "EXTERNAL_AGENT_LINE_SHORTENED": "~",  # devvi/sdk.py
        "DEVVI_UNKNOWN_TURN_END": "no end",  # devvi/events.py
        "APPROVAL_ARGUMENTS_CHANGED_RESULT": "{content} [{note}]",  # mcp_server.py
    }.items():
        monkeypatch.setattr(prompts, name, value)

    # The calling model's tool result: the footer's tool counts and refused requests.
    external = ScriptedExternalAgent(
        [[ToolStep("read"), ToolStep("read"), ToolStep("bash"), PermissionStep("edit"), "ok"]],
        name="coder",
    )
    model = ScriptedModel([call("coder", task="Go."), "Done."])
    result = await Agent(model, tools=[external.as_tool()]).run("go")
    (content,) = [p.content for m in result.messages for p in m.tool_results]
    assert "3 tool calls (read x2 | bash x1)" in content
    assert "1 permission refused (edit: (none))" in content

    # The approval texts, and the stand-in tool of a permission decision.
    assert ExternalApproval(False, reason="risky").content.startswith("Declined because risky.")
    stand_in = ExternalTool("coder.bash", agent="coder", external_name="bash")
    assert stand_in.description == "coder runs bash."
    with pytest.raises(ToolError, match="^Not here.$"):
        stand_in.function()

    # The external agent's prompt text of a task in several parts.
    scripted = ScriptedExternalAgent([["ok"]], name="coder")
    await scripted.run([TextPart(text="a"), TextPart(text="b")])
    assert scripted.calls[0].task == "a / b"

    # The model passes a session the tool does not know: with few sessions kept, and with
    # the tool's full 50 (the call before fills its list).
    @tool
    def fill(ctx: RunContext) -> str:
        """Pretend the coder tool opened 50 sessions."""
        old = [{"session_id": f"old-{i}", "cwd": None, "backend": None} for i in range(50)]
        ctx.state[STATE_SESSIONS] = {"coder": old}
        return "filled"

    model = ScriptedModel(
        [
            call("coder", task="Go on.", session="s-9"),
            call("fill"),
            call("coder", task="Go on.", session="s-9"),
            "Done.",
        ]
    )
    result = await Agent(model, tools=[external.as_tool(), fill]).run("go")
    unknown, _, forgotten = [p.content for m in result.messages for p in m.tool_results]
    assert unknown == prompts.EXTERNAL_AGENT_FAILED.format(name="coder", error="never saw s-9")
    assert forgotten == prompts.EXTERNAL_AGENT_FAILED.format(name="coder", error="forgot s-9 (50)")

    # ACP: a tool call's result content, and a stop reason the model reads.
    pieces = [{"type": "content", "content": {"type": "text", "text": "a"}}, {"type": "diff"}]
    assert mapping.content_text(pieces) == "a + <diff >"
    failed = mapping.outcome_for_prompt_error(JsonRpcError(-32000, ""), cancelling=False)
    assert failed.stop_reason == "code -32000"

    # Devvi: the tool description, a shortened stop reason, an unknown turn end.
    devvi = DevviAgent(
        workspace=tmp_path, transport="sdk", command=["node", "devvi.js"], home=tmp_path / "h"
    )
    assert devvi.as_tool().description.startswith(prompts.EXTERNAL_AGENT_CONTINUE_HINT + " ")
    assert devvi_sdk._first_line("x" * 400, 10) == "x" * 9 + "~"
    assert devvi_events.turn_end_outcome(None).stop_reason == "no end"

    # The MCP tool server: a result that ran with an approver's edited arguments.
    ran = ToolResultPart(tool_call_id="c", name="send", content="sent")
    edited = _note_edit(ran, _Verdict(ToolCallPart(id="c", name="send"), edited=True))
    assert edited.content.startswith("sent [") and edited.content.endswith("]")


@pytest.mark.parametrize("with_tools", [False, True])
async def test_core_unknown_tool_uses_shared_prompt_joiners(monkeypatch, with_tools):
    monkeypatch.setattr(prompts, "UNKNOWN_TOOL_SEPARATOR", " / ")
    monkeypatch.setattr(prompts, "UNKNOWN_TOOL_NONE", "nothing")

    @tool
    def one() -> str:
        """Return one."""
        return "one"

    @tool
    def two() -> str:
        """Return two."""
        return "two"

    model = ScriptedModel([call("missing"), "done"])
    await Agent(model, tools=[one, two] if with_tools else []).run("go")
    content = model.requests[1].messages[-1].tool_results[0].content
    assert content == prompts.UNKNOWN_TOOL.format(
        name="missing", available="one / two" if with_tools else "nothing"
    )


@pytest.mark.parametrize("outside", [False, True])
async def test_core_edited_tool_result_uses_shared_prompt_layout(monkeypatch, outside):
    from qagents import Approval, Permissions

    monkeypatch.setattr(prompts, "APPROVAL_ARGUMENTS_CHANGED_RESULT", "{content} <{note}>")

    @tool(requires_approval=True)
    def send(value: str) -> str:
        """Send the approved value."""
        return value

    @tool
    async def delegate(ctx: RunContext) -> str:
        """Ask the run to call send."""
        return (await ctx.call_tool(send, {"value": "original"}, source="test")).content

    permissions = Permissions(
        on_ask=lambda *_: Approval(approved=True, arguments={"value": "edited"})
    )
    calls = [call("delegate")] if outside else [call("send", value="original")]
    model = ScriptedModel([*calls, "done"])
    await Agent(model, tools=[delegate] if outside else [send], permissions=permissions).run("go")
    content = model.requests[1].messages[-1].tool_results[0].content
    assert content.startswith("edited <The approver changed the arguments") and content.endswith(
        ">"
    )


async def test_core_tool_and_fan_out_errors_use_shared_exception_text(monkeypatch):
    from qagents.models.testing import FunctionModel
    from qagents.multi import fan_out

    monkeypatch.setattr(prompts, "TOOL_FAILED_EXCEPTION", "<{type}> {error}")

    @tool
    def broken() -> str:
        """Raise an ordinary tool error."""
        raise ValueError("bad input")

    def fail(request):
        raise RuntimeError("worker unavailable")

    worker = Agent(FunctionModel(fail), name="worker")
    model = ScriptedModel([call("broken"), call("worker_parallel", tasks=["go"]), "done"])
    result = await Agent(model, tools=[broken, fan_out(worker)]).run("go")
    contents = [p.content for m in result.messages for p in m.tool_results]
    assert contents[0] == "'broken' failed: <ValueError> bad input"
    assert "<RuntimeError> worker unavailable" in contents[1]


def test_pyproject_has_the_external_agent_extras_and_bundles_the_docs():
    import tomllib
    from pathlib import Path

    from qagents.docs_mcp import sources

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    assert extras["acp"] == [] and extras["devvi"] == []
    bundled = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    package = Path(sources.__file__).resolve().parents[2]  # the folder that holds qagents/
    for source, target in bundled.items():
        assert (root / source).exists(), source
        assert (package / target).is_relative_to(sources.BUNDLE_DIR), target
    for needed in ("docs", "examples", "README.md", "AGENTS.md", sources.SKILL_PATH, "mkdocs.yml"):
        assert bundled[needed] == f"qagents/docs_mcp/_bundled/{needed}"


def test_the_agents_md_layout_names_only_paths_that_exist_and_every_cli_command():
    """The docs server hands AGENTS.md to coding agents: its layout must not send them to a
    file the external/ split removed, and it must list every subcommand of ``qagents``."""
    import re
    from pathlib import Path

    from qagents.cli import build_parser

    root = Path(__file__).resolve().parents[1]
    text = (root / "AGENTS.md").read_text(encoding="utf-8")
    layout = re.search(r"### Layout\s+```text\n(.*?)^```", text, re.DOTALL | re.MULTILINE)
    assert layout, "AGENTS.md Part 2 needs a ### Layout block"
    named = re.findall(r"^(src/qagents/\S+)", layout.group(1), re.MULTILINE)
    assert "src/qagents/external/" in named
    for path in named:
        assert (root / path).exists(), f"AGENTS.md's layout names {path}, which does not exist"
    listed = re.search(r"cli\.py \(the qagents command: ([^)]*)\)", layout.group(1))
    assert listed, "the layout must list the qagents subcommands"
    parser = build_parser()
    commands = next(a for a in parser._actions if a.dest == "command").choices
    assert listed.group(1).split(", ") == list(commands)
