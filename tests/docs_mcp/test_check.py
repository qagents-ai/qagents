"""check_code: invented names are caught, correct code passes."""

from __future__ import annotations

import sys
import textwrap

import pytest

import qagents

from .conftest import ROOT, python_blocks

CORRECT = '''
import asyncio

from qagents import Agent, Limits, RunContext, Session, SQLiteStore, Workflow, tool
from qagents.models.testing import FunctionModel, ScriptedModel, call


@tool(requires_approval=True, idempotent=True)
async def refund(ctx: RunContext, order_id: str) -> str:
    """Refund an order."""
    return f"{order_id} {ctx.operation_id} {ctx.deps}"


async def main() -> None:
    model = ScriptedModel([call("refund", order_id="A-1"), "Done."])
    agent = Agent(model, instructions="Be brief.", tools=[refund], limits=Limits(max_turns=5))
    session = Session()
    result = await agent.run("Refund A-1", session=session)
    for pending in result.pending_approvals:
        print(pending.tool_call_id, pending.name, pending.arguments)
    done = await agent.resume(session, approvals={result.pending_approvals[0].tool_call_id: True})
    print(done.output, done.usage.requests, model.requests[0].messages[0].role)
    async with SQLiteStore("jobs.db") as store:
        wf = Workflow(store, id="job")
        step = await wf.run_agent("one", agent, "Go", limits=Limits(max_turns=2))
        await wf.step("two", lambda: step.text)
    stream = agent.stream("hi")
    async for event in stream:
        print(event.type)
    print(stream.result.output, FunctionModel(lambda request: "x").requests)


asyncio.run(main())
'''


def problems(service, code):
    return [p.message for p in service.check(textwrap.dedent(code)).problems]


def test_correct_code_passes(service):
    report = service.check(CORRECT)
    assert report.problems == []
    assert report.counts["calls"] >= 15 and report.counts["attributes"] >= 15
    text = report.render(service.version)
    assert text.startswith(f"# check_code: no problems (qagents {service.version})")


WRONG = """
from qagents import Agent, Tools, ScriptedModel
from qagents.testing import FunctionModel

agent = Agent("qcloud:x", system_prompt="Be brief.", max_iterations=3)


async def main():
    result = agent.run("hi")
    later = await agent.run("hi", thread_id="t")
    print(later.final_output)
    for pending in later.pending_approvals:
        print(pending.tool_call_idd)
    agent.run_sync("x")
"""


def test_invented_names_are_caught_with_the_real_names(service):
    found = problems(service, WRONG)
    expected = [
        "`qagents` has no `Tools`. Did you mean: `Tool`",
        "`ScriptedModel` is in `qagents.models.testing`: "
        "`from qagents.models.testing import ScriptedModel`",
        "`qagents.testing` does not exist",
        "`Agent()` has no parameter `system_prompt`. qagents calls it `instructions`",
        "`Agent()` has no parameter `max_iterations`.",
        "Turn limits are `limits=Limits(max_turns=...)`",
        "`Agent.run()` has no parameter `thread_id`. Conversations are `session=Session(",
        "`later (RunResult)` has no attribute `final_output`. Did you mean: `output`?",
        "`pending (PendingApproval)` has no attribute `tool_call_idd`. Did you mean: `tool_call_id`?",
        "`Agent.run()` is a coroutine function and is not awaited here",
        "`Agent.run_sync()` starts its own event loop and fails inside `async def`",
    ]
    for piece in expected:
        assert any(piece in message for message in found), (piece, found)
    report = service.check(WRONG)
    assert len(report.warnings) == 2 and len(report.errors) == 8


def test_the_report_lists_problems_with_lines(service):
    text = service.check_code(WRONG)
    assert text.startswith("# check_code: 8 problem(s)")
    assert "## Problems" in text and "## Warnings" in text
    assert "1. line 2: `qagents` has no `Tools`" in text
    assert "Static check only" in text


def test_syntax_errors(service):
    report = service.check("def broken(:\n    pass\n")
    assert len(report.problems) == 1 and "does not compile" in report.problems[0].message


def test_top_level_await_is_allowed(service):
    assert problems(service, "from qagents import Agent\nr = await Agent('x').run('hi')\n") == []


def test_keyword_arguments_passed_on_are_checked(service):
    found = problems(
        service,
        """
        from qagents import Agent, Workflow
        agent = Agent("x")
        agent.run_sync("hi", limit=3)
        async def go(wf: Workflow):
            await wf.run_agent("s", agent, "p", sesion=None)
    """,
    )
    assert any(
        "`Agent.run_sync()` has no parameter `limit`. Did you mean: `limits`" in m for m in found
    )
    assert any("`Workflow.run_agent()` has no parameter `sesion`" in m for m in found)


def test_options_passed_to_the_base_class_are_checked(service):
    """``DevviAgent`` and friends pass ``**options`` to ``super().__init__``: typos there count."""
    found = problems(
        service,
        """
        from qagents.external import ScriptedExternalAgent
        from qagents.integrations.acp import ACPAgent
        from qagents.integrations.devvi import DevviAgent
        DevviAgent(workspace=".", on_permision="pause")
        ScriptedExternalAgent([], cancel_grace=1)
        ACPAgent(["agent", "--acp"], permisions=None)
    """,
    )
    for piece in [
        "has no parameter `on_permision`. Did you mean: `on_permission`",
        "has no parameter `cancel_grace`. Did you mean: `cancel_grace_s`",
        "has no parameter `permisions`. Did you mean: `permissions`",
    ]:
        assert any(piece in message for message in found), (piece, found)
    assert len(found) == 3, found


def test_options_the_base_class_accepts_pass(service):
    code = """
        from qagents import Permissions
        from qagents.external import ScriptedExternalAgent
        from qagents.integrations.devvi import DevviAgent
        DevviAgent(workspace=".", on_permission="pause", allow_unmetered=True, track_changes=False)
        ScriptedExternalAgent(
            [], workspace=".", cancel_grace_s=1.0, max_external_turns=3,
            permissions=Permissions(allow=["scripted.*"]), approval_timeout_s=5.0,
        )
    """
    assert problems(service, code) == []


def test_decorators_and_tools(service):
    found = problems(
        service,
        """
        from qagents import tool
        @tool(read_only=True, needs_approval=True)
        def lookup(city: str) -> str:
            '''Weather.'''
            return city
        lookup.requires_approvall
    """,
    )
    assert any(
        "`tool()` has no parameter `needs_approval`. Did you mean: `requires_approval`" in m
        for m in found
    )
    assert any("`lookup (Tool)` has no attribute `requires_approvall`" in m for m in found)


def test_positional_and_missing_arguments(service):
    found = problems(
        service,
        """
        from qagents import Workflow, InMemoryStore
        Workflow(InMemoryStore())
        Workflow(InMemoryStore(), "id", "extra")
    """,
    )
    assert any("is missing required argument(s): id" in m for m in found)
    assert any("takes at most 2 positional argument(s)" in m for m in found)


def test_classmethods_and_context_managers(service):
    found = problems(
        service,
        """
        from qagents.toolkits.mcp import MCPToolkit
        async def go():
            async with MCPToolkit.stdio("npx", ["server"], prefixx="gh") as tk:
                await tk.toolz()
    """,
    )
    assert any("has no parameter `prefixx`. Did you mean: `prefix`?" in m for m in found), found
    assert any("has no attribute `toolz`. Did you mean: `tools`" in m for m in found)


MODEL_WITH_ITS_OWN_INIT = """
    from qagents.external import ExternalApproval
    yes = ExternalApproval(True, reason="ok")
    no = ExternalApproval(approved=False)
    default = ExternalApproval()
    named = ExternalApproval(False, tool_call_id="c1", name="coder", is_error=True)
"""


def test_a_model_with_its_own_init_is_checked_against_that_init(service):
    """``ExternalApproval.__init__`` fills in ``tool_call_id``, ``name`` and ``content``.

    pydantic's own signature of the class lists them as required keywords, so a check
    against it rejected the documented ``ExternalApproval(approved=...)``.
    """
    assert problems(service, MODEL_WITH_ITS_OWN_INIT) == []
    namespace: dict[str, object] = {}
    exec(textwrap.dedent(MODEL_WITH_ITS_OWN_INIT), namespace)  # noqa: S102 - it really works
    assert namespace["no"].is_error and namespace["named"].tool_call_id == "c1"

    found = problems(
        service,
        """
        from qagents.external import ExternalApproval
        ExternalApproval(True, reasn="x")
        ExternalApproval(True, False)
        ExternalApproval(tool_call_idd="c1")
    """,
    )
    assert any("has no parameter `reasn`. Did you mean: `reason`" in m for m in found), found
    assert any("takes at most 1 positional argument(s)" in m for m in found), found
    assert any("has no parameter `tool_call_idd`. Did you mean: `tool_call_id`" in m for m in found)
    assert not any("missing required" in m for m in found) and len(found) == 3, found


def test_a_model_without_its_own_init_still_needs_its_fields(service):
    found = problems(
        service,
        """
        from qagents.messages import ToolResultPart
        ToolResultPart(content="x")
    """,
    )
    assert any("is missing required argument(s): tool_call_id, name" in m for m in found), found


def test_attributes_annotated_where_they_are_assigned_are_followed(service):
    """``self.result: ExternalResult | None = None`` in ``__init__`` types ``turn.result``."""
    found = problems(
        service,
        """
        from qagents import Agent
        from qagents.external import ScriptedExternalAgent

        async def main():
            coder = ScriptedExternalAgent([])
            turn = coder.stream("task")
            async for event in turn:
                print(event.type)
            print(turn.result.changed_files, turn.result.session_id, turn.result.changed)
            stream = Agent("qcloud:x").stream("hi")
            async for event in stream:
                print(event.type)
            print(stream.result.output, stream.result.usage, stream.result.outputs)
    """,
    )
    assert any(
        "`result (ExternalResult)` has no attribute `changed`. Did you mean: `changed_files`" in m
        for m in found
    ), found
    assert any(
        "`result (RunResult)` has no attribute `outputs`. Did you mean: `output`" in m
        for m in found
    ), found
    assert len(found) == 2, found


def test_an_attribute_assigned_without_annotation_stays_unchecked():
    """``self.context = context`` says nothing about the type: it is known, and left alone."""
    from qagents.docs_mcp.check import _instance_member
    from qagents.external import ExternalResult, ExternalTurn

    assert _instance_member(ExternalTurn, "result") == (True, "attribute", ExternalResult | None)
    assert _instance_member(ExternalTurn, "context") == (True, "unknown", None)


def test_code_without_qagents_is_only_compiled(service):
    report = service.check("import os\nprint(os.getcwd())\n")
    assert report.problems == []
    assert "only its syntax was checked" in report.render(service.version)


def test_a_missing_extra_is_reported_as_such(service, monkeypatch):
    qagents.MCPToolkit  # noqa: B018 - resolve the lazy name, so monkeypatch can restore it
    monkeypatch.delattr(qagents, "MCPToolkit", raising=False)
    monkeypatch.delitem(sys.modules, "qagents.toolkits.mcp", raising=False)
    monkeypatch.setitem(sys.modules, "mcp_types", None)  # importing the SDK now fails
    found = problems(service, "from qagents.toolkits.mcp import MCPToolkit\n")
    assert len(found) == 1 and "pip install 'qagents[mcp]'" in found[0]


@pytest.mark.parametrize("page", ["AGENTS.md", "skills/qagents/SKILL.md"])
def test_the_documentation_for_agents_uses_only_real_names(service, page):
    blocks = python_blocks(ROOT / page)
    assert blocks
    for line, code in blocks:
        report = service.check(code)
        assert report.problems == [], f"{page}:{line}: {report.problems}"


def test_main_modules_are_never_imported(service, monkeypatch):
    monkeypatch.delitem(sys.modules, "qagents.__main__", raising=False)
    monkeypatch.delitem(sys.modules, "qagents.docs_mcp.__main__", raising=False)
    code = "import qagents.__main__\nfrom qagents.docs_mcp import __main__\n"
    assert service.check(code).problems == []
    assert "has no public name" in service.get_api("qagents.__main__")
    assert "qagents.__main__" not in sys.modules
    assert "qagents.docs_mcp.__main__" not in sys.modules
