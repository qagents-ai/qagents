"""A lead agent plans a code change, delegates the edits to Devvi, then reviews the result.

The lead is a qagents `Agent`. Devvi (devvi.sh) is a coding agent with its own loop, tools,
sandbox and model, so it is not a model and not an MCP server: qagents drives it over ACP
as an external agent (`DevviAgent`), and the lead sees it as one tool,
``coder(task, session=None)``. Each call is a child run of the lead's tool call: Devvi's
events reach the lead's stream (printed with a ``[coder]`` prefix), its permission requests
are decided by the lead's `Permissions` on Devvi's own tool names, and the lead's limits
stop it. The lead reads only the coder's final message and the footer qagents adds (session,
tool calls, refused permissions, changed files), then checks the file itself with read-only
file tools.

What Devvi asks about: in ``workspace-write`` mode it sends a permission request only for
an escalation, a step that asks to leave its sandbox (a command or a file write with
``sandbox_permissions`` and a ``justification``). Reads, edits and plain commands inside the
workspace run without a request, and so do web fetches: Devvi's web tools have no approval
step, so a rule cannot stop them. This example therefore removes them with a patch row
(``{"id": "tool-web", "disabled": True}``), sends every escalated ``coder.bash`` or
``coder.pwsh`` command (``pwsh`` is Devvi's shell on Windows) to a reviewer that allows test
runs only, and refuses every other escalation. The deny rule on ``coder.web_fetch`` is
defense in depth, in case a later Devvi asks.

Run it from the project root (Windows PowerShell shown; use ``export`` elsewhere):

    python examples/devvi_coder.py --workspace C:\\tmp\\devvi-sandbox --dry-run

    $env:QAGENTS_DEVVI_BIN = "C:\\...\\devvi-harness\\apps\\cli\\lib\\bin.js"
    $env:DEEPSEEK_API_KEY = "<your key>"       # Devvi's model route; never commit it
    $env:QAGENTS_LEAD_MODEL = "anthropic:claude-opus-5-5"   # or qcloud:<model-id>
    python examples/devvi_coder.py --workspace C:\\tmp\\devvi-sandbox

``--dry-run`` needs no Devvi, no API key and no network: a `FunctionModel` plays the lead,
and the fake ACP peer of the test suite (``tests/integrations/fakes``) plays a Devvi turn
shaped like Devvi's recorded escalations: it reads and edits the file without asking, asks
to run the test outside its sandbox (the reviewer allows a test run), then asks to run
``pip install`` outside it (the reviewer refuses). Everything else is the real code path:
the launcher, the ACP client, the permission bridge and the footer. The dry run seeds the
workspace with a small parser and its test if they are missing; it never overwrites a file.

Without ``--dry-run`` the lead runs on ``--lead`` (or ``QAGENTS_LEAD_MODEL``), any qagents
model string such as ``anthropic:claude-opus-5-5`` or ``qcloud:<model-id>``, and Devvi on its
own route (``--devvi-model provider/model`` to choose one it offers; check with
``python -m qagents.integrations.devvi doctor``). Use a disposable workspace: Devvi edits it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qagents import Agent, Approval, Limits, Permissions, RunContext, Tool
from qagents.integrations.devvi import DevviAgent
from qagents.messages import ToolCallPart
from qagents.models import ModelRequest
from qagents.models.testing import FunctionModel, call
from qagents.toolkits.files import file_tools

NAME = "devvi_coder"
DESCRIPTION = "A lead agent plans a fix, delegates the edits to Devvi over ACP, then reviews it"

ROOT = Path(__file__).resolve().parents[1]
FAKES = ROOT / "tests" / "integrations" / "fakes"
ACP_FIXTURES = ROOT / "tests" / "fixtures" / "devvi" / "acp"

TASK = (
    "tests/test_parser.py fails: parse_line in src/parser.py drops the first field of every "
    "line. Fix it, run the test, and tell me what changed."
)

LEAD_INSTRUCTIONS = (
    "You lead a small code change. You never edit files yourself.\n"
    "1. Plan the fix in one or two sentences.\n"
    "2. Call coder once with a self-contained task: the file, the bug, the expected behaviour, "
    "and that it must run `python -m pytest tests/test_parser.py -q` when done.\n"
    "3. Read the changed file with read_file to review the fix.\n"
    "4. Answer in at most four sentences: what changed, whether the test passed, and which "
    "permissions were refused."
)

SEED = {
    "src/parser.py": (
        "def parse_line(line: str) -> list[str]:\n"
        '    """Split a comma-separated line into its fields."""\n'
        '    fields = line.strip().split(",")\n'
        "    return [fields[i].strip() for i in range(1, len(fields))]\n"
    ),
    "tests/test_parser.py": (
        "from src.parser import parse_line\n\n\n"
        "def test_parse_line():\n"
        '    assert parse_line("a, b ,c") == ["a", "b", "c"]\n'
    ),
}
"""The dry run's workspace: a parser with an off-by-one bug, and its test."""

FIXED = SEED["src/parser.py"].replace("range(1, len(fields))", "range(len(fields))")
"""``src/parser.py`` after the fake Devvi's edit."""

TEST_COMMANDS = ("python -m pytest", "pytest")
"""Commands the reviewer lets Devvi run outside its sandbox."""

SHELL = "pwsh" if sys.platform == "win32" else "bash"
"""Devvi's shell tool: its base profile mounts ``pwsh`` on Windows and ``bash`` elsewhere."""

NO_WEB = {"id": "tool-web", "disabled": True}
"""A Devvi patch row that removes ``web_fetch`` and ``web_search``, which never ask."""


# -- the rules: Permissions on Devvi's tool names ----------------------------------------------


def review(ctx: RunContext[Any], tool: Tool, request: ToolCallPart, reason: Any) -> Approval:
    """The reviewer (``on_ask``): it allows test runs and refuses everything else it is asked.

    Devvi asks only for escalations, so every request here wants to leave Devvi's sandbox.
    The reviewer looks at the command only: the ``justification`` and ``reason`` are text
    Devvi's model wrote. In a real setup this could ask a person (a UI, a chat message);
    here it is a rule.
    """
    command = str(request.arguments.get("command", ""))
    if tool.name.endswith((".bash", ".pwsh")) and command.startswith(TEST_COMMANDS):
        if not re.search(r"[;&|<>`$]", command):  # one plain command, nothing chained
            return Approval(approved=True)
    what = command or json.dumps(request.arguments)[:80]
    return Approval(approved=False, reason=f"the reviewer allows only test runs, not {what}")


def coder_permissions(name: str = "coder") -> Permissions:
    """What Devvi may do, by its own tool names prefixed with the tool name the lead sees.

    Devvi asks only for escalations (a step that wants to leave its sandbox): escalated
    commands go to the reviewer, and so does any other request, since a request no rule
    matches asks. Devvi does not ask before ``web_fetch``; `make_coder` removes the web
    tools, and the deny rule only matters if a later Devvi asks.
    """
    return Permissions(
        ask=[f"{name}.bash", f"{name}.pwsh"],
        deny=[f"{name}.web_fetch", f"{name}.web_search"],
        on_ask=review,
    )


# -- the agents --------------------------------------------------------------------------------


def make_coder(workspace: Path, **options: Any) -> DevviAgent:
    """Devvi as the coder: ACP (the default transport), one process for this workspace."""
    return DevviAgent(
        workspace=workspace,
        name="coder",
        permission_mode="workspace-write",  # Devvi's sandbox: writes inside the workspace only
        patches=[NO_WEB],  # Devvi never asks before a web fetch: remove its web tools
        max_external_turns=40,  # tool calls per delegation before qagents stops the turn
        allow_unmetered=True,  # ACP reports no tokens: the lead's cost cap counts the lead only
        **options,
    )


def make_lead(model: Any, coder: DevviAgent, workspace: Path, max_cost: float) -> Agent:
    return Agent(
        model,
        name="lead",
        instructions=LEAD_INSTRUCTIONS,
        tools=[coder.as_tool(name="coder", timeout=1800), *file_tools(workspace)],
        permissions=coder_permissions("coder"),
        limits=Limits(max_turns=8, max_cost_usd=max_cost, timeout_s=3600),
    )


# -- the dry run: a scripted lead and the fake Devvi --------------------------------------------


def lead_policy(request: ModelRequest) -> Any:
    """The scripted lead: delegate, review, answer. A function of each request."""
    last = request.messages[-1]
    if last.role != "tool":
        return call(
            "coder",
            task=(
                "In src/parser.py, parse_line skips the first field: it iterates from index 1. "
                "Make it return every field, stripped. Then run "
                "`python -m pytest tests/test_parser.py -q` and report the result."
            ),
        )
    result = last.tool_results[0]
    if result.name == "coder":
        return call("read_file", path="src/parser.py")
    if "range(len(fields))" not in result.content:
        return "The review failed: src/parser.py still skips the first field."
    return (
        "Devvi fixed parse_line in src/parser.py to keep every field (range(len(fields))). "
        "The reviewer let it run the test outside its sandbox, and the test passed. The "
        "reviewer refused its request to run pip install outside the sandbox."
    )


def fake_devvi_turn(directory: Path) -> Path:
    """A Devvi turn for the fake ACP peer: Devvi's recorded handshake, then a scripted turn.

    Like the real Devvi in ``workspace-write``, the fake asks only about escalations, shaped
    as in Devvi's ``snapshots/acp/escalation-approved`` (``tests/fixtures/devvi/acp``): the
    ``tool_call`` update with ``sandbox_permissions`` and ``justification`` in its
    ``rawInput``, then ``session/request_permission`` with ``allow-once`` and
    ``reject-once``. The read and the edit run without a request.
    """
    handshake = (ACP_FIXTURES / "handshake.jsonl").read_text(encoding="utf-8").splitlines()[1:5]

    def update(update_kind: str, **fields: Any) -> dict[str, Any]:
        body = {"sessionUpdate": update_kind, **fields}
        return {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": "{{sessionId}}", "update": body},
        }

    def tool(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return update(
            "tool_call", toolCallId=call_id, title=name, kind="other", status="in_progress",
            rawInput=arguments,
        )  # fmt: skip

    def done(call_id: str, text: str, status: str = "completed") -> dict[str, Any]:
        content = [{"type": "content", "content": {"type": "text", "text": text}}]
        return update("tool_call_update", toolCallId=call_id, status=status, content=content)

    def ask(request_id: int, call_id: str, chosen: str) -> list[dict[str, Any]]:
        options = [
            {"optionId": "allow-once", "name": "Allow once", "kind": "allow_once"},
            {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"},
        ]
        params = {"sessionId": "{{sessionId}}", "toolCall": {"toolCallId": call_id}}
        return [
            {"jsonrpc": "2.0", "id": request_id, "method": "session/request_permission",
             "params": {**params, "options": options}},
            {"answer": {"id": request_id, "accept": [
                {"outcome": {"outcome": "selected", "optionId": chosen}}]}},
        ]  # fmt: skip

    def text(update_kind: str, value: str) -> dict[str, Any]:
        content = {"type": "text", "text": value}
        return update(update_kind, messageId="{{messageId}}", content=content)

    def escalated(command: str, description: str, justification: str) -> dict[str, Any]:
        """A shell call that asks to leave the sandbox, as Devvi's model writes one."""
        return {
            "command": command,
            "description": description,
            "sandbox_permissions": "danger-full-access",
            "justification": justification,
        }

    test = "python -m pytest tests/test_parser.py -q"
    upgrade = "python -m pip install --upgrade pytest"
    refused = (
        'Error: the user rejected escalating this command to "danger-full-access"; it stays '
        "denied, so stop and explain instead of working around it"
    )  # Devvi's own text for a rejected escalation
    rows: list[Any] = [
        *handshake,
        {"expect": {"id": 3, "method": "session/prompt", "params": {"prompt": "$list"}}},
        text("agent_thought_chunk", "Read the parser, fix the loop, run the test."),
        tool("t1", "read", {"path": "src/parser.py"}),
        done("t1", SEED["src/parser.py"]),
        tool("t2", "edit", {"path": "src/parser.py", "old": "range(1, len(fields))",
                            "new": "range(len(fields))"}),
        {"write_file": {"path": "src/parser.py", "text": FIXED}},  # the fake's edit
        done("t2", "Edited src/parser.py"),
        tool("t3", SHELL, escalated(test, "Run the parser test",
                                    "the sandboxed shell cannot start Python in this workspace")),
        *ask(1, "t3", "allow-once"),
        done("t3", "1 passed in 0.02s"),
        tool("t4", SHELL, escalated(upgrade, "Update pytest",
                                    "installing a package needs the network")),
        *ask(2, "t4", "reject-once"),
        done("t4", refused, status="failed"),
        text("agent_message_chunk", "Fixed parse_line: it now keeps the first field. "
                                    "tests/test_parser.py passes (1 passed). I did not update "
                                    "pytest: that escalation was refused."),
        {"jsonrpc": "2.0", "id": 3, "result": {"stopReason": "end_turn"}},
    ]  # fmt: skip
    path = directory / "devvi-turn.jsonl"
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return path


def seed(workspace: Path) -> list[str]:
    """Write the dry run's files that are missing; never overwrite one.

    Raises:
        SystemExit: The workspace has a ``src/parser.py`` of its own, which the fake Devvi's
            edit would replace.
    """
    parser = workspace / "src" / "parser.py"
    if parser.exists() and parser.read_text(encoding="utf-8") not in (SEED["src/parser.py"], FIXED):
        raise SystemExit(
            f"{parser} is not the dry run's file; use an empty workspace for --dry-run."
        )
    written = []
    for relative, content in SEED.items():
        target = workspace / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(relative)
    return written


# -- running it ---------------------------------------------------------------------------------


@dataclass
class Outcome:
    ok: bool
    summary: str
    answer: str = ""
    decisions: list[str] = field(default_factory=list)
    footer: str = ""


def show(event: Any, outcome: Outcome) -> None:
    """Print the lead's events, and the coder's with a ``[coder]`` prefix."""
    if event.type == "subagent_event":
        inner = event.event
        if inner.type == "tool_call_started":
            print(f"  [coder] {inner.call.name} {json.dumps(inner.call.arguments)[:90]}")
        elif inner.type == "external_event" and inner.kind == "permission_decided":
            data = inner.data
            line = f"{data['tool']}: {data['outcome']} by {data['rule']}"
            outcome.decisions.append(line)
            print(f"  [coder] permission {line}")
        elif inner.type == "text_delta":
            print(f"  [coder] says: {inner.text}")
    elif event.type == "tool_call_started":
        print(f"lead calls {event.call.name} {json.dumps(event.call.arguments)[:90]}")
    elif event.type == "tool_call_finished" and event.call.name == "coder":
        outcome.footer = event.result.content.rsplit("\n\n", 1)[-1]
        print(f"lead reads the coder's footer: {outcome.footer}")


async def run(workspace: Path, *, dry_run: bool, lead: str | None, devvi_model: str | None,
              max_cost: float) -> Outcome:  # fmt: skip
    """Run the example once; the outcome says whether it did what it shows."""
    outcome = Outcome(ok=False, summary="")
    with tempfile.TemporaryDirectory(prefix="qagents-devvi-") as scratch:
        options: dict[str, Any] = {}
        if dry_run:
            seeded = seed(workspace)
            if seeded:
                print(f"seeded the workspace with {', '.join(seeded)}")
            fake = [sys.executable, "-I", str(FAKES / "fake_devvi_acp.py")]
            options["command"] = [*fake, "--script", str(fake_devvi_turn(Path(scratch)))]
            options["home"] = Path(scratch) / "devvi-home"  # never the user's Devvi home
            model: Any = FunctionModel(lead_policy)
        else:
            model = lead
            if devvi_model:
                provider, _, model_id = devvi_model.partition("/")
                options["model"] = (provider, model_id)
        coder = make_coder(workspace, **options)
        agent = make_lead(model, coder, workspace, max_cost)
        async with coder:
            stream = agent.stream(TASK)
            async for event in stream:
                show(event, outcome)
        result = stream.result
    assert result is not None
    outcome.answer = str(result.output or "")
    print(f"lead: {outcome.answer}")
    completed = result.status == "completed" and "; completed" in outcome.footer
    if dry_run:
        expected = [f"coder.{SHELL}: allow by on_ask", f"coder.{SHELL}: deny by on_ask"]
        reviewed = "range(len(fields))" in outcome.answer  # the lead read the fixed file
        outcome.ok = completed and reviewed and outcome.decisions == expected
    else:
        outcome.ok = completed
    outcome.summary = (
        f"{result.status}; coder: {outcome.footer or 'not called'}; "
        f"{len(outcome.decisions)} permission decisions"
    )
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python examples/devvi_coder.py", description=DESCRIPTION)
    parser.add_argument("--workspace", required=True, type=Path, help="a disposable directory")
    parser.add_argument("--dry-run", action="store_true", help="offline: scripted lead, fake Devvi")
    parser.add_argument(
        "--lead",
        default=os.environ.get("QAGENTS_LEAD_MODEL"),
        help="the lead's model, e.g. anthropic:claude-opus-5-5 (default: QAGENTS_LEAD_MODEL)",
    )
    parser.add_argument("--devvi-model", help="Devvi's model as provider/model, if not its default")
    parser.add_argument("--max-cost", type=float, default=0.50, help="USD for the lead's run")
    options = parser.parse_args(argv)
    if not options.dry_run and not options.lead:
        parser.error("give --lead (or set QAGENTS_LEAD_MODEL), or use --dry-run")
    if options.devvi_model and "/" not in options.devvi_model:
        parser.error("--devvi-model is provider/model, e.g. deepseek-official/deepseek-v4-pro")
    workspace = options.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    outcome = asyncio.run(
        run(
            workspace,
            dry_run=options.dry_run,
            lead=options.lead,
            devvi_model=options.devvi_model,
            max_cost=options.max_cost,
        )
    )
    print(f"{'ok' if outcome.ok else 'FAILED'}: {outcome.summary}")
    return 0 if outcome.ok else 1


if __name__ == "__main__":
    sys.exit(main())
