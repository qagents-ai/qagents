"""The external-agent pages say what the code does.

Pages: ``docs/concepts/external-agents.md`` and ``docs/integrations/{devvi,acp,mcp-server}.md``.
The offline tests run the pages' own snippets and rules against the real classes, and compare
the pages' tables (options and their defaults, stop reasons, event kinds, protocol steps) with
the code in both directions. One ACP test drives the fake ACP agent
(``tests/integrations/fakes/fake_acp_agent.py``) with a rule written as the ACP page teaches.
The two ``devvi_mock`` tests (opt-in, ``-m devvi_mock``, see ``conftest.py``) check what the
Devvi page says Devvi asks permission for, against the real Devvi CLI and the local mock model.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qagents import Agent, Limits, Permissions, prompts
from qagents import prompts as devvi_prompts
from qagents import prompts as texts
from qagents.external import (
    ExternalAgent,
    ExternalAgentConnectionError,
    ExternalCapabilities,
    PermissionStep,
    ScriptedExternalAgent,
    ToolStep,
    WaitStep,
)
from qagents.integrations._process import ManagedProcess
from qagents.integrations.acp import (
    TITLES_ARE_TOOL_NAMES,
    ACPAgent,
    default_tool_name,
    title_tool_name,
)
from qagents.integrations.acp import client as acp_client
from qagents.integrations.acp.mapping import outcome_for_stop_reason
from qagents.integrations.devvi import DevviAgent
from qagents.integrations.devvi.events import TURN_END_KINDS, turn_end_outcome
from qagents.integrations.devvi.patches import devvi_patch, render_patch
from qagents.messages import Usage
from qagents.models.testing import ScriptedModel, call

from .fakes import script

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src" / "qagents"
DOCS = ROOT / "docs"
CONCEPT = DOCS / "concepts" / "external-agents.md"
DEVVI = DOCS / "integrations" / "devvi.md"
ACP = DOCS / "integrations" / "acp.md"
MCP_SERVER = DOCS / "integrations" / "mcp-server.md"
PAGES = [CONCEPT, DEVVI, ACP, MCP_SERVER]
FIXTURE_HOME = ROOT / "tests" / "fixtures" / "devvi" / "home"
ACP_FIXTURES = ROOT / "tests" / "fixtures" / "devvi" / "acp"
EXAMPLE = ROOT / "examples" / "devvi_coder.py"
COMPAT = HERE / "devvi_node_compat.mjs"

FENCE = re.compile(r"^```(\w+)[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
HEADING = re.compile(r"^(#{1,6}) (.+)$")


# --- reading the pages ---------------------------------------------------------------------------


def text(page: Path) -> str:
    return page.read_text(encoding="utf-8")


def flat(page: Path) -> str:
    """The page with every run of whitespace made one space, so prose checks ignore wrapping."""
    return " ".join(text(page).split())


def blocks(page: Path, language: str = "python") -> list[str]:
    """The fenced blocks of ``language`` that start at the left margin."""
    return [match.group(2) for match in FENCE.finditer(text(page)) if match.group(1) == language]


def block_with(page: Path, needle: str, language: str = "python") -> str:
    found = [block for block in blocks(page, language) if needle in block]
    assert len(found) == 1, f"{page.name}: expected one {language} block with {needle!r}"
    return found[0]


def section(page: Path, title: str) -> str:
    """The lines under the heading ``title``, up to the next heading of its level or higher."""
    lines = text(page).splitlines()
    level = None
    out: list[str] = []
    fenced = False
    for line in lines:
        if line.startswith("```"):
            fenced = not fenced
        heading = None if fenced else HEADING.match(line)
        if level is None:
            if heading and heading.group(2) == title:
                level = len(heading.group(1))
            continue
        if heading and len(heading.group(1)) <= level:
            break
        out.append(line)
    assert level is not None, f"{page.name} has no heading {title!r}"
    return "\n".join(out)


def flat_section(page: Path, title: str) -> str:
    """`section` with every run of whitespace made one space."""
    return " ".join(section(page, title).split())


def first_table(lines: str) -> list[list[str]]:
    """The rows of the first Markdown table in ``lines``, header and rule left out."""
    rows: list[list[str]] = []
    started = False
    for line in lines.splitlines():
        if line.startswith("|"):
            started = True
            if set(line) <= set("|-: "):
                continue  # the rule under the header
            rows.append([cell.strip() for cell in line.strip().strip("|").split(" | ")])
        elif started:
            break
    return rows[1:]


def parse(source: str) -> ast.Module:
    flags = ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    return compile(source, "<docs>", "exec", flags=flags)


def strings(node: ast.AST) -> list[str] | None:
    """A list or tuple literal of strings, else ``None``."""
    if isinstance(node, ast.List | ast.Tuple) and all(
        isinstance(item, ast.Constant) and isinstance(item.value, str) for item in node.elts
    ):
        return [item.value for item in node.elts]  # type: ignore[attr-defined]
    return None


def permission_calls(page: Path) -> list[dict[str, Any]]:
    """The literal arguments of every ``Permissions(...)`` call in the page's Python blocks."""
    found = []
    for source in blocks(page):
        for node in ast.walk(parse(source)):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != "Permissions":
                continue
            rules: dict[str, Any] = {"allow": [], "ask": [], "deny": [], "mode": "default"}
            for keyword in node.keywords:
                if keyword.arg in ("allow", "ask", "deny"):
                    values = strings(keyword.value)
                    if values is not None:
                        rules[keyword.arg] = values
                elif keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                    rules["mode"] = keyword.value.value
            found.append(rules)
    return found


def decision_on(rules: Permissions, name: str):
    """What ``rules`` decide for a permission request named ``name``.

    The tool stands in for the permission bridge's synthetic tool: it requires approval.
    """
    tool = SimpleNamespace(name=name, read_only=False, requires_approval=True)
    return rules.decide(None, tool, None)  # type: ignore[arg-type]


def tool_results(result: Any) -> list[Any]:
    return [part for message in result.messages for part in message.tool_results]


# --- permission rules -----------------------------------------------------------------------------


async def test_every_allow_rule_in_the_pages_allows_what_it_names() -> None:
    """An ask or deny pattern that also matches would win over the allow rule (acp.md had
    ``allow=["coder.read"], ask=["coder.*"]``, which refuses ``coder.read``)."""
    checked = 0
    for page in PAGES:
        for rules in permission_calls(page):
            permissions = Permissions(
                allow=rules["allow"], ask=rules["ask"], deny=rules["deny"], mode=rules["mode"]
            )
            for pattern in rules["allow"]:
                name = pattern.replace("*", "x").replace("?", "x")
                decision = await decision_on(permissions, name)
                assert decision.behavior == "allow", f"{page.name}: {pattern} in {rules}"
                checked += 1
    assert checked >= 4


def acp_quick_start_rules() -> Permissions:
    source = blocks(ACP)[0]  # the quick start
    assert "coder = ACPAgent(" in source and "lead = Agent(" in source
    for node in ast.walk(parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Permissions":
            return eval(ast.unparse(node), {"Permissions": Permissions})  # noqa: S307
    raise AssertionError("the ACP quick start has no Permissions(...)")


def read_then_execute() -> ScriptedExternalAgent:
    return ScriptedExternalAgent(
        [[PermissionStep("read", {"path": "a.py"}), PermissionStep("execute", {"cmd": "ls"})]],
        name="coder",
    )


async def test_the_acp_quick_start_allows_reads_and_asks_about_commands() -> None:
    coder = read_then_execute()
    await coder.run("t", permissions=acp_quick_start_rules())
    assert [(a.outcome, a.decided_by) for a in coder.answers] == [
        ("allow", "rules"),
        ("deny", "no_approver"),  # "ask" without an on_ask
    ]


async def test_the_acp_page_explains_why_an_overlapping_ask_rule_refuses_reads() -> None:
    assert '`allow=["coder.read"], ask=["coder.*"]` asks about `coder.read` too' in flat(ACP)
    coder = read_then_execute()
    await coder.run("t", permissions=Permissions(allow=["coder.read"], ask=["coder.*"]))
    assert [(a.outcome, a.decided_by) for a in coder.answers] == [
        ("deny", "no_approver"),
        ("deny", "no_approver"),
    ]


async def test_the_acp_page_ask_all_rule_runs_the_coder_tool() -> None:
    assert '`Permissions(allow=["coder", "coder.read"], mode="ask_all")`' in flat(ACP)
    coder = read_then_execute()
    lead = Agent(
        ScriptedModel([call("coder", task="t"), "ok"]),
        tools=[coder.as_tool()],
        permissions=Permissions(allow=["coder", "coder.read"], mode="ask_all"),
    )
    result = await lead.run("go")
    assert result.status == "completed" and result.output == "ok"
    assert [(a.outcome, a.decided_by) for a in coder.answers] == [
        ("allow", "rules"),
        ("deny", "no_approver"),
    ]


def test_deny_rules_on_tools_devvi_never_asks_about_are_labelled() -> None:
    """Devvi never asks before web_fetch or web_search: a deny rule on them is only
    defense in depth, and the pages must say so where they write one."""
    seen = 0
    for page in (CONCEPT, DEVVI):
        for source in blocks(page):
            for line in source.splitlines():
                if "deny=" in line and re.search(r"\.web_(fetch|search)", line):
                    assert "defense in depth" in line, f"{page.name}: {line.strip()}"
                    seen += 1
    assert seen >= 2
    permissions = " ".join(section(DEVVI, "Permissions").split())
    assert "are never consulted" in permissions
    assert '{"id": "tool-web", "disabled": True}' in permissions
    assert "is never consulted" in flat(CONCEPT)


def test_the_no_web_patch_row_is_written_as_the_page_shows(tmp_path: Path) -> None:
    source = block_with(DEVVI, "NO_WEB = ")
    namespace: dict[str, Any] = {}
    exec(compile(source.split("\ncoder = ")[0], "devvi.md", "exec"), namespace)  # noqa: S102
    row = namespace["NO_WEB"]
    assert row == {"id": "tool-web", "disabled": True}
    DevviAgent(workspace=tmp_path, home=tmp_path / "home", patches=[row], allow_unmetered=True)
    rendered = render_patch(devvi_patch(tmp_path / "home", extra=[row]))
    assert '"id": "tool-web",\n    "disabled": true' in rendered


# --- the concept page's own examples ------------------------------------------------------------


async def test_the_concept_page_test_example_passes() -> None:
    source = block_with(CONCEPT, "async def test_")
    namespace: dict[str, Any] = {}
    exec(compile(source, "external-agents.md", "exec"), namespace)  # noqa: S102
    tests = [value for key, value in namespace.items() if key.startswith("test_")]
    assert len(tests) == 1
    await tests[0]()


async def test_the_concept_page_footer_is_what_the_lead_model_reads() -> None:
    """The footer under "Using an external agent" comes from the page's scripted scenario."""
    source = block_with(CONCEPT, "async def test_")
    namespace: dict[str, Any] = {}
    exec(compile(source, "external-agents.md", "exec"), namespace)  # noqa: S102
    coder = ScriptedExternalAgent(
        [
            [
                "Reading the parser.",
                ToolStep("read", {"path": "src/parser.py"}, result="def parse_line..."),
                ToolStep("pwsh", {"command": "pytest -q"}, result="1 passed"),
                ToolStep("pwsh", namespace["ESCALATION"], ask=True),
                Usage(requests=1, input_tokens=120, output_tokens=40),
                "Fixed parse_line; the test passes.",
            ]
        ],
        name="coder",
    )
    model = ScriptedModel([call("coder", task="t"), "Done."])
    lead = Agent(model, tools=[coder.as_tool()], permissions=Permissions(ask=["coder.pwsh"]))
    await lead.run("Fix the parser.")
    content = model.requests[1].messages[-1].tool_results[0].content
    shown = block_with(CONCEPT, "[coder: session coder-1;", "text").strip()
    assert content == shown


# --- budgets, timeouts and the unmetered refusal --------------------------------------------------


def waiting_coder() -> ScriptedExternalAgent:
    return ScriptedExternalAgent([["working", WaitStep()]], name="coder", cancel_grace_s=0.0)


async def test_a_short_tool_timeout_ends_the_call_without_the_session() -> None:
    """``tool_timeout_s <= 2 * (cancel_grace_s + 5)``: the loop's own timeout ends the call."""
    assert "more than `2 × (cancel_grace_s + 5)` (20 s by default)" in flat(CONCEPT)
    model = ScriptedModel([call("coder", task="t"), "ok"])
    lead = Agent(model, tools=[waiting_coder().as_tool()], limits=Limits(tool_timeout_s=1))
    result = await lead.run("go")
    content = tool_results(result)[0].content
    assert content == prompts.TOOL_TIMEOUT.format(name="coder", seconds=1)
    assert "session" not in content


async def test_a_longer_tool_timeout_cancels_the_turn_gracefully_with_the_session() -> None:
    """``tool_timeout_s > 2 * (cancel_grace_s + 5)``: cancelled at ``tool_timeout_s - margin``."""
    coder = waiting_coder()
    model = ScriptedModel([call("coder", task="t"), "ok"])
    lead = Agent(model, tools=[coder.as_tool()], limits=Limits(tool_timeout_s=10.2))
    started = time.monotonic()
    result = await lead.run("go")
    elapsed = time.monotonic() - started
    content = tool_results(result)[0].content
    assert content.startswith(texts.EXTERNAL_AGENT_STOPPED.format(name="coder", reason="timeout"))
    assert "session coder-1" in content
    assert coder.cancel_requests == ["timeout"]
    assert elapsed < 10.2  # at 10.2 - (0 + 5) = 5.2 s, before the loop's own timeout


async def test_an_unmetered_agent_is_refused_after_the_handshake_without_a_session() -> None:
    page = flat(CONCEPT)
    assert "refused before a session or a turn starts" in page
    assert "the handshake comes first" in page
    assert "no prompt is sent" in flat(ACP)
    coder = ScriptedExternalAgent([["never"]], name="coder", capabilities=ExternalCapabilities())
    lead = Agent(
        ScriptedModel([call("coder", task="t"), "ok"]),
        tools=[coder.as_tool()],
        limits=Limits(max_cost_usd=1.0),
    )
    result = await lead.run("go")
    assert tool_results(result)[0].content == texts.EXTERNAL_AGENT_UNMETERED.format(name="coder")
    assert coder.calls == [] and coder.prompts == []
    assert coder.connects <= 1  # the handshake may already have run


def test_every_external_agent_recipe_gives_as_tool_a_timeout() -> None:
    """Without one, the lead's 120 s ``tool_timeout_s`` cuts a coding turn at about 110 s."""
    checked = 0
    for page in PAGES:
        for source in blocks(page):
            if "DevviAgent(" not in source and "ACPAgent(" not in source:
                continue
            for node in ast.walk(parse(source)):
                if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "as_tool":
                    keywords = {keyword.arg for keyword in node.keywords}
                    assert "timeout" in keywords, f"{page.name}: {ast.unparse(node)}"
                    checked += 1
    assert checked >= 6


# --- signatures and option tables -----------------------------------------------------------------


def option_names(rows: list[list[str]]) -> set[str]:
    return {name for row in rows for name in re.findall(r"`([a-z_]+)`", row[0])}


def parameters(cls: type) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(cls.__init__).parameters)


def own_parameters(cls: type) -> set[str]:
    """The named parameters of ``cls.__init__``: no ``self``, no ``**options``."""
    return {
        name
        for name, p in parameters(cls).items()
        if name != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
    }


LITERAL_DEFAULT = re.compile(r'`(True|False|None|"[^"`]*")`|(\d+(?:\.\d+)?)')


def check_documented_defaults(rows: list[list[str]], cls: type) -> int:
    """Each table default that is a plain literal (a number, ``True``, ``False``, ``None``
    or a string) is the constructor's default. Returns how many were checked."""
    params = parameters(cls)
    checked = 0
    for row in rows:
        names = re.findall(r"`([a-z_]+)`", row[0])
        match = LITERAL_DEFAULT.fullmatch(row[1])
        if len(names) != 1 or match is None:
            continue
        default = params[names[0]].default
        if match.group(2) is not None:
            assert float(match.group(2)) == default, (cls.__name__, row)
        else:
            assert ast.literal_eval(match.group(1)) == default, (cls.__name__, row)
        checked += 1
    return checked


def test_devvi_agent_takes_every_option_by_keyword(tmp_path: Path) -> None:
    configuration = section(DEVVI, "Configuration")
    assert "`DevviAgent(*, workspace, ...)`" in configuration
    params = parameters(DevviAgent)
    kinds = {name: p.kind for name, p in params.items() if name != "self"}
    assert all(
        kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
        for kind in kinds.values()
    )
    with pytest.raises(TypeError):
        DevviAgent(str(tmp_path))  # type: ignore[misc]
    rows = first_table(configuration)
    documented = option_names(rows)
    assert documented and documented <= set(params)
    inherited = re.search(
        r"The `ExternalAgent` options work too: (.+?)\.\n", configuration, re.DOTALL
    )
    assert inherited is not None
    names = set(re.findall(r"`([a-z_]+)`", inherited.group(1)))
    assert names and names <= set(parameters(ExternalAgent))
    # the other direction: every option is on the page
    missing = own_parameters(DevviAgent) - documented
    assert not missing, f"not in the Configuration table: {sorted(missing)}"
    missing = own_parameters(ExternalAgent) - documented - names
    assert not missing, f"ExternalAgent options not named: {sorted(missing)}"
    assert check_documented_defaults(rows, DevviAgent) >= 10


def test_acp_agent_options_match_its_constructor() -> None:
    """Both directions: ``keep_sessions`` was added to the code but not to the table."""
    options = section(ACP, "Options")
    assert "`ACPAgent(command, *, ...)`" in options
    params = parameters(ACPAgent)
    assert params["command"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    rows = first_table(options)
    documented = option_names(rows)
    assert documented and documented <= set(params)
    inherited = re.search(r"Plus the `ExternalAgent` options: (.+?)\.\n", options, re.DOTALL)
    assert inherited is not None
    names = set(re.findall(r"`([a-z_]+)`", inherited.group(1)))
    assert names and names <= set(parameters(ExternalAgent))
    missing = own_parameters(ACPAgent) - documented
    assert not missing, f"not in the Options table: {sorted(missing)}"
    missing = own_parameters(ExternalAgent) - documented - names
    assert not missing, f"ExternalAgent options not named: {sorted(missing)}"
    assert check_documented_defaults(rows, ACPAgent) >= 8


# --- the Devvi page's recipes and claims --------------------------------------------------------


def test_the_session_log_recipe_finds_the_log(capsys: pytest.CaptureFixture[str]) -> None:
    source = block_with(DEVVI, "find_session_log(")
    assert "find_session_log(home," not in text(DEVVI)  # the home is not a sessions directory
    namespace: dict[str, Any] = {"home": FIXTURE_HOME, "session_id": "qa-fixture"}
    exec(compile(source, "devvi.md", "exec"), namespace)  # noqa: S102
    assert namespace["path"] is not None
    assert namespace["path"].name == "session.v4.jsonl"
    assert capsys.readouterr().out.splitlines()[0].startswith("1 completed")
    missing: dict[str, Any] = {"home": FIXTURE_HOME, "session_id": "qa-no-such-session"}
    exec(compile(source, "devvi.md", "exec"), missing)  # noqa: S102
    assert missing["path"] is None  # the page says: None when there is no log


def test_the_devvi_page_states_the_measured_sandbox_root() -> None:
    page = flat(DEVVI)
    assert "serves only the sessions of the directory" not in page
    assert "roots each session's file sandbox at that session's `cwd`" in page
    assert "not measured" not in flat(CONCEPT)


def test_the_devvi_troubleshooting_covers_the_failures_seen_in_tier_one() -> None:
    rows = first_table(section(DEVVI, "Troubleshooting"))
    not_offered = next(row for row in rows if "is not offered by the session" in row[0])
    assert "DEEPSEEK_API_KEY" in not_offered[1]
    assert any("access denied" in row[0] for row in rows)
    windows = section(DEVVI, "Windows notes")
    assert "ConstrainedLanguage" in windows and "TEMP" in windows
    assert "dsh-spill-" in section(DEVVI, "What qagents writes, and where")


def test_the_doctor_sample_is_real_output() -> None:
    sample = block_with(DEVVI, "[  ok] ACP: ", "text")
    acp, models = sample.strip().splitlines()
    assert acp.startswith(
        "[  ok] ACP: deepseek-harness-acp 0.0.1 (ACP v1): MCP over HTTP yes, image prompts no, "
        "session resume yes; ready in "
    )
    assert "ready in 0.3 s" not in acp  # the fake's figure; the real CLI needs about 2.5 s
    assert models.startswith("[  ok] ACP models: current deepseek-official/deepseek-v4-flash; ")


def marker_uses() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "tests").rglob("*.py")
        if path.name != "conftest.py"
    )


def test_every_marker_command_in_the_pages_selects_tests() -> None:
    """``pytest -m devvi_live`` was documented while no test carried the marker."""
    used = marker_uses()
    commands = {m for page in PAGES for m in re.findall(r"pytest -m (\w+)", text(page))}
    assert commands
    for marker in commands:
        assert re.search(rf"pytest\.mark\.{marker}\b", used), f"no test is marked {marker}"


def test_every_test_the_pages_cite_exists() -> None:
    defined = "\n".join(path.read_text(encoding="utf-8") for path in HERE.glob("test_*.py"))
    cited = {name for page in PAGES for name in re.findall(r"`(test_[a-z0-9_]+)`", text(page))}
    assert len(cited) >= 6
    for name in cited:
        assert re.search(rf"^(async )?def {name}\(", defined, re.MULTILINE), name


def test_the_mcp_server_page_loads_credentials_through_deps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = flat(MCP_SERVER)
    assert "from their own environment" not in page
    assert "KEY|PASSWORD|SECRET|TOKEN" in page
    source = block_with(MCP_SERVER, "def make_deps()")
    folder = tmp_path / ".my_app"
    folder.mkdir()
    (folder / "tickets.json").write_text('{"token": "fixture-value"}', encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    namespace: dict[str, Any] = {}
    exec(compile(source, "mcp-server.md", "exec"), namespace)  # noqa: S102
    assert namespace["make_deps"]() == {"tickets_token": "fixture-value"}


# --- the ACP page: tool names, protocol steps, updates ------------------------------------------


def test_the_acp_tool_name_rule_is_the_code_rule() -> None:
    """The page taught "the title when it looks like a tool name" for every agent after the
    code stopped using titles, so rules written from it (``coder.rm``) never matched."""
    names = flat_section(ACP, "Tool names for permission rules")
    assert "`title`, when it looks like a tool name" not in names
    assert "The `title` is not used" in names
    assert "`default_tool_name`" in names and "`title_tool_name`" in names
    # the page's examples, against the code
    assert default_tool_name({"title": "rm", "kind": "execute"}) == "execute"
    assert default_tool_name({"title": "Run ls -la", "kind": "execute"}) == "execute"
    assert default_tool_name({"title": "make"}) == "unknown"
    assert default_tool_name({"name": "read_file", "title": "rm", "kind": "read"}) == "read_file"
    assert title_tool_name({"title": "bash", "kind": "other"}) == "bash"
    assert title_tool_name({"title": "mcp__db__query", "kind": "other"}) == "mcp__db__query"
    assert title_tool_name({"title": "Run ls -la", "kind": "execute"}) == "execute"
    devvi = re.search(r"`agentInfo.name` `([a-z-]+)`", names)
    assert devvi is not None and devvi.group(1) in TITLES_ARE_TOOL_NAMES
    handshake = json.loads((ACP_FIXTURES / "handshake.jsonl").read_text("utf-8").splitlines()[2])
    assert handshake["result"]["agentInfo"]["name"] == devvi.group(1)  # Devvi's recorded name
    options = {row[0]: row[1] for row in first_table(section(ACP, "Options"))}
    assert options["`tool_name`"].startswith("`default_tool_name` (`title_tool_name` for Devvi)")


def generic_agent_runs_rm(path: Path) -> Path:
    """A fake ACP agent that is not Devvi, whose tool call has the title ``rm`` and the kind
    ``execute``; it accepts either answer to its permission request."""
    handshake = (ACP_FIXTURES / "handshake.jsonl").read_text(encoding="utf-8").splitlines()[1:]
    initialize = json.loads(handshake[1])
    initialize["result"]["agentInfo"] = {"name": "other-agent", "version": "1.0"}
    session = "{{sessionId}}"

    def update(body: dict[str, Any]) -> dict[str, Any]:
        params = {"sessionId": session, "update": body}
        return {"jsonrpc": "2.0", "method": "session/update", "params": params}

    options = [
        {"optionId": "allow-once", "name": "Allow once", "kind": "allow_once"},
        {"optionId": "reject-once", "name": "Reject", "kind": "reject_once"},
    ]
    answers = [{"outcome": {"outcome": "selected", "optionId": o["optionId"]}} for o in options]
    tool_call = {
        "sessionUpdate": "tool_call",
        "toolCallId": "c1",
        "title": "rm",
        "kind": "execute",
        "status": "pending",
        "rawInput": {"command": "rm -rf build"},
    }
    request = {"sessionId": session, "toolCall": {"toolCallId": "c1"}, "options": options}
    text = {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "Done."}}
    rows = [
        handshake[0],
        initialize,
        handshake[2],
        handshake[3],
        {
            "expect": {
                "id": 3,
                "method": "session/prompt",
                "params": {"sessionId": session, "prompt": "$list"},
            }
        },
        update(tool_call),
        {"jsonrpc": "2.0", "id": 1, "method": "session/request_permission", "params": request},
        {"answer": {"id": 1, "accept": answers}},
        update(text),
        {"jsonrpc": "2.0", "id": 3, "result": {"stopReason": "end_turn"}},
    ]
    path.write_text(
        "\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.devvi_fake
@pytest.mark.parametrize(
    ("deny", "decided"),
    [("coder.execute", "deny"), ("coder.rm", "allow")],
    ids=["a rule on the kind refuses", "a rule on the title never matches"],
)
async def test_a_rule_written_from_the_acp_page_matches_what_the_agent_sends(
    tmp_path: Path, deny: str, decided: str
) -> None:
    page = flat_section(ACP, "Tool names for permission rules")
    assert f'`deny=["{deny}"]`' in page
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = ACPAgent(
        script("fake_acp_agent", "--script", str(generic_agent_runs_rm(tmp_path / "rm.jsonl"))),
        name="coder",
        workspace=workspace,
        track_changes=False,
        close_grace_s=2.0,
        permissions=Permissions(allow=["coder.*"], deny=[deny]),
    )
    async with agent:
        result = await agent.run("Clean the build folder.")
    assert result.status == "completed", result.raw
    assert [(p.tool, p.outcome, p.decided_by) for p in result.permissions] == [
        ("coder.execute", decided, "rules")
    ]


def test_the_acp_page_names_every_request_the_client_sends() -> None:
    """The client module's own list of steps, against the page's: ``session/close`` (after
    a turn, for idle sessions beyond ``keep_sessions``) was missing from the page."""
    steps = flat_section(ACP, "What the client does")
    sent = set(re.findall(r"``(session/[a-z_]+|initialize|authenticate)\b", acp_client.__doc__))
    assert {"initialize", "session/new", "session/prompt", "session/close"} <= sent
    for method in sorted(sent):
        assert f"`{method}" in steps, method
    assert "beyond `keep_sessions` (default 1)" in steps
    assert str(parameters(ACPAgent)["keep_sessions"].default) == "1"
    assert (
        "A cancel that comes before `session/prompt` is sent ends the turn at once, without "
        "sending it and without `session/cancel`" in steps
    )
    assert "`$/cancel_request`, and the process is kept" in steps  # a setup request given up
    assert "continued directly while it is still open" in steps
    assert "continued directly while it is still open" in " ".join(acp_client.__doc__.split())


def test_the_acp_updates_table_says_new_input_is_an_event() -> None:
    rows = {row[0]: row[1] for row in first_table(section(ACP, "Updates and events"))}
    update = rows["`tool_call_update`"]
    assert "no event" not in update and "`tool_arguments`" in update
    assert "renames the call" in update


def test_the_acp_permission_requests_may_describe_their_call() -> None:
    answers = flat_section(ACP, "Permission answers")
    assert "carry only the tool call id" not in answers  # that is Devvi's shape, not ACP's
    assert "may describe it" in answers and "Devvi sends only the id" in answers


# --- the Devvi page: stop reasons, the dry run, re-attach, Windows, offline tests ----------------


def test_the_devvi_transports_report_the_shared_stop_reasons() -> None:
    """The table said an SDK turn ends with ``completed``; it ends with ``end_turn``."""
    rows = first_table(section(DEVVI, "Two transports"))
    assert not any(row[0].startswith("Stop reason") for row in rows)
    own = next(row for row in rows if row[0] == "Devvi's own stop word")
    assert all('`raw["agent_stop_reason"]`' in cell for cell in own[1:])
    page = flat_section(DEVVI, "Two transports")
    claim = re.search(
        r"Both transports report the same `stop_reason` for the same outcome: (.+?)\. So", page
    )
    assert claim is not None
    pairs = re.findall(r"`([a-z_]+)` \(status `([a-z_]+)`\)", claim.group(1))
    shared = [word for word, _ in pairs] + re.findall(r"and `([a-z_]+)`$", claim.group(1))
    assert shared == ["end_turn", "max_tokens", "cancelled"]
    statuses = [*(status for _, status in pairs), "cancelled"]
    for word, status in zip(shared, statuses, strict=True):  # ACP's own words are the same
        outcome = outcome_for_stop_reason({"stopReason": word})
        assert (outcome.status, outcome.stop_reason) == (status, word)
    sdk = re.search(r"Devvi's SDK words `([a-z-]+)`, `([a-z-]+)` and `([a-z-]+)` stay in", page)
    assert sdk is not None
    for kind, word, status in zip(sdk.groups(), shared, statuses, strict=True):
        assert TURN_END_KINDS[kind] == (status, word)
        outcome = turn_end_outcome({"kind": kind})
        assert outcome.stop_reason == word and outcome.raw["agent_stop_reason"] == kind


def test_the_dry_run_decisions_on_the_page_are_the_examples() -> None:
    """The page described a deny rule refusing a request; the example's reviewer refuses."""
    expected = None
    for node in ast.walk(ast.parse(EXAMPLE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "expected" for target in node.targets
        ):
            expected = node.value
    assert expected is not None, "examples/devvi_coder.py has no `expected = [...]`"
    decisions = eval(  # noqa: S307 - a list of f-strings from the example
        compile(ast.Expression(expected), "devvi_coder.py", "eval"), {"SHELL": "pwsh"}
    )
    assert len(decisions) == 2
    page = flat(DEVVI)
    for decision in decisions:
        assert f"`{decision}`" in page, decision
    assert "a request that a deny rule refuses" not in page
    assert "No deny rule decides anything in it" in page


def test_a_repeat_whose_session_is_gone_is_documented_as_not_sent_again() -> None:
    devvi, concept = flat(DEVVI), flat(CONCEPT)
    assert "a prompt that never arrived is sent as new" not in devvi
    for page in (devvi, concept):
        assert "`DEVVI_NOT_REATTACHED`" in page and "the task is not sent again" in page
    assert "`ACP_NOT_REATTACHED`" in concept
    assert "`EXTERNAL_AGENT_INTERRUPTED_NO_SESSION`" in concept
    model_reads = texts.EXTERNAL_AGENT_INTERRUPTED_NO_SESSION.format(
        name="coder", reason=devvi_prompts.DEVVI_NOT_REATTACHED.format(session="qa-...")
    )
    rows = first_table(section(DEVVI, "Troubleshooting"))
    row = next(row for row in rows if "which Devvi no longer has" in row[0])
    shown = re.search(r"`([^`]+)`", row[0])
    assert shown is not None and model_reads.startswith(shown.group(1))


def test_the_windows_notes_describe_the_kill_the_code_does() -> None:
    for page in PAGES:
        assert "taskkill" not in text(page), page.name
    notes = flat_section(DEVVI, "Windows notes")
    assert "Job Object" in notes and "process snapshot by parent id" in notes
    assert "Killing starts no other program" in notes
    assert "starts no other program" in " ".join((ManagedProcess.kill.__doc__ or "").split())


async def test_the_offline_lead_recipe_passes_and_mirrors_devvi_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = block_with(DEVVI, "def scripted_devvi()")
    namespace: dict[str, Any] = {}
    exec(compile(source, "devvi.md", "exec"), namespace)  # noqa: S102
    await namespace["test_the_lead_delegates_and_the_test_run_is_allowed"]()
    # the two settings mirror DevviAgent on ACP
    scripted = namespace["scripted_devvi"]()
    real = DevviAgent(workspace=tmp_path, home=tmp_path / "home")
    assert scripted.resumable_turns is True and real.resumable_turns is True
    acp = ACPAgent([sys.executable], name="coder", workspace=tmp_path)

    async def connected() -> None:
        return None

    monkeypatch.setattr(acp, "ensure_connected", connected)
    assert (await acp.capabilities()).token_usage is False
    assert (await scripted.capabilities()).token_usage is False
    assert (await ScriptedExternalAgent([]).capabilities()).token_usage is True  # the default
    # a coder built without allow_unmetered is refused, with the text the page quotes
    unmetered = ScriptedExternalAgent(
        [["never"]], name="coder", capabilities=ExternalCapabilities()
    )
    lead = namespace["make_lead"](ScriptedModel([call("coder", task="t"), "ok"]), unmetered)
    refusal = tool_results(await lead.run("go"))[0].content
    assert refusal == texts.EXTERNAL_AGENT_UNMETERED.format(name="coder")
    quoted = re.search(
        r"\(the model reads `(coder was not started: [^`]+)`\)",
        flat_section(DEVVI, "Test your lead offline"),
    )
    assert quoted is not None and refusal.startswith(quoted.group(1))
    assert "testing-offline-with-scriptedexternalagent" in section(DEVVI, "Test your lead offline")


# --- the concept page: events, adapters, event loops, git ---------------------------------------


def emitted_event_kinds() -> set[str]:
    """The ``external_event`` kinds the foundation and the two adapters emit by name."""
    paths = [
        *(SRC / "external").glob("*.py"),
        *(SRC / "integrations" / "acp").glob("*.py"),
        *(SRC / "integrations" / "devvi").glob("*.py"),
    ]
    kinds = {"image", "audio"}  # acp/mapping.py: ``self.turn.event(kind, ...)`` for these two
    for path in paths:
        source = path.read_text(encoding="utf-8")
        kinds |= set(re.findall(r'\.event\(\s*"([a-z_]+)"', source))
        kinds |= set(re.findall(r'_report\(\s*\w+,\s*"([a-z_]+)"', source))
        kinds |= set(re.findall(r'\bkind="([a-z_]+)"', source))
    return kinds


def test_every_external_event_kind_the_code_emits_is_documented() -> None:
    """``tool_arguments`` and ``approval_refused`` were emitted but not in the table."""
    events = section(CONCEPT, "Events")
    kinds_table = events.split('`ExternalAgentEvent` (`type="external_event"`) kinds:', 1)[1]
    documented = {
        name for row in first_table(kinds_table) for name in re.findall(r"`([a-z_]+)`", row[0])
    }
    emitted = emitted_event_kinds()
    assert {"tool_arguments", "approval_refused", "turn_finished"} <= emitted
    missing = emitted - documented
    assert not missing, f"emitted but not in the events table: {sorted(missing)}"


async def test_new_input_for_a_started_call_emits_tool_arguments() -> None:
    def new_input(turn: Any) -> None:
        turn.tool_started("c1", "execute", {"command": "ls"})
        turn.tool_started("c1", "execute", {"command": "ls -la"})
        turn.tool_finished("c1", "ok")

    coder = ScriptedExternalAgent([[new_input, "done"]], name="coder")
    async with coder.stream("t") as turn:
        events = [event async for event in turn]
    found = [e for e in events if e.type == "external_event" and e.kind == "tool_arguments"]
    assert len(found) == 1 and found[0].data["arguments"] == {"command": "ls -la"}
    started = [e for e in events if e.type == "tool_call_started"]
    assert started[0].call.arguments == {"command": "ls"}  # the started event keeps the first
    row = re.search(
        r"\| `tool_arguments` \| a started tool call got new input: `\{([a-z_, ]+)\}`",
        text(CONCEPT),
    )
    assert row is not None and set(row.group(1).split(", ")) == set(found[0].data)


async def test_a_connection_error_with_ambiguous_false_is_a_plain_error() -> None:
    assert "Pass `ambiguous=False` when nothing of the turn reached the agent" in flat(CONCEPT)
    plain = ScriptedExternalAgent([[ExternalAgentConnectionError("gone", ambiguous=False)]])
    broken = ScriptedExternalAgent([[ExternalAgentConnectionError("gone")]])
    first, second = await plain.run("t"), await broken.run("t")
    assert (first.status, first.raw.get("ambiguous")) == ("error", None)
    assert (second.status, second.raw.get("ambiguous")) == ("interrupted", True)


def test_an_agent_used_across_run_sync_calls_restarts_on_each_loop() -> None:
    assert "stops its process and starts a new one on the next loop" in flat(CONCEPT)
    coder = ScriptedExternalAgent([["first"], ["second"]], name="coder")
    model = ScriptedModel([call("coder", task="a"), "one", call("coder", task="b"), "two"])
    lead = Agent(model, tools=[coder.as_tool(timeout=60)])
    assert lead.run_sync("go").output == "one"
    assert lead.run_sync("again").output == "two"
    assert (coder.connects, coder.disconnects) == (2, 1)


def test_git_change_tracking_is_described_as_the_code_runs_it() -> None:
    page = flat(CONCEPT)
    code = " ".join((ExternalAgent.__doc__ or "").split())
    for program in ("fsmonitor", "filters", "external diff", "textconv", "pager"):
        assert program in page and program in code, program
    assert "Submodules' work trees are not looked into" in page
    assert "Submodules' work trees are not looked into" in code


# --- tier 1: what the real Devvi asks about (opt-in, -m devvi_mock) ------------------------------


@pytest.fixture
def devvi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """`DevviAgent` options for the real CLI, in a private home, workspace, profile and TEMP."""
    configured = os.environ.get("QAGENTS_DEVVI_BIN")
    if not configured or not Path(configured).is_file():
        pytest.skip("set QAGENTS_DEVVI_BIN to Devvi's built apps/cli/lib/bin.js")
    node = os.environ.get("QAGENTS_NODE") or shutil.which("node")
    if not node:
        pytest.skip("Node.js is not on PATH (or set QAGENTS_NODE)")
    out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=60)
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", out.stdout)
    if match is None:
        pytest.skip(f"cannot read the version of {node}")
    cli = str(Path(configured).resolve())
    if tuple(int(part) for part in match.groups()) < (24, 9, 0):
        launch: dict[str, Any] = {"command": [node, "--import", COMPAT.as_uri(), cli]}
    else:
        launch = {"devvi_bin": cli, "node": node}
    tempfile.gettempdir()  # cache this process's temp folder before TEMP changes below
    base = os.environ.get("QAGENTS_DEVVI_TMP")
    if base:
        Path(base).mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="dd-", dir=base))
    else:
        root = tmp_path
    user = root / "user"
    for sub in ("", "AppData/Roaming", "AppData/Local", "temp"):
        (user / sub).mkdir(parents=True, exist_ok=True)
    for name, value in {
        "USERPROFILE": user,
        "HOME": user,
        "APPDATA": user / "AppData" / "Roaming",
        "LOCALAPPDATA": user / "AppData" / "Local",
        "TEMP": user / "temp",
        "TMP": user / "temp",
    }.items():
        monkeypatch.setenv(name, str(value))
    for name in ("DEVVI_HOME", "DSH_HOME", "QAGENTS_DEVVI_HOME"):
        monkeypatch.delenv(name, raising=False)
    for folder in ("home", "ws"):
        (root / folder).mkdir()
    yield {"workspace": root / "ws", "home": root / "home", **launch}
    if base:
        shutil.rmtree(root, ignore_errors=True)


def mock_coder(devvi: dict[str, Any], model: Any, *extra_rows: dict[str, Any]) -> DevviAgent:
    from .mock_model_server import MOCK_ROUTE, mock_models_row

    return DevviAgent(
        name="coder",
        env=model.env(),
        env_passthrough=(),  # nothing of this process's environment, no real key
        model=MOCK_ROUTE,
        patches=[mock_models_row(), *extra_rows],
        track_changes=False,
        permissions=Permissions(deny=["coder.web_fetch", "coder.web_search"]),
        **devvi,
    )


@pytest.mark.devvi_mock
async def test_a_web_fetch_runs_without_a_permission_request(devvi: dict[str, Any]) -> None:
    """Under ``workspace-write`` an ordinary command and a web fetch run without a request,
    so the deny rule on ``coder.web_fetch`` is never consulted."""
    from .mock_model_server import IS_WINDOWS, SHELL, MockModel, Text, ToolUse, shell_arguments

    command = "Write-Output hello" if IS_WINDOWS else "echo hello"
    with MockModel() as model:
        model.on_prompt("FETCH", ToolUse(SHELL, shell_arguments(command, "Say hello"), id="sh-1"))
        model.after_tool("sh-1", ToolUse("web_fetch", {"url": "http://127.0.0.1:9/"}, id="f-1"))
        model.after_tool("f-1", Text("done"))
        async with mock_coder(devvi, model) as agent:
            result = await agent.run("FETCH a page.")
        assert model.errors == []
    assert result.status == "completed" and result.text == "done"
    assert result.permissions == []  # Devvi asked nothing: the deny rule never ran
    names = [tool_call.name for tool_call in result.tool_calls]
    assert "web_fetch" in names and len(names) == 2
    assert {"web_fetch", "web_search"} <= model.requests[0].tools


@pytest.mark.devvi_mock
async def test_a_disabled_tool_web_row_removes_devvis_web_tools(devvi: dict[str, Any]) -> None:
    from .mock_model_server import MockModel, Text

    source = block_with(DEVVI, "NO_WEB = ")
    namespace: dict[str, Any] = {}
    exec(compile(source.split("\ncoder = ")[0], "devvi.md", "exec"), namespace)  # noqa: S102
    with MockModel() as model:
        model.on_prompt("TOOLS", Text("no web"))
        async with mock_coder(devvi, model, namespace["NO_WEB"]) as agent:
            result = await agent.run("TOOLS: which tools do you have?")
        assert model.errors == []
    assert result.status == "completed" and result.text == "no web"
    offered = set().union(*(request.tools for request in model.requests))
    assert offered and not offered & {"web_fetch", "web_search"}
