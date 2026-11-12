"""``examples/devvi_coder.py --dry-run``: the lead agent and the Devvi coder, offline.

The dry run uses a `FunctionModel` for the lead and the fake ACP peer for Devvi
(``fakes/fake_devvi_acp.py``), so it needs no Devvi, no key and no network. It runs once
in this process (to check the outcome) and once as the command a user types.
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from qagents import Approval
from qagents.messages import ToolCallPart

pytestmark = pytest.mark.devvi_fake

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "devvi_coder.py"


@pytest.fixture(scope="module")
def example() -> dict[str, Any]:
    """The example's module globals (``run``, ``main``, ``review``, ...)."""
    return runpy.run_path(str(EXAMPLE), run_name="devvi_coder")


async def test_the_dry_run_delegates_reviews_and_applies_the_rules(
    example: dict[str, Any], tmp_path: Path
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outcome = await example["run"](
        workspace, dry_run=True, lead=None, devvi_model=None, max_cost=0.5
    )
    assert outcome.ok, outcome.summary
    shell = example["SHELL"]
    assert outcome.decisions == [
        f"coder.{shell}: allow by on_ask",
        f"coder.{shell}: deny by on_ask",
    ]
    assert outcome.footer.startswith("[coder: session fake-session-1; completed; 4 tool calls")
    assert f"1 permission refused ({shell}" in outcome.footer
    assert "pip install --upgrade pytest" in outcome.footer
    fixed = (workspace / "src" / "parser.py").read_text(encoding="utf-8")
    assert fixed == example["FIXED"]
    again = await example["run"](workspace, dry_run=True, lead=None, devvi_model=None, max_cost=0.5)
    assert again.ok  # a second dry run in the same workspace works too


def test_the_reviewer_allows_test_runs_only(example: dict[str, Any]) -> None:
    from qagents.external import ExternalTool

    review = example["review"]
    bash = ExternalTool("coder.bash", agent="coder", external_name="bash")
    edit = ExternalTool("coder.edit", agent="coder", external_name="edit")

    def asks(tool: Any, **arguments: Any) -> Approval:
        return review(None, tool, ToolCallPart(id="c", name=tool.name, arguments=arguments), None)

    assert asks(bash, command="python -m pytest tests/test_parser.py -q").approved
    assert not asks(bash, command="pytest -q; rm -rf build").approved
    assert not asks(bash, command="rm -rf build").approved
    assert not asks(edit, path="../outside.py").approved
    permissions = example["coder_permissions"]("coder")
    assert {"coder.bash", "coder.pwsh"} <= set(permissions.ask)


def test_the_coder_removes_the_web_tools_devvi_never_asks_about(
    example: dict[str, Any], tmp_path: Path
) -> None:
    """Devvi runs web_fetch without a permission request, so a deny rule alone would not
    stop it: the coder is started with a patch row that removes Devvi's web tools."""
    coder = example["make_coder"](tmp_path)
    assert {"id": "tool-web", "disabled": True} in coder._launch_options["patch_rows"]


def test_the_dry_run_asks_only_about_escalations(example: dict[str, Any], tmp_path: Path) -> None:
    """As the real Devvi in workspace-write: plain steps run unasked; each request is an
    escalation of the shell tool with sandbox_permissions and a justification."""
    rows = [
        json.loads(line)
        for line in example["fake_devvi_turn"](tmp_path).read_text().splitlines()
        if line.strip()
    ]
    calls = {
        row["params"]["update"]["toolCallId"]: row["params"]["update"]
        for row in rows
        if row.get("method") == "session/update"
        and row["params"]["update"]["sessionUpdate"] == "tool_call"
    }
    asked = [
        row["params"]["toolCall"]["toolCallId"]
        for row in rows
        if row.get("method") == "session/request_permission"
    ]
    assert asked == ["t3", "t4"] and set(calls) == {"t1", "t2", "t3", "t4"}
    for call_id in asked:
        update = calls[call_id]
        assert update["title"] == example["SHELL"]
        assert update["rawInput"]["sandbox_permissions"] == "danger-full-access"
        assert update["rawInput"]["justification"]
    assert not any("web_" in update["title"] for update in calls.values())


def test_the_dry_run_never_replaces_a_file_of_the_workspace(
    example: dict[str, Any], tmp_path: Path
) -> None:
    mine = tmp_path / "ws" / "src" / "parser.py"
    mine.parent.mkdir(parents=True)
    mine.write_text("# my own parser\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not the dry run's file"):
        example["seed"](tmp_path / "ws")
    assert mine.read_text(encoding="utf-8") == "# my own parser\n"


def test_the_command_line(tmp_path: Path) -> None:
    environment = {key: value for key, value in os.environ.items() if key != "QAGENTS_LEAD_MODEL"}
    done = subprocess.run(
        [sys.executable, str(EXAMPLE), "--workspace", str(tmp_path / "sandbox"), "--dry-run"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    lines = done.stdout.splitlines()
    assert lines[0] == "seeded the workspace with src/parser.py, tests/test_parser.py"
    shell = "pwsh" if sys.platform == "win32" else "bash"
    assert f"  [coder] permission coder.{shell}: allow by on_ask" in lines
    assert f"  [coder] permission coder.{shell}: deny by on_ask" in lines
    assert lines[-1].startswith("ok: completed; coder: [coder: session fake-session-1; completed")
    refused = subprocess.run(
        [sys.executable, str(EXAMPLE), "--workspace", str(tmp_path / "live")],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert refused.returncode == 2 and "give --lead" in refused.stderr
