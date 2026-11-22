"""The generated docs files of this checkout are current, and the docs say what the code does."""

from __future__ import annotations

import asyncio
import os
import runpy
import subprocess
import sys

from qagents import Agent, Permissions, RunContext, tool
from qagents.models.testing import ScriptedModel, call

from .conftest import ROOT


def test_llms_files_in_docs_are_current(capsys):
    # docs/llms.txt and docs/llms-full.txt are published with the site and bundled in the
    # wheel; a stale copy tells coding agents what the code no longer does.
    script = runpy.run_path(str(ROOT / "scripts" / "build_llms_txt.py"))
    assert script["main"](["--check"]) == 0, capsys.readouterr().out


def test_llms_build_does_not_depend_on_the_hash_seed(tmp_path):
    # The API reference shows frozenset values (LIMIT_REASONS), whose repr order follows the
    # string hash seed; two runs of an unchanged checkout must write the same files.
    script = ROOT / "scripts" / "build_llms_txt.py"
    texts = []
    for seed in ("1", "2"):
        out = tmp_path / seed
        env = {**os.environ, "PYTHONHASHSEED": seed}
        subprocess.run(
            [sys.executable, str(script), "--out", str(out)], env=env, check=True,
            capture_output=True,
        )  # fmt: skip
        texts.append((out / "llms-full.txt").read_text(encoding="utf-8"))
    assert texts[0] == texts[1]


async def test_parallel_tools_keeps_served_calls_apart_per_calling_call_only():
    active = peak = 0

    @tool
    async def write_ticket(n: int) -> str:
        """Writes: not read-only, so it must run alone among its caller's served calls."""
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.2)
        active -= 1
        return "ok"

    @tool(read_only=True)
    async def relay(ctx: RunContext, n: int) -> str:
        """Serves write_ticket to another program, as tool_server does."""
        result = await ctx.call_tool(write_ticket, {"n": n}, source="mcp")
        return result.content

    model = ScriptedModel([[call("relay", n=1), call("relay", n=2)], "done"])
    agent = Agent(model, tools=[relay, write_ticket], permissions=Permissions(allow=["*"]))
    result = await agent.run("go")
    assert result.output == "done"
    # Two calling calls: their served writers are not kept apart from each other.
    assert peak == 2
    page = (ROOT / "docs" / "integrations" / "mcp-server.md").read_text(encoding="utf-8")
    text = " ".join(page.split())
    assert "keeps it apart from the run's other calls" not in text
    assert "applies among the calls served through the same calling tool call" in text
    assert "Calls served through different calling calls" in text
