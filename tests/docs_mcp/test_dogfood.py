"""The programs written while dogfooding the server, using only its command line.

Each was written by an agent that knew nothing about qagents: an agent with a tool that needs
approval (paused and resumed), a workflow with two durable agent steps, and a supervisor with
two sub-agents. They must pass ``check_code`` with no problems and run with scripted models.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

PROGRAMS = Path(__file__).resolve().parent / "programs"
NAMES = ["approval_agent.py", "two_step_workflow.py", "supervisor_team.py"]


@pytest.mark.parametrize("name", NAMES)
def test_the_program_checks_clean(service, name):
    report = service.check((PROGRAMS / name).read_text(encoding="utf-8"))
    assert report.problems == []
    assert report.counts["calls"] >= 8


@pytest.mark.parametrize("name", NAMES)
async def test_the_program_runs(name, capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    module = runpy.run_path(str(PROGRAMS / name), run_name="dogfood")
    await module["main"]()
    assert capsys.readouterr().out.strip()
