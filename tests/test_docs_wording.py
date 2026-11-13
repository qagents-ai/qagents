"""Stage-0 workflow guidance and release-status regression checks."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_side_effect_in_step_advice():
    paths = [ROOT / "src/qagents/workflow.py", ROOT / "AGENTS.md", ROOT / "README.md"]
    paths += list((ROOT / "docs").rglob("*.md")) + list((ROOT / "docs").glob("llms*.txt"))
    paths += list((ROOT / "examples").rglob("*.py"))
    unsafe = re.compile(
        r"\b(?:wf|workflow)\.step\(\s*[\"'](?:publish|send|refund|charge|delete|remind|close|code)[\"']"
    )
    found = [
        str(path.relative_to(ROOT))
        for path in paths
        if unsafe.search(path.read_text(encoding="utf-8"))
    ]
    assert not found, f"External changes still recommended inside plain workflow steps: {found}"


def test_readme_status_matches_package_version():
    from qagents import __version__

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    status = re.search(r"\*\*Status:\s*([^,\s]+)", readme)
    assert status is not None
    assert status.group(1) == __version__


def test_design_explains_runtime_lease_recovery():
    design = (ROOT / "docs/design.md").read_text(encoding="utf-8")
    liveness = design.split("**Liveness.**", 1)[1].split("\n- ", 1)[0]
    assert "in 0.1" not in liveness
    assert "AgentRuntime" in liveness and "recover()" in liveness
    assert "lease expired" in liveness
