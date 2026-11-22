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


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_serving_checklist_signs_approvals_for_pauses_and_effect_recovery():
    """The production checklist once said approvals are not signed (0.1). Without the
    secret, approval-gated effects tools lose D3/D4 recovery (AF-5 review)."""
    serving = (ROOT / "docs/serving.md").read_text(encoding="utf-8")
    checklist = _flat(serving.split("## Before production", 1)[1].split("\n## ", 1)[0])
    assert "not signed" not in checklist
    assert "Set `Permissions(approval_secret=...)`" in checklist
    assert "the same on every worker" in checklist
    assert "guides/recover-effect.md#effects-tools-that-ask-for-approval" in checklist


def test_effects_approval_docs_name_can_use_tool_and_the_effects_the_secret_concerns():
    """A can_use_tool "ask" holds the same way, and only effects D3 or D4 can re-send
    depend on the secret; the others keep their own hold reason (AF-5 review)."""
    guide = (ROOT / "docs/guides/recover-effect.md").read_text(encoding="utf-8")
    section = _flat(
        guide.split("## Effects tools that ask for approval", 1)[1].split("\n## ", 1)[0]
    )
    assert 'your `can_use_tool` returns `"ask"`' in section
    assert "`Dedupe` or `Unique` contract, or with an authoritative lookup" in section
    assert "held for its own reason, such as `RESPONSE_LOST`" in section
    assert "`NOT_YET_QUIESCENT`, with a `retry_after`" in section
    assert "Both reasons appear only where a re-send (D3 or D4) would otherwise go out" in section
    concept = _flat((ROOT / "docs/concepts/permissions.md").read_text(encoding="utf-8"))
    assert "where D3 or D4 would re-send it" in concept and "`can_use_tool`" in concept
    assert "also through `agent.recheck`" in concept
    capabilities = _flat((ROOT / "docs/capabilities.md").read_text(encoding="utf-8"))
    assert (
        "(the agent warns when it is built, or at run start for a toolkit's tools)" in capabilities
    )
