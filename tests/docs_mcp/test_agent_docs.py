"""AGENTS.md, the Claude Code skill and docs/coding-agents.md: present, valid, and true."""

from __future__ import annotations

import ast
import re

from .conftest import ROOT, python_blocks

ALLOWED_FENCES = {"python", "text", "bash", "powershell", "toml", "json"}


def fences(text: str) -> list[str]:
    return re.findall(r"^```(\w*)", text, re.MULTILINE)[::2]  # opening fences only


def test_agents_md_has_both_parts():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for heading in [
        "## Part 1: writing code with qagents",
        "### Look it up, never guess",
        "### Golden rules",
        "### Core patterns",
        "### Testing",
        "## Part 2: working on the qagents repository",
        "### Principles",
    ]:
        assert heading in text
    assert "python -m qagents.docs_mcp check" in text and "src/qagents/prompts.py" in text


PACKAGE = ROOT / "src" / "qagents"


def _layout() -> str:
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    match = re.search(r"### Layout\s+```text\n(.*?)^```", text, re.DOTALL | re.MULTILINE)
    assert match, "AGENTS.md Part 2 needs a ### Layout block"
    return match.group(1)


def test_the_layout_names_every_module_and_integration():
    layout = _layout()
    missing = []
    for path in sorted(PACKAGE.iterdir()):
        if path.name.startswith("_"):
            continue
        if path.suffix == ".py" and path.name not in layout:
            missing.append(path.name)
        if (path / "__init__.py").is_file() and f"src/qagents/{path.name}/" not in layout:
            missing.append(f"{path.name}/")
    for path in sorted((PACKAGE / "integrations").iterdir()):
        if path.name.startswith("_"):
            continue
        is_package = (path / "__init__.py").is_file()
        if (path.suffix == ".py" or is_package) and not re.search(rf"\b{path.stem}\b", layout):
            missing.append(f"integrations/{path.name}")
    assert not missing, f"AGENTS.md's layout does not mention: {missing}"


def test_every_prompts_module_is_named_and_every_named_one_exists():
    """The "Nothing hidden" rule says where model-facing strings live: it must stay true."""
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for path in sorted(PACKAGE.rglob("*prompts.py")):
        short = path.relative_to(PACKAGE).as_posix()
        assert short in text, f"AGENTS.md does not say that {short} holds model-facing strings"
    named = re.findall(r"`((?:src/qagents/)?(?:\w+/)*\w*prompts\.py)`", text)
    assert "src/qagents/prompts.py" in named
    for name in named:
        path = ROOT / name if name.startswith("src/") else PACKAGE / name
        assert path.is_file(), f"AGENTS.md names {name}, which does not exist"


def test_the_skill_has_frontmatter_and_stays_short():
    text = (ROOT / "skills/qagents/SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"^---\nname: (.+)\ndescription: (.+)\n---\n", text)
    assert match, "SKILL.md must start with name and description frontmatter"
    assert match.group(1) == "qagents"
    assert "imports qagents" in match.group(2) and len(match.group(2)) < 1024
    assert len(text.splitlines()) <= 200
    assert "python -m qagents.docs_mcp" in text and "get_api" in text


def test_coding_agents_page():
    text = (ROOT / "docs/coding-agents.md").read_text(encoding="utf-8")
    for needle in [
        "claude mcp add qagents-docs -- python -m qagents.docs_mcp",
        "codex mcp add qagents-docs -- python -m qagents.docs_mcp",
        "[mcp_servers.qagents-docs]",
        ".cursor/mcp.json",
        ".vscode/mcp.json",
        "## Tool access and agent control",
        "(concepts/external-agents.md)",
        "llms.txt",
        "AGENTS.md",
    ]:
        assert needle in text, needle


def test_coding_agents_page_says_how_a_pip_install_gets_the_skill():
    """The skill is not in your project: the page names a command that writes it, and the
    repository folder for an install without a copy. Both must work."""
    from qagents.docs_mcp.sources import SKILL_URL

    text = (ROOT / "docs/coding-agents.md").read_text(encoding="utf-8")
    section = text.split("**The skill.**", 1)[1].split("\n## ", 1)[0]
    assert "python -m qagents.docs_mcp skill --to .claude/skills/qagents" in section
    assert 'read_page("skills/qagents/SKILL.md")' in section
    assert f"]({SKILL_URL})" in section
    assert "Copy the `skills/qagents` folder" not in text  # there is no folder in a pip install


AGENT_FACING = ["docs/coding-agents.md", "skills/qagents/SKILL.md", "AGENTS.md"]


def test_no_page_claims_mcp_cannot_hand_a_coding_agent_a_task():
    """Codex serves itself over MCP (codex mcp-server), so MCPToolkit can send it a task. What
    MCP cannot do is follow the turn: the pages must say that, not the absolute claim."""
    for page in AGENT_FACING:
        text = " ".join((ROOT / page).read_text(encoding="utf-8").split())
        assert "cannot start a coding session" not in text, page
        assert "not MCP:" not in text, page
    text = (ROOT / "docs/coding-agents.md").read_text(encoding="utf-8")
    section = text.split("## Tool access and agent control", 1)[1].split("\n## ", 1)[0]
    section = " ".join(section.split())
    for needle in [
        "(Codex: `codex mcp-server`) can be called through `MCPToolkit`",
        "each turn is then one opaque call",
        "decide its permission requests one by one",
        "drive it over ACP or the Devvi SDK",
    ]:
        assert needle in section, needle


def test_coding_agents_page_says_how_a_project_gets_the_agents_md_rules(service):
    """qagents' AGENTS.md is read only inside a qagents checkout; the page must say so."""
    text = (ROOT / "docs/coding-agents.md").read_text(encoding="utf-8")
    section = text.split("## AGENTS.md", 1)[1].split("\n## ", 1)[0]
    assert "Copy Part 1" in section and "your project's `AGENTS.md`" in section
    assert 'read_page("AGENTS.md")' in section
    assert "python -m qagents.docs_mcp read AGENTS.md" in section
    assert "read them automatically" not in text and "read it automatically" not in text
    # What the page tells agents to run must work.
    assert service.read_page("AGENTS.md").startswith("<!-- source: AGENTS.md ")
    part_1 = service.read_page("AGENTS.md", "part-1-writing-code-with-qagents")
    assert "## Part 1: writing code with qagents" in part_1 and "## Part 2" not in part_1


def test_code_fences_use_the_allowed_languages():
    for page in ["AGENTS.md", "skills/qagents/SKILL.md", "docs/coding-agents.md"]:
        text = (ROOT / page).read_text(encoding="utf-8")
        assert set(fences(text)) <= ALLOWED_FENCES, page
        for line, code in python_blocks(ROOT / page):
            compile(code, f"{page}:{line}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


async def test_the_test_examples_in_the_docs_pass():
    """The ``async def test_...`` blocks of AGENTS.md and the skill run and pass."""
    ran = 0
    for page in ["AGENTS.md", "skills/qagents/SKILL.md"]:
        for line, code in python_blocks(ROOT / page):
            if "async def test_" not in code:
                continue
            namespace: dict[str, object] = {}
            exec(compile(code, f"{page}:{line}", "exec"), namespace)  # noqa: S102 - our docs
            for name, value in namespace.items():
                if name.startswith("test_") and callable(value):
                    await value()
                    ran += 1
    assert ran >= 2
