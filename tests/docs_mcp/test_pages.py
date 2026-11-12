"""read_page, list_pages and the examples gallery."""

from __future__ import annotations

from qagents.docs_mcp.service import DocsService
from qagents.docs_mcp.sources import find_source

from .conftest import ROOT


def test_read_a_section_by_anchor_or_heading(service):
    by_anchor = service.read_page("concepts/permissions.md", "pause-and-resume")
    by_title = service.read_page("concepts/permissions.md", "Pause and resume")
    assert by_anchor == by_title
    assert by_anchor.startswith("<!-- source: concepts/permissions.md#pause-and-resume |")
    assert "## Pause and resume" in by_anchor
    assert "agent.resume(" in by_anchor
    assert "## The decision order" not in by_anchor  # only the section


def test_loose_paths(service):
    for path in ["docs/concepts/permissions.md", "concepts/permissions", "/concepts/permissions/"]:
        assert "# Permissions and approvals" in service.read_page(path)
    assert "# Testing agents" in service.read_page("testing")
    assert "# qagents examples" in service.read_page("examples/README.md")


def test_unknown_page_and_section_suggest_close_names(service):
    missing = service.read_page("concepts/permisions.md")
    assert missing.startswith("There is no page") and "concepts/permissions.md" in missing
    assert "## Pause and resume" in service.read_page("concepts/permissions.md", "pause-and")
    section = service.read_page("concepts/permissions.md", "pause-resume")
    assert "has no section" in section and "pause-and-resume" in section
    assert "Sections of concepts/permissions.md:" in section


def test_reference_pages_get_the_live_api(service):
    text = service.read_page("reference/agent.md")
    assert "::: qagents.agent.Agent" not in text
    assert "## qagents.agent.Agent (class)" in text or "## qagents.Agent (class)" in text
    assert "async run(" in text  # the live signature of a method


def test_snippet_includes_are_filled_in(service):
    text = service.read_page("providers.md")
    assert "--8<--" not in text
    assert "class EchoModel" in text or "def stream" in text  # examples/custom_provider.py


def test_list_pages(service):
    text = service.list_pages()
    assert text.startswith(f"# qagents {service.version} documentation")
    assert "Source: " in text and "built in" in text
    assert "## Concepts" in text
    assert "- `concepts/permissions.md` **Permissions and approvals**: " in text
    assert text.index("`index.md`") < text.index("`concepts/loop.md`") < text.index("`testing.md`")


def test_the_skill_is_a_page(service):
    """An agent connected to the server can read the skill and hand it to its user."""
    file = (ROOT / "skills/qagents/SKILL.md").read_text(encoding="utf-8")
    text = service.read_page("skills/qagents/SKILL.md")
    assert text.startswith("<!-- source: skills/qagents/SKILL.md | ")
    assert text.split("\n", 1)[1] == file  # the whole file, front matter included
    assert service.page_markdown("skills/qagents/SKILL.md") == file
    assert "## Workflow" in service.read_page("skills/qagents/SKILL.md", "workflow")
    page = service.corpus.pages["skills/qagents/SKILL.md"]
    assert not page.summary.startswith(("---", "name:"))  # not the front matter
    assert "- `skills/qagents/SKILL.md` **The Claude Code skill**: " in service.list_pages()


def test_examples(service):
    gallery = service.list_examples()
    assert "## Patterns (examples/patterns)" in gallery
    assert "- `supervisor`: " in gallery and "Mirrors: " in gallery
    example = service.read_example("supervisor")
    assert example.startswith("# Example supervisor: ")
    assert "Source: `examples/patterns/supervisor.py`" in example
    assert "Offline test: `tests/examples/test_supervisor.py`" in example
    assert "```python" in example and "as_tool" in example
    assert service.read_example("patterns/supervisor.py") == example
    assert service.read_example("supervisr").startswith("There is no example")


def test_a_fake_checkout(fake_checkout):
    docs = DocsService(source=find_source(fake_checkout))
    pages = docs.list_pages()
    assert "## Guide" in pages and "`guide/widgets.md` **Widgets**" in pages
    assert "A fake README." in docs.read_page("README.md")
    text = docs.read_page("guide/widgets.md", "making-a-widget_1")
    assert "A second section with the same title." in text
    assert "`gadget`: A gadget from widgets." in docs.list_examples()
    assert "concepts" not in pages
