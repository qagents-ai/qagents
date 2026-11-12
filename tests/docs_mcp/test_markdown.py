"""Headings, anchors, the nav and page summaries."""

from __future__ import annotations

import pytest

from qagents.docs_mcp import markdown


@pytest.mark.parametrize(
    ("heading", "anchor"),
    [
        ("Pause and resume", "pause-and-resume"),
        ("Approvals, inline", "approvals-inline"),
        ("`ctx.operation_id`", "ctxoperation_id"),
        ("Streaming, cancel and voice", "streaming-cancel-and-voice"),
        ("What `run()` returns", "what-run-returns"),
        ("[Links](other.md) are text", "links-are-text"),
    ],
)
def test_anchors_match_mkdocs(heading, anchor):
    assert markdown.slugify(markdown.strip_inline(heading)) == anchor


def test_sections_skip_code_blocks_and_number_duplicate_anchors(fake_checkout):
    lines = (fake_checkout / "docs/guide/widgets.md").read_text(encoding="utf-8").splitlines()
    sections = markdown.parse_sections(lines)
    assert [s.anchor for s in sections] == [
        "widgets", "making-a-widget", "sizes-small-and-large", "making-a-widget_1",
    ]  # fmt: skip
    first = sections[1]
    assert first.trail == ("Widgets", "Making a widget")
    assert lines[first.start] == "## Making a widget"
    # The H2 section includes its H3; its own body stops at the H3.
    assert "Sizes change the widget." in "\n".join(lines[first.start : first.end])
    assert "Sizes change the widget." not in "\n".join(lines[first.start : first.body_end])
    assert sections[2].trail == ("Widgets", "Making a widget", "Sizes small and large")


def test_explicit_ids_win():
    sections = markdown.parse_sections(["# Title", "", "## Custom one {#mine}", "text"])
    assert sections[1].anchor == "mine" and sections[1].title == "Custom one"


def test_nav_order_groups_and_external_links_left_out(fake_checkout):
    nav = markdown.parse_nav(fake_checkout / "mkdocs.yml")
    assert [(e.title, e.path, e.group) for e in nav] == [
        ("Home", "index.md", ()),
        ("Widgets", "guide/widgets.md", ("Guide",)),
    ]


def test_nav_of_this_project_lists_every_page_in_order(service):
    nav = service.corpus.nav
    assert nav[0].path == "index.md"
    assert any(e.path == "concepts/permissions.md" and e.group == ("Concepts",) for e in nav)


def test_summaries():
    assert markdown.summary(["# T", "", "One. Two."]) == "One. Two."  # short: two sentences
    long_first = "A sentence that is long enough to stand on its own as a summary of the page."
    assert markdown.summary(["# T", "", f"{long_first} Second."]) == long_first
    assert markdown.summary(["# T", "", "## Install", "", "text", "## Use"]) == (
        "Sections: Install, Use."
    )
    assert markdown.summary(["# Agent", "", "::: qagents.agent.Agent", "::: qagents.tools"]) == (
        "API reference: Agent, qagents.tools."
    )


def test_front_matter_is_blanked_but_keeps_the_line_numbers():
    lines = ["---", "name: qagents", "description: A skill.", "---", "", "# qagents", "", "Hi."]
    visible = markdown.without_front_matter(lines)
    assert visible == ["", "", "", "", "", "# qagents", "", "Hi."]
    assert markdown.summary(visible) == "Hi."
    assert [s.title for s in markdown.parse_sections(visible)] == ["qagents"]
    assert markdown.without_front_matter(["# T", "---"]) == ["# T", "---"]  # a rule, not YAML
    assert markdown.without_front_matter(["---", "never closed"]) == ["---", "never closed"]


def test_mkdocs_setting(fake_checkout):
    yml = fake_checkout / "mkdocs.yml"
    assert markdown.mkdocs_setting(yml, "site_description") == "A fake library for tests."
    assert markdown.mkdocs_setting(yml, "site_url") is None
