"""scripts/build_llms_txt.py writes llms.txt and llms-full.txt in the llmstxt.org format."""

from __future__ import annotations

import re
import runpy

import pytest

from .conftest import ROOT


@pytest.fixture(scope="module")
def script():
    return runpy.run_path(str(ROOT / "scripts" / "build_llms_txt.py"))


def test_writes_both_files(script, tmp_path, capsys, service):
    assert script["main"](["--out", str(tmp_path)]) == 0
    assert "wrote" in capsys.readouterr().out
    index = (tmp_path / "llms.txt").read_text(encoding="utf-8")
    full = (tmp_path / "llms-full.txt").read_text(encoding="utf-8")

    lines = index.splitlines()
    assert lines[0] == "# QAgents" and lines[2].startswith("> ")
    assert "## Concepts" in index
    links = re.findall(r"^- \[([^\]]+)\]\(([^)]+)\): (.+)$", index, re.MULTILINE)
    assert len(links) >= 30
    assert (
        "Permissions and approvals",
        "https://qagents-ai.github.io/qagents/concepts/permissions/",
    ) in [(t, u) for t, u, _ in links]
    assert "research/" not in index and "<!-- page: research.md " not in full

    order = [entry.path for entry in service.corpus.nav]
    pages = re.findall(r"^<!-- page: (\S+) \|", full, re.MULTILINE)
    assert pages[: len(order)] == order  # every nav page, in nav order
    assert "::: qagents" not in full and "--8<--" not in full
    assert "- `async run(" in full  # the API reference, filled in
    # Files from outside docs/ are not pages of the site: neither file lists them.
    for path in ("README.md", "AGENTS.md", "skills/qagents/SKILL.md"):
        assert path in service.corpus.pages
        assert f"<!-- page: {path} " not in full
    assert "The Claude Code skill" not in index and "](AGENTS/)" not in index


def test_check_mode(script, tmp_path):
    assert script["main"](["--out", str(tmp_path), "--check"]) == 1  # nothing written yet
    assert script["main"](["--out", str(tmp_path)]) == 0
    assert script["main"](["--out", str(tmp_path), "--check"]) == 0
    (tmp_path / "llms.txt").write_text("stale", encoding="utf-8")
    assert script["main"](["--out", str(tmp_path), "--check"]) == 1


def test_absolute_links_with_a_site_url(script, tmp_path):
    script["main"](["--out", str(tmp_path), "--site-url", "https://docs.example.org/qagents/"])
    index = (tmp_path / "llms.txt").read_text(encoding="utf-8")
    assert "(https://docs.example.org/qagents/concepts/tools/)" in index
    assert "[llms-full.txt](https://docs.example.org/qagents/llms-full.txt)" in index
