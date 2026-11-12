"""Where the docs come from: --docs-dir, the bundle, the checkout, QAGENTS_DOCS_DIR."""

from __future__ import annotations

from pathlib import Path

import pytest

from qagents.docs_mcp import markdown, sources
from qagents.docs_mcp.service import DocsService

from .conftest import ROOT


@pytest.fixture
def no_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(sources, "BUNDLE_DIR", tmp_path / "no-bundle")
    monkeypatch.delenv(sources.ENV_VAR, raising=False)


def test_an_editable_install_reads_its_checkout(no_bundle):
    source = sources.find_source()
    assert source.kind == "checkout" and source.root == ROOT
    assert source.docs_dir == ROOT / "docs" and source.source_version == source.version
    assert "editable install" in source.describe()


def test_the_option_wins(no_bundle, fake_checkout):
    source = sources.find_source(fake_checkout)
    assert source.kind == "flag" and source.root == fake_checkout.resolve()
    assert sources.find_source(fake_checkout / "docs").root == fake_checkout.resolve()


def test_the_bundle_wins_over_the_checkout_and_the_variable(monkeypatch, fake_checkout):
    monkeypatch.setattr(sources, "BUNDLE_DIR", fake_checkout)
    monkeypatch.setenv(sources.ENV_VAR, str(ROOT))
    source = sources.find_source()
    assert source.kind == "bundled" and source.root == fake_checkout
    assert any("ignored" in note for note in source.notes)


def test_the_variable_is_used_when_there_is_no_checkout(no_bundle, monkeypatch, fake_checkout):
    monkeypatch.setattr(sources, "_is_qagents_checkout", lambda root: False)
    monkeypatch.setenv(sources.ENV_VAR, str(fake_checkout / "docs"))
    source = sources.find_source()
    assert source.kind == "env" and source.root == fake_checkout.resolve()
    assert "Widgets" in DocsService(source=source).list_pages()


def test_without_docs_the_api_still_answers(no_bundle, monkeypatch, tmp_path):
    monkeypatch.setattr(sources, "_is_qagents_checkout", lambda root: False)
    monkeypatch.setenv(sources.ENV_VAR, str(tmp_path / "missing"))
    source = sources.find_source()
    assert source.kind == "none" and any("ignored" in note for note in source.notes)
    docs = DocsService(source=source)
    assert "No documentation folder was found" in docs.list_pages()
    assert docs.get_api("Agent").startswith("# qagents.Agent (class)")
    assert "[api]" in docs.search("Agent.resume")


def test_a_version_mismatch_is_reported(no_bundle, fake_checkout):
    (fake_checkout / "pyproject.toml").write_text(
        '[project]\nname = "qagents"\nversion = "0.0.1"\n', encoding="utf-8"
    )
    source = sources.find_source(fake_checkout)
    assert source.source_version == "0.0.1"
    assert any("qagents 0.0.1" in note for note in source.notes)
    assert "qagents 0.0.1" in DocsService(source=source).list_pages()


def test_snippet_paths_stay_inside_the_checkout(fake_checkout):
    source = sources.find_source(fake_checkout)
    assert source.resolve_include("README.md") == fake_checkout.resolve() / "README.md"
    assert source.resolve_include("../outside.txt") is None
    assert source.resolve_include(str(Path(fake_checkout).resolve() / "README.md")) is None
    included = source.resolve_include("src/qagents/prompts.py")
    assert included is not None and included.name == "prompts.py"  # from the installed package


def test_a_bundle_with_the_skill_serves_it(monkeypatch, fake_checkout):
    """A wheel bundles skills/qagents/SKILL.md next to AGENTS.md (pyproject force-include)."""
    skill = fake_checkout / "skills" / "qagents" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: fake\ndescription: x\n---\n\n# fake\n\nA skill.\n", "utf-8")
    monkeypatch.setattr(sources, "BUNDLE_DIR", fake_checkout)
    source = sources.find_source()
    assert source.kind == "bundled" and source.skill_md == skill
    docs = DocsService(source=source)
    assert docs.read_page(sources.SKILL_PATH).endswith("# fake\n\nA skill.\n")
    assert "`skills/qagents/SKILL.md` **The Claude Code skill**: A skill." in docs.list_pages()


def test_the_repository_url_is_the_one_in_mkdocs():
    assert markdown.mkdocs_setting(ROOT / "mkdocs.yml", "repo_url") == sources.REPOSITORY_URL
    assert sources.SKILL_URL.startswith(sources.REPOSITORY_URL + "/")
    assert sources.SKILL_URL.endswith("/" + sources.SKILL_PATH.rsplit("/", 1)[0])
