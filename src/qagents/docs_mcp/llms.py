"""``llms.txt`` and ``llms-full.txt`` (https://llmstxt.org) from the docs corpus.

``llms.txt`` is a Markdown index: an H1 with the project name, a blockquote summary, a short
paragraph, then one H2 per nav section with a link and a one-line description per page.
Pages that are mostly not English go under ``## Optional``, which readers may skip.
``llms-full.txt`` is every page in nav order, with snippet includes and API directives
filled in, so one fetch gives a model the whole documentation.
"""

from __future__ import annotations

from collections.abc import Callable

from .corpus import EXTRA_PAGES, Corpus, Page
from .markdown import mkdocs_setting

DEFAULT_BRANCH = "main"
"""Branch used for links to example files in the repository."""


def page_url(page: Page, site_url: str | None) -> str:
    """The published URL of a page (MkDocs directory URLs), relative when no site URL."""
    path = page.path.removesuffix(".md")
    if path == "index":
        path = ""
    elif path.endswith("/index"):
        path = path[: -len("index")]
    else:
        path += "/"
    base = (site_url or "").rstrip("/")
    if base:
        return f"{base}/{path}" if path else f"{base}/"
    return path or "./"


def _mostly_not_english(page: Page) -> bool:
    text = "\n".join(page.lines)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    non_ascii = sum(1 for c in letters if not c.isascii())
    return non_ascii / len(letters) > 0.3


def _summary(corpus: Corpus) -> str:
    description = mkdocs_setting(corpus.source.mkdocs_yml, "site_description")
    if description:
        return description
    readme = corpus.pages.get("README.md")
    return readme.summary if readme else "A Python agent framework."


def build_llms_txt(corpus: Corpus, *, site_url: str | None = None) -> str:
    """The ``llms.txt`` index of the documentation."""
    site_url = site_url or mkdocs_setting(corpus.source.mkdocs_yml, "site_url")
    name = mkdocs_setting(corpus.source.mkdocs_yml, "site_name") or "qagents"
    version = corpus.source.version
    full = f"{site_url.rstrip('/')}/llms-full.txt" if site_url else "llms-full.txt"
    lines = [
        f"# {name}",
        "",
        f"> {_summary(corpus)}",
        "",
        f"This index describes qagents {version}. Every page below, in this order, is also in "
        f"one file: [llms-full.txt]({full}). Coding agents can query the documentation of the "
        "installed version with the docs MCP server, `python -m qagents.docs_mcp` (stdio), or "
        'its command line, `python -m qagents.docs_mcp search "<query>"`.',
    ]
    sections: dict[str, list[str]] = {}
    optional: list[str] = []
    in_nav = {entry.path for entry in corpus.nav}
    for page in corpus.ordered_pages():
        if page.path not in in_nav and page.path not in EXTRA_PAGES:
            group = "Other pages"
        elif page.path in in_nav:
            group = " / ".join(page.group) or "Start"
        else:
            continue
        description = f": {page.summary}" if page.summary else ""
        line = f"- [{page.display_title}]({page_url(page, site_url)}){description}"
        if _mostly_not_english(page):
            optional.append(line)
        else:
            sections.setdefault(group, []).append(line)
    for group, items in sections.items():
        lines += ["", f"## {group}", "", *items]
    examples = _example_lines(corpus)
    if examples:
        lines += ["", "## Examples", "", *examples]
    if optional:
        lines += ["", "## Optional", "", *optional]
    return "\n".join(lines) + "\n"


def _example_lines(corpus: Corpus) -> list[str]:
    repo = mkdocs_setting(corpus.source.mkdocs_yml, "repo_url")
    lines = []
    for example in corpus.examples.values():
        if not example.name.startswith("patterns/"):
            continue
        link = f"{repo.rstrip('/')}/blob/{DEFAULT_BRANCH}/{example.rel}" if repo else example.rel
        lines.append(f"- [{example.short}]({link}): {example.description}.")
    return lines


def build_llms_full(
    corpus: Corpus,
    *,
    site_url: str | None = None,
    render_api: Callable[[str], str] | None = None,
) -> str:
    """Every page in nav order (then the pages the nav leaves out), as one Markdown file."""
    site_url = site_url or mkdocs_setting(corpus.source.mkdocs_yml, "site_url")
    name = mkdocs_setting(corpus.source.mkdocs_yml, "site_name") or "qagents"
    lines = [
        f"# {name} {corpus.source.version}: the full documentation",
        "",
        f"> {_summary(corpus)}",
        "",
        "Every documentation page, in the order of the site navigation. Each page starts with "
        "a line that gives its path and its URL.",
    ]
    for page in corpus.ordered_pages():
        if page.path in EXTRA_PAGES:
            continue  # README, AGENTS.md, the skill: not pages of the site
        body = corpus.expand(page, 0, len(page.lines), render_api)
        lines += [
            "",
            "---",
            "",
            f"<!-- page: {page.path} | {page_url(page, site_url)} -->",
            "",
            body,
        ]
    return "\n".join(lines) + "\n"
