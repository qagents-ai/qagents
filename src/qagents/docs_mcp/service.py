"""The answers of the docs server, as compact Markdown that cites its sources.

`DocsService` is what the MCP tools and the command line call. Every method returns a
string ready to show to a model or a person; nothing here depends on the MCP SDK.
"""

from __future__ import annotations

import difflib
import re
import threading
from collections.abc import Iterable

from . import prompts
from .api import ApiLookup, Target
from .check import Report, check_code
from .corpus import Corpus, Example, Page
from .markdown import Section, mkdocs_setting, slugify
from .search import snippet
from .sources import DocsSource

KINDS = ("docs", "example", "api")
PAGE_LIMIT = 60_000
"""A whole page longer than this many characters is cut, with a list of its sections."""
EXAMPLE_LIMIT = 80_000


class DocsService:
    """Search, read and check against the installed qagents documentation."""

    def __init__(self, corpus: Corpus | None = None, *, source: DocsSource | None = None) -> None:
        self.corpus = corpus or Corpus.load(source)
        self.lookup = ApiLookup(self.corpus.api)
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        return self.corpus.source.version

    # -- search ----------------------------------------------------------------------

    def search(self, query: str, limit: int = 8, kind: str | None = None) -> str:
        """Ranked results for a query, each with a snippet, its source and how to read it."""
        query = query.strip()
        limit = max(1, min(int(limit), 25))
        kinds: Iterable[str] | None = None
        if kind:
            wanted = kind.strip().lower().rstrip("s")
            wanted = {"doc": "docs", "page": "docs", "code": "api"}.get(wanted, wanted)
            if wanted not in KINDS:
                return f"Unknown kind {kind!r}: use one of {', '.join(KINDS)}, or leave it empty.\n"
            kinds = [wanted]
        if not query:
            return prompts.NO_RESULTS.format(query=query) + "\n"
        hits, total = self.corpus.index.search(query, limit=limit, kinds=kinds)
        if not hits:
            text = prompts.NO_RESULTS.format(query=query)
            close = self.lookup.suggest(query)
            if close:
                text += " " + prompts.DID_YOU_MEAN.format(names=", ".join(close))
            return text + "\n"
        lines = [
            f"# Search: {query!r} ({len(hits)} of {total} matches, qagents {self.version})",
            "",
        ]
        for number, hit in enumerate(hits, start=1):
            doc = hit.doc
            lines.append(f"{number}. **{doc.title}** [{doc.kind}] `{doc.cite}`")
            text = snippet(doc.text, hit.matched)
            if text:
                lines.append(f"   {text}")
            lines.append(f"   Read: {doc.read}")
        if self.corpus.source.kind == "none":
            lines += ["", prompts.NO_DOCS_FOLDER]
        return "\n".join(lines) + "\n"

    # -- pages -----------------------------------------------------------------------

    def list_pages(self) -> str:
        """The table of contents with summaries, the version and where the docs come from."""
        source = self.corpus.source
        lines = [
            f"# qagents {self.version} documentation",
            "",
            f"Source: {source.describe()}.",
        ]
        lines += [f"Note: {note}" for note in source.notes]
        if source.kind == "none":
            lines.append(prompts.NO_DOCS_FOLDER)
        lines += [
            f"Index: {len(self.corpus.pages)} pages, {len(self.corpus.examples)} examples, "
            f"{len(self.corpus.api)} API entries, built in "
            f"{self.corpus.load_seconds * 1000:.0f} ms.",
            "",
        ]
        current: tuple[str, ...] | None = None
        in_nav = {entry.path for entry in self.corpus.nav}
        others_started = False
        for page in self.corpus.ordered_pages():
            if page.path in in_nav:
                if page.group != current:
                    current = page.group
                    lines += ["", f"## {' / '.join(current) or 'Start'}"]
            elif not others_started:
                others_started = True
                lines += ["", "## Other pages"]
            summary = f": {page.summary}" if page.summary else ""
            lines.append(f"- `{page.path}` **{page.display_title}**{summary}")
        lines += [
            "",
            "Read one with read_page(path) or read_page(path, section=...). Examples: "
            "list_examples().",
        ]
        return "\n".join(lines).replace("\n\n\n", "\n\n") + "\n"

    def read_page(self, path: str, section: str | None = None) -> str:
        """A page, or one section of it, with includes and API directives filled in."""
        page, close = self.corpus.find_page(path)
        if page is None:
            text = prompts.NOT_FOUND_PAGE.format(path=path)
            if close:
                text += " " + prompts.DID_YOU_MEAN.format(names=", ".join(close))
            return text + " list_pages() shows every page.\n"
        header = f"<!-- source: {page.path} | qagents {self.version} -->"
        if section:
            found = self._find_section(page, section)
            if found is None:
                names = [s.anchor for s in page.sections if s.anchor]
                close = difflib.get_close_matches(slugify(section), names, n=4, cutoff=0.4)
                text = prompts.NOT_FOUND_SECTION.format(path=page.path, section=section)
                if close:
                    text += " " + prompts.DID_YOU_MEAN.format(names=", ".join(close))
                return text + "\n\n" + self._section_list(page) + "\n"
            body = self.corpus.expand(page, found.start, found.end, self._render_directive)
            cite = f"{page.path}#{found.anchor}" if found.level > 1 else page.path
            return f"<!-- source: {cite} | qagents {self.version} -->\n{body}\n"
        body = self.corpus.expand(page, 0, len(page.lines), self._render_directive)
        if len(body) > PAGE_LIMIT:
            cut = body[:PAGE_LIMIT].rsplit("\n", 1)[0]
            anchors = ", ".join(s.anchor for s in page.sections if s.level == 2)
            body = f"{cut}\n\n{prompts.PAGE_TRUNCATED.format(limit=PAGE_LIMIT, sections=anchors)}"
        return f"{header}\n{body}\n"

    def page_markdown(self, path: str) -> str | None:
        """The raw Markdown of a page for a resource read, or None."""
        page, _ = self.corpus.find_page(path)
        if page is None:
            return None
        return self.corpus.expand(page, 0, len(page.lines), self._render_directive) + "\n"

    def _find_section(self, page: Page, wanted: str) -> Section | None:
        wanted = wanted.strip().lstrip("#").strip()
        for section in page.sections:
            if section.anchor and section.anchor == wanted:
                return section
        slug = slugify(wanted)
        lowered = wanted.lower()
        for section in page.sections:
            if section.anchor == slug or section.title.lower() == lowered:
                return section
        prefixed = [s for s in page.sections if s.anchor.startswith(slug) and slug]
        return prefixed[0] if len(prefixed) == 1 else None

    def _section_list(self, page: Page) -> str:
        lines = [f"Sections of {page.path}:"]
        for section in page.sections:
            if section.level >= 2:
                indent = "  " * (section.level - 2)
                lines.append(f"{indent}- `{section.anchor}`: {section.title}")
        return "\n".join(lines)

    def _render_directive(self, path: str) -> str:
        resolution = self.lookup.resolve(path)
        if resolution.target is None:
            return f"::: {path}"
        text = self.lookup.render(resolution.target, self.version)
        return re.sub(r"^# ", "## ", text, count=1).rstrip()

    # -- API -------------------------------------------------------------------------

    def get_api(self, symbol: str) -> str:
        """Signature, docstring, members and location of a public qagents object."""
        with self._lock:  # imports of optional extras are not reentrant across threads
            resolution = self.lookup.resolve(symbol)
            if resolution.target is not None:
                text = self.lookup.render(resolution.target, self.version)
                return text + self._related(resolution.target, resolution.also)
        if resolution.candidates:
            lines = [prompts.AMBIGUOUS_SYMBOL.format(symbol=symbol), ""]
            lines += [f"- `{name}`" for name in resolution.candidates]
            return "\n".join(lines) + "\n"
        text = prompts.NOT_FOUND_SYMBOL.format(version=self.version, symbol=symbol)
        if resolution.suggestions:
            text += " " + prompts.DID_YOU_MEAN.format(
                names=", ".join(f"`{name}`" for name in resolution.suggestions)
            )
        text += " search_docs() finds names by what they do."
        return text + "\n"

    def _related(self, target: Target, also: list[str]) -> str:
        """Guide sections and examples that use the object, and other objects of its name."""
        lines: list[str] = []
        if target.kind in ("class", "function", "method", "property", "field"):
            parts = target.path.removeprefix("qagents.").split(".")
            member = target.kind != "class" and len(parts) > 1
            query = ".".join(parts[-2:]) if member else parts[-1]
            hits, _ = self.corpus.index.search(query, limit=3, kinds=["docs", "example"])
            if hits:
                lines += ["", prompts.API_SEE_ALSO]
                lines += [f"- `{hit.doc.cite}` {hit.doc.title}: {hit.doc.read}" for hit in hits]
        if also:
            names = ", ".join(f"`{name}`" for name in also)
            lines += ["", prompts.API_ALSO_NAMED.format(names=names)]
        return ("\n".join(lines) + "\n") if lines else ""

    # -- examples --------------------------------------------------------------------

    def list_examples(self) -> str:
        examples = self.corpus.examples
        if not examples:
            return prompts.NO_DOCS_FOLDER + "\n"
        lines = [f"# qagents {self.version} examples", ""]
        patterns = [e for e in examples.values() if e.name.startswith("patterns/")]
        others = [e for e in examples.values() if not e.name.startswith("patterns/")]
        if patterns:
            lines += ["## Patterns (examples/patterns)", ""]
            for example in patterns:
                mirrors = f" Mirrors: {example.inspired_by}." if example.inspired_by else ""
                lines.append(f"- `{example.short}`: {example.description}.{mirrors}")
            lines.append("")
        if others:
            lines += ["## Other examples (examples/)", ""]
            for example in others:
                lines.append(f"- `{example.short}`: {example.description}.")
            lines.append("")
        lines.append("Read one with read_example(name). Patterns run offline in their tests.")
        return "\n".join(lines) + "\n"

    def find_example(self, name: str) -> tuple[Example | None, list[str]]:
        wanted = name.strip().replace("\\", "/").removesuffix(".py")
        for prefix in ("qagents://examples/", "examples/", "examples.", "./"):
            wanted = wanted.removeprefix(prefix)
        wanted = wanted.replace(".", "/")
        examples = self.corpus.examples
        if wanted in examples:
            return examples[wanted], []
        by_short = [e for e in examples.values() if e.short == wanted.rsplit("/", 1)[-1]]
        if len(by_short) == 1:
            return by_short[0], []
        names = [e.short for e in examples.values()]
        return None, difflib.get_close_matches(wanted.rsplit("/", 1)[-1], names, n=4, cutoff=0.4)

    def read_example(self, name: str) -> str:
        example, close = self.find_example(name)
        if example is None:
            text = prompts.NOT_FOUND_EXAMPLE.format(name=name)
            if close:
                text += " " + prompts.DID_YOU_MEAN.format(names=", ".join(close))
            return text + " list_examples() shows them all.\n"
        lines = [
            f"# Example {example.short}: {example.description}",
            "",
            f"Source: `{example.rel}` | qagents {self.version}",
        ]
        if example.test:
            lines.append(f"Offline test: `{example.test}`")
        code = example.code
        if len(code) > EXAMPLE_LIMIT:
            code = code[:EXAMPLE_LIMIT].rsplit("\n", 1)[0] + "\n# ... (cut)"
        lines += ["", "```python", code.rstrip("\n"), "```"]
        return "\n".join(lines) + "\n"

    def example_source(self, name: str) -> str | None:
        example, _ = self.find_example(name)
        return example.code if example is not None else None

    # -- check -----------------------------------------------------------------------

    def check(self, code: str) -> Report:
        with self._lock:
            return check_code(code, self.lookup)

    def check_code(self, code: str) -> str:
        return self.check(code).render(self.version)

    # -- llms.txt --------------------------------------------------------------------

    def llms_txt(self, site_url: str | None = None) -> str:
        """The llms.txt index: title, summary, and the pages as links with descriptions."""
        from .llms import build_llms_txt

        return build_llms_txt(self.corpus, site_url=site_url)

    def site_url(self) -> str | None:
        return mkdocs_setting(self.corpus.source.mkdocs_yml, "site_url")
