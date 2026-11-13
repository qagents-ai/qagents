"""Everything the docs server knows, loaded once at startup.

`Corpus.load` reads the docs pages (in ``mkdocs.yml`` nav order), the README, the examples
and the public API of the installed package, and builds the search index. It imports no
qagents module beyond the package itself, so it is fast and needs no optional extras.
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from . import markdown, prompts
from .apiindex import ApiEntry, build_api_index
from .search import BM25Index, SearchDoc
from .sources import SKILL_PATH, DocsSource, find_source

EXTRA_PAGES = ("README.md", "examples/README.md", "AGENTS.md", SKILL_PATH)
"""Pages from outside the docs folder, in the order `Corpus.ordered_pages` lists them last.
They are not pages of the published site, so ``llms.txt`` leaves them out."""

PRIORS = {
    "docs": 1.0,
    "README.md": 0.9,
    "examples/README.md": 0.95,
    "AGENTS.md": 0.7,
    SKILL_PATH: 0.6,
}
"""Search weight of each kind of page; API entries and examples have their own below."""
PREFIX_PRIORS = {
    "reference/": 0.6,
    "research.md": 0.6,
}
"""Search weight of docs pages by path (a folder ends with ``/``), when `PRIORS` has none.
The API reference pages hold only ``::: name`` lines (the objects themselves are indexed as
API entries), and the research report is mostly in Greek."""


@dataclass(frozen=True)
class OffTopic:
    """How a page on one integration weighs for a query that does not name its topic."""

    weight: float
    """The page's search weight for such a query."""
    words: str = ""
    """Words that name the topic, though the page may not use them (``codex``)."""
    titles: bool = True
    """Whether the words of the page's titles and file name name its topic too (those not
    common to the whole corpus)."""


_CODING_AGENTS = "coding coder claude codex gemini"
"""What people call the coding agents that the external-agent pages put to work, and that
the tool server serves tools to."""
OFF_TOPIC_PRIORS = {
    "concepts/external-agents.md": OffTopic(0.5, f"acp devvi worker drive {_CODING_AGENTS}"),
    "integrations/acp.md": OffTopic(0.5, f"external worker drive {_CODING_AGENTS}"),
    "integrations/devvi.md": OffTopic(0.5, f"external worker drive {_CODING_AGENTS}"),
    "integrations/mcp-server.md": OffTopic(0.5, f"expose external {_CODING_AGENTS}"),
    "integrations/": OffTopic(0.5),
    "coding-agents.md": OffTopic(0.5, "docs documentation llms", titles=False),
}
"""The pages on one integration, by path (a folder ends with ``/``), and what names their
topic (`OffTopic`).

These pages use the core words, ``agent``, ``tool``, ``session``, ``resume``, about their
own topic, so a plain question about the core (``agent tools``) would otherwise land on
them. A question about their topic names it and finds them at full weight: a word of the
page's title or file name (``devvi``, ``acp``, ``mcp``, ``external``), or one of its
`OffTopic.words`. The external-agent pages are about putting another coding agent to work,
so the agents people name for that (Claude Code, Codex, Gemini CLI) name their topic. The
page on the docs server is about the docs: the coding agents in its title are its readers,
and a question about driving Claude Code is not about it."""
API_PRIOR = 0.85
EXAMPLE_PRIOR = 0.9


_T = TypeVar("_T")


def _by_prefix(path: str, table: dict[str, _T]) -> _T | None:
    """The value of the first key that is the path, or a folder (``.../``) holding it."""
    for prefix, prior in table.items():
        if path == prefix or (prefix.endswith("/") and path.startswith(prefix)):
            return prior
    return None


def page_prior(path: str) -> float:
    """The search weight of a page: `PRIORS`, then `PREFIX_PRIORS`, else 1."""
    if path in PRIORS:
        return PRIORS[path]
    prior = _by_prefix(path, PREFIX_PRIORS)
    return PRIORS["docs"] if prior is None else prior


_DIRECTIVE = re.compile(r"^:::\s+([\w.]+)\s*$")
_SNIPPET = re.compile(r"""^[ \t]*-{2}8<-{2}[ \t]+["']([^"']+)["'][ \t]*$""")


@dataclass
class Page:
    """One Markdown page."""

    path: str
    """How tools and resources name it: relative to the docs folder (``concepts/tools.md``),
    or one of `EXTRA_PAGES` (``README.md``, ``AGENTS.md``, ...) for the files outside it."""
    file: Path
    title: str
    summary: str
    lines: list[str]
    sections: list[markdown.Section]
    nav_title: str | None = None
    group: tuple[str, ...] = ()
    """The nav sections that contain it; empty when it is not in the nav."""

    @property
    def display_title(self) -> str:
        return self.nav_title or self.title or self.path

    def section(self, anchor: str) -> markdown.Section | None:
        return next((s for s in self.sections if s.anchor == anchor), None)


@dataclass
class Example:
    """One runnable example module."""

    name: str
    """``patterns/supervisor`` for a pattern, ``quickstart`` for the other examples."""
    file: Path
    rel: str
    """Path shown in answers, for example ``examples/patterns/supervisor.py``."""
    description: str
    docstring: str
    inspired_by: str | None
    code: str
    uses: tuple[str, ...]
    """qagents names the code uses (imports and attributes), for search."""
    test: str | None = None
    """The offline test of the example, when the checkout has one."""

    @property
    def short(self) -> str:
        return self.name.rsplit("/", 1)[-1]


@dataclass
class Corpus:
    source: DocsSource
    pages: dict[str, Page]
    nav: list[markdown.NavEntry]
    examples: dict[str, Example]
    api: list[ApiEntry]
    index: BM25Index
    load_seconds: float
    api_by_path: dict[str, ApiEntry] = field(default_factory=dict)
    api_by_public: dict[str, ApiEntry] = field(default_factory=dict)

    @classmethod
    def load(cls, source: DocsSource | None = None) -> Corpus:
        """Read every page, example and API docstring and build the search index."""
        started = time.perf_counter()
        source = source or find_source()
        pages = _load_pages(source)
        nav = markdown.parse_nav(source.mkdocs_yml) if source.mkdocs_yml else []
        for entry in nav:
            page = pages.get(entry.path)
            if page is not None:
                page.nav_title = entry.title or page.title
                page.group = entry.group
        examples = _load_examples(source)
        display_root = source.root if _inside(source.package_dir, source.root) else None
        api = build_api_index(source.package_dir, display_root or source.package_dir.parent)
        docs = [*_page_docs(pages), *_example_docs(examples), *_api_docs(api)]
        index = BM25Index(docs)
        corpus = cls(source, pages, nav, examples, api, index, 0.0)
        corpus.api_by_path = {entry.path: entry for entry in api}
        corpus.api_by_public = {entry.public: entry for entry in api}
        corpus.load_seconds = time.perf_counter() - started
        return corpus

    def ordered_pages(self) -> list[Page]:
        """Pages in nav order, then the pages the nav does not list, then the extra files."""
        seen: set[str] = set()
        ordered: list[Page] = []
        for entry in self.nav:
            page = self.pages.get(entry.path)
            if page is not None and entry.path not in seen:
                ordered.append(page)
                seen.add(entry.path)
        rest = sorted(p for p in self.pages if p not in seen and p not in EXTRA_PAGES)
        extra = [p for p in EXTRA_PAGES if p in self.pages and p not in seen]
        ordered.extend(self.pages[p] for p in [*rest, *extra])
        return ordered

    def find_page(self, path: str) -> tuple[Page | None, list[str]]:
        """The page a loosely written path names, and close names when there is none.

        Accepts ``concepts/tools.md``, ``docs/concepts/tools.md``, ``concepts/tools``,
        ``tools`` (when only one page has that name), a nav title such as ``Tools``, and
        a published URL path such as ``concepts/tools/``.
        """
        wanted = path.strip().strip("/").replace("\\", "/")
        wanted = re.sub(r"#.*$", "", wanted)
        for prefix in ("qagents://docs/", "docs/", "./"):
            if wanted.startswith(prefix):
                wanted = wanted[len(prefix) :]
        if wanted in self.pages:
            return self.pages[wanted], []
        candidates = [wanted, f"{wanted}.md", f"{wanted}/index.md"]
        if wanted.endswith(".html"):
            candidates.append(wanted[: -len(".html")] + ".md")
        for candidate in candidates:
            for key in (candidate, candidate.lower()):
                if key in self.pages:
                    return self.pages[key], []
        lowered = wanted.lower().removesuffix(".md")
        by_stem = [p for p in self.pages.values() if Path(p.path).stem.lower() == lowered]
        if len(by_stem) == 1:
            return by_stem[0], []
        by_title = [
            p
            for p in self.pages.values()
            if lowered in {p.title.lower(), (p.nav_title or "").lower()}
        ]
        if len(by_title) == 1:
            return by_title[0], []
        import difflib

        names = list(self.pages)
        close = difflib.get_close_matches(wanted, names, n=5, cutoff=0.4)
        close += [p.path for p in by_stem + by_title if p.path not in close]
        return None, close

    def expand(self, page: Page, start: int, end: int, render_api: object = None) -> str:
        """Lines ``start:end`` of a page with snippet includes and API directives filled in.

        Args:
            render_api: A function from an object path (``qagents.agent.Agent``) to Markdown,
                for ``::: path`` lines. Without it they are left as they are.
        """
        out: list[str] = []
        in_code = markdown.code_mask(page.lines)
        for index in range(start, end):
            line = page.lines[index]
            snippet = _SNIPPET.match(line)
            if snippet:
                included = self.source.resolve_include(snippet.group(1))
                if included is not None:
                    try:
                        out.extend(included.read_text(encoding="utf-8").splitlines())
                        continue
                    except (OSError, UnicodeDecodeError):
                        pass
            directive = _DIRECTIVE.match(line) if not in_code[index] else None
            if directive and callable(render_api):
                out.append(render_api(directive.group(1)))
                continue
            out.append(line)
        return "\n".join(out).strip("\n")


def _inside(path: Path, root: Path | None) -> bool:
    if root is None:
        return False
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_lines(file: Path) -> list[str] | None:
    try:
        return file.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError):
        return None


def _make_page(path: str, file: Path) -> Page | None:
    lines = _read_lines(file)
    if lines is None:
        return None
    visible = markdown.without_front_matter(lines)  # the page keeps it; sections do not
    sections = markdown.parse_sections(visible)
    title = next((s.title for s in sections if s.level == 1), "") or Path(path).stem
    return Page(path, file, title, markdown.summary(visible), lines, sections)


def _load_pages(source: DocsSource) -> dict[str, Page]:
    pages: dict[str, Page] = {}
    docs_dir = source.docs_dir
    if docs_dir is not None:
        for file in sorted(docs_dir.rglob("*.md")):
            relative = file.relative_to(docs_dir).as_posix()
            if any(part.startswith(".") for part in Path(relative).parts):
                continue
            page = _make_page(relative, file)
            if page is not None:
                pages[relative] = page
    extras = {
        "README.md": source.readme,
        "examples/README.md": (
            source.examples_dir / "README.md" if source.examples_dir is not None else None
        ),
        "AGENTS.md": source.agents_md,
        SKILL_PATH: source.skill_md,
    }
    for path, file in extras.items():
        if file is not None and file.is_file() and path not in pages:
            page = _make_page(path, file)
            if page is not None:
                if path == SKILL_PATH:
                    page.nav_title = prompts.SKILL_PAGE_TITLE  # its own title is "qagents"
                pages[path] = page
    return pages


def _load_examples(source: DocsSource) -> dict[str, Example]:
    examples: dict[str, Example] = {}
    folder = source.examples_dir
    if folder is None:
        return examples
    files = sorted(folder.glob("patterns/*.py")) + sorted(folder.glob("*.py"))
    for file in files:
        if file.name.startswith("_"):
            continue
        relative = file.relative_to(folder).with_suffix("").as_posix()
        try:
            code = file.read_text(encoding="utf-8")
            tree = ast.parse(code)
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
        docstring = ast.get_docstring(tree) or ""
        constants = _string_constants(tree)
        description = constants.get("DESCRIPTION") or docstring.strip().split("\n", 1)[0]
        test = None
        if source.tests_dir is not None:
            candidate = source.tests_dir / "examples" / f"test_{file.stem}.py"
            if candidate.is_file():
                test = f"tests/examples/{candidate.name}"
        examples[relative] = Example(
            name=relative,
            file=file,
            rel=f"examples/{file.relative_to(folder).as_posix()}",
            description=description.rstrip("."),
            docstring=docstring,
            inspired_by=constants.get("INSPIRED_BY"),
            code=code,
            uses=_qagents_names(tree),
            test=test,
        )
    return examples


def _string_constants(tree: ast.Module) -> dict[str, str]:
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        found[target.id] = node.value.value
    return found


def _qagents_names(tree: ast.Module) -> tuple[str, ...]:
    """Names imported from qagents and attribute names used anywhere in the code."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("qagents"):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    return tuple(sorted(name for name in names if name and not name.startswith("_")))


# --- search documents ----------------------------------------------------------------


def _page_docs(pages: dict[str, Page]) -> list[SearchDoc]:
    docs: list[SearchDoc] = []
    for page in pages.values():
        prior = page_prior(page.path)
        name = Path(page.path).stem.replace("-", " ")  # mcp-server.md: "mcp server"
        if page.path in EXTRA_PAGES or name == "index":
            name = ""
        # page.title is never empty: a page without an H1 is titled by its file name
        titles = tuple(dict.fromkeys(t for t in (page.title, page.nav_title, name) if t))
        off_topic = _by_prefix(page.path, OFF_TOPIC_PRIORS) or OffTopic(1.0)
        topic = " ".join(titles) if off_topic.weight != 1.0 and off_topic.titles else ""
        for section in page.sections:
            body = "\n".join(
                page.lines[section.start + (1 if section.level else 0) : section.body_end]
            )
            if not body.strip() and section.level != 1:
                continue
            # a section of only ``::: name`` lines says nothing of its own: its title is not
            # matched as a heading, so the API entries it lists rank above it
            stub = bool(body.strip()) and all(
                not line.strip() or _DIRECTIVE.match(line.strip()) for line in body.splitlines()
            )
            heading = section.title if section.level > 1 else titles[0]
            above = titles if section.level > 1 else titles[1:]
            trail = " > ".join(section.trail) if section.trail else page.title
            title = trail if section.level else page.title
            anchor = section.anchor if section.level > 1 else ""
            cite = f"{page.path}#{anchor}" if anchor else page.path
            read = (
                f'read_page("{page.path}", section="{anchor}")'
                if anchor
                else f'read_page("{page.path}")'
            )
            docs.append(
                SearchDoc(
                    kind="docs",
                    title=title,
                    text=body,
                    cite=cite,
                    read=read,
                    group=page.path,
                    boost=prior,
                    extra=page.summary if section.level <= 1 else "",
                    heading="" if stub else heading,
                    titles=() if stub else above,
                    topic=topic,
                    topic_words=off_topic.words,
                    off_topic=off_topic.weight,
                )
            )
    return docs


def _example_docs(examples: dict[str, Example]) -> list[SearchDoc]:
    docs = []
    for example in examples.values():
        first = example.docstring.strip().split("\n", 1)[0].rstrip(".")
        same = first.lower() == example.description.lower()
        text = example.docstring if same else f"{example.description}.\n{example.docstring}"
        docs.append(
            SearchDoc(
                kind="example",
                title=f"Example {example.short}: {example.description}",
                text=text,
                cite=f"{example.rel}:1",
                read=f'read_example("{example.name}")',
                group=example.rel,
                boost=EXAMPLE_PRIOR,
                names=(example.short.lower(), example.name.lower()),
                extra=" ".join(example.uses),
                titles=(example.description,),
            )
        )
    return docs


def _api_docs(api: list[ApiEntry]) -> list[SearchDoc]:
    docs = []
    for entry in api:
        if entry.kind == "attribute" and not entry.doc:
            continue
        short = entry.public.removeprefix("qagents.")
        names = {entry.public.lower(), short.lower(), entry.name.lower(), entry.path.lower()}
        text = f"{entry.signature}\n{entry.doc}" if entry.signature else entry.doc
        if entry.kind == "attribute" and entry.value:
            text = f"{text}\n{entry.value}"
        docs.append(
            SearchDoc(
                kind="api",
                title=f"{short} ({entry.kind})",
                text=text,
                cite=f"{entry.file}:{entry.line}",
                read=f'get_api("{entry.public}")',
                group=entry.path.rsplit(".", 1)[0]
                if entry.kind in ("method", "property")
                else entry.path,
                boost=API_PRIOR * (0.8 if entry.kind == "module" else 1.0),
                names=tuple(sorted(names)),
            )
        )
    return docs
