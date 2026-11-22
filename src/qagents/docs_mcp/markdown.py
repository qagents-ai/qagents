"""Just enough Markdown for the docs server: headings, anchors, the nav, summaries.

Anchors are made the way MkDocs makes them (Python-Markdown's ``toc`` extension with its
default ``slugify``), so ``page.md#anchor`` in an answer is also a working link on the
published site.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")
_ATTR_ID = re.compile(r"\s*\{[^}]*#([\w-]+)[^}]*\}\s*$")
_ATTR_LIST = re.compile(r"\s*\{:?[^}]*\}\s*$")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_REF_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_HTML_TAG = re.compile(r"<[^>]+>")
_UNIQUE_SUFFIX = re.compile(r"^(.*)_([0-9]+)$")


@dataclass(frozen=True)
class Section:
    """One heading of a page and the lines under it."""

    anchor: str
    """The heading's id on the published page; empty for the text above the first heading."""
    title: str
    """The heading as plain text."""
    level: int
    """1 for ``#``, 2 for ``##`` and so on; 0 for the text above the first heading."""
    trail: tuple[str, ...]
    """The titles of the enclosing headings and this one, outermost first."""
    start: int
    """Index of the heading line in the page's lines (0-based)."""
    body_end: int
    """End (exclusive) of this heading's own text, before the next heading of any level."""
    end: int
    """End (exclusive) of the section with its subsections, before the next heading of the
    same or a higher level."""


def strip_inline(text: str) -> str:
    """Markdown inline markup removed: links keep their text, code keeps its content."""
    text = _LINK.sub(r"\1", text)
    text = _REF_LINK.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    text = text.replace("`", "")
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)
    return text.strip()


def slugify(text: str) -> str:
    """The id Python-Markdown's default ``slugify`` gives a heading's text."""
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def _unique(anchor: str, used: set[str]) -> str:
    while anchor in used or not anchor:
        match = _UNIQUE_SUFFIX.match(anchor)
        if match:
            anchor = f"{match.group(1)}_{int(match.group(2)) + 1}"
        else:
            anchor = f"{anchor}_1"
    used.add(anchor)
    return anchor


def without_front_matter(lines: list[str]) -> list[str]:
    """The lines with a leading YAML front matter block (``---`` to ``---``) blanked out.

    The line count stays the same, so line numbers still match the file. A Claude Code
    skill (``SKILL.md``) and MkDocs pages with metadata start this way.
    """
    if not lines or lines[0].strip() != "---":
        return lines
    for index in range(1, len(lines)):
        if lines[index].strip() in ("---", "..."):
            return [""] * (index + 1) + lines[index + 1 :]
    return lines


def code_mask(lines: list[str]) -> list[bool]:
    """For each line, whether it is inside a fenced code block (fence lines included)."""
    mask: list[bool] = []
    fence: str | None = None
    for line in lines:
        match = _FENCE.match(line)
        if fence is None:
            if match:
                fence = match.group(1)
                mask.append(True)
                continue
            mask.append(False)
        else:
            mask.append(True)
            if match and match.group(1)[0] == fence[0] and len(match.group(1)) >= len(fence):
                if not match.group(2).strip():
                    fence = None
    return mask


def parse_sections(lines: list[str]) -> list[Section]:
    """The headings of a page, in order, with the line ranges they cover.

    The first entry covers the text above the first heading (level 0) when there is any.
    Lines inside fenced code blocks are never headings.
    """
    in_code = code_mask(lines)
    found: list[tuple[int, int, str, str]] = []  # (line, level, title, anchor)
    used: set[str] = set()
    for index, line in enumerate(lines):
        if in_code[index]:
            continue
        match = _HEADING.match(line)
        if not match:
            continue
        raw = match.group(2)
        explicit = _ATTR_ID.search(raw)
        raw = _ATTR_LIST.sub("", raw)
        title = strip_inline(raw)
        anchor = explicit.group(1) if explicit else slugify(title)
        found.append((index, len(match.group(1)), title, _unique(anchor, used)))

    sections: list[Section] = []
    first = found[0][0] if found else len(lines)
    if any(line.strip() for line in lines[:first]):
        sections.append(Section("", "", 0, (), 0, first, first))
    stack: list[tuple[int, str]] = []
    for position, (index, level, title, anchor) in enumerate(found):
        body_end = found[position + 1][0] if position + 1 < len(found) else len(lines)
        end = len(lines)
        for later_index, later_level, _, _ in found[position + 1 :]:
            if later_level <= level:
                end = later_index
                break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        trail = tuple(name for _, name in stack)
        sections.append(Section(anchor, title, level, trail, index, body_end, end))
    return sections


def summary(lines: list[str], limit: int = 200) -> str:
    """A one-line description of a page, as plain text.

    The first paragraph under the title, cut to its first sentence (two when the first is
    short). A page with no paragraph before its first ``##`` is described by its section
    titles; a page of API directives (``::: qagents.x.Y``) by the names it documents.
    """
    in_code = code_mask(lines)
    paragraph: list[str] = []
    titles: list[str] = []
    directives: list[str] = []
    in_intro = True
    for index, line in enumerate(lines):
        stripped = line.strip()
        heading = None if in_code[index] else _HEADING.match(line)
        if heading:
            if len(heading.group(1)) >= 2:
                in_intro = False
                if len(heading.group(1)) == 2:
                    titles.append(strip_inline(_ATTR_LIST.sub("", heading.group(2))))
            continue
        if in_code[index] or not in_intro:
            continue
        if stripped.startswith(":::"):
            target = stripped[3:].strip()
            last = target.rsplit(".", 1)[-1]
            directives.append(last if last[:1].isupper() or "_" in last else target)
            continue
        if paragraph:
            if not stripped or stripped.startswith(("!!!", "???", "|", "<", "--8<--", ">", "- ")):
                in_intro = False
                continue
            paragraph.append(stripped)
            continue
        if not stripped or stripped.startswith(
            ("!!!", "???", "|", "<", "--8<--", ">", "- [", "---")
        ):
            continue
        if stripped.startswith("    ") or line.startswith("    "):
            continue  # the body of an admonition
        paragraph.append(stripped)
    text = strip_inline(" ".join(paragraph))
    if text:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z`\[(])", text)
        result = sentences[0]
        if len(result) < 60 and len(sentences) > 1:
            result = f"{result} {sentences[1]}"
    elif directives:
        result = "API reference: " + ", ".join(directives) + "."
    elif titles:
        result = "Sections: " + ", ".join(titles) + "."
    else:
        result = ""
    if len(result) > limit:
        result = result[: limit - 3].rsplit(" ", 1)[0] + "..."
    return result


@dataclass(frozen=True)
class NavEntry:
    """One page in the site navigation."""

    title: str
    path: str
    """Relative to the docs folder, with forward slashes."""
    group: tuple[str, ...]
    """The nav sections that contain the page, outermost first."""


def parse_nav(mkdocs_yml: Path) -> list[NavEntry]:
    """The pages of ``nav:`` in ``mkdocs.yml``, in order.

    Reads the small YAML subset MkDocs navs use (``- Title: page.md``, ``- page.md`` and
    nested sections). Anything else in the file is ignored. Returns an empty list when the
    file has no ``nav``; external links are left out.
    """
    try:
        lines = mkdocs_yml.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    entries: list[NavEntry] = []
    inside = False
    groups: list[tuple[int, str]] = []
    for line in lines:
        if not inside:
            if re.match(r"^nav\s*:\s*$", line):
                inside = True
            continue
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            break  # the next top-level key
        item = line.strip()
        if not item.startswith("- "):
            continue
        item = item[2:].strip()
        while groups and groups[-1][0] >= indent:
            groups.pop()
        title, separator, value = item.partition(":")
        if separator and not value.strip() and not item.endswith("://"):
            groups.append((indent, _unquote(title)))
            continue
        if separator and not _looks_like_url(item):
            title, value = _unquote(title), _unquote(value)
        else:
            title, value = "", _unquote(item)
        if _looks_like_url(value) or not value.endswith(".md"):
            continue
        entries.append(NavEntry(title, value, tuple(name for _, name in groups)))
    return entries


def _looks_like_url(value: str) -> bool:
    return bool(re.match(r"^[a-z][a-z0-9+.-]*://", value.strip().strip("'\"")))


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def mkdocs_setting(mkdocs_yml: Path | None, key: str) -> str | None:
    """A top-level scalar setting of ``mkdocs.yml``, such as ``site_url``, or None."""
    if mkdocs_yml is None:
        return None
    try:
        text = mkdocs_yml.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(rf"^{re.escape(key)}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    if not match:
        return None
    value = _unquote(match.group(1).split(" #", 1)[0])
    return value or None
