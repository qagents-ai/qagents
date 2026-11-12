"""Fixtures of the docs server tests: the real corpus of this checkout, and a small fake one.

A test marked ``docs_benchmark`` measures wall-clock time. It is skipped unless the ``-m``
expression names the marker (``pytest tests/docs_mcp -m docs_benchmark``), so a slow or busy
machine never fails the default suite.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from qagents.docs_mcp.service import DocsService
from qagents.docs_mcp.sources import find_source

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
FENCE = re.compile(r"^```python[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
BENCHMARK = "docs_benchmark"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", f"{BENCHMARK}: a wall-clock bound on the docs server (opt-in: -m {BENCHMARK})"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    expression = config.getoption("markexpr") or ""
    named = re.findall(r"(\bnot\s+)?\b([A-Za-z_]\w*)\b", expression)
    if any(name == BENCHMARK and not negation for negation, name in named):
        return
    for item in items:
        if HERE not in Path(str(item.path)).resolve().parents:
            continue  # the hook sees every test of the session; only ours are opt-in
        if item.get_closest_marker(BENCHMARK) is not None:
            item.add_marker(pytest.mark.skip(reason=f"opt-in: run with -m {BENCHMARK}"))


@pytest.fixture(scope="session")
def service() -> DocsService:
    """The docs of this checkout, loaded once for the whole session."""
    return DocsService(source=find_source(ROOT))


def python_blocks(path: Path) -> list[tuple[int, str]]:
    """The ``python`` code blocks of a Markdown file, with the line each starts on."""
    text = path.read_text(encoding="utf-8")
    return [
        (text[: match.start()].count("\n") + 1, match.group(1))
        for match in FENCE.finditer(text)
        if "--8<--" not in match.group(1)
    ]


@pytest.fixture
def fake_checkout(tmp_path: Path) -> Path:
    """A tiny qagents-like checkout: two pages, a nav, an example and a README."""
    root = tmp_path / "checkout"
    (root / "docs" / "guide").mkdir(parents=True)
    (root / "examples" / "patterns").mkdir(parents=True)
    (root / "mkdocs.yml").write_text(
        textwrap.dedent(
            """\
            site_name: fakedocs
            site_description: A fake library for tests.
            nav:
              - Home: index.md
              - Guide:
                  - Widgets: guide/widgets.md
              - External: https://example.com
            markdown_extensions:
              - toc
            """
        ),
        encoding="utf-8",
    )
    (root / "docs" / "index.md").write_text(
        "# Fake\n\nThe fake library makes widgets. It is only for tests.\n", encoding="utf-8"
    )
    (root / "docs" / "guide" / "widgets.md").write_text(
        textwrap.dedent(
            """\
            # Widgets

            Widgets are made by `make_widget`.

            ## Making a widget

            Call it with a size.

            ```bash
            # not a heading: it is inside a code block
            echo widget
            ```

            ### Sizes `small` and `large`

            Sizes change the widget.

            ## Making a widget

            A second section with the same title.
            """
        ),
        encoding="utf-8",
    )
    (root / "README.md").write_text("# fake\n\nA fake README.\n", encoding="utf-8")
    (root / "examples" / "patterns" / "gadget.py").write_text(
        '"""Make a gadget out of widgets."""\n\nNAME = "gadget"\n'
        'DESCRIPTION = "A gadget from widgets"\n',
        encoding="utf-8",
    )
    return root
