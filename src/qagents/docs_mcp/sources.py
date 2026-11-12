"""Where the documentation comes from.

The docs server answers from the documentation of the qagents that is installed, so what it
says matches the code an agent writes against. It looks in this order and uses the first
place that has a ``docs`` folder:

1. ``--docs-dir`` on the command line (an explicit choice).
2. The copy bundled in the wheel, under ``qagents/docs_mcp/_bundled``.
3. The repository checkout next to the package, for editable installs
   (``<checkout>/src/qagents`` with ``<checkout>/docs``).
4. The directory in the ``QAGENTS_DOCS_DIR`` environment variable: a qagents checkout,
   or its ``docs`` folder.

Without any of them the server still answers from the docstrings of the installed package.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SourceKind = Literal["flag", "bundled", "checkout", "env", "none"]

ENV_VAR = "QAGENTS_DOCS_DIR"
"""Environment variable naming a qagents checkout (or its docs folder) to read docs from."""

BUNDLE_DIR = Path(__file__).resolve().parent / "_bundled"
"""Where a wheel built with the force-include rules keeps docs, examples and README."""

SKILL_PATH = "skills/qagents/SKILL.md"
"""The Claude Code skill, relative to the checkout (or the bundle) root."""

REPOSITORY_URL = "https://github.com/mpvasilis/qagents"
"""Where the qagents sources are published; ``repo_url`` in ``mkdocs.yml`` says the same."""

SKILL_URL = f"{REPOSITORY_URL}/tree/main/skills/qagents"
"""The skill folder in the repository, for an install that has no copy of it."""


@dataclass(frozen=True)
class DocsSource:
    """The files the docs server reads, and how they were found."""

    kind: SourceKind
    root: Path | None
    """The directory that holds ``docs/``, ``examples/``, ``README.md``, ``AGENTS.md``,
    ``skills/qagents/SKILL.md`` and ``mkdocs.yml``."""
    package_dir: Path
    """The installed ``qagents`` package, whose source files give the API reference."""
    version: str
    """The installed qagents version."""
    source_version: str | None = None
    """The version written in the checkout's ``pyproject.toml``, when there is one."""
    notes: tuple[str, ...] = field(default_factory=tuple)
    """Warnings about the choice, for example a version mismatch."""

    @property
    def docs_dir(self) -> Path | None:
        return _existing_dir(self.root / "docs") if self.root else None

    @property
    def examples_dir(self) -> Path | None:
        return _existing_dir(self.root / "examples") if self.root else None

    @property
    def readme(self) -> Path | None:
        return _existing_file(self.root / "README.md") if self.root else None

    @property
    def agents_md(self) -> Path | None:
        return _existing_file(self.root / "AGENTS.md") if self.root else None

    @property
    def skill_md(self) -> Path | None:
        """The Claude Code skill, ``skills/qagents/SKILL.md``."""
        return _existing_file(self.root / SKILL_PATH) if self.root else None

    @property
    def mkdocs_yml(self) -> Path | None:
        return _existing_file(self.root / "mkdocs.yml") if self.root else None

    @property
    def tests_dir(self) -> Path | None:
        return _existing_dir(self.root / "tests") if self.root else None

    def describe(self) -> str:
        """One line that says where the docs come from."""
        where = {
            "flag": "the --docs-dir option",
            "bundled": "the copy bundled with the installed package",
            "checkout": "the repository checkout of the installed package (editable install)",
            "env": f"the {ENV_VAR} environment variable",
            "none": "no docs folder: only the docstrings of the installed package",
        }[self.kind]
        if self.root is None:
            return where
        return f"{where}: {self.root}"

    def resolve_include(self, relative: str) -> Path | None:
        """The file a ``--8<--`` snippet line names, or None.

        Snippet paths are relative to the repository root. In a bundled copy the package
        sources are not under ``src/``, so ``src/qagents/...`` is read from the installed
        package instead.
        """
        relative = relative.strip().strip("\"'")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            return None
        if self.root is not None:
            candidate = _existing_file(self.root / relative)
            if candidate is not None:
                return candidate
        prefix = "src/qagents/"
        if relative.startswith(prefix):
            return _existing_file(self.package_dir / relative[len(prefix) :])
        return None


def find_source(docs_dir: str | os.PathLike[str] | None = None) -> DocsSource:
    """Find the documentation of the installed qagents.

    Args:
        docs_dir: An explicit qagents checkout or docs folder; it wins over everything else.

    Returns:
        The source to read. Its ``kind`` is ``"none"`` when no docs folder was found.
    """
    import qagents

    package_dir = Path(qagents.__file__).resolve().parent
    version = qagents.__version__
    notes: list[str] = []
    env_value = os.environ.get(ENV_VAR, "").strip()

    if docs_dir is not None:
        root = _as_root(Path(docs_dir))
        if root is None:
            notes.append(f"--docs-dir {docs_dir} has no docs folder; it was ignored.")
        else:
            return _with_version("flag", root, package_dir, version, notes)

    if _existing_dir(BUNDLE_DIR / "docs") is not None:
        if env_value:
            notes.append(f"{ENV_VAR} is set but ignored: the bundled docs match this version.")
        return DocsSource("bundled", BUNDLE_DIR, package_dir, version, None, tuple(notes))

    checkout = package_dir.parent.parent
    if package_dir.parent.name == "src" and _is_qagents_checkout(checkout):
        if env_value:
            notes.append(f"{ENV_VAR} is set but ignored: the checkout matches this install.")
        return _with_version("checkout", checkout, package_dir, version, notes)

    if env_value:
        root = _as_root(Path(env_value))
        if root is not None:
            return _with_version("env", root, package_dir, version, notes)
        notes.append(f"{ENV_VAR}={env_value} has no docs folder; it was ignored.")

    return DocsSource("none", None, package_dir, version, None, tuple(notes))


def _with_version(
    kind: SourceKind, root: Path, package_dir: Path, version: str, notes: list[str]
) -> DocsSource:
    source_version = _checkout_version(root)
    if source_version is not None and source_version != version:
        notes.append(
            f"The docs are from qagents {source_version} but qagents {version} is installed; "
            "answers about names and signatures come from the installed package."
        )
    return DocsSource(kind, root, package_dir, version, source_version, tuple(notes))


def _as_root(path: Path) -> Path | None:
    """The checkout root for a path that is a checkout or its docs folder."""
    path = path.expanduser().resolve()
    if _existing_dir(path / "docs") is not None:
        return path
    if path.name == "docs" and path.is_dir():
        return path.parent
    return None


def _is_qagents_checkout(root: Path) -> bool:
    if _existing_dir(root / "docs") is None:
        return False
    return _project_table(root).get("name") == "qagents"


def _checkout_version(root: Path) -> str | None:
    version = _project_table(root).get("version")
    return version if isinstance(version, str) else None


def _project_table(root: Path) -> dict[str, object]:
    """The ``[project]`` table of ``root/pyproject.toml``, or an empty dict."""
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError):  # TOMLDecodeError and UnicodeDecodeError are ValueErrors
        return {}
    project = data.get("project")
    return project if isinstance(project, dict) else {}


def _existing_dir(path: Path) -> Path | None:
    return path if path.is_dir() else None


def _existing_file(path: Path) -> Path | None:
    return path if path.is_file() else None
