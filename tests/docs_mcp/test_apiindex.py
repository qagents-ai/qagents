"""The static API index: which public name each object gets."""

from __future__ import annotations

import textwrap
from pathlib import Path

from qagents.docs_mcp.apiindex import build_api_index


def write(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    return root


def publics(package: Path) -> dict[str, str]:
    return {entry.path: entry.public for entry in build_api_index(package, package.parent)}


def test_a_package_re_exports_only_what_its_all_lists(tmp_path):
    package = write(
        tmp_path / "qagents",
        {
            "__init__.py": "from .core import Base, helper\n\n__all__ = ['Base']\n",
            "core.py": "class Base:\n    'A base.'\n\n\ndef helper():\n    'Help.'\n",
        },
    )
    found = publics(package)
    assert found["qagents.core.Base"] == "qagents.Base"
    assert found["qagents.core.helper"] == "qagents.core.helper"  # imported, not exported


def test_without_all_every_public_import_counts(tmp_path):
    package = write(
        tmp_path / "qagents",
        {"__init__.py": "from .core import Base\n", "core.py": "class Base:\n    'A base.'\n"},
    )
    assert publics(package)["qagents.core.Base"] == "qagents.Base"


def test_a_longer_alias_never_replaces_the_defining_path(tmp_path):
    """An integration that imports a core class does not become the class's home."""
    package = write(
        tmp_path / "qagents",
        {
            "__init__.py": "",
            "external.py": "class ExternalAgent:\n    'The base.'\n\n    def run(self):\n"
            "        'Run.'\n",
            "integrations/__init__.py": "",
            "integrations/devvi/__init__.py": """\
                from ...external import ExternalAgent
                from .agent import DevviAgent

                __all__ = ["DevviAgent", "ExternalAgent"]
                """,
            "integrations/devvi/agent.py": "class DevviAgent:\n    'Devvi.'\n",
        },
    )
    found = publics(package)
    assert found["qagents.external.ExternalAgent"] == "qagents.external.ExternalAgent"
    assert found["qagents.external.ExternalAgent.run"] == "qagents.external.ExternalAgent.run"
    assert found["qagents.integrations.devvi.agent.DevviAgent"] == (
        "qagents.integrations.devvi.DevviAgent"
    )


def test_lazy_names_and_extended_all(tmp_path):
    package = write(
        tmp_path / "qagents",
        {
            "__init__.py": """\
                from .core import Base, Extra

                __all__ = ["Base"]
                __all__ += ["Extra"]
                _LAZY = {"Toolkit": "qagents.toolkits.mcp"}
                """,
            "core.py": "class Base:\n    'A base.'\n\n\nclass Extra:\n    'More.'\n",
            "toolkits/__init__.py": "",
            "toolkits/mcp.py": "class Toolkit:\n    'Needs an extra.'\n",
        },
    )
    found = publics(package)
    assert found["qagents.core.Extra"] == "qagents.Extra"
    assert found["qagents.toolkits.mcp.Toolkit"] == "qagents.Toolkit"
