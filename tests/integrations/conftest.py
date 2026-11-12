"""Markers and opt-in rules for the integration tests.

| Marker | What the test needs | Runs |
|---|---|---|
| ``devvi_fake`` | a fake peer process (``tests/integrations/fakes``), no Devvi | always (deselect with ``-m "not devvi_fake"``) |
| ``devvi_mock`` | a real Devvi CLI against a local mock model, no API key | only with ``-m devvi_mock`` |
| ``devvi_live`` | a real Devvi CLI and a paid model route | only with ``-m devvi_live`` |

The markers are registered here rather than in ``pyproject.toml``. A test marked
``devvi_mock`` or ``devvi_live`` is skipped unless the ``-m`` expression names that
marker without a ``not`` in front of it, so ``pytest tests/integrations`` stays offline
and free.

The ``isolated_home`` fixture gives a test an empty home directory (``HOME``,
``USERPROFILE``, ``APPDATA``, ``LOCALAPPDATA``, ``DEVVI_HOME``, ``DSH_HOME`` point into
``tmp_path``) so that nothing reads or writes the user's own configuration.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

OPT_IN = {
    "devvi_mock": "needs the Devvi CLI and runs it against a local mock model (opt-in)",
    "devvi_live": "needs the Devvi CLI and a paid model route; costs money (opt-in)",
}
MARKERS = {"devvi_fake": "starts a fake peer process; no Devvi needed", **OPT_IN}


def pytest_configure(config: pytest.Config) -> None:
    for name, text in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {text}")


def selected_markers(expression: str) -> set[str]:
    """Names the ``-m`` expression mentions without a ``not`` directly in front."""
    names: set[str] = set()
    for negation, name in re.findall(r"(\bnot\s+)?\b([A-Za-z_][A-Za-z0-9_]*)\b", expression):
        if name not in ("and", "or", "not") and not negation:
            names.add(name)
    return names


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    selected = selected_markers(config.getoption("markexpr") or "")
    for item in items:
        if HERE not in Path(str(item.path)).resolve().parents:
            continue  # the hook sees every test of the session; only ours are opt-in
        for name in OPT_IN:
            if item.get_closest_marker(name) is not None and name not in selected:
                item.add_marker(pytest.mark.skip(reason=f"opt-in: run with -m {name}"))


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty home for code that looks up home or config directories."""
    home = tmp_path / "home"
    for sub in ("", "AppData/Roaming", "AppData/Local", ".devvi"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(home / "AppData" / "Local"))
    monkeypatch.setenv("DEVVI_HOME", str(home / ".devvi"))
    monkeypatch.setenv("DSH_HOME", str(home / ".devvi"))
    return home
