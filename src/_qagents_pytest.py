"""Lightweight pytest entry point; importing it never imports qagents."""

import pytest


@pytest.fixture
def crash_env(tmp_path):
    from qagents.testing.crash.plugin import CrashRunner

    return CrashRunner(tmp_path)
