"""Pytest plugin for a single owned child process with one seeded mutation."""

import inspect
import json
import os
from functools import wraps
from pathlib import Path
from pkgutil import resolve_name

import pytest

from .catalogue import CASES

_calls = 0
_failed = []
_passed = []


def pytest_configure(config):
    global _calls
    key = os.environ["QAGENTS_MUTATION_ID"]
    case = next(item for item in CASES if item.id == key)
    seam = case.seam.replace("tests/effects/mutants/", f"{__package__}.")
    module, attr = seam.rsplit(".", 1)
    target = resolve_name(module)
    original = getattr(target, attr)

    if os.environ["QAGENTS_MUTATION_MODE"] == "baseline":
        return

    def replace(args, kwargs):
        global _calls
        _calls += 1
        return case.replacement(original, *args, **kwargs)

    if inspect.iscoroutinefunction(original):

        @wraps(original)
        async def seeded(*args, **kwargs):
            value = replace(args, kwargs)
            return await value if inspect.isawaitable(value) else value
    else:

        @wraps(original)
        def seeded(*args, **kwargs):
            return replace(args, kwargs)

    setattr(target, attr, seeded)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.when == "call" and report.passed:
        _passed.append(report.nodeid)
    if report.failed and call.excinfo is not None:
        failure = call.excinfo
        oracle = failure.errisinstance(AssertionError) or (
            failure.errisinstance(pytest.fail.Exception)
            and str(failure.value).startswith("DID NOT RAISE")
        )
        _failed.append({"node": report.nodeid, "phase": report.when, "oracle": oracle})
    return report


def pytest_sessionfinish(session, exitstatus):
    evidence = {
        "calls": _calls,
        "failed": _failed,
        "passed": _passed,
        "exitstatus": int(exitstatus),
    }
    Path(os.environ["QAGENTS_MUTATION_RESULT"]).write_text(json.dumps(evidence), encoding="utf-8")
