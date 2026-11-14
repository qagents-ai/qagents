"""Enforce the frozen design independently of each owner's implementation."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from pydantic_core import PydanticUndefined

from qagents.effects._types import MISSING

from ._inventory import INTERFACES, SEAMS

ROOT = Path(__file__).resolve().parents[2]


def resolve(module, name):
    value = importlib.import_module(module)
    for part in name.split("."):
        value = inspect.getattr_static(value, part)
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
    return value


def default_value(expression):
    return eval(expression, {"__builtins__": {}, "frozenset": frozenset, "MISSING": MISSING})


@pytest.mark.parametrize(
    "module,name,owner,kind,contract,partial",
    INTERFACES,
    ids=[f"owner_{owner}:{module}.{name}" for module, name, owner, *_ in INTERFACES],
)
def test_stage1_frozen_interfaces(module, name, owner, kind, contract, partial):
    value = resolve(module, name)
    if kind == "call":
        params = inspect.signature(value).parameters
        expected = [item[0] for item in contract]
        if not partial:
            assert list(params) == expected
        else:
            assert set(expected) <= params.keys()
        for parameter, category, default in contract:
            actual = params[parameter]
            assert actual.kind.name == category, parameter
            # The plan's abbreviated Agent signatures omit this pre-existing default.
            if partial and module == "qagents.agent" and parameter == "prompt":
                default = "None"
            assert actual.default == (
                inspect.Parameter.empty if default is None else default_value(default)
            ), parameter
        return
    annotations = inspect.get_annotations(value)
    fields = (
        value.model_fields
        if hasattr(value, "model_fields")
        else {field.name: field for field in dataclasses.fields(value)}
        if dataclasses.is_dataclass(value)
        else {}
    )
    for field_name, default in contract:
        # Effect and Tool store some frozen fields on instances rather than the class.
        instance = value
        if name == "Effect":
            from .test_testing_effects import refund

            instance = refund
        elif name == "Tool":
            from qagents import tool

            def sample():
                return None

            instance = tool(sample)
        assert field_name in annotations or field_name in fields or hasattr(instance, field_name)
        if default is None:
            continue
        field = fields.get(field_name)
        factory = None if field is None else field.default_factory
        if default.startswith("field(default_factory=") or default.startswith(
            "Field(default_factory="
        ):
            assert factory.__name__ == default.split("=", 1)[1].split(")", 1)[0]
        else:
            expected = getattr(value, default) if default in annotations else default_value(default)
            actual = (
                getattr(instance, field_name, dataclasses.MISSING)
                if field is None
                else field.default
            )
            if actual is dataclasses.MISSING or actual is PydanticUndefined:
                assert factory() == expected
            else:
                assert actual == expected


@pytest.mark.parametrize(
    "number,seam,owner", SEAMS, ids=[f"owner_{owner}:{number}" for number, _, owner in SEAMS]
)
def test_owner_mutant_seams(number, seam, owner):
    module_path = seam.replace("tests/effects/mutants/", f"{__package__}.mutants.")
    parts = module_path.split(".")
    for index in range(len(parts), 0, -1):
        try:
            value = resolve(".".join(parts[:index]), ".".join(parts[index:]))
        except (ModuleNotFoundError, AttributeError):
            continue
        assert callable(value)
        break
    else:
        pytest.fail(f"{number} seam missing: {seam}")


def test_stage1_mutant_seams_are_never_imported_by_name():
    names = {seam.rsplit(".", 1)[1] for _, seam, _ in SEAMS}
    failures = []
    for path in (ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                for imported in node.names:
                    if imported.name in names:
                        failures.append(f"{path.relative_to(ROOT)}:{node.lineno}:{imported.name}")
    assert failures == []


def test_stage1_foundation_imports_obey_layering():
    foundation = {"_canonical", "_types", "_kinds", "_declare", "_fold", "_ports", "contracts"}
    allowed = foundation | {"errors", "prompts", "effects"}
    for name in foundation:
        tree = ast.parse((ROOT / f"src/qagents/effects/{name}.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                assert (
                    node.module is None
                    or not node.module.startswith("qagents.")
                    or node.module.split(".")[1] in allowed
                )
                if node.level:
                    assert node.module is None or node.module.split(".")[0] in allowed
                    if node.module is None:
                        assert {item.name for item in node.names} <= foundation
            elif isinstance(node, ast.Import):
                assert not any(
                    item.name == "qagents"
                    or item.name.startswith("qagents.")
                    and item.name.split(".")[1] not in allowed
                    for item in node.names
                )


@pytest.mark.parametrize("module", ["_engine", "_table", "_quiescence", "_operator", "_claims"])
def test_stage1_runtime_modules_import_tools_and_agent_only_inside_functions(module):
    tree = ast.parse((ROOT / f"src/qagents/effects/{module}.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not any(item.name in {"qagents.tools", "qagents.agent"} for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "qagents":
                assert not {item.name for item in node.names} & {"tools", "agent"}
            assert node.module not in {"qagents.tools", "qagents.agent"}
            if node.level == 2:
                assert node.module not in {"tools", "agent"}
                if node.module is None:
                    assert not {item.name for item in node.names} & {"tools", "agent"}


@pytest.mark.parametrize(
    "module",
    [
        "qagents.effects",
        "qagents.testing.effects",
        "qagents.testing.stores",
        "qagents.testing.crash",
        "_qagents_pytest",
    ],
)
def test_stage1_fresh_imports_need_no_optional_extras(module):
    script = "import importlib,json,sys; importlib.import_module(sys.argv[1]); print(json.dumps(sorted(sys.modules)))"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-c", script, module], check=True, capture_output=True, text=True, env=env
    )
    loaded = json.loads(result.stdout)
    extras = {
        "mcp",
        "anthropic",
        "mlflow",
        "psycopg",
        "psycopg_pool",
        "opentelemetry",
        "jsonschema",
        "boto3",
    }
    assert not extras.intersection({name.split(".")[0] for name in loaded})
    if module == "_qagents_pytest":
        assert not any(name == "qagents" or name.startswith("qagents.") for name in loaded)
    if module == "qagents.effects":
        assert not any(name.startswith("qagents.testing") for name in loaded)


def test_stage1_new_fields_and_plugin_packaging_are_explicit():
    from qagents import RunContext, RunResult
    from qagents.session import Entry, PendingInput, SessionStatus
    from qagents.testing.crash import Faults, UnscriptedRequest

    first, second = Entry(kind="unknown"), Entry(kind="unknown")
    assert first.id and first.id != second.id
    assert Entry(kind="unknown", id="persisted").id == "persisted"
    assert RunContext().tool_attempt == 1 and not RunContext().is_rerun
    assert RunResult.model_fields["held_effects"].default_factory() == []
    assert PendingInput.model_fields["held"].default is None
    assert SessionStatus(session_id="s").lease_epoch == 0
    assert Faults.DEFAULT_KINDS == Faults.KINDS
    assert len(Faults.KINDS) == 38
    assert all(kinds <= Faults.KINDS for kinds in Faults.ALIASES.values())
    assert not UnscriptedRequest("missing").retryable
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["entry-points"]["pytest11"] == {"_qagents_pytest": "_qagents_pytest"}


def test_stage1_public_events_have_distinct_names_and_types():
    from qagents import events
    from qagents.effects import EffectRecord, HeldEffect, Settle

    assert len(events.__all__) == len(set(events.__all__))
    names = {
        events.EffectStarted.__name__,
        events.EffectFinished.__name__,
        events.EffectSettled.__name__,
        EffectRecord.__name__,
        HeldEffect.__name__,
        Settle.__name__,
    }
    assert len(names) == 6
