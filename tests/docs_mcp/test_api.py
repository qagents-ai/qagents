"""get_api: live introspection of the installed qagents."""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

import qagents
from qagents.docs_mcp.api import (
    MODEL_FIELDS,
    call_signature,
    forwarded_parameters,
    forwarding_target,
    model_init,
)


def test_a_class(service):
    text = service.get_api("Agent")
    assert text.startswith("# qagents.Agent (class)")
    assert "Source: `src/qagents/agent.py:" in text
    assert "Import: `from qagents import Agent`" in text
    for parameter in inspect.signature(qagents.Agent).parameters:
        assert f"    {parameter}:" in text  # every constructor parameter, one per line
    assert "- `async run(" in text and "- `async resume(" in text
    assert "See also (guides and examples):" in text


@pytest.mark.parametrize("symbol", ["qagents.Agent.run", "Agent.run", "agent.run", "`Agent.run()`"])
def test_a_method_however_it_is_written(service, symbol):
    text = service.get_api(symbol)
    assert text.startswith("# qagents.Agent.run (method)")
    assert "async def run(" in text and "session: Session | None = None" in text
    assert "call it with `await`" in text


def test_kwargs_that_are_passed_on_are_named(service):
    text = service.get_api("Agent.run_sync")
    assert "are passed to `Agent.run`" in text


@pytest.mark.parametrize("name", ["ScriptedExternalAgent", "DevviAgent", "ACPAgent"])
def test_constructor_options_passed_to_the_base_class_are_named(service, name):
    """``super().__init__(..., **options)`` is followed to `ExternalAgent`'s constructor."""
    text = service.get_api(name)
    assert re.search(r"are passed to `qagents\.(external\.)?ExternalAgent`", text), text[:600]


@pytest.mark.parametrize(
    "name", ["ExternalAgent", "ExternalTurnContext", "ExternalOutcome", "ExternalCapabilities"]
)
def test_a_class_imported_by_an_integration_keeps_its_own_public_name(service, name):
    """`qagents.integrations.devvi` imports these for itself; they are not its re-exports."""
    text = service.get_api(name)
    heading, _, rest = text.partition("\n")
    # The shortest name it is exported by: qagents' own (lazy) name, else qagents.external's.
    public = f"qagents.{name}" if name in qagents._LAZY else f"qagents.external.{name}"
    assert heading == f"# {public} (class)", heading
    assert "Source: `src/qagents/external/" in rest  # the package's modules define it
    assert "integrations.devvi" not in heading and "from qagents.integrations" not in rest
    listing = service.search(name, limit=5, kind="api")
    assert f"integrations.devvi.{name}" not in listing
    assert f'get_api("{public}")' in listing


def test_public_names_follow_chained_re_exports():
    """``qagents.external`` re-exports its modules' classes and ``qagents`` re-exports those
    names lazily: the public name is the last link of the chain."""
    from qagents.docs_mcp.apiindex import _public_aliases, _public_name

    aliases = _public_aliases(Path(qagents.__file__).parent)
    assert aliases["qagents.external.permissions.ExternalApproval"] == (
        "qagents.external.ExternalApproval"
    )
    assert _public_name("qagents.external.permissions.ExternalApproval", aliases) == (
        "qagents.ExternalApproval"
    )
    assert _public_name("qagents.external.agent.ExternalAgent.capabilities", aliases) == (
        "qagents.ExternalAgent.capabilities"
    )
    assert _public_name("a.b.c", {"a.b.c": "a.b", "a.b": "a.c", "a.c": "a.b"}) in ("a.b", "a.c")


class _Base:
    __module__ = "qagents.fixture"  # forwarding is only followed into qagents classes

    def __init__(self, *, name: str, grace_s: float = 1.0) -> None:
        self.name = name


class _Middle(_Base):
    __module__ = "qagents.fixture"

    def __init__(self, *, mode: str = "a", **options: Any) -> None:
        super().__init__(**options)


class _Leaf(_Middle):
    __module__ = "qagents.fixture"

    def __init__(self, turns: list[str], **options: Any) -> None:
        super(_Leaf, self).__init__(**options)  # noqa: UP008 - the two-argument form, on purpose


class _ToObject:
    __module__ = "qagents.fixture"

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)


def test_super_chains_are_followed_to_the_end():
    assert forwarding_target(_Leaf, None) is _Middle.__init__
    assert forwarding_target(_Middle, None) is _Base.__init__
    assert {p.name for p in forwarded_parameters(_Leaf, None)} == {"mode", "name", "grace_s"}
    assert forwarded_parameters(_Base, None) is None  # no **kwargs at all
    assert forwarded_parameters(_ToObject, None) is None  # object.__init__ is not followed


class _Fields(BaseModel):
    __module__ = "qagents.fixture"

    tool_call_id: str
    content: str
    count: int = 0


class _FillsIn(_Fields):
    """A model whose own ``__init__`` fills in required fields, as ExternalApproval does."""

    __module__ = "qagents.fixture"

    def __init__(self, approved: bool = True, *, reason: str | None = None, **data: Any) -> None:
        data.setdefault("tool_call_id", "")
        data.setdefault("content", reason or ("yes" if approved else "no"))
        super().__init__(**data)


class _AnyKeyword(_FillsIn):
    __module__ = "qagents.fixture"
    model_config = ConfigDict(extra="allow")


def test_a_model_with_its_own_init_is_described_by_that_init():
    # pydantic's merged signature calls the fields the __init__ fills in required:
    pydantic = inspect.signature(_FillsIn).parameters
    assert pydantic["tool_call_id"].default is inspect.Parameter.empty
    assert _FillsIn().content == "yes"  # but they are not
    assert list(call_signature(_FillsIn).parameters) == ["approved", "reason", "data"]
    assert call_signature(_Fields) == inspect.signature(_Fields)  # no own __init__: pydantic's
    assert model_init(_Fields) is None and model_init(_FillsIn) is _FillsIn.__init__
    assert forwarding_target(_FillsIn, None) is MODEL_FIELDS
    keywords = [p.name for p in forwarded_parameters(_FillsIn, None)]
    assert keywords == ["tool_call_id", "content", "count"]
    assert forwarded_parameters(_AnyKeyword, None) is None  # extra="allow": any keyword


def test_external_approval_shows_its_own_constructor(service):
    text = service.get_api("ExternalApproval")
    signature = text.split("```python\n", 1)[1].split("\n", 1)[0]
    assert signature == (
        "class ExternalApproval(approved: bool = True, *, reason: str | None = None, **data: Any)"
    )
    assert "set the fields of `qagents.ExternalApproval` by name" in text  # the public name
    assert "- `tool_call_id: str` (from `ToolResultPart`)" in text  # the fields, as before


def test_a_module(service):
    text = service.get_api("qagents.toolkits.files")
    assert text.startswith("# qagents.toolkits.files (module)")
    assert "## Members" in text and "`file_tools(" in text


def test_names_found_in_submodules(service):
    text = service.get_api("ScriptedModel")
    assert text.startswith("# qagents.models.testing.ScriptedModel (class)")
    assert "Import: `from qagents.models.testing import ScriptedModel`" in text
    assert "requests: list[ModelRequest]" in text  # set in __init__, still listed
    assert "From the docstring of `qagents.models.testing`" in text  # the module's example


def test_permissions_and_its_rules(service):
    text = service.get_api("Permissions")
    assert text.startswith("# qagents.Permissions (class)")
    assert "on_ask" in text and "can_use_tool" in text


def test_pydantic_models_show_fields_not_pydantic_docs(service):
    text = service.get_api("RunResult")
    assert "## Fields" in text and "- `pending_approvals: list[PendingApproval]" in text
    assert "model_post_init" not in text
    message = service.get_api("Message")
    assert (
        "__pydantic" not in message and "A base class for creating Pydantic models" not in message
    )
    assert "parts: list[TextPart | ImagePart" in message and "FieldInfo(" not in message
    assert "- `tool_results -> list[ToolResultPart]`" in message


def test_fields_and_attributes(service):
    field = service.get_api("RunResult.pending_approvals")
    assert field.startswith("# qagents.RunResult.pending_approvals (field)")
    assert "pending_approvals: list[PendingApproval]" in field
    constant = service.get_api("qagents.prompts.OUTPUT_RETRY")
    assert "(attribute)" in constant and "OUTPUT_RETRY = " in constant


def test_a_bare_name_prefers_the_module_level_object(service):
    text = service.get_api("call")
    assert text.startswith("# qagents.models.testing.call (function)")
    assert "Other objects with this name: `qagents.events.ToolCallFinished.call`" in text


def test_ambiguous_names_list_the_candidates(service):
    text = service.get_api("resume")
    assert text.startswith("Several qagents objects are called 'resume'")
    assert "- `qagents.Agent.resume`" in text


def test_unknown_names_are_refused_with_suggestions(service):
    text = service.get_api("Agnet")
    assert "has no public name 'Agnet'" in text and "`qagents.Agent`" in text
    assert "Do not use it" in text
    assert "has no public name" in service.get_api("qagents.Agent.final_output")


def test_lazy_extras(service):
    text = service.get_api("MCPToolkit")  # the mcp extra is installed in the dev environment
    assert text.startswith("# qagents.MCPToolkit (class)")
    assert "pip install 'qagents[mcp]'" in text


def hide_the_mcp_sdk(monkeypatch):
    """Make ``import mcp_types`` fail, as it does without the qagents[mcp] extra."""
    qagents.MCPToolkit  # noqa: B018 - resolve the lazy name, so monkeypatch can restore it
    monkeypatch.delattr(qagents, "MCPToolkit", raising=False)
    monkeypatch.delitem(sys.modules, "qagents.toolkits.mcp", raising=False)
    monkeypatch.setitem(sys.modules, "mcp_types", None)  # importing the SDK now fails


def test_a_missing_extra_falls_back_to_the_source(service, monkeypatch):
    hide_the_mcp_sdk(monkeypatch)
    text = service.get_api("MCPToolkit")
    assert text.startswith("# qagents.MCPToolkit (class)")
    assert "is not installed), so this comes from the source" in text
    assert "pip install 'qagents[mcp]'" in text
    assert "Source: `src/qagents/toolkits/mcp.py:" in text


EXTERNAL_CONSTANTS = {
    "APPROVAL_KEY": ("permissions.py", "``ToolResultPart.metadata`` key of an `ExternalApproval`"),
    "CHECKPOINT_KIND": ("contract.py", "Journal entry kind for facts a tool call records"),
    "ExternalStatus": ("contract.py", "How an external turn ended."),
    "PermissionPosture": ("permissions.py", "What a permission request that needs a human gets"),
    "STATE_GRANTS": ("tool.py", "``ctx.state`` key: permission requests that paused the run"),
}


@pytest.mark.parametrize("name", sorted(EXTERNAL_CONSTANTS))
def test_a_re_exported_constant_shows_its_attribute_docstring(service, name):
    """A str or a ``Literal`` has no ``__module__``: its entry is found in the module that
    defines it, which holds the same object."""
    module, doc = EXTERNAL_CONSTANTS[name]
    text = service.get_api(f"qagents.external.{name}")
    assert text.startswith(f"# qagents.external.{name} (attribute)"), text[:200]
    assert f"Source: `src/qagents/external/{module}:" in text
    assert doc in text and "No docstring." not in text


def test_a_module_lists_its_constants_with_their_docstrings(service):
    text = service.get_api("qagents.external")
    for name, (_, doc) in EXTERNAL_CONSTANTS.items():
        line = next(line for line in text.splitlines() if line.startswith(f"- `{name}` "))
        assert line.startswith(f"- `{name}` (attribute): {doc}"), line


def test_a_prompt_constant_shows_its_docstring(service):
    text = service.get_api("qagents.prompts.OUTPUT_TOOL_ACCEPTED")
    assert "OUTPUT_TOOL_ACCEPTED = 'Final answer accepted.'" in text
    assert "Result of a final-answer call that passed validation." in text
    assert "Source: `src/qagents/prompts.py:" in text and "No docstring." not in text


def test_a_constant_is_not_given_the_docstring_of_another_object(service):
    """A re-exported name counts only when the defining module holds the very same object."""
    import qagents.external.permissions as permissions

    lookup = service.lookup
    external = sys.modules["qagents.external"]
    assert lookup.module_attribute_entry(external, "APPROVAL_KEY", "another value") is None
    found = lookup.module_attribute_entry(external, "APPROVAL_KEY", permissions.APPROVAL_KEY)
    assert found is not None and found.path == "qagents.external.permissions.APPROVAL_KEY"
