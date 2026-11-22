"""State snapshots, receipt validation and admission checks before an effect can send."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, field_serializer

from qagents.effects import _engine as engine
from qagents.effects._canonical import LIMIT_BYTES, jcs
from qagents.effects._declare import Key, Unique, effect
from qagents.effects._ports import Timing
from qagents.effects._types import MISSING, Attempt, EffectContext
from qagents.errors import ConfigurationError
from qagents.session import EFFECTS_REQUIRED, SessionStore


class Request(BaseModel):
    order: str
    amount: int = 1


class Receipt(BaseModel):
    id: str
    details: dict = {}


def declaration(**kwargs):
    @effect("orders.reserve", key=Key("order", "order"), **kwargs)
    async def reserve(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        raise AssertionError("Admission and pure helpers must never invoke send")

    return reserve


class RegistryStore(SessionStore):
    features = EFFECTS_REQUIRED

    def __init__(self, *, outcome="registered", features=None):
        self.outcome = outcome
        self.calls = []
        if features is not None:
            self.features = features

    async def register_effect(self, *args):
        self.calls.append(args)
        return self.outcome

    async def load(self, session_id):
        raise AssertionError("Admission must not load a journal")

    async def append(self, session_id, entries):
        raise AssertionError("Admission must not append to a journal")

    async def delete(self, session_id):
        raise AssertionError("Admission must not delete a journal")


def test_receipt_is_validated_and_snapshotted_independently():
    original = Receipt(id="receipt", details={"lines": [1]})
    result, error = engine.receipt_json(declaration(), original)
    assert error is None
    assert result == {"id": "receipt", "details": {"lines": [1]}}
    original.details["lines"].append(2)
    assert result["details"]["lines"] == [1]
    result["details"]["lines"].append(3)
    assert original.details["lines"] == [1, 2]


def test_receipt_dict_uses_schema_defaults_and_is_not_mutated():
    original = {"id": "r"}
    assert engine.receipt_json(declaration(), original) == ({"id": "r", "details": {}}, None)
    assert original == {"id": "r"}


@pytest.mark.parametrize("receipt", [{}, {"id": []}, Receipt.model_construct(id=[])])
def test_invalid_receipt_records_validation_error(receipt):
    result, error = engine.receipt_json(declaration(), receipt)
    assert result is None
    assert "ValidationError" in error


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 2**53, "\ud800"])
def test_noncanonical_receipt_values_are_rejected(value):
    result, error = engine.receipt_json(declaration(), Receipt(id="r", details={"x": value}))
    assert result is None and error


def test_receipt_canonical_byte_limit_is_inclusive_and_counts_utf8():
    blank = {"id": "", "details": {}}
    room = LIMIT_BYTES - len(jcs(blank))
    blank["id"] = "é" * (room // 2) + "x" * (room % 2)
    result, error = engine.receipt_json(declaration(), blank)
    assert error is None and len(jcs(result)) == LIMIT_BYTES
    blank["id"] += "x"
    result, error = engine.receipt_json(declaration(), blank)
    assert result is None and str(LIMIT_BYTES) in error


def test_receipt_custom_serializer_must_round_trip_through_its_schema():
    class BadReceipt(BaseModel):
        quantity: int

        @field_serializer("quantity")
        def incompatible(self, value):
            return "not a number"

    result, error = engine.receipt_json(
        SimpleNamespace(name="bad", receipt_type=BadReceipt), BadReceipt(quantity=3)
    )
    assert result is None and "ValidationError" in error


def test_no_on_commit_means_no_patch():
    assert engine.on_commit_patch(declaration(), {}, {}, {}) == (None, None)


def test_on_commit_uses_typed_copies_and_returns_a_detached_nested_patch():
    eff = declaration()
    captured = []

    @eff.on_commit
    def update(state, request, receipt):
        assert isinstance(request, Request) and isinstance(receipt, Receipt)
        state["orders"].append(request.order)
        state["receipt"] = receipt.id
        del state["obsolete"]
        request.order = "mutated"
        receipt.details["tags"].append("local")
        captured.append(state)

    before = {"orders": ["old"], "obsolete": True, "same": 7}
    request = {"order": "new", "amount": 1}
    receipt = {"id": "r", "details": {"tags": []}}
    patch, error = engine.on_commit_patch(eff, before, request, receipt)
    assert error is None
    assert patch.model_dump() == {
        "set": {"orders": ["old", "new"], "receipt": "r"},
        "unset": ["obsolete"],
    }
    assert before == {"orders": ["old"], "obsolete": True, "same": 7}
    assert request["order"] == "new" and receipt["details"]["tags"] == []
    captured[0]["orders"].append("late mutation")
    assert patch.set["orders"] == ["old", "new"]


def test_on_commit_diffs_json_types_including_nested_bool_and_number():
    eff = declaration()

    @eff.on_commit
    def update(state, request, receipt):
        state.update(value=True, nested={"value": False}, equivalent=1.0)

    patch, error = engine.on_commit_patch(
        eff, {"value": 1, "nested": {"value": 0}, "equivalent": 1}, {"order": "o"}, {"id": "r"}
    )
    assert error is None
    assert patch.set == {"value": True, "nested": {"value": False}}


@pytest.mark.parametrize("failure", ["raise", "non_json", "bad_key", "awaitable"])
def test_on_commit_failure_preserves_prior_state_without_partial_patch(failure):
    eff = declaration()
    before = {"items": [1]}

    async def unfinished():
        raise AssertionError("must not execute")

    @eff.on_commit
    def update(state, request, receipt):
        state["items"].append(2)
        if failure == "raise":
            raise RuntimeError("failed after mutation")
        if failure == "non_json":
            state["bad"] = float("nan")
        if failure == "bad_key":
            state[1] = "bad key"
        if failure == "awaitable":
            return unfinished()

    patch, error = engine.on_commit_patch(eff, before, {"order": "o"}, {"id": "r"})
    assert patch is None and error
    assert before == {"items": [1]}


def test_on_commit_receives_prior_committed_patch():
    eff = declaration()

    @eff.on_commit
    def update(state, request, receipt):
        state["count"] = state.get("count", 0) + 1

    one, error = engine.on_commit_patch(eff, {}, {"order": "o"}, {"id": "r"})
    two, second_error = engine.on_commit_patch(eff, one.set, {"order": "o"}, {"id": "r"})
    assert error is second_error is None
    assert one.set == {"count": 1} and two.set == {"count": 2}


def test_pure_helper_does_not_swallow_process_control():
    eff = declaration()

    @eff.on_commit
    def cancelled(state, request, receipt):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        engine.on_commit_patch(eff, {}, {"order": "o"}, {"id": "r"})


def test_declaration_checks_lease_and_tool_bounds_and_sequential_warning():
    eff = declaration(timeout_s=20)
    timing = Timing(30, 3, 1, rho=0)
    with pytest.raises(ConfigurationError, match="lease_ttl_s"):
        engine.check_declarations([eff], timing=Timing(25, 3, 1), tool_timeout_s=None)
    with pytest.raises(ConfigurationError, match="tool timeout_s"):
        engine.check_declarations([eff], timing=timing, tool_timeout_s=48.999)
    assert engine.check_declarations([eff], timing=timing, tool_timeout_s=49) == []
    assert len(engine.check_declarations([eff, eff], timing=timing, tool_timeout_s=49)) == 1
    assert engine.check_declarations([eff, eff], timing=timing, tool_timeout_s=98) == []


def test_unique_lookup_is_checked_after_decorators_have_attached():
    eff = declaration(dedupe=Unique("unique order", (KeyError,)))
    with pytest.raises(ConfigurationError, match="Unique needs a lookup"):
        engine.check_declarations([eff], timing=Timing(30, 3, 1), tool_timeout_s=None)
    eff.lookup(lambda ctx, inquiry: None)
    assert engine.check_declarations([eff], timing=Timing(30, 3, 1), tool_timeout_s=None) == []


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "20"])
def test_invalid_tool_timeout_fails_at_build_time(timeout):
    with pytest.raises(ConfigurationError, match="tool timeout_s"):
        engine.check_declarations([declaration()], timing=Timing(30, 3, 1), tool_timeout_s=timeout)


async def test_ordinary_run_does_not_require_effect_features():
    assert engine.check_declarations([], timing=Timing(0.1, 0.01, 1), tool_timeout_s=0.01) == []
    await engine.check_run_start(
        [], store=RegistryStore(features=frozenset()), deps=MISSING, session_id="s"
    )


@pytest.mark.parametrize("missing", sorted(EFFECTS_REQUIRED))
async def test_each_required_store_feature_is_checked_before_registry(missing):
    store = RegistryStore(features=EFFECTS_REQUIRED - {missing})
    with pytest.raises(ConfigurationError, match=missing):
        await engine.check_run_start([declaration()], store=store, deps=None, session_id="s")
    assert store.calls == []


@pytest.mark.parametrize("outcome", ["registered", "matches"])
async def test_run_start_checks_exact_registry_binding(outcome):
    store, eff = RegistryStore(outcome=outcome), declaration()
    await engine.check_run_start([eff], store=store, deps=MISSING, session_id="s")
    assert store.calls == [(eff.name, eff.version, eff.key_spec_digest, eff.request_schema_digest)]


async def test_changed_key_fails_before_the_run_sends():
    with pytest.raises(ConfigurationError, match="orders.reserve v1 changed its key; bump version"):
        await engine.check_run_start(
            [declaration()], store=RegistryStore(outcome="key_changed"), deps=None, session_id="s"
        )


async def test_changed_schema_warns_without_implying_key_change(caplog):
    await engine.check_run_start(
        [declaration()], store=RegistryStore(outcome="schema_changed"), deps=None, session_id="s"
    )
    assert "request schema" in caplog.text and "bump version" in caplog.text


async def test_invalid_registry_result_is_not_accepted():
    with pytest.raises(ConfigurationError, match="invalid effect registry result"):
        await engine.check_run_start(
            [declaration()], store=RegistryStore(outcome=None), deps=None, session_id="s"
        )


@pytest.mark.parametrize("namespace", ["tenant:42", "session:forged", lambda deps: "session:s"])
async def test_every_custom_namespace_requires_claims_even_with_session_prefix(namespace):
    store, eff = RegistryStore(), declaration(namespace=namespace)
    with pytest.raises(ConfigurationError, match="claims"):
        await engine.check_run_start([eff], store=store, deps=None, session_id="s")
    assert store.calls == []
    with pytest.raises(ConfigurationError, match="claims"):
        engine._require_claims(eff, "session:s", store)


async def test_namespace_resolves_with_explicit_none_and_claims_are_optional_for_session():
    calls = []

    def namespace(deps):
        calls.append(deps)
        return "tenant:public" if deps is None else f"tenant:{deps}"

    store = RegistryStore(features=EFFECTS_REQUIRED | {"claims"})
    await engine.check_run_start(
        [declaration(namespace=namespace)], store=store, deps=None, session_id="s"
    )
    assert calls == [None] and len(store.calls) == 1
    engine._require_claims(declaration(), "session:s", RegistryStore())


async def test_missing_deps_are_not_passed_to_a_namespace_function():
    def namespace(deps):
        pytest.fail("missing dependency sentinel must not reach user callback")

    with pytest.raises(ConfigurationError, match="deps="):
        await engine.check_run_start(
            [declaration(namespace=namespace)], store=RegistryStore(), deps=MISSING, session_id="s"
        )


@pytest.mark.parametrize("resolved", [None, "", "\ud800", 42])
async def test_invalid_resolved_namespace_fails_without_registry_write(resolved):
    store = RegistryStore(features=EFFECTS_REQUIRED | {"claims"})
    with pytest.raises(ConfigurationError, match="namespace"):
        await engine.check_run_start(
            [declaration(), declaration(namespace=lambda deps: resolved)],
            store=store,
            deps=None,
            session_id="s",
        )
    assert store.calls == []


def test_patch_round_trips_in_the_commit_json():
    eff = declaration()
    eff.on_commit(lambda state, request, receipt: state.update(receipts=[receipt.id]))
    patch, error = engine.on_commit_patch(eff, {}, {"order": "o"}, {"id": "r"})
    assert error is None
    assert json.loads(jcs(patch.model_dump())) == {"set": {"receipts": ["r"]}, "unset": []}
