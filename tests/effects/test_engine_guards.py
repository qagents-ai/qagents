"""Boundary checks for the engine's sending conditions and destination contracts."""

from collections import Counter
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from qagents.effects import Attempt, Dedupe, EffectContext, Key, Unique, effect
from qagents.effects import _canonical as canonical
from qagents.effects import _engine as engine
from qagents.effects._ports import Authorization, Timing
from qagents.effects._types import HoldReason
from qagents.errors import ConfigurationError, EffectRejected, ToolError

from ._support import MemoryJournal


class Request(BaseModel):
    id: str
    values: list[int] = []


class Receipt(BaseModel):
    id: str


def declared(**options):
    @effect("send", key=options.pop("key", Key.per_call()), timeout_s=2, **options)
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        return Receipt(id=attempt.request.id)

    return send


def preparation_scope(**changes):
    values = dict(
        effects=frozenset({"send"}),
        composition_refusal=None,
        session_id="session",
        journal=MemoryJournal(),
        key_override=None,
        tool_name="tool",
        occurrences=Counter(),
        deps=None,
        operation_id="run:1:call",
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "changes",
    [
        {"effects": frozenset()},
        {"composition_refusal": "in-memory child is not fenced"},
        {"session_id": "different"},
        {"key_override": "tool-cannot-override"},
    ],
)
async def test_p0_refuses_invalid_scope_before_journal_write(changes):
    scope = preparation_scope(**changes)
    with pytest.raises(ConfigurationError):
        await engine.perform(declared(), Request(id="R1"), scope=scope)
    assert scope.journal.entries == []


def test_request_and_destination_are_detached_before_the_first_await():
    deps = {"account": ["one"]}
    scope = preparation_scope(deps=deps)
    send = declared(key=Key("request", "id"), destination=lambda deps: deps)
    request = Request(id="R1", values=[1])
    prepared = engine._prepare(send, request, scope, None)
    request.values.append(2)
    deps["account"].append("two")
    assert prepared.frozen.json == {"id": "R1", "values": [1]}
    assert prepared.destination == {"account": ["one"]}
    assert prepared.namespace == "session:session" and prepared.key == "request:R1"
    assert scope.journal.entries == []


def test_plain_dict_request_is_not_coerced_into_an_effect_request():
    scope = preparation_scope()
    with pytest.raises(TypeError, match="requires Request"):
        engine._prepare(declared(), {"id": "R1"}, scope, None)
    assert scope.journal.entries == []


def test_m12_second_per_call_invocation_requires_explicit_occurrence():
    scope = SimpleNamespace(occurrences=Counter())
    send = declared()
    assert engine._occurrence_for(scope, send, None) is None
    with pytest.raises(ConfigurationError, match="occurrence"):
        engine._occurrence_for(scope, send, None)
    assert engine._occurrence_for(scope, send, 4) == 4
    assert scope.occurrences[send.name] == 2


@pytest.mark.parametrize("occurrence", [-1, True, 1.0, "one"])
def test_invalid_occurrence_is_not_counted(occurrence):
    scope = SimpleNamespace(occurrences=Counter())
    with pytest.raises(ConfigurationError):
        engine._occurrence_for(scope, declared(), occurrence)
    assert not scope.occurrences


def test_business_key_can_be_revisited_without_positional_identity():
    scope = SimpleNamespace(occurrences=Counter())
    send = declared(key=Key("business", "id"))
    assert engine._occurrence_for(scope, send, None) is None
    assert engine._occurrence_for(scope, send, None) is None


def test_m13_sender_receives_frozen_request_even_after_caller_mutation():
    send = declared()
    original = Request(id="R1", values=[1])
    frozen = canonical.freeze(send, original)
    original.values.append(2)
    rebuilt = engine._request_for_send(send, frozen, original)
    assert rebuilt is not original and rebuilt.values == [1]
    rebuilt.values.append(3)
    assert frozen.json["values"] == [1]


def test_m4_toolerror_is_unknown_and_warned_once(monkeypatch, caplog):
    monkeypatch.setattr(engine, "_TOOLERROR_WARNED", False)
    send = declared()
    assert engine._classify_send_error(send, ToolError("lost response")) == "unknown"
    assert engine._classify_send_error(send, ToolError("another")) == "unknown"
    assert len(caplog.records) == 1
    assert "unknown outcome" in caplog.records[0].message
    assert engine._classify_send_error(send, EffectRejected("no application")) == "rejected"


def test_destination_contract_classifies_rejection_and_unique_duplicate():
    assert (
        engine._classify_send_error(declared(rejected_on=(ValueError,)), ValueError()) == "rejected"
    )
    unique = declared(dedupe=Unique(contract="destination uniqueness", on_duplicate=(KeyError,)))
    assert engine._classify_send_error(unique, KeyError()) == "duplicate"
    assert engine._classify_send_error(unique, RuntimeError()) == "unknown"


@pytest.mark.parametrize("remaining,renewed", [(4.0, 0), (3.99, 1)])
async def test_m7_margin_boundary_renews_at_most_once(remaining, renewed):
    class Lease:
        calls = 0

        def __init__(self):
            self.seconds = remaining

        def lost(self):
            return False

        def remaining(self):
            return self.seconds

        async def renew(self):
            self.calls += 1
            self.seconds = 30.0
            return True

    lease = Lease()
    journal = SimpleNamespace(lease=lease, timing=Timing(30, 2, 1))
    assert await engine._self_fence(journal, declared())
    assert lease.calls == renewed


@pytest.mark.parametrize("result", [False, RuntimeError("offline"), "still_short", "lost"])
async def test_m7_failed_or_late_renewal_never_authorizes_send(result):
    class Lease:
        calls = 0

        def lost(self):
            return result == "lost"

        def remaining(self):
            return 3.0

        async def renew(self):
            self.calls += 1
            if isinstance(result, Exception):
                raise result
            return result is not False

    lease = Lease()
    journal = SimpleNamespace(lease=lease, timing=Timing(30, 2, 1))
    assert not await engine._self_fence(journal, declared())
    assert lease.calls == (0 if result == "lost" else 1)


def test_m17_automatic_resend_requires_original_proof_and_store_clock_expiry():
    scope = SimpleNamespace(
        authorization=Authorization("ask", expires_at=1000, verifiable=True),
        original_authorization=Authorization("ask", expires_at=100, verifiable=True),
    )
    assert engine._authorization_valid("D3", scope, 100) is None
    assert engine._authorization_valid("D3", scope, 100.001) == HoldReason.APPROVAL_EXPIRED
    assert engine._authorization_valid("D1", scope, 100.001) is None
    scope.original_authorization = Authorization("ask", verifiable=False)
    assert engine._authorization_valid("D4", scope, 0) == HoldReason.APPROVAL_UNVERIFIABLE
    scope.original_authorization = None
    assert engine._authorization_valid("D3", scope, 0) == HoldReason.APPROVAL_UNVERIFIABLE


def test_d5_requires_settlement_identity_and_d2_changes_only_replayed_rejection_key():
    scope = SimpleNamespace(authorization=Authorization("resend"))
    assert engine._authorization_valid("D5", scope, 0) == HoldReason.APPROVAL_UNVERIFIABLE
    scope.authorization = Authorization("resend", settlement_seq=12)
    assert engine._authorization_valid("D5", scope, 0) is None
    original = declared(dedupe=Dedupe(1000, "contract", replays_rejections=False))
    replayed = declared(dedupe=Dedupe(1000, "contract", replays_rejections=True))
    assert engine._d2_plan(original, SimpleNamespace(generation=2)) == (False, 2)
    assert engine._d2_plan(replayed, SimpleNamespace(generation=2)) == (False, 3)
