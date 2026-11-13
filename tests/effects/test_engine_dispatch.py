"""Actual dispatch records its intent before user code and its outcome before returning."""

import asyncio
from dataclasses import replace

import pytest
from pydantic import BaseModel

from qagents.effects import Attempt, EffectContext, Key, current_attempt, effect
from qagents.effects import _engine as engine
from qagents.effects import _fold as fold
from qagents.effects import _kinds as kinds
from qagents.effects import _ports as ports
from qagents.effects._ports import Authorization, CallScope, Timing
from qagents.errors import EffectRejected, LeaseError, ToolError

from ._support import MemoryJournal


class Request(BaseModel):
    id: str
    amount: int = 1


class Receipt(BaseModel):
    id: str


def host(journal=None):
    journal = journal or MemoryJournal()
    state, emitted = {}, []

    async def memo(name, value):
        raise AssertionError("dispatch must not call memo")

    def patch(data):
        state.update(data.set)
        for key in data.unset:
            state.pop(key, None)

    scope = CallScope(
        "session",
        "run",
        "tool",
        "call",
        "run:1:call",
        1,
        frozenset({"refund"}),
        Authorization("allow"),
        Authorization("allow"),
        "args",
        "args",
        None,
        None,
        emitted.append,
        journal,
        None,
        {},
        memo,
        lambda: state,
        patch,
    )
    return scope, state, emitted


def declared(fn, timeout=0.05):
    return effect("refund", key=Key("request", "id"), timeout_s=timeout)(fn)


async def dispatch(send, scope, request=None, *, rule="D1"):
    request = request or Request(id="R1")
    prepared = engine._prepare(send, request, scope, None)
    op = scope.journal.view().get(prepared.namespace, send.name, prepared.key)
    if op is None:
        op = fold._new_operation(
            kinds.EffectRefData(
                namespace=prepared.namespace,
                name=send.name,
                key=prepared.key,
                record_id=engine.canonical.record_id(prepared.namespace, send.name, prepared.key),
            )
        )
    token = engine._DISPATCH_INPUT.set(prepared)
    try:
        return await engine._shield(
            engine._dispatch(send, op, prepared.frozen, scope, rule, op.generation), scope=scope
        )
    finally:
        engine._DISPATCH_INPUT.reset(token)


async def test_e1_intent_returns_before_send_and_commit_before_return():
    scope, state, emitted = host()

    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        scope.journal.assert_send_ready()
        assert current_attempt() is attempt and ports.attempt_live(attempt)
        assert attempt.request.id == "R1" and attempt.key == "request:R1"
        assert attempt.idempotency_key.startswith("q1_")
        assert attempt.not_after == pytest.approx(100 + 0.05 + 6)
        scope.journal.now = 101.25
        return Receipt(id="receipt")

    send = declared(send)

    @send.on_commit
    def committed(values, request, receipt):
        values["receipt"] = receipt.id

    outcome = await dispatch(send, scope)
    assert outcome.kind == "committed" and outcome.receipt == {"id": "receipt"}
    assert [item.kind for item in scope.journal.entries] == ["effect_intent", "effect_committed"]
    assert scope.journal.entries[-1].data["ended_at"] == 101.25
    assert scope.committed == [("refund", "request:R1")]
    assert state == {"receipt": "receipt"}
    assert emitted[0].type == "effect_started"


async def test_request_mutation_while_intent_is_writing_does_not_reach_sender():
    scope, _, _ = host()
    request = Request(id="R1", amount=2)

    async def before(entries):
        if entries[0].kind == "effect_intent":
            request.amount = 99

    scope.journal.before_append = before

    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        assert attempt.request.amount == 2
        return Receipt(id="receipt")

    assert (await dispatch(declared(send), scope, request)).kind == "committed"
    assert scope.journal.entries[0].data["request"]["amount"] == 2


async def test_e1_commit_unwritable_logged(caplog):
    scope, _, _ = host()

    async def before(entries):
        if entries[0].kind == "effect_committed":
            raise OSError("disk unavailable")

    scope.journal.before_append = before

    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        return Receipt(id="important-receipt")

    with pytest.raises(OSError, match="disk unavailable"):
        await dispatch(declared(send), scope)
    assert "important-receipt" in caplog.text
    assert "could not be journaled" in caplog.text
    assert scope.committed == []


async def test_e2_invalid_receipt_still_records_commit():
    scope, _, _ = host()

    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        return {"missing": "id"}

    outcome = await dispatch(declared(send), scope)
    assert outcome.kind == "committed" and outcome.receipt_error
    assert scope.journal.entries[-1].kind == "effect_committed"
    assert scope.journal.entries[-1].data["receipt"] is None
    assert scope.journal.view().records("session")[0].status == "committed"


@pytest.mark.parametrize(
    "error,kind",
    [
        (EffectRejected("no"), "rejected"),
        (ToolError("lost"), "held"),
        (RuntimeError("network"), "held"),
    ],
)
async def test_e3_e4_classify_only_declared_rejections(error, kind):
    scope, _, _ = host()

    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        raise error

    outcome = await dispatch(declared(send), scope)
    assert outcome.kind == kind and outcome.error is error
    assert scope.journal.entries[-1].data["committed"] is (False if kind == "rejected" else None)
    assert bool(scope.rejected) == (kind == "rejected")


async def test_f1_failed_first_fence_writes_and_sends_nothing(monkeypatch):
    scope, _, _ = host()

    async def refuse(*args):
        return False

    monkeypatch.setattr(engine, "_self_fence", refuse)

    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        raise AssertionError("must not send")

    with pytest.raises(LeaseError):
        await dispatch(declared(send), scope)
    assert scope.journal.entries == []


async def test_e7_failed_second_fence_records_not_sent(monkeypatch):
    scope, _, _ = host()
    checks = []

    async def fence(*args):
        checks.append(True)
        return len(checks) == 1

    monkeypatch.setattr(engine, "_self_fence", fence)

    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        raise AssertionError("must not send")

    outcome = await dispatch(declared(send), scope)
    assert outcome.kind == "not_sent"
    assert scope.journal.entries[-1].data["why"] == "not_sent"
    assert scope.journal.view().records("session") == []
    assert len(checks) == 2


async def test_e5_async_timeout_cancels_and_records_actual_end():
    scope, _, _ = host()
    ended = []

    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        try:
            await asyncio.Event().wait()
        finally:
            ended.append(True)

    outcome = await dispatch(declared(send, timeout=0.01), scope)
    assert outcome.kind == "held" and outcome.error_kind == "timeout"
    assert ended == [True]
    assert scope.journal.entries[-1].data["send_ended"]


async def test_e5o_resistant_send_is_tracked_until_late_commit_and_end():
    scope, state, _ = host()
    scope.journal.timing = Timing(60, 0.01, 1)
    release = asyncio.Event()

    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return Receipt(id="late")

    outcome = await dispatch(declared(send, timeout=0.01), scope)
    assert outcome.kind == "held"
    assert scope.journal.entries[-1].data["overran"]
    assert not scope.journal.entries[-1].data["send_ended"]
    assert scope.journal.lease.running_sends() > 0
    release.set()
    await scope.journal.lease.sends_drained()
    assert [item.kind for item in scope.journal.entries][-2:] == [
        "effect_committed",
        "effect_send_ended",
    ]
    assert scope.journal.view().records("session")[0].status == "committed"
    assert scope.journal.view().records("session")[0].attempts[0].overran


async def test_e6_caller_cancellation_waits_for_commit():
    scope, _, _ = host()
    started, release = asyncio.Event(), asyncio.Event()

    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        started.set()
        await release.wait()
        return Receipt(id="done")

    task = asyncio.create_task(dispatch(declared(send, timeout=1), scope))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert scope.committed == [("refund", "request:R1")]
    assert scope.journal.entries[-1].kind == "effect_committed"


async def test_approval_expiring_during_intent_append_prevents_send():
    scope, _, _ = host()
    scope.authorization = replace(scope.authorization, expires_at=100.5)

    async def after(entries):
        if entries[0].kind == "effect_intent":
            scope.journal.now = 101

    scope.journal.after_append = after

    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        raise AssertionError("expired decision must not send")

    outcome = await dispatch(declared(send), scope)
    assert outcome.kind == "not_sent" and outcome.reason == "APPROVAL_EXPIRED"


async def test_transport_delayed_past_deadline_does_not_enter_send_callback():
    scope, _, _ = host()

    class Transport:
        async def send(self, effect, ctx, attempt, call_send):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                return await call_send()

    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        raise AssertionError("a delayed callback cannot start after its deadline")

    token = ports.TRANSPORT.set(Transport())
    try:
        outcome = await dispatch(declared(send, timeout=0.01), scope)
    finally:
        ports.TRANSPORT.reset(token)
    assert outcome.kind == "not_sent"
    assert scope.journal.entries[-1].data["why"] == "not_sent"


@pytest.mark.parametrize("returned_id,kind", [("looked-up", "committed"), ("different", "held")])
async def test_commit_observed_during_send_is_not_patched_again(returned_id, kind):
    from qagents.effects._canonical import identity_digest
    from qagents.session import Entry

    scope, state, _ = host()
    patches = []

    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        recorded = scope.journal.entries[0]
        ref = {name: recorded.data[name] for name in ("namespace", "name", "key", "record_id")}
        receipt = {"id": "looked-up"}
        await scope.journal.append_effect(
            [
                Entry(
                    kind="effect_committed",
                    data=kinds.EffectCommittedData(
                        **ref,
                        attempt=1,
                        intent_seq=recorded.seq,
                        via="lookup",
                        receipt=receipt,
                        identity_digest=identity_digest(receipt, None),
                    ).model_dump(mode="json"),
                )
            ]
        )
        state["count"] = 1  # patch already applied by the lookup writer
        return Receipt(id=returned_id)

    send = declared(send)

    @send.on_commit
    def patch(values, request, receipt):
        patches.append(receipt.id)
        values["count"] = values.get("count", 0) + 1

    outcome = await dispatch(send, scope)
    assert outcome.kind == kind
    assert patches == [] and state == {"count": 1}
