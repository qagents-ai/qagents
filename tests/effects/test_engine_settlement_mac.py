"""With an approval_secret, an operator resend sends (D5) only under a valid MAC (SPEC 5.3).

A settlement written straight to the store, without the operator path, must not
authorize a send: no MAC, a wrong MAC, a MAC over other fields or from another secret,
or a genuine answer copied to a later position. Each is held as APPROVAL_UNVERIFIABLE
until an operator answers again, also on a key no attempt used. The operator path checks a
new resend against the record without such an entry, so one cannot reopen a not_applied,
closed_unknown, rejected or unused record for a real resend. Without a secret the journal
is trusted, as before.
"""

from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from qagents import Agent, RunContext, Session, SQLiteStore, tool
from qagents.effects import Attempt, Dedupe, EffectContext, Key, Settle, effect
from qagents.effects import _canonical as canonical
from qagents.effects import _engine as engine
from qagents.effects import _kinds as kinds
from qagents.effects import _operator as operator
from qagents.effects._types import HoldReason
from qagents.errors import EffectHeld, EffectRejected, SettlementConflict
from qagents.models.testing import FunctionModel, call
from qagents.permissions import Permissions
from qagents.session import Entry, InMemoryStore, LeasedWriter

from .test_engine_dispatch import Receipt, Request, host

SECRET = "settlement-test-fixture-secret-32b"
OTHER_SECRET = "another-settlement-fixture-secret!"
SESSION = "session"


@pytest.fixture(params=("memory", "sqlite"))
async def store(request, tmp_path):
    now = [1000.0]
    value = (
        InMemoryStore(clock=lambda: now[0])
        if request.param == "memory"
        else SQLiteStore(tmp_path / "settle-mac.sqlite", clock=lambda: now[0])
    )
    async with value:
        yield value, now


def refund(calls, *, failures=1, dedupe=None):
    @effect("refund", key=Key("request", "id"), dedupe=dedupe)
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        calls.append(attempt.idempotency_key)
        if len(calls) <= failures:
            raise ConnectionError("ack lost")
        return Receipt(id="paid")

    return send


@asynccontextmanager
async def opened(db, permissions):
    writer = LeasedWriter(
        db, SESSION, worker="test", ttl_s=30, purpose="run", journal_acquisition=True
    )
    assert await writer.open()
    try:
        await writer.load()
        scope, _, _ = host(writer)
        yield replace(scope, session_id=SESSION, permissions=permissions)
    finally:
        await writer.close()


async def perform(db, send, permissions):
    async with opened(db, permissions) as scope:
        return await engine.perform(send, Request(id="R1"), scope=scope)


async def perform_held(db, send, permissions):
    with pytest.raises(EffectHeld) as held:
        await perform(db, send, permissions)
    return held.value.held


async def record(db):
    (only,) = await Session(db, id=SESSION).effects()
    return only


async def settle(db, send, permissions, answer):
    return await operator.settle(
        Session(db, id=SESSION),
        answer,
        effects={send.name: send},
        permissions=permissions,
        lease_ttl_s=30,
        worker="operator",
    )


async def append_straight(db, data):
    """A writer with store access that bypasses agent.settle and the operator checks."""
    assert await db.acquire_lease(SESSION, "forger", 30)
    try:
        await db.append_leased(SESSION, [Entry(kind="effect_settled", data=data)], "forger")
    finally:
        await db.release_lease(SESSION, "forger")


def resend_data(current, **changes):
    return {
        "namespace": current.namespace,
        "name": current.effect,
        "key": current.key,
        "record_id": current.id,
        "outcome": "resend",
        "by": "ops",
        "evidence": "destination audit",
        "reason": None,
        "expected_revision": current.revision,
        "mac": None,
        "covers_attempts": [attempt.number for attempt in current.attempts],
        **changes,
    }


def signed(secret, data, *, session_id=SESSION):
    fields = kinds.settlement_mac_fields(session_id, kinds.KIND_EFFECT_SETTLED, data)
    return Permissions(approval_secret=secret).sign_settlement(fields)


FORGERIES = {
    "no_mac": lambda data: None,
    "wrong_mac": lambda data: "0" * 64,
    "mac_of_other_outcome": lambda data: signed(SECRET, {**data, "outcome": "not_applied"}),
    "mac_of_other_session": lambda data: signed(SECRET, data, session_id="other-session"),
    "mac_from_other_secret": lambda data: signed(OTHER_SECRET, data),
}


def intents(entries):
    return [entry for entry in entries if entry.kind == "effect_intent"]


def assert_unverifiable(held):
    assert held.reason == HoldReason.APPROVAL_UNVERIFIABLE
    assert held.detail == engine._RESEND_UNVERIFIABLE
    assert held.retry_after is None


@pytest.mark.parametrize("forgery", sorted(FORGERIES))
async def test_forged_resend_is_held_and_never_sent(store, forgery):
    db, now = store
    calls = []
    send = refund(calls)
    permissions = Permissions(approval_secret=SECRET)
    await perform_held(db, send, permissions)
    now[0] += permissions.resend_min_age_s + 1
    data = resend_data(await record(db))
    await append_straight(db, {**data, "mac": FORGERIES[forgery](data)})
    assert_unverifiable(await perform_held(db, send, permissions))
    assert len(calls) == 1
    entries = await db.load(SESSION)
    assert [entry.data["rule"] for entry in intents(entries)] == ["D1"]
    assert entries[-1].kind == "effect_observed"
    assert entries[-1].data["reason"] == HoldReason.APPROVAL_UNVERIFIABLE
    current = await record(db)
    assert current.status == "held"
    assert_unverifiable(current.hold)
    # An operator answers again; that genuine resend sends once, under the same key.
    await settle(db, send, permissions, Settle.resend(current, by="ops", evidence="re-checked"))
    assert (await perform(db, send, permissions)).id == "paid"
    assert len(calls) == 2 and len(set(calls)) == 1
    assert [entry.data["rule"] for entry in intents(await db.load(SESSION))] == ["D1", "D5"]


async def test_genuine_operator_resend_is_verified_and_sent_once(store):
    db, now = store
    calls = []
    send = refund(calls)
    permissions = Permissions(approval_secret=SECRET)
    await perform_held(db, send, permissions)
    now[0] += permissions.resend_min_age_s + 1
    await settle(db, send, permissions, Settle.resend(await record(db), by="ops", evidence="audit"))
    assert (await perform(db, send, permissions)).id == "paid"
    entries = await db.load(SESSION)
    settled = next(entry for entry in entries if entry.kind == "effect_settled")
    (resend,) = intents(entries)[1:]
    assert resend.data["rule"] == "D5"
    assert resend.data["authorization"]["settlement_seq"] == settled.seq
    assert resend.data["authorization"]["verifiable"] is True
    assert len(calls) == 2 and len(set(calls)) == 1


async def test_genuine_resend_copied_after_its_attempt_is_not_used_again(store):
    db, now = store
    calls = []
    send = refund(calls, failures=2)
    permissions = Permissions(approval_secret=SECRET)
    await perform_held(db, send, permissions)
    now[0] += permissions.resend_min_age_s + 1
    await settle(db, send, permissions, Settle.resend(await record(db), by="ops", evidence="audit"))
    await perform_held(db, send, permissions)  # D5 sends, and its answer is lost again
    assert len(calls) == 2
    genuine = next(e for e in await db.load(SESSION) if e.kind == "effect_settled")
    assert Permissions(approval_secret=SECRET).verify_settlement(
        kinds.settlement_mac_fields(SESSION, genuine.kind, genuine.data), genuine.data["mac"]
    )
    await append_straight(db, dict(genuine.data))
    assert_unverifiable(await perform_held(db, send, permissions))
    assert len(calls) == 2


async def test_forged_resend_cannot_reopen_a_not_applied_record_for_a_dedupe_resend(store):
    db, now = store
    calls = []
    send = refund(
        calls,
        failures=2,
        dedupe=Dedupe(3600, contract="fixture", concurrent="waits", replays_errors=False),
    )
    permissions = Permissions(approval_secret=SECRET)
    await perform_held(db, send, permissions)  # D1 and one D3, both answers lost
    assert len(calls) == 2
    await settle(
        db, send, permissions, Settle.not_applied(await record(db), by="ops", evidence="ledger")
    )
    closed = await record(db)
    assert closed.status == "not_applied"
    await append_straight(db, resend_data(closed))
    assert (await record(db)).status == "held"  # the fold trusts the entry; the engine does not
    held = await perform_held(db, send, permissions)
    assert len(calls) == 2
    assert_unverifiable(held)


async def test_without_a_secret_a_journaled_resend_is_trusted_as_before(store):
    db, now = store
    calls = []
    send = refund(calls)
    permissions = Permissions()
    await perform_held(db, send, permissions)
    await append_straight(db, resend_data(await record(db)))
    assert (await perform(db, send, permissions)).id == "paid"
    assert [entry.data["rule"] for entry in intents(await db.load(SESSION))] == ["D1", "D5"]
    assert len(calls) == 2


async def test_unverifiable_resend_authorization_never_dispatches():
    scope = replace(
        host()[0],
        authorization=engine.ports.Authorization("resend", settlement_seq=7, verifiable=False),
    )
    assert engine._authorization_valid("D5", scope, 0) == HoldReason.APPROVAL_UNVERIFIABLE
    scope = replace(scope, authorization=engine.ports.Authorization("resend", settlement_seq=7))
    assert engine._authorization_valid("D5", scope, 0) is None


def refund_agent(destination, path, now):
    @effect("refund", key=Key("request", "id"))
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        destination.append(attempt.idempotency_key)
        if len(destination) == 1:
            raise ConnectionError("ack lost")
        return Receipt(id="paid")

    @tool(effects=[send])
    async def refund_order(ctx: RunContext, id: str) -> str:
        """Refund an order in the test destination."""
        return (await ctx.perform(send, Request(id=id))).id

    agent = Agent(
        FunctionModel(
            lambda request: (
                "Refunded."
                if request.messages[-1].role == "tool"
                else call("refund_order", id="R1")
            )
        ),
        tools=[refund_order],
        permissions=Permissions(approval_secret=SECRET),
    )
    return agent, SQLiteStore(path, clock=lambda: now[0])


async def test_agent_settle_resend_survives_a_crash_and_a_forgery_does_not(tmp_path):
    """Each block is a new process: a new store handle and agent fold the file again."""
    path, now, destination = tmp_path / "agent.sqlite", [1000.0], []
    agent, db = refund_agent(destination, path, now)
    async with db:
        paused = await agent.run("Refund R1", session=Session(db, id=SESSION))
        assert paused.stop_reason == "effect_held" and len(destination) == 1
        now[0] += 901
        await append_straight(db, resend_data(await record(db)))  # then the worker dies
    agent, db = refund_agent(destination, path, now)
    async with db:
        again = await agent.resume(Session(db, id=SESSION))
        assert again.status == "paused" and again.stop_reason == "effect_held"
        assert_unverifiable(again.held_effects[0])
        assert len(destination) == 1
    agent, db = refund_agent(destination, path, now)
    async with db:
        session = Session(db, id=SESSION)
        current = await record(db)
        assert_unverifiable(current.hold)  # the rebuilt fold still holds it
        await agent.settle(session, Settle.resend(current, by="ops", evidence="audit"))
    agent, db = refund_agent(destination, path, now)  # a crash between settle and resume
    async with db:
        done = await agent.resume(Session(db, id=SESSION))
        assert done.status == "completed" and done.output == "Refunded."
        rules = [entry.data["rule"] for entry in intents(await db.load(SESSION))]
    assert rules == ["D1", "D5"]
    assert len(destination) == 2 and len(set(destination)) == 1


class Crash(BaseException):
    """The worker dies at a journal append (not an Exception: nothing retries it)."""


def crash_at(db, number, phase):
    """Kill the worker at the number-th effect batch from now, before or after it commits."""
    append = db.append_leased
    seen = [0]

    async def append_leased(session_id, entries, owner):
        if any(entry.kind.startswith("effect_") for entry in entries):
            seen[0] += 1
            if seen[0] == number:
                if phase == "before":
                    raise Crash()
                await append(session_id, entries, owner)
                raise Crash()
        return await append(session_id, entries, owner)

    db.append_leased = append_leased
    return lambda: setattr(db, "append_leased", append)


# The effect batches after the first hold: the operator's effect_settled, then the D5 intent
# and its commit. A crash before or after each one; each recovery folds the journal again.
STEPS = [(number, phase) for number in (1, 2, 3) for phase in ("before", "after")]


@pytest.mark.parametrize("step", STEPS, ids=[f"{phase}-{number}" for number, phase in STEPS])
async def test_a_crash_at_every_step_of_a_signed_resend_recovers_from_the_journal(store, step):
    db, now = store
    calls = []
    send = refund(calls)
    permissions = Permissions(approval_secret=SECRET)
    await perform_held(db, send, permissions)
    now[0] += permissions.resend_min_age_s + 1
    restore = crash_at(db, *step)
    try:
        await settle(db, send, permissions, Settle.resend(await record(db), by="ops", evidence="a"))
        receipt = await perform(db, send, permissions)
    except Crash:
        receipt = None
    finally:
        restore()
    assert receipt is None  # the worker died at the step
    now[0] += 120  # the dead worker's lease expires
    entries = await db.load(SESSION)
    if not any(entry.kind == "effect_settled" for entry in entries):
        await settle(db, send, permissions, Settle.resend(await record(db), by="ops", evidence="b"))
    try:
        receipt = await perform(db, send, permissions)
    except EffectHeld as error:
        # The D5 intent is durable, so the resend is spent: the outcome is unknown again,
        # never a second D5 from the same answer. The operator checks and answers again.
        assert error.held.reason == HoldReason.DISPATCH_UNKNOWN
        assert step in ((2, "after"), (3, "before"))
        current = await record(db)
        assert current.settlement.voided
        if len(calls) == 2:  # the destination applied the D5 attempt before the crash
            answer = Settle.committed(current, Receipt(id="paid"), by="ops", evidence="ledger")
        else:
            answer = Settle.resend(current, by="ops", evidence="ledger shows none")
        now[0] += permissions.resend_min_age_s + 1
        await settle(db, send, permissions, answer)
        receipt = await perform(db, send, permissions)
    assert receipt.id == "paid"
    assert len(calls) == 2 and len(set(calls)) == 1  # D1 (not applied), then one application
    final = await record(db)
    assert final.status == "committed"
    rules = [entry.data["rule"] for entry in intents(await db.load(SESSION))]
    assert rules[0] == "D1" and set(rules[1:]) == {"D5"}


@pytest.mark.parametrize("phase", ["before", "after"])
async def test_a_crash_while_holding_a_forged_resend_still_holds_after_recovery(store, phase):
    db, now = store
    calls = []
    send = refund(calls)
    permissions = Permissions(approval_secret=SECRET)
    await perform_held(db, send, permissions)
    await append_straight(db, resend_data(await record(db)))
    restore = crash_at(db, 1, phase)  # the hold's effect_observed
    try:
        with pytest.raises(Crash):
            await perform(db, send, permissions)
    finally:
        restore()
    now[0] += 120
    assert_unverifiable(await perform_held(db, send, permissions))
    assert len(calls) == 1
    assert_unverifiable((await record(db)).hold)


def fresh_resend():
    """A forged resend for a key that no attempt ever used: the fold makes it a held record."""
    namespace, name, key = "session:session", "refund", "request:R1"
    return {
        "namespace": namespace,
        "name": name,
        "key": key,
        "record_id": canonical.record_id(namespace, name, key),
        "outcome": "resend",
        "by": "ops",
        "evidence": "destination audit",
        "reason": None,
        "expected_revision": 0,
        "mac": None,
        "covers_attempts": [],
    }


@pytest.mark.parametrize(
    "dedupe",
    [None, Dedupe(3600, contract="fixture", concurrent="waits")],
    ids=["no_dedupe", "dedupe"],
)
async def test_forged_resend_on_a_never_attempted_key_is_held_not_an_error(store, dedupe):
    db, _ = store
    calls = []
    send = refund(calls, failures=0, dedupe=dedupe)
    permissions = Permissions(approval_secret=SECRET)
    await append_straight(db, fresh_resend())
    assert (await record(db)).status == "held"  # the fold has no secret
    assert_unverifiable(await perform_held(db, send, permissions))
    assert calls == []
    entries = await db.load(SESSION)
    assert intents(entries) == []
    assert entries[-1].kind == "effect_observed"
    assert entries[-1].data["reason"] == HoldReason.APPROVAL_UNVERIFIABLE


def declining(calls):
    """The destination refuses the first request: a rejected record, R(n)."""

    @effect("refund", key=Key("request", "id"))
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        calls.append(attempt.idempotency_key)
        if len(calls) == 1:
            raise EffectRejected("card declined")
        return Receipt(id="paid")

    return send


async def _not_applied(db, send, permissions):
    await perform_held(db, send, permissions)
    held = await record(db)
    await settle(db, send, permissions, Settle.not_applied(held, by="ops", evidence="ledger"))


async def _close_unknown(db, send, permissions):
    await perform_held(db, send, permissions)
    held = await record(db)
    await settle(db, send, permissions, Settle.close_unknown(held, by="ops", reason="gave up"))


async def _rejected(db, send, permissions):
    with pytest.raises(EffectRejected):
        await perform(db, send, permissions)


async def _never_attempted(db, send, permissions):
    return None


# What a record is without the forged resend: (make the send, reach that status, the sends
# so far, the effect_settled outcomes after the forgery).
NOT_HELD = {
    "not_applied": (refund, _not_applied, 1, ["not_applied", "resend"]),
    "closed_unknown": (refund, _close_unknown, 1, ["close_unknown", "resend"]),
    "rejected": (declining, _rejected, 1, ["resend"]),
    "never_attempted": (lambda calls: refund(calls, failures=0), _never_attempted, 0, ["resend"]),
}


async def forge_on(db, status, calls, permissions):
    make, reach, _, _ = NOT_HELD[status]
    send = make(calls)
    await reach(db, send, permissions)
    if status != "never_attempted":
        assert (await record(db)).status == status
    await append_straight(
        db, fresh_resend() if status == "never_attempted" else resend_data(await record(db))
    )
    reopened = await record(db)
    assert reopened.status == "held"  # the fold has no secret
    return send, reopened


@pytest.mark.parametrize("status", sorted(NOT_HELD))
async def test_forged_resend_cannot_reopen_a_record_for_an_operator_resend(store, status):
    """M25 and O3r: only a held record (D) can be resent. A forged resend makes the fold
    report a closed, rejected or unused record as held; the operator path must refuse a
    real resend there, so D5 never sends it."""
    db, now = store
    calls = []
    permissions = Permissions(approval_secret=SECRET)
    send, reopened = await forge_on(db, status, calls, permissions)
    now[0] += permissions.resend_min_age_s + 1
    with pytest.raises(SettlementConflict) as refused:
        await settle(db, send, permissions, Settle.resend(reopened, by="ops", evidence="told to"))
    assert refused.value.reason == "already_settled"
    assert "does not verify under approval_secret" in str(refused.value)
    assert_unverifiable(await perform_held(db, send, permissions))
    _, _, sends, outcomes = NOT_HELD[status]
    assert len(calls) == sends
    entries = await db.load(SESSION)
    assert [entry.data["rule"] for entry in intents(entries)] == ["D1"] * sends
    assert [entry.data["outcome"] for entry in entries if entry.kind == "effect_settled"] == (
        outcomes
    )


@pytest.mark.parametrize("status", sorted(NOT_HELD))
async def test_an_operator_closes_a_record_a_forged_resend_reopened(store, status):
    """The other answers send nothing, so the hold's "settle again" works: not_applied closes
    the record again, and the tool sees EffectRejected with no send."""
    db, now = store
    calls = []
    permissions = Permissions(approval_secret=SECRET)
    send, reopened = await forge_on(db, status, calls, permissions)
    closed = await settle(
        db, send, permissions, Settle.not_applied(reopened, by="ops", evidence="ledger")
    )
    assert closed.status == "not_applied"
    with pytest.raises(EffectRejected):
        await perform(db, send, permissions)
    assert len(calls) == NOT_HELD[status][2]


async def test_without_a_secret_an_operator_resend_on_a_journaled_resend_is_accepted(store):
    """Without a secret the journaled resend is trusted, as before: the record is D."""
    db, now = store
    calls = []
    permissions = Permissions()
    send, reopened = await forge_on(db, "not_applied", calls, permissions)
    now[0] += permissions.resend_min_age_s + 1
    accepted = await settle(
        db, send, permissions, Settle.resend(reopened, by="ops", evidence="told to")
    )
    assert accepted.status == "held" and accepted.settlement.outcome == "resend"
