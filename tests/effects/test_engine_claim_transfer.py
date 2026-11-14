"""A retry after another session's rejection runs the declared lookup first (SPEC 4.4 D2, 7.10).

Session "one" sends and the destination refuses, so the business key's claim is rejected.
Session "two" then performs the same business key. The claim moves to "two" in a journaled
effect_claimed entry that carries the earlier attempts, and only then do the R(n) rows decide:
the lookup runs first, Found commits without a send, and only an absence leads to D2.
"""

from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from qagents import Agent, RunContext, tool
from qagents.effects import Absent, Attempt, Dedupe, EffectContext, Found, Key, effect
from qagents.effects import _engine as engine
from qagents.effects._canonical import idempotency_key
from qagents.effects._fold import EffectView, fold_effects
from qagents.errors import EffectRejected
from qagents.models.testing import ScriptedModel, call
from qagents.session import InMemoryStore, LeasedWriter, Session, SQLiteStore

from .test_engine_dispatch import Receipt, Request, host

KEY = ("shop", "refund", "request:R1")


class Refused(Exception):
    """A destination answer the declaration lists in rejected_on."""


class Destination:
    """Applies each idempotency key once; finds an application by its operation tag."""

    def __init__(self, mode="reject"):
        self.mode = mode
        self.applied = []  # (idempotency_key, operation_tag, receipt)
        self.events = []  # ("send" | "lookup", session_id, idempotency key or Inquiry)

    def sends(self, session=None):
        return [e[2] for e in self.events if e[0] == "send" and session in (None, e[1])]

    def lookups(self, session=None):
        return [e[2] for e in self.events if e[0] == "lookup" and session in (None, e[1])]


def declare(destination, *, replays_rejections=False, lookup=True):
    contract = Dedupe(
        3600, contract="fixture", concurrent="waits", replays_rejections=replays_rejections
    )

    @effect(
        "refund",
        key=Key("request", "id"),
        namespace="shop",
        dedupe=contract,
        rejected_on=(Refused,),
        receipt_identity=("id",),
    )
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        destination.events.append(("send", ctx.session_id, attempt.idempotency_key))
        for key, _, receipt in destination.applied:
            if key == attempt.idempotency_key:
                return receipt
        if destination.mode == "reject":
            raise EffectRejected("card declined")
        receipt = Receipt(id=f"refund-{len(destination.applied) + 1}")
        destination.applied.append((attempt.idempotency_key, attempt.operation_tag, receipt))
        if destination.mode == "apply_then_refuse":
            raise Refused("refused")  # rejection-shaped, but the destination applied it
        return receipt

    if lookup:

        @send.lookup
        async def find(ctx, inquiry):
            destination.events.append(("lookup", ctx.session_id, inquiry))
            for _, tag, receipt in destination.applied:
                if tag == inquiry.operation_tag:
                    return Found(receipt, tag=tag)
            return Absent()

    return send


@pytest.fixture(params=("memory", "sqlite"))
async def store(request, tmp_path):
    now = [1000.0]

    def clock():
        return now[0]

    value = (
        InMemoryStore(clock=clock)
        if request.param == "memory"
        else SQLiteStore(tmp_path / "claims.sqlite", clock=clock)
    )
    async with value:
        yield value, now


@asynccontextmanager
async def opened(db, session, operation):
    writer = LeasedWriter(
        db, session, worker="test", ttl_s=30, purpose="run", journal_acquisition=True
    )
    assert await writer.open()
    try:
        await writer.load()
        scope, _, _ = host(writer)
        yield replace(scope, session_id=session, operation_id=operation)
    finally:
        await writer.close()


async def perform(db, send, session, operation):
    async with opened(db, session, operation) as scope:
        return await engine.perform(send, Request(id="R1"), scope=scope)


async def reject_in_session_one(db, destination, send):
    with pytest.raises(EffectRejected):
        await perform(db, send, "one", "run-1:1:call")
    claim = await db.get_claim(*KEY)
    assert claim.session_id == "one" and claim.status == "rejected"
    return claim


def kinds_of(entries):
    return [entry.kind for entry in entries if entry.kind != "lease_acquired"]


@pytest.mark.parametrize("replays_rejections", [False, True])
async def test_retry_in_another_session_looks_up_first_and_sends_d2_on_absence(
    store, replays_rejections
):
    db, _ = store
    destination = Destination()
    send = declare(destination, replays_rejections=replays_rejections)
    earlier = await reject_in_session_one(db, destination, send)
    [first_key] = destination.sends("one")

    destination.mode = "apply"
    receipt = await perform(db, send, "two", "run-2:1:call")

    # The lookup ran before any send in session two, and it asked about session one's key.
    assert destination.events[1][:2] == ("lookup", "two")
    [inquiry] = destination.lookups("two")
    assert inquiry.idempotency_keys == (first_key,)
    assert [(a.number, a.generation, a.outcome) for a in inquiry.attempts] == [(1, 0, "rejected")]
    assert inquiry.first_use_at == earlier.first_use_at
    # D2 then sent under the key SPEC 4.4 prescribes: the same one, or generation + 1.
    [second_key] = destination.sends("two")
    salt = await db.salt()
    generation = int(replays_rejections)
    assert second_key == idempotency_key(salt, *KEY, generation, 64)
    assert (second_key == first_key) is (not replays_rejections)
    assert receipt.id == "refund-1" and len(destination.applied) == 1

    entries = await db.load("two")
    assert kinds_of(entries) == [
        "effect_claimed",
        "effect_observed",
        "effect_intent",
        "effect_committed",
    ]
    taken = entries[[e.kind for e in entries].index("effect_claimed")]
    assert taken.data["previous_owner"]["session_id"] == "one"
    assert [a["idempotency_key"] for a in taken.data["attempts"]] == [first_key]
    [intent] = [e for e in entries if e.kind == "effect_intent"]
    assert intent.data["rule"] == "D2" and intent.data["attempt"] == 2
    assert intent.data["generation"] == generation
    assert intent.data["first_use_at"] == earlier.first_use_at
    assert intent.seq > taken.seq  # record first, act second
    claim = await db.get_claim(*KEY)
    assert claim.session_id == "two" and claim.status == "committed"
    assert [(a.session_id, a.idempotency_key, a.outcome) for a in claim.attempts] == [
        ("one", first_key, "rejected"),
        ("two", second_key, "committed"),
    ]


async def test_retry_in_another_session_commits_a_found_result_without_sending(store):
    db, _ = store
    destination = Destination()
    send = declare(destination)
    await reject_in_session_one(db, destination, send)
    # The destination now holds the operation (for example, applied out of band by tag).
    tag = engine.canonical.operation_tag(
        *KEY, engine.canonical.freeze(send, Request(id="R1")).digest
    )
    destination.applied.append(("elsewhere", tag, Receipt(id="found-1")))

    receipt = await perform(db, send, "two", "run-2:1:call")

    assert receipt.id == "found-1"
    assert destination.sends("two") == [] and len(destination.lookups("two")) == 1
    entries = await db.load("two")
    assert kinds_of(entries) == ["effect_claimed", "effect_observed", "effect_committed"]
    [commit] = [e for e in entries if e.kind == "effect_committed"]
    assert commit.data["via"] == "lookup" and commit.data["attempt"] is None
    view = fold_effects(entries)
    assert view.get(*KEY).state == "C" and not view.get(*KEY).findings
    claim = await db.get_claim(*KEY)
    assert claim.session_id == "two" and claim.status == "committed"
    assert claim.receipt == {"id": "found-1"}
    # A third session replays the committed claim; nothing else is sent or looked up.
    assert (await perform(db, send, "three", "run-3:1:call")).id == "found-1"
    assert destination.sends("three") == [] and destination.lookups("three") == []


@pytest.mark.parametrize("replays_rejections", [False, True])
async def test_applied_then_rejection_shaped_error_is_found_not_applied_again(
    store, replays_rejections
):
    """commit_then_reject_shaped_error: one application, also under a fresh key (STRIPE)."""
    db, _ = store
    destination = Destination("apply_then_refuse")
    send = declare(destination, replays_rejections=replays_rejections)
    await reject_in_session_one(db, destination, send)
    assert len(destination.applied) == 1  # session one saw "refused", but it applied

    destination.mode = "apply"
    receipt = await perform(db, send, "two", "run-2:1:call")

    assert receipt.id == "refund-1"
    assert len(destination.applied) == 1
    assert destination.sends("two") == [] and len(destination.lookups("two")) == 1
    assert (await db.get_claim(*KEY)).status == "committed"


async def test_retry_in_another_session_without_a_lookup_still_sends_d2(store):
    db, _ = store
    destination = Destination()
    send = declare(destination, lookup=False)
    await reject_in_session_one(db, destination, send)
    destination.mode = "apply"
    assert (await perform(db, send, "two", "run-2:1:call")).id == "refund-1"
    assert len(destination.sends("two")) == 1
    assert kinds_of(await db.load("two")) == ["effect_claimed", "effect_intent", "effect_committed"]


async def test_within_session_retry_is_unchanged(store):
    """Control: inside one session the earlier attempt is local; nothing is transferred."""
    db, _ = store
    destination = Destination()
    send = declare(destination)
    async with opened(db, "one", "run-1:1:call") as scope:
        with pytest.raises(EffectRejected):
            await engine.perform(send, Request(id="R1"), scope=scope)
        destination.mode = "apply"
        later = replace(scope, operation_id="run-1:2:call")
        assert (await engine.perform(send, Request(id="R1"), scope=later)).id == "refund-1"
    [inquiry] = destination.lookups("one")
    assert [(a.number, a.outcome) for a in inquiry.attempts] == [(1, "rejected")]
    assert [e[0] for e in destination.events] == ["send", "lookup", "send"]
    assert kinds_of(await db.load("one")) == [
        "effect_intent",
        "effect_failed",
        "effect_observed",
        "effect_intent",
        "effect_committed",
    ]
    claim = await db.get_claim(*KEY)
    assert claim.session_id == "one" and claim.status == "committed"


class Crash(BaseException):
    """The worker dies at a journal append (not an Exception: nothing retries it)."""


def crash_at(db, session, number, phase):
    """Kill the worker at the number-th effect batch of the session, before or after it commits."""
    append = db.append_leased
    seen = [0]

    async def append_leased(session_id, entries, owner):
        if session_id == session and any(e.kind.startswith("effect_") for e in entries):
            seen[0] += 1
            if seen[0] == number:
                if phase == "before":
                    raise Crash()
                await append(session_id, entries, owner)
                raise Crash()
        return await append(session_id, entries, owner)

    db.append_leased = append_leased
    return lambda: setattr(db, "append_leased", append)


# Session two's effect batches: claimed, observed, committed (found) or claimed, observed,
# intent, committed (absent). A crash before or after each one, then a recovery of the call.
STEPS = [(number, phase) for number in range(1, 5) for phase in ("before", "after")]


@pytest.mark.parametrize("found", [True, False], ids=["found", "absent"])
@pytest.mark.parametrize("step", STEPS, ids=[f"{phase}-{n}" for n, phase in STEPS])
async def test_a_crash_at_every_new_step_recovers_from_the_journal(store, found, step):
    db, now = store
    destination = Destination()
    send = declare(destination)
    await reject_in_session_one(db, destination, send)
    destination.mode = "apply"
    if found:
        tag = engine.canonical.operation_tag(
            *KEY, engine.canonical.freeze(send, Request(id="R1")).digest
        )
        destination.applied.append(("elsewhere", tag, Receipt(id="found-1")))

    restore = crash_at(db, "two", *step)
    try:
        receipt = await perform(db, send, "two", "run-2:1:call")
    except Crash:
        receipt = None
    finally:
        restore()
    entries = await db.load("two")
    # The fold of every durable prefix matches the incremental fold.
    view = EffectView()
    for index, item in enumerate(entries):
        view.apply(item)
        assert view.records("two") == fold_effects(entries[: index + 1]).records("two")
    assert (receipt is None) is (step[0] <= (3 if found else 4))  # the worker died there
    if receipt is None:
        now[0] += 120  # the dead worker's lease expires; the call is recovered
        receipt = await perform(db, send, "two", "run-2:1:call")

    assert receipt.id == ("found-1" if found else "refund-1")
    assert len(destination.applied) == 1
    # In every leg the lookup ran before session two sent anything.
    two = [e[0] for e in destination.events if e[1] == "two"]
    assert two and two[0] == "lookup"
    if found:
        assert destination.sends("two") == []
    final = fold_effects(await db.load("two")).get(*KEY)
    assert final.state == "C" and not final.findings
    claim = await db.get_claim(*KEY)
    assert claim.session_id == "two" and claim.status == "committed"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_agent_run_in_a_second_session_looks_up_before_retrying(backend, tmp_path):
    """The fable-skeptic s3 probe as a test: lookups were 0 across sessions, 1 within one."""
    destination = Destination()
    send = declare(destination)

    @tool(effects=[send])
    async def refund(ctx: RunContext, id: str) -> str:
        """Refund the order."""
        try:
            return (await ctx.perform(send, Request(id=id))).id
        except EffectRejected:
            return "rejected"

    def agent():
        return Agent(ScriptedModel([call("refund", id="R1"), "Done."]), tools=[refund])

    db = InMemoryStore() if backend == "memory" else SQLiteStore(tmp_path / "agent.sqlite")
    async with db:
        first = await agent().run("Refund", session=Session(db, id="s1"))
        destination.mode = "apply"
        second = await agent().run("Refund", session=Session(db, id="s2"))
        records = await Session(db, id="s2").effects()
    assert first.status == second.status == "completed"
    assert [(kind, session) for kind, session, _ in destination.events] == [
        ("send", "s1"),
        ("lookup", "s2"),
        ("send", "s2"),
    ]
    assert len(destination.applied) == 1
    assert [(r.status, r.via) for r in records] == [("committed", "direct")]
