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
from qagents.effects import (
    Absent,
    Attempt,
    Dedupe,
    EffectContext,
    Found,
    HoldReason,
    Key,
    effect,
)
from qagents.effects import _engine as engine
from qagents.effects._canonical import idempotency_key
from qagents.effects._fold import EffectView, fold_effects
from qagents.errors import EffectHeld, EffectRejected, LeaseError
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


def declare(destination, *, replays_rejections=False, lookup=True, timeout_s=None):
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
        **({} if timeout_s is None else {"timeout_s": timeout_s}),
    )
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        destination.events.append(("send", ctx.session_id, attempt.idempotency_key))
        for key, _, receipt in destination.applied:
            if key == attempt.idempotency_key:
                return receipt
        if destination.mode == "reject":
            raise EffectRejected("card declined")
        if destination.mode == "error":
            raise RuntimeError("connection reset")  # neither declared nor definite: held
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
async def opened(db, session, operation, ttl_s=30):
    writer = LeasedWriter(
        db, session, worker="test", ttl_s=ttl_s, purpose="run", journal_acquisition=True
    )
    assert await writer.open()
    try:
        await writer.load()
        scope, _, _ = host(writer)
        yield replace(scope, session_id=session, operation_id=operation)
    finally:
        await writer.close()


async def perform(db, send, session, operation, ttl_s=30):
    async with opened(db, session, operation, ttl_s) as scope:
        return await engine.perform(send, Request(id="R1"), scope=scope)


async def reject_in_session_one(db, destination, send, ttl_s=30):
    with pytest.raises(EffectRejected):
        await perform(db, send, "one", "run-1:1:call", ttl_s)
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


async def test_a_takeover_whose_self_fence_fails_journals_nothing_and_keeps_the_claim(store):
    """SPEC 7.1 C1: nothing is journaled if the self-fence fails (review AF-2-self-fence).

    Session two's 30 s lease cannot cover timeout_s=60 plus m, so it may not send. Before
    the fix it still moved the claim and looked up, and the owner's own retry then failed
    with ClaimConflict.
    """
    db, _ = store
    destination = Destination()
    send = declare(destination, timeout_s=60)
    earlier = await reject_in_session_one(db, destination, send, ttl_s=600)

    destination.mode = "apply"
    with pytest.raises(LeaseError):
        await perform(db, send, "two", "run-2:1:call", ttl_s=30)

    assert kinds_of(await db.load("two")) == []
    assert destination.events[1:] == []  # no lookup and no send by session two
    assert await db.get_claim(*KEY) == earlier
    # The owner, with a lease long enough to send, retries in a new call.
    assert (await perform(db, send, "one", "run-1:2:call", ttl_s=600)).id == "refund-1"
    assert len(destination.applied) == 1


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


# A record already in R(n) reads the claim again before L2 or D2 (SPEC 7.1 C1, 7.10). Review
# AF-2-old-owner-blocked and AF-2-old-owner-lockout: a taker that journaled the transfer and
# died before sending left the claim's earlier owner with ClaimConflict on every retry.


async def outcome_of(awaitable):
    """The receipt, or the error raised, so that a test asserts which one it got."""
    try:
        return await awaitable
    except Exception as error:
        return error


async def take_over_and_die(db, send, session="two"):
    """The session journals the claim transfer, then dies before its lookup is journaled."""
    restore = crash_at(db, session, 2, "before")
    try:
        with pytest.raises(Crash):
            await perform(db, send, session, f"run-{session}:1:call")
    finally:
        restore()
    assert kinds_of(await db.load(session)) == ["effect_claimed"]
    claim = await db.get_claim(*KEY)
    assert claim.session_id == session and claim.status == "rejected"
    return claim


@pytest.mark.parametrize("replays_rejections", [False, True])
async def test_the_owner_takes_its_claim_back_from_a_taker_that_sent_nothing(
    store, replays_rejections
):
    db, now = store
    destination = Destination()
    send = declare(destination, replays_rejections=replays_rejections)
    await reject_in_session_one(db, destination, send)
    [first_key] = destination.sends("one")
    destination.mode = "apply"
    taken = await take_over_and_die(db, send)
    now[0] += 120

    receipt = await outcome_of(perform(db, send, "one", "run-1:2:call"))

    assert isinstance(receipt, Receipt), receipt
    assert receipt.id == "refund-1" and len(destination.applied) == 1
    assert [e[:2] for e in destination.events] == [
        ("send", "one"),
        ("lookup", "two"),
        ("lookup", "one"),
        ("send", "one"),
    ]
    # The owner's own earlier attempt is in the claim's history; the lookup asks once.
    inquiry = destination.lookups("one")[0]
    assert inquiry.idempotency_keys == (first_key,)
    assert [(a.number, a.generation, a.outcome) for a in inquiry.attempts] == [(1, 0, "rejected")]
    entries = await db.load("one")
    assert kinds_of(entries) == [
        "effect_intent",
        "effect_failed",
        "effect_claimed",
        "effect_observed",
        "effect_intent",
        "effect_committed",
    ]
    [back] = [e for e in entries if e.kind == "effect_claimed"]
    assert back.data["previous_owner"]["session_id"] == "two"
    assert back.data["attempts"] == [a.model_dump(mode="json") for a in taken.attempts]
    intent = [e for e in entries if e.kind == "effect_intent"][-1]
    assert (intent.data["rule"], intent.data["attempt"]) == ("D2", 2)
    assert intent.data["generation"] == int(replays_rejections)
    assert intent.seq > back.seq  # record first, act second
    claim = await db.get_claim(*KEY)
    assert claim.session_id == "one" and claim.status == "committed"
    # The taker, recovered later, replays that commit; it neither looks up nor sends.
    assert (await perform(db, send, "two", "run-two:1:call")).id == "refund-1"
    assert len(destination.events) == 4
    assert [e.data.get("via") for e in await db.load("two")][-1] == "claim"


@pytest.mark.parametrize("replays_rejections", [False, True])
async def test_the_owner_takes_back_a_claim_refused_again_elsewhere_with_every_key(
    store, replays_rejections
):
    """The lookup asks about the newer owner's key too, and D2 numbers after it."""
    db, _ = store
    destination = Destination()
    send = declare(destination, replays_rejections=replays_rejections)
    await reject_in_session_one(db, destination, send)
    with pytest.raises(EffectRejected):
        await perform(db, send, "two", "run-2:1:call")  # looks up, then D2 is refused too
    claim = await db.get_claim(*KEY)
    assert claim.session_id == "two" and claim.status == "rejected"
    destination.mode = "apply"

    receipt = await outcome_of(perform(db, send, "one", "run-1:2:call"))

    assert isinstance(receipt, Receipt), receipt
    assert receipt.id == "refund-1" and len(destination.applied) == 1
    generation = int(replays_rejections)
    [inquiry] = destination.lookups("one")
    keys = tuple(dict.fromkeys(a.idempotency_key for a in claim.attempts))
    assert inquiry.idempotency_keys == keys
    assert [(a.number, a.generation, a.outcome) for a in inquiry.attempts] == [
        (1, 0, "rejected"),
        (2, generation, "rejected"),
    ]
    intent = [e for e in await db.load("one") if e.kind == "effect_intent"][-1]
    assert (intent.data["rule"], intent.data["attempt"]) == ("D2", 3)
    salt = await db.salt()
    assert intent.data["idempotency_key"] == idempotency_key(salt, *KEY, 2 * generation, 64)
    claim = await db.get_claim(*KEY)
    assert [(a.session_id, a.number, a.outcome) for a in claim.attempts] == [
        ("one", 1, "rejected"),
        ("two", 2, "rejected"),
        ("one", 3, "committed"),
    ]


async def test_a_rejected_record_holds_while_another_session_may_still_apply(store):
    db, _ = store
    destination = Destination()
    send = declare(destination)
    await reject_in_session_one(db, destination, send)
    destination.mode = "error"
    with pytest.raises(EffectHeld):
        await perform(db, send, "two", "run-2:1:call")
    claim = await db.get_claim(*KEY)
    assert claim.session_id == "two" and claim.status == "held"
    before = kinds_of(await db.load("one"))
    seen = len(destination.events)

    held = await outcome_of(perform(db, send, "one", "run-1:2:call"))

    assert isinstance(held, EffectHeld), held
    assert held.held.reason == HoldReason.CLAIMED_ELSEWHERE and "two" in held.held.detail
    assert kinds_of(await db.load("one")) == before  # nothing journaled
    assert len(destination.events) == seen  # neither a lookup nor a send


async def test_a_rejected_record_replays_a_commit_another_session_made(store):
    db, _ = store
    destination = Destination()
    send = declare(destination)
    await reject_in_session_one(db, destination, send)
    destination.mode = "apply"
    assert (await perform(db, send, "two", "run-2:1:call")).id == "refund-1"
    seen = len(destination.events)
    # The call whose own attempt was refused still gets its recorded rejection (P3r).
    with pytest.raises(EffectRejected):
        await perform(db, send, "one", "run-1:1:call")

    receipt = await outcome_of(perform(db, send, "one", "run-1:2:call"))

    assert isinstance(receipt, Receipt), receipt
    assert receipt.id == "refund-1" and len(destination.applied) == 1
    assert len(destination.events) == seen
    entries = await db.load("one")
    assert entries[-1].kind == "effect_committed" and entries[-1].data["via"] == "claim"
    assert entries[-1].data["claim_owner"]["session_id"] == "two"
    final = fold_effects(entries).get(*KEY)
    assert final.state == "C" and not final.findings


@pytest.mark.parametrize("found", [True, False], ids=["found", "absent"])
@pytest.mark.parametrize("step", STEPS, ids=[f"{phase}-{n}" for n, phase in STEPS])
async def test_a_crash_at_every_step_of_taking_a_claim_back_recovers_from_the_journal(
    store, found, step
):
    """Session one's batches: claimed, observed, committed (found) or claimed, observed,
    intent, committed (absent). A kill before or after each, then a recovery of the call."""
    db, now = store
    destination = Destination("apply_then_refuse" if found else "reject")
    send = declare(destination, replays_rejections=True)
    await reject_in_session_one(db, destination, send)
    destination.mode = "apply"
    await take_over_and_die(db, send)
    now[0] += 120

    restore = crash_at(db, "one", *step)
    try:
        receipt = await perform(db, send, "one", "run-1:2:call")
    except Crash:
        receipt = None
    finally:
        restore()
    entries = await db.load("one")
    view = EffectView()
    for index, item in enumerate(entries):
        view.apply(item)
        assert view.records("one") == fold_effects(entries[: index + 1]).records("one")
    assert (receipt is None) is (step[0] <= (3 if found else 4))  # the worker died there
    if receipt is None:
        now[0] += 120
        receipt = await perform(db, send, "one", "run-1:2:call")

    assert receipt.id == "refund-1"
    assert len(destination.applied) == 1
    retry = destination.events[2:]  # after one's first send and two's lookup
    assert retry and retry[0][0] == "lookup" and {e[1] for e in retry} == {"one"}
    if found:
        assert [e for e in retry if e[0] == "send"] == []
    final = fold_effects(await db.load("one")).get(*KEY)
    assert final.state == "C" and not final.findings
    claim = await db.get_claim(*KEY)
    assert claim.session_id == "one" and claim.status == "committed"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_agent_owner_session_retries_after_a_taker_died_before_sending(backend, tmp_path):
    """Review probe p8: the customer's own session s1 retries after a run in s2 died."""
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
        await agent().run("Refund", session=Session(db, id="s1"))
        destination.mode = "apply"
        restore = crash_at(db, "s2", 2, "before")
        try:
            with pytest.raises(Crash):
                await agent().run("Refund", session=Session(db, id="s2"))
        finally:
            restore()
        assert (await db.get_claim(*KEY)).session_id == "s2"
        again = await agent().run("Refund again", session=Session(db, id="s1"))
        records = await Session(db, id="s1").effects()
    assert again.status == "completed"
    assert [(kind, session) for kind, session, _ in destination.events] == [
        ("send", "s1"),
        ("lookup", "s2"),
        ("lookup", "s1"),
        ("send", "s1"),
    ]
    assert len(destination.applied) == 1
    assert [(r.status, r.via) for r in records] == [("committed", "direct")]
