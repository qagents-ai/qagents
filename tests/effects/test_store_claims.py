"""Cross-session claims and metadata are committed with journal batches."""

import pytest

from qagents.errors import ClaimConflict, SessionError
from qagents.session import Entry, InMemoryStore, SQLiteStore

from ._samples import claimed, committed, failed, intent, reference


@pytest.fixture(params=["memory", "sqlite"])
async def db(request, tmp_path):
    store = InMemoryStore() if request.param == "memory" else SQLiteStore(tmp_path / "claims.db")
    for session in ("a", "b"):
        assert await store.acquire_lease(session, session, 30)
    yield store
    await store.aclose()


async def write(db, session, kind, data):
    return await db.append_leased(session, [Entry(kind=kind, data=data)], session)


async def test_claim_conflict_rolls_back_the_whole_batch(db):
    await write(db, "a", "effect_intent", intent())
    with pytest.raises(ClaimConflict):
        await db.append_leased(
            "b", [Entry(kind="custom"), Entry(kind="effect_intent", data=intent())], "b"
        )
    assert await db.load("b") == []
    claim = await db.get_claim("shop", "refund", "R1")
    assert claim.session_id == "a" and claim.status == "sent"


@pytest.mark.parametrize("why", ["rejected", "not_sent"])
async def test_definite_no_send_claim_moves_preserving_first_use(db, why):
    await write(db, "a", "effect_intent", intent(first_use_at=77))
    await write(db, "a", "effect_failed", failed(why=why))
    await write(db, "b", "effect_intent", intent(generation=1, first_use_at=999))
    claim = await db.get_claim("shop", "refund", "R1")
    assert claim.session_id == "b" and claim.first_use_at == 77 and claim.generation == 1


async def test_unknown_then_second_attempt_not_sent_cannot_move_claim(db):
    await write(db, "a", "effect_intent", intent())
    await write(db, "a", "effect_failed", failed())
    await write(db, "a", "effect_intent", intent(attempt=2))
    await write(db, "a", "effect_failed", failed(attempt=2, intent_seq=3, why="not_sent"))
    assert (await db.get_claim("shop", "refund", "R1")).status == "held"
    with pytest.raises(ClaimConflict):
        await write(db, "b", "effect_intent", intent())


async def test_different_descriptor_cannot_reuse_rejected_claim(db):
    await write(db, "a", "effect_intent", intent())
    await write(db, "a", "effect_failed", failed(why="rejected"))
    with pytest.raises(ClaimConflict):
        await write(db, "b", "effect_intent", intent(request_digest="different"))


async def test_committed_claim_survives_delete_and_replay_preserves_owner(db):
    await write(db, "a", "effect_intent", intent())
    await write(db, "a", "effect_committed", committed())
    old = await db.get_claim("shop", "refund", "R1")
    await db.delete("a")
    await write(db, "b", "effect_committed", committed(via="claim", attempt=None, intent_seq=None))
    assert await db.get_claim("shop", "refund", "R1") == old
    assert old.receipt == {"id": "refund-1", "status": "pending"}
    assert await db.purge_claims(older_than=old.updated_at) == 0
    assert await db.purge_claims(older_than=old.updated_at + 1) == 1


async def test_unleased_effects_and_sticky_lifecycle_refusal(db):
    with pytest.raises(SessionError):
        await db.append("a", [Entry(kind="effect_intent", data=intent())])
    await write(db, "a", "effect_intent", intent())
    with pytest.raises(SessionError):
        await db.append("a", [Entry(kind="run_started", run_id="r")])
    await db.append("a", [Entry(kind="workflow_signal", data={"value": 1})])
    assert (await db.status("a")).has_effects


async def test_registry_mismatch_does_not_replace_first_contract(db):
    assert await db.register_effect("refund", "1", "key-a", "schema-a") == "registered"
    assert await db.register_effect("refund", "1", "key-b", "schema-a") == "key_changed"
    assert await db.register_effect("refund", "1", "key-a", "schema-b") == "schema_changed"
    assert await db.register_effect("refund", "1", "key-a", "schema-a") == "matches"


async def test_sqlite_metadata_survives_reopen(tmp_path):
    path = tmp_path / "metadata.db"
    async with SQLiteStore(path) as db:
        salt = await db.salt()
        assert len(salt) == 32
        await db.register_effect("refund", "1", "key", "schema")
    async with SQLiteStore(path) as db:
        assert await db.salt() == salt
        assert await db.register_effect("refund", "1", "key", "schema") == "matches"


async def test_invalid_or_oversized_terminal_leaves_claim_and_journal_unchanged(db):
    await write(db, "a", "effect_intent", intent())
    before = await db.get_claim("shop", "refund", "R1")
    for data in (
        committed(receipt={"id": "r", "payload": "x" * 17000}),
        {"namespace": "shop", "name": "refund", "key": "R1"},
    ):
        with pytest.raises(SessionError):
            await write(db, "a", "effect_committed", data)
        assert await db.get_claim("shop", "refund", "R1") == before
        assert len(await db.load("a")) == 1


async def transfer_of(db, **changes):
    """The effect_claimed payload for the current claim, as the engine journals it."""
    claim = await db.get_claim("shop", "refund", "R1")
    return claimed(
        previous_owner={
            "session_id": claim.session_id,
            "incarnation": claim.incarnation,
            "record_id": claim.record_id,
        },
        generation=claim.generation,
        first_use_at=claim.first_use_at,
        attempts=[item.model_dump(mode="json") for item in claim.attempts],
        **changes,
    )


async def test_claim_keeps_every_attempt_and_outcome_across_owners(db):
    await write(db, "a", "effect_intent", intent(first_use_at=77))
    await write(db, "a", "effect_failed", failed(why="rejected"))
    claim = await db.get_claim("shop", "refund", "R1")
    assert [(a.session_id, a.number, a.idempotency_key, a.outcome) for a in claim.attempts] == [
        ("a", 1, "q1_0", "rejected")
    ]
    await write(db, "b", "effect_claimed", await transfer_of(db))
    moved = await db.get_claim("shop", "refund", "R1")
    assert (moved.session_id, moved.status, moved.generation, moved.first_use_at) == (
        "b",
        "rejected",
        0,
        77,
    )
    assert moved.attempts == claim.attempts
    await write(db, "b", "effect_intent", intent(attempt=2, rule="D2", first_use_at=77))
    await write(db, "b", "effect_committed", committed(attempt=2, intent_seq=2))
    final = await db.get_claim("shop", "refund", "R1")
    assert [(a.session_id, a.number, a.outcome) for a in final.attempts] == [
        ("a", 1, "rejected"),
        ("b", 2, "committed"),
    ]
    assert final.status == "committed" and final.first_use_at == 77


async def test_only_the_new_owner_writes_after_the_claim_moved(db):
    await write(db, "a", "effect_intent", intent())
    await write(db, "a", "effect_failed", failed(why="rejected"))
    stale = await transfer_of(db)
    await write(db, "b", "effect_claimed", stale)
    observed = {**reference(), "attempts": [], "verdict": "absent", "reason": "LOOKUP_INCONCLUSIVE"}
    with pytest.raises(ClaimConflict):
        await write(db, "a", "effect_observed", observed)
    assert await db.acquire_lease("c", "c", 30)
    with pytest.raises(ClaimConflict):  # a transfer naming the old owner is now stale
        await write(db, "c", "effect_claimed", stale)
    await write(db, "b", "effect_observed", observed)
    claim = await db.get_claim("shop", "refund", "R1")
    assert (claim.session_id, claim.status) == ("b", "rejected")  # reconciled from b's fold


@pytest.mark.parametrize(
    "change",
    [
        {"previous_owner": {"session_id": "other", "incarnation": "x", "record_id": "y"}},
        {"generation": 1},
        {"first_use_at": 5.0},
        {"attempts": []},
        {"request_digest": "different"},
    ],
    ids=["owner", "generation", "first_use", "attempts", "descriptor"],
)
async def test_a_claim_moves_only_while_it_matches_what_the_new_owner_read(db, change):
    await write(db, "a", "effect_intent", intent())
    await write(db, "a", "effect_failed", failed(why="rejected"))
    before = await db.get_claim("shop", "refund", "R1")
    stale = {**await transfer_of(db), **change}
    with pytest.raises(ClaimConflict):
        await db.append_leased(
            "b", [Entry(kind="custom"), Entry(kind="effect_claimed", data=stale)], "b"
        )
    assert await db.load("b") == []  # the whole batch is refused
    assert await db.get_claim("shop", "refund", "R1") == before


@pytest.mark.parametrize("why", [None, "unknown", "committed"])
async def test_only_a_rejected_or_never_sent_claim_can_be_taken_over(db, why):
    if why is not None:
        await write(db, "a", "effect_intent", intent())
        if why == "committed":
            await write(db, "a", "effect_committed", committed())
        else:
            await write(db, "a", "effect_failed", failed(why=why))
        payload = await transfer_of(db)
    else:
        payload = claimed()
    with pytest.raises(ClaimConflict):
        await write(db, "b", "effect_claimed", payload)
    assert await db.load("b") == []


@pytest.mark.parametrize("why", ["rejected", "not_sent"])
async def test_a_transfer_moves_a_claim_only_under_the_status_it_read(db, why):
    """AF-2 recheck: a claim whose last attempt was never sent moves too, so a record in
    R(n) can take it back; the status it names is compared like the rest of the claim."""
    await write(db, "a", "effect_intent", intent(first_use_at=77))
    await write(db, "a", "effect_failed", failed(why=why))
    before = await db.get_claim("shop", "refund", "R1")
    assert before.status == why
    other = "not_sent" if why == "rejected" else "rejected"
    with pytest.raises(ClaimConflict):
        await write(db, "b", "effect_claimed", await transfer_of(db, status=other))
    assert await db.load("b") == []
    assert await db.get_claim("shop", "refund", "R1") == before

    await write(db, "b", "effect_claimed", await transfer_of(db, status=why))
    moved = await db.get_claim("shop", "refund", "R1")
    assert (moved.session_id, moved.generation, moved.first_use_at) == ("b", 0, 77)
    assert moved.attempts == before.attempts
    assert moved.status == "rejected"  # reconciled from b's fold, R(n): a lookup runs first
    observed = {**reference(), "attempts": [], "verdict": "absent", "reason": "LOOKUP_INCONCLUSIVE"}
    with pytest.raises(ClaimConflict):  # only the new owner journals its lookup now
        await write(db, "a", "effect_observed", observed)
    await write(db, "b", "effect_observed", observed)
