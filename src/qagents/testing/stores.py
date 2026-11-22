"""Executable store checks; replication durability needs deployment-specific tests."""

from __future__ import annotations

import asyncio
import inspect
import json
import subprocess
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..effects._canonical import digest, record_id
from ..errors import ClaimConflict, EntryConflict, LeaseError, SessionError
from ..session import Entry, SessionStore, SQLiteStore


@dataclass
class StoreReport:
    ok: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    not_checked: list[str] = field(default_factory=list)


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(detail)


def _intent(namespace: str) -> Entry:
    return Entry(
        kind="effect_intent",
        data={
            "namespace": namespace,
            "name": "store-check",
            "key": "one",
            "record_id": record_id(namespace, "store-check", "one"),
            "key_spec_digest": "key",
            "version": "1",
            "operation_id": "check:1:call",
            "tool_call_id": "call",
            "occurrence": None,
            "attempt": 1,
            "generation": 0,
            "idempotency_key": "probe",
            "operation_tag": "probe",
            "request_digest": digest({"id": "one"}),
            "destination_digest": digest(None),
            "request": {"id": "one"},
            "destination": None,
            "epoch": 1,
            "store_ts": 1.0,
            "first_use_at": 1.0,
            "not_after": None,
            "may_overrun": False,
            "authorization": {"decision": "allow"},
            "rule": "D1",
            "timing": {
                "timeout_s": 1,
                "applies_within_s": None,
                "m_s": 2,
                "sigma_s": 1,
                "window_s": None,
                "margin_s": None,
                "unique": True,
            },
            "receipt_identity": None,
        },
    )


async def _sqlite_kill(path: str, session_id: str) -> int:
    script = """import asyncio, json, os, sys
sys.path[:] = json.loads(sys.argv[3])
from qagents.session import SQLiteStore, Entry
async def run():
    async with SQLiteStore(sys.argv[1]) as store:
        await store.append(sys.argv[2], [Entry(id="kill-check", kind="custom")])
        print(json.dumps({"ack": "written", "pid": os.getpid()}), flush=True)
        await asyncio.to_thread(sys.stdin.buffer.read, 1)
asyncio.run(run())
"""
    # Windows SelectorEventLoop (required by psycopg) has no asyncio subprocess API.
    # Only creation is synchronous; pipe reads and process waits run on worker threads.
    process = subprocess.Popen(  # noqa: ASYNC220
        [
            getattr(sys, "_base_executable", sys.executable),
            "-c",
            script,
            path,
            session_id,
            json.dumps(sys.path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    identified = False
    try:
        line = await asyncio.wait_for(asyncio.to_thread(process.stdout.readline), 15)
        ready = json.loads(line)
        _require(ready.get("ack") == "written", "The subprocess did not acknowledge its write.")
        _require(
            ready.get("pid") == process.pid,
            "The launched process is not the SQLite writer; no kill result can be claimed.",
        )
        identified = True
    finally:
        if not identified and process.stdin is not None:
            # A launcher mismatch is not a valid crash test. EOF lets its writer
            # exit normally instead of leaving an unidentified descendant alive.
            process.stdin.close()
            process.stdin = None
        process.kill()
        await asyncio.to_thread(process.communicate, timeout=5)
    _require(process.returncode is not None, "The acknowledged SQLite writer did not exit.")
    return process.pid


async def check_store(
    factory: Callable[[], SessionStore | Awaitable[SessionStore]],
    *,
    kill: Callable[[], Awaitable[None]] | None = None,
) -> StoreReport:
    """Exercise a disposable backend; factory must reopen the same database.

    SQLite includes a real subprocess kill after an acknowledged append. Other
    stores can supply a deployment-specific kill callback, or the report names
    the durability check as untested. Probe journals and claims are retained.
    """
    made = factory()
    store = await made if inspect.isawaitable(made) else made
    report = StoreReport(ok=True)
    prefix = "store-check:" + uuid.uuid4().hex

    async def check(name, operation):
        try:
            await operation()
        except Exception as error:
            report.ok = False
            report.checks.append((name, False, f"{type(error).__name__}: {error}"))
        else:
            report.checks.append((name, True, "passed"))

    async def atomic():
        session = prefix + ":atomic"
        entries = [Entry(kind="custom", data={"i": n}) for n in range(3)]
        seqs = await store.append(session, entries)
        _require([e.seq for e in await store.load(session)] == [1, 2, 3], "Noncontiguous seq.")
        if "entry_ids" in store.features:
            _require(seqs == [1, 2, 3], "Append did not return positions.")
            _require(await store.append(session, entries) == [1, 2, 3], "Replay wrote new entries.")
            try:
                await store.append(session, [entries[0], Entry(kind="custom")])
            except EntryConflict:
                pass
            else:
                raise AssertionError("Partial overlap was accepted.")
        try:
            await store.append(
                session, [Entry(kind="custom"), Entry(kind="bad", data={"x": object()})]
            )
        except (SessionError, TypeError, ValueError):
            pass
        else:
            raise AssertionError("Non-JSON data was accepted.")
        _require(len(await store.load(session)) == 3, "Failed write published part of its batch.")

    async def leases():
        session = prefix + ":leases"
        _require(await store.acquire_lease(session, "one", 30), "Initial lease refused.")
        _require(not await store.acquire_lease(session, "two", 30), "Concurrent owner admitted.")
        batch = [Entry(kind="custom")]
        await store.append_leased(session, batch, "one")
        await store.release_lease(session, "one")
        _require(await store.acquire_lease(session, "two", 30), "Released lease not available.")
        if "entry_ids" in store.features:
            _require(
                await store.append_leased(session, batch, "one") == [1],
                "Duplicate ids were checked after the lease.",
            )
        try:
            await store.append_leased(session, [Entry(kind="custom")], "one")
        except LeaseError:
            pass
        else:
            raise AssertionError("Stale owner wrote a fresh entry.")
        _require(len(await store.load(session)) == 1, "Refused write changed the journal.")

    async def epochs():
        session = prefix + ":epochs"
        first = await store.acquire_lease_grant(session, "one", 0.03)
        second = await store.acquire_lease_grant(session, "one", 0.03)
        _require(first.epoch == second.epoch, "Same-token renewal changed epoch.")
        if store.takeover_grace_s > 0.05:
            await asyncio.sleep(0.04)
            _require(
                await store.acquire_lease_grant(session, "two", 30) is None,
                "Takeover ignored configured grace.",
            )
        await store.delete(session)
        next_grant = await store.acquire_lease_grant(session, "two", 30)
        _require(next_grant.epoch > first.epoch, "Delete reset epoch high-water.")

    async def clock():
        value = await store.now()
        _require(
            isinstance(value, (int, float)) and float("-inf") < value < float("inf"),
            "Store clock is not a finite timestamp.",
        )

    async def metadata():
        effect = prefix + ":effect"
        for key, schema, expected in (
            ("key", "schema", "registered"),
            ("other", "schema", "key_changed"),
            ("key", "other", "schema_changed"),
            ("key", "schema", "matches"),
        ):
            _require(
                await store.register_effect(effect, "1", key, schema) == expected,
                "Registry did not preserve the original contract.",
            )
        salt = await store.salt()
        _require(
            isinstance(salt, bytes) and len(salt) >= 16 and salt == await store.salt(),
            "Salt is absent, too short, or unstable.",
        )

    async def claims():
        session, other = prefix + ":claim", prefix + ":other"
        intent = _intent(prefix)
        try:
            await store.append(session, [intent])
        except SessionError:
            pass
        else:
            raise AssertionError("Unleased effect intent was accepted.")
        await store.acquire_lease(session, "owner", 30)
        await store.append_leased(session, [intent], "owner")
        claim = await store.get_claim(prefix, "store-check", "one")
        _require(claim is not None and claim.session_id == session, "Intent has no atomic claim.")
        await store.acquire_lease(other, "other", 30)
        try:
            await store.append_leased(other, [_intent(prefix)], "other")
        except ClaimConflict:
            pass
        else:
            raise AssertionError("Competing session took a live claim.")
        await store.delete(session, force=True)
        _require(
            await store.get_claim(prefix, "store-check", "one") is not None,
            "Delete removed claim tombstone.",
        )

    async def durability():
        session = prefix + ":kill"
        if kill is not None:
            await store.append(session, [Entry(id="kill-check", kind="custom")])
            await kill()
        else:
            await _sqlite_kill(store.path, session)
        made = factory()
        reopened = await made if inspect.isawaitable(made) else made
        try:
            _require(
                [e.id for e in await reopened.load(session)] == ["kill-check"],
                "Acknowledged entry lost after writer kill.",
            )
        finally:
            await reopened.aclose()

    try:
        await check("atomic append, ids and decided failure", atomic)
        for capability, name, operation in (
            ("leases", "lease fencing", leases),
            ("epochs", "epochs and grace", epochs),
            ("now", "store clock", clock),
            ("registry", "registry and salt", metadata),
            ("claims", "claims and tombstones", claims),
        ):
            if capability in store.features:
                await check(name, operation)
            else:
                report.not_checked.append(name)
        if kill is not None or isinstance(store, SQLiteStore) and store.path != ":memory:":
            await check("durability after killed writer", durability)
        else:
            report.not_checked.append("durability after killed writer")
        report.not_checked.append("replication durability")
    finally:
        await store.aclose()
    return report
