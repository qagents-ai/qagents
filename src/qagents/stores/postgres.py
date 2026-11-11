"""Session journals in PostgreSQL, shared by workers on several hosts.

    store = PostgresStore("postgresql://agents@db.internal/agents")
    await agent.run("Hello", session=Session(store, id="user-42"))

Needs psycopg 3 and psycopg-pool (``pip install "qagents[postgres]"``). The
store opens a connection pool on first use and creates its tables if they do
not exist: ``entries`` (one row per journal entry, ``data`` as ``jsonb``,
primary key ``(session_id, seq)``) and ``sessions`` (run status and lease per
session), in their own schema, ``qagents`` by default.

What it guarantees:

- `append` is one transaction that locks the session's ``sessions`` row, so
  concurrent writers to one session are serialised: entries keep their
  order and ``seq`` is gap-free per session. The run status is updated in
  the same transaction.
- `append_leased` checks the lease under that same row lock, so a worker
  whose lease was taken over cannot write any more.
- Leases and ``updated_at`` use the database server's clock
  (``clock_timestamp()``), so clock skew between workers does not matter.
  `SessionStatus.lease_expires_at` is in that clock too. ``Entry.ts`` is
  still set by the process that created the entry.

Two properties of ``jsonb`` to know about: it does not keep the order of
object keys (a loaded entry compares equal, but its keys may come back in
another order), and it cannot store the NUL character (``\\u0000``); an
entry that contains one is refused with a `SessionError` that says so.

On Windows, psycopg's async mode needs a selector event loop instead of
asyncio's default proactor loop:

    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(main())
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ..errors import ConfigurationError, SessionError
from ..session import (
    Entry,
    LeaseError,
    SessionStatus,
    SessionStore,
    _check_limit,
    _check_ttl,
    _lease_lost,
    _status_filter,
    _status_row,
    advance_status,
)

if TYPE_CHECKING:
    from psycopg import AsyncConnection
    from psycopg_pool import AsyncConnectionPool

DSN_ENV = "QAGENTS_POSTGRES_DSN"
"""Environment variable read for the connection string when ``dsn`` is not given."""

SCHEMA_VERSION = 1
"""Version of the tables this module creates, kept in ``<schema>.schema_version``."""

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}")
_STATUS_COLUMNS = (
    "session_id, status, open_run_id, agent, failures,"
    " extract(epoch FROM updated_at)::float8, lease_owner,"
    " extract(epoch FROM lease_expires_at)::float8"
)


class PostgresStore(SessionStore):
    """Journals, run status and leases in PostgreSQL.

    Args:
        dsn: A libpq connection string or URI. Defaults to the environment
            variable ``QAGENTS_POSTGRES_DSN``; when that is unset too, libpq's
            own defaults and ``PG*`` variables apply.
        pool_size: Most connections the pool opens. Each `append` holds one
            for the length of its transaction.
        schema: Schema for the tables; created when missing. Use different
            schemas to keep separate applications apart in one database.
        connect_timeout_s: Seconds to wait for the first connection before
            failing with a `SessionError`.

    Raises:
        ConfigurationError: psycopg or psycopg-pool is not installed, or an
            argument is invalid.
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        pool_size: int = 10,
        schema: str = "qagents",
        connect_timeout_s: float = 30.0,
    ) -> None:
        if not _IDENTIFIER.fullmatch(schema):
            raise ConfigurationError(
                f"schema={schema!r} is not a plain identifier. Use letters, digits and "
                "underscores, starting with a letter or underscore, at most 63 characters."
            )
        if pool_size < 1:
            raise ConfigurationError(f"pool_size must be at least 1, not {pool_size}.")
        try:
            import psycopg  # noqa: F401
            import psycopg_pool  # noqa: F401
        except ImportError as error:
            raise ConfigurationError(
                "PostgresStore needs psycopg 3 and psycopg-pool. Install them with "
                f"pip install 'qagents[postgres]'. The import failed with: {error}"
            ) from error
        self.schema = schema
        self.pool_size = pool_size
        self.connect_timeout_s = connect_timeout_s
        self._dsn = dsn if dsn is not None else os.environ.get(DSN_ENV, "")
        self._pool: AsyncConnectionPool | None = None
        self._opening: asyncio.Lock | None = None
        self._q = f'"{schema}"'

    def __repr__(self) -> str:  # never show the DSN: it may hold a password
        return f"PostgresStore(schema={self.schema!r}, pool_size={self.pool_size})"

    # -- connection pool -------------------------------------------------------------

    async def _ready(self) -> AsyncConnectionPool:
        if self._pool is None:
            if self._opening is None:
                self._opening = asyncio.Lock()
            async with self._opening:
                if self._pool is None:
                    self._pool = await self._open()
        return self._pool

    async def _open(self) -> AsyncConnectionPool:
        from psycopg_pool import AsyncConnectionPool, PoolTimeout

        _check_event_loop()
        pool = AsyncConnectionPool(
            self._dsn,
            min_size=1,
            max_size=self.pool_size,
            kwargs={"autocommit": True},
            open=False,
            name=f"qagents-{self.schema}",
        )
        try:
            await pool.open(wait=True, timeout=self.connect_timeout_s)
            async with pool.connection() as connection:
                await self._create_schema(connection)
        except BaseException as error:
            await pool.close()
            if isinstance(error, PoolTimeout):
                raise SessionError(
                    f"Could not connect to PostgreSQL within {self.connect_timeout_s:g} s. "
                    f"Check the connection string (dsn= or {DSN_ENV}) and that the server "
                    "accepts connections."
                ) from error
            raise
        return pool

    async def _create_schema(self, connection: AsyncConnection[Any]) -> None:
        q = self._q
        async with connection.transaction():
            # Workers starting together must not race on CREATE TABLE.
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))", (f"qagents.schema:{self.schema}",)
            )
            await connection.execute(f"CREATE SCHEMA IF NOT EXISTS {q}")
            await connection.execute(
                f"CREATE TABLE IF NOT EXISTS {q}.schema_version (version integer NOT NULL)"
            )
            cursor = await connection.execute(f"SELECT max(version) FROM {q}.schema_version")
            row = await cursor.fetchone()
            version = row[0] if row is not None else None
            if version is not None and version > SCHEMA_VERSION:
                raise SessionError(
                    f"The PostgreSQL schema {self.schema!r} was created by a newer qagents "
                    f"(schema version {version}; this version reads up to {SCHEMA_VERSION}). "
                    "Upgrade qagents."
                )
            if version is not None:
                return
            await connection.execute(
                f"CREATE TABLE IF NOT EXISTS {q}.entries ("
                " session_id text NOT NULL, seq bigint NOT NULL, ts double precision NOT NULL,"
                " run_id text, kind text NOT NULL, data jsonb NOT NULL,"
                " PRIMARY KEY (session_id, seq))"
            )
            await connection.execute(
                f"CREATE TABLE IF NOT EXISTS {q}.sessions ("
                " session_id text PRIMARY KEY, status text NOT NULL DEFAULT 'idle',"
                " open_run_id text, agent text, failures integer NOT NULL DEFAULT 0,"
                " last_seq bigint NOT NULL DEFAULT 0, updated_at timestamptz,"
                " lease_owner text, lease_expires_at timestamptz)"
            )
            await connection.execute(
                f"CREATE INDEX IF NOT EXISTS sessions_by_status"
                f" ON {q}.sessions (status, updated_at)"
            )
            await connection.execute(
                f"INSERT INTO {q}.schema_version (version) VALUES (%s)", (SCHEMA_VERSION,)
            )

    async def aclose(self) -> None:
        """Close the connection pool. The store opens a new one if it is used again."""
        if self._pool is not None:
            pool, self._pool = self._pool, None
            await pool.close()

    # -- journal ---------------------------------------------------------------------

    async def append(self, session_id: str, entries: list[Entry]) -> None:
        await self._append(session_id, entries, None)

    async def append_leased(self, session_id: str, entries: list[Entry], owner: str) -> None:
        await self._append(session_id, entries, owner)

    async def _append(self, session_id: str, entries: list[Entry], owner: str | None) -> None:
        if not entries:
            return
        import psycopg

        documents = [json.dumps(entry.data, ensure_ascii=False) for entry in entries]
        q = self._q
        pool = await self._ready()
        async with pool.connection() as connection:
            try:
                async with connection.transaction():
                    await connection.execute(
                        f"INSERT INTO {q}.sessions (session_id) VALUES (%s)"
                        " ON CONFLICT (session_id) DO NOTHING",
                        (session_id,),
                    )
                    cursor = await connection.execute(
                        "SELECT status, open_run_id, agent, failures, last_seq, lease_owner"
                        f" FROM {q}.sessions WHERE session_id = %s FOR UPDATE",
                        (session_id,),
                    )
                    found = await cursor.fetchone()
                    assert found is not None  # inserted above, and rows are never deleted here
                    if owner is not None and found[5] != owner:
                        raise LeaseError(_lease_lost(session_id, owner, found[5]))
                    status = SessionStatus(
                        session_id=session_id,
                        status=found[0],
                        open_run_id=found[1],
                        agent=found[2],
                        failures=found[3],
                    )
                    last = found[4]
                    rows = []
                    for offset, (entry, document) in enumerate(
                        zip(entries, documents, strict=True), start=1
                    ):
                        rows.append(
                            (
                                session_id,
                                last + offset,
                                entry.ts,
                                entry.run_id,
                                entry.kind,
                                document,
                            )
                        )
                        advance_status(status, entry)
                    async with connection.cursor() as insert:
                        await insert.executemany(
                            f"INSERT INTO {q}.entries (session_id, seq, ts, run_id, kind, data)"
                            " VALUES (%s, %s, %s, %s, %s, %s::jsonb)",
                            rows,
                        )
                    await connection.execute(
                        f"UPDATE {q}.sessions SET status = %s, open_run_id = %s, agent = %s,"
                        " failures = %s, last_seq = %s, updated_at = clock_timestamp()"
                        " WHERE session_id = %s",
                        (
                            status.status,
                            status.open_run_id,
                            status.agent,
                            status.failures,
                            last + len(entries),
                            session_id,
                        ),
                    )
            except psycopg.errors.UntranslatableCharacter as error:
                kinds = sorted({entry.kind for entry in entries})
                raise SessionError(
                    f"PostgreSQL jsonb cannot store the NUL character (\\u0000), which appears "
                    f"in a {', '.join(kinds)} entry of session {session_id!r}. Nothing was "
                    "written. Keep NUL characters out of messages and tool results, for "
                    "example by returning binary data as base64."
                ) from error

    async def load(self, session_id: str) -> list[Entry]:
        pool = await self._ready()
        async with pool.connection() as connection:
            cursor = await connection.execute(
                f"SELECT seq, ts, run_id, kind, data FROM {self._q}.entries"
                " WHERE session_id = %s ORDER BY seq",
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [
            Entry(seq=seq, ts=ts, run_id=run_id, kind=kind, data=data)
            for seq, ts, run_id, kind, data in rows
        ]

    async def delete(self, session_id: str) -> None:
        pool = await self._ready()
        async with pool.connection() as connection, connection.transaction():
            await connection.execute(
                f"DELETE FROM {self._q}.entries WHERE session_id = %s", (session_id,)
            )
            await connection.execute(
                f"DELETE FROM {self._q}.sessions WHERE session_id = %s", (session_id,)
            )

    # -- run status ------------------------------------------------------------------

    async def status(self, session_id: str) -> SessionStatus:
        pool = await self._ready()
        async with pool.connection() as connection:
            cursor = await connection.execute(
                f"SELECT {_STATUS_COLUMNS} FROM {self._q}.sessions WHERE session_id = %s",
                (session_id,),
            )
            found = await cursor.fetchone()
        return _status_row(found) if found is not None else SessionStatus(session_id=session_id)

    async def list_sessions(
        self,
        status: str | Iterable[str] | None = None,
        prefix: str | None = None,
        limit: int | None = None,
    ) -> list[SessionStatus]:
        wanted = _status_filter(status)
        _check_limit(limit)
        clauses, params = ["updated_at IS NOT NULL"], []
        if wanted is not None:
            clauses.append("status = ANY(%s)")
            params.append(sorted(wanted))
        if prefix is not None:
            clauses.append("starts_with(session_id, %s)")
            params.append(prefix)
        query = (
            f"SELECT {_STATUS_COLUMNS} FROM {self._q}.sessions WHERE {' AND '.join(clauses)}"
            " ORDER BY updated_at, session_id"
        )
        if limit is not None:
            query += " LIMIT %s"
            params.append(limit)
        pool = await self._ready()
        async with pool.connection() as connection:
            cursor = await connection.execute(query, params)
            rows = await cursor.fetchall()
        return [_status_row(row) for row in rows]

    # -- leases ----------------------------------------------------------------------

    async def acquire_lease(self, session_id: str, owner: str, ttl_s: float) -> bool:
        _check_ttl(ttl_s)
        pool = await self._ready()
        async with pool.connection() as connection:
            cursor = await connection.execute(
                f"INSERT INTO {self._q}.sessions AS s (session_id, lease_owner, lease_expires_at)"
                " VALUES (%s, %s, clock_timestamp() + make_interval(secs => %s))"
                " ON CONFLICT (session_id) DO UPDATE SET lease_owner = excluded.lease_owner,"
                " lease_expires_at = excluded.lease_expires_at"
                " WHERE s.lease_owner IS NULL OR s.lease_owner = excluded.lease_owner"
                " OR s.lease_expires_at <= clock_timestamp()",
                (session_id, owner, float(ttl_s)),
            )
            return cursor.rowcount == 1

    async def renew_lease(self, session_id: str, owner: str, ttl_s: float) -> bool:
        _check_ttl(ttl_s)
        pool = await self._ready()
        async with pool.connection() as connection:
            cursor = await connection.execute(
                f"UPDATE {self._q}.sessions"
                " SET lease_expires_at = clock_timestamp() + make_interval(secs => %s)"
                " WHERE session_id = %s AND lease_owner = %s",
                (float(ttl_s), session_id, owner),
            )
            return cursor.rowcount == 1

    async def release_lease(self, session_id: str, owner: str) -> None:
        pool = await self._ready()
        async with pool.connection() as connection:
            await connection.execute(
                f"UPDATE {self._q}.sessions SET lease_owner = NULL, lease_expires_at = NULL"
                " WHERE session_id = %s AND lease_owner = %s",
                (session_id, owner),
            )


def _check_event_loop() -> None:
    if sys.platform != "win32":
        return
    if isinstance(asyncio.get_running_loop(), asyncio.ProactorEventLoop):
        raise ConfigurationError(
            "psycopg's async mode cannot run on the ProactorEventLoop that asyncio uses on "
            "Windows by default. Run your program on a selector loop instead: "
            "with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner: "
            "runner.run(main())"
        )
