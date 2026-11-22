# Sessions and recovery

A session is a conversation that outlives a process.

```python
from qagents import Agent, Session, SQLiteStore

store = SQLiteStore("agents.db")
session = Session(store, id="user-42")

await agent.run("My name is Maria.", session=session)
await agent.run("What is my name?", session=session)  # sees the first exchange
```

`Session()` alone keeps the conversation in memory. A store and an id make it durable. The id is
yours to choose; include the tenant and the user in it when your application has several.

## The journal

A session is an append-only journal of JSON entries. The loop writes an entry for every message,
every tool start and every tool result at the moment it happens.

| Entry | Written when |
|---|---|
| `run_started` | a run begins |
| `message` | a message joins the conversation |
| `tool_started` | a tool call begins executing |
| `tool_result` | a tool call has a result |
| `agent_switched` | a handoff changes the active agent |
| `run_paused` | the run stops to wait for approvals |
| `run_finished` | the run ends (`completed`, `limit_reached`, `cancelled`, `abandoned`) |
| `run_failed` | the run raised; it stays open and can be resumed |
| `compaction` | the history sent to the model was replaced by a shorter one |
| `workflow_step` | a workflow step finished |

The current state is derived by reading the journal: `await session.state()` returns the
messages, the open run if there is one, recorded tool results and pending approvals. Nothing is
pickled, and nothing depends on the shape of your code, so a run can be resumed by a different
process, after a deploy, or after the agent's tools changed.

```python
for entry in await session.entries():
    print(entry.seq, entry.kind, entry.run_id)
```

## Resuming

`agent.resume(session)` continues the session's unfinished run: after a
[pause for approval](permissions.md#pause-and-resume), a crash, or a provider error.

```python
state = await session.state()
if state.open_run_id is not None:
    result = await agent.resume(session)
```

### What is guaranteed

Stated plainly, because the details matter when tools have side effects:

- **A tool call with a recorded result never runs again.** Its result is read from the journal.
- **A tool call that had started when the process died** has an unknown outcome. It is re-run
  only if the tool is declared `read_only` or `idempotent`. Otherwise the model is told that the
  outcome is unknown and that it should check before repeating.
- **A tool call that had not started** runs normally, after the usual permission check.
- **A model request that was in flight** is sent again. Its first response was lost, so this
  costs tokens, but no tool had acted on it. The model's response is always journaled before any
  tool it requested starts.
- **Delivery is at-least-once**, as in every agent framework. Exactly-once needs cooperation from
  the system your tool calls: see below.

`agent.resume` continues a run. To recover interrupted runs automatically, run
`AgentRuntime` and call `recover()`; it uses leases and heartbeats to avoid taking an active
worker's session. Workflow sessions are recovered by resuming their workflow instead.
Without a runtime, resume from your own job runner or on the user's next request.

### Idempotency

Every tool call has `ctx.operation_id`. It is the same when the call is repeated after a resume,
and different for every other call. Pass it to the system the tool talks to:

```python
@tool(idempotent=True)
async def charge_card(ctx: RunContext[Deps], amount_cents: int) -> str:
    """Charge the customer's saved card."""
    charge = await ctx.deps.payments.charge(
        amount=amount_cents,
        idempotency_key=ctx.operation_id,  # a repeat returns the first charge
    )
    return f"Charged, receipt {charge.receipt}"
```

Declaring the tool `idempotent=True` is your promise that a repeat is safe; with the key above it
is.

### A new message while tool calls are open

If the user sends a new message while a run is paused or was interrupted, `agent.run` closes the
open tool calls with a result that says they were not run (or that their outcome is unknown), and
marks the old run `abandoned`. The conversation stays valid for the provider, and the model knows
what did not happen.

## Stores

| Store | Use it for |
|---|---|
| `InMemoryStore` | tests, and sessions that need not survive a restart |
| `SQLiteStore(path)` | one host; WAL mode, calls run on worker threads |
| `PostgresStore(dsn)` | sessions shared by workers on several hosts; install `qagents[postgres]` |

A basic store implements three methods. A custom backend can implement this interface:

```python
from qagents.session import Entry, SessionStore


class MyStore(SessionStore):
    async def append(self, session_id: str, entries: list[Entry]) -> None:
        """Durably add entries to the end of the session, in order."""

    async def load(self, session_id: str) -> list[Entry]:
        """Return every entry of the session, oldest first."""

    async def delete(self, session_id: str) -> None:
        """Remove the session."""
```

`append` must be atomic and must preserve order. Entries are plain JSON (`entry.model_dump()`).

The built-in stores also support leases for runtime workers; a custom store needs the lease
methods to support `AgentRuntime`. `PostgresStore` opens a connection pool on first use
and keeps its tables in a separate schema.

!!! note "Journal snapshots"
    State is derived by reading the whole session journal. Long-lived sessions with thousands
    of entries will want snapshots, which are not implemented yet.
