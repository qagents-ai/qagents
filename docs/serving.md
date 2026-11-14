# Serving over HTTP

qagents is a library: it runs inside your web service, worker or script. This page shows the
usual integration, a streaming chat endpoint with human approvals, using FastAPI.

The example below is `examples/serve_fastapi.py`, and it is exercised by the test suite.

```python title="examples/serve_fastapi.py"
--8<-- "examples/serve_fastapi.py"
```

## How the pieces map

| HTTP | qagents |
|---|---|
| `POST /chat` with a session id and a message | `agent.stream(message, session=Session(store, id=...))` |
| the response body | one server-sent event per run event, JSON-encoded with `event.model_dump_json()` |
| the client disconnects | the stream is closed, which stops the run where it is |
| `GET /sessions/{id}/pending` | `(await session.state()).pending_approvals` |
| `POST /approve` | `agent.resume_stream(session, approvals=...)` |

The server keeps nothing in memory between requests. Each request builds a `Session` over the
shared store, so any replica can serve any request, and a paused run can be approved by a
different process than the one that started it.

## A browser client

A minimal sketch of the receiving side. Unlike the server above, it is not part of the test suite.

```javascript
const response = await fetch("/chat", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({session_id: "s-1", message: "Email Maria about the meeting"}),
});
const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
let buffer = "";
for (;;) {
  const {value, done} = await reader.read();
  if (done) break;
  buffer += value;
  const messages = buffer.split("\n\n");
  buffer = messages.pop();
  for (const message of messages) {
    const event = JSON.parse(message.split("\ndata: ")[1]);
    if (event.type === "text_delta") appendText(event.text);
    if (event.type === "approval_required") showApprovalDialog(event.approval);
  }
}
```

## Before production

- **Authenticate and authorise.** Check who is calling, and that the session id belongs to them,
  before reading or resuming a session.
- **Sign approvals.** Set `Permissions(approval_secret=...)` from your secret store, the same on
  every worker. It signs paused calls, so `resume` accepts only answers that carry their token
  ([signed pauses](concepts/permissions.md#signed-pauses)). For effects tools that ask for
  approval it also lets an effect whose outcome is unknown be re-sent under its decision; see
  [Effects tools that ask for approval](guides/recover-effect.md#effects-tools-that-ask-for-approval).
- **One model client per process.** Build the agent, and its `QCloudModel`, once at startup.
- **Choose the store.** `SQLiteStore` suits one host. Several replicas need a shared store; a
  Postgres store is on the [roadmap](design.md#roadmap), and the interface is
  [three methods](concepts/sessions.md#stores).
- **Set limits.** `Limits(max_turns=..., max_cost_usd=..., timeout_s=...)` on every agent that
  faces users.
- **Disconnects leave a run open.** When a client drops mid-run, the next message on that session
  closes the open tool calls and carries on. To finish the work regardless of the client, run the
  agent in a background task and stream from a queue instead.
- **Long-running work** belongs in a worker, not in a request: use a
  [workflow](concepts/multi-agent.md#workflows) driven by your job queue.
