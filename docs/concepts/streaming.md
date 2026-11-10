# Streaming and cancel

`agent.stream()` starts a run and returns a `RunStream`: an async iterator of events, plus
control over the run.

```python
stream = agent.stream("Tell me about Athens")
async for event in stream:
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
result = stream.result  # the same RunResult agent.run() returns
```

For the answer text only, use `stream.text()`:

```python
async for piece in agent.stream("Tell me about Athens").text():
    print(piece, end="", flush=True)
```

## Events

Every event is a pydantic model with a `type` field.

| `type` | When | Useful fields |
|---|---|---|
| `run_started` | the run begins | `run_id`, `session_id`, `agent`, `resumed` |
| `turn_started` | a model request is about to be sent | `turn`, `agent` |
| `text_delta` | the model wrote some answer text | `text` |
| `reasoning_delta` | the model exposed some reasoning | `text` |
| `tool_call_delta` | the model is writing a tool call | `index`, `name`, `arguments_delta` |
| `model_responded` | one model response is complete | `response` (message, usage, finish reason) |
| `tool_call_started` | a tool begins executing | `call` |
| `tool_call_finished` | a tool call has a result: ran, failed, refused or cancelled | `call`, `result`, `duration_ms` |
| `approval_required` | a tool call waits for a human | `approval` |
| `agent_switched` | a handoff changed the active agent | `from_agent`, `to_agent` |
| `subagent_event` | an event from an agent running as a tool | `agent`, `tool_call_id`, `event` |
| `run_finished` | always last, unless the run raised | `result` |

Events serialise to JSON as they are, which makes forwarding them to a browser a one-liner:

```python
async for event in agent.stream(prompt, session=session):
    await websocket.send_text(event.model_dump_json())
```

See [serving over HTTP](../serving.md) for a complete server-sent-events endpoint.

`Hooks.on_event` receives the same events, for logging or metrics without consuming the stream.

## Cancel

```python
stream = agent.stream("Write a long story")
async for event in stream:
    if user_pressed_stop():
        stream.cancel()  # keep iterating: run_finished still arrives
print(stream.result.status)  # "cancelled"
```

`cancel()` stops the run at the next chunk or tool boundary:

- The connection to the model is closed, which makes QCloud abort the upstream request.
- The text produced so far is kept in the conversation as an assistant message with
  `metadata == {"interrupted": True}`.
- Tool calls that were running are cancelled and get a result saying so, so the history stays
  valid.
- The next `agent.run` on that session continues from the interrupted answer.

`Limits(timeout_s=...)` uses the same mechanism and ends the run as `limit_reached` with
`stop_reason == "timeout"`.

## Real-time and voice

For a voice agent the path above is the barge-in path: when the user starts speaking, call
`stream.cancel("barge_in")`, then start a new run with what they said. The model already knows
what it managed to say, because the partial answer is in the history.

What qagents gives a real-time pipeline: token-level streaming with no buffering in the loop,
cancellation that reaches the provider, and time to first token measurable from `turn_started`
to the first `text_delta`. What it does not include: speech-to-text, text-to-speech, audio
transports, and turn detection. Those belong to your media stack.

Two things to keep off the hot path: approvals that wait for a human (use permission rules that
decide immediately), and slow `before_model` hooks.
