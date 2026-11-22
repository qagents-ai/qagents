# The loop

An agent is a model, instructions and tools, run in a loop until the model answers.

## One rule decides the next step

Before every step the loop looks at the last message of the conversation:

| Last message | Next step |
|---|---|
| assistant message with tool calls that have no results | run the tools |
| assistant message without tool calls | that is the final answer |
| anything else (a user message, tool results) | ask the model |

The next step depends only on the conversation, and the conversation is written to the
[session journal](sessions.md) as it grows. That is why a run can stop anywhere and continue
later: the loop that resumes is the same loop, reading the same history.

```
run_started
  turn 1:  model request  ->  assistant message (maybe with tool calls)
           tool calls     ->  tool results
  turn 2:  model request  ->  assistant message
run_finished
```

## What a run returns

`agent.run()` returns a [`RunResult`](../reference/events.md). Check `status` before `output`:

| `status` | Meaning | What to do |
|---|---|---|
| `completed` | The agent answered. `output` is the text, or the validated `output_type` value. | Use it. |
| `paused` | Tool calls in `pending_approvals` wait for a decision. | [`agent.resume`](permissions.md#pause-and-resume) |
| `limit_reached` | A limit stopped the run; `stop_reason` names it. | Report it, or continue with a new `run` on the session. |
| `cancelled` | `RunStream.cancel()` was called. | Continue with a new `run` on the session. |

Failures are exceptions, not statuses: a provider error after retries (`ModelError`), a hook that
raised, or a final answer that never matched `output_type` (`OutputValidationError`). The session
keeps everything that happened before the failure, so `agent.resume(session)` continues the run.

`result.messages` is the whole conversation, `result.new_messages` only what this run added,
`result.usage` the tokens and cost, and `result.session` the session to pass to the next call.

## Limits

Every run has hard stops. The defaults are deliberately not "unlimited".

```python
from qagents import Agent, Limits

agent = Agent(
    "qcloud:<model-id>",
    limits=Limits(max_turns=12, max_total_tokens=200_000, max_cost_usd=0.50, timeout_s=300),
)
result = await agent.run("...", limits=Limits(max_turns=3))  # per-run override
```

| Limit | Default | Checked |
|---|---|---|
| `max_turns` | 30 | before each model request |
| `max_total_tokens` | off | before each model request |
| `max_cost_usd` | off | before each model request; needs a model that reports cost |
| `timeout_s` | off | continuously: cancels the request or tool in flight |
| `tool_timeout_s` | 120 | per tool call, unless the tool sets its own `timeout` |
| `output_retries` | 2 | extra attempts when the answer does not match `output_type` |
| `model_retries` | 2 | extra attempts for model errors that are safe to repeat |

A run stopped by a limit has `status == "limit_reached"` and `stop_reason` set to `max_turns`,
`max_total_tokens`, `max_cost_usd` or `timeout`. Its history is still valid. To let it go on,
start a run on the same session without a prompt and with a higher limit:

```python
result = await agent.run("Research the topic", session=session, limits=Limits(max_turns=5))
if result.stop_reason == "max_turns":
    result = await agent.run(session=session, limits=Limits(max_turns=20))
```

## Retries and billing

Model errors carry two flags. `retryable` means the request could succeed if sent again.
`ambiguous` means the provider may already have processed, and billed, the request: a timeout
after sending, a dropped connection, a 5xx, a stream that ended early.

The loop repeats a request only when it is `retryable` and not `ambiguous` (a 429 is the typical
case), and never after part of the answer was already streamed to you. To also repeat ambiguous
requests, set `Limits(retry_ambiguous=True)` and accept the possible double charge.

## Instructions

`instructions` is fixed text, or a function of the run context:

```python
async def instructions(ctx: RunContext[Deps]) -> str:
    return f"You are helping {ctx.deps.user_name}. Today is {ctx.deps.today}."


agent = Agent("qcloud:<model-id>", instructions=instructions)
await agent.run("...", deps=Deps(user_name="Maria", today="2026-11-11"))
```

The instructions are sent as the system message of every request and are not stored in the
conversation, so each agent in a [handoff](multi-agent.md#handoffs) applies its own.

## Settings

`ModelSettings` holds sampling and tool-choice settings; unset fields are not sent.

```python
from qagents import ModelSettings

agent = Agent("qcloud:<model-id>", settings=ModelSettings(temperature=0.2, max_tokens=800))
await agent.run("...", settings=ModelSettings(max_tokens=200))  # merged over the agent's
```

`ModelSettings(extra={...})` passes provider-specific request fields as they are. QCloud
publishes an allow-list, so an unknown field fails when the agent is built.
