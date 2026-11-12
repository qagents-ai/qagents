# Hooks, context and tracing

## Hooks

Hooks put your code at fixed points of the loop. Subclass `Hooks` and override what you need.

```
on_run_start
repeat:
    before_model  ->  model request  ->  after_model
    for each tool call:
        before_tool  ->  tool  ->  after_tool
on_run_end
```

```python
from qagents import Agent, Hooks, ToolResultPart


class Audit(Hooks):
    async def before_model(self, ctx, request):
        log.info("run=%s turn=%s messages=%d", ctx.run_id, ctx.turn, len(request.messages))

    async def before_tool(self, ctx, call):
        if call.name == "send_email" and not call.arguments["to"].endswith("@example.com"):
            return ToolResultPart(
                tool_call_id=call.id,
                name=call.name,
                is_error=True,
                content="Emails can only be sent to example.com addresses.",
            )

    async def after_tool(self, ctx, call, result):
        return result.model_copy(update={"content": redact(result.content)})


agent = Agent("qcloud:<model-id>", hooks=[Audit()])
```

| Hook | Receives | May return |
|---|---|---|
| `on_run_start(ctx)` | the run context | nothing |
| `before_model(ctx, request)` | the exact request about to be sent | a replacement request |
| `after_model(ctx, response)` | the model's response | a replacement response |
| `before_tool(ctx, call)` | an authorised tool call | a changed call, or a result to skip the tool |
| `after_tool(ctx, call, result)` | the tool's result | a replacement result |
| `on_event(ctx, event)` | every event of the run | nothing |
| `on_run_end(ctx, result, error)` | the result, or the exception | nothing |

Rules:

- Returning `None` changes nothing.
- Several hooks run in the order given; each sees what the previous one returned.
- **A hook that raises stops the run with that exception.** Hooks are part of your control path,
  so a broken one fails closed instead of being skipped.
- `before_tool` runs after the [permission check](permissions.md). Use permissions to decide
  whether a call may run, and hooks to observe or adjust calls that may.

`before_model` is also the place to verify what is sent: the request it receives is complete,
including the system message and tool definitions.

## Long conversations

Each model request carries the whole history, so long runs eventually fill the context window.
Two mechanisms keep them in bounds.

**Tool results are capped.** A result longer than `Limits.max_tool_result_chars` keeps its start
and end, with a marker in the middle saying how much was omitted.

**Compaction** replaces the older part of the history with a summary:

```python
from qagents import Agent, SummarizingCompactor

agent = Agent(
    "qcloud:<model-id>",
    compactor=SummarizingCompactor(trigger_tokens=60_000, keep_last=8),
)
```

When the next request would hold more than `trigger_tokens` tokens (or a `trigger_fraction`
of the model's context window), everything except the last
`keep_last` messages is summarised by a model request (the agent's own model, or `model=` for a
cheaper one), and the summary takes its place. A tool call is never separated from its result.
The summarising prompt is `qagents.prompts.SUMMARY_REQUEST`.

Compaction changes what is sent, not what happened: the original messages stay in the session
journal, and a `compaction` entry records the replacement. For your own strategy, subclass
`Compactor` and return a shorter list of messages, or `None` to keep the history.

## Tracing

`OTelHooks` turns a run into OpenTelemetry spans.

```python
from qagents.tracing import OTelHooks

agent = Agent("qcloud:<model-id>", hooks=[OTelHooks()])
```

```
invoke_agent <agent>
  chat <model>            one per model request: tokens, cost, finish reason
  execute_tool <tool>     one per tool call: error status when the tool failed
```

It uses the tracer provider your application configured, so the spans go wherever your other
spans go. Any backend that accepts OTLP works (Langfuse, Phoenix, LangSmith, Logfire, Grafana,
Datadog). Install with `pip install "qagents[otel]"`.

- Attribute names follow the OpenTelemetry GenAI semantic conventions (`gen_ai.*`). Those are
  still marked "Development" upstream and may be renamed; they are kept in one dictionary,
  `qagents.tracing.ATTR`.
- Message and tool content is **not** recorded unless you pass `record_content=True`, because it
  often contains personal data.
- A run that is cancelled as an asyncio task, or a stream closed early, still reaches
  `on_run_end`, with the `CancelledError` or `GeneratorExit` as `error`, so its root span is
  closed.
- Error text (a failed tool's status description, exception messages and stack traces) counts
  as content: without `record_content=True` a span records only the exception's type.
