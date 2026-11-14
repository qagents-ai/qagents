# Several agents and workflows

There are three ways to combine qagents agents, and a fourth for another agent program. Pick by
who should control the flow.

| Pattern | Who decides what happens next | Context |
|---|---|---|
| [Sub-agent](#sub-agents) | the calling agent's model | the sub-agent starts fresh and returns one answer |
| [Handoff](#handoffs) | the current agent's model, once | the next agent continues the same conversation |
| [Workflow](#workflows) | your code | each agent step has its own session |
| [External agent](external-agents.md) | the calling agent's model | another agent program with its own context, tools and model; returns its final message and a footer |

Start with a single orchestrator. Give specialist workers separate context and let one owner
write each shared resource.

## Sub-agents

Pass an agent in another agent's `tools`. It becomes a tool with one argument, `task`.

```python
researcher = Agent(
    "qcloud:<model-id>",
    name="researcher",
    description="Finds facts in the knowledge base and cites them.",
    instructions="Research thoroughly. Answer with facts and sources only.",
    tools=[search],
)
lead = Agent("qcloud:<model-id>", name="lead", tools=[researcher])
result = await lead.run("How fast did solar grow in 2025?")
```

- The sub-agent sees only the task text, not the caller's conversation. The tool description
  tells the calling model to make the task self-contained.
- The caller sees only the sub-agent's final answer, so long tool output stays out of its context.
- The sub-agent's usage is added to the caller's `result.usage`.
- Its events arrive on the caller's stream as `subagent_event`.
- If the sub-agent stops early (a limit, a pause), the caller's model gets an error result.

`agent.as_tool(name=..., description=..., read_only=True)` gives you the tool explicitly, for
example to let several read-only sub-agents run in parallel.

!!! note "Limits and approvals of sub-agents"
    A sub-agent cannot pause the caller for approval. Give sub-agents permission rules that
    decide immediately, or an `on_ask` handler; a child whose permissions have none uses the
    caller's `on_ask`, and the caller's hooks with `propagate = True` run in it too. A child's
    token, cost and time budgets are capped at what remains of the caller's, and children that
    run at once share it live. Runs nest at most `Limits.max_depth` levels (5 by default, the
    lowest value along the chain applies); a deeper child is not started, and the caller's
    model gets `prompts.SUBAGENT_TOO_DEEP`.

## Handoffs

A handoff transfers the conversation to another agent, which answers the user directly.

```python
billing = Agent(
    "qcloud:<model-id>",
    name="billing",
    description="Handles invoices, payments and refunds.",
    instructions="You are the billing specialist.",
    tools=[get_invoice, issue_refund],
)
triage = Agent(
    "qcloud:<model-id>",
    name="triage",
    instructions="Route the customer to the right specialist.",
    handoffs=[billing],
)
result = await triage.run("How much is my invoice?", session=session)
print(result.agent)  # "billing"
```

The triage model sees a tool named `transfer_to_billing`. When it calls it, the billing agent's
model, instructions, tools and `output_type` take over, on the same history. `result.agent` names
the agent that was active when the run stopped; a later `agent.resume(session)` continues with
that agent.

The agent you called `run` on still owns the run's `limits`, `permissions` and `hooks`. For the
next user message, call `run` on whichever agent should start: `billing.run(...)` to stay with
the specialist, `triage.run(...)` to route again.

## Workflows

When your code should decide the order, write a normal async function and wrap each unit of work
in a `Workflow` step.

```python
from qagents import Workflow


async def weekly_report(topic: str, store) -> str:
    wf = Workflow(store, id=f"report:{topic}:2026-W46")
    notes = await wf.run_agent("research", researcher, f"Research {topic}")
    draft = await wf.run_agent("write", writer, f"Write a report from these notes:\n{notes.text}")
    await wf.run_agent("publish", publisher, draft.text)
    return draft.text
```

External changes belong in durable agent tools. Pass `ctx.operation_id` to a service that
enforces an idempotency key; set `idempotent=True` only with that guarantee. Otherwise keep
the tool non-idempotent and reconcile an unknown outcome before trying again. An unfinished
plain `Workflow.step` may execute again after a crash.


- `wf.step(name, function)` records a pure or read-only function's result. When the same workflow
  id runs again, a finished step returns the recorded result; an unfinished function may repeat.
- `wf.run_agent(name, agent, prompt)` runs an agent in its own durable session. If the step was
  interrupted, the agent run is resumed at the exact tool call rather than restarted.
- Results must be JSON-serialisable. Pass `type=` to get a recorded value back as a model:
  `await wf.step("fetch", fetch, type=Report)`.
- Step names are the durable identity of the work. Do not rename them while runs are in flight,
  and make them unique (`f"summarise-{index}"` in a loop).

Parallel branches are plain asyncio. No branch waits for a slower one:

```python
import asyncio

europe, asia = await asyncio.gather(
    wf.run_agent("research-europe", researcher, "Research the EU market"),
    wf.run_agent("research-asia", researcher, "Research the Asian market"),
)
```

Approvals inside a workflow: a paused agent step is returned as it is and is not recorded as
finished. Call the same step again with the decisions to continue it.

```python
result = await wf.run_agent("send", mailer, "Send the newsletter")
if result.status == "paused":
    decisions = {p.tool_call_id: True for p in result.pending_approvals}
    result = await wf.run_agent("send", mailer, "Send the newsletter", approvals=decisions)
```

What is guaranteed: a step that finished never runs again. A plain `step` that was in progress
when the process died runs again from its start, so make side-effecting steps idempotent.


## Send messages between teammates

Use a `TeamMailbox` for a team that needs to exchange information while
its child runs are active. Register a unique name for each participant.
Give each agent the tools from its own endpoint. The trusted host binds
the sender identity, and each endpoint reads its own inbox.

```console
python -m examples.team_messages
```

```python
--8<-- "examples/team_messages.py"
```

The researcher receives a request from the coordinator, sends evidence
to the reviewer, and the reviewer replies to the coordinator. The child
runs use `parent=ctx` to share the coordinator's run limits.

Each inbox holds up to 64 messages by default. A full inbox returns a
tool error. A receive takes messages in arrival order and can wait for up
to 30 seconds. Use one mailbox per team execution. Messages live in
process memory and expire when the application discards that mailbox.
