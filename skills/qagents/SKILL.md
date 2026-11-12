---
name: qagents
description: Write, fix, review or test Python code that uses the qagents agent framework. Use it whenever code imports qagents (Agent, tool, Workflow, Session, Permissions, ScriptedModel, ...) or the user asks to build an agent, a tool, a multi-agent workflow or a test with qagents. It looks up the installed qagents docs and API (the qagents-docs MCP server, or python -m qagents.docs_mcp) and checks code for invented names before it runs.
---

# qagents

qagents is a small Python agent framework: one loop, typed tools, durable sessions. It is
young, and names from other frameworks (LangGraph, CrewAI, OpenAI Agents SDK, pydantic-ai)
mostly do not exist in it. Work from the docs of the installed version, never from memory.

## How to look things up

If the `qagents-docs` MCP server is connected, use its tools. Otherwise run the same commands
in the shell, with the Python of the project (for example `.venv/Scripts/python.exe` on
Windows, `.venv/bin/python` elsewhere):

| Question | MCP tool | Shell |
|---|---|---|
| How do I ...? | `search_docs("...")` | `python -m qagents.docs_mcp search "..."` |
| Read the page or section a result cites | `read_page(path, section)` | `python -m qagents.docs_mcp read <path> [section]` |
| What pages are there? | `list_pages()` | `python -m qagents.docs_mcp pages` |
| What does X take and return? | `get_api("X")` | `python -m qagents.docs_mcp api X` |
| Is there an example of ...? | `list_examples()`, `read_example(name)` | `python -m qagents.docs_mcp examples`, `... example <name>` |
| Is my code right? | `check_code(source)` | `python -m qagents.docs_mcp check file.py` |

`get_api` accepts `Agent`, `Agent.run`, `qagents.toolkits.files`, `ScriptedModel`,
`RunResult.pending_approvals` and similar. Its answer is read from the installed package, so
it wins over a guide page or your memory when they disagree.

## Workflow

1. `search_docs` with the task in plain words. Read the top guide section and the closest
   example (`read_example`).
2. `get_api` for every class, function and method you are about to call. Use only the
   parameters it lists. If it says the name does not exist, do not use it.
3. Write the code and an offline test with `ScriptedModel` or `FunctionModel`.
4. `check_code` on each file. Fix every problem; it names the closest real names.
5. Run the test.

## Cheat sheet (check details with get_api)

```python
from qagents import Agent, Limits, Permissions, RunContext, Session, SQLiteStore, Workflow, tool
from qagents.models.testing import FunctionModel, ScriptedModel, call


@tool(read_only=True)  # flags: read_only, requires_approval, idempotent, timeout, deferred
def lookup(city: str) -> str:
    """The docstring is the description the model reads."""
    return f"Sunny in {city}"


@tool(requires_approval=True)  # the run pauses until a person decides
def refund(ctx: RunContext, order_id: str) -> str:
    """Refund an order."""
    return f"refunded {order_id} (key {ctx.operation_id})"  # stable idempotency key


agent = Agent(
    "qcloud:<model-id>",  # or any Model, for example ScriptedModel([...]) in tests
    instructions="Be brief.",  # not system_prompt
    tools=[lookup, refund],
    limits=Limits(max_turns=10, max_cost_usd=0.50),  # not max_iterations
    permissions=Permissions(allow=["lookup"], deny=["delete_*"]),  # deny, ask, allow rules
)


async def flow(session: Session) -> None:
    result = await agent.run("Refund A-1", session=session)  # always await
    if result.status == "paused":  # completed | paused | limit_reached | cancelled | blocked
        decisions = {p.tool_call_id: True for p in result.pending_approvals}
        result = await agent.resume(session, approvals=decisions)
    print(result.output)  # not final_output; result.text is the plain text
```

- Several agents: an agent in `tools=[worker]` (or `worker.as_tool(read_only=True)`) is a
  sub-agent with one argument, `task`; `handoffs=[other]` hands the conversation over.
- Durable steps: `wf = Workflow(store, id="job-1")`, then `await wf.run_agent("name", agent,
  prompt)` and `await wf.step("name", function)`. A finished step never runs again.
- Typed answers: `Agent(..., output_type=MyModel)`; `result.output` is a `MyModel`.
- Streaming: `stream = agent.stream("..."); async for event in stream:` with
  `event.type == "text_delta"`; then `stream.result`.
- Durable conversations: `async with SQLiteStore("agents.db") as store:` and
  `Session(store, id="user-42")`.

## Testing

```python
from qagents import Agent, Session, tool
from qagents.models.testing import ScriptedModel, call


@tool(requires_approval=True)
def refund(order_id: str) -> str:
    """Refund an order."""
    return f"refunded {order_id}"


async def test_refund_pauses_then_runs():
    model = ScriptedModel([call("refund", order_id="A-1"), "Done."])  # tool call, then answer
    agent = Agent(model, tools=[refund])
    session = Session()
    paused = await agent.run("Refund A-1", session=session)
    assert paused.status == "paused"
    done = await agent.resume(session, approvals={paused.pending_approvals[0].tool_call_id: True})
    assert done.output == "Done."
    assert model.requests[1].messages[-1].tool_results[0].content == "refunded A-1"
```

`FunctionModel(lambda request: ...)` returns a reply computed from the request (text, a
`call(...)`, or a list of calls). A sub-agent gets its own scripted model.

## Mistakes the checker catches

| Written | qagents |
|---|---|
| `Agent(system_prompt=...)` | `Agent(instructions=...)` |
| `Agent(max_iterations=5)` | `Agent(limits=Limits(max_turns=5))` |
| `result.final_output`, `result.data` | `result.output` |
| `agent.run(...)` without `await` | `await agent.run(...)` |
| `@agent.tool` | `@tool` on a function, then `tools=[...]` |
| `from qagents import ScriptedModel` | `from qagents.models.testing import ScriptedModel` |
| `agent.run(..., approvals=...)` | `agent.resume(session, approvals=...)` |

The checker is static: it follows names imported from qagents and variables assigned from
them. A clean report means no invented names, not that the logic is right; run the test.

## More

- `docs/coding-agents.md`: connecting the MCP server to Claude Code, Codex, Cursor, VS Code.
- `AGENTS.md` at the repository root: rules for code that uses qagents and for work on qagents.
- To make another coding agent do work for a qagents agent, drive it over ACP or the Devvi
  SDK. Over MCP (Codex: `codex mcp-server`) each turn is one opaque call that qagents cannot
  follow, decide permissions in, or cancel: `read_page("concepts/external-agents.md")`.
