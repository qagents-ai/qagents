# Tools

A tool is a Python function the model can call. Decorate it with `@tool`.

```python
from typing import Literal
from qagents import tool


@tool(read_only=True)
async def search_orders(customer: str, status: Literal["open", "closed"] = "open") -> list[dict]:
    """Find a customer's orders.

    Args:
        customer: Customer email.
        status: Which orders to return.
    """
    return [{"id": "A-1001", "status": status, "customer": customer}]
```

## What the model sees

The signature becomes a JSON Schema and the docstring becomes the description. For the function
above the model receives:

```json
{
  "name": "search_orders",
  "description": "Find a customer's orders.",
  "parameters": {
    "type": "object",
    "properties": {
      "customer": {"type": "string", "description": "Customer email."},
      "status": {"type": "string", "enum": ["open", "closed"], "default": "open",
                 "description": "Which orders to return."}
    },
    "required": ["customer"],
    "additionalProperties": false
  }
}
```

Inspect it with `search_orders.spec`. Parameters can be anything pydantic understands, including
nested models. `*args` and `**kwargs` are rejected, because they cannot be described to a model.
Sync functions run on a worker thread; async functions run on the event loop.

Write descriptions for the model: say what the tool does, when to use it, and what each argument
means. That text is most of what the model knows about your system.

## The run context

A first parameter annotated `RunContext` (or named `ctx`) is filled in by the loop and hidden
from the model.

```python
from dataclasses import dataclass
from qagents import Agent, RunContext, tool


@dataclass
class Deps:
    db: "Database"
    user_id: str


@tool(read_only=True)
async def my_orders(ctx: RunContext[Deps]) -> list[dict]:
    """List the current user's orders."""
    return await ctx.deps.db.orders(ctx.deps.user_id)


agent = Agent("qcloud:<model-id>", tools=[my_orders])
await agent.run("What did I order?", deps=Deps(db=db, user_id="u-42"))
```

| Field | What it is |
|---|---|
| `deps` | whatever you passed as `deps=`: connections, the current user, configuration |
| `run_id`, `session_id`, `agent_name`, `turn` | where the call is happening |
| `tool_call_id` | the model's id for this call |
| `operation_id` | a stable id for this call: [use it as an idempotency key](sessions.md#idempotency) |
| `usage`, `messages` | snapshots of the run so far |
| `emit(event)` | put a custom event on the run's stream |
| `add_usage(usage)` | count model usage your tool caused in the run total |

## Results and errors

Whatever the function returns is rendered to text for the model: strings as they are, everything
else as JSON, `None` as `OK`. Long results are shortened in the middle
(`Limits.max_tool_result_chars`, default 60,000 characters).

Mistakes go back to the model as error results, so it can correct itself:

| Situation | What the model reads |
|---|---|
| unknown tool name | `There is no tool named 'x'. Available tools: ...` |
| arguments are not valid JSON | a request to call again with a valid JSON object |
| arguments fail validation | the failing fields and messages |
| your tool raises `ToolError("...")` | your message |
| your tool raises anything else | `'x' failed: <Type>: <message>` (also logged with a traceback) |
| the tool exceeds its timeout | `'x' did not finish within N seconds and was stopped.` |

Raise `ToolError` when the model can do something about the failure, and write the message for
the model: what went wrong and what to try instead.

```python
from qagents import ToolError


@tool
def refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    if amount > 100:
        raise ToolError("Refunds above 100 EUR need a human. Tell the customer you escalated it.")
    return f"Refunded {amount:.2f} EUR."
```

## Flags

| Option | Meaning |
|---|---|
| `@tool(read_only=True)` | changes nothing: may run in parallel with other read-only calls |
| `@tool(requires_approval=True)` | asks a human first, unless a permission rule allows it |
| `@tool(idempotent=True)` | safe to repeat after a crash |
| `@tool(timeout=10)` | seconds; overrides `Limits.tool_timeout_s` |
| `@tool(name="lookup", description="...")` | override what is derived from the function |

**Parallelism.** When the model requests several tools in one turn, calls to `read_only` tools
run concurrently and all other calls run one at a time, in the order the model wrote them. This
keeps two writers from racing. Change it with `Limits(parallel_tools="all")` or `"none"`.

## Toolkits

A list of tools is already a toolkit: `tools=[file_tools("./workspace"), get_order]` flattens
nested lists. For tools that are only known at run time, subclass `Toolkit`:

```python
from qagents import Tool, Toolkit


class TenantTools(Toolkit):
    def __init__(self, tenant: str) -> None:
        self.tenant = tenant

    async def tools(self) -> list[Tool]:
        return await load_tools_for(self.tenant)
```

`tools()` is called when a run starts. See [ready-made tools](../toolkits.md) for the bundled
ones.
