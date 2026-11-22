# Getting started

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/qagents-ai/qagents.git
cd qagents
python -m venv .venv
.venv/Scripts/activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

The QCloud provider uses the `qcloud-ai` SDK. Until it is published on PyPI, install it from
source:

```bash
pip install "git+https://github.com/qcloud-host/qcloud-python.git"
```

## See it run without a key

`examples/offline_tour.py` uses a scripted model, so it needs no network:

```bash
python examples/offline_tour.py
```

It shows a streamed tool loop, a pause for approval that is resumed by a different agent object,
and a handoff.

## Your first agent on QCloud

Set `QCLOUD_API_KEY`. Pick a model id from the catalog that lists the `tools` capability
(`client.models.list()` in the SDK, or the public catalog at
`https://platform.qcloud.host/api/public/models`).

```python
import asyncio
from qagents import Agent, tool


@tool(read_only=True)
def get_order(order_id: str) -> dict:
    """Look up an order by its id.

    Args:
        order_id: The order id, for example "A-1001".
    """
    return {"id": order_id, "status": "delivered", "total_eur": 42.5}


agent = Agent(
    "qcloud:<model-id>",
    instructions="You are a support assistant for an online shop. Be brief.",
    tools=[get_order],
)


async def main() -> None:
    result = await agent.run("Has order A-1001 arrived?")
    print(result.output)
    print(result.usage.total_tokens, result.usage.cost_usd)


asyncio.run(main())
```

`"qcloud:<model-id>"` builds a [`QCloudModel`](providers.md#qcloud). To share one SDK client per
process, build the model yourself:

```python
from qcloud_ai import AsyncQCloud
from qagents import Agent, QCloudModel

client = AsyncQCloud()  # reads QCLOUD_API_KEY
agent = Agent(QCloudModel("<model-id>", client=client))
```

## Stream the answer

```python
async for event in agent.stream("Has order A-1001 arrived?"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
    elif event.type == "tool_call_finished":
        print(f"\n[{event.call.name} -> {event.result.content}]")
```

## Keep the conversation

```python
from qagents import Session, SQLiteStore

store = SQLiteStore("agents.db")
await agent.run("My name is Maria.", session=Session(store, id="user-42"))
await agent.run("What is my name?", session=Session(store, id="user-42"))
```

## Check the live platform

`examples/live_acceptance.py` runs a handful of small paid requests against the real API
(streaming, a tool loop, structured output, error mapping) under a spend cap:

```bash
python examples/live_acceptance.py --dry-run   # no key: shows the model and the worst-case cost
python examples/live_acceptance.py             # needs QCLOUD_API_KEY
```

It reads the key from `QCLOUD_API_KEY`, a `.env` file next to `pyproject.toml`, or your Windows
user environment variables, and never prints it.

## Next

- [The loop](concepts/loop.md): what happens in a run, limits and stop reasons.
- [Tools](concepts/tools.md): schemas, context, errors and flags.
- [Sessions and recovery](concepts/sessions.md): what is guaranteed after a crash.
