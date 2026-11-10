"""A complete model provider in thirty lines: it echoes the last message.

python examples/custom_provider.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from qagents import Agent, Message, Model, Usage, register_provider
from qagents.models import ModelRequest, ModelResponse, ModelStreamEvent, ResponseDone, TextDelta


class EchoModel(Model):
    provider = "echo"

    def __init__(self, name: str) -> None:
        self.name = name

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        text = f"You said: {request.messages[-1].text}"
        yield TextDelta(text=text)
        yield ResponseDone(
            response=ModelResponse(
                message=Message.assistant(text),
                usage=Usage(requests=1),
                model=self.name,
                provider=self.provider,
            )
        )


register_provider("echo", EchoModel)  # now Agent("echo:<anything>") works


async def main() -> str:
    result = await Agent("echo:demo").run("hello")
    print(result.output)
    return result.output


if __name__ == "__main__":
    asyncio.run(main())
