"""Models for tests: no network, fully deterministic.


    from qagents.models.testing import ScriptedModel, call

    model = ScriptedModel([
        call("get_weather", city="Athens"),     # first reply: a tool call
        "It is sunny in Athens.",               # second reply: the answer
    ])
    agent = Agent(model, tools=[get_weather])
    result = await agent.run("Weather in Athens?")
    assert model.requests[1].messages[-1].role == "tool"

Every request the agent made is kept in ``model.requests``, so a test can
assert on the exact messages, tools and settings that were sent.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import uuid
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any

from ..messages import Message, ToolCallPart, Usage
from .base import Model, ModelRequest, ModelResponse, ModelStreamEvent, ResponseDone, TextDelta

Reply = Any
"""What a scripted reply can be: text, tool call(s), a full message or
response, an exception to raise, or a function of the request returning one."""


def call(tool_name: str, /, **arguments: Any) -> ToolCallPart:
    """A tool call for a scripted reply."""
    return ToolCallPart(id=f"call_{uuid.uuid4().hex[:12]}", name=tool_name, arguments=arguments)


class ScriptedModel(Model):
    """Answers with the given replies, in order."""

    provider = "test"

    def __init__(
        self,
        replies: Iterable[Reply],
        *,
        name: str = "scripted",
        usage: Usage | None = None,
        delay: float = 0.0,
    ) -> None:
        self.name = name
        self.requests: list[ModelRequest] = []
        self._replies = list(replies)
        self._usage = usage or Usage(requests=1)
        self.delay = delay
        """Seconds to wait before each streamed piece of text."""

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if not self._replies:
            raise AssertionError(
                f"{type(self).__name__} was asked for reply #{len(self.requests)} "
                "but the script has no more replies."
            )
        async for event in self._play(self._replies.pop(0), request):
            yield event

    async def _play(self, reply: Reply, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if callable(reply) and not isinstance(reply, type):
            reply = reply(request)
            if inspect.isawaitable(reply):
                reply = await reply
        if isinstance(reply, BaseException):
            raise reply
        response = self._as_response(reply)
        for piece in re.findall(r"\S+\s*|\s+", response.message.text):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield TextDelta(text=piece)
        yield ResponseDone(response=response)

    def _as_response(self, reply: Reply) -> ModelResponse:
        if isinstance(reply, ModelResponse):
            return reply
        if isinstance(reply, Message):
            message = reply
        elif isinstance(reply, str):
            message = Message.assistant(reply)
        elif isinstance(reply, ToolCallPart):
            message = Message.assistant(tool_calls=[reply])
        elif isinstance(reply, list | tuple) and all(isinstance(x, ToolCallPart) for x in reply):
            message = Message.assistant(tool_calls=list(reply))
        else:
            raise TypeError(f"Cannot use {reply!r} as a scripted model reply.")
        return ModelResponse(
            message=message,
            finish_reason="tool_calls" if message.tool_calls else "stop",
            usage=self._usage.model_copy(),
            model=self.name,
            provider=self.provider,
        )


class FunctionModel(ScriptedModel):
    """Answers every request by calling ``function(request)``."""

    def __init__(
        self,
        function: Callable[[ModelRequest], Reply],
        *,
        name: str = "function",
        usage: Usage | None = None,
        delay: float = 0.0,
    ) -> None:
        super().__init__([], name=name, usage=usage, delay=delay)
        self._function = function

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        async for event in self._play(self._function, request):
            yield event
