"""A human in the loop, as a tool the model calls.

``ask_user`` is a deferred tool: the loop never runs it. When the model calls
it, the run pauses with the question in ``result.pending_inputs``; your
application shows it to the user, and the answer goes back with
``agent.resume``:

    agent = Agent("qcloud:<model-id>", tools=[ask_user])
    result = await agent.run("Book me a table for Friday", session=session)
    while result.status == "paused" and result.pending_inputs:
        question = result.pending_inputs[0]
        answer = input(question.arguments["question"] + " ")
        result = await agent.resume(session, results={question.tool_call_id: answer})

The pause is journaled like any other, so the answer can arrive minutes or
days later, in another process.

The arguments are validated before the run pauses, so ``question`` is always
there. The tool's description is `qagents.prompts.ASK_USER_DESCRIPTION`, and
its parameters are described by ``ASK_USER_QUESTION_DESCRIPTION`` and
``ASK_USER_CHOICES_DESCRIPTION``; all three are read when this module is
imported. To word them differently, set ``ask_user.description`` and
``ask_user.parameters``, or build your own tool with ``tool(deferred=True)``.
"""

from __future__ import annotations

from .. import prompts
from ..tools import tool


@tool(read_only=True, deferred=True, description=prompts.ASK_USER_DESCRIPTION)
def ask_user(question: str, choices: list[str] | None = None) -> str:
    """Ask the user a question and wait for the answer.

    The parameter descriptions the model sees are in `qagents.prompts`.
    """
    raise AssertionError("ask_user is deferred: the agent loop never calls it")


ask_user.parameters["properties"]["question"]["description"] = prompts.ASK_USER_QUESTION_DESCRIPTION
ask_user.parameters["properties"]["choices"]["description"] = prompts.ASK_USER_CHOICES_DESCRIPTION


__all__ = ["ask_user"]
