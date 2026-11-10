"""Every piece of text qagents itself puts in front of a model.

There are no hidden prompts anywhere else in the package. The agent loop sends
your instructions, your messages and your tool descriptions; the strings below
are the only words it adds, and each one is a plain module attribute you can
read, log or replace (``qagents.prompts.TOOL_FAILED = "..."``).
"""

from __future__ import annotations

# --- structured output -------------------------------------------------------

OUTPUT_INSTRUCTIONS = (
    "Reply with a single JSON value that conforms to this JSON Schema. "
    "Do not wrap it in code fences and do not add any other text.\n\n{schema}"
)
"""Appended to the instructions when ``output_mode="prompted"``."""

OUTPUT_RETRY = (
    "Your reply could not be used as the final answer: {error}\n"
    "Reply again with only the corrected JSON."
)
"""Sent as a user message when the final answer fails validation."""

# --- tool results written by the loop, not by a tool -------------------------

UNKNOWN_TOOL = "There is no tool named {name!r}. Available tools: {available}."

INVALID_ARGUMENTS_JSON = (
    "The arguments for {name!r} were not a valid JSON object ({error}). "
    "Call the tool again with a valid JSON object."
)

INVALID_ARGUMENTS = (
    "Invalid arguments for {name!r}:\n{error}\nCall the tool again with corrected arguments."
)

PERMISSION_DENIED = "{name!r} was not run: {reason}"

APPROVAL_DECLINED = "The user declined this tool call."

TOOL_TIMEOUT = "{name!r} did not finish within {seconds:g} seconds and was stopped."

TOOL_FAILED = "{name!r} failed: {error}"

TOOL_INTERRUPTED = (
    "{name!r} had started when the run stopped, and it never reported a result. "
    "Its outcome is unknown: check what happened before calling it again."
)
"""A tool had started when the process stopped and it is not safe to repeat."""

TOOL_NOT_EXECUTED = "{name!r} was not run because the run was interrupted before it started."
"""A tool call was still waiting when a new run started on the same session."""

TOOL_CANCELLED = "{name!r} was cancelled before it finished."

TRUNCATED = "\n[... {omitted} characters omitted ...]\n"
"""Marks where a long tool result was shortened; ``omitted`` is a character count."""

# --- multi-agent --------------------------------------------------------------

HANDOFF_DESCRIPTION = "Hand the conversation over to {name}. {description}"
"""Description of the tool that hands the conversation to another agent."""

HANDOFF_RESULT = "The conversation is now handled by {name}."

HANDOFF_DUPLICATE = "Only one handoff can happen per turn; this one was ignored."

SUBAGENT_DESCRIPTION = (
    "Delegate a task to {name}. It works in its own conversation and sees only the "
    "task text you write, so include everything it needs."
)
"""Default description of an agent exposed as a tool with ``agent.as_tool()``."""

SUBAGENT_STOPPED = "{name} stopped before finishing ({reason})."

# --- context compaction --------------------------------------------------------

SUMMARY_REQUEST = (
    "Summarise the conversation above so that work can continue from the summary alone. "
    "Keep: the user's goals and constraints, decisions made, facts and identifiers "
    "learned from tools, what was done, and what remains to do. Leave out pleasantries."
)
"""Instructions for the summarising request made by ``SummarizingCompactor``."""

SUMMARY_PREFIX = "Summary of the earlier conversation:\n\n"
"""Prefix of the message that replaces the summarised part of the history."""
