"""Every piece of text the docs server puts in front of a coding agent's model.

The server's instructions, its tool descriptions, its two prompts and the fixed sentences in
its answers are plain module attributes here, as in `qagents.prompts`. Nothing else in
`qagents.docs_mcp` writes prose for a model. Formatting (headings, lists, code fences) is
built in the modules that use these strings.
"""

from __future__ import annotations

# --- the server -----------------------------------------------------------------------

DOCS_MCP_NAME = "qagents-docs"
"""The MCP server name. Clients show tools as ``mcp__qagents-docs__<tool>`` or similar."""

DOCS_MCP_INSTRUCTIONS = (
    "Documentation of the installed qagents {version}, a Python agent framework. Answers "
    "come from the docs, the examples and the live package, so they match the code you "
    "write against.\n"
    "How to use it:\n"
    "1. search_docs with the task in plain words, then read_page or read_example for the "
    "best hits.\n"
    "2. get_api for every qagents name before you use it: it shows the real signature.\n"
    "3. check_code on the code you wrote: it finds invented imports, attributes and keyword "
    "arguments, with the closest real names.\n"
    "Never guess a qagents name. If a name is not found, it does not exist in this version. "
    "When a guide page and get_api disagree, get_api is right: it reads the installed code."
)
"""Sent to the client in the MCP handshake. ``{version}`` is the installed version."""

# --- tool descriptions ----------------------------------------------------------------

SEARCH_DOCS_DESCRIPTION = (
    "Search the qagents documentation: guide pages, the README, the runnable examples and the "
    "docstrings of the public API. Ranked BM25 over sections. Use plain words for a task "
    "('pause a tool call for human approval', 'test an agent without a model') or a name "
    "('Agent.resume', 'ScriptedModel'). Each result cites its source (page.md#anchor or "
    "file:line) and says which tool reads it in full."
)
SEARCH_QUERY_DESCRIPTION = "Words, a question, or a qagents name such as Agent.run."
SEARCH_LIMIT_DESCRIPTION = "How many results to return (1 to 25)."
SEARCH_KIND_DESCRIPTION = (
    "Only results of one kind: 'docs' (pages), 'example' or 'api' (docstrings). Empty for all."
)

READ_PAGE_DESCRIPTION = (
    "Read one documentation page, or one section of it. Paths are those list_pages and "
    "search_docs show, for example 'concepts/permissions.md'; 'README.md', "
    "'examples/README.md', 'AGENTS.md' and the Claude Code skill 'skills/qagents/SKILL.md' "
    "work too. The section is an anchor such as 'pause-and-resume' or a "
    "heading's text. API reference pages come with the live signatures and docstrings."
)
READ_PAGE_PATH_DESCRIPTION = "The page, for example 'concepts/tools.md' or 'testing.md'."
READ_PAGE_SECTION_DESCRIPTION = (
    "Optional: an anchor ('flags') or a heading ('Flags'). Without it, the whole page."
)

LIST_PAGES_DESCRIPTION = (
    "The table of contents of the qagents documentation, in site order, with a one-line "
    "summary of each page, the installed qagents version and where the docs were read from."
)

GET_API_DESCRIPTION = (
    "The signature, docstring, members and source location of a public qagents object, read "
    "from the installed package by live introspection. Accepts 'Agent', 'qagents.Agent.run', "
    "'Agent.resume', 'qagents.toolkits.files', 'Permissions', 'ScriptedModel' and similar. "
    "Use it before you call anything: names and keyword arguments you have not seen here "
    "probably do not exist."
)
GET_API_SYMBOL_DESCRIPTION = "A qagents name: a module, class, function, method or attribute."

LIST_EXAMPLES_DESCRIPTION = (
    "The examples gallery: runnable agent patterns (supervisor, human in the loop, workflows, "
    "handoffs, RAG, MCP tools, ...) and the other example programs, each with what it shows."
)

READ_EXAMPLE_DESCRIPTION = (
    "The full code of one example, with its docstring that explains what it shows and how "
    "to run it. Names come from list_examples, for example 'supervisor' or "
    "'patterns/human_in_the_loop'."
)
READ_EXAMPLE_NAME_DESCRIPTION = "The example, for example 'supervisor' or 'offline_tour'."

CHECK_CODE_DESCRIPTION = (
    "Check a Python snippet against the installed qagents without running it: it compiles; "
    "every qagents import and attribute exists; keyword and positional arguments of qagents "
    "calls match their signatures; coroutines such as Agent.run are awaited. Each problem "
    "comes with the closest real names. Run it on code you wrote before you run the code."
)
CHECK_CODE_CODE_DESCRIPTION = "The Python source to check. Top-level await is allowed."

# --- resources ------------------------------------------------------------------------

LLMS_TXT_RESOURCE_DESCRIPTION = (
    "The llms.txt index of the qagents documentation: every page with a one-line summary."
)
DOC_RESOURCE_DESCRIPTION = "A qagents documentation page, as Markdown."
EXAMPLE_RESOURCE_DESCRIPTION = "A qagents example program, as Python source."

# --- prompts --------------------------------------------------------------------------

BUILD_AGENT_PROMPT_DESCRIPTION = (
    "Build a new agent with qagents, the qagents way: find the pattern in the docs, check "
    "every name, write an offline test with ScriptedModel or FunctionModel."
)
BUILD_AGENT_TASK_DESCRIPTION = "What the agent should do."

BUILD_AGENT_PROMPT = """\
Build this with qagents {version}: {task}

Work from the installed qagents documentation, through the qagents-docs tools:

1. search_docs for the task in plain words. Read the best page sections (read_page) and the
   closest example (list_examples, read_example). Copy the pattern, not your memory of
   another framework.
2. get_api for every qagents name you use (classes, functions, methods). Use only the
   parameters it shows. If a name is not there, it does not exist.
3. Write the agent:
   - Tools are typed Python functions with a docstring and @tool. Mark read-only tools
     read_only=True. Tools with side effects get requires_approval=True, or a Permissions
     rule, and pass ctx.operation_id to the system they change as the idempotency key.
   - Put limits on every run: Limits(max_turns=..., max_cost_usd=..., timeout_s=...).
   - Keep state in a Session (SQLiteStore for anything that must survive a restart).
   - Several agents: a sub-agent as a tool (tools=[worker] or worker.as_tool()), a handoff
     (handoffs=[...]), or a Workflow with step and run_agent for durable multi-step jobs.
4. Write an offline test with ScriptedModel (or FunctionModel) from qagents.models.testing:
   script the tool calls and the answer, assert on result.status, result.output and
   model.requests. Cover the pause and resume path if a tool needs approval.
5. Run check_code on every file you wrote, fix what it reports, then run the test.
"""
"""The ``build_agent`` prompt. ``{task}`` is what the user asked for."""

REVIEW_PROMPT_DESCRIPTION = (
    "Review Python code that uses qagents against the installed API and the qagents "
    "principles. The static check of the code is included."
)
REVIEW_CODE_DESCRIPTION = "The Python source to review."

REVIEW_PROMPT = """\
Review this code, which uses qagents {version}. The static check against the installed
package found:

{check}

Then review it against the qagents principles, using the qagents-docs tools (get_api,
search_docs, read_page) to confirm every point before you make it:

- Names: every qagents import, attribute and keyword argument exists (the check above).
- Side effects: tools that change something outside the conversation have
  requires_approval=True or a Permissions rule (ask or deny), and are not read_only.
- Idempotency: a tool that may be repeated after a crash uses ctx.operation_id as the
  idempotency key, or is marked idempotent=True only if repeating it is harmless.
- Read-only: only tools without side effects are read_only=True (they may run in parallel).
- Limits: runs have Limits (turns, cost, time); nothing loops without a bound.
- Pauses: a paused run (result.status == "paused") is handled: pending_approvals are
  answered with agent.resume(session, approvals=...), from the same session.
- Durability: state that must survive a restart lives in a Session or a Workflow over a
  persistent store, not in globals.
- Nothing hidden: no prompt text is smuggled into tool results; model-facing strings are
  visible in the code.
- Async: Agent.run, resume and Workflow steps are awaited; run_sync is not used inside a
  running event loop.
- Tests: there is an offline test with ScriptedModel or FunctionModel.

Code:

```python
{code}
```
"""
"""The ``review_code`` prompt. ``{check}`` is the check_code report of ``{code}``."""

# --- answers --------------------------------------------------------------------------

NOT_FOUND_PAGE = "There is no page {path!r}."
NOT_FOUND_SECTION = "Page {path} has no section {section!r}."
NOT_FOUND_EXAMPLE = "There is no example {name!r}."
NOT_FOUND_SYMBOL = (
    "qagents {version} has no public name {symbol!r}. Do not use it: it does not exist in "
    "the installed version."
)
AMBIGUOUS_SYMBOL = "Several qagents objects are called {symbol!r}; ask for one of these:"
DID_YOU_MEAN = "Did you mean: {names}?"
NO_RESULTS = "No results for {query!r}. Try other words, a qagents name, or list_pages to browse."
NO_DOCS_FOLDER = (
    "No documentation folder was found, so only the docstrings of the installed package are "
    "searchable. Set QAGENTS_DOCS_DIR to a qagents checkout (or its docs folder) to add the "
    "guides and examples."
)
SKILL_PAGE_TITLE = "The Claude Code skill"
"""How list_pages and the resources name ``skills/qagents/SKILL.md`` (its heading is
"qagents")."""

PAGE_TRUNCATED = (
    "The page is longer than {limit} characters and was cut here. Read one section with "
    "section=..., one of: {sections}"
)

API_NO_DOCSTRING = "No docstring."
API_MODULE_NOTES = "From the docstring of `{module}`:"
API_SEE_ALSO = "See also (guides and examples):"
API_ALSO_NAMED = "Other objects with this name: {names}."
API_ASYNC_NOTE = "This is a coroutine function: call it with `await`."
API_FORWARDS = "Its `**kwargs` are passed to `{target}`, which accepts them."
API_FORWARDS_FIELDS = (
    "Its `**kwargs` set the fields of `{model}` by name. Its own `__init__` can fill in "
    "fields itself, so a field listed without a default may still be optional: the "
    "docstring says what to pass."
)
"""For a pydantic model whose own ``__init__`` passes ``**data`` to the model: the
signature above is that ``__init__``, not pydantic's list of fields."""
API_MEMBER_HINT = 'Details of one member: get_api("{path}.<member>").'
API_NEEDS_EXTRA = "Needs the optional extra: `pip install 'qagents[{extra}]'`."
API_MISSING_EXTRA = (
    "Could not import it here ({missing} is not installed), so this comes from the source. "
    "It needs `pip install 'qagents[{extra}]'`."
)

# --- check_code -----------------------------------------------------------------------

CHECK_SYNTAX = "The code does not compile: {message}."
CHECK_NO_MODULE = "`{module}` does not exist in qagents {version}."
CHECK_NO_NAME = "`{owner}` has no `{name}`."
CHECK_ELSEWHERE = "`{name}` is in `{module}`: `{line}`."
CHECK_NO_ATTRIBUTE = "`{owner}` has no attribute `{name}`."
CHECK_UNKNOWN_KEYWORD = "`{callable}()` has no parameter `{name}`."
CHECK_POSITIONAL = (
    "`{callable}()` takes at most {count} positional argument(s) ({names}); pass the rest by "
    "keyword."
)
CHECK_MISSING = "`{callable}()` is missing required argument(s): {names}."
CHECK_ACCEPTS = "It accepts: {names}."
CHECK_IMPORT_FAILED = (
    "`{module}` could not be imported here ({error}). It may need an optional extra: "
    "`pip install 'qagents[{extra}]'`."
)
CHECK_NOT_AWAITED = (
    "`{callable}()` is a coroutine function and is not awaited here: write `await {call}`, "
    "or use `run_sync` outside async code."
)
CHECK_RUN_SYNC_IN_ASYNC = (
    "`{callable}()` starts its own event loop and fails inside `async def`; use "
    "`await {replacement}(...)` there."
)
CHECK_OK = "No problems found."
CHECK_SUMMARY = (
    "Checked against the installed qagents {version}: {imports} import(s), {calls} call(s) "
    "and {attributes} attribute(s) of qagents objects."
)
CHECK_NOTHING = (
    "The code uses no qagents names that a static check can follow, so only its syntax was checked."
)
CHECK_LIMITS = (
    "Static check only: it follows names imported from qagents and the variables assigned "
    "from them, and does not run the code. Test behaviour with ScriptedModel or FunctionModel."
)

KEYWORD_HINTS: dict[str, str] = {
    "system_prompt": "qagents calls it `instructions`: `Agent(model, instructions=...)`.",
    "system": "qagents calls it `instructions`: `Agent(model, instructions=...)`.",
    "system_message": "qagents calls it `instructions`: `Agent(model, instructions=...)`.",
    "role": "Role, goal and backstory go into `instructions=...`.",
    "goal": "Role, goal and backstory go into `instructions=...`.",
    "backstory": "Role, goal and backstory go into `instructions=...`.",
    "llm": "The model is the first argument: `Agent(model, ...)`.",
    "model_name": 'The model is the first argument: `Agent("provider:model-id", ...)`.',
    "max_iterations": "Turn limits are `limits=Limits(max_turns=...)`.",
    "max_iter": "Turn limits are `limits=Limits(max_turns=...)`.",
    "max_steps": "Turn limits are `limits=Limits(max_turns=...)`.",
    "recursion_limit": "Turn limits are `limits=Limits(max_turns=...)`.",
    "max_turns": "Turn limits are `limits=Limits(max_turns=...)`.",
    "max_cost_usd": "Cost limits are `limits=Limits(max_cost_usd=...)`.",
    "timeout_s": "Time limits are `limits=Limits(timeout_s=...)`.",
    "result_type": "The answer's type is `output_type=...`.",
    "response_model": "The answer's type is `output_type=...`.",
    "output_schema": "The answer's type is `output_type=...`.",
    "output_pydantic": "The answer's type is `output_type=...`.",
    "response_format": "The answer's type is `output_type=...` (with `output_mode=...`).",
    "functions": "Tools are `tools=[...]`.",
    "temperature": "Model settings are `settings=ModelSettings(temperature=...)`.",
    "max_tokens": "Model settings are `settings=ModelSettings(max_tokens=...)`.",
    "thread_id": "Conversations are `session=Session(store, id=...)`.",
    "session_id": "Conversations are `session=Session(store, id=...)`.",
    "conversation_id": "Conversations are `session=Session(store, id=...)`.",
    "checkpointer": "Durable conversations are `session=Session(SQLiteStore(path), id=...)`.",
    "memory": "Conversation memory is a `Session`: `session=Session(store, id=...)`.",
    "callbacks": "Code inside the loop is `hooks=[MyHooks()]` with a `Hooks` subclass.",
    "verbose": "There is no verbose flag: stream events with `agent.stream(...)`.",
    "message": 'The prompt is the first positional argument: `agent.run("...")`.',
    "input": 'The prompt is the first positional argument: `agent.run("...")`.',
    "user_input": 'The prompt is the first positional argument: `agent.run("...")`.',
    "query": 'The prompt is the first positional argument: `agent.run("...")`.',
    "task": 'The prompt is the first positional argument: `agent.run("...")`.',
    "approvals": "Answer approvals with `agent.resume(session, approvals={call_id: True})`.",
    "needs_approval": "Mark a tool with `@tool(requires_approval=True)`.",
    "require_approval": "The flag is `requires_approval=True`.",
    "readonly": "The flag is `read_only=True`.",
}
"""Hints for keyword arguments that other frameworks use, shown when one is unknown."""

NAME_HINTS: dict[str, str] = {
    "function_tool": "Use `@tool`: `from qagents import tool`.",
    "FunctionTool": "Use `@tool` on a function, or `Tool`.",
    "BaseTool": "Use `@tool` on a typed function, or a `Toolkit`.",
    "Runner": 'There is no Runner: `result = await agent.run("...")`.',
    "TestModel": "Use `ScriptedModel` or `FunctionModel` from `qagents.models.testing`.",
    "MockModel": "Use `ScriptedModel` or `FunctionModel` from `qagents.models.testing`.",
    "FakeModel": "Use `ScriptedModel` or `FunctionModel` from `qagents.models.testing`.",
    "StateGraph": "There is no graph: write an async function; `Workflow` makes steps durable.",
    "Graph": "There is no graph: write an async function; `Workflow` makes steps durable.",
    "Crew": "There is no Crew: an async function that calls agents, or a manager agent with "
    "the others as tools.",
    "Task": "A task is the prompt of `agent.run(...)` or `Workflow.run_agent(...)`.",
    "Process": "A process is your async function; `Workflow` makes its steps durable.",
    "create_react_agent": "Use `Agent(model, instructions=..., tools=[...])`.",
    "create_supervisor": "Give the supervisor its workers as tools: `tools=[worker]`.",
    "MemorySaver": "Use `Session(InMemoryStore(), id=...)` or `SQLiteStore(path)`.",
    "SqliteSaver": "Use `Session(SQLiteStore(path), id=...)`.",
    "interrupt": "Pause with `@tool(requires_approval=True)` or a deferred tool, then "
    "`agent.resume(...)`.",
    "Command": "Resume a paused run with `agent.resume(session, approvals=..., results=...)`.",
    "Handoff": "Use `Agent(..., handoffs=[other_agent])`.",
    "handoff": "Use `Agent(..., handoffs=[other_agent])`.",
    "OpenAIModel": "Use `OpenAIChatModel` (any Chat Completions server).",
    "ChatOpenAI": "Use `OpenAIChatModel` (any Chat Completions server).",
    "ChatAnthropic": "Use `AnthropicModel` (extra `anthropic`).",
    "Memory": "Conversation memory is a `Session`; files are `notes_tools(directory)`.",
}
"""Hints for names that other frameworks have and qagents does not."""

ATTRIBUTE_HINTS: dict[str, str] = {
    "final_output": "The answer is `result.output` (`result.text` for the plain text).",
    "data": "The answer is `result.output` (`result.text` for the plain text).",
    "response": "The answer is `result.output` (`result.text` for the plain text).",
    "content": "The answer is `result.output` (`result.text` for the plain text).",
    "answer": "The answer is `result.output` (`result.text` for the plain text).",
    "result": "The answer is `result.output` (`result.text` for the plain text).",
    "final_answer": "The answer is `result.output` (`result.text` for the plain text).",
    "interrupts": "Pending approvals are `result.pending_approvals`.",
    "pending": "Pending calls are `result.pending_approvals` and `result.pending_inputs`.",
    "tool": "Decorate a function with `@tool` and pass it in `tools=[...]`.",
    "tool_plain": "Decorate a function with `@tool` and pass it in `tools=[...]`.",
    "invoke": 'Call `await agent.run("...")`.',
    "ainvoke": 'Call `await agent.run("...")`.',
    "kickoff": 'Call `await agent.run("...")`.',
    "chat": 'Call `await agent.run("...", session=session)`.',
    "run_stream": 'Stream with `agent.stream("...")`.',
    "approve": "Approve with `agent.resume(session, approvals={call_id: True})`.",
}
"""Hints for attributes that other frameworks' objects have, shown when one is unknown."""

# --- the skill command (python -m qagents.docs_mcp skill) ------------------------------

SKILL_MISSING = (
    "This install has no copy of the Claude Code skill (skills/qagents/SKILL.md). Get the "
    "skills/qagents folder from the qagents repository: {url}"
)
SKILL_WRITTEN = "Wrote {path}: the Claude Code skill of qagents {version}."
SKILL_UNCHANGED = "{path} is already the skill of qagents {version}; nothing to write."
SKILL_EXISTS = "{path} exists and is different. Nothing was written. Pass --force to replace it."
