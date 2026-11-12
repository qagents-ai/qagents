"""The ``qagents`` command line: chat with an agent, run one prompt, list models.

    python -m qagents chat --model qcloud:<model-id> --tools files,time
    python -m qagents run "Summarise README.md" --model qcloud:<model-id> --tools files
    python -m qagents models qcloud

``--model`` takes a ``provider:model-id`` string and goes through the provider
registry, so every provider registered with `register_provider` works. The
command adds no instructions of its own: the model sees ``--instructions``, the
conversation and the tool descriptions, nothing else.

Tools with effects outside the conversation ask first: writing and editing
files, shell commands, saving and deleting notes, and fetching web pages,
since a URL can carry workspace data out. The prompt shows the whole call,
every argument in full, and waits for ``y``; any other answer, or the end of
input, declines.
``chat`` streams the answer and shows each tool call on one line. ``run``
writes tool activity and approval prompts to stderr, so stdout holds only the
answer (or the JSON result) and the usage. Answers are read from stdin, so
with ``run -`` (the prompt on stdin) every call that needs approval is declined.

Text from the model and from tools reaches the terminal with control
characters (which start terminal escape sequences) and bidirectional controls
escaped, as in ``\\x1b``, so it cannot restyle the terminal or forge a prompt.

``--max-cost`` covers the whole command, runs that failed included: in
``chat`` each prompt gets what is left of the budget. A request that failed
is not counted, since its usage is unknown. It needs a model whose prices are
known (QCloud catalog models), and the command refuses to start otherwise.

Exit codes: 0 when the agent answered, 1 when the run failed or stopped
without an answer, 2 for a mistake in the command line, 130 after Ctrl-C.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import threading
import unicodedata
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO, Any

import httpx

from .agent import Agent, Limits, RunStream
from .errors import ConfigurationError, QAgentsError
from .messages import ToolCallPart, Usage
from .models.base import ModelSettings, resolve_model
from .permissions import Permissions
from .results import RunResult
from .session import Session, SessionStore, SQLiteStore
from .tools import RunContext, Tool

TOOL_SETS: dict[str, str] = {
    "files": "list, read and search files in --workspace; write and edit them after approval",
    "web": "fetch public web pages, each after approval",
    "notes": "keep notes as Markdown files in <workspace>/notes; save and delete after approval",
    "time": "tell the current date and time",
    "shell": "run shell commands in --workspace, each after approval",
    "todo": "keep a todo list for multi-step work in the session state",
    "ask": "let the agent ask you questions while it works",
}
"""The names ``--tools`` accepts, with what each gives the agent."""

QCLOUD_BASE_URL = "https://platform.qcloud.host/api/v1"
"""Used when neither ``--base-url`` nor ``QCLOUD_BASE_URL`` is set."""

DEFAULT_SESSION = "default"
"""Session id used with ``--db`` when ``--session`` is not given."""

_PREVIEW = 100
"""Characters of a tool's arguments or result shown on its line, and the
longest call an approval prompt shows on one line."""

_ASK_FIRST = ("save_note", "delete_note", "fetch_url")
"""Tools the command asks about although they do not require approval: notes
are overwritten and deleted, and a fetched URL can carry workspace data out."""

_BIDI = frozenset("\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")
"""Bidirectional controls: they reorder the text around them."""


class UsageError(Exception):
    """A mistake in the command line, reported with exit code 2."""


# --- argument parsing --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the ``qagents`` command.

    Returns:
        A parser with the ``chat``, ``run`` and ``models`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="qagents",
        description="Chat with an agent, run one prompt, or list the models of a provider.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    agent_options = argparse.ArgumentParser(add_help=False)
    agent_options.add_argument(
        "--model",
        default=os.environ.get("QAGENTS_MODEL"),
        help="provider:model-id, for example qcloud:<model-id> (default: $QAGENTS_MODEL)",
    )
    agent_options.add_argument(
        "--instructions", default=None, help="instructions (system prompt) for the agent"
    )
    agent_options.add_argument(
        "--tools",
        default="",
        metavar="LIST",
        help=f"comma-separated tool sets: {', '.join(TOOL_SETS)}",
    )
    agent_options.add_argument(
        "--workspace",
        default=".",
        metavar="DIR",
        help="directory the files, notes and shell tools work in (default: .)",
    )
    agent_options.add_argument(
        "--db", default=None, metavar="PATH", help="SQLite file that keeps the conversation"
    )
    agent_options.add_argument(
        "--session",
        default=None,
        metavar="ID",
        help=f"conversation id in --db (default: {DEFAULT_SESSION})",
    )
    agent_options.add_argument(
        "--max-cost",
        type=float,
        default=None,
        metavar="USD",
        help="stop when the model requests have cost this much (needs a priced model)",
    )
    agent_options.add_argument(
        "--max-turns",
        type=int,
        default=None,
        metavar="N",
        help=f"model requests per prompt (default: {Limits().max_turns})",
    )
    agent_options.add_argument(
        "--max-tokens", type=int, default=None, metavar="N", help="output tokens per response"
    )

    chat = commands.add_parser(
        "chat",
        parents=[agent_options],
        help="talk with an agent; /usage, /reset and /exit inside",
        description="Talk with an agent. Inside: /usage, /reset, /help, /exit.",
    )
    chat.set_defaults(handler=_chat)

    run = commands.add_parser(
        "run",
        parents=[agent_options],
        help="run one prompt and print the answer",
        description="Run one prompt, print the answer and the usage, exit non-zero on failure.",
    )
    run.add_argument("prompt", help="the prompt; - reads it from stdin")
    run.add_argument("--json", action="store_true", help="print the whole run result as JSON")
    run.set_defaults(handler=_run)

    models = commands.add_parser(
        "models",
        help="list the models a provider offers",
        description="List the public QCloud catalog: models that can call tools, with prices.",
    )
    models.add_argument("provider", choices=["qcloud"], help="the provider whose catalog to list")
    models.add_argument(
        "--all", action="store_true", help="include models without the tools capability"
    )
    models.add_argument(
        "--search", default=None, metavar="TEXT", help="only models whose id or name contains TEXT"
    )
    models.add_argument("--json", action="store_true", help="print the models as JSON")
    models.add_argument(
        "--base-url",
        default=None,
        help=f"QCloud API base URL (default: $QCLOUD_BASE_URL or {QCLOUD_BASE_URL})",
    )
    models.set_defaults(handler=_models)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Run the ``qagents`` command.

    Args:
        argv: The arguments after the program name; defaults to ``sys.argv[1:]``.
        stdin: Where prompts, answers to questions and approvals are read from.
        stdout: Where answers and results go.
        stderr: Where errors (and, for ``run``, tool activity) go.

    Returns:
        The exit code: 0 on success, 1 when the run failed or stopped without
        an answer, 2 for a command-line mistake, 130 after Ctrl-C.
    """
    console = _Console(
        stdin if stdin is not None else sys.stdin,
        stdout if stdout is not None else _tolerant(sys.stdout),
        stderr if stderr is not None else _tolerant(sys.stderr),
    )
    parser = build_parser()
    try:
        with contextlib.redirect_stdout(console.out), contextlib.redirect_stderr(console.err):
            args = parser.parse_args(argv)
    except SystemExit as exit_:
        if exit_.code is None:
            return 0
        return exit_.code if isinstance(exit_.code, int) else 2
    try:
        return asyncio.run(args.handler(args, console))
    except UsageError as error:
        console.error(f"qagents {args.command}: error: {error}")
        return 2
    except QAgentsError as error:
        console.error(f"error: {error}")
        return 1
    except KeyboardInterrupt:
        console.error("\ninterrupted")
        return 130


# --- the commands ---------------------------------------------------------------------------


async def _chat(args: argparse.Namespace, console: _Console) -> int:
    agent, store = _build_agent(args, console, activity=console.out)
    try:
        await _check_cost_limit(agent, args)
        session = _session(store, args)
        spent = _Spent()
        console.say(
            f"qagents chat with {agent.model.provider}:{agent.model.name}"
            f"{_tools_note(args)}, {_session_note(session, store)}."
        )
        console.say("Type /help for commands, /exit to leave.")
        while True:
            line = await console.ask("\nyou> ")
            if line is None:
                console.say("")
                return 0
            text = line.strip()
            if not text:
                continue
            if text.startswith("/"):
                command = text.split()[0].lower()
                if command in ("/exit", "/quit"):
                    return 0
                if command == "/reset":
                    session = _fresh_session(store, session)
                    console.say(f"Started a new conversation ({_session_note(session, store)}).")
                elif command == "/usage":
                    console.say(_usage_text(spent.usage))
                elif command == "/help":
                    console.say(
                        "/usage  tokens and cost so far\n"
                        "/reset  start a new conversation\n"
                        "/exit   leave (Ctrl-D works too)"
                    )
                else:
                    console.say(f"Unknown command {command}. Commands: /usage, /reset, /exit.")
                continue
            limits = _limits(args, spent.usage)
            if limits is None:
                console.say(
                    f"The --max-cost budget of ${args.max_cost:g} is used up "
                    f"({_usage_text(spent.usage)}). Leave with /exit."
                )
                continue
            try:
                result = await _converse(agent, text, session, limits, args, spent, console)
            except QAgentsError as error:  # what the run used is counted all the same
                console.say(f"error: {error}")
                continue
            if not result.completed:
                console.say(f"[the run stopped: {_stop_text(result)}]")
    finally:
        await _close(agent, store)


async def _run(args: argparse.Namespace, console: _Console) -> int:
    prompt = args.prompt
    if prompt == "-":
        prompt = await _read(console.stdin.read)
    if not prompt.strip():
        raise UsageError("the prompt is empty.")
    agent, store = _build_agent(args, console, activity=console.err)
    try:
        await _check_cost_limit(agent, args)
        session = _session(store, args)
        limits = _limits(args, Usage())
        assert limits is not None
        spent = _Spent()
        try:
            result = await _converse(
                agent, prompt, session, limits, args, spent, console, stream_text=False
            )
        except QAgentsError as error:
            console.error(f"error: {error}")
            console.error(f"[usage] {_usage_text(spent.usage)}")
            if args.json:
                failure = {"type": type(error).__name__, "message": str(error)}
                usage = spent.usage.model_dump(mode="json")
                data = {"status": "error", "error": failure, "usage": usage}
                console.write(json.dumps(data, indent=2) + "\n")
            return 1
    finally:
        await _close(agent, store)

    if args.json:
        data = result.model_dump(mode="json")
        data["usage"] = spent.usage.model_dump(mode="json")  # every part of a resumed run
        console.write(json.dumps(data, indent=2) + "\n")
    else:
        text = result.text.rstrip("\n")
        if text:
            console.write((_visible(text) if _is_terminal(console.out) else text) + "\n")
        console.write(f"\n[usage] {_usage_text(spent.usage)}\n")
    if not result.completed:
        console.error(f"The run stopped without an answer: {_stop_text(result)}.")
        return 1
    return 0


async def _models(args: argparse.Namespace, console: _Console) -> int:
    base = (args.base_url or os.environ.get("QCLOUD_BASE_URL") or QCLOUD_BASE_URL).rstrip("/")
    url = base.removesuffix("/v1") + "/public/models"
    try:
        response = await asyncio.to_thread(
            httpx.get, url, timeout=30.0, headers={"accept": "application/json"}
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        console.error(
            f"Could not read the QCloud catalog at {url}: {error}. "
            "Check the connection, or point --base-url (or QCLOUD_BASE_URL) at the API."
        )
        return 1
    listed = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(listed, list):
        console.error(f"The catalog at {url} did not return a model list ('data' is missing).")
        return 1
    search = (args.search or "").lower()
    models = sorted(
        (
            entry
            for entry in listed
            if isinstance(entry, dict)
            and isinstance(entry.get("id"), str)
            and (args.all or "tools" in (entry.get("capabilities") or []))
            and search in f"{entry['id']} {entry.get('display_name') or ''}".lower()
        ),
        key=lambda entry: entry["id"],
    )
    rows = [_catalog_row(entry) for entry in models]
    if args.json:
        console.write(json.dumps(rows, indent=2) + "\n")
        return 0
    if not rows:
        kind = "models" if args.all else "models that can call tools"
        matching = f" matching {args.search!r}" if args.search else ""
        console.write(f"The catalog lists no {kind}{matching}.\n")
        return 0
    table = [["MODEL", "CONTEXT", "INPUT $/1M", "OUTPUT $/1M", "CAPABILITIES"]]
    for row in rows:
        table.append(
            [
                row["id"],
                f"{row['context_length']:,}" if row["context_length"] else "-",
                _price(row["input_per_million"]),
                _price(row["output_per_million"]),
                ", ".join(row["capabilities"]),
            ]
        )
    widths = [max(len(line[column]) for line in table) for column in range(len(table[0]))]
    for line in table:
        cells = [line[0].ljust(widths[0])]
        cells += [cell.rjust(width) for cell, width in zip(line[1:4], widths[1:4], strict=True)]
        cells.append(line[4])
        console.write("  ".join(cells).rstrip() + "\n")
    console.write(f"\n{len(rows)} models. Prices are USD per million tokens. ")
    console.write(f"Use one with --model qcloud:{rows[0]['id']}\n")
    catalog = payload.get("catalog")
    if isinstance(catalog, dict) and catalog.get("stale"):
        updated = catalog.get("last_updated_at") or "unknown"
        console.error(f"Note: the catalog marks itself as stale (last updated: {updated}).")
    return 0


# --- running the agent ----------------------------------------------------------------------


async def _converse(
    agent: Agent,
    prompt: str,
    session: Session,
    limits: Limits,
    args: argparse.Namespace,
    spent: _Spent,
    console: _Console,
    *,
    stream_text: bool = True,
) -> RunResult:
    """Run ``prompt``, then answer the questions of a paused run until it stops for good.

    The run's usage is counted in ``spent``, also when a part raises. Every
    resume gets the same ``limits``: they count the whole run, resumes included.

    Returns:
        The last result.
    """
    activity = console.out if stream_text else console.err
    spent.start()
    stream = agent.stream(prompt, session=session, limits=limits)
    result = await _show(stream, console, stream_text, spent)
    while result.status == "paused":
        approvals: dict[str, bool] = {}
        for approval in result.pending_approvals:
            approvals[approval.tool_call_id] = await _approve(
                console, activity, _call_text(approval.name, approval.arguments)
            )
        results: dict[str, Any] = {}
        for pending in getattr(result, "pending_inputs", []):
            answer = await console.ask(_question_text(pending.name, pending.arguments), activity)
            if answer is None:
                return result  # no more input: leave the run paused
            results[pending.tool_call_id] = answer.rstrip("\r\n")
        if _limits(args, spent.usage) is None:
            return result  # the budget is used up: leave the run paused
        options: dict[str, Any] = {"approvals": approvals, "limits": limits}
        if results:
            options["results"] = results
        result = await _show(agent.resume_stream(session, **options), console, stream_text, spent)
    return result


async def _show(
    stream: RunStream, console: _Console, stream_text: bool, spent: _Spent
) -> RunResult:
    """Consume a run, writing its answer text and a line per tool call."""
    activity = console.out if stream_text else console.err
    try:
        async with stream:
            async for event in stream:
                kind = event.type
                if kind == "text_delta" and stream_text:
                    console.text(event.text)
                elif kind in ("tool_call_started", "tool_call_finished", "agent_switched"):
                    console.note(_activity_text(event), activity)
                elif kind == "subagent_event":
                    inner = event.event
                    if getattr(inner, "type", None) in ("tool_call_started", "tool_call_finished"):
                        console.note(f"  [{event.agent}]{_activity_text(inner)}", activity)
    finally:
        console.end_text()
        spent.add(stream)
    assert stream.result is not None
    return stream.result


def _build_agent(
    args: argparse.Namespace, console: _Console, *, activity: IO[str]
) -> tuple[Agent, SessionStore | None]:
    if not args.model:
        raise UsageError(
            "--model is required: give provider:model-id, for example qcloud:<model-id>, "
            "or set QAGENTS_MODEL. List QCloud models with: python -m qagents models qcloud"
        )
    if args.session is not None and args.db is None:
        raise UsageError("--session names a conversation stored with --db; add --db PATH.")
    if args.max_cost is not None and args.max_cost <= 0:
        raise UsageError("--max-cost must be more than 0.")
    if args.max_turns is not None and args.max_turns < 1:
        raise UsageError("--max-turns must be at least 1.")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise UsageError("--max-tokens must be at least 1.")
    workspace = Path(args.workspace)
    if not workspace.is_dir():
        raise UsageError(f"--workspace {args.workspace} is not a directory.")

    async def approve(ctx: RunContext[Any], tool: Tool, call: ToolCallPart, reason: Any) -> bool:
        return await _approve(console, activity, _call_text(call.name, call.arguments))

    try:
        tools = _build_tools(_tool_names(args.tools), workspace)
        model = resolve_model(args.model)
        agent = Agent(
            model,
            instructions=args.instructions,
            tools=tools,
            settings=ModelSettings(max_tokens=args.max_tokens) if args.max_tokens else None,
            permissions=Permissions(ask=_ASK_FIRST, on_ask=approve),
        )
    except (ConfigurationError, ValueError) as error:
        raise UsageError(str(error)) from error
    return agent, (SQLiteStore(args.db) if args.db else None)


def _tool_names(listed: str) -> list[str]:
    names: list[str] = []
    for raw in listed.split(","):
        name = raw.strip().lower()
        if not name or name in names:
            continue
        if name not in TOOL_SETS:
            raise UsageError(f"unknown tool set {name!r}. Choose from: {', '.join(TOOL_SETS)}.")
        names.append(name)
    return names


def _build_tools(names: list[str], workspace: Path) -> list[Any]:
    from .toolkits import current_time, file_tools, notes_tools, shell_tool, web_tools

    tools: list[Any] = []
    for name in names:
        if name == "files":
            tools.append(file_tools(workspace, writable=True))
        elif name == "web":
            tools.append(web_tools())
        elif name == "notes":
            tools.append(notes_tools(workspace / "notes"))
        elif name == "time":
            tools.append(current_time)
        elif name == "shell":
            tools.append(shell_tool(workspace))
        elif name == "todo":
            from .toolkits.planning import todo_tools

            tools.append(todo_tools())
        elif name == "ask":
            from .toolkits.human import ask_user

            tools.append(ask_user)
    return tools


async def _check_cost_limit(agent: Agent, args: argparse.Namespace) -> None:
    if args.max_cost is None:
        return
    capabilities = await agent.model.capabilities()
    if (
        capabilities.input_price_per_million is None
        or capabilities.output_price_per_million is None
    ):
        raise UsageError(
            f"--max-cost needs a model with known prices, and {agent.model.provider}:"
            f"{agent.model.name} reports none, so the limit could not be enforced. "
            "Remove --max-cost, or use --max-turns and --max-tokens to bound the run."
        )


def _limits(args: argparse.Namespace, spent: Usage) -> Limits | None:
    """Limits for the next run, or ``None`` when the cost budget is used up."""
    limits = Limits()
    if args.max_turns is not None:
        limits.max_turns = args.max_turns
    if args.max_cost is not None:
        remaining = args.max_cost - (spent.cost_usd or 0.0)
        if remaining <= 0:
            return None
        limits.max_cost_usd = remaining
    return limits


def _session(store: SessionStore | None, args: argparse.Namespace) -> Session:
    if store is None:
        return Session()
    return Session(store, id=args.session or DEFAULT_SESSION)


def _fresh_session(store: SessionStore | None, current: Session) -> Session:
    if store is None:
        return Session()
    base = current.id.split("~", 1)[0]
    return Session(store, id=f"{base}~{uuid.uuid4().hex[:8]}")


async def _approve(console: _Console, stream: IO[str], call: str) -> bool:
    """Ask whether ``call``, as `_call_text` shows it, may run."""
    answer = await console.ask(f"  [approve] {call}? [y/N] ", stream)
    if answer is None:
        console.note("  (no answer: declined)", stream)
        return False
    return answer.strip().lower() in ("y", "yes")


async def _close(agent: Agent, store: SessionStore | None) -> None:
    with contextlib.suppress(Exception):
        await agent.model.aclose()
    if store is not None:
        with contextlib.suppress(Exception):
            await store.aclose()


# --- text ----------------------------------------------------------------------------------


def _activity_text(event: Any) -> str:
    if event.type == "agent_switched":
        return f"  [handoff] {event.from_agent} -> {event.to_agent}"
    if event.type == "tool_call_started":
        return f"  -> {event.call.name}({_arguments_text(event.call.arguments)})"
    result = event.result
    first = result.content.strip().splitlines()[0] if result.content.strip() else ""
    if result.is_error:
        return f"  !! {event.call.name}: {_clip(first, _PREVIEW)}"
    size = f" ({len(result.content):,} chars)" if len(result.content) > _PREVIEW else ""
    return f"  <- {event.call.name}: {_clip(first, _PREVIEW)}{size}"


def _arguments_text(arguments: Mapping[str, Any], limit: int | None = _PREVIEW) -> str:
    """The arguments on one line, clipped to ``limit`` characters unless it is ``None``."""
    pieces = [
        f"{_key(name)}={json.dumps(value, ensure_ascii=False, default=str)}"
        for name, value in arguments.items()
    ]
    text = ", ".join(pieces)
    return text if limit is None else _clip(text, limit)


def _call_text(name: str, arguments: Mapping[str, Any]) -> str:
    """A tool call as approval prompts show it: every argument in full, nothing clipped.

    A short call fits on one line. A longer one gets a line per argument, as
    indented JSON, and text with line breaks is shown line by line after a
    ``|``. Control and format characters are escaped, so what is shown is
    exactly what will run.
    """
    line = f"{name}({_arguments_text(arguments, limit=None)})"
    if len(line) <= _PREVIEW:
        return _visible(line, strict=True)
    lines = [f"{name}("]
    for key, value in arguments.items():
        if isinstance(value, str) and "\n" in value:
            lines.append(f"    {_key(key)} = {len(value):,} characters:")
            lines.extend(f"      | {text}" for text in value.split("\n"))
        else:
            text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
            lines.append(f"    {_key(key)} = " + text.replace("\n", "\n    "))
    lines.append("  )")
    return "\n".join(_visible(item, strict=True) for item in lines)


def _key(name: str) -> str:
    return name if name.isidentifier() else json.dumps(name, ensure_ascii=False)


def _question_text(name: str, arguments: dict[str, Any]) -> str:
    question = arguments.get("question")
    if isinstance(question, str):
        choices = arguments.get("choices")
        listed = f" ({' / '.join(map(str, choices))})" if isinstance(choices, list) else ""
        return f"  [{name}] {question}{listed}\n  answer> "
    return f"  [{name}] {name}({_arguments_text(arguments)}) waits for a result.\n  result> "


def _usage_text(usage: Usage) -> str:
    cost = f"${usage.cost_usd:.6f}" if usage.cost_usd is not None else "cost not reported"
    requests = "1 request" if usage.requests == 1 else f"{usage.requests} requests"
    return (
        f"{requests}, {usage.input_tokens:,} input + {usage.output_tokens:,} output tokens, {cost}"
    )


def _stop_text(result: RunResult) -> str:
    if result.status == "paused":
        waiting = len(result.pending_approvals) + len(getattr(result, "pending_inputs", []))
        return f"paused, {waiting} tool call(s) wait for an answer"
    return result.stop_reason or result.status


def _tools_note(args: argparse.Namespace) -> str:
    names = _tool_names(args.tools)
    return f" (tools: {', '.join(names)})" if names else " (no tools)"


def _session_note(session: Session, store: SessionStore | None) -> str:
    if store is None:
        return "conversation kept in memory"
    return f"session {session.id!r} in {getattr(store, 'path', 'the store')}"


def _catalog_row(entry: dict[str, Any]) -> dict[str, Any]:
    pricing = entry.get("pricing") if isinstance(entry.get("pricing"), dict) else {}
    return {
        "id": entry["id"],
        "name": entry.get("display_name"),
        "context_length": entry.get("context_length"),
        "max_output_tokens": entry.get("max_output_tokens"),
        "capabilities": [str(item) for item in entry.get("capabilities") or []],
        "input_per_million": _number(pricing.get("input_per_million")),
        "output_per_million": _number(pricing.get("output_per_million")),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _price(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}" if value >= 0.01 or value == 0 else f"{value:.4f}"


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _visible(text: str, *, strict: bool = False) -> str:
    """``text``, safe for a terminal: what could restyle, move or hide output is escaped.

    Control characters other than newline and tab (they start terminal escape
    sequences, ``\\r`` moves to the start of the line), bidirectional controls
    and lone surrogates become escapes such as ``\\x1b``. ``strict`` also
    escapes tabs and the other format characters, such as zero-width spaces.
    """
    if not strict:
        text = text.replace("\r\n", "\n")
    escaped = []
    for char in text:
        category = unicodedata.category(char)
        if char == "\n" or (char == "\t" and not strict):
            escaped.append(char)
        elif category in ("Cc", "Cs") or char in _BIDI or (strict and category == "Cf"):
            escaped.append(char.encode("unicode_escape").decode("ascii"))
        else:
            escaped.append(char)
    return "".join(escaped)


def _is_terminal(stream: IO[str]) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


class _Spent:
    """What the command's runs have used so far, runs that raised included."""

    def __init__(self) -> None:
        self.usage = Usage()
        self._before = Usage()
        """What the runs before the current one used."""
        self._run = Usage()
        """What the current run used, across its resumes."""

    def start(self) -> None:
        """A new run begins: what the earlier ones used is final."""
        self._before, self._run = self.usage, Usage()

    def add(self, stream: RunStream) -> None:
        """Count one part of the current run: its start, or a resume.

        A resume reports the usage of the whole run (it continues from the
        journal), so it replaces what the earlier parts reported. A resume
        that failed before it loaded the run reports nothing and changes nothing.
        """
        # The run's running total is kept when the run raises; stream.result is not.
        used = stream._run.usage
        if used != Usage():
            self._run = used
        self.usage = self._before.add(self._run)


def _tolerant(stream: IO[str]) -> IO[str]:
    """``stream``, made to replace characters its encoding cannot write instead of failing."""
    reconfigure = getattr(stream, "reconfigure", None)
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
    if reconfigure is not None and encoding != "utf8":
        with contextlib.suppress(Exception):
            reconfigure(errors="replace")
    return stream


class _Console:
    """The command's input and output, with the bookkeeping streamed text needs.

    Everything but `write` goes through `_visible`, so text from the model
    and from tools cannot reach the terminal as escape sequences.
    """

    def __init__(self, stdin: IO[str], out: IO[str], err: IO[str]) -> None:
        self.stdin = stdin
        self.out = out
        self.err = err
        self._open_line = False
        """Streamed answer text was written to ``out`` without a final newline."""

    def write(self, text: str) -> None:
        """``text`` as it is: for output the command builds itself, such as JSON."""
        self.out.write(text)
        self.out.flush()

    def text(self, text: str) -> None:
        """Streamed answer text."""
        if text:
            self.write(_visible(text))
            self._open_line = not text.endswith("\n")

    def end_text(self) -> None:
        if self._open_line:
            self.write("\n")
            self._open_line = False

    def say(self, line: str) -> None:
        self.end_text()
        self.write(_visible(line) + "\n")

    def note(self, line: str, stream: IO[str]) -> None:
        """A line of activity, on ``out`` (chat) or ``err`` (run)."""
        if stream is self.out:
            self.say(line)
        else:
            stream.write(_visible(line) + "\n")
            stream.flush()

    def error(self, line: str) -> None:
        self.end_text()
        self.err.write(_visible(line) + "\n")
        self.err.flush()

    async def ask(self, prompt: str, stream: IO[str] | None = None) -> str | None:
        """Show ``prompt`` and read one line; ``None`` at the end of input."""
        target = stream if stream is not None else self.out
        if target is self.out:
            self.end_text()
        target.write(_visible(prompt))
        target.flush()
        line = await _read(self.stdin.readline)
        return line if line else None


async def _read(function: Callable[[], str]) -> str:
    """Run a blocking read on a daemon thread, so that Ctrl-C at a prompt exits at once.

    On a thread of the default executor, ``asyncio.run`` would wait at shutdown
    until the read returned.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def deliver(value: str, error: BaseException | None) -> None:
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(value)

    def read() -> None:
        value, error = "", None
        try:
            value = function()
        except BaseException as caught:  # handed to the waiting coroutine
            error = caught
        with contextlib.suppress(RuntimeError):  # the loop has closed meanwhile
            loop.call_soon_threadsafe(deliver, value, error)

    threading.Thread(target=read, name="qagents-input", daemon=True).start()
    return await future


__all__ = ["DEFAULT_SESSION", "QCLOUD_BASE_URL", "TOOL_SETS", "UsageError", "build_parser", "main"]
