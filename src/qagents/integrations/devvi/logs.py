"""Read Devvi's session logs (``$DEVVI_HOME/sessions``), read-only.

    log = read_session_log(find_session_log(sessions_dir(home), "qa-5f0c..."))
    for turn in log.turns():
        print(turn.number, turn.end_kind, turn.final_text)
    print(to_markdown(log))

Devvi writes one log per session at
``<root>/<project key of the cwd>/<session id, escaped>/session.v4.jsonl``
(``packages/session/session-persistence-jsonl``, format version 4): a header line
``{"type": "session", "version": 4, "id", "createdAt", "cwd", ...}``, then one
``SessionEvent`` per line (``{"type", "seq", "time", "data", ...}``). qagents' default
patch keeps the logs uncompressed (``compression: "none"``); a ``.jsonl.zstd`` log cannot
be read here (the standard library has no Zstandard before Python 3.14).

Rules that keep this safe:

* Nothing is ever written. Devvi owns the log (a single-writer lock, ``session.lock``,
  and a named semaphore on Windows).
* Only complete lines count: a last line without its newline is a torn write and is
  ignored, as Devvi does when it recovers.
* A line that is not JSON in the middle of the log is an error (`DevviLogError`): the log
  is damaged, so nothing read from it is trusted.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...external import ExternalAgentError
from .events import content_text, user_receipt

FORMAT_VERSION = 4
LOG_NAME = f"session.v{FORMAT_VERSION}.jsonl"
COMPRESSED_SUFFIX = ".zstd"
_SAFE = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")


class DevviLogError(ExternalAgentError):
    """A Devvi session log cannot be read (missing, compressed, damaged, another format)."""


# --- where logs live --------------------------------------------------------------------------


def encode_segment(raw: str) -> str:
    """A session id as Devvi's one-path-segment form (``encodeSegment``): safe characters
    stay, every other UTF-16 code unit becomes ``~XXXX``."""
    if not raw:
        raise ValueError("cannot encode an empty path segment")
    if raw == ".":
        return "~002E"
    if raw == "..":
        return "~002E~002E"
    return "".join(
        char if char in _SAFE else "".join(f"~{unit:04X}" for unit in _utf16_units(char))
        for char in raw
    )


def project_key(cwd: str) -> str:
    """Devvi's directory name for a project path (``projectKey``): separators and the drive
    colon become ``-`` (runs collapse), other unsafe code units ``~XXXX``, bounded length."""
    if not cwd:
        raise ValueError("cannot encode an empty project path")
    readable: list[str] = []
    separator_run = False
    for char in cwd:
        if char in "/\\:":
            if not separator_run:
                readable.append("-")
            separator_run = True
        elif char != "~" and char in _SAFE:
            readable.append(char)
            separator_run = False
        else:
            readable.append("".join(f"~{unit:04X}" for unit in _utf16_units(char)))
            separator_run = False
    slug = "".join(readable).lstrip("-") or "root"
    return f"--{slug[:251]}--"


def _utf16_units(char: str) -> list[int]:
    data = char.encode("utf-16-le", errors="surrogatepass")
    return [int.from_bytes(data[index : index + 2], "little") for index in range(0, len(data), 2)]


def sessions_dir(home: str | Path) -> Path:
    """``<home>/sessions``: where qagents' default patch puts the logs."""
    return Path(home) / "sessions"


def session_log_path(root: str | Path, session_id: str, cwd: str | None) -> Path:
    """Where Devvi writes the log of ``session_id`` with header ``cwd`` under ``root``."""
    project = project_key(cwd) if cwd else "_no-cwd"
    return Path(root) / project / encode_segment(session_id) / LOG_NAME


def find_session_log(root: str | Path, session_id: str, *, cwd: str | None = None) -> Path | None:
    """The log file of ``session_id`` under ``root`` (a sessions directory), or ``None``.

    With ``cwd`` the expected path is tried first; then every project directory is
    searched (the project key is lossy, and the cwd may be unknown). A compressed log is
    returned too, so the caller can say why it cannot be read.
    """
    base = Path(root)
    if cwd:
        expected = session_log_path(base, session_id, cwd)
        for candidate in (expected, expected.with_name(expected.name + COMPRESSED_SUFFIX)):
            if candidate.is_file():
                return candidate
    segment = encode_segment(session_id)
    try:
        projects = sorted(entry for entry in base.iterdir() if entry.is_dir())
    except OSError:
        return None
    for project in projects:
        for name in (LOG_NAME, LOG_NAME + COMPRESSED_SUFFIX):
            candidate = project / segment / name
            if candidate.is_file():
                return candidate
    return None


# --- reading ------------------------------------------------------------------------------


@dataclass
class TurnRecord:
    """One turn of a session, as the log shows it."""

    number: int | None
    """``turn/start.turn``; ``None`` when a prompt was received but no turn started."""
    receipt_index: int | None = None
    """Index (in `SessionLog.events`) of the ``agent/inbox/spliced`` that received its prompt."""
    message_ids: list[str] = field(default_factory=list)
    """Ids of the user messages that started it."""
    prompt_text: str = ""
    start_index: int | None = None
    end_index: int | None = None
    end_reason: dict[str, Any] | None = None
    """``turn/end.reason``; ``None`` while the turn is open (or was cut off by a crash)."""
    final_text: str = ""
    """Text of the turn's last assistant message."""
    usage: list[dict[str, Any]] = field(default_factory=list)
    """The raw ``usage`` of each assistant message."""
    tool_calls: list[str] = field(default_factory=list)

    @property
    def end_kind(self) -> str | None:
        kind = self.end_reason.get("kind") if self.end_reason else None
        return kind if isinstance(kind, str) else None

    @property
    def finished(self) -> bool:
        """The turn ended by itself (not open, not closed after a crash or an abort)."""
        return self.end_kind is not None and self.end_kind not in ("interrupted", "aborted")

    @property
    def last_index(self) -> int | None:
        return self.end_index if self.end_index is not None else self.start_index


@dataclass
class SessionLog:
    """A parsed session log: the header and the events, in order."""

    path: Path
    header: dict[str, Any]
    events: list[dict[str, Any]]
    torn_tail: bool = False
    """The last line was incomplete and was ignored."""

    @property
    def session_id(self) -> str:
        return str(self.header.get("id") or "")

    @property
    def cwd(self) -> str | None:
        value = self.header.get("cwd")
        return value if isinstance(value, str) else None

    def turns(self) -> list[TurnRecord]:
        """The turns, each with the prompt that started it (see `TurnRecord`)."""
        turns: list[TurnRecord] = []
        waiting: TurnRecord | None = None
        current: TurnRecord | None = None
        for index, event in enumerate(self.events):
            kind = event.get("type")
            data = event.get("data") if isinstance(event.get("data"), Mapping) else {}
            received = user_receipt(event)
            if received and (current is None or current.end_reason is not None):
                if waiting is None:
                    waiting = TurnRecord(number=None, receipt_index=index)
                    turns.append(waiting)
                waiting.message_ids.extend(message_id for message_id, _ in received)
                texts = [text for _, text in received if text]
                waiting.prompt_text = "\n\n".join([waiting.prompt_text, *texts]).strip()
            elif kind == "turn/start":
                number = data.get("turn")
                if waiting is not None:
                    current, waiting = waiting, None
                else:
                    current = TurnRecord(number=None)
                    turns.append(current)
                current.number = number if isinstance(number, int) else None
                current.start_index = index
            elif current is not None and current.end_reason is None:
                if kind == "turn/end":
                    reason = data.get("reason")
                    current.end_reason = dict(reason) if isinstance(reason, Mapping) else {}
                    current.end_index = index
                elif kind == "assistant/message":
                    message = data.get("message")
                    text = (
                        content_text(message.get("content")) if isinstance(message, Mapping) else ""
                    )
                    if text:
                        current.final_text = text
                    if isinstance(data.get("usage"), Mapping):
                        current.usage.append(dict(data["usage"]))
                elif kind == "tool/call" and isinstance(data.get("name"), str):
                    current.tool_calls.append(data["name"])
        return turns

    def last_turn(self) -> TurnRecord | None:
        turns = self.turns()
        return turns[-1] if turns else None

    def turn_events(self, turn: TurnRecord) -> list[dict[str, Any]]:
        """The events from the turn's receipt (or start) through its end (or the log's end)."""
        first = turn.receipt_index if turn.receipt_index is not None else turn.start_index
        if first is None:
            return []
        last = turn.end_index if turn.end_index is not None else len(self.events) - 1
        return self.events[first : last + 1]


def read_session_log(path: str | Path) -> SessionLog:
    """Parse a session log (format v4, uncompressed).

    Raises:
        DevviLogError: The file is missing or compressed, its header is not a v4 header,
            or a complete line is not a JSON object.
    """
    file = Path(path)
    if file.name.endswith(COMPRESSED_SUFFIX):
        raise DevviLogError(
            f"{file} is compressed (Zstandard). qagents reads uncompressed logs only: keep "
            "the default patch row session-persistence-jsonl with compression 'none'."
        )
    try:
        data = file.read_bytes()
    except OSError as error:
        raise DevviLogError(f"Cannot read the Devvi session log {file}: {error}") from error
    lines = data.split(b"\n")
    torn = lines[-1] != b""
    complete = lines[:-1]
    records: list[dict[str, Any]] = []
    for number, raw in enumerate(complete, start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise DevviLogError(f"{file}: line {number} is not JSON ({error})") from error
        if not isinstance(value, dict):
            raise DevviLogError(f"{file}: line {number} is not a JSON object")
        records.append(value)
    if not records:
        raise DevviLogError(f"{file} has no header line")
    header = records[0]
    if header.get("type") != "session" or header.get("version") != FORMAT_VERSION:
        raise DevviLogError(
            f"{file} is not a session format v{FORMAT_VERSION} log "
            f"(header type {header.get('type')!r}, version {header.get('version')!r})."
        )
    return SessionLog(path=file, header=header, events=records[1:], torn_tail=torn)


def peek_session_log(path: str | Path) -> tuple[dict[str, Any], int]:
    """The header of a log and how many events it holds now, without parsing the events.

    The count is what `read_session_log` returns in ``SessionLog.events`` for the same
    bytes (complete, non-blank lines after the header), so an event index taken from it
    can be compared with `TurnRecord.receipt_index` later.

    Raises:
        DevviLogError: As `read_session_log` does for the file and its header.
    """
    file = Path(path)
    if file.name.endswith(COMPRESSED_SUFFIX):
        raise DevviLogError(f"{file} is compressed (Zstandard); qagents cannot read it.")
    try:
        data = file.read_bytes()
    except OSError as error:
        raise DevviLogError(f"Cannot read the Devvi session log {file}: {error}") from error
    complete = [raw for raw in data.split(b"\n")[:-1] if raw.strip()]
    if not complete:
        raise DevviLogError(f"{file} has no header line")
    try:
        header = json.loads(complete[0].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise DevviLogError(f"{file}: the header is not JSON ({error})") from error
    if (
        not isinstance(header, dict)
        or header.get("type") != "session"
        or header.get("version") != FORMAT_VERSION
    ):
        raise DevviLogError(f"{file} is not a session format v{FORMAT_VERSION} log.")
    return header, len(complete) - 1


@dataclass(frozen=True)
class SessionSummary:
    """One log found by `list_sessions`."""

    session_id: str
    path: Path
    cwd: str | None
    created_at: int | None
    parent_session: str | None
    compressed: bool


def list_sessions(root: str | Path) -> list[SessionSummary]:
    """Every session log under ``root`` (a sessions directory), newest first.

    Only header lines are read. Unreadable logs are skipped.
    """
    found: list[SessionSummary] = []
    for path in _log_files(Path(root)):
        compressed = path.name.endswith(COMPRESSED_SUFFIX)
        header: dict[str, Any] = {}
        if not compressed:
            try:
                with path.open("rb") as handle:
                    header = json.loads(handle.readline().decode("utf-8"))
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            if not isinstance(header, dict) or header.get("type") != "session":
                continue
        created = header.get("createdAt")
        parent = header.get("parentSession")
        found.append(
            SessionSummary(
                session_id=str(header.get("id") or path.parent.name),
                path=path,
                cwd=header.get("cwd") if isinstance(header.get("cwd"), str) else None,
                created_at=created if isinstance(created, int) else None,
                parent_session=parent if isinstance(parent, str) else None,
                compressed=compressed,
            )
        )
    found.sort(key=lambda item: item.created_at or 0, reverse=True)
    return found


def _log_files(root: Path) -> Iterator[Path]:
    try:
        projects = [entry for entry in root.iterdir() if entry.is_dir()]
    except OSError:
        return
    for project in projects:
        try:
            sessions = [entry for entry in project.iterdir() if entry.is_dir()]
        except OSError:
            continue
        for session in sessions:
            for name in (LOG_NAME, LOG_NAME + COMPRESSED_SUFFIX):
                if (session / name).is_file():
                    yield session / name


# --- matching a prompt (crash re-attach) ---------------------------------------------------------


def find_prompt_turn(
    log: SessionLog,
    *,
    message_id: str | None = None,
    prompts: Sequence[str] = (),
    after: int | None = None,
) -> TurnRecord | None:
    """The turn a prompt started, found by its message id or, failing that, its text.

    Without a message id, only the session's most recent user prompt is considered, and it
    matches when its text is one of ``prompts`` and, with ``after``, when it was received
    at event index ``after`` or later. ``after`` is how many events the log held before the
    call sent its first prompt (`peek_session_log`): in a session that earlier calls
    continued, their prompts come before it, and one with the same text (``"Continue"``,
    or the resume prompt of an earlier re-attach) is not this call's. qagents runs one turn
    per session at a time, so a prompt of this call is always the session's last one.
    """
    turns = log.turns()
    if message_id:
        for turn in reversed(turns):
            if message_id in turn.message_ids:
                return turn
        return None
    for turn in reversed(turns):
        if turn.receipt_index is None:
            continue
        if after is not None and turn.receipt_index < after:
            return None  # the last prompt is older than this call
        wanted = {text.strip() for text in prompts if text.strip()}
        return turn if turn.prompt_text.strip() in wanted else None
    return None


# --- Markdown -----------------------------------------------------------------------------------


def to_markdown(log: SessionLog, *, reasoning: bool = False, results_limit: int = 2000) -> str:
    """A readable transcript of the log for reviews and evals (prompts, replies, tool calls).

    A read-only reimplementation in the spirit of Devvi's ``formatSessionMarkdown``; it is
    not byte-identical to it.
    """
    lines = [f"# Devvi session {log.session_id}", ""]
    if log.cwd:
        lines += [f"Working directory: `{log.cwd}`", ""]
    for event in log.events:
        kind = event.get("type")
        data = event.get("data") if isinstance(event.get("data"), Mapping) else {}
        if received := user_receipt(event):
            for _, text in received:
                lines += ["## User", "", text, ""]
        elif kind == "assistant/message":
            message = data.get("message") if isinstance(data.get("message"), Mapping) else {}
            for block in message.get("content") or []:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "text" and block.get("text"):
                    lines += ["## Assistant", "", str(block["text"]), ""]
                elif block.get("type") == "reasoning" and reasoning and block.get("text"):
                    lines += ["> " + line for line in str(block["text"]).splitlines()] + [""]
        elif kind == "tool/call":
            lines += [f"### Tool call `{data.get('name')}`", "", "```json"]
            lines += [str(data.get("arguments") or "{}"), "```", ""]
        elif kind == "tool/result":
            message = data.get("message") if isinstance(data.get("message"), Mapping) else {}
            text = content_text(message.get("content"))
            if len(text) > results_limit:
                text = text[:results_limit] + "\n..."
            label = "Tool error" if message.get("isError") else "Tool result"
            lines += [f"### {label}", "", "```text", text, "```", ""]
        elif kind == "turn/end":
            reason = data.get("reason") if isinstance(data.get("reason"), Mapping) else {}
            lines += [f"*Turn {data.get('turn')} ended: {reason.get('kind')}*", ""]
    return "\n".join(lines).rstrip() + "\n"
