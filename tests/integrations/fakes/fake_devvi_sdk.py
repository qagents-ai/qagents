"""A fake Devvi CLI that speaks Devvi's SDK JSON-RPC over stdio, for the SDK transport tests.

Started as ``[python, -I, fake_devvi_sdk.py, *options]``; the launcher appends Devvi's own
flags (``--profile sdk --patch FILE``), which are read too. Standard library only.

    --version                 print the version (``--version-text``, default 0.2.1-rc.3)
    --script FILE             the turns to play: a script of notifications (``*.jsonl``,
                              turns separated by ``{"$turn": true}``) or a Devvi session log
                              (first line ``{"type": "session", ...}``; one turn per prompt)
    --record FILE             append what arrives (JSON lines) and a start line
    --missing                 session/resume always answers "not found"
    --no-working-directory    session/working-directory/get is an unknown method
    --no-message-id           answer session/prompt with {} (a protocol violation)
    --ignore-shutdown         never answer shutdown (the client must kill the process)
    --fail-initialize TEXT    answer initialize with this error
    --exit-on-start CODE      exit at once (after writing --stderr)
    --stderr TEXT             write this to stderr at start
    --delay SECONDS           pause between notifications

What it does like Devvi (``packages/sdk/server/src/server.ts``):

* ``initialize`` answers ``serverInfo {deepseek-harness-sdk-runtime, 0.0.1}``; the cwd must
  be absolute.
* ``session/prompt`` on an unknown id creates the session (writes the log header), sends
  the receipt (``agent/inbox/spliced`` with the new message id) before its answer
  ``{messageId}``, then plays the next turn. A session whose log exists but that was not
  resumed is refused (the real server's create would collide with the stored session).
* ``session/resume`` opens a stored session (``session "<id>" not found`` otherwise;
  "already open" in the same process) and closes an open last turn with
  ``turn/end {kind: interrupted}``, as Devvi's crash repair does.
* ``session/working-directory/get`` answers ``{cwd}`` of an open session: the cwd of its
  log header, kept on resume as Devvi keeps it (``preserving the stored session identity and
  directory``), whatever directory the process was initialized with.
* ``session/wait`` answers once the turn's notifications are all sent.
* ``shutdown`` closes an open turn as ``aborted``, answers ``{}`` and exits 0. EOF exits 0.
* Session logs are written as Devvi writes them: ``<root>/<project key>/<id>/session.v4.jsonl``,
  the root taken from the ``session-persistence-jsonl`` row of the patch
  (``.jsonl.zstd`` when that row asks for ``zstd``; the content stays plain).

Script tokens are replaced in every string: ``{{session:1}}`` and ``{{sessionId}}`` by the
session id, ``{{session:2}}`` by a child id, ``{{cwd}}`` by the cwd, and the message token
of the turn's first user receipt (``{{message:1}}``) by the prompt's message id.

Directives in scripts: ``{"$approval": {"toolName", "callId"?, "reason"?, "sessionId"?},
"$on": {"allowed-once": [...], "rejected": [...], "cancelled": [...]}}`` asks the client
and plays the branch of its answer (the ``approval/asked`` and ``approval/decided`` events
are sent around it); ``{"$hang": true}`` stops sending until shutdown; ``{"$exit": code,
"$stderr": text}`` exits; ``{"$sleep": seconds}`` waits; ``{"$if_cwd": name, "$then":
[...], "$else": [...]}`` plays ``$then`` when the session's cwd is a folder called
``name``, else ``$else`` (one script, different turns per workspace). A turn holding
``{"$no_answer": true}`` never answers ``session/prompt``: it plays up to that item, then
stops sending until shutdown (the client dies waiting for the answer).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

VERSION = "0.2.1-rc.3"
LOG_NAME = "session.v4.jsonl"
SAFE = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
WRITE = threading.Lock()
STOP = threading.Event()


def encode_segment(raw: str) -> str:
    if raw == ".":
        return "~002E"
    if raw == "..":
        return "~002E~002E"
    out = []
    for char in raw:
        if char in SAFE:
            out.append(char)
        else:
            data = char.encode("utf-16-le", errors="surrogatepass")
            for index in range(0, len(data), 2):
                out.append(f"~{int.from_bytes(data[index : index + 2], 'little'):04X}")
    return "".join(out)


def project_key(cwd: str) -> str:
    readable, run = [], False
    for char in cwd:
        if char in "/\\:":
            if not run:
                readable.append("-")
            run = True
        elif char != "~" and char in SAFE:
            readable.append(char)
            run = False
        else:
            readable.append(encode_segment(char))
            run = False
    slug = "".join(readable).lstrip("-") or "root"
    return "--" + slug[:251] + "--"


def send(message: dict) -> None:
    data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
    with WRITE:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()


class Recorder:
    def __init__(self, path: str | None) -> None:
        self.path = path
        self.lock = threading.Lock()

    def write(self, item: dict) -> None:
        if not self.path:
            return
        with self.lock, open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


# --- scripts ---------------------------------------------------------------------------------


def is_user_receipt(event: dict) -> bool:
    if event.get("type") != "agent/inbox/spliced":
        return False
    inserted = (event.get("data") or {}).get("inserted") or []
    return any(
        isinstance(item, dict) and (item.get("source") or {}).get("kind", "user") == "user"
        for item in inserted
    )


def notification(event: dict, session: str = "{{session:1}}") -> dict:
    return {"method": "session.event", "params": {"sessionId": session, "event": event}}


def status(value: str, session: str = "{{session:1}}") -> dict:
    return {"method": "session.status", "params": {"sessionId": session, "status": value}}


def log_segments(rows: list[dict]) -> list[list[dict]]:
    """A Devvi session log as turns: one per user receipt, with statuses around them."""
    segments: list[list[dict]] = []
    current: list[dict] | None = None
    closed = True
    for index, row in enumerate(rows):
        event = dict(row)
        event.setdefault("seq", index)
        event.setdefault("time", 0)
        kind = event.get("type")
        if is_user_receipt(event) and closed:
            current = [notification(event), status("running")]
            segments.append(current)
            closed = False
            continue
        if current is None:
            continue
        data = event.get("data") or {}
        if kind == "approval/asked":
            ask = {"toolName": data.get("toolName")}
            for key in ("callId", "reason"):
                if key in data:
                    ask[key] = data[key]
            current.append({"$approval": ask})
            continue
        if kind == "approval/decided":
            continue  # sent by the $approval directive, with the client's real answer
        current.append(notification(event))
        if kind == "turn/end":
            current.append(status("idle"))
            closed = True
    return segments


def load_script(path: str | None) -> list[list[dict]]:
    if not path:
        return []
    items = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if items and items[0].get("type") == "session":
        return log_segments(items[1:])
    segments: list[list[dict]] = [[]]
    for item in items:
        if item.get("$turn"):
            segments.append([])
        else:
            segments[-1].append(item)
    return [segment for segment in segments if segment]


def message_token(segment: list[dict]) -> str | None:
    for item in segment:
        event = (item.get("params") or {}).get("event") if item.get("method") else None
        if isinstance(event, dict) and is_user_receipt(event):
            for message in event["data"]["inserted"]:
                if isinstance(message.get("id"), str):
                    return message["id"]
    return None


def substitute(value, tokens: dict):
    if isinstance(value, str):
        for token, replacement in tokens.items():
            if token in value:
                value = value.replace(token, replacement)
        return value
    if isinstance(value, list):
        return [substitute(item, tokens) for item in value]
    if isinstance(value, dict):
        return {key: substitute(item, tokens) for key, item in value.items()}
    return value


# --- sessions --------------------------------------------------------------------------------


class Session:
    def __init__(self, session_id: str, cwd: str, path: Path) -> None:
        self.id = session_id
        self.cwd = cwd
        self.path = path
        self.seq = 0
        self.prompts = 0
        self.open_turn: int | None = None

    def append(self, event: dict) -> None:
        self.seq = max(self.seq, int(event.get("seq") or 0) + 1)
        kind = event.get("type")
        if kind == "turn/start":
            self.open_turn = (event.get("data") or {}).get("turn")
        elif kind == "turn/end":
            self.open_turn = None
        elif is_user_receipt(event):
            self.prompts += 1
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def close_open_turn(self, kind: str) -> None:
        if self.open_turn is None:
            return
        event = {
            "type": "turn/end",
            "seq": self.seq,
            "time": int(time.time() * 1000),
            "data": {"turn": self.open_turn, "reason": {"kind": kind}},
        }
        self.append(event)
        send(notification(event, self.id))


class Fake:
    def __init__(self, options: argparse.Namespace, patches: list[str]) -> None:
        self.options = options
        self.recorder = Recorder(options.record)
        self.segments = load_script(options.script)
        self.sessions: dict[str, Session] = {}
        self.cwd = os.getcwd()
        self.root, self.suffix = self._persistence(patches)
        self.answers: dict[str, queue.Queue] = {}
        self.player: threading.Thread | None = None
        self.playing: Session | None = None
        self.next_id = 0

    def _persistence(self, patches: list[str]) -> tuple[Path, str]:
        home = os.environ.get("DEVVI_HOME") or os.environ.get("DSH_HOME") or "."
        root, suffix = Path(home) / "sessions", ""
        for patch in patches:
            try:
                rows = json.loads(Path(patch).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for row in rows if isinstance(rows, list) else []:
                if isinstance(row, dict) and row.get("id") == "session-persistence-jsonl":
                    config = row.get("config") or {}
                    root = Path(config.get("root") or root)
                    suffix = ".zstd" if config.get("compression", "zstd") == "zstd" else ""
        return root, suffix

    def log_path(self, session_id: str, cwd: str) -> Path:
        return self.root / project_key(cwd) / encode_segment(session_id) / (LOG_NAME + self.suffix)

    def find_log(self, session_id: str) -> Path | None:
        if not self.root.is_dir():
            return None
        for project in self.root.iterdir():
            candidate = project / encode_segment(session_id) / (LOG_NAME + self.suffix)
            if candidate.is_file():
                return candidate
        return None

    # -- requests -------------------------------------------------------------------------------

    def handle(self, message: dict) -> None:
        method, request_id, params = message.get("method"), message.get("id"), message.get("params")
        if method is None:  # an answer to one of our requests
            waiting = self.answers.pop(str(request_id), None)
            if waiting is not None:
                waiting.put(message)
            return
        if request_id is None:
            return
        try:
            result = self.dispatch(method, params or {}, request_id)
        except FakeError as error:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": error.code, "message": str(error)},
                }
            )
            return
        if result is not None:
            send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def dispatch(self, method: str, params: dict, request_id):
        if method == "initialize":
            if self.options.fail_initialize:
                raise FakeError(self.options.fail_initialize)
            cwd = params.get("cwd")
            if not isinstance(cwd, str) or not os.path.isabs(cwd):
                raise FakeError("initialize cwd must be absolute")
            self.cwd = cwd
            return {"serverInfo": {"name": "deepseek-harness-sdk-runtime", "version": "0.0.1"}}
        if method == "session/resume":
            return self.resume(params.get("sessionId"))
        if method == "session/prompt":
            return self.prompt(params, request_id)
        if method == "session/working-directory/get" and not self.options.no_working_directory:
            session = self.sessions.get(params.get("sessionId"))
            if session is None:
                raise FakeError(f"SDK session is not open: {params.get('sessionId')}")
            return {"cwd": session.cwd}
        if method == "session/wait":
            threading.Thread(target=self.wait_then_answer, args=(request_id,), daemon=True).start()
            return None
        if method == "shutdown":
            if self.options.ignore_shutdown:
                return None
            STOP.set()
            for session in list(self.sessions.values()):
                session.close_open_turn("aborted")  # Devvi disposes its agents on shutdown
            send({"jsonrpc": "2.0", "id": request_id, "result": {}})
            os._exit(0)
        raise FakeError(f"unknown DeepSeek Harness SDK runtime method: {method}", -32601)

    def resume(self, session_id) -> dict:
        if not isinstance(session_id, str) or not session_id:
            raise FakeError("session/resume requires a non-empty sessionId string")
        if session_id in self.sessions:
            raise FakeError(f"SDK session is already open or opening: {session_id}")
        path = None if self.options.missing else self.find_log(session_id)
        if path is None:
            raise FakeError(f'session "{session_id}" not found')
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        session = Session(session_id, rows[0].get("cwd") or self.cwd, path)
        for index, row in enumerate(rows[1:]):
            kind = row.get("type")
            session.seq = max(session.seq, int(row.get("seq", index)) + 1)
            if kind == "turn/start":
                session.open_turn = (row.get("data") or {}).get("turn")
            elif kind == "turn/end":
                session.open_turn = None
            elif is_user_receipt(row):
                session.prompts += 1
        self.sessions[session_id] = session
        session.close_open_turn("interrupted")
        return {}

    def prompt(self, params: dict, request_id):
        session_id = params.get("sessionId")
        blocks = params.get("contentBlocks")
        if not isinstance(session_id, str) or not isinstance(blocks, list):
            raise FakeError("session/prompt requires sessionId and contentBlocks", -32602)
        session = self.sessions.get(session_id)
        if session is None:
            if self.find_log(session_id) is not None:
                raise FakeError(f'session "{session_id}" already exists')
            path = self.log_path(session_id, self.cwd)
            path.parent.mkdir(parents=True, exist_ok=True)
            header = {
                "type": "session", "version": 4, "id": session_id,
                "createdAt": int(time.time() * 1000), "cwd": self.cwd,
                "isSeeded": False, "delegationDepth": 0,
            }  # fmt: skip
            path.write_text(json.dumps(header) + "\n", encoding="utf-8", newline="\n")
            session = Session(session_id, self.cwd, path)
            self.sessions[session_id] = session
        if self.player is not None and self.player.is_alive():
            raise FakeError("fake: a turn is still running")
        index = session.prompts
        if index >= len(self.segments):
            raise FakeError("fake: no scripted turn left")
        segment = self.segments[index]
        message_id = "msg-" + uuid.uuid4().hex[:12]
        tokens = {
            "{{session:1}}": session_id,
            "{{sessionId}}": session_id,
            "{{session:2}}": session_id + "-child-1",
            "{{cwd}}": session.cwd,
        }
        token = message_token(segment)
        if token:
            tokens[token] = message_id
        items = [substitute(item, tokens) for item in segment]
        first = items[0] if items else None
        rest = items
        if first is not None and first.get("method") == "session.event":
            event = first["params"]["event"]
            if is_user_receipt(event):  # the receipt comes before the answer, as in Devvi
                self.emit(session, first)
                rest = items[1:]
        if not any("$no_answer" in item for item in rest):
            result = {} if self.options.no_message_id else {"messageId": message_id}
            send({"jsonrpc": "2.0", "id": request_id, "result": result})
        self.playing = session
        self.player = threading.Thread(target=self.play, args=(session, rest), daemon=True)
        self.player.start()
        return None

    # -- playing --------------------------------------------------------------------------------

    def emit(self, session: Session, item: dict) -> None:
        params = item.get("params") or {}
        if item.get("method") == "session.event" and params.get("sessionId") == session.id:
            session.append(params["event"])
        send({"jsonrpc": "2.0", **item})

    def play(self, session: Session, items: list[dict]) -> None:
        try:
            self._items(session, items)
        finally:
            if self.playing is session and not STOP.is_set():
                self.playing = None

    def _items(self, session: Session, items: list[dict]) -> None:
        for item in items:
            if STOP.is_set():
                return
            if self.options.delay:
                time.sleep(self.options.delay)
            if "$hang" in item or "$no_answer" in item:
                STOP.wait()
                return
            if "$exit" in item:
                if item.get("$stderr"):
                    sys.stderr.write(item["$stderr"] + "\n")
                    sys.stderr.flush()
                os._exit(int(item["$exit"]))
            if "$sleep" in item:
                time.sleep(float(item["$sleep"]))
                continue
            if "$if_cwd" in item:
                branch = "$then" if Path(session.cwd).name == item["$if_cwd"] else "$else"
                self._items(session, item.get(branch) or [])
                continue
            if "$approval" in item:
                outcome = self.ask(session, item["$approval"])
                self._items(session, (item.get("$on") or {}).get(outcome, []))
                continue
            self.emit(session, item)

    def ask(self, session: Session, ask: dict) -> str:
        self.next_id += 1
        approval_id = f"approval-{self.next_id}"
        request_id = f"fake-{self.next_id}"
        params = {"sessionId": ask.get("sessionId") or session.id, "toolName": ask.get("toolName")}
        for key in ("callId", "reason"):
            if ask.get(key) is not None:
                params[key] = ask[key]
        asked = {"type": "approval/asked", "seq": session.seq, "time": 0,
                 "data": {"id": approval_id, **{k: v for k, v in params.items() if k != "sessionId"}}}  # fmt: skip
        self.emit(session, notification(asked, session.id))
        waiting: queue.Queue = queue.Queue()
        self.answers[request_id] = waiting
        send({"jsonrpc": "2.0", "id": request_id, "method": "approval/request", "params": params})
        answer = waiting.get()
        result = answer.get("result") if isinstance(answer.get("result"), dict) else {}
        outcome = result.get("outcome")
        if outcome not in ("allowed-once", "rejected", "cancelled"):
            outcome = "cancelled"  # anything else refuses, as Devvi's server does
        decided = {"type": "approval/decided", "seq": session.seq, "time": 0,
                   "data": {"id": approval_id, "outcome": outcome}}  # fmt: skip
        self.emit(session, notification(decided, session.id))
        return outcome

    def wait_then_answer(self, request_id) -> None:
        player = self.player
        if player is not None:
            player.join()
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})


class FakeError(Exception):
    def __init__(self, message: str, code: int = -32603) -> None:
        super().__init__(message)
        self.code = code


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--version-text", default=VERSION)
    parser.add_argument("--script")
    parser.add_argument("--record")
    parser.add_argument("--missing", action="store_true")
    parser.add_argument("--no-working-directory", action="store_true")
    parser.add_argument("--no-message-id", action="store_true")
    parser.add_argument("--ignore-shutdown", action="store_true")
    parser.add_argument("--fail-initialize")
    parser.add_argument("--exit-on-start", type=int)
    parser.add_argument("--stderr")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--profile")
    parser.add_argument("--patch", action="append", default=[])
    parser.add_argument("--dump-config", action="store_true")
    options, _ = parser.parse_known_args(argv)
    if options.stderr:
        sys.stderr.write(options.stderr + "\n")
        sys.stderr.flush()
    if options.exit_on_start is not None:
        return options.exit_on_start
    if options.version and not options.profile:
        print(options.version_text)
        return 0
    if options.dump_config:
        for patch in options.patch:
            print(Path(patch).read_text(encoding="utf-8"))
        return 0
    fake = Fake(options, options.patch)
    names = sorted(os.environ)
    fake.recorder.write(
        {
            "start": {
                "argv": argv,
                "pid": os.getpid(),
                "cwd": os.getcwd(),
                "env_names": names,
                "home": {key: os.environ.get(key) for key in ("DEVVI_HOME", "DSH_HOME")},
                "permission_mode": os.environ.get("DSH_PERMISSION_MODE"),
                "telemetry_disabled": os.environ.get("DSH_TELEMETRY_DISABLED"),
            }
        }
    )
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            STOP.set()
            os._exit(0)
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        fake.recorder.write(message)
        fake.handle(message)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
