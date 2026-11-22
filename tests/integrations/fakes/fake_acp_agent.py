"""A scripted ACP v1 agent for the tests (standard library only).

Run as ``python -I fake_acp_agent.py --script FIXTURE [options]``. It plays a fixture
from ``tests/fixtures/devvi/acp`` (or one a test writes): it reads the client's JSON-RPC
lines on stdin, checks each against the script, and writes the recorded agent lines on
stdout. A mismatch is written to stderr and ends the process with exit code 3, so the
client sees the connection break and the test sees why in the stderr tail.

A fixture is JSON Lines. Each line is one of:

| Line | Does |
|---|---|
| a JSON-RPC message (has ``"jsonrpc"``) | written as recorded, byte for byte, after the placeholders are replaced; a response gets the id of the client request it answers |
| ``{"fixture": ...}``, ``{"comment": ...}`` | nothing (a header or a note) |
| ``{"expect": {"id", "method", "params"}}`` | reads the next client message and checks it: same method, the params given (a subset); ``id`` is the recorded id, mapped to the client's own |
| ``{"answer": {"id", "accept": [result, ...]}, "otherwise": FILE}`` | waits for the client's answer to the agent's request ``id``; if it is not one of ``accept``, continues in ``FILE`` after its own ``answer`` line for that id (a branch), else fails |
| ``{"raw": TEXT}`` | writes ``TEXT`` and a newline (a malformed line) |
| ``{"big_chunk": N}`` | writes an ``agent_message_chunk`` whose text is ``N`` bytes of ``x`` |
| ``{"sleep": SECONDS}`` | waits |
| ``{"exit": CODE, "stderr": TEXT}`` | writes ``TEXT`` to stderr and exits with ``CODE`` |
| ``{"hang": true}`` | stops reading and writing, and never exits by itself |
| ``{"write_file": {"path", "text"}}`` | writes ``text`` to ``path`` under the session's ``cwd`` (an edit the agent makes) |

A branch continues in another recording: Devvi's approved and rejected transcripts were
recorded separately, so the tool call ids after the branch are that recording's own.

Placeholders, as Devvi's snapshots write them: ``{{sessionId}}`` (``--session-id``,
default ``fake-session-1``), ``{{messageId}}`` (``fake-message-1``), ``{{cwd}}`` (the
``cwd`` of the client's ``session/new``, ``session/resume`` or ``session/load``) and
``"{{usedTokens}}"`` (the number 1234). They are replaced in recorded lines and in
expectations.

Param patterns in ``expect``: ``"$any"`` (present), ``"$absent"``, ``"$absolute"`` (an
absolute path), ``"$list"``, ``"$string"``; a dict matches a subset; anything else must
be equal.

After the last line it reads until end of input; a request then is a mismatch, a
notification is ignored.

Options:

* ``--log FILE``: append every client message to ``FILE``, one JSON line each.
* ``--hang-after-cancel``: once a ``session/cancel`` matched, behave like ``{"hang": true}``
  (an agent that does not stop).
* ``--child HEARTBEAT``: at start, start a child process that appends a byte to
  ``HEARTBEAT`` every 50 ms, and write its pid to ``HEARTBEAT.pid``, to prove that the
  whole process tree is killed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class Mismatch(Exception):
    pass


class Script:
    def __init__(self, path: Path, args: argparse.Namespace) -> None:
        self.path = path
        self.args = args
        self.lines = self._load(path)
        self.position = 0
        self.ids: dict[Any, Any] = {}
        self.queue: list[dict[str, Any]] = []
        self.values = {
            "{{sessionId}}": args.session_id,
            "{{messageId}}": args.message_id,
            "{{cwd}}": "",
        }

    @staticmethod
    def _load(path: Path) -> list[str]:
        text = path.read_bytes().decode("utf-8")
        return [line for line in text.splitlines() if line.strip()]

    # -- placeholders ---------------------------------------------------------------

    def substitute(self, text: str) -> str:
        text = text.replace('"{{usedTokens}}"', "1234")
        for token, value in self.values.items():
            text = text.replace(token, json.dumps(value)[1:-1])
        return text

    def substitute_value(self, value: Any) -> Any:
        if isinstance(value, str):
            for token, real in self.values.items():
                value = value.replace(token, real)
            return value
        if isinstance(value, list):
            return [self.substitute_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self.substitute_value(item) for key, item in value.items()}
        return value

    # -- io ---------------------------------------------------------------------------

    def write(self, text: str) -> None:
        sys.stdout.buffer.write(text.encode("utf-8") + b"\n")
        sys.stdout.buffer.flush()

    def read(self) -> dict[str, Any] | None:
        if self.queue:
            return self.queue.pop(0)
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        message = json.loads(line)
        if self.args.log:
            with open(self.args.log, "a", encoding="utf-8") as file:
                file.write(json.dumps(message) + "\n")
        return message

    # -- playing ------------------------------------------------------------------------

    def play(self) -> int:
        while self.position < len(self.lines):
            line = self.lines[self.position]
            self.position += 1
            item = json.loads(line)
            if "jsonrpc" in item:
                self.write_recorded(line, item)
            elif "expect" in item:
                self.expect(item["expect"])
            elif "answer" in item:
                self.answer(item)
            elif "raw" in item:
                self.write(item["raw"])
            elif "big_chunk" in item:
                self.big_chunk(int(item["big_chunk"]))
            elif "sleep" in item:
                time.sleep(float(item["sleep"]))
            elif "exit" in item:
                sys.stderr.write(item.get("stderr", ""))
                sys.stderr.flush()
                return int(item["exit"])
            elif "hang" in item:
                hang()
            elif "write_file" in item:
                self.write_file(item["write_file"])
            elif "fixture" in item or "comment" in item:
                continue
            else:
                raise Mismatch(f"{self.path.name}: unknown directive {line}")
        return self.drain()

    def write_recorded(self, line: str, message: dict[str, Any]) -> None:
        text = self.substitute(line)
        if "method" not in message and "id" in message:  # a response to a client request
            recorded = message["id"]
            if recorded not in self.ids:
                raise Mismatch(f"no client request was mapped to recorded id {recorded}")
            pattern = re.compile(r'"id"\s*:\s*' + re.escape(json.dumps(recorded)) + r"(?![0-9])")
            if not pattern.search(text):
                raise Mismatch(f"cannot find the id {recorded} in the recorded response")
            text = pattern.sub(lambda _: f'"id":{json.dumps(self.ids[recorded])}', text, count=1)
        self.write(text)

    def expect(self, wanted: dict[str, Any]) -> None:
        message = self.read()
        if message is None:
            raise Mismatch(f"end of input; expected {wanted.get('method')}")
        method = wanted.get("method")
        if message.get("method") != method:
            raise Mismatch(f"expected {method}, got {json.dumps(message)[:500]}")
        if "id" in wanted:
            if "id" not in message:
                raise Mismatch(f"expected the request {method}, got a notification")
            self.ids[wanted["id"]] = message["id"]
        elif "id" in message:
            raise Mismatch(f"expected the notification {method}, got a request")
        params = message.get("params")
        if isinstance(params, dict) and isinstance(params.get("cwd"), str):
            self.values["{{cwd}}"] = params["cwd"]
        problem = match(self.substitute_value(wanted.get("params", {})), params, "params")
        if problem:
            raise Mismatch(f"{method}: {problem}")
        if method == "session/cancel" and self.args.hang_after_cancel:
            hang()

    def answer(self, directive: dict[str, Any]) -> None:
        wanted = directive["answer"]
        request_id = wanted["id"]
        overtaking: list[dict[str, Any]] = []  # a cancel that overtook the answer, say
        while True:
            message = self.read()
            if message is None:
                raise Mismatch(f"end of input; expected the answer to request {request_id}")
            if "method" not in message and message.get("id") == request_id:
                break
            overtaking.append(message)
        self.queue = overtaking + self.queue
        if "error" in message:
            result: Any = {"error": message["error"]}
        else:
            result = message.get("result")
        accepted = [self.substitute_value(item) for item in wanted.get("accept", [])]
        if result in accepted:
            return
        self.branch(directive, request_id, result, set())

    def branch(self, directive: dict[str, Any], request_id: Any, result: Any, seen: set) -> None:
        other = directive.get("otherwise")
        if not other or other in seen:
            raise Mismatch(f"answer {json.dumps(result)} to request {request_id} fits no branch")
        seen.add(other)
        path = self.path.parent / other
        lines = self._load(path)
        for index, line in enumerate(lines):
            item = json.loads(line)
            if "answer" in item and item["answer"].get("id") == request_id:
                accepted = [self.substitute_value(entry) for entry in item["answer"]["accept"]]
                if result in accepted:
                    self.path, self.lines, self.position = path, lines, index + 1
                    return
                self.branch(item, request_id, result, seen)
                return
        raise Mismatch(f"{other} has no answer for request {request_id}")

    def write_file(self, directive: dict[str, Any]) -> None:
        root = self.values["{{cwd}}"]
        if not root:
            raise Mismatch("write_file needs a session cwd first")
        target = Path(root) / directive["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(directive["text"], encoding="utf-8")

    def big_chunk(self, size: int) -> None:
        update = {
            "sessionUpdate": "agent_message_chunk",
            "messageId": self.args.message_id,
            "content": {"type": "text", "text": "x" * size},
        }
        message = {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": self.args.session_id, "update": update},
        }
        self.write(json.dumps(message, separators=(",", ":")))

    def drain(self) -> int:
        while True:
            message = self.read()
            if message is None:
                return 0
            if "method" in message and "id" in message:
                raise Mismatch(f"the script has ended; unexpected {json.dumps(message)[:500]}")


def match(pattern: Any, value: Any, where: str) -> str | None:
    if pattern == "$any":
        return None
    if pattern == "$absolute":
        ok = isinstance(value, str) and os.path.isabs(value)
        return None if ok else f"{where} should be an absolute path, not {value!r}"
    if pattern == "$list":
        return None if isinstance(value, list) else f"{where} should be a list, not {value!r}"
    if pattern == "$string":
        return None if isinstance(value, str) else f"{where} should be a string, not {value!r}"
    if isinstance(pattern, dict):
        if not isinstance(value, dict):
            return f"{where} should be an object, not {value!r}"
        for key, item in pattern.items():
            if item == "$absent":
                if key in value:
                    return f"{where}.{key} should be absent"
                continue
            if key not in value:
                return f"{where}.{key} is missing"
            problem = match(item, value[key], f"{where}.{key}")
            if problem:
                return problem
        return None
    return None if pattern == value else f"{where} should be {pattern!r}, not {value!r}"


def hang() -> None:
    while True:
        time.sleep(1)


def heartbeat(path: str) -> None:
    while True:
        try:
            with open(path, "ab") as file:
                file.write(b".")
        except OSError:
            pass
        time.sleep(0.05)


def start_child(path: str) -> None:
    child = subprocess.Popen(
        [sys.executable, "-I", os.path.abspath(__file__), "--heartbeat", path],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    while not os.path.exists(path) and time.monotonic() < deadline:
        time.sleep(0.02)
    Path(path + ".pid").write_text(str(child.pid), encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script")
    parser.add_argument("--session-id", default="fake-session-1")
    parser.add_argument("--message-id", default="fake-message-1")
    parser.add_argument("--log")
    parser.add_argument("--hang-after-cancel", action="store_true")
    parser.add_argument("--child")
    parser.add_argument("--heartbeat")
    args = parser.parse_args()
    if args.heartbeat:
        heartbeat(args.heartbeat)
        return 0
    if args.child:
        start_child(args.child)
    script = Script(Path(args.script), args)
    try:
        return script.play()
    except Mismatch as error:
        sys.stderr.write(f"fake_acp_agent: {error}\n")
        sys.stderr.flush()
        return 3


if __name__ == "__main__":
    sys.exit(main())
