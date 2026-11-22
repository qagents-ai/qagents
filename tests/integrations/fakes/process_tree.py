"""A process with a child that writes a heartbeat file, for the tree-kill tests.

Standard library only; run as ``python -I process_tree.py MODE [ARG]``.

| Mode | Does |
|---|---|
| ``orphan HEARTBEAT`` | waits for a ``go`` line, starts the child, prints ``ready <child pid>``, then exits at end of stdin and leaves the child running (the child inherits stdout and stderr, so it holds the pipes open) |
| ``stubborn HEARTBEAT`` | the same, but ignores end of stdin and SIGTERM and runs until killed |
| ``polite`` | prints ``ready``, exits with code 0 at end of stdin |
| ``env`` | prints its environment as one JSON line and exits |
| ``stderr SIZE`` | writes ``SIZE`` bytes to stderr, ending with ``THE END``, and exits with code 3 |
| ``child HEARTBEAT`` | appends one byte to ``HEARTBEAT`` every 50 ms, forever |

The test reads the heartbeat file's size: while it grows, the child lives.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time


def child(heartbeat: str) -> None:
    while True:
        try:
            with open(heartbeat, "ab") as file:
                file.write(b".")
        except OSError:
            pass
        time.sleep(0.05)


def start_child(heartbeat: str) -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        [sys.executable, "-I", os.path.abspath(__file__), "child", heartbeat],
        stdin=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    while not os.path.exists(heartbeat) and time.monotonic() < deadline:
        time.sleep(0.02)
    return process


def main() -> int:
    mode = sys.argv[1]
    if mode == "child":
        child(sys.argv[2])
        return 0
    if mode == "env":
        print(json.dumps(dict(os.environ)), flush=True)
        return 0
    if mode == "stderr":
        size = int(sys.argv[2])
        sys.stderr.write("e" * max(0, size - 7) + "THE END")
        sys.stderr.flush()
        return 3
    if mode == "polite":
        print("ready", flush=True)
        sys.stdin.read()
        return 0
    heartbeat = sys.argv[2]
    if sys.stdin.readline().strip() != "go":
        return 2
    if mode == "stubborn" and hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    process = start_child(heartbeat)
    print(f"ready {process.pid}", flush=True)
    sys.stdin.read()  # end of stdin
    if mode == "orphan":
        return 0
    while True:
        time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
