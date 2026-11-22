"""A scripted JSON-RPC 2.0 peer over stdio, for the tests (standard library only).

Run as ``python -I jsonrpc_peer.py``. It reads one JSON message per line on stdin and
answers on stdout. Requests it understands:

| Method | Does |
|---|---|
| ``echo`` | answers ``params`` |
| ``big`` ``{size}`` | answers a string of ``size`` bytes (one long line) |
| ``ask`` ``{method, params}`` | sends that request to the client, waits for the answer, answers ``{"answer": ...}`` or ``{"error": ...}`` |
| ``ticks`` ``{count}`` | sends ``count`` ``tick`` notifications ``{"n": i}``, then answers ``{"sent": count}`` |
| ``garbage`` ``{lines}`` | writes ``lines`` lines that are not JSON, then answers ``"after"`` |
| ``fail`` ``{code, message}`` | answers with that error |
| ``exit`` ``{code, stderr}`` | writes ``stderr`` to stderr and exits with ``code``, without answering |
| ``hang`` | never answers |

Anything else is answered with -32601. Notifications from the client are ignored.
"""

from __future__ import annotations

import json
import sys


def write(message: dict) -> None:
    sys.stdout.buffer.write(json.dumps(message).encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()


def read() -> dict | None:
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    return json.loads(line)


def main() -> int:
    next_id = 1000
    while True:
        message = read()
        if message is None:
            return 0
        if "method" not in message or "id" not in message:
            continue  # a notification or a stray response
        request_id, method = message["id"], message["method"]
        params = message.get("params") or {}
        if method == "echo":
            write({"jsonrpc": "2.0", "id": request_id, "result": params})
        elif method == "big":
            write({"jsonrpc": "2.0", "id": request_id, "result": "x" * int(params["size"])})
        elif method == "ask":
            next_id += 1
            write(
                {
                    "jsonrpc": "2.0",
                    "id": next_id,
                    "method": params["method"],
                    "params": params.get("params"),
                }
            )
            while True:
                answer = read()
                if answer is None:
                    return 0
                if answer.get("id") == next_id and "method" not in answer:
                    break
            if "error" in answer:
                write({"jsonrpc": "2.0", "id": request_id, "result": {"error": answer["error"]}})
            else:
                write({"jsonrpc": "2.0", "id": request_id, "result": {"answer": answer["result"]}})
        elif method == "ticks":
            for n in range(int(params["count"])):
                write({"jsonrpc": "2.0", "method": "tick", "params": {"n": n}})
            write({"jsonrpc": "2.0", "id": request_id, "result": {"sent": int(params["count"])}})
        elif method == "garbage":
            for n in range(int(params["lines"])):
                sys.stdout.buffer.write(f"this is not json {n}\n".encode())
            sys.stdout.buffer.flush()
            write({"jsonrpc": "2.0", "id": request_id, "result": "after"})
        elif method == "fail":
            error = {"code": int(params["code"]), "message": params["message"]}
            write({"jsonrpc": "2.0", "id": request_id, "error": error})
        elif method == "exit":
            sys.stderr.write(params.get("stderr", ""))
            sys.stderr.flush()
            return int(params.get("code", 0))
        elif method == "hang":
            continue
        else:
            error = {"code": -32601, "message": f"Method not found: {method}"}
            write({"jsonrpc": "2.0", "id": request_id, "error": error})


if __name__ == "__main__":
    sys.exit(main())
