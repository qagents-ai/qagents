"""Stateful loopback fixture for a small Stripe API v1 refund contract."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from json import dumps
from threading import Event, Lock, Thread
from urllib.parse import parse_qs, urlsplit

WINDOW_S = 86400


def _error(code, *, kind="invalid_request_error"):
    return {"error": {"type": kind, "code": code}}


@dataclass(frozen=True)
class CachedResponse:
    parameters: dict
    expires: float
    status: int
    body: dict


class RefundSandbox:
    """Own a synthetic charge, its refund ledger and deterministic fault controls."""

    def __init__(self):
        self.refunds = []
        self.attempts = Counter()
        self.clock = 0.0
        self.cache = {}
        self.pending = set()
        self.lock = Lock()
        self.entered = Event()
        self.release = Event()
        self.delay_key = None
        self.error_key = None
        self.refusal_key = None
        self.disconnect_once = False
        self.hide_next_lookup = False

    def refund(self, key, fields):
        with self.lock:
            self.attempts[key] += 1
            if key is None or not 1 <= len(key) <= 255:
                return 400, _error("invalid_idempotency_key"), False
            try:
                amount = int(fields["amount"][0])
            except (KeyError, ValueError):
                amount = 0
            if amount <= 0 or fields.get("charge") != ["ch_fixture"]:
                return 400, _error("parameter_invalid"), False
            saved = self.cache.get(key)
            if saved is not None and self.clock < saved.expires:
                if fields != saved.parameters:
                    return 400, _error("parameters_changed", kind="idempotency_error"), False
                return saved.status, saved.body, True
            if key in self.pending:
                return 409, _error("idempotency_key_in_use"), False
            self.pending.add(key)
            expires = self.clock + WINDOW_S

        try:
            if key == self.delay_key:
                self.entered.set()
                if not self.release.wait(10):
                    raise TimeoutError("probe never released the in-flight request")
            with self.lock:
                if key == self.error_key:
                    status, body = 500, _error("execution_error", kind="api_error")
                elif key == self.refusal_key:
                    status, body = 400, _error("charge_already_refunded")
                else:
                    status = 200
                    body = {"id": f"re_{len(self.refunds) + 1}", "amount": amount}
                    if tags := fields.get("metadata[operation_tag]"):
                        body["metadata"] = {"operation_tag": tags[0]}
                    self.refunds.append(body)
                self.cache[key] = CachedResponse(fields, expires, status, body)
                if status == 200 and self.disconnect_once:
                    self.disconnect_once = False
                    return None
                return status, body, False
        finally:
            with self.lock:
                self.pending.remove(key)


@contextmanager
def stripe_http_sandbox():
    sandbox = RefundSandbox()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server protocol method
            path = urlsplit(self.path)
            if path.path != "/v1/refunds" or parse_qs(path.query) != {"charge": ["ch_fixture"]}:
                self.send_error(404)
                return
            with sandbox.lock:
                rows = [] if sandbox.hide_next_lookup else list(sandbox.refunds)
                sandbox.hide_next_lookup = False
            self.reply(200, {"data": rows}, False)

        def do_POST(self):  # noqa: N802 - http.server protocol method
            if self.path != "/v1/refunds":
                self.send_error(404)
                return
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                self.send_error(400)
                return
            fields = parse_qs(self.rfile.read(size).decode("ascii"))
            result = sandbox.refund(self.headers.get("Idempotency-Key"), fields)
            if result is None:
                self.close_connection = True
                return
            self.reply(*result)

        def reply(self, status, body, replayed):
            payload = dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            if replayed:
                self.send_header("Idempotent-Replayed", "true")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", sandbox
        finally:
            sandbox.release.set()
            server.shutdown()
            thread.join()
