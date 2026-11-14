"""Boundary checks for the engine's sending conditions and destination contracts."""

import asyncio
from collections import Counter
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from qagents.effects import Attempt, Dedupe, EffectContext, Key, Unique, effect
from qagents.effects import _canonical as canonical
from qagents.effects import _engine as engine
from qagents.effects._ports import Authorization, Timing
from qagents.effects._types import HoldReason
from qagents.errors import ConfigurationError, EffectRejected, ToolError

from ._support import MemoryJournal


class Request(BaseModel):
    id: str
    values: list[int] = []


class Receipt(BaseModel):
    id: str


def declared(**options):
    @effect("send", key=options.pop("key", Key.per_call()), timeout_s=2, **options)
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        return Receipt(id=attempt.request.id)

    return send


def preparation_scope(**changes):
    values = dict(
        effects=frozenset({"send"}),
        composition_refusal=None,
        session_id="session",
        journal=MemoryJournal(),
        key_override=None,
        tool_name="tool",
        occurrences=Counter(),
        deps=None,
        operation_id="run:1:call",
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "changes",
    [
        {"effects": frozenset()},
        {"composition_refusal": "in-memory child is not fenced"},
        {"session_id": "different"},
        {"key_override": "tool-cannot-override"},
    ],
)
async def test_p0_refuses_invalid_scope_before_journal_write(changes):
    scope = preparation_scope(**changes)
    with pytest.raises(ConfigurationError):
        await engine.perform(declared(), Request(id="R1"), scope=scope)
    assert scope.journal.entries == []


def test_request_and_destination_are_detached_before_the_first_await():
    deps = {"account": ["one"]}
    scope = preparation_scope(deps=deps)
    send = declared(key=Key("request", "id"), destination=lambda deps: deps)
    request = Request(id="R1", values=[1])
    prepared = engine._prepare(send, request, scope, None)
    request.values.append(2)
    deps["account"].append("two")
    assert prepared.frozen.json == {"id": "R1", "values": [1]}
    assert prepared.destination == {"account": ["one"]}
    assert prepared.namespace == "session:session" and prepared.key == "request:R1"
    assert scope.journal.entries == []


def test_plain_dict_request_is_not_coerced_into_an_effect_request():
    scope = preparation_scope()
    with pytest.raises(TypeError, match="requires Request"):
        engine._prepare(declared(), {"id": "R1"}, scope, None)
    assert scope.journal.entries == []


def test_m12_second_per_call_invocation_requires_explicit_occurrence():
    scope = SimpleNamespace(occurrences=Counter())
    send = declared()
    assert engine._occurrence_for(scope, send, None) is None
    with pytest.raises(ConfigurationError, match="occurrence"):
        engine._occurrence_for(scope, send, None)
    assert engine._occurrence_for(scope, send, 4) == 4
    assert scope.occurrences[send.name] == 2


@pytest.mark.parametrize("occurrence", [-1, True, 1.0, "one"])
def test_invalid_occurrence_is_not_counted(occurrence):
    scope = SimpleNamespace(occurrences=Counter())
    with pytest.raises(ConfigurationError):
        engine._occurrence_for(scope, declared(), occurrence)
    assert not scope.occurrences


def test_business_key_can_be_revisited_without_positional_identity():
    scope = SimpleNamespace(occurrences=Counter())
    send = declared(key=Key("business", "id"))
    assert engine._occurrence_for(scope, send, None) is None
    assert engine._occurrence_for(scope, send, None) is None


def test_m13_sender_receives_frozen_request_even_after_caller_mutation():
    send = declared()
    original = Request(id="R1", values=[1])
    frozen = canonical.freeze(send, original)
    original.values.append(2)
    rebuilt = engine._request_for_send(send, frozen, original)
    assert rebuilt is not original and rebuilt.values == [1]
    rebuilt.values.append(3)
    assert frozen.json["values"] == [1]


def test_m4_toolerror_is_unknown_and_warned_once(monkeypatch, caplog):
    monkeypatch.setattr(engine, "_TOOLERROR_WARNED", False)
    send = declared()
    assert engine._classify_send_error(send, ToolError("lost response")) == "unknown"
    assert engine._classify_send_error(send, ToolError("another")) == "unknown"
    assert len(caplog.records) == 1
    assert "unknown outcome" in caplog.records[0].message
    assert engine._classify_send_error(send, EffectRejected("no application")) == "rejected"


def test_destination_contract_classifies_rejection_and_unique_duplicate():
    assert (
        engine._classify_send_error(declared(rejected_on=(ValueError,)), ValueError()) == "rejected"
    )
    unique = declared(dedupe=Unique(contract="destination uniqueness", on_duplicate=(KeyError,)))
    assert engine._classify_send_error(unique, KeyError()) == "duplicate"
    assert engine._classify_send_error(unique, RuntimeError()) == "unknown"


class _Refused(Exception):
    """A client's definite refusal type, the kind rejected_on lists."""


@pytest.mark.parametrize(
    "base",
    [TimeoutError, ConnectionResetError, httpx.ReadTimeout, httpx.RemoteProtocolError],
    ids=lambda base: base.__name__,
)
def test_no_answer_error_is_unknown_even_when_it_derives_from_a_declared_refusal(base):
    send = declared(rejected_on=(_Refused,))
    lost = type(f"Refused{base.__name__}", (_Refused, base), {})("response lost")
    assert isinstance(lost, send.rejected_on)
    assert engine._classify_send_error(send, lost) == "unknown"
    assert engine._classify_send_error(send, base("response lost")) == "unknown"
    assert engine._classify_send_error(send, _Refused("declined")) == "rejected"


@pytest.mark.parametrize(
    "base",
    [TimeoutError, ConnectionResetError, httpx.ReadTimeout, httpx.RemoteProtocolError],
    ids=lambda base: base.__name__,
)
def test_no_answer_error_is_unknown_even_if_rejected_on_skipped_the_declaration_check(base):
    # The shape of the analysis probe: an effect whose rejected_on lists a base class of
    # the transport errors. The declaration refuses it now; classification must not rely
    # on that, because a worker leg rebuilds effects without re-declaring them.
    send = declared()
    send.rejected_on = (OSError, httpx.HTTPError)
    assert engine._classify_send_error(send, base("response lost")) == "unknown"


def test_no_answer_overrides_effect_rejected_and_cancellation_bases():
    send = declared(rejected_on=(_Refused,))
    lost = type("RejectedTimeout", (EffectRejected, TimeoutError), {})("response lost")
    assert engine._classify_send_error(send, lost) == "unknown"
    cancelled = type("RefusedCancel", (_Refused, asyncio.CancelledError), {})("stopped")
    assert engine._classify_send_error(send, cancelled) == "unknown"
    assert engine._classify_send_error(send, asyncio.CancelledError()) == "unknown"


def _wrapped(inner: BaseException, outer: BaseException, style: str) -> BaseException:
    """``outer`` as a client raises it while handling ``inner``, with Python's own links."""
    try:
        try:
            raise inner
        except BaseException as error:
            if style == "from":
                raise outer from error
            if style == "from_none":
                raise outer from None
            raise outer  # noqa: B904
    except BaseException as raised:
        return raised
    raise AssertionError("nothing was raised")


@pytest.mark.parametrize("style", ["from", "context", "from_none"])
@pytest.mark.parametrize(
    "inner",
    [
        TimeoutError,
        ConnectionResetError,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
        asyncio.CancelledError,
    ],
    ids=lambda inner: inner.__name__,
)
def test_refusal_raised_from_or_while_handling_a_no_answer_error_is_unknown(inner, style):
    # AF-1 review: clients wrap the transport error instead of inheriting from it,
    # `except httpx.HTTPError as e: raise PaymentsError(...) from e`.
    send = declared(rejected_on=(_Refused,))
    refused = _wrapped(inner("response lost"), _Refused("payments request failed"), style)
    assert engine._classify_send_error(send, refused) == "unknown"
    rejected = _wrapped(inner("response lost"), EffectRejected("refused"), style)
    assert engine._classify_send_error(send, rejected) == "unknown"


def test_a_no_answer_error_deeper_in_the_chain_is_unknown():
    # The chain requests builds: a socket timeout, the urllib3 error raised from it,
    # and the requests error raised while handling that.
    class PoolReadTimeout(Exception):
        pass

    pool = _wrapped(TimeoutError("timed out"), PoolReadTimeout("read timed out"), "from")
    refused = _wrapped(pool, _Refused("read timed out"), "context")
    assert engine._classify_send_error(declared(rejected_on=(_Refused,)), refused) == "unknown"


def test_a_refusal_from_a_received_response_or_an_unrelated_error_stays_rejected():
    send = declared(rejected_on=(_Refused,))
    request = httpx.Request("POST", "https://payments.example/charges")
    status = httpx.HTTPStatusError(
        "400 Bad Request", request=request, response=httpx.Response(400, request=request)
    )
    for style in ("from", "context"):
        assert engine._classify_send_error(send, _wrapped(status, _Refused("bad"), style)) == (
            "rejected"
        )
        assert engine._classify_send_error(send, _wrapped(KeyError("x"), _Refused(), style)) == (
            "rejected"
        )


def test_an_exception_group_holding_a_no_answer_error_is_unknown():
    # AF-1 review: an asyncio.TaskGroup child's timeout reaches the engine inside a group.
    send = declared()
    send.rejected_on = (_Refused, BaseExceptionGroup)  # refused at declaration now
    timeout = ExceptionGroup("charge", [ValueError("audit"), httpx.ReadTimeout("lost")])
    assert engine._classify_send_error(send, timeout) == "unknown"
    cancelled = BaseExceptionGroup("charge", [asyncio.CancelledError()])
    assert engine._classify_send_error(send, cancelled) == "unknown"
    nested = ExceptionGroup("outer", [ExceptionGroup("inner", [ConnectionResetError()])])
    assert engine._classify_send_error(send, nested) == "unknown"
    wrapped = _wrapped(timeout, _Refused("payments request failed"), "from")
    assert engine._classify_send_error(declared(rejected_on=(_Refused,)), wrapped) == "unknown"
    refusals = type("Refusals", (_Refused, ExceptionGroup), {})("charge", [ValueError("bad")])
    assert engine._classify_send_error(send, refusals) == "rejected"


def _lost_responses():
    """Errors that mean no answer came back and carry no cause or context to read."""
    import http.client
    import ssl
    import warnings

    cases = [
        (EOFError, asyncio.IncompleteReadError(b"", 10)),
        (http.client.HTTPException, http.client.IncompleteRead(b"partial")),
        (http.client.HTTPException, http.client.BadStatusLine("")),
        (ssl.SSLError, ssl.SSLEOFError(8, "EOF occurred in violation of protocol")),
        (ssl.SSLError, ssl.SSLZeroReturnError(6, "TLS/SSL connection has been closed")),
    ]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # requests warns about its optional extras
            import requests
    except ImportError:
        pass
    else:
        cases.append((requests.RequestException, requests.ConnectionError("aborted")))
        cases.append((requests.RequestException, requests.exceptions.ChunkedEncodingError()))
    try:
        import aiohttp
    except ImportError:
        pass
    else:
        cases.append((aiohttp.ClientError, aiohttp.ServerDisconnectedError()))
        cases.append((aiohttp.ClientError, aiohttp.ClientPayloadError("not completed")))
    return cases


_LOST = _lost_responses()


@pytest.mark.parametrize("base,lost", _LOST, ids=[type(lost).__name__ for _, lost in _LOST])
def test_a_response_cut_short_or_a_client_library_disconnect_is_unknown(base, lost):
    # AF-1 review: these were accepted bases and classified as rejected.
    send = declared()
    send.rejected_on = (base,)  # refused at declaration now
    assert isinstance(lost, base)
    assert engine._classify_send_error(send, lost) == "unknown"


_DISCONNECTED = "Server disconnected without sending a response."


@pytest.mark.parametrize("module", ["httpx2", "httpcore", "httpcore2"])
def test_a_disconnect_reported_by_httpx2_or_httpcore_is_unknown(module):
    # AF-1 review: httpx2 (the transport of anthropic>=2.8 and mcp>=2.3) shares no class
    # with httpx, and a server that closes without answering leaves no builtin error in
    # the chain, so httpx2.RemoteProtocolError counted as a refusal.
    library = pytest.importorskip(module)
    base = library.HTTPError if module == "httpx2" else library.ProtocolError
    send = declared()
    send.rejected_on = (base,)  # refused at declaration now
    lost = library.RemoteProtocolError(_DISCONNECTED)
    assert isinstance(lost, base)
    assert engine._classify_send_error(send, lost) == "unknown"


@pytest.mark.parametrize("style", ["from", "context", "from_none"])
def test_a_refusal_raised_from_or_while_handling_an_httpx2_disconnect_is_unknown(style):
    httpx2 = pytest.importorskip("httpx2")
    httpcore2 = pytest.importorskip("httpcore2")
    # The chain httpx2 builds (its error raised from httpcore2's), wrapped by the client.
    core = httpcore2.RemoteProtocolError(_DISCONNECTED)
    lost = _wrapped(core, httpx2.RemoteProtocolError(_DISCONNECTED), "from")
    refused = _wrapped(lost, _Refused("payments request failed"), style)
    assert engine._classify_send_error(declared(rejected_on=(_Refused,)), refused) == "unknown"
    # A response that was received is still a refusal, raised directly or wrapped.
    request = httpx2.Request("POST", "https://payments.example/charges")
    status = httpx2.HTTPStatusError(
        "400 Bad Request", request=request, response=httpx2.Response(400, request=request)
    )
    received = declared(rejected_on=(httpx2.HTTPStatusError,))
    assert engine._classify_send_error(received, status) == "rejected"
    wrapped = _wrapped(status, _Refused("bad"), style)
    assert engine._classify_send_error(declared(rejected_on=(_Refused,)), wrapped) == "rejected"


def test_an_sdk_connection_error_raised_from_an_httpx2_disconnect_is_unknown():
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    # anthropic>=2.8 raises APIConnectionError from the httpx2 error. The SDK's catch-all
    # base is accepted at declaration, so only the chain makes the disconnect unknown.
    send = declared(rejected_on=(anthropic.APIError,))
    request = httpx2.Request("POST", "https://api.anthropic.example/v1/messages")
    connection = anthropic.APIConnectionError(request=request)
    lost = _wrapped(httpx2.RemoteProtocolError(_DISCONNECTED), connection, "from")
    assert engine._classify_send_error(send, lost) == "unknown"
    response = httpx2.Response(400, request=request)
    refused = anthropic.BadRequestError("invalid request", response=response, body=None)
    assert engine._classify_send_error(send, refused) == "rejected"


def test_a_cycle_in_the_error_chain_ends():
    send = declared(rejected_on=(_Refused,))
    refused, other = _Refused("declined"), ValueError("cached")
    refused.__context__, other.__context__ = other, refused
    assert engine._classify_send_error(send, refused) == "rejected"
    lost = TimeoutError("lost")
    other.__cause__, lost.__context__ = lost, refused
    assert engine._classify_send_error(send, refused) == "unknown"


@pytest.mark.parametrize("remaining,renewed", [(4.0, 0), (3.99, 1)])
async def test_m7_margin_boundary_renews_at_most_once(remaining, renewed):
    class Lease:
        calls = 0

        def __init__(self):
            self.seconds = remaining

        def lost(self):
            return False

        def remaining(self):
            return self.seconds

        async def renew(self):
            self.calls += 1
            self.seconds = 30.0
            return True

    lease = Lease()
    journal = SimpleNamespace(lease=lease, timing=Timing(30, 2, 1))
    assert await engine._self_fence(journal, declared())
    assert lease.calls == renewed


@pytest.mark.parametrize("result", [False, RuntimeError("offline"), "still_short", "lost"])
async def test_m7_failed_or_late_renewal_never_authorizes_send(result):
    class Lease:
        calls = 0

        def lost(self):
            return result == "lost"

        def remaining(self):
            return 3.0

        async def renew(self):
            self.calls += 1
            if isinstance(result, Exception):
                raise result
            return result is not False

    lease = Lease()
    journal = SimpleNamespace(lease=lease, timing=Timing(30, 2, 1))
    assert not await engine._self_fence(journal, declared())
    assert lease.calls == (0 if result == "lost" else 1)


def test_m17_automatic_resend_requires_original_proof_and_store_clock_expiry():
    scope = SimpleNamespace(
        authorization=Authorization("ask", expires_at=1000, verifiable=True),
        original_authorization=Authorization("ask", expires_at=100, verifiable=True),
    )
    assert engine._authorization_valid("D3", scope, 100) is None
    assert engine._authorization_valid("D3", scope, 100.001) == HoldReason.APPROVAL_EXPIRED
    assert engine._authorization_valid("D1", scope, 100.001) is None
    scope.original_authorization = Authorization("ask", verifiable=False)
    assert engine._authorization_valid("D4", scope, 0) == HoldReason.APPROVAL_UNVERIFIABLE
    scope.original_authorization = None
    assert engine._authorization_valid("D3", scope, 0) == HoldReason.APPROVAL_UNVERIFIABLE


def test_d5_requires_settlement_identity_and_d2_changes_only_replayed_rejection_key():
    scope = SimpleNamespace(authorization=Authorization("resend"))
    assert engine._authorization_valid("D5", scope, 0) == HoldReason.APPROVAL_UNVERIFIABLE
    scope.authorization = Authorization("resend", settlement_seq=12)
    assert engine._authorization_valid("D5", scope, 0) is None
    original = declared(dedupe=Dedupe(1000, "contract", replays_rejections=False))
    replayed = declared(dedupe=Dedupe(1000, "contract", replays_rejections=True))
    assert engine._d2_plan(original, SimpleNamespace(generation=2)) == (False, 2)
    assert engine._d2_plan(replayed, SimpleNamespace(generation=2)) == (False, 3)
