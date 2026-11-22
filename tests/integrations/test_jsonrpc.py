"""The JSON-RPC 2.0 peer: both directions, order, errors, EOF, size limits, backpressure."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from qagents import prompts as texts
from qagents.external import (
    ExternalAgentConnectionError,
    ExternalAgentProtocolError,
    ExternalAgentTimeoutError,
)
from qagents.integrations._jsonrpc import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    REQUEST_CANCELLED,
    JsonRpcError,
    JsonRpcPeer,
)
from qagents.integrations._process import ManagedProcess, build_env

from .fakes import script
from .fakes.streams import RawSide, peer_pair

MIB = 1024 * 1024


async def test_requests_in_both_directions() -> None:
    a, b = peer_pair()

    async def add(params: dict) -> int:
        return params["x"] + params["y"]

    async def greet(params: dict) -> str:
        return f"hello {params['name']}"

    b.on_request("add", add)
    a.on_request("greet", greet)
    a.start()
    b.start()
    assert await a.request("add", {"x": 2, "y": 3}) == 5
    assert await b.request("greet", {"name": "qagents"}) == "hello qagents"
    await a.aclose()
    await b.aclose()


async def test_concurrent_requests_resolve_by_id_out_of_order() -> None:
    side = RawSide()
    side.peer.start()
    first = asyncio.create_task(side.peer.request("slow"))
    second = asyncio.create_task(side.peer.request("fast"))
    one, two = await side.receive(), await side.receive()
    assert (one["id"], two["id"]) == (1, 2)
    assert one["method"] == "slow" and two["method"] == "fast"
    side.send({"jsonrpc": "2.0", "id": 2, "result": "fast done"})
    side.send({"jsonrpc": "2.0", "id": 1, "result": "slow done"})
    assert await second == "fast done"
    assert await first == "slow done"
    await side.peer.aclose()


async def test_error_answers_raise_json_rpc_error() -> None:
    side = RawSide()
    side.peer.start()
    pending = asyncio.create_task(side.peer.request("session/prompt", {"sessionId": "s"}))
    sent = await side.receive()
    error = {"code": -32603, "message": "Internal error: turn failed: boom", "data": {"x": 1}}
    side.send({"jsonrpc": "2.0", "id": sent["id"], "error": error})
    with pytest.raises(JsonRpcError) as raised:
        await pending
    assert raised.value.code == -32603
    assert raised.value.message == "Internal error: turn failed: boom"
    assert raised.value.data == {"x": 1}
    assert raised.value.method == "session/prompt"
    await side.peer.aclose()


async def test_incoming_requests_get_results_and_error_replies() -> None:
    side = RawSide()

    async def ok(params: dict) -> dict:
        return {"seen": params}

    async def refuse(params: dict) -> None:
        raise JsonRpcError(INVALID_PARAMS, "Invalid params: cwd must be absolute")

    async def broken(params: dict) -> None:
        raise RuntimeError("secret detail that must not leak")

    side.peer.on_request("ok", ok)
    side.peer.on_request("refuse", refuse)
    side.peer.on_request("broken", broken)
    side.peer.start()
    side.send({"jsonrpc": "2.0", "id": 7, "method": "ok", "params": {"a": 1}})
    assert await side.receive() == {"jsonrpc": "2.0", "id": 7, "result": {"seen": {"a": 1}}}
    side.send({"jsonrpc": "2.0", "id": 8, "method": "refuse", "params": {}})
    answer = await side.receive()
    assert answer["error"] == {
        "code": INVALID_PARAMS,
        "message": "Invalid params: cwd must be absolute",
    }
    side.send({"jsonrpc": "2.0", "id": "s-9", "method": "broken"})
    answer = await side.receive()
    assert answer["id"] == "s-9" and answer["error"]["code"] == INTERNAL_ERROR
    assert "secret" not in answer["error"]["message"]
    side.send({"jsonrpc": "2.0", "id": 10, "method": "fs/read_text_file", "params": {}})
    answer = await side.receive()
    assert answer["error"]["code"] == METHOD_NOT_FOUND
    await side.peer.aclose()


async def test_notifications_are_handled_in_order_before_a_following_request() -> None:
    side = RawSide()
    seen: list[str] = []

    def update(params: dict) -> None:
        seen.append(f"update {params['n']}")

    async def slow_update(params: dict) -> None:
        await asyncio.sleep(0.05)  # still finished before the request handler starts
        seen.append(f"slow {params['n']}")

    async def permission(params: dict) -> dict:
        seen.append("permission")
        return {"outcome": {"outcome": "cancelled"}}

    side.peer.on_notification("session/update", update)
    side.peer.on_notification("slow", slow_update)
    side.peer.on_request("session/request_permission", permission)
    side.peer.start()
    side.send({"jsonrpc": "2.0", "method": "session/update", "params": {"n": 1}})
    side.send({"jsonrpc": "2.0", "method": "slow", "params": {"n": 2}})
    side.send({"jsonrpc": "2.0", "method": "session/update", "params": {"n": 3}})
    side.send({"jsonrpc": "2.0", "id": 1, "method": "session/request_permission", "params": {}})
    await side.receive()
    assert seen == ["update 1", "slow 2", "update 3", "permission"]
    await side.peer.aclose()


async def test_a_slow_request_handler_does_not_block_the_stream() -> None:
    side = RawSide()
    release = asyncio.Event()
    updates: list[int] = []

    async def wait_for_human(params: dict) -> str:
        await release.wait()
        return "answered"

    side.peer.on_request("ask", wait_for_human)
    side.peer.on_notification("tick", lambda params: updates.append(params["n"]))
    side.peer.start()
    side.send({"jsonrpc": "2.0", "id": 1, "method": "ask"})
    for n in range(3):
        side.send({"jsonrpc": "2.0", "method": "tick", "params": {"n": n}})
    for _ in range(50):
        if len(updates) == 3:
            break
        await asyncio.sleep(0.01)
    assert updates == [0, 1, 2]
    release.set()
    assert (await side.receive())["result"] == "answered"
    await side.peer.aclose()


async def test_failing_notification_handler_is_logged_and_the_connection_stays(
    caplog: pytest.LogCaptureFixture,
) -> None:
    side = RawSide()

    def broken(params: object) -> None:
        raise ValueError("bad update")

    side.peer.on_notification("update", broken)
    side.peer.start()
    side.send({"jsonrpc": "2.0", "method": "update", "params": {}})
    pending = asyncio.create_task(side.peer.request("echo"))
    sent = await side.receive()
    side.send({"jsonrpc": "2.0", "id": sent["id"], "result": "still here"})
    assert await pending == "still here"
    assert "bad update" in caplog.text
    await side.peer.aclose()


async def test_eof_fails_every_pending_request_with_a_clear_error() -> None:
    side = RawSide()
    side.peer.start()
    first = asyncio.create_task(side.peer.request("session/prompt"))
    second = asyncio.create_task(side.peer.request("session/new"))
    await side.receive()
    await side.receive()
    side.eof()
    for task, method in ((first, "session/prompt"), (second, "session/new")):
        with pytest.raises(ExternalAgentConnectionError) as raised:
            await asyncio.wait_for(task, 2)
        assert method in str(raised.value)
        assert "closed the connection" in str(raised.value)
    assert side.peer.closed
    with pytest.raises(ExternalAgentConnectionError):
        await side.peer.request("after")
    with pytest.raises(ExternalAgentConnectionError):
        await side.peer.notify("after")


async def test_close_callbacks_and_wait_closed() -> None:
    side = RawSide()
    seen: list[ExternalAgentConnectionError] = []
    side.peer.on_close(seen.append)
    side.peer.start()
    side.eof()
    error = await asyncio.wait_for(side.peer.wait_closed(), 2)
    assert seen == [error]
    late: list[ExternalAgentConnectionError] = []
    side.peer.on_close(late.append)  # registered after the close: called at once
    assert late == [error]


async def test_a_20_mib_line_is_accepted_without_stalling() -> None:
    side = RawSide()
    side.peer.start()
    pending = asyncio.create_task(side.peer.request("big"))
    sent = await side.receive()
    payload = "y" * (20 * MIB)
    began = time.perf_counter()
    side.send({"jsonrpc": "2.0", "id": sent["id"], "result": payload})
    result = await asyncio.wait_for(pending, 30)
    assert len(result) == 20 * MIB
    assert time.perf_counter() - began < 20
    await side.peer.aclose()


async def test_a_line_over_the_limit_closes_the_connection_and_fails_pending_calls() -> None:
    side = RawSide(max_line_bytes=MIB)
    side.peer.start()
    pending = asyncio.create_task(side.peer.request("big"))
    await side.receive()
    side.send_raw(b'{"jsonrpc": "2.0", "id": 1, "result": "' + b"z" * (2 * MIB))  # no end yet
    with pytest.raises(ExternalAgentProtocolError) as raised:
        await asyncio.wait_for(pending, 5)
    assert "longer than" in str(raised.value)
    assert side.peer.closed


async def test_a_long_line_is_refused_also_when_its_newline_comes_in_the_same_read() -> None:
    side = RawSide(max_line_bytes=100)
    side.peer.start()
    pending = asyncio.create_task(side.peer.request("big"))
    sent = await side.receive()
    answer = {"jsonrpc": "2.0", "id": sent["id"], "result": "z" * 500}
    side.send_raw(json.dumps(answer).encode() + b"\n")  # the whole line in one chunk
    with pytest.raises(ExternalAgentProtocolError) as raised:
        await asyncio.wait_for(pending, 5)
    assert raised.value.reason == texts.EXTERNAL_REASON_TOO_LARGE
    assert side.peer.closed

    exact = RawSide(max_line_bytes=100)  # a line of exactly the limit is still accepted
    exact.peer.start()
    pending = asyncio.create_task(exact.peer.request("small"))
    sent = await exact.receive()
    head = b'{"jsonrpc":"2.0","id":%d,"result":"' % sent["id"]
    line = head + b"z" * (100 - len(head) - 2) + b'"}'
    assert len(line) == 100
    exact.send_raw(line + b"\n")
    assert (await asyncio.wait_for(pending, 5)).startswith("z")
    await exact.peer.aclose()


def test_connection_reasons_come_from_the_prompts_module() -> None:
    error = ExternalAgentConnectionError("details for developers")
    assert error.reason == texts.EXTERNAL_REASON_CONNECTION_LOST


async def test_malformed_lines_are_skipped_until_too_many_in_a_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    side = RawSide()
    side.peer.start()
    pending = asyncio.create_task(side.peer.request("echo"))
    sent = await side.receive()
    side.send_raw(b"not json\n")
    side.send_raw(b'{"jsonrpc": "1.0", "id": 99}\n')
    side.send_raw(b"\n   \n")  # blank lines do not count
    side.send({"jsonrpc": "2.0", "id": sent["id"], "result": "fine"})
    assert await pending == "fine"
    assert side.peer.malformed == 2
    assert "Skipped a line" in caplog.text

    pending = asyncio.create_task(side.peer.request("echo"))
    await side.receive()
    side.send_raw(b"garbage 1\ngarbage 2\n[1, 2]\n")
    with pytest.raises(ExternalAgentProtocolError) as raised:
        await asyncio.wait_for(pending, 5)
    assert "malformed lines in a row" in str(raised.value)


async def test_a_response_whose_id_cannot_answer_a_request_is_skipped_as_malformed() -> None:
    side = RawSide(max_malformed=4)
    side.peer.start()
    pending = asyncio.create_task(side.peer.request("ping"))
    sent = await side.receive()
    side.send({"jsonrpc": "2.0", "id": [sent["id"]], "result": {}})  # unhashable: a list
    side.send({"jsonrpc": "2.0", "id": {"n": sent["id"]}, "error": {"code": 1}})
    side.send({"jsonrpc": "2.0", "id": True, "result": "not for request 1"})
    side.send({"jsonrpc": "2.0", "id": sent["id"], "result": "pong"})  # the real answer
    assert await asyncio.wait_for(pending, 5) == "pong"
    assert side.peer.malformed == 3 and not side.peer.closed
    await side.peer.aclose()


async def test_a_batch_is_handled_item_by_item() -> None:
    side = RawSide()

    async def echo(params: object) -> object:
        return params

    side.peer.on_request("echo", echo)
    seen: list[object] = []
    side.peer.on_notification("note", seen.append)
    side.peer.start()
    side.send(
        [
            {"jsonrpc": "2.0", "method": "note", "params": {"n": 1}},
            {"jsonrpc": "2.0", "id": 5, "method": "echo", "params": [1, 2]},
        ]
    )
    assert await side.receive() == {"jsonrpc": "2.0", "id": 5, "result": [1, 2]}
    assert seen == [{"n": 1}]
    await side.peer.aclose()


async def test_writes_wait_for_the_peer_to_drain() -> None:
    side = RawSide()
    gate = asyncio.Event()
    peer_writer = side.peer._writer
    peer_writer.gate = gate  # the other side stops reading
    side.peer.start()
    first = asyncio.create_task(side.peer.notify("one"))
    second = asyncio.create_task(side.peer.notify("two"))
    await asyncio.sleep(0.05)
    assert not first.done() and not second.done()  # held by drain, the second by the lock
    assert peer_writer.drains == 1
    gate.set()
    await asyncio.wait_for(asyncio.gather(first, second), 2)
    assert (await side.receive())["method"] == "one"
    assert (await side.receive())["method"] == "two"
    await side.peer.aclose()


async def test_a_broken_pipe_closes_the_peer() -> None:
    side = RawSide()
    side.peer.start()
    side.peer._writer.fail_with = BrokenPipeError("pipe is broken")
    with pytest.raises(ExternalAgentConnectionError) as raised:
        await side.peer.request("echo")
    assert "Writing to peer failed" in str(raised.value)
    assert side.peer.closed


async def test_timeout_and_cancel_are_announced_with_the_cancel_method() -> None:
    side = RawSide(cancel_method="$/cancel_request")
    side.peer.start()
    with pytest.raises(ExternalAgentTimeoutError):
        await side.peer.request("session/prompt", timeout=0.05)
    sent = await side.receive()
    cancel = await side.receive()
    assert cancel == {
        "jsonrpc": "2.0",
        "method": "$/cancel_request",
        "params": {"requestId": sent["id"]},
    }
    side.send({"jsonrpc": "2.0", "id": sent["id"], "result": "late"})  # ignored

    task = asyncio.create_task(side.peer.request("session/prompt"))
    sent = await side.receive()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await side.receive())["params"] == {"requestId": sent["id"]}
    assert not side.peer.closed
    await side.peer.aclose()


async def test_the_peer_can_cancel_a_request_we_are_handling() -> None:
    side = RawSide(cancel_method="$/cancel_request")
    started = asyncio.Event()

    async def long(params: object) -> str:
        started.set()
        await asyncio.sleep(30)
        return "never"

    side.peer.on_request("session/request_permission", long)
    side.peer.start()
    side.send({"jsonrpc": "2.0", "id": 3, "method": "session/request_permission"})
    await asyncio.wait_for(started.wait(), 2)
    side.send({"jsonrpc": "2.0", "method": "$/cancel_request", "params": {"requestId": 3}})
    answer = await side.receive()
    assert answer["id"] == 3 and answer["error"]["code"] == REQUEST_CANCELLED
    await side.peer.aclose()


async def test_aclose_fails_pending_requests_and_closes_the_writer() -> None:
    side = RawSide()
    side.peer.start()
    pending = asyncio.create_task(side.peer.request("hang"))
    await side.receive()
    await side.peer.aclose()
    with pytest.raises(ExternalAgentConnectionError) as raised:
        await pending
    assert "closed by qagents" in str(raised.value)
    assert side.peer._writer.closed


# -- over a real subprocess --------------------------------------------------------------


async def _fake_peer(**options: object) -> tuple[ManagedProcess, JsonRpcPeer]:
    process = await ManagedProcess.start(script("jsonrpc_peer"), env=build_env(), name="fake-peer")
    peer = JsonRpcPeer.for_process(process, **options)
    return process, peer


@pytest.mark.devvi_fake
async def test_subprocess_round_trip_and_peer_initiated_requests() -> None:
    process, peer = await _fake_peer()

    async def permission(params: dict) -> dict:
        return {"outcome": {"outcome": "selected", "optionId": params["options"][0]}}

    ticks: list[int] = []
    peer.on_request("session/request_permission", permission)
    peer.on_notification("tick", lambda params: ticks.append(params["n"]))
    peer.start()
    try:
        assert await peer.request("echo", {"a": [1, "é"]}, timeout=10) == {"a": [1, "é"]}
        asked = await peer.request(
            "ask",
            {"method": "session/request_permission", "params": {"options": ["allow-once"]}},
            timeout=10,
        )
        assert asked == {"answer": {"outcome": {"outcome": "selected", "optionId": "allow-once"}}}
        unknown = await peer.request("ask", {"method": "terminal/create"}, timeout=10)
        assert unknown["error"]["code"] == METHOD_NOT_FOUND
        assert await peer.request("ticks", {"count": 5}, timeout=10) == {"sent": 5}
        assert ticks == [0, 1, 2, 3, 4]
        with pytest.raises(JsonRpcError) as raised:
            await peer.request("fail", {"code": -32602, "message": "Invalid params: x"}, timeout=10)
        assert raised.value.code == -32602
    finally:
        await peer.aclose()
        await process.close(grace=5)
    assert process.returncode == 0
    assert not process.killed


@pytest.mark.devvi_fake
async def test_subprocess_20_mib_line() -> None:
    process, peer = await _fake_peer()
    peer.start()
    try:
        result = await peer.request("big", {"size": 20 * MIB}, timeout=60)
        assert len(result) == 20 * MIB
    finally:
        await peer.aclose()
        await process.close()


@pytest.mark.devvi_fake
async def test_subprocess_line_over_the_limit_fails_clearly() -> None:
    process, peer = await _fake_peer(max_line_bytes=MIB)
    peer.start()
    try:
        with pytest.raises(ExternalAgentProtocolError):
            await peer.request("big", {"size": 4 * MIB}, timeout=30)
    finally:
        await peer.aclose()
        await process.close(grace=1)


@pytest.mark.devvi_fake
async def test_subprocess_garbage_then_recovery_and_too_much_garbage() -> None:
    process, peer = await _fake_peer()
    peer.start()
    try:
        assert await peer.request("garbage", {"lines": 2}, timeout=10) == "after"
        with pytest.raises(ExternalAgentProtocolError):
            await peer.request("garbage", {"lines": 3}, timeout=10)
    finally:
        await peer.aclose()
        await process.close(grace=1)


@pytest.mark.devvi_fake
async def test_subprocess_exit_mid_request_names_the_exit_code_and_stderr() -> None:
    process, peer = await _fake_peer()
    peer.start()
    try:
        with pytest.raises(ExternalAgentConnectionError) as raised:
            await peer.request("exit", {"code": 3, "stderr": "fatal: no model route\n"}, timeout=10)
        error = raised.value
        assert error.returncode == 3
        assert error.reason == "the process exited with code 3"
        assert "fatal: no model route" in str(error)
        assert error.stderr_tail is not None and "no model route" in error.stderr_tail
        assert "'exit'" in str(error)
        assert "stderr" not in error.reason  # the short reason is safe to show a model
    finally:
        await peer.aclose()
        await process.close(grace=1)


def test_error_object_shape() -> None:
    error = JsonRpcError(-32602, "Invalid params", {"field": "cwd"}, method="session/new")
    assert error.to_object() == {
        "code": -32602,
        "message": "Invalid params",
        "data": {"field": "cwd"},
    }
    assert "session/new" in str(error)
    assert json.dumps(JsonRpcError(-1, "x").to_object()) == '{"code": -1, "message": "x"}'
