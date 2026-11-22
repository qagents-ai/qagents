"""Recovery model replies follow durable turns and account for every consultation."""

import asyncio

import pytest

from qagents.effects import _ports
from qagents.errors import ModelError
from qagents.messages import Message, ReasoningPart, ToolCallPart, Usage
from qagents.models.base import ModelRequest, ModelResponse
from qagents.models.recording import RecordingModel
from qagents.models.testing import ScriptedModel
from qagents.testing.crash import _model


def request(text="question"):
    return ModelRequest(messages=[Message.user(text)])


def test_unscripted_request_is_a_definite_non_retryable_model_error():
    error = _model.UnscriptedRequest("no reply")
    assert isinstance(error, ModelError)
    assert error.retryable is False
    assert error.ambiguous is False


async def ask(model, turn, req=None):
    token = _ports.MODEL_TURN.set(
        None if turn is None else _ports.ModelTurn("session", "run", turn)
    )
    try:
        return await model.request(req or request())
    finally:
        _ports.MODEL_TURN.reset(token)


def test_turn_survives_compaction_and_crash():
    async def scenario():
        ledger = _model.ModelLedger(["first", "second", "third"])
        assert (await ask(ledger.for_leg(0), 0)).message.text == "first"
        # A compacted context contains no assistant messages, but two are durable.
        compacted = request("summary of two prior turns")
        assert (await ask(ledger.for_leg(1, recovering=True), 2, compacted)).message.text == "third"
        assert [item.turn for item in ledger.calls] == [0, 2]

    asyncio.run(scenario())


def test_same_turn_is_consulted_again_and_spend_is_shared_across_legs():
    async def scenario():
        response = ModelResponse(
            message=Message.assistant("answer"),
            usage=Usage(requests=1, input_tokens=9, output_tokens=3, cost_usd=0.01),
        )
        ledger = _model.ModelLedger([response])
        await ask(ledger.for_leg(0), 0)
        await ask(ledger.for_leg(1, recovering=True), 0)
        assert len(ledger.calls) == 2
        assert ledger.usage.requests == 2
        assert ledger.usage.total_tokens == 24
        assert ledger.usage.cost_usd == pytest.approx(0.02)

    asyncio.run(scenario())


def test_requests_without_turn_match_only_when_and_do_not_consume_main_script():
    async def scenario():
        ledger = _model.ModelLedger(
            [_model.when(lambda req: req.messages[0].text == "summary", "summary reply"), "main"]
        )
        model = ledger.for_leg(0)
        assert (await ask(model, None, request("summary"))).message.text == "summary reply"
        assert (await ask(model, 0)).message.text == "main"
        with pytest.raises(_model.UnscriptedRequest):
            await ask(model, None)
        assert ledger.calls[-1].outcome == "inconclusive"

    asyncio.run(scenario())


@pytest.mark.parametrize("turn", [-1, 1, True])
def test_unavailable_turn_never_falls_back_to_message_count(turn):
    async def scenario():
        ledger = _model.ModelLedger(["only"])
        with pytest.raises(_model.UnscriptedRequest):
            await ask(ledger.for_leg(0), turn)

    asyncio.run(scenario())


def test_model_snapshots_cannot_be_mutated_by_a_consumer():
    async def scenario():
        response = ModelResponse(
            message=Message(
                role="assistant",
                parts=[
                    ReasoningPart(text="fixture", opaque={"signature": "fixture-only"}),
                    ToolCallPart(id="c1", name="refund", arguments={"nested": {"amount": 10}}),
                ],
            )
        )
        ledger = _model.ModelLedger([response])
        req = request()
        first = await ask(ledger.for_leg(0), 0, req)
        first.message.tool_calls[0].arguments["nested"]["amount"] = 99
        req.messages.clear()
        second = await ask(ledger.for_leg(1), 0)
        assert second.message.tool_calls[0].arguments["nested"]["amount"] == 10
        assert ledger.calls[0].request.messages
        assert ledger.calls[0].response.message.parts[0].opaque == {"signature": "fixture-only"}

    asyncio.run(scenario())


def test_async_script_and_script_errors_are_recorded():
    async def reply(req):
        return "async answer"

    async def scenario():
        ledger = _model.ModelLedger([reply, ValueError("fixture error")])
        model = ledger.for_leg(0)
        assert (await ask(model, 0)).message.text == "async answer"
        with pytest.raises(ValueError, match="fixture error"):
            await ask(model, 1)
        assert ledger.calls[-1].error_type == "ValueError"
        assert ledger.calls[-1].outcome == "failed_unexpected"

    asyncio.run(scenario())


def test_cassette_clients_are_fresh_per_leg_and_misses_are_inconclusive(tmp_path):
    async def scenario():
        path = tmp_path / "model.json"
        recorder = RecordingModel(ScriptedModel(["recorded"]), path)
        await recorder.request(request())
        await recorder.aclose()
        ledger = _model.ModelLedger(_model.Cassette(path))
        assert (await ask(ledger.for_leg(0), 0)).message.text == "recorded"
        assert (await ask(ledger.for_leg(1), 0)).message.text == "recorded"
        with pytest.raises(_model.CassetteMissError):
            await ask(ledger.for_leg(2), 0, request("different"))
        assert ledger.calls[-1].outcome == "inconclusive"

    asyncio.run(scenario())


def test_cassette_miss_uses_mutant_seam(tmp_path, monkeypatch):
    async def scenario():
        path = tmp_path / "model.json"
        recorder = RecordingModel(ScriptedModel(["recorded"]), path)
        await recorder.request(request())
        ledger = _model.ModelLedger(_model.Cassette(path))
        monkeypatch.setattr(_model, "_on_cassette_miss", lambda error: "seeded-bug")
        with pytest.raises(_model.CassetteMissError):
            await ask(ledger.for_leg(0), 0, request("miss"))
        assert ledger.calls[-1].outcome == "seeded-bug"

    asyncio.run(scenario())


def effect_call():
    return ToolCallPart(
        id="original",
        name="refund",
        arguments={"amount": 10, "request_id": "R1"},
        raw_arguments='{"amount":10,"request_id":"R1"}',
    )


@pytest.mark.parametrize(
    "kind", ["fresh_ids", "retry_effect", "retry_changed", "permuted", "abandon"]
)
def test_perturbations_never_change_reference_leg(kind):
    async def scenario():
        ledger = _model.ModelLedger([effect_call()])
        answer = await ask(ledger.for_leg(0, perturbation=kind, effect_tools={"refund"}), 0)
        assert answer.message.tool_calls[0] == effect_call()

    asyncio.run(scenario())


def test_retry_effect_repeats_last_effect_under_new_deterministic_id_once():
    async def scenario():
        ledger = _model.ModelLedger([effect_call(), "finished", "next"], seed=7)
        await ask(ledger.for_leg(0), 0)
        model = ledger.for_leg(
            1, recovering=True, perturbation="retry_effect", effect_tools={"refund"}
        )
        retry = await ask(model, 1)
        assert retry.message.tool_calls[0].id != "original"
        assert retry.message.tool_calls[0].arguments == effect_call().arguments
        assert (await ask(model, 2)).message.text == "next"
        return retry.message.tool_calls[0].id

    assert asyncio.run(scenario()) == asyncio.run(scenario())


def test_retry_changed_prioritizes_measured_key_field_and_updates_raw_arguments():
    async def scenario():
        ledger = _model.ModelLedger([effect_call(), "finished"])
        await ask(ledger.for_leg(0), 0)
        model = ledger.for_leg(
            1,
            recovering=True,
            perturbation="retry_changed",
            effect_tools={"refund"},
            key_sensitive_fields={"refund": ("request_id",)},
        )
        response = await ask(model, 1)
        changed = response.message.tool_calls[0]
        assert changed.arguments["request_id"] != "R1"
        assert changed.arguments["amount"] == 10
        assert changed.raw_arguments is None
        assert ledger.calls[-1].changed_field == "request_id"
        assert ledger.calls[-1].key_sensitive is True

    asyncio.run(scenario())


def test_change_candidates_change_one_field_without_touching_original():
    original = effect_call()
    candidates = _model.change_candidates(original, key_sensitive_fields=("request_id",))
    assert [name for name, _ in candidates] == ["request_id", "amount"]
    for field, candidate in candidates:
        assert [
            key for key in original.arguments if candidate.arguments[key] != original.arguments[key]
        ] == [field]
    assert original == effect_call()


def test_retry_changed_can_target_non_key_field_without_claiming_it_is_key_sensitive():
    async def scenario():
        ledger = _model.ModelLedger([effect_call(), "finished"])
        await ask(ledger.for_leg(0), 0)
        model = ledger.for_leg(
            1,
            recovering=True,
            perturbation="retry_changed",
            effect_tools={"refund"},
            key_sensitive_fields={"refund": ("request_id",)},
            change_field="amount",
        )
        changed = (await ask(model, 1)).message.tool_calls[0]
        assert changed.arguments == {"amount": 11, "request_id": "R1"}
        assert ledger.calls[-1].changed_field == "amount"
        assert not ledger.calls[-1].key_sensitive

    asyncio.run(scenario())


def test_retry_changed_without_prior_effect_is_inconclusive():
    async def scenario():
        ledger = _model.ModelLedger(["done"])
        model = ledger.for_leg(
            1, recovering=True, perturbation="retry_changed", effect_tools={"refund"}
        )
        with pytest.raises(_model.UnscriptedRequest):
            await ask(model, 0)
        assert ledger.calls[-1].outcome == "inconclusive"

    asyncio.run(scenario())


def test_fresh_ids_preserve_arguments_and_reasoning_and_permuted_changes_order():
    async def scenario():
        calls = [effect_call(), ToolCallPart(id="c2", name="other", arguments={})]
        ledger = _model.ModelLedger([calls])
        fresh = await ask(ledger.for_leg(1, recovering=True, perturbation="fresh_ids"), 0)
        assert [call.id for call in fresh.message.tool_calls] != ["original", "c2"]
        assert [call.arguments for call in fresh.message.tool_calls] == [
            call.arguments for call in calls
        ]
        reversed_reply = await ask(ledger.for_leg(2, recovering=True, perturbation="permuted"), 0)
        assert [call.name for call in reversed_reply.message.tool_calls] == ["other", "refund"]
        abandoned = await ask(ledger.for_leg(3, recovering=True, perturbation="abandon"), 0)
        assert not abandoned.message.tool_calls
        assert abandoned.finish_reason == "stop"

    asyncio.run(scenario())
