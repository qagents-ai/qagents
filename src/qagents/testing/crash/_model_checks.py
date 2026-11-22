"""Check recovered model messages against durable history and local wire encoders."""

from collections import Counter, defaultdict

from qagents.effects._canonical import jcs
from qagents.models._openai_wire import build_chat_body
from qagents.models.recording import request_fingerprint
from qagents.session import derive_state

from ._invariants import Finding, _check, _read


def check_model_calls(calls, evidence, adapters):
    pairs = defaultdict(list)
    for item in evidence:
        pairs[(item.leg, item.consultation)].append(item)
    ordinals = Counter()
    findings = {key: [] for key in ("I4a", "I4b", "I4c")}
    for call in calls:
        number = ordinals[call.leg]
        ordinals[call.leg] += 1
        if call.turn is None:
            continue  # Auxiliary model requests are matched by when(), not by turn.
        request = call.request
        if not _paired(request.messages):
            findings["I4a"].append(
                Finding(
                    "I4a",
                    "unpaired_tool_messages",
                    f"Tool messages were unpaired in leg {call.leg}.",
                )
            )
        captured = pairs[(call.leg, number)]
        if len(captured) != 1:
            findings["I4b"].append(
                Finding(
                    "I4b",
                    "model_prefix_missing",
                    "Model request has no unique durable prefix.",
                    "inconclusive",
                )
            )
        else:
            observed = captured[0]
            history = derive_state(_read(observed.snapshot)).messages
            observed_history = [msg for msg in request.messages if msg.role != "system"]
            expected_history = [msg for msg in history if msg.role != "system"]
            expected = request.model_copy(
                update={
                    "messages": [
                        *[msg for msg in request.messages if msg.role == "system"],
                        *expected_history,
                    ]
                }
            )
            if (
                request_fingerprint(request) != observed.fingerprint
                or request_fingerprint(expected) != call.fingerprint
                or _reasoning(observed_history) != _reasoning(expected_history)
            ):
                findings["I4b"].append(
                    Finding(
                        "I4b",
                        "model_history_differs",
                        "Model history differs from its durable prefix.",
                    )
                )
        for adapter in adapters:
            try:
                if adapter == "openai_chat":
                    build_chat_body("offline-test", request, stream=True)
                elif adapter == "anthropic":
                    from qagents.models.anthropic import AnthropicModel

                    AnthropicModel(api_key="offline-test").build_params(request)
            except Exception as error:
                findings["I4c"].append(
                    Finding(
                        "I4c",
                        "model_request_cannot_encode",
                        f"{adapter} refused a recorded request ({type(error).__name__}).",
                    )
                )
    return tuple(_check(key, result) for key, result in findings.items())


def _paired(messages):
    waiting = set()
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            if waiting or len({item.id for item in message.tool_calls}) != len(message.tool_calls):
                return False
            waiting = {item.id for item in message.tool_calls}
        elif message.role == "tool":
            for result in message.tool_results:
                if result.tool_call_id not in waiting:
                    return False
                waiting.remove(result.tool_call_id)
        elif waiting:
            return False
    return not waiting


def _reasoning(messages):
    return [
        jcs(part.model_dump(mode="json"))
        for message in messages
        for part in message.parts
        if part.type == "reasoning"
    ]
