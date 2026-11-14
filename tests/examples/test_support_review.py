"""The README application exercises approvals and a real durable local destination."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest

from examples.support_review import (
    DEMO_ORDERS,
    ReviewQueue,
    SupportDeps,
    build_agent,
    configured_model,
)
from qagents import Approval, Session, SQLiteStore
from qagents.errors import ToolError
from qagents.models.testing import ScriptedModel, call


def count_cases(path):
    with closing(sqlite3.connect(path)) as db:
        return db.execute("SELECT COUNT(*) FROM cases").fetchone()[0]


async def test_case_is_written_only_after_approval_and_survives_store_reopen(tmp_path):
    path = tmp_path / "queue.db"
    deps = SupportDeps(DEMO_ORDERS, ReviewQueue(path))
    model = ScriptedModel(
        [
            call("inspect_order", order_id="A-1001"),
            call("queue_review", order_id="A-1001", reason="Arrived damaged"),
            "The case is awaiting human review.",
        ]
    )
    agent = build_agent(model)
    async with SQLiteStore(tmp_path / "session.db") as store:
        paused = await agent.run(
            "Inspect and open a review for A-1001", session=Session(store, id="s"), deps=deps
        )
        assert paused.status == "paused" and count_cases(path) == 0
        pending = paused.pending_approvals[0]
    async with SQLiteStore(tmp_path / "session.db") as reopened:
        result = await agent.resume(
            Session(reopened, id="s"),
            deps=deps,
            approvals={pending.tool_call_id: Approval.for_call(pending, by="tester")},
        )
        assert result.status == "completed" and count_cases(path) == 1
    assert len(model.requests) == 3
    assert agent.limits.max_total_tokens == 12000


async def test_declined_case_never_reaches_the_destination(tmp_path):
    path = tmp_path / "queue.db"
    deps = SupportDeps(DEMO_ORDERS, ReviewQueue(path))
    agent = build_agent(
        ScriptedModel(
            [call("queue_review", order_id="A-1001", reason="Damage"), "No case was queued."]
        )
    )
    async with SQLiteStore(tmp_path / "session.db") as store:
        session = Session(store, id="s")
        paused = await agent.run("Open a case", session=session, deps=deps)
        pending = paused.pending_approvals[0]
        result = await agent.resume(
            session,
            deps=deps,
            approvals={pending.tool_call_id: Approval.for_call(pending, approved=False)},
        )
    assert result.status == "completed" and count_cases(path) == 0


def test_destination_deduplicates_concurrent_retries_and_rejects_changed_payload(tmp_path):
    path = tmp_path / "queue.db"
    queue = ReviewQueue(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda _: queue.enqueue("same-op", "A-1001", "Damage"), range(12)))
    assert all(receipt == receipts[0] for receipt in receipts)
    assert count_cases(path) == 1
    assert ReviewQueue(path).enqueue("same-op", "A-1001", "Damage") == receipts[0]
    with pytest.raises(ToolError, match="different review request"):
        queue.enqueue("same-op", "A-1002", "Changed")
    assert count_cases(path) == 1


@pytest.mark.parametrize(
    "provider,prefix", [("qcloud", "QCLOUD"), ("openai", "OPENAI"), ("anthropic", "ANTHROPIC")]
)
def test_provider_requires_its_own_configuration_without_fallback(monkeypatch, provider, prefix):
    monkeypatch.delenv(prefix + "_MODEL", raising=False)
    monkeypatch.delenv(prefix + "_API_KEY", raising=False)
    with pytest.raises(ValueError, match=prefix + "_MODEL"):
        configured_model(provider)


@pytest.mark.parametrize(
    "provider,prefix,class_name",
    [
        ("qcloud", "QCLOUD", "QCloudModel"),
        ("openai", "OPENAI", "OpenAIChatModel"),
        ("anthropic", "ANTHROPIC", "AnthropicModel"),
    ],
)
def test_provider_factory_selects_requested_adapter(monkeypatch, provider, prefix, class_name):
    import qagents

    captured = []
    sentinel = object()

    def construct(model, **kwargs):
        captured.append((model, kwargs))
        return sentinel

    monkeypatch.setattr(qagents, class_name, construct)
    monkeypatch.setenv(prefix + "_MODEL", "explicit-model")
    monkeypatch.setenv(prefix + "_API_KEY", "fixture-key")
    assert configured_model(provider) is sentinel
    assert captured[0][0] == "explicit-model"
    assert captured[0][1]["api_key"] == "fixture-key"
    if provider == "openai":
        assert captured[0][1]["base_url"] == "https://api.openai.com/v1"
        assert captured[0][1]["max_tokens_param"] == "max_completion_tokens"
