from __future__ import annotations

import json

from examples.scoped_memory import Identity, RunAudit, demo, preference_agent, recall
from qagents import Session, SQLiteStore
from qagents.models.testing import ScriptedModel, call


async def test_approved_preferences_survive_reopen_without_crossing_identity_boundaries(tmp_path):
    result = await demo(tmp_path / "preferences.db")
    assert result["same_user_after_reopen"] == "el"
    assert result["another_user"] == result["another_tenant"] == "unset"
    assert [row["status"] for row in result["audit"]] == [
        "paused",
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    assert all(
        set(row) == {"run_id", "status", "requests", "total_tokens"} for row in result["audit"]
    )
    assert "Remember that I prefer Greek" not in json.dumps(result["audit"])


async def test_declined_memory_write_changes_no_session_state(tmp_path):
    audit = RunAudit()
    model = ScriptedModel(
        [call("remember_preference", key="language", value="private"), "Declined."]
    )
    agent = preference_agent(model, audit)
    async with SQLiteStore(tmp_path / "preferences.db") as store:
        session = Session(store, id=Identity("tenant", "user").session_id)
        paused = await agent.run("Remember a preference.", session=session)
        assert "preferences" not in (await session.state()).values
        assert not any(entry.kind == "tool_started" for entry in await session.entries())
        completed = await agent.resume(
            session, approvals={paused.pending_approvals[0].tool_call_id: False}
        )
        assert completed.completed
        assert "preferences" not in (await session.state()).values
        assert await recall(store, Identity("tenant", "user"), audit) == "unset"


def test_scope_ids_do_not_collide_when_identifiers_contain_delimiters():
    identities = [
        Identity("a:b", "c"),
        Identity("a", "b:c"),
        Identity("a/b", "c"),
        Identity("a%2Fb", "c"),
    ]
    assert len({identity.session_id for identity in identities}) == len(identities)


async def test_approval_resumes_after_reopening_the_store_with_a_new_agent(tmp_path):
    database = tmp_path / "preferences.db"
    identity = Identity("tenant", "alice")
    audit = RunAudit()
    first = preference_agent(
        ScriptedModel([call("remember_preference", key="language", value="el")]), audit
    )
    async with SQLiteStore(database) as store:
        paused = await first.run("Remember Greek.", session=Session(store, id=identity.session_id))
    second = preference_agent(ScriptedModel(["Saved."]), audit)
    async with SQLiteStore(database) as store:
        session = Session(store, id=identity.session_id)
        result = await second.resume(
            session, approvals={paused.pending_approvals[0].tool_call_id: True}
        )
        assert result.completed and result.run_id == paused.run_id
        assert (await session.state()).values["preferences"] == {"language": "el"}
        starts = [entry for entry in await session.entries() if entry.kind == "tool_started"]
        assert len(starts) == 1
        assert await recall(store, identity, audit) == "el"


async def test_scope_identity_is_not_a_model_controlled_tool_argument(tmp_path):
    audit = RunAudit()
    model = ScriptedModel(["done"])
    agent = preference_agent(model, audit)
    async with SQLiteStore(tmp_path / "preferences.db") as store:
        await agent.run("Go.", session=Session(store, id=Identity("tenant", "alice").session_id))
    for definition in model.requests[0].tools:
        assert set(definition.parameters["properties"]) <= {"key", "value"}
