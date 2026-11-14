"""An approval-gated effects tool without an approval_secret warns at build (SPEC 5.1, 5.4).

D3 and D4 re-send only under a verifiable decision, and without a secret no decision
verifies, so such an effect is never re-sent automatically: where D3 or D4 would re-send it,
one whose outcome is in doubt is held as APPROVAL_UNVERIFIABLE until an operator settles
it. The hold is right (fail closed); the warning tells the developer before the first crash
does. An effect that no row would re-send keeps its own hold reason (SPEC 7.1 H1), so a tool
with only such effects is not named.
"""

import asyncio
import logging

import pytest
from pydantic import BaseModel

from qagents import Agent, Permissions, RunContext, Session, SQLiteStore, Toolkit, tool
from qagents.effects import Absent, Attempt, Dedupe, EffectContext, Key, Unique, effect
from qagents.effects._types import HoldReason
from qagents.models.testing import FunctionModel, call

SECRET = "unsigned-approvals-fixture-secret"
LOGGER = "qagents"


class Request(BaseModel):
    id: str


class Receipt(BaseModel):
    id: str


def refund_effect(destination=None, *, slow_first=False):
    @effect(
        "refund",
        key=Key("request", "id"),
        dedupe=Dedupe(window_s=24 * 3600, contract="test destination", concurrent="rejects"),
        timeout_s=0.05,
    )
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        if destination is not None:
            destination.append(attempt.idempotency_key)
            if slow_first and len(destination) == 1:
                await asyncio.sleep(1)  # past timeout_s: the outcome is in doubt
        return Receipt(id="paid")

    return send


def refund_tool(name="refund_order", *, requires_approval=True, send=None):
    send = send or refund_effect()

    @tool(name=name, effects=[send], requires_approval=requires_approval)
    async def refund_order(ctx: RunContext, id: str) -> str:
        """Refund an order in the test destination."""
        return (await ctx.perform(send, Request(id=id))).id

    return refund_order


def model():
    return FunctionModel(
        lambda request: (
            "Refunded." if request.messages[-1].role == "tool" else call("refund_order", id="R1")
        )
    )


def warnings_about_secret(caplog):
    return [
        record
        for record in caplog.records
        if record.name == LOGGER
        and record.levelno == logging.WARNING
        and "approval_secret" in record.getMessage()
    ]


@pytest.mark.parametrize(
    "permissions,requires_approval",
    [
        (Permissions(), True),
        (Permissions(ask=["refund_*"]), False),
        (Permissions(mode="ask_all"), False),
        (Permissions(on_ask=lambda *args: True), True),  # answered in-process, still unsigned
    ],
    ids=["requires_approval", "ask_pattern", "ask_all", "on_ask"],
)
def test_build_warns_once_for_an_approval_gated_effects_tool_without_a_secret(
    caplog, permissions, requires_approval
):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    Agent(
        model(),
        tools=[refund_tool(requires_approval=requires_approval)],
        permissions=permissions,
    )
    records = warnings_about_secret(caplog)
    assert len(records) == 1  # one warning, not one per check
    record = records[0]
    message = record.getMessage()
    assert message.startswith("refund_order: this effects tool asks for approval")
    assert "APPROVAL_UNVERIFIABLE" in message and "D3 and D4" in message
    assert "an operator settles it" in message
    assert "Set Permissions(approval_secret=...)" in message


@pytest.mark.parametrize(
    "permissions,requires_approval",
    [
        (Permissions(approval_secret=SECRET), True),
        (Permissions(approval_secret=SECRET, mode="ask_all"), False),
        (Permissions(), False),  # an effects tool that runs without asking
        (Permissions(allow=["refund_*"]), True),  # an allow pattern answers first
        (Permissions(mode="allow_all"), True),
        (Permissions(deny=["refund_*"]), True),  # never runs, so never re-sent
    ],
    ids=["secret", "secret_ask_all", "no_approval", "allow_pattern", "allow_all", "denied"],
)
def test_build_does_not_warn_with_a_secret_or_without_an_approval_gated_effect(
    caplog, permissions, requires_approval
):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    Agent(
        model(),
        tools=[refund_tool(requires_approval=requires_approval)],
        permissions=permissions,
    )
    assert warnings_about_secret(caplog) == []


def test_an_approval_gated_tool_without_effects_does_not_warn(caplog):
    @tool(requires_approval=True)
    def delete_order(id: str) -> str:
        """Delete an order."""
        return "deleted"

    caplog.set_level(logging.WARNING, logger=LOGGER)
    Agent(model(), tools=[delete_order])
    assert warnings_about_secret(caplog) == []


class Duplicate(Exception):
    pass


def shaped_effect(shape):
    """One effect per destination contract; D3 or D4 can re-send only the first three."""
    contracts = {
        "dedupe": {"dedupe": Dedupe(window_s=3600, contract="test", concurrent="rejects")},
        "unique": {"dedupe": Unique(contract="test unique key", on_duplicate=(Duplicate,))},
        "authoritative": {"applies_within_s": 1.0},
    }

    @effect(f"{shape}_notice", key=Key("notice", "id"), **contracts.get(shape, {}))
    async def notice(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        return Receipt(id="sent")

    if shape in ("unique", "authoritative", "reconcilable"):  # Unique needs a lookup too

        @notice.lookup(authoritative=shape == "authoritative")
        async def find(ctx, inquiry):
            return Absent()

    assert notice.klass == {"dedupe": "deduplicated"}.get(shape, shape)
    return notice


def notice_tool(name, *effects):
    @tool(name=name, effects=list(effects), requires_approval=True)
    async def notify(ctx: RunContext, id: str) -> str:
        """Tell the customer."""
        return "sent"

    return notify


@pytest.mark.parametrize("shape", ["dedupe", "unique", "authoritative"])
def test_build_warns_for_each_effect_contract_with_an_automatic_resend(caplog, shape):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    Agent(model(), tools=[notice_tool("notify_customer", shaped_effect(shape))])
    assert [record.getMessage().split(":")[0] for record in warnings_about_secret(caplog)] == [
        "notify_customer"
    ]


@pytest.mark.parametrize("shape", ["opaque", "reconcilable"])
def test_an_approval_gated_effects_tool_with_no_automatic_resend_does_not_warn(caplog, shape):
    """No D3 or D4 can ever re-send these effects, so the secret changes nothing for them:
    an effect in doubt is held for its own reason, with or without it (SPEC 7.1 H1)."""
    caplog.set_level(logging.WARNING, logger=LOGGER)
    Agent(model(), tools=[notice_tool("notify_customer", shaped_effect(shape))])
    assert warnings_about_secret(caplog) == []


def test_one_warning_names_every_approval_gated_effects_tool(caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    refund = refund_tool(send=refund_effect())
    # One effect with an automatic re-send is enough to list the tool.
    notify = notice_tool("notify_customer", shaped_effect("opaque"), shaped_effect("unique"))
    tools = [refund, notify, notice_tool("post_letter", shaped_effect("opaque"))]
    Agent(model(), tools=[*tools, refund_tool("plain", requires_approval=False)])
    records = warnings_about_secret(caplog)
    assert len(records) == 1  # one warning, not one per check
    record = records[0]
    assert record.getMessage().startswith(
        "refund_order, notify_customer: these effects tools ask for approval"
    )


class RefundToolkit(Toolkit):
    def __init__(self, tool_):
        self.tool = tool_

    async def tools(self):
        return [self.tool]


async def test_run_start_warns_once_for_a_toolkit_tool_and_never_repeats(caplog, tmp_path):
    """A toolkit's tools are known only at run start; each tool is warned about once."""
    caplog.set_level(logging.WARNING, logger=LOGGER)
    agent = Agent(model(), tools=[RefundToolkit(refund_tool())])
    assert warnings_about_secret(caplog) == []  # nothing is known at build
    async with SQLiteStore(tmp_path / "toolkit.sqlite") as db:
        for number in range(2):
            paused = await agent.run("Refund R1", session=Session(db, id=f"s{number}"))
            assert paused.status == "paused"  # it asks: the approval pauses the run
    records = warnings_about_secret(caplog)
    assert len(records) == 1  # one warning, not one per check
    record = records[0]
    assert record.getMessage().startswith("refund_order: this effects tool asks")


async def test_static_tools_are_not_warned_about_again_at_run_start(caplog, tmp_path):
    caplog.set_level(logging.WARNING, logger=LOGGER)
    agent = Agent(model(), tools=[refund_tool()])
    assert len(warnings_about_secret(caplog)) == 1
    async with SQLiteStore(tmp_path / "static.sqlite") as db:
        await agent.run("Refund R1", session=Session(db, id="s"))
    assert len(warnings_about_secret(caplog)) == 1


def gated_agent(destination, path, secret):
    agent = Agent(
        model(),
        tools=[refund_tool(send=refund_effect(destination, slow_first=True))],
        permissions=Permissions(approval_secret=secret, on_ask=lambda *args: True),
    )
    return agent, SQLiteStore(path)


async def test_without_a_secret_the_effect_in_doubt_holds_unverifiable_and_is_not_resent(
    tmp_path,
):
    """What the warning says: no automatic same-key re-send, in the run or after a crash."""
    path, destination = tmp_path / "unsigned.sqlite", []
    agent, db = gated_agent(destination, path, None)
    async with db:
        first = await agent.run("Refund R1", session=Session(db, id="s"))
    assert first.stop_reason == "effect_held"
    [held] = first.held_effects
    assert held.reason == HoldReason.APPROVAL_UNVERIFIABLE and held.retry_after is None
    agent, db = gated_agent(destination, path, None)  # a new process
    async with db:
        again = await agent.resume(Session(db, id="s"))
    assert [item.reason for item in again.held_effects] == [HoldReason.APPROVAL_UNVERIFIABLE]
    assert len(destination) == 1


async def test_with_a_secret_the_effect_in_doubt_is_resent_under_its_key(tmp_path):
    path, destination = tmp_path / "signed.sqlite", []
    agent, db = gated_agent(destination, path, SECRET)
    async with db:
        done = await agent.run("Refund R1", session=Session(db, id="s"))
    assert done.status == "completed" and done.output == "Refunded."
    assert len(destination) == 2 and len(set(destination)) == 1  # D3: the same key


def undeduplicated_effect(destination, *, authoritative):
    """No dedupe contract: only D4, after an authoritative absence, can re-send it."""

    @effect(
        "refund",
        key=Key("request", "id"),
        timeout_s=0.05,
        **({"applies_within_s": 0.01} if authoritative else {}),
    )
    async def send(ctx: EffectContext, attempt: Attempt[Request]) -> Receipt:
        destination.append(attempt.idempotency_key)
        if len(destination) == 1:
            await asyncio.sleep(1)  # past timeout_s: the outcome is in doubt
        return Receipt(id="paid")

    if authoritative:

        @send.lookup(authoritative=True)
        async def find(ctx, inquiry):
            return Absent()

    return send


def clocked_agent(destination, secret, *, authoritative):
    send = undeduplicated_effect(destination, authoritative=authoritative)
    return Agent(
        model(),
        tools=[refund_tool(send=send)],
        permissions=Permissions(approval_secret=secret, on_ask=lambda *args: True),
    )


@pytest.mark.parametrize(
    "authoritative,reason",
    [(False, HoldReason.RESPONSE_LOST), (True, HoldReason.NOT_YET_QUIESCENT)],
    ids=["opaque", "authoritative_before_quiescence"],
)
async def test_an_effect_no_row_would_resend_holds_for_its_own_reason_without_a_secret(
    tmp_path, authoritative, reason
):
    """SPEC 7.1 H1: an unverifiable decision is the reason only on a row that would send.

    Here no row would, with or without a secret: an opaque effect has no automatic re-send,
    and an authoritative absence counts only after quiescence. So the hold is the same as
    under a valid decision, retry_after included.
    """
    holds = {}
    for secret in (None, SECRET):
        destination = []
        agent = clocked_agent(destination, secret, authoritative=authoritative)
        async with SQLiteStore(tmp_path / f"{secret is not None}.sqlite") as db:
            paused = await agent.run("Refund R1", session=Session(db, id="s"))
        [held] = paused.held_effects
        holds[secret] = (held.reason, held.retry_after is not None, len(destination))
    assert holds[None] == holds[SECRET] == (reason, authoritative, 1)


@pytest.mark.parametrize("secret", [None, SECRET], ids=["no_secret", "secret"])
async def test_once_d4_would_send_an_unverifiable_decision_holds_the_effect(tmp_path, secret):
    """After quiescence D4 would send: without a secret the decision is now the reason."""
    destination, now = [], [1000.0]
    agent = clocked_agent(destination, secret, authoritative=True)
    async with SQLiteStore(tmp_path / "d4.sqlite", clock=lambda: now[0]) as db:
        paused = await agent.run("Refund R1", session=Session(db, id="s"))
        [held] = paused.held_effects
        assert held.reason == HoldReason.NOT_YET_QUIESCENT and held.retry_after is not None
        now[0] += 3600  # the store clock passes every attempt's quiescence
        again = await agent.resume(Session(db, id="s"))
    if secret is None:
        assert [(item.reason, item.retry_after) for item in again.held_effects] == [
            (HoldReason.APPROVAL_UNVERIFIABLE, None)
        ]
        assert len(destination) == 1
    else:
        assert again.status == "completed" and again.output == "Refunded."
        assert len(destination) == 2 and len(set(destination)) == 1  # D4: the same key
