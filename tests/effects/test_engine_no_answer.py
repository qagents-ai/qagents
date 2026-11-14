"""A send that applied and then lost its response is never re-sent under a new key.

Regression for the analysis probe s5 (findings fable-2): with ``rejected_on`` naming a
base class of a timeout, the timeout counted as a definite refusal. Under
``replays_rejections=True`` (the Stripe preset) the model's retry took D2 with a new
idempotency key, and the destination applied the charge twice. The ``wrapped_*`` cases
are the AF-1 review's shape: the client wraps the httpx timeout in its own refusal type
(``raise PaymentsError(...) from error``), as most clients and SDKs do.
"""

import asyncio

import httpx
import pytest
from pydantic import BaseModel

from qagents import Agent, RunContext, tool
from qagents.effects import Absent, Attempt, EffectContext, Found, HoldReason, Key, effect
from qagents.effects.contracts import STRIPE_IDEMPOTENCY_V1
from qagents.errors import EffectRejected
from qagents.models.testing import ScriptedModel, call
from qagents.session import InMemoryStore, Session, SQLiteStore


class Charge(BaseModel):
    order: str


class ChargeReceipt(BaseModel):
    order: str


class PaymentsError(Exception):
    """A client's base error, listed in rejected_on for the refusals it receives."""


class PaymentsTimeout(PaymentsError, TimeoutError):
    """The client's read timeout: no answer, although it derives from PaymentsError."""


class PaymentsReadTimeout(PaymentsError, httpx.ReadTimeout):
    """The same for a client built on httpx."""


class Destination:
    """A Stripe-like destination: at most one application per idempotency key."""

    def __init__(self, lost: type[BaseException]) -> None:
        self.lost = lost
        self.keys: list[str] = []
        self.applied: dict[str, str] = {}

    def charge(self, attempt: Attempt[Charge]) -> ChargeReceipt:
        self.keys.append(attempt.idempotency_key)
        first = not self.applied
        self.applied.setdefault(attempt.idempotency_key, attempt.operation_tag)
        if first:
            raise self.lost("read timed out after the destination applied the charge")
        return ChargeReceipt(order=attempt.request.order)


# (raised error, rejected_on forced past the declaration check as in the probe, how the
# send wraps a transport error in PaymentsError: None, "from", "context" or "from_none";
# "task_group" runs the charge in an asyncio.TaskGroup, so it raises an ExceptionGroup)
CASES = {
    "timeout_subclass": (PaymentsTimeout, None, None),
    "httpx_subclass": (PaymentsReadTimeout, None, None),
    "undeclared_oserror": (TimeoutError, (OSError,), None),
    "wrapped_from": (httpx.ReadTimeout, None, "from"),
    "wrapped_context": (httpx.ReadTimeout, None, "context"),
    "wrapped_from_none": (httpx.ReadTimeout, None, "from_none"),
    "task_group": (httpx.ReadTimeout, (ExceptionGroup,), "task_group"),
    "task_group_wrapped": (httpx.ReadTimeout, None, "task_group_wrapped"),
}


@pytest.fixture(params=("memory", "sqlite"))
async def store(request, tmp_path):
    value = InMemoryStore() if request.param == "memory" else SQLiteStore(tmp_path / "s5.sqlite")
    async with value:
        yield value


def build(destination: Destination, forced: tuple | None, wrap: str | None, *, lookup: bool):
    looked: list[str] = []

    @effect(
        "shop.charge",
        key=Key("charge", "order"),
        namespace="shop",
        timeout_s=2,
        rejected_on=(PaymentsError,),
        dedupe=STRIPE_IDEMPOTENCY_V1,
    )
    async def charge(ctx: EffectContext, attempt: Attempt[Charge]) -> ChargeReceipt:
        if wrap is None:
            return destination.charge(attempt)
        if wrap.startswith("task_group"):

            async def leg() -> ChargeReceipt:
                return destination.charge(attempt)

            try:
                async with asyncio.TaskGroup() as group:  # say, the charge and an audit call
                    task = group.create_task(leg())
            except* httpx.HTTPError as lost:
                if wrap == "task_group_wrapped":
                    raise PaymentsError("payments request failed") from lost
                raise
            return task.result()
        try:
            return destination.charge(attempt)
        except httpx.HTTPError as error:
            # The usual client wrapper: its own error type, listed in rejected_on.
            if wrap == "from":
                raise PaymentsError(f"payments request failed: {error}") from error
            if wrap == "context":
                raise PaymentsError(f"payments request failed: {error}")  # noqa: B904
            raise PaymentsError(f"payments request failed: {error}") from None

    if forced is not None:
        # The probe's declaration, which @effect refuses now. A worker leg rebuilds
        # effects without declaring them again, so classification must not rely on it.
        charge.rejected_on = forced

    if lookup:

        @charge.lookup
        async def find(ctx: EffectContext, inquiry) -> Found | Absent:
            looked.append(inquiry.operation_tag)
            if inquiry.operation_tag in destination.applied.values():
                return Found(ChargeReceipt(order=inquiry.request.order), tag=inquiry.operation_tag)
            return Absent()

    @tool(effects=[charge])
    async def pay(ctx: RunContext, order: str) -> str:
        """Charge the order."""
        try:
            return (await ctx.perform(charge, Charge(order=order))).order
        except EffectRejected:
            return "rejected"

    return pay, looked


def model() -> ScriptedModel:
    # The model retries the charge once, as in the probe.
    return ScriptedModel([call("pay", order="O1"), call("pay", order="O1"), "Done."])


@pytest.mark.parametrize("case", CASES)
async def test_no_answer_after_application_is_held_not_resent_under_a_new_key(store, case):
    raised, forced, wrap = CASES[case]
    destination = Destination(raised)
    pay, _ = build(destination, forced, wrap, lookup=False)
    result = await Agent(model(), tools=[pay]).run("pay", session=Session(store, id="s"))
    assert len(destination.applied) == 1, destination.keys
    assert len(set(destination.keys)) == 1, destination.keys
    # Stripe replays errors for a key, so no same-key re-send helps either (SPEC 7.1 E4).
    assert len(destination.keys) == 1
    assert result.status == "paused"
    assert [held.reason for held in result.held_effects] == [HoldReason.RESPONSE_LOST]


@pytest.mark.parametrize("case", CASES)
async def test_no_answer_after_application_is_looked_up_not_resent(store, case):
    raised, forced, wrap = CASES[case]
    destination = Destination(raised)
    pay, looked = build(destination, forced, wrap, lookup=True)
    scripted = model()
    result = await Agent(scripted, tools=[pay]).run("pay", session=Session(store, id="s"))
    assert result.status == "completed"
    assert len(destination.applied) == 1 and len(destination.keys) == 1
    assert len(looked) == 1
    # The first call is answered from the lookup, never reported to the model as refused.
    assert scripted.requests[1].messages[-1].tool_results[0].content == "O1"


class Stalled:
    """A destination that applies the first charge and never answers it in time."""

    def __init__(self, style: str) -> None:
        self.style = style
        self.keys: list[str] = []
        self.applied: dict[str, str] = {}

    async def charge(self, attempt: Attempt[Charge]) -> ChargeReceipt:
        self.keys.append(attempt.idempotency_key)
        first = not self.applied
        self.applied.setdefault(attempt.idempotency_key, attempt.operation_tag)
        if not first:
            return ChargeReceipt(order=attempt.request.order)
        aborted = False
        try:
            await asyncio.sleep(30)  # the response does not arrive before timeout_s
        except asyncio.CancelledError as error:
            # A client that turns the engine's cancellation into its own error type.
            if self.style == "from":
                raise PaymentsError("request aborted") from error
            if self.style == "context":
                raise PaymentsError("request aborted")  # noqa: B904
            aborted = True
        if aborted:
            raise PaymentsError("request aborted")  # after the handler: no chain to read
        return ChargeReceipt(order=attempt.request.order)


@pytest.mark.parametrize("style", ["from", "context", "after"])
async def test_refusal_raised_after_the_engine_cancelled_the_send_is_its_timeout(store, style):
    # AF-1 review: SPEC 7.1 E5 cancels an async send at its deadline, "Then E4". A
    # declared refusal type the send raises in reaction is that timeout, not E3.
    destination = Stalled(style)

    @effect(
        "shop.charge",
        key=Key("charge", "order"),
        namespace="shop",
        timeout_s=0.3,
        rejected_on=(PaymentsError,),
        dedupe=STRIPE_IDEMPOTENCY_V1,
    )
    async def charge(ctx: EffectContext, attempt: Attempt[Charge]) -> ChargeReceipt:
        return await destination.charge(attempt)

    @tool(effects=[charge])
    async def pay(ctx: RunContext, order: str) -> str:
        """Charge the order."""
        try:
            return (await ctx.perform(charge, Charge(order=order))).order
        except EffectRejected:
            return "rejected"

    await Agent(model(), tools=[pay]).run("pay", session=Session(store, id="s"))
    assert len(destination.applied) == 1, destination.keys
    assert len(set(destination.keys)) == 1, destination.keys
    failed = [entry.data for entry in await store.load("s") if entry.kind == "effect_failed"]
    assert [(data["why"], data["error_kind"], data["committed"]) for data in failed] == [
        ("unknown", "timeout", None)
    ]
