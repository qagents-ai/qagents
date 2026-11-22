# Effects 0.3 preview


Effects makes an external action, such as a credit or a refund, a declared operation with a
journal of intents and receipts. Before every send, the worker journals a durable intent with
the frozen request. After a crash or a lost response, recovery decides from journaled
evidence, the destination's declared contract or a lookup whether it may send again. When the
evidence cannot decide, the effect is held, and a held effect is not sent again until a lookup
or an operator settles it. An unknown outcome is sent again automatically only under a
declared `Dedupe` window or `Unique` contract, with the same idempotency key, or after an
authoritative lookup finds it absent. For a tool that asks for approval, it also needs a
decision signed with `approval_secret`.

Effects is on `main` and not released. The package version is still 0.2.0 and the changelog
has no Effects entry. It works on `InMemoryStore` and `SQLiteStore` only.

```python
from pydantic import BaseModel
from qagents import RunContext, tool
from qagents.effects import Attempt, EffectContext, Key, Settle, effect


class Credit(BaseModel):
    order: str
    amount: int


class Receipt(BaseModel):
    order: str
    amount: int


@effect("billing.credit", key=Key("request", "order"), timeout_s=5)
async def credit(ctx: EffectContext, attempt: Attempt[Credit]) -> Receipt:
    """Send the credit with the attempt's durable idempotency key."""
    return await ctx.deps.billing.credit(attempt.request, key=attempt.idempotency_key)


@tool(effects=[credit])
async def credit_order(ctx: RunContext, order: str) -> Receipt:
    """Credit the order for ten units."""
    return await ctx.perform(credit, Credit(order=order, amount=10))


# After a lost response the run stops with stop_reason "effect_held". An operator who has
# checked the destination settles the held effect, and the run can then resume:
#
# await agent.settle(session, Settle.committed(
#     result.held_effects[0], receipt=verified, by="ops@example.com", evidence="ledger row 991",
# ))
```

[Recover an external action](guides/recover-effect.md) runs this flow end to end with two
`AgentRuntime` workers, and runs the crash harness from the command line.

<details>
<summary>Effects API</summary>

| Feature | API |
|---|---|
| A typed effect with a Pydantic request and receipt. The send function receives an `Attempt` that carries a durable idempotency key. The declaration is checked when it is declared and when the agent is built. | `@effect(name, key=..., dedupe=..., timeout_s=..., on_unknown=...)`, `Attempt`, `EffectContext`, `@tool(effects=[...])`, `ctx.perform` |
| A business key rendered from a prefix and scalar request fields (`Key("request", "order")` renders `request:<order>`), or one key per tool call. Per-attempt idempotency keys are derived from it. | `Key("request", "order")`, `Key.per_call()`, `Attempt.idempotency_key` |
| Destination contracts: an idempotency-key window, or a business identity that the destination rejects as a duplicate. The strongest declared contract sets the effect class. | `Dedupe(window_s, contract, concurrent=..., replays_errors=..., replays_rejections=..., binds_params=...)`, `Unique(...)` |
| Definite refusals only. `rejected_on` lists the exceptions your client raises for a refusal it received. Timeouts, lost or refused connections, cancellations, responses cut short and client-library transport errors (httpx, httpx2, httpcore, requests, urllib3, aiohttp) are refused there at declaration, with their base classes, subclasses and exception groups. A send that raises one of them, or raises any error from or while handling one, has an unknown outcome. So does any error an async send raises after the engine's own `timeout_s` cancelled it. | `@effect(..., rejected_on=(...))` |
| A lookup asks the destination whether an earlier attempt applied. A recheck runs lookups without sending. | `effect.lookup(fn, authoritative=...)`, `Found`, `NotYetVisible`, `Absent`, `Unknown` |
| Cross-session claims. A session that takes over a business key whose send was rejected in another session first fences itself, journals the transfer with every earlier attempt, and runs the declared lookup before any new send. The earlier owner can take back a claim that the taker never sent. | internal: claim transfer, `Session.effects(...)` |
| Decisions from journaled evidence only: an ordered 42-row transition table in which five rules (D1-D5) may authorize a send, each after a durable intent with a frozen request and fence. | internal: `qagents.effects._table` |
| Held effects: when the journal cannot decide an outcome, the effect is held with one of 17 hold reasons. With the default `on_unknown="pause"` the run pauses with `stop_reason == "effect_held"`. `"fail_run"` fails the run and `"tell_model"` reports the unknown outcome to the model. | `RunResult.held_effects`, `HeldEffect`, `HoldReason`, `EffectHeld` |
| Operator settlements with attribution and evidence, checked under the session lease against fresh evidence and, when the answer is built from a `HeldEffect` or `EffectRecord`, the record revision. `Settle` refuses an empty or non-string record id. | `Settle.committed`, `not_applied`, `resend`, `close_unknown`. `Agent.settle`, `Agent.recheck`, `AgentRuntime.settle`, `AgentRuntime.recheck` |
| Signed operator answers. Under `approval_secret`, a resend is used only when its MAC and record revision verify. A forged or copied resend authorizes no send and holds the effect as `APPROVAL_UNVERIFIABLE`. The other answers send nothing and are not verified when a run reads them. | `Permissions(approval_secret=...)`, `Permissions.verify_settlement` |
| Effects tools that ask for approval re-send an effect in doubt (rules D3 and D4) only under the original decision, and only if it verifies and has not expired, so they need `approval_secret`. Without it the agent logs a warning when it is built, or at run start for a toolkit's tools, and such an effect is held as `APPROVAL_UNVERIFIABLE` unless a lookup finds it. | `Permissions(approval_secret=...)` |
| Effect events, and effect records listed by status and namespace. | `EffectStarted`, `EffectFinished`, `EffectSettled`, `Session.effects(...)`, `EffectRecord` |
| A Stripe idempotency preset declared from the Stripe API v1 documentation and marked "probes pending". | `qagents.effects.contracts.STRIPE_IDEMPOTENCY_V1` |
| Strict RFC 8785 canonical JSON with the safe-integer limit for effect requests, receipts, memo values and approval digests. Effect requests and receipts are also limited to 16 KiB. | internal: `qagents.effects._canonical` |
| Unit-test helpers for send and lookup functions, without running an agent. | `qagents.testing.effects.attempt_for`, `inquiry_for` |

</details>

<details>
<summary>Crash harness and evidence</summary>

| Feature | API |
|---|---|
| Runs an offline scenario, derives crash and fault points from the durable boundaries it observes, and replays each point with recovery legs. It checks destination applications, journal prefixes, decisions, usage and surfaced holds. | `qagents.testing.crash.crash_test(build, prompt, store="memory" \| "sqlite", faults=..., points=..., isolation=...)`, `CrashEnv`, `World` |
| 38 named faults: kills before or after writes, freezes, lost, late or error responses, rejections, lease loss, clock steps, approval redelivery, competing runs and operator races. | `Faults.default()`, `Faults.quick()`, `Faults.parse(...)` |
| Invariants over observed destination applications and every durable journal prefix, plus your own. | `crash_test(invariants=[...])` |
| Model checks: recovered model requests are compared with durable history and with the OpenAI and Anthropic wire encodings. The repository tests also check a bounded model of the authoritative-lookup protocol. | `Cassette`, `when`, `UnscriptedRequest` |
| Destination fakes, and a client transport that runs your real send and lookup code against them. | `Fake`, `FakeStripeRefunds`, `CrashEnv.service`, `transport="client"` |
| Real worker kills: each leg runs in a fresh interpreter and is killed at the selected boundary before it receives the response. Needs SQLite, freeze mode and an importable build function. | `crash_test(isolation="subprocess")` |
| Per-point verdicts (safe, violated, inconclusive) with coverage gaps. Pin one point and replay it. | `CrashReport.explain`, `to_json`, `to_junit`, `pin`, `assert_recovers` |
| A command line with JSON and JUnit reports. It exits 1 on a violation, 2 on a usage error and 3 when the report is not ok (inconclusive or vacuous). With `--strict`, any report that is not strictly ok, gaps included, exits 4 instead. | `qagents crashtest MODULE:FUNCTION PROMPT --cassette FILE` |
| A pytest fixture that runs `crash_test` in the test's `tmp_path`. | `crash_env` |
| Store conformance through a reopenable factory. For SQLite it kills a real writer subprocess after an acknowledged append. | `qagents.testing.stores.check_store` |
| Real-kill lane (repository only): in 14 cases the test kills its own worker process (`Process.kill()`) just before or after intent, commit and tool-result writes, or after a destination application. The lane has 26 process tests in all. | `pytest -m process tests/process` with `QAGENTS_RUN_PROCESS_TESTS=1` |
| Mutation catalogue (repository only): 68 seeded recovery defects, each run in an isolated interpreter against an unmodified baseline. | `pytest -m mutants tests/effects/mutants` with `QAGENTS_RUN_MUTATIONS=1` |
| A stateful Stripe-like refund endpoint on a loopback port, and an opt-in live probe that accepts only `sk_test_` keys. | `tests/probes` |

</details>

## Availability and limits

- `PostgresStore` lacks the store features effects need, so effect tools are refused there.
- Effect-level approvals (`approval="effect"`) and `Workflow.effect` raise `ConfigurationError`.
  `Workflow.step` stays at-least-once.
- There is no hermetic-replay invariant (I9) and no operator command line. Settling needs Python.
- The Stripe preset has not been checked against Stripe. Only the loopback simulation has run;
  the live sandbox probe has not.
- `OTelHooks`, `MLflowHooks` and the AG-UI projection do not handle effect events.
- There is no Effects reference page yet. `AGENTS.md` and the skill still teach the 0.2
  `ctx.operation_id` pattern, and the docs server does not resolve bare names such as `crash_test`.

## Recovery review

The 2026-11-14 review found safety issues in transport errors, claim transfers, operator
settlements and lease loss. The fixes have regression tests and seeded mutations.
The unresolved operator path is described under [Open issues](#open-issues).

- A timeout, a lost connection or a cut-short response from the standard library, httpx,
  httpx2, httpcore, requests, urllib3 or aiohttp is no longer a definite refusal, even when the
  client wraps it in a declared refusal type. With `replays_rejections=True`, which the Stripe preset sets, the
  review's destination saw two keys and two applications. It now sees one of each.
- A cross-session send (D2) after a rejected claim runs the declared lookup first.
- Operator resends are verified under `approval_secret` before anything is sent, and `Settle`
  refuses an empty or non-string record id.
- Approval-gated effects tools without an `approval_secret` still lose automatic re-sends
  (D3/D4), but the agent now warns about them when it is built, or at run start for a toolkit's
  tools.
- `AgentRuntime.start`, `resume` and `recover`, and the operator lease of `settle` and
  `recheck`, no longer leave a stray `CancelledError` when the lease is lost.

## Open issues

- A rejected record whose declared lookup is inconclusive is held, in the same session or
  after a claim transfer, but has no operator exit yet: `settle` answers `already_settled`,
  `recheck` and the held listings skip it, and `effects()` shows it as `rejected`. It fails
  closed: nothing is sent, and a resume runs the lookup again.
- The no-answer set is a fixed list. Errors of other transports, such as h11, websockets, gRPC
  and botocore, are not in it, so `rejected_on` accepts their base types, and SDK types such as
  `anthropic.APIError`, `APIConnectionError` and `APITimeoutError`. Such an error counts as
  unknown only when a listed error is in its cause chain.
- A `Settle` built from a record id string skips the revision check, so it cannot see a change
  made after the operator looked. Pass the `HeldEffect` or `EffectRecord` instead.
- Workers built before these fixes ignore claim transfers and use an unverified resend. Do not
  run them beside workers built after.
- `crash_test` with `Faults.default()` reports violations on correct fixtures (about 40 of 364
  points. 4 of 35 with `Faults.quick()`), and the guide's `qagents crashtest` command fails on
  correct agents. `budget=` drops depth-2 points without saying so.

Effects is an exercised preview with real-kill evidence for the cases it tests. It is not a
released guarantee.
