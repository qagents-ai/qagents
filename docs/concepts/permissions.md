# Permissions and approvals

Whether a tool call runs is decided by ordinary code, outside the model. A prompt cannot talk its
way past a rule.

## The decision order

For every tool call the loop asks `Permissions.decide`, which checks in this order and stops at
the first match:

1. `deny` patterns: the call is refused.
2. `can_use_tool`: your function may allow, ask or deny, or return `None` for "no opinion".
3. `ask` patterns: a human is asked.
4. `allow` patterns: the call runs.
5. The `mode` default.

The first rule that answers decides. `deny` is checked first, so a deny always wins; an allow
from `can_use_tool` skips the `ask` rules, `requires_approval` and the mode. If anything goes wrong
while deciding (your callback raises, or returns something that is not a decision), the call is
refused.

```python
from qagents import Agent, Permissions

agent = Agent(
    "qcloud:<model-id>",
    tools=[read_order, refund_order, send_email, delete_customer],
    permissions=Permissions(
        allow=["read_*"],
        ask=["send_*", "refund_*"],
        deny=["delete_*"],
    ),
)
```

Patterns are shell-style and match tool names.

| `mode` | Calls that no rule matched |
|---|---|
| `"default"` | tools declared `requires_approval=True` ask; the rest run |
| `"allow_all"` | run |
| `"ask_all"` | ask |
| `"read_only"` | run only if the tool is declared `read_only=True`; otherwise refused |

A refused call is not an exception. The model receives an error result such as
`'delete_customer' was not run: a deny rule matches this tool.` and can tell the user.

## Decisions that depend on the arguments

```python
def refunds_up_to_20(ctx, tool, call):
    if tool.name == "refund_order":
        return "allow" if call.arguments["amount"] <= 20 else "ask"
    return None  # no opinion: continue with the rules


permissions = Permissions(can_use_tool=refunds_up_to_20)
```

`can_use_tool` may be async, and may return `"allow"`, `"ask"`, `"deny"`, a bool, a `Decision`
with a reason, or `None`.

## Approvals, inline

When the answer is "ask" and you gave an `on_ask` handler, it is awaited on the spot. Use this
where a person is present: a terminal, a chat UI holding a connection open, a Slack button.

```python
from qagents import Approval, Permissions


async def ask_in_slack(ctx, tool, call, reason) -> Approval:
    approved = await slack.confirm(f"Allow {tool.name}({call.arguments})?")
    return Approval(approved=approved, reason=None if approved else "Declined in Slack.")


permissions = Permissions(on_ask=ask_in_slack)
```

An `Approval` can also carry `arguments`, to run the tool with values the approver edited. If the
handler raises, the call is refused.

## Pause and resume

Without `on_ask`, the run pauses. Nothing is held in memory: the pause is written to the session,
so the approval can arrive minutes or days later, in another process.

```python
result = await agent.run("Refund order A-1001", session=session)

if result.status == "paused":
    for pending in result.pending_approvals:
        print(pending.tool_call_id, pending.name, pending.arguments, pending.reason)

    result = await agent.resume(
        session,
        approvals={result.pending_approvals[0].tool_call_id: True},
    )
```

`approvals` maps tool call ids to `True`, `False` or an `Approval`. What to know:

- Calls of the same turn that needed no approval ran before the pause. They are not run again.
- Approve only some of the pending calls and the run pauses again with the rest.
- A declined call becomes an error result with your reason, and the model continues from it.
- If you did not pass a session, one was created: use `result.session`.
- The pending calls are also in `(await session.state()).pending_approvals`, which is what an
  approvals page would read.

### Signed pauses

Configure `Permissions(approval_secret=...)` to sign paused approvals and deferred inputs.
Return `Approval.for_call(pending)` or `InputResult.for_call(pending, value)` when resuming;
these carry the recorded token. A missing token or a changed signed call raises
`ApprovalTokenError` before a tool runs. Authenticate whoever calls `resume` in your
application: the token protects the paused call's integrity, not the reviewer's identity
or the rest of the journal.

Effects tools need the secret as well. When an effect's outcome is unknown, a tool that asks
for approval re-sends it automatically (rules D3 and D4) only under its original decision,
and only if that decision verifies. Without `approval_secret` no decision verifies, so such
an effect is never re-sent automatically, in the run or after a crash: unless a declared
lookup finds it, it is held as `APPROVAL_UNVERIFIABLE` until an operator settles it with
`agent.settle`. The agent logs a warning when it is built (or, for a toolkit's tools, when
a run starts). See
[Effects tools that ask for approval](../guides/recover-effect.md#effects-tools-that-ask-for-approval).

Signed JSON integers must be between `-(2**53 - 1)` and `2**53 - 1`, inclusive. Larger
integers are refused rather than rounded. PostgreSQL JSONB can normalize an integral float
to an integer; a large float normalized outside this range also fails verification on
resume. Use strings for identifiers or exact quantities beyond the safe integer range.
