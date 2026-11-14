# Recover an external action after a lost response

A provider can apply a write and lose the response. The worker needs a
durable intent before it sends the request. During recovery, the recorded
receipt, the destination's lookup and idempotency contract, or an operator
decision determine how execution continues.

This example uses a scripted model and two SQLite databases. One database
holds the run journal. The other simulates the destination's durable ledger.
The first worker commits a credit and then loses its response. A second
worker opens the journal, records an operator's verified receipt, and resumes
the agent. The example checks that the destination contains one credit.

```bash
python -m examples.recover_credit --directory work/credit-recovery-demo
```

Use a fresh directory for each run. The example uses the Effects runtime. In an application, verify the destination record against
the business identity and amount, then supply the authenticated operator's
identity to `Settle.committed()`.

```python
--8<-- "examples/recover_credit.py"
```

The [Gym recovery benchmark](https://github.com/mpvasilis/qagent-gym/blob/main/docs/recovery-benchmark.md)
uses worker process termination and a separate HTTP destination. It includes
trials with destination deduplication disabled to confirm that the benchmark
detects duplicate credits.

## Effects tools that ask for approval

A tool asks for approval when it is declared `requires_approval=True`, matches
an `ask` pattern, runs under `Permissions(mode="ask_all")`, or when your
`can_use_tool` returns `"ask"` for the call. Its effects are sent under the
decision a person made. When an attempt's outcome is unknown, the automatic
re-sends need that decision again: rule D3 (the same key, inside the
destination's dedupe window) and rule D4 (after an authoritative absence)
send only under the original decision, and only if it verifies and has not
expired. A decision verifies only under `Permissions(approval_secret=...)`,
which signs the journaled decision.

Only the recovery of effects that D3 or D4 can re-send depends on the secret:
those with a `Dedupe` or `Unique` contract, or with an authoritative lookup. An effect with
neither is never re-sent automatically, with or without the secret, and is
held for its own reason, such as `RESPONSE_LOST`.

Without `approval_secret`:

- No effect of such a tool is re-sent automatically, in the same run or after
  a crash. Where D3 or D4 would re-send it, and unless a declared lookup finds
  it, an effect whose outcome is unknown is held with `APPROVAL_UNVERIFIABLE`
  and no `retry_after`. Before then it keeps its own reason: an authoritative
  absence seen too early is `NOT_YET_QUIESCENT`, with a `retry_after`.
- The held effect waits for `agent.recheck` (its lookup) or for an operator's
  answer through `agent.settle`: `Settle.committed`, `Settle.not_applied`,
  `Settle.resend` or `Settle.close_unknown`.
- A call whose effect had not started is asked again after a crash.
- The agent logs a warning when it is built, or at run start for tools a
  toolkit provides. It names the tools and the fix. With `can_use_tool` the
  build cannot tell which calls ask, so the warning names every effects tool
  that D3 or D4 can re-send and that no `deny` pattern refuses, and says it
  may ask.

```python
import os

from qagents import Permissions

permissions = Permissions(approval_secret=os.environ["APPROVAL_SECRET"])
```

Give every worker and every restart the same secret: a decision signed under
another secret does not verify either. The secret also signs paused approvals
and operator answers; see [signed pauses](../concepts/permissions.md#signed-pauses).

Two hold reasons concern the decision:

| Reason | Cause | Next step |
|---|---|---|
| `APPROVAL_EXPIRED` | The decision is older than `approval_ttl_s` (store clock) when a re-send would go out. | Recheck, or settle as an operator. |
| `APPROVAL_UNVERIFIABLE` | The decision does not verify: no `approval_secret`, a MAC that does not match, or an operator `resend` that does not verify. | Recheck, or settle as an operator. Set `approval_secret` for later calls. |

Both reasons appear only where a re-send (D3 or D4) would otherwise go out,
except an operator `resend` that does not verify, which is always
`APPROVAL_UNVERIFIABLE`. A held effect never asks the person again: the first
intent spent the decision.


## Run the offline crash harness

Build an agent with the model supplied by the harness and define its
destination through the crash environment. Record a cassette with
the same model requests and synthetic fixture data.

```bash
qagents crashtest my_fixture:build "Credit DEMO-1" --cassette fixture.json \
  --store sqlite --transport fake --faults kill_after,advance_boundaries \
  --json results.json --junit results.xml
```

The CLI exits with 1 for a violation, 2 for a usage error and 3 for
an inconclusive report. With strict mode it exits with 4 when the
coverage or provider evidence is incomplete. A strict run requires
exercise through the client transport and evidence for destination
assumptions. The in-process harness uses a virtual clock. The Gym
benchmark measures real worker termination on the local machine.

## Run with a fresh worker process

An importable factory can also run in a new interpreter on each leg.
Use SQLite for the journal and pass an offline script or cassette. The
parent owns the destination and terminates the worker before replying at
the selected boundary.

```console
qagents crashtest my_fixture:build "Credit DEMO-1" --cassette fixture.json \
  --store sqlite --isolation subprocess --transport client \
  --faults kill_before,kill_after,recovery_kill,commit_then_crash,commit_then_lose_response,advance_boundaries \
  --points all-raw --json subprocess-results.json
```

Subprocess factories and custom scenarios must be importable by module
name. Define request and receipt types and simulated client classes at
module scope so both processes can resolve them. `env.service(...)`
provides the worker with a proxy to the parent-owned fake. The worker
uses the supplied offline model and receives a minimal environment.
Use trusted fixture code, as worker processes still have the application's
filesystem and network access. Freeze-and-thaw faults use in-process
isolation because they require a surviving worker.

Run a worker against a simulated Stripe refund with no credentials.
The destination commits the refund and the harness terminates the
worker before it receives the receipt. A fresh worker replays the
same key and the destination records one application.

```console
python -m examples.subprocess_refund
```

The command writes `refund-recovery.json` and prints the selected point's
verdict. The report also shows any gaps in the broader coverage.

```python
--8<-- "examples/subprocess_refund.py"
```

## Test the recovery boundaries

Run the Effects process tests from the development checkout. The test
controller kills its own worker just before or after an SQLite journal
write, or after a destination application and before the worker receives
a response. Every recovery starts a new interpreter and checks the
journal prefix and the destination ledger. The in-process explorer
also tests a conflicting approval delivery and competing attempts to
run, resume, or settle while an effect is held.

```powershell
$env:QAGENTS_RUN_PROCESS_TESTS = "1"
python -m pytest -q -m process tests/process
$env:QAGENTS_RUN_MUTATIONS = "1"
python -m pytest -q -m mutants tests/effects/mutants
```

Each mutation runs in an isolated interpreter with an
unmodified baseline. The test requires an exercised fault and an
assertion that detects it. The suite covers runtime, store and report
checks, including whether a reference run applied an effect.

## Run the refund probes without credentials

The local sandbox runs a stateful refund endpoint on a random loopback
port. Its clock can jump to the simulated 24-hour boundary and its
fault controls can return cached rejections or HTTP 500 responses,
hold an in-flight request, and drop a response after recording a refund.

```bash
python -m pytest -q -m "not network" tests/probes
```

The recovery test reopens an SQLite journal after the response is lost,
finds the receipt through the simulated provider's list endpoint and
resumes the agent. It checks that the provider received one refund POST.

Set `QAGENTS_LOCAL_STRIPE_PROBE_DIR` to keep the JSON observations.
The evidence has scope `local_http_simulation`. Only the provider's
sandbox probe can write a passing provider mapping for
`crash_test(probes=...)`.

## Run the optional Stripe provider probes

Load `STRIPE_TEST_SECRET_KEY` from your sandbox's secret store and run
the probe from the development checkout.

```powershell
$env:QAGENTS_RUN_STRIPE_PROBES = "1"
$env:QAGENTS_STRIPE_PROBE_DIR = "work/stripe-probes"
python -m pytest -q -rs tests/probes/test_stripe_idempotency.py
```

The probe writes an evidence file and a mapping for
`crash_test(probes=...)`. The evidence file records the observed
retention interval in minutes. Set `QAGENTS_STRIPE_RETENTION_MINUTES`
to test a longer interval up to 1439 minutes. A partial run leaves the
provider assurance pending. For cached server errors, the probe requires
an actual HTTP 500 and a matching replay from Stripe.

See [Stripe API v1 idempotency](https://docs.stripe.com/api/idempotent_requests)
and [Stripe replay headers](https://docs.stripe.com/error-low-level#sending-idempotency-keys).
