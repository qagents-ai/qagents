# Devvi SDK fixtures

Copied byte for byte from Devvi's committed snapshots (`devvi-harness`, commit
`705975265c5be24ad6f41bf398a473237792ea5c`, 2026-11-06, Devvi 0.2.1-rc.3). MIT licensed:
see `../LICENSE`. Tests never read the Devvi repository.

| File | Source in devvi-harness |
|---|---|
| `text-turn.notifications.jsonl`, `.session.v4.jsonl`, `.result.json` | `snapshots/sdk/text-turn/notifications.expected.jsonl`, `session.v4.jsonl`, `result.expected.json` |
| `bash-tool.notifications.jsonl`, `.session.v4.jsonl`, `.result.json` | `snapshots/sdk/bash-tool/...` |
| `approval.session.v4.jsonl` | `snapshots/sdk/approval/session.v4.jsonl` |
| `session-resume.session.v4.jsonl` | `snapshots/sdk/session-resume/session.v4.jsonl` |
| `max-tokens-continue.session.v4.jsonl` | `snapshots/sdk/max-tokens-continue/session.v4.jsonl` |
| `subagent-spawn-in-process.notifications.jsonl`, `.session.v4.jsonl`, `.session.1.v4.jsonl`, `.result.json` | `snapshots/sdk/subagent-spawn-in-process/...` |

One derived file: `subagent-spawn-in-process.tokenized.notifications.jsonl`. The snapshot
normalises the parent and the child session ids to the same token (`{{sessionId}}`), so a
client cannot tell them apart. The derived file gives the child `{{session:2}}`: a
`session.event` belongs to the parent when its `seq` continues the parent's numbering
(child sessions start again at 0), and a `session.status` belongs to the session of the
event before it. Nothing else changed.

The snapshots replace volatile values with tokens (`{{session:1}}`, `{{message:1}}`,
`{{cwd}}`). `tests/integrations/fakes/fake_devvi_sdk.py` puts real values in their place
and plays the rest unchanged. A session log (`*.session.v4.jsonl`) plays one turn per
prompt; its `approval/asked` rows become real `approval/request` calls.
