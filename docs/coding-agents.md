# Docs for coding agents

Coding agents (Claude Code, Codex CLI, Cursor, VS Code, any MCP client) write better qagents
code when they read the documentation of the version you have installed instead of guessing
from other frameworks. The installed documentation is available through these interfaces:

| Way | For | Needs |
|---|---|---|
| The **docs MCP server**, `python -m qagents.docs_mcp` | agents that speak MCP | The `mcp` installation extra |
| The **command line**, `python -m qagents.docs_mcp search "..."` | agents with a shell but no MCP | qagents |
| **`llms.txt`** and **`llms-full.txt`** on the docs site | chat assistants and web tools | a browser |
| **`AGENTS.md`** and the **Claude Code skill** | rules the agent reads up front | a copy in your project (`python -m qagents.docs_mcp skill`) |

The server and the command line answer from the installed package: the API comes from the code
that will run, and the guides from the docs that ship with it. When a guide and the API
disagree, the API answer wins.

## What the server offers

Tools, all read-only:

| Tool | What it returns |
|---|---|
| `search_docs(query, limit)` | Ranked sections of the guides, the README, the examples and the API docstrings, each with a snippet and its source (`page.md#anchor` or `file:line`) |
| `read_page(path, section)` | A whole page or one section, with the API reference filled in from the installed package |
| `list_pages()` | The table of contents with a one-line summary per page, the version, and where the docs were read from |
| `get_api(symbol)` | Signature, docstring, members and source location of any public name, such as `Agent.run` or `qagents.toolkits.files` |
| `list_examples()`, `read_example(name)` | The examples gallery and the code of one example |
| `check_code(code)` | A static check of a snippet: imports, attributes and keyword arguments against the installed signatures, and coroutines that are not awaited, each with the closest real name |

Resources: `qagents://llms.txt`, `qagents://docs/{path}` (every page, and also `README.md`,
`AGENTS.md` and the skill, `skills/qagents/SKILL.md`) and `qagents://examples/{name}`. Prompts: `build_agent` (build a new agent the qagents way) and
`review_code` (review code against the installed API and the qagents principles. It includes
the `check_code` report).

From your authorized QAgents checkout, install the MCP extra and check that the
server can find the documentation. QAgents is not published on PyPI:

```bash
python -m pip install -e ".[mcp]"
python -m qagents.docs_mcp pages
```

## Claude Code

```bash
claude mcp add qagents-docs -- python -m qagents.docs_mcp
```

Everything after `--` is the command Claude Code starts. Use the interpreter of the environment
where qagents is installed. On Windows with a virtual environment:

```powershell
claude mcp add qagents-docs -- C:\path\to\project\.venv\Scripts\python.exe -m qagents.docs_mcp
```

`--scope project` (`-s project`) writes the server to `.mcp.json` in the project, so everyone
who clones it gets it. The default scope, `local`, is only for you in this project. The
project file looks like this:

```json
{
  "mcpServers": {
    "qagents-docs": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "qagents.docs_mcp"]
    }
  }
}
```

**The skill.** `skills/qagents/SKILL.md` tells Claude Code when to use the server and gives it a
short cheat sheet. Write the copy of your installed version into your project:

```bash
python -m qagents.docs_mcp skill --to .claude/skills/qagents
```

Use `--to ~/.claude/skills/qagents` instead for all your projects. The command writes
`SKILL.md` into that folder as UTF-8, and does not replace a different `SKILL.md` there unless
you add `--force`. `python -m qagents.docs_mcp skill` prints the file instead (on Windows
PowerShell 5.1, `>` would save it as UTF-16, so prefer `--to`). Agents connected to the server
read it with `read_page("skills/qagents/SKILL.md")`. If your install has no copy, the command
says so. The skill is also in the qagents repository, in the
[`skills/qagents`](https://github.com/qagents-ai/qagents/tree/main/skills/qagents) folder.
Claude Code loads it whenever code imports qagents or you ask for a qagents agent.

## Codex CLI

```bash
codex mcp add qagents-docs -- python -m qagents.docs_mcp
```

Or add it to `~/.codex/config.toml` yourself:

```toml
[mcp_servers.qagents-docs]
command = "python"
args = ["-m", "qagents.docs_mcp"]
```

`codex mcp list` shows the configured servers. Codex reads the `AGENTS.md` of the repository
it works in, so in your own project it does not see the one in qagents. Give it the rules as
described in [AGENTS.md](#agentsmd) below.

## Cursor

Add the server to `.cursor/mcp.json` in your project (or `~/.cursor/mcp.json` for every
project):

```json
{
  "mcpServers": {
    "qagents-docs": {
      "command": "python",
      "args": ["-m", "qagents.docs_mcp"]
    }
  }
}
```

The Cursor command line can also add it to your user profile:

```bash
cursor --add-mcp '{"name":"qagents-docs","command":"python","args":["-m","qagents.docs_mcp"]}'
```

## VS Code

GitHub Copilot's agent mode reads `.vscode/mcp.json` in the workspace. Note the top-level key
is `servers`:

```json
{
  "servers": {
    "qagents-docs": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "qagents.docs_mcp"]
    }
  }
}
```

Or add it to your user profile from the command line:

```bash
code --add-mcp '{"name":"qagents-docs","command":"python","args":["-m","qagents.docs_mcp"]}'
```

## Any MCP client

The server speaks MCP over stdio. Configure the command `python` with the arguments
`-m qagents.docs_mcp`, using the Python of the environment where qagents is installed. It needs
no network access, no API key and no environment variables.

## Without MCP: the command line

Every tool has a command that prints the same Markdown:

```bash
python -m qagents.docs_mcp search "pause a tool call for approval"
python -m qagents.docs_mcp read concepts/permissions.md pause-and-resume
python -m qagents.docs_mcp pages
python -m qagents.docs_mcp api Agent.resume
python -m qagents.docs_mcp examples
python -m qagents.docs_mcp example supervisor
python -m qagents.docs_mcp check my_agent.py
python -m qagents.docs_mcp llms
python -m qagents.docs_mcp skill --to .claude/skills/qagents
```

`check` exits with status 1 when it finds problems, so it also works as a pre-commit step.
`qagents docs ...` (or `python -m qagents docs ...`) is the same command line under a shorter
name: `qagents docs search "pause a tool call for approval"`.
`search` takes `--limit N` and `--kind docs|example|api`.

## llms.txt

The published site serves [`llms.txt`](llms.txt), an index of every page with a one-line
description in the [llms.txt](https://llmstxt.org) format, and `llms-full.txt`, every page in
one file. Give either to a chat assistant that cannot run the server. They are written by
`scripts/build_llms_txt.py` from the same corpus the server reads, and MkDocs copies them from
`docs/` into the site.

## AGENTS.md

`AGENTS.md` at the root of the qagents repository has two parts. Part 1 has rules and patterns
for agents that write code with qagents (look names up, never invent them, mark side effects,
test with `ScriptedModel`). Part 2 has the layout, commands and principles for agents that work
on qagents itself.

Codex, Cursor and other tools read the `AGENTS.md` at the root of the repository they work in.
Inside a qagents checkout that is this file. In your own project that depends on qagents,
nothing reads it for you. Do one of these:

- Copy Part 1 of qagents' `AGENTS.md` into your project's `AGENTS.md` (create the file at your
  repository root if you have none). Codex and Cursor then read the rules on their own.
- Or tell the agent to read it from the docs server: `read_page("AGENTS.md")`, or
  `python -m qagents.docs_mcp read AGENTS.md` on the command line. The server has the copy of
  the installed version.

## Where the docs come from

The server uses the first of these that has a `docs` folder, and `list_pages()` says which:

1. `--docs-dir DIR`, when given: `python -m qagents.docs_mcp --docs-dir DIR`.
2. The copy bundled in the installed wheel.
3. The repository checkout, for an editable install (`pip install -e .`).
4. The `QAGENTS_DOCS_DIR` environment variable: a qagents checkout or its `docs` folder.

The API reference always comes from the installed package, so `get_api` and `check_code` are
right even when the guides come from another version (`list_pages()` then warns).

## Tool access and agent control

MCP is how you give an agent tools and context, such as these docs: the agent is the client
and decides when to call the server.

MCP is a tool protocol. An agent that serves itself as an MCP tool (Codex:
`codex mcp-server`) can be called through `MCPToolkit`, but each turn is then one opaque call:
qagents cannot follow its steps, decide its permission requests one by one, or cancel it and
keep the session. To make a coding agent work for a qagents agent, drive it over ACP or the
Devvi SDK.

ACP is the Agent Client Protocol, which many coding agents speak. The Devvi SDK
is Devvi's own protocol. See [External agents](concepts/external-agents.md). The two directions
combine: a qagents agent drives a coding agent over ACP, and that coding agent reads these
docs over MCP.

## Troubleshooting

| Symptom | Fix |
|---|---|
| The client cannot start the server | Use the full path of the environment's Python, for example `.venv\Scripts\python.exe` on Windows |
| `needs the MCP Python SDK 2.x` | Run `python -m pip install -e ".[mcp]"` from the QAgents checkout. The command line works without it |
| `list_pages()` says "no docs folder" | Point `QAGENTS_DOCS_DIR` at a qagents checkout, or use `--docs-dir` |
| Answers mention another version | The guides come from another checkout. `get_api` still reads the installed code |
