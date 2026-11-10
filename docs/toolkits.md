# Ready-made tools

`qagents.toolkits` has factories that return ordinary tools. Give them to an agent like your own.

```python
from qagents import Agent, Permissions
from qagents.toolkits import current_time, file_tools, notes_tools, shell_tool, web_tools

agent = Agent(
    "qcloud:<model-id>",
    tools=[
        file_tools("./workspace", writable=True),
        web_tools(allow_hosts=["docs.example.com", "*.wikipedia.org"]),
        notes_tools("./memory"),
        current_time,
    ],
    permissions=Permissions(allow=["write_file"]),  # edits and other risky calls still ask
)
```

| Factory | Tools | Read-only | Asks for approval |
|---|---|---|---|
| `file_tools(root)` | `list_files`, `read_file`, `search_files` | yes | no |
| `file_tools(root, writable=True)` | adds `write_file`, `edit_file` | no | yes |
| `shell_tool(cwd)` | `run_command` | no | yes |
| `web_tools()` | `fetch_url` | yes | no |
| `notes_tools(directory)` | `list_notes`, `read_note`, `save_note`, `delete_note` | reads only | no |
| `current_time` | `current_time` | yes | no |

## Files

`file_tools(root, writable=False, max_read_chars=100_000, max_results=200)`

Every path the model gives is resolved, following symlinks, and must end up inside `root`.
Anything else is refused with a message that tells the model to use a relative path. Directory
listings and searches skip symlinks and the usual noise (`.git`, `node_modules`, `.venv`).

- `read_file` returns numbered lines and tells the model how to continue reading a long file.
- `search_files` takes a regular expression and an optional file-name pattern.
- `edit_file` replaces one exact piece of text, and refuses when the text is missing or appears
  more than once, so the model cannot silently change the wrong place.

## Shell

`shell_tool(cwd, timeout=60, max_output_chars=30_000, env=None)`

`run_command` runs a command line with the rights of the agent process and returns the exit code
and the combined output. A command that exceeds the timeout is stopped together with the
processes it started.

!!! danger "This is not a sandbox"
    The tool asks for approval by default. If you allow it by rule, or feed the agent untrusted
    input, run the agent inside a container or VM and let that be the boundary. Pass `env=` to
    keep the agent's own secrets out of the commands' environment.

## Web

`web_tools(allow_hosts=None, allow_private=False, max_bytes=1_000_000, max_chars=40_000, timeout=20, max_redirects=5)`

`fetch_url` fetches a page and returns its readable text (HTML is reduced to visible text). The
same checks apply to the first URL and to every redirect:

- only `http` and `https`;
- the host must match `allow_hosts` when you give one;
- the host must not resolve to a private, loopback or link-local address, which blocks requests
  to internal services and cloud metadata endpoints.

The address check resolves the name once and the HTTP client resolves it again when connecting,
so a hostile DNS server could answer differently. When URLs come from untrusted parties, also
restrict outbound traffic at the network level.

Fetched pages are data for the model to read. Pages can contain text written to look like
instructions; this is why tools with side effects ask for approval by default.

## Notes (memory)

`notes_tools(directory, max_note_chars=20_000)`

Each note is a Markdown file. The agent decides what is worth writing down and reads it back in a
later run. Memory stays inspectable: open the folder and you see exactly what the agent knows.
Use one directory per user or tenant.

There is no vector store. The published benchmarks for specialised memory layers are disputed,
and an agent with plain file tools is competitive with them; see the
[research report](research.md).

## Your own toolkit

A function that returns a list of tools is a toolkit. Close over what the tools need:

```python
from qagents import Tool, ToolError, tool


def crm_tools(client: "CrmClient") -> list[Tool]:
    @tool(read_only=True)
    async def find_customer(email: str) -> dict:
        """Find a customer by email address."""
        customer = await client.find(email)
        if customer is None:
            raise ToolError(f"No customer with email {email}.")
        return customer

    @tool(requires_approval=True)
    async def close_account(customer_id: str) -> str:
        """Close a customer's account. This cannot be undone."""
        await client.close(customer_id)
        return "Account closed."

    return [find_customer, close_account]
```
