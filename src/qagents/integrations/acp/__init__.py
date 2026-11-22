"""Drive any ACP v1 agent from qagents: `ACPAgent`, plus the MCP servers to hand it.

The Agent Client Protocol (https://agentclientprotocol.com) is how a program drives a
coding agent: start a session in a workspace, send a prompt, follow what it does, answer
its permission requests, cancel it. MCP is the other direction: it gives an agent tools.
So `ACPAgent` drives the agent over ACP, and `MCPServerStdio` / `MCPServerHTTP` hand it
extra tools (for example qagents tools served over MCP) for each session.

    from qagents import Agent, Permissions
    from qagents.integrations.acp import ACPAgent

    coder = ACPAgent(["node", "C:/tools/agent/bin.js", "--acp"], name="coder",
                     workspace="C:/work/repo", passthrough=["SOME_API_KEY"])
    lead = Agent(model, tools=[coder.as_tool(timeout=1800)],
                 permissions=Permissions(allow=["coder.read"],
                                         ask=["coder.edit", "coder.execute"]))
    async with coder:
        result = await lead.run("Make tests/test_parser.py pass.")

Verified against Devvi (``devvi --profile acp``, ACP v1) through its recorded
transcripts; other ACP agents are expected to work but have no recorded fixture yet.
The update-to-event and stop-reason tables are in `qagents.integrations.acp.mapping`.
"""

from __future__ import annotations

from .client import (
    CLIENT_CAPABILITIES,
    ACPAgent,
    MCPServer,
    MCPServerHTTP,
    MCPServers,
    MCPServerStdio,
)
from .mapping import (
    PROTOCOL_VERSION,
    STOP_REASONS,
    TITLES_ARE_TOOL_NAMES,
    TurnUpdates,
    default_tool_name,
    title_tool_name,
)

__all__ = [
    "CLIENT_CAPABILITIES",
    "PROTOCOL_VERSION",
    "STOP_REASONS",
    "TITLES_ARE_TOOL_NAMES",
    "ACPAgent",
    "MCPServer",
    "MCPServerHTTP",
    "MCPServerStdio",
    "MCPServers",
    "TurnUpdates",
    "default_tool_name",
    "title_tool_name",
]
