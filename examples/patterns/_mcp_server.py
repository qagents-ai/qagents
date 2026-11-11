"""The inventory MCP server of the ``mcp_tools`` example (not an example of its own: the leading
underscore keeps it out of ``examples.run_all``).

A small stock store that speaks the Model Context Protocol over stdin and stdout, built with
the official MCP Python SDK. It keeps the stock in a JSON file, so the example can check what
changed; the file is created with the fixture stock below when it does not exist.

    python examples/patterns/_mcp_server.py <stock.json>

Tools, with the annotations a client reads to decide what needs approval:

- ``list_items``: every item with its SKU, name and quantity (read-only).
- ``get_item``: one item (read-only).
- ``update_stock``: add or remove units (``destructiveHint``: it changes the stock).

The MCP SDK is imported only when the server is built, so importing this module needs nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

STOCK = {  # fixture data, written to the stock file when it does not exist
    "CB-01": {"name": "USB-C cable, 1 m", "quantity": 12},
    "MN-27": {"name": "27-inch monitor", "quantity": 4},
    "KB-10": {"name": "Wireless keyboard", "quantity": 9},
}


def build_server(path: Path) -> Any:
    """The MCP server, reading and writing the stock in ``path``."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp_types import ToolAnnotations

    read_only = ToolAnnotations(read_only_hint=True)
    destructive = ToolAnnotations(read_only_hint=False, destructive_hint=True)
    server = MCPServer("inventory", instructions="The stock of a small electronics shop.")

    def load() -> dict[str, dict[str, Any]]:
        return json.loads(path.read_text(encoding="utf-8"))

    def line(sku: str, item: dict[str, Any]) -> str:
        return f"{sku}: {item['name']}, {item['quantity']} in stock"

    def find(items: dict[str, dict[str, Any]], sku: str) -> dict[str, Any]:
        if sku not in items:
            raise ToolError(f"There is no item {sku!r}. The SKUs are {', '.join(items)}.")
        return items[sku]

    @server.tool(annotations=read_only)
    def list_items() -> str:
        """List every item in stock: its SKU, name and quantity."""
        return "\n".join(line(sku, item) for sku, item in load().items())

    @server.tool(annotations=read_only)
    def get_item(sku: str) -> str:
        """Show one item: its name and the quantity in stock. sku is the item's SKU, like CB-01."""
        return line(sku, find(load(), sku))

    @server.tool(annotations=destructive)
    def update_stock(sku: str, change: int) -> str:
        """Change the quantity in stock of one item. sku is the item's SKU, like CB-01; change
        is the number of units to add (positive, for a delivery) or remove (negative)."""
        items = load()
        item = find(items, sku)
        before, after = item["quantity"], item["quantity"] + change
        if after < 0:
            raise ToolError(f"{sku} has only {before} in stock; cannot remove {-change}.")
        item["quantity"] = after
        path.write_text(json.dumps(items, indent=2), encoding="utf-8")
        return f"{sku}: {before} -> {after} in stock"

    return server


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python examples/patterns/_mcp_server.py <stock.json>")
    path = Path(argv[0])
    if not path.exists():
        path.write_text(json.dumps(STOCK, indent=2), encoding="utf-8")
    build_server(path).run("stdio")


if __name__ == "__main__":
    main(sys.argv[1:])
