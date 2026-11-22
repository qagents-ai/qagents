"""The docs MCP server, on the official MCP Python SDK (the ``qagents[mcp]`` extra).

Run it over stdio with ``python -m qagents.docs_mcp``. It has seven read-only tools
(``search_docs``, ``read_page``, ``list_pages``, ``get_api``, ``list_examples``,
``read_example``, ``check_code``), the resources ``qagents://llms.txt``,
``qagents://docs/{path}`` and ``qagents://examples/{name}``, and two prompts
(``build_agent`` and ``review_code``). Every answer comes from the installed qagents.
"""

# No ``from __future__ import annotations``: the SDK reads the tool signatures at run time,
# and the Annotated[...] descriptions must be real objects.

from typing import Annotated, Any

from pydantic import Field

from . import prompts
from .service import DocsService

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ResourceNotFoundError
    from mcp_types import ToolAnnotations
except ImportError as error:  # pragma: no cover - depends on the environment
    raise ImportError(
        "The qagents docs MCP server needs the MCP Python SDK 2.x: pip install 'qagents[mcp]'. "
        'Without it, use the command line: python -m qagents.docs_mcp search "<query>". '
        f"Import failed with: {error}"
    ) from error

READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)


def build_server(service: DocsService | None = None) -> Any:
    """The MCP server with every tool, resource and prompt registered.

    Args:
        service: The docs to serve; by default those of the installed qagents.

    Returns:
        An ``mcp.server.mcpserver.MCPServer``. Call ``.run("stdio")`` to serve it.
    """
    docs = service or DocsService()
    version = docs.version
    server = MCPServer(
        prompts.DOCS_MCP_NAME,
        title="qagents documentation",
        instructions=prompts.DOCS_MCP_INSTRUCTIONS.format(version=version),
        version=version,
    )

    @server.tool(
        name="search_docs",
        title="Search the qagents docs",
        description=prompts.SEARCH_DOCS_DESCRIPTION,
        annotations=READ_ONLY,
        structured_output=False,
    )
    def search_docs(
        query: Annotated[str, Field(description=prompts.SEARCH_QUERY_DESCRIPTION)],
        limit: Annotated[int, Field(description=prompts.SEARCH_LIMIT_DESCRIPTION, ge=1, le=25)] = 8,
        kind: Annotated[str, Field(description=prompts.SEARCH_KIND_DESCRIPTION)] = "",
    ) -> str:
        return docs.search(query, limit, kind or None)

    @server.tool(
        name="read_page",
        title="Read a qagents docs page",
        description=prompts.READ_PAGE_DESCRIPTION,
        annotations=READ_ONLY,
        structured_output=False,
    )
    def read_page(
        path: Annotated[str, Field(description=prompts.READ_PAGE_PATH_DESCRIPTION)],
        section: Annotated[
            str | None, Field(description=prompts.READ_PAGE_SECTION_DESCRIPTION)
        ] = None,
    ) -> str:
        return docs.read_page(path, section or None)

    @server.tool(
        name="list_pages",
        title="List the qagents docs pages",
        description=prompts.LIST_PAGES_DESCRIPTION,
        annotations=READ_ONLY,
        structured_output=False,
    )
    def list_pages() -> str:
        return docs.list_pages()

    @server.tool(
        name="get_api",
        title="Look up a qagents name",
        description=prompts.GET_API_DESCRIPTION,
        annotations=READ_ONLY,
        structured_output=False,
    )
    def get_api(
        symbol: Annotated[str, Field(description=prompts.GET_API_SYMBOL_DESCRIPTION)],
    ) -> str:
        return docs.get_api(symbol)

    @server.tool(
        name="list_examples",
        title="List the qagents examples",
        description=prompts.LIST_EXAMPLES_DESCRIPTION,
        annotations=READ_ONLY,
        structured_output=False,
    )
    def list_examples() -> str:
        return docs.list_examples()

    @server.tool(
        name="read_example",
        title="Read a qagents example",
        description=prompts.READ_EXAMPLE_DESCRIPTION,
        annotations=READ_ONLY,
        structured_output=False,
    )
    def read_example(
        name: Annotated[str, Field(description=prompts.READ_EXAMPLE_NAME_DESCRIPTION)],
    ) -> str:
        return docs.read_example(name)

    @server.tool(
        name="check_code",
        title="Check code against the installed qagents",
        description=prompts.CHECK_CODE_DESCRIPTION,
        annotations=READ_ONLY,
        structured_output=False,
    )
    def check_code(
        code: Annotated[str, Field(description=prompts.CHECK_CODE_CODE_DESCRIPTION)],
    ) -> str:
        return docs.check_code(code)

    # -- resources -----------------------------------------------------------------

    @server.resource(
        "qagents://llms.txt",
        name="llms.txt",
        title="qagents llms.txt",
        description=prompts.LLMS_TXT_RESOURCE_DESCRIPTION,
        mime_type="text/markdown",
    )
    def llms_txt() -> str:
        return docs.llms_txt()

    for page in docs.corpus.ordered_pages():
        _add_page_resource(server, docs, page.path, page.display_title, page.summary)

    @server.resource(
        "qagents://docs/{+path}",
        name="docs-page",
        title="qagents docs page",
        description=prompts.DOC_RESOURCE_DESCRIPTION,
        mime_type="text/markdown",
    )
    def doc_page(path: str) -> str:
        text = docs.page_markdown(path)
        if text is None:
            raise ResourceNotFoundError(prompts.NOT_FOUND_PAGE.format(path=path))
        return text

    @server.resource(
        "qagents://examples/{+name}",
        name="example",
        title="qagents example",
        description=prompts.EXAMPLE_RESOURCE_DESCRIPTION,
        mime_type="text/x-python",
    )
    def example(name: str) -> str:
        code = docs.example_source(name)
        if code is None:
            raise ResourceNotFoundError(prompts.NOT_FOUND_EXAMPLE.format(name=name))
        return code

    # -- prompts -------------------------------------------------------------------

    @server.prompt(
        name="build_agent",
        title="Build a qagents agent",
        description=prompts.BUILD_AGENT_PROMPT_DESCRIPTION,
    )
    def build_agent(
        task: Annotated[str, Field(description=prompts.BUILD_AGENT_TASK_DESCRIPTION)],
    ) -> str:
        return prompts.BUILD_AGENT_PROMPT.format(version=version, task=task)

    @server.prompt(
        name="review_code",
        title="Review qagents code",
        description=prompts.REVIEW_PROMPT_DESCRIPTION,
    )
    def review_code(
        code: Annotated[str, Field(description=prompts.REVIEW_CODE_DESCRIPTION)],
    ) -> str:
        report = docs.check_code(code)
        return prompts.REVIEW_PROMPT.format(version=version, check=report.strip(), code=code)

    return server


def _add_page_resource(server: Any, docs: DocsService, path: str, title: str, summary: str) -> None:
    def read() -> str:
        text = docs.page_markdown(path)
        if text is None:  # pragma: no cover - the page was listed a moment ago
            raise ResourceNotFoundError(prompts.NOT_FOUND_PAGE.format(path=path))
        return text

    server.resource(
        f"qagents://docs/{path}",
        name=path,
        title=title,
        description=summary or prompts.DOC_RESOURCE_DESCRIPTION,
        mime_type="text/markdown",
    )(read)


def serve_stdio(service: DocsService | None = None) -> None:
    """Serve the docs over stdio until the client closes the connection."""
    build_server(service).run("stdio")
