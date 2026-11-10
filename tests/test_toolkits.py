from __future__ import annotations

import os
import sys

import httpx
import pytest

from qagents import Agent, RunContext, ToolError
from qagents.models.testing import ScriptedModel, call
from qagents.toolkits import current_time, file_tools, notes_tools, shell_tool, web_tools
from qagents.toolkits.web import html_to_text


def by_name(tools):
    return {t.name: t for t in tools}


async def run(target, **arguments):
    return await target.run(RunContext(), arguments)


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text(
        "import os\n\ndef main():\n    return 42\n", encoding="utf-8"
    )
    (root / "README.md").write_text("# Demo\nA demo project.\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "junk.js").write_text("main", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    return root


async def test_file_tools_read_list_and_search(workspace):
    tools = by_name(file_tools(workspace))
    assert set(tools) == {"list_files", "read_file", "search_files"}  # no write tools by default
    assert all(t.read_only for t in tools.values())

    assert (await run(tools["list_files"])).splitlines() == ["README.md", "src/app.py"]
    assert await run(tools["list_files"], pattern="*.py") == "src/app.py"
    assert await run(tools["read_file"], path="src/app.py", start_line=3, max_lines=1) == (
        "3\tdef main():\n[1 more lines; continue with start_line=4]"
    )
    assert await run(tools["search_files"], pattern=r"def \w+") == "src/app.py:3: def main():"
    assert await run(tools["search_files"], pattern="main", glob="*.md") == "No matches."
    with pytest.raises(ToolError, match="Invalid regular expression"):
        await run(tools["search_files"], pattern="(")


@pytest.mark.parametrize("path", ["../secret.txt", "src/../../secret.txt", "{abs}"])
async def test_file_tools_never_leave_the_workspace(workspace, path):
    tools = by_name(file_tools(workspace, writable=True))
    target = path.replace("{abs}", str(workspace.parent / "secret.txt"))
    with pytest.raises(ToolError, match="outside the workspace"):
        await run(tools["read_file"], path=target)
    with pytest.raises(ToolError, match="outside the workspace"):
        await run(tools["write_file"], path=target, content="overwritten")
    with pytest.raises(ToolError, match="outside the workspace"):
        await run(tools["list_files"], path=os.path.dirname(target) or "..")
    assert (workspace.parent / "secret.txt").read_text(encoding="utf-8") == "TOP SECRET"


@pytest.mark.skipif(sys.platform == "win32", reason="creating symlinks needs privileges on Windows")
async def test_file_tools_do_not_follow_links_out_of_the_workspace(workspace):
    (workspace / "link.txt").symlink_to(workspace.parent / "secret.txt")
    tools = by_name(file_tools(workspace))
    with pytest.raises(ToolError, match="outside the workspace"):
        await run(tools["read_file"], path="link.txt")
    assert "TOP SECRET" not in await run(tools["search_files"], pattern="SECRET")


async def test_file_writes_need_approval_and_edits_must_be_unambiguous(workspace):
    tools = by_name(file_tools(workspace, writable=True))
    assert tools["write_file"].requires_approval and tools["edit_file"].requires_approval

    assert await run(tools["write_file"], path="docs/notes.txt", content="a b a") == (
        "Wrote 5 characters to docs/notes.txt."
    )
    with pytest.raises(ToolError, match="appears 2 times"):
        await run(tools["edit_file"], path="docs/notes.txt", old_text="a", new_text="c")
    with pytest.raises(ToolError, match="was not found"):
        await run(tools["edit_file"], path="docs/notes.txt", old_text="zzz", new_text="c")
    await run(tools["edit_file"], path="docs/notes.txt", old_text="b", new_text="B")
    assert (workspace / "docs" / "notes.txt").read_text(encoding="utf-8") == "a B a"

    # Through an agent, the write pauses for approval by default.
    model = ScriptedModel([call("write_file", path="x.txt", content="hi")])
    paused = await Agent(model, tools=[file_tools(workspace, writable=True)]).run("write it")
    assert paused.status == "paused" and not (workspace / "x.txt").exists()


async def test_notes_are_plain_files(tmp_path):
    tools = by_name(notes_tools(tmp_path / "memory"))
    assert await run(tools["list_notes"]) == "There are no notes yet."
    await run(tools["save_note"], name="customer-prefs", content="Prefers email.\nNever call.")
    assert (
        (tmp_path / "memory" / "customer-prefs.md")
        .read_text(encoding="utf-8")
        .startswith("Prefers")
    )
    assert await run(tools["list_notes"]) == "customer-prefs: Prefers email."
    assert await run(tools["read_note"], name="customer-prefs") == "Prefers email.\nNever call."
    for bad in ["../escape", "a/b", "", ".hidden"]:
        with pytest.raises(ToolError, match="note name"):
            await run(tools["save_note"], name=bad, content="x")
    await run(tools["delete_note"], name="customer-prefs")
    with pytest.raises(ToolError, match="no note named"):
        await run(tools["read_note"], name="customer-prefs")


async def test_shell_tool_reports_exit_code_and_output(tmp_path):
    command = shell_tool(tmp_path, timeout=20)
    assert command.requires_approval
    output = await run(command, command="echo hello-from-shell")
    assert output.startswith("exit code 0") and "hello-from-shell" in output
    failed = await run(command, command=f'"{sys.executable}" -c "import sys; sys.exit(3)"')
    assert failed.startswith("exit code 3")


async def test_shell_tool_stops_commands_that_run_too_long(tmp_path):
    command = shell_tool(tmp_path, timeout=0.5)
    with pytest.raises(ToolError, match="stopped after 0.5 seconds"):
        await run(command, command=f'"{sys.executable}" -c "import time; time.sleep(30)"')


def web(handler, **options):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    (fetch,) = web_tools(http_client=client, **options)
    return fetch


async def test_fetch_returns_readable_text():
    page = (
        "<html><head><title>T</title><style>p{color:red}</style><script>alert(1)</script></head>"
        "<body><h1>Opening hours</h1><p>Mon&ndash;Fri <b>9 to 5</b></p></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=page)

    text = await run(web(handler), url="http://93.184.216.34/hours")
    assert text.endswith("T\nOpening hours\nMon–Fri 9 to 5")
    assert "alert" not in text and "color:red" not in text
    assert html_to_text("<div>a</div><div>b <i>c</i></div>") == "a\nb c"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost:8080/",
        "http://10.0.0.5/internal",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "file:///etc/passwd",
        "ftp://example.com/file",
    ],
)
async def test_fetch_refuses_non_public_targets(url):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the request must not be sent")

    with pytest.raises(ToolError, match="non-public address|Only http"):
        await run(web(handler), url=url)


async def test_fetch_checks_every_redirect_and_the_allow_list():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "93.184.216.34":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})
        raise AssertionError("the redirect target must not be fetched")

    with pytest.raises(ToolError, match="non-public address"):
        await run(web(handler), url="http://93.184.216.34/start")

    def never(request: httpx.Request) -> httpx.Response:
        raise AssertionError("the request must not be sent")

    with pytest.raises(ToolError, match="not allowed here"):
        await run(web(never, allow_hosts=["docs.example.com"]), url="http://93.184.216.34/")

    def missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nothing here")

    with pytest.raises(ToolError, match="answered HTTP 404"):
        await run(web(missing, allow_private=True), url="http://intranet.test/page")


async def test_current_time():
    assert (await run(current_time)).endswith("UTC")
    with pytest.raises(ToolError, match="Unknown time zone"):
        await run(current_time, timezone="Mars/Olympus")
