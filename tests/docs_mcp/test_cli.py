"""The command-line fallback prints what the MCP tools return."""

from __future__ import annotations

import io
import re
import subprocess
import sys

from qagents.docs_mcp import cli
from qagents.docs_mcp.sources import SKILL_URL

from .conftest import ROOT


def run(*argv: str, stdin: str = "") -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(
        ["--docs-dir", str(ROOT), *argv], stdin=io.StringIO(stdin), stdout=out, stderr=err
    )
    return code, out.getvalue(), err.getvalue()


def test_search(service):
    code, out, _ = run("search", "pause", "and", "resume", "--limit", "3")
    assert code == 0
    assert out == cli.shell_hints(service.search("pause and resume", 3))
    assert "Read: python -m qagents.docs_mcp read concepts/permissions.md pause-and-resume" in out


def test_search_without_results_fails():
    code, out, _ = run("search", "zzzqqqxxx")
    assert code == 1 and out.startswith("No results")


def test_read(service):
    code, out, _ = run("read", "concepts/permissions.md", "pause-and-resume")
    assert code == 0 and out == service.read_page("concepts/permissions.md", "pause-and-resume")
    code, out, _ = run("read", "nope.md")
    assert code == 1 and out.startswith("There is no page")


def test_pages_api_examples_example_llms(service):
    timing = re.compile(r"built in \d+ ms")
    pages = run("pages")[1]
    assert timing.sub("", pages) == timing.sub("", cli.shell_hints(service.list_pages()))
    assert "python -m qagents.docs_mcp read <path> [section]" in pages
    code, out, _ = run("api", "Agent.resume")
    assert code == 0 and out.startswith("# qagents.Agent.resume (method)")
    code, out, _ = run("api", "Agnet")
    assert code == 1 and "has no public name" in out
    assert run("examples")[1] == cli.shell_hints(service.list_examples())
    assert run("example", "supervisor")[1] == service.read_example("supervisor")
    code, out, _ = run("llms")
    assert code == 0 and out.startswith("# QAgents")
    code, out, _ = run("llms", "--full")
    assert code == 0 and "<!-- page: concepts/permissions.md" in out


def test_check_files_and_stdin(tmp_path):
    good = tmp_path / "good.py"
    good.write_text("from qagents import Agent\nAgent('x', instructions='hi')\n", encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text("from qagents import Agent\nAgent('x', system_prompt='hi')\n", encoding="utf-8")
    code, out, _ = run("check", str(good))
    assert code == 0 and "no problems" in out
    code, out, _ = run("check", str(good), str(bad))
    assert code == 1 and f"<!-- {bad} -->" in out and "system_prompt" in out
    code, out, _ = run("check", "-", stdin="from qagents import Tools\n")
    assert code == 1 and "`qagents` has no `Tools`" in out
    code, _, err = run("check", str(tmp_path / "missing.py"))
    assert code == 1 and "cannot read" in err


SKILL = ROOT / "skills" / "qagents" / "SKILL.md"


def test_skill_prints_the_file_of_the_installed_version():
    code, out, _ = run("skill")
    assert code == 0 and out == SKILL.read_text(encoding="utf-8")
    assert out.startswith("---\nname: qagents\n")


def test_skill_writes_the_file_and_keeps_a_different_one(tmp_path):
    folder = tmp_path / ".claude" / "skills" / "qagents"
    written = folder / "SKILL.md"
    code, out, _ = run("skill", "--to", str(folder))
    assert code == 0 and "Wrote" in out and str(written) in out
    assert written.read_bytes() == SKILL.read_text(encoding="utf-8").encode("utf-8")  # LF, no BOM
    code, out, _ = run("skill", "--to", str(folder))
    assert code == 0 and "nothing to write" in out

    written.write_text("my own skill\n", encoding="utf-8")
    code, _, err = run("skill", "--to", str(folder))
    assert code == 1 and "--force" in err
    assert written.read_text(encoding="utf-8") == "my own skill\n"  # untouched
    code, _, _ = run("skill", "--to", str(folder), "--force")
    assert code == 0 and written.read_text(encoding="utf-8") == SKILL.read_text(encoding="utf-8")

    named = tmp_path / "elsewhere" / "SKILL.md"  # a path to the file itself works too
    assert run("skill", "--to", str(named))[0] == 0 and named.is_file()


def test_skill_says_where_to_get_it_when_the_install_has_none(fake_checkout):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["--docs-dir", str(fake_checkout), "skill"], stdout=out, stderr=err)
    assert code == 1 and out.getvalue() == ""
    assert SKILL_URL in err.getvalue() and "skills/qagents" in err.getvalue()


def test_bad_command_line():
    assert run("search")[0] == 2
    assert run("frobnicate")[0] == 2


def test_python_dash_m_runs_the_command_line():
    completed = subprocess.run(
        [sys.executable, "-m", "qagents.docs_mcp", "--docs-dir", str(ROOT), "api", "tool"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("# qagents.tool (function)")
