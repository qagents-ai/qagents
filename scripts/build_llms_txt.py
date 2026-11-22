"""Write docs/llms.txt and docs/llms-full.txt (https://llmstxt.org) from the documentation.

    python scripts/build_llms_txt.py                      # into docs/, from this checkout
    python scripts/build_llms_txt.py --out site-extra/    # somewhere else
    python scripts/build_llms_txt.py --site-url https://example.org/qagents/
    python scripts/build_llms_txt.py --check              # exit 1 if docs/ is out of date

``llms.txt`` is the index: the project name, a one-line summary, then every page of the
``mkdocs.yml`` nav as a link with a one-line description, grouped by nav section.
``llms-full.txt`` is every page in nav order, with snippet includes and the API reference
(``::: qagents...`` directives) filled in from the installed package.

MkDocs copies files from ``docs/`` that are not Markdown into the built site unchanged, so
both files are published at the site root next to ``index.html``. Run this script after
changing the docs or the public API, before ``mkdocs build``. Links use ``site_url`` from
``mkdocs.yml`` when it is set (or ``--site-url``), and are relative otherwise.

It needs qagents importable (``pip install -e .``); it reads the same corpus as the docs MCP
server, ``python -m qagents.docs_mcp``. The output does not depend on the interpreter's hash
seed: the API reference shows values such as frozensets, whose order follows string hashes, so
the script runs itself again with ``PYTHONHASHSEED=0`` when the seed is not already fixed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASH_SEED = "0"


def build(root: Path, site_url: str | None = None) -> dict[str, str]:
    """The contents of ``llms.txt`` and ``llms-full.txt`` for the checkout at ``root``."""
    from qagents.docs_mcp.llms import build_llms_full, build_llms_txt
    from qagents.docs_mcp.service import DocsService
    from qagents.docs_mcp.sources import find_source

    source = find_source(root)
    if source.kind != "flag":
        raise SystemExit(f"{root} has no docs folder.")
    service = DocsService(source=source)
    return {
        "llms.txt": build_llms_txt(service.corpus, site_url=site_url),
        "llms-full.txt": build_llms_full(
            service.corpus, site_url=site_url, render_api=service._render_directive
        ),
    }


def main(argv: list[str] | None = None) -> int:
    if os.environ.get("PYTHONHASHSEED") != HASH_SEED:
        # Set and frozenset reprs follow the hash seed: build with a fixed one, or two runs
        # of an unchanged checkout write different files and --check never settles.
        args = sys.argv[1:] if argv is None else argv
        env = {**os.environ, "PYTHONHASHSEED": HASH_SEED, "PYTHONIOENCODING": "utf-8"}
        child = subprocess.run(
            [sys.executable, __file__, *args],
            env=env, capture_output=True, text=True, encoding="utf-8", check=False,
        )  # fmt: skip
        sys.stdout.write(child.stdout)  # through sys.stdout, so callers can capture it
        sys.stderr.write(child.stderr)
        return child.returncode
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="the qagents checkout")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: docs/)")
    parser.add_argument("--site-url", default=None, help="absolute base URL for the links")
    parser.add_argument(
        "--check", action="store_true", help="write nothing; exit 1 if the files differ"
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = (args.out or root / "docs").resolve()
    files = build(root, args.site_url)
    if args.check:
        stale = [
            name
            for name, text in files.items()
            if not (out / name).is_file() or (out / name).read_text(encoding="utf-8") != text
        ]
        for name in stale:
            print(f"{out / name} is out of date: run python scripts/build_llms_txt.py")
        return 1 if stale else 0
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {out / name} ({len(text.encode('utf-8')):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
