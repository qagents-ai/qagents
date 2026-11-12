"""A static index of the public qagents API, read from the package's source files.

It parses the source with `ast` and never imports a module, so building it is fast and does
not need the optional extras (``mcp``, ``mlflow``, ``psycopg``, ...). `qagents.docs_mcp.api`
uses live introspection on top of it when an agent asks about one name.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ApiKind = Literal["module", "class", "function", "method", "property", "attribute"]

SKIPPED_PACKAGES = ("docs_mcp",)
"""Subpackages left out of the API index: the docs server documents qagents, not itself."""


@dataclass(frozen=True)
class ApiEntry:
    """One documented public object of the installed package."""

    path: str
    """Where it is defined, for example ``qagents.agent.Agent.run``."""
    public: str
    """The shortest public name for it, for example ``qagents.Agent.run``."""
    kind: ApiKind
    signature: str
    """The parameters as written in the source, ``self`` left out; empty for modules and
    attributes."""
    doc: str
    file: str
    """The source file, relative to the checkout or to ``site-packages``."""
    line: int
    value: str = ""
    """For attributes, the source of the assigned value (shortened)."""
    bases: tuple[str, ...] = ()
    """For classes, the base classes as written in the source."""

    @property
    def name(self) -> str:
        return self.path.rsplit(".", 1)[-1]

    @property
    def summary(self) -> str:
        first = self.doc.strip().split("\n\n", 1)[0]
        return " ".join(first.split())


def build_api_index(package_dir: Path, display_root: Path) -> list[ApiEntry]:
    """Every public module, class, function, method, property and documented attribute.

    Args:
        package_dir: The installed ``qagents`` package directory.
        display_root: Paths in `ApiEntry.file` are relative to this directory.
    """
    aliases = _public_aliases(package_dir)
    entries: list[ApiEntry] = []
    for file, module in _modules(package_dir):
        try:
            source = file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file))
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
        try:
            shown = file.relative_to(display_root).as_posix()
        except ValueError:
            shown = file.as_posix()
        entries.extend(_module_entries(tree, module, shown, aliases))
    return entries


def _modules(package_dir: Path) -> Iterator[tuple[Path, str]]:
    for file in sorted(package_dir.rglob("*.py")):
        parts = file.relative_to(package_dir).with_suffix("").parts
        if parts and parts[0] in SKIPPED_PACKAGES:
            continue
        if any(part.startswith("_") and part != "__init__" for part in parts):
            continue
        names = ["qagents", *(part for part in parts if part != "__init__")]
        yield file, ".".join(names)


def _public_aliases(package_dir: Path) -> dict[str, str]:
    """Map each defining path to a shorter public path re-exported by a package.

    ``from .agent import Agent`` in ``qagents/__init__.py`` makes ``qagents.Agent`` the
    public name of ``qagents.agent.Agent``. Names in a package's ``_LAZY`` mapping count too.
    A package with a literal ``__all__`` re-exports only the imported names listed in it: the
    others are imports it uses itself. An alias counts only when it is shorter than the path
    that defines the object, so ``qagents.external.ExternalAgent`` stays the public name even
    though ``qagents.integrations.devvi`` imports it.
    """
    aliases: dict[str, str] = {}
    for init in sorted(package_dir.rglob("__init__.py")):
        parts = init.relative_to(package_dir).parent.parts
        if parts and parts[0] in SKIPPED_PACKAGES:
            continue
        if any(part.startswith("_") for part in parts):
            continue
        package = ".".join(["qagents", *parts])
        try:
            tree = ast.parse(init.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
        exported = _dunder_all(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level >= 1 and node.module:
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                for alias in node.names:
                    shown = alias.asname or alias.name
                    if alias.name == "*" or shown.startswith("_"):
                        continue
                    if exported is not None and shown not in exported:
                        continue
                    target = f"{base}.{node.module}.{alias.name}"
                    _keep_shortest(aliases, target, f"{package}.{shown}")
            elif isinstance(node, ast.Assign | ast.AnnAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == "_LAZY" for t in targets):
                    value = node.value
                    if isinstance(value, ast.Dict):
                        for key, module in zip(value.keys, value.values, strict=False):
                            if isinstance(key, ast.Constant) and isinstance(module, ast.Constant):
                                target = f"{module.value}.{key.value}"
                                _keep_shortest(aliases, target, f"{package}.{key.value}")
    return aliases


def _dunder_all(tree: ast.Module) -> set[str] | None:
    """The names in a module's literal ``__all__``; None when it has none or computes it."""
    names: set[str] = set()
    found = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign | ast.AugAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        if not isinstance(value, ast.List | ast.Tuple):
            return None  # built at import time: every import may be a re-export
        items = [item.value for item in value.elts if isinstance(item, ast.Constant)]
        if len(items) != len(value.elts) or not all(isinstance(i, str) for i in items):
            return None
        if not isinstance(node, ast.AugAssign):
            names = set()
        names.update(items)
        found = True
    return names if found else None


def _keep_shortest(aliases: dict[str, str], target: str, public: str) -> None:
    """Record ``public`` for ``target`` when it is shorter than both it and the alias so far."""
    if len(public) >= len(target):
        return
    current = aliases.get(target)
    if current is None or len(public) < len(current):
        aliases[target] = public


def _public_name(path: str, aliases: dict[str, str]) -> str:
    """The shortest public spelling of ``path``: the alias of its longest prefix, then the
    alias of that spelling, until none applies. Re-exports chain: a class of
    ``qagents.external.agent`` is re-exported by ``qagents.external``, whose name
    ``qagents`` re-exports in turn (``_LAZY``), so its public name is ``qagents.X``."""
    seen: set[str] = set()
    while path not in seen:  # every alias is shorter than what it names: this ends
        seen.add(path)
        parts = path.split(".")
        for cut in range(len(parts), 1, -1):
            prefix = ".".join(parts[:cut])
            if prefix in aliases:
                path = ".".join([aliases[prefix], *parts[cut:]])
                break
        else:
            return path
    return path


def _module_entries(
    tree: ast.Module, module: str, file: str, aliases: dict[str, str]
) -> Iterator[ApiEntry]:
    yield ApiEntry(
        module, _public_name(module, aliases), "module", "", ast.get_docstring(tree) or "",
        file, 1,
    )  # fmt: skip
    for node, doc in _with_attribute_docs(tree.body):
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            yield from _class_entries(node, f"{module}.{node.name}", file, aliases)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name.startswith("_"):
                continue
            path = f"{module}.{node.name}"
            yield ApiEntry(
                path, _public_name(path, aliases), "function", _signature(node),
                ast.get_docstring(node) or "", file, node.lineno,
            )  # fmt: skip
        elif doc is not None:
            for name in _assigned_names(node):
                if name.startswith("_"):
                    continue
                path = f"{module}.{name}"
                yield ApiEntry(
                    path, _public_name(path, aliases), "attribute", "", doc, file, node.lineno,
                    value=_value_source(node),
                )  # fmt: skip


def _class_entries(
    node: ast.ClassDef, path: str, file: str, aliases: dict[str, str]
) -> Iterator[ApiEntry]:
    init = next(
        (
            item
            for item in node.body
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name == "__init__"
        ),
        None,
    )
    yield ApiEntry(
        path, _public_name(path, aliases), "class", _signature(init) if init else "",
        ast.get_docstring(node) or "", file, node.lineno,
        bases=tuple(ast.unparse(base) for base in node.bases),
    )  # fmt: skip
    for item, doc in _with_attribute_docs(node.body):
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
            if item.name.startswith("_"):
                continue
            decorators = {ast.unparse(d) for d in item.decorator_list}
            if any(d.endswith(".setter") or d.endswith(".deleter") for d in decorators):
                continue
            kind: ApiKind = "property" if decorators & {"property", "cached_property"} else "method"
            member = f"{path}.{item.name}"
            yield ApiEntry(
                member, _public_name(member, aliases), kind,
                "" if kind == "property" else _signature(item),
                ast.get_docstring(item) or "", file, item.lineno,
                value=_return_source(item) if kind == "property" else "",
            )  # fmt: skip
        elif doc is not None or isinstance(item, ast.AnnAssign):
            for name in _assigned_names(item):
                if name.startswith("_") or name == "model_config":
                    continue
                member = f"{path}.{name}"
                yield ApiEntry(
                    member, _public_name(member, aliases), "attribute", "", doc or "", file,
                    item.lineno, value=_value_source(item),
                )  # fmt: skip


def _with_attribute_docs(body: list[ast.stmt]) -> Iterator[tuple[ast.stmt, str | None]]:
    """Each statement with the string literal right after it, the attribute docstring."""
    for index, node in enumerate(body):
        doc = None
        if isinstance(node, ast.Assign | ast.AnnAssign) and index + 1 < len(body):
            following = body[index + 1]
            if (
                isinstance(following, ast.Expr)
                and isinstance(following.value, ast.Constant)
                and isinstance(following.value.value, str)
            ):
                doc = inspect.cleandoc(following.value.value)
        yield node, doc


def _assigned_names(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    if isinstance(node, ast.Assign):
        return [target.id for target in node.targets if isinstance(target, ast.Name)]
    return []


def _value_source(node: ast.stmt, limit: int = 400) -> str:
    if isinstance(node, ast.AnnAssign):
        annotation = ast.unparse(node.annotation)
        if node.value is None:
            return f": {annotation}"
        text = f": {annotation} = {ast.unparse(node.value)}"
    elif isinstance(node, ast.Assign):
        text = f" = {ast.unparse(node.value)}"
    else:
        return ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _return_source(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return f": {ast.unparse(node.returns)}" if node.returns is not None else ""


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    if positional and positional[0].arg in ("self", "cls"):
        positional = positional[1:]
        skipped = 1
    else:
        skipped = 0
    defaults = [None] * (len(args.posonlyargs) + len(args.args) - len(args.defaults))
    defaults += list(args.defaults)
    defaults = defaults[skipped:]
    pieces: list[str] = []
    posonly = max(len(args.posonlyargs) - skipped, 0)
    for index, (arg, default) in enumerate(zip(positional, defaults, strict=False)):
        pieces.append(_param(arg, default))
        if posonly and index == posonly - 1:
            pieces.append("/")
    if args.vararg is not None:
        pieces.append("*" + _param(args.vararg, None))
    elif args.kwonlyargs:
        pieces.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=False):
        pieces.append(_param(arg, default))
    if args.kwarg is not None:
        pieces.append("**" + _param(args.kwarg, None))
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"({', '.join(pieces)}){returns}"


def _param(arg: ast.arg, default: ast.expr | None) -> str:
    text = arg.arg
    if arg.annotation is not None:
        text += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            text += f" = {ast.unparse(default)}"
    elif default is not None:
        text += f"={ast.unparse(default)}"
    return text
