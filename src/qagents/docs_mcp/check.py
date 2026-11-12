"""A static check of Python code against the installed qagents: what ``check_code`` answers.

It never runs the code. It compiles it, then follows the names imported from qagents and the
variables assigned from them:

* every ``import`` and ``from ... import`` of qagents exists;
* every attribute read from a qagents module, class or instance exists;
* every call of a qagents callable passes keyword and positional arguments its signature
  accepts (``**kwargs`` that a function passes on, as ``Agent.run_sync`` does to
  ``Agent.run``, are checked against the receiving function);
* coroutine functions such as ``Agent.run`` are awaited, and ``run_sync`` is not called
  inside ``async def``.

Types are followed through constructors, return annotations, annotated parameters and
assignments, pydantic fields and properties, attributes annotated where a method assigns them
(``self.result: RunResult | None = None``), ``for`` loops over typed lists and ``with``
blocks. A qagents pydantic model with its own ``__init__`` is checked against that
``__init__``, not against pydantic's list of fields. Anything it cannot follow is left alone
rather than guessed, so a report with no problems means "nothing wrong that a static check
can see".
"""

from __future__ import annotations

import ast
import difflib
import importlib
import inspect
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from . import prompts
from .api import (
    EXTRAS,
    ApiLookup,
    call_signature,
    element_class,
    forwarded_parameters,
    resolve_annotation,
    return_type,
    safe_to_import,
    single_class,
)

_MISSING = object()


@dataclass(frozen=True)
class Problem:
    line: int
    column: int
    message: str
    severity: str = "error"
    """``"error"`` for code that will fail, ``"warning"`` for code that probably does."""


@dataclass
class Report:
    problems: list[Problem] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "error"]

    @property
    def warnings(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "warning"]

    def render(self, version: str) -> str:
        errors, warnings = self.errors, self.warnings
        if errors:
            title = f"# check_code: {len(errors)} problem(s)"
        elif warnings:
            title = f"# check_code: {len(warnings)} warning(s)"
        else:
            title = "# check_code: no problems"
        lines = [f"{title} (qagents {version})", ""]
        for heading, items in (("Problems", errors), ("Warnings", warnings)):
            if not items:
                continue
            if errors and warnings:
                lines += [f"## {heading}", ""]
            for number, problem in enumerate(sorted(items, key=_position), start=1):
                lines.append(f"{number}. line {problem.line}: {problem.message}")
            lines.append("")
        if not errors and not warnings:
            lines += [prompts.CHECK_OK, ""]
        checked = self.counts["imports"] + self.counts["calls"] + self.counts["attributes"]
        if checked or self.counts["names"]:
            lines.append(
                prompts.CHECK_SUMMARY.format(
                    version=version,
                    imports=self.counts["imports"],
                    calls=self.counts["calls"],
                    attributes=self.counts["attributes"],
                )
            )
        else:
            lines.append(prompts.CHECK_NOTHING)
        lines.append(prompts.CHECK_LIMITS)
        return "\n".join(lines) + "\n"


def _position(problem: Problem) -> tuple[int, int]:
    return problem.line, problem.column


@dataclass(frozen=True)
class Value:
    """What the checker knows about an expression."""

    kind: str
    """``module``, ``object`` (a class or function itself), ``instance`` or ``method``."""
    obj: Any
    owner: type | None = None
    """For ``method``: the class of the instance it was read from."""
    item: type | None = None
    """For containers: the class of their items."""
    label: str = ""
    """How messages name it, for example ``qagents.Agent`` or ``agent``."""


class Checker(ast.NodeVisitor):
    """Walks a module, keeping one namespace per function, and records problems."""

    def __init__(self, lookup: ApiLookup) -> None:
        self.lookup = lookup
        self.report = Report()
        self._awaited: set[int] = set()
        self._done: dict[int, Value | None] = {}
        self._async_depth = 0

    # -- entry -----------------------------------------------------------------------

    def check(self, code: str) -> Report:
        source = textwrap.dedent(code)
        try:
            tree = compile(
                source, "<snippet>", "exec",
                flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )  # fmt: skip
        except SyntaxError as error:
            message = prompts.CHECK_SYNTAX.format(message=error.msg)
            if error.text:
                message += f" `{error.text.strip()}`"
            self.report.problems.append(Problem(error.lineno or 1, error.offset or 0, message))
            return self.report
        self.block(tree.body, {})
        return self.report

    # -- statements ------------------------------------------------------------------

    def block(self, body: list[ast.stmt], scope: dict[str, Value]) -> None:
        for statement in body:
            self.statement(statement, scope)

    def statement(self, node: ast.stmt, scope: dict[str, Value]) -> None:
        if isinstance(node, ast.Import):
            self.import_(node, scope)
        elif isinstance(node, ast.ImportFrom):
            self.import_from(node, scope)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            self.function(node, scope)
        elif isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                self.expr(decorator, scope)
            for base in node.bases:
                self.expr(base, scope)
            for keyword in node.keywords:
                self.expr(keyword.value, scope)
            self.block(node.body, dict(scope))
            scope.pop(node.name, None)
        elif isinstance(node, ast.Assign):
            value = self.expr(node.value, scope)
            self.not_awaited(node.value, scope)
            for target in node.targets:
                self.bind(target, value, scope)
        elif isinstance(node, ast.AnnAssign):
            value = self.expr(node.value, scope) if node.value is not None else None
            if node.value is not None:
                self.not_awaited(node.value, scope)
            annotated = self.annotation_value(node.annotation, scope)
            self.bind(node.target, annotated or value, scope)
        elif isinstance(node, ast.AugAssign):
            self.expr(node.value, scope)
            self.expr(node.target, scope)
        elif isinstance(node, ast.Expr):
            self.expr(node.value, scope)
            self.not_awaited(node.value, scope)
        elif isinstance(node, ast.For | ast.AsyncFor):
            iterated = self.expr(node.iter, scope)
            item = Value("instance", iterated.item) if iterated and iterated.item else None
            self.bind(node.target, item, scope)
            self.block(node.body, scope)
            self.block(node.orelse, scope)
        elif isinstance(node, ast.With | ast.AsyncWith):
            for item in node.items:
                value = self.expr(item.context_expr, scope)
                if item.optional_vars is not None:
                    self.bind(item.optional_vars, self.entered(value, node), scope)
            self.block(node.body, scope)
        elif isinstance(node, ast.If | ast.While):
            self.expr(node.test, scope)
            self.block(node.body, scope)
            self.block(node.orelse, scope)
        elif isinstance(node, ast.Try | ast.TryStar):
            self.block(node.body, scope)
            for handler in node.handlers:
                if handler.type is not None:
                    self.expr(handler.type, scope)
                if handler.name:
                    scope.pop(handler.name, None)
                self.block(handler.body, scope)
            self.block(node.orelse, scope)
            self.block(node.finalbody, scope)
        elif isinstance(node, ast.Match):
            self.expr(node.subject, scope)
            for case in node.cases:
                if case.guard is not None:
                    self.expr(case.guard, scope)
                self.block(case.body, scope)
        else:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self.expr(child, scope)
                elif isinstance(child, ast.stmt):
                    self.statement(child, scope)

    def function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, scope: dict[str, Value]
    ) -> None:
        decorated: Value | None = None
        for decorator in node.decorator_list:
            value = self.expr(decorator, scope)
            if self._is_tool_decorator(value):
                from qagents.tools import Tool

                decorated = Value("instance", Tool, label=node.name)
        arguments = node.args
        for default in [*arguments.defaults, *arguments.kw_defaults]:
            if default is not None:
                self.expr(default, scope)
        inner = dict(scope)
        for argument in [
            *arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
            *(a for a in (arguments.vararg, arguments.kwarg) if a is not None),
        ]:  # fmt: skip
            inner.pop(argument.arg, None)
            if argument.annotation is not None:
                annotated = self.annotation_value(argument.annotation, scope)
                if annotated is not None and argument not in (arguments.vararg, arguments.kwarg):
                    inner[argument.arg] = annotated
        if node.returns is not None:
            self.expr(node.returns, scope)
        is_async = isinstance(node, ast.AsyncFunctionDef)
        self._async_depth += is_async
        try:
            self.block(node.body, inner)
        finally:
            self._async_depth -= is_async
        if decorated is not None:
            scope[node.name] = decorated
        else:
            scope.pop(node.name, None)

    def _is_tool_decorator(self, value: Value | None) -> bool:
        if value is None:
            return False
        try:
            from qagents.tools import tool
        except ImportError:  # pragma: no cover
            return False
        if value.kind == "object" and value.obj is tool:
            return True
        return value.kind == "instance" and value.obj is _ToolDecorator

    def entered(self, value: Value | None, node: ast.With | ast.AsyncWith) -> Value | None:
        """What ``with value as name`` binds: the ``__aenter__``/``__enter__`` result."""
        if value is None or value.kind != "instance" or not inspect.isclass(value.obj):
            return None
        method = "__aenter__" if isinstance(node, ast.AsyncWith) else "__enter__"
        function = getattr(value.obj, method, None)
        if function is None:
            return None
        returned = return_type(function)
        return Value("instance", returned or value.obj, label=value.label)

    def bind(self, target: ast.expr, value: Value | None, scope: dict[str, Value]) -> None:
        if isinstance(target, ast.Name):
            if value is None:
                scope.pop(target.id, None)
            else:
                scope[target.id] = Value(
                    value.kind, value.obj, value.owner, value.item, value.label or target.id
                )
        elif isinstance(target, ast.Tuple | ast.List):
            for element in target.elts:
                self.bind(element, None, scope)
        elif isinstance(target, ast.Starred):
            self.bind(target.value, None, scope)
        else:
            self.expr(target, scope)

    def annotation_value(self, annotation: ast.expr, scope: dict[str, Value]) -> Value | None:
        """An instance of the class an annotation names, for ``x: Agent`` or ``ctx: RunContext``."""
        node = annotation
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                node = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                return None
        else:
            self.expr(annotation, scope)
        if isinstance(node, ast.Subscript):
            node = node.value
        value = self._peek(node, scope)
        if value is not None and value.kind == "object" and inspect.isclass(value.obj):
            return Value("instance", value.obj)
        return None

    def _peek(self, node: ast.expr, scope: dict[str, Value]) -> Value | None:
        """Resolve a name or dotted name without recording anything."""
        if isinstance(node, ast.Name):
            return scope.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self._peek(node.value, scope)
            if base is None or base.kind not in ("module", "object"):
                return None
            value = _safe_getattr(base.obj, node.attr)
            if value is _MISSING:
                return None
            return Value("module" if inspect.ismodule(value) else "object", value)
        return None

    # -- imports ---------------------------------------------------------------------

    def import_(self, node: ast.Import, scope: dict[str, Value]) -> None:
        for alias in node.names:
            if alias.name != "qagents" and not alias.name.startswith("qagents."):
                scope.pop((alias.asname or alias.name).split(".")[0], None)
                continue
            self.report.counts["imports"] += 1
            module = self.import_module(alias.name, node)
            if module is None:
                continue
            if alias.asname:
                scope[alias.asname] = Value("module", module, label=alias.asname)
            else:
                root = importlib.import_module("qagents")
                scope["qagents"] = Value("module", root, label="qagents")

    def import_from(self, node: ast.ImportFrom, scope: dict[str, Value]) -> None:
        name = node.module or ""
        if node.level or (name != "qagents" and not name.startswith("qagents.")):
            for alias in node.names:
                scope.pop(alias.asname or alias.name, None)
            return
        self.report.counts["imports"] += 1
        module = self.import_module(name, node, imported=[a.name for a in node.names])
        if module is None:
            for alias in node.names:
                scope.pop(alias.asname or alias.name, None)
            return
        for alias in node.names:
            if alias.name == "*":
                continue
            self.report.counts["names"] += 1
            value = self.module_attribute(module, alias.name, node)
            if value is None:
                scope.pop(alias.asname or alias.name, None)
            else:
                scope[alias.asname or alias.name] = Value(
                    value.kind, value.obj, label=alias.asname or alias.name
                )

    def import_module(self, name: str, node: ast.stmt, imported: list[str] | None = None) -> Any:
        if not safe_to_import(name):
            return None  # never run a __main__ module; nothing to follow in it
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name and (name == error.name or name.startswith(f"{error.name}.")):
                message = prompts.CHECK_NO_MODULE.format(module=name, version=_version())
                message += self._module_suggestions(name, imported or [])
                self.problem(node, message)
            else:
                self.problem(node, self._import_failed(name, error))
        except ImportError as error:
            self.problem(node, self._import_failed(name, error))
        return None

    def _import_failed(self, module: str, error: ImportError) -> str:
        extra = next(
            (extra for prefix, extra in EXTRAS.items() if module.startswith(prefix)), "all"
        )
        return prompts.CHECK_IMPORT_FAILED.format(module=module, error=error, extra=extra)

    def _module_suggestions(self, module: str, names: list[str]) -> str:
        hints: list[str] = []
        for name in names:
            for line in self._where_is(name):
                hints.append(
                    prompts.CHECK_ELSEWHERE.format(name=name, module=line[0], line=line[1])
                )
        if hints:
            return " " + " ".join(hints)
        modules = sorted({e.path for e in self.lookup.entries if e.kind == "module"})
        close = difflib.get_close_matches(module, modules, n=3, cutoff=0.6)
        return " " + prompts.DID_YOU_MEAN.format(names=_names(close)) if close else ""

    def _where_is(self, name: str) -> list[tuple[str, str]]:
        """Modules that export ``name``, with the import line to use."""
        found: list[tuple[str, str]] = []
        for entry in self.lookup.entries:
            if entry.kind in ("class", "function", "module", "attribute") and entry.name == name:
                if (
                    entry.kind == "attribute"
                    and entry.path.count(".") > 2
                    and entry.path.split(".")[-2][:1].isupper()
                ):
                    continue  # a class attribute, not importable
                public = entry.public
                module, _, last = public.rpartition(".")
                line = f"from {module} import {last}"
                if all(line != existing for _, existing in found):
                    found.append((module, line))
        found.sort(key=lambda item: (item[0].count("."), item[0]))
        return found[:2]

    def module_attribute(self, module: Any, name: str, node: ast.AST) -> Value | None:
        try:
            value = getattr(module, name, _MISSING)
        except ImportError as error:
            self.problem(node, self._import_failed(f"{module.__name__}.{name}", error))
            return None
        if value is _MISSING and not safe_to_import(f"{module.__name__}.{name}"):
            return None
        if value is _MISSING:
            try:
                value = importlib.import_module(f"{module.__name__}.{name}")
            except ModuleNotFoundError as error:
                if not (error.name and error.name.startswith(f"{module.__name__}.{name}")):
                    self.problem(node, self._import_failed(f"{module.__name__}.{name}", error))
                    return None
                value = _MISSING
            except ImportError as error:
                self.problem(node, self._import_failed(f"{module.__name__}.{name}", error))
                return None
        if value is _MISSING:
            message = prompts.CHECK_NO_NAME.format(owner=module.__name__, name=name)
            self.problem(node, message + self._name_suggestions(module, name))
            return None
        return Value("module" if inspect.ismodule(value) else "object", value, label=name)

    def _name_suggestions(self, module: Any, name: str) -> str:
        parts: list[str] = []
        elsewhere = [item for item in self._where_is(name) if item[0] != module.__name__]
        for where, line in elsewhere:
            parts.append(prompts.CHECK_ELSEWHERE.format(name=name, module=where, line=line))
        if not elsewhere:
            names = _public_names(module)
            close = _close(name, names)
            if close:
                parts.append(prompts.DID_YOU_MEAN.format(names=_names(close)))
        hint = prompts.NAME_HINTS.get(name)
        if hint:
            parts.append(hint)
        return (" " + " ".join(parts)) if parts else ""

    # -- expressions -----------------------------------------------------------------

    def expr(self, node: ast.expr | None, scope: dict[str, Value]) -> Value | None:
        if node is None:
            return None
        key = id(node)
        if key in self._done:
            return self._done[key]
        self._done[key] = None
        value = self._expr(node, scope)
        self._done[key] = value
        return value

    def _expr(self, node: ast.expr, scope: dict[str, Value]) -> Value | None:
        if isinstance(node, ast.Name):
            return scope.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.expr(node.value, scope)
            return self.attribute(base, node) if base is not None else None
        if isinstance(node, ast.Await):
            if isinstance(node.value, ast.Call):
                self._awaited.add(id(node.value))
            return self.expr(node.value, scope)
        if isinstance(node, ast.Call):
            function = self.expr(node.func, scope)
            for argument in node.args:
                self.expr(argument.value if isinstance(argument, ast.Starred) else argument, scope)
            for keyword in node.keywords:
                self.expr(keyword.value, scope)
            return self.call(function, node) if function is not None else None
        if isinstance(node, ast.Subscript):
            base = self.expr(node.value, scope)
            self.expr(node.slice, scope)
            if base is None:
                return None
            if base.item is not None:
                return Value("instance", base.item)
            if base.kind == "object" and inspect.isclass(base.obj):
                return base  # a generic alias such as RunContext[Deps]
            return None
        if isinstance(node, ast.Lambda):
            inner = dict(scope)
            for argument in node.args.args + node.args.kwonlyargs:
                inner.pop(argument.arg, None)
            self.expr(node.body, inner)
            return None
        if isinstance(node, ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp):
            inner = dict(scope)
            for generator in node.generators:
                iterated = self.expr(generator.iter, inner)
                item = Value("instance", iterated.item) if iterated and iterated.item else None
                self.bind(generator.target, item, inner)
                for condition in generator.ifs:
                    self.expr(condition, inner)
            if isinstance(node, ast.DictComp):
                self.expr(node.key, inner)
                self.expr(node.value, inner)
            else:
                self.expr(node.elt, inner)
            return None
        if isinstance(node, ast.NamedExpr):
            value = self.expr(node.value, scope)
            self.bind(node.target, value, scope)
            return value
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.expr(child, scope)
            elif isinstance(child, ast.comprehension | ast.keyword):
                for grandchild in ast.iter_child_nodes(child):
                    if isinstance(grandchild, ast.expr):
                        self.expr(grandchild, scope)
        return None

    def attribute(self, base: Value, node: ast.Attribute) -> Value | None:
        name = node.attr
        if base.kind == "module":
            self.report.counts["attributes"] += 1
            return self.module_attribute(base.obj, name, node)
        if base.kind == "object" and inspect.isclass(base.obj):
            cls = base.obj
            if not _is_qagents(cls):
                return None
            self.report.counts["attributes"] += 1
            value = _safe_getattr(cls, name)
            if value is _MISSING:
                if name in getattr(cls, "model_fields", {}) or name in _annotations(cls):
                    return None  # an instance attribute read from the class: leave it
                self.problem(
                    node,
                    prompts.CHECK_NO_ATTRIBUTE.format(owner=_label(cls), name=name)
                    + self._attribute_suggestions(cls, name),
                )
                return None
            return Value("module" if inspect.ismodule(value) else "object", value)
        if base.kind == "instance" and inspect.isclass(base.obj):
            cls = base.obj
            if not _is_qagents(cls):
                return None
            self.report.counts["attributes"] += 1
            known, kind, annotation = _instance_member(cls, name)
            if not known:
                owner = f"{base.label} ({_label(cls)})" if base.label else _label(cls)
                self.problem(
                    node,
                    prompts.CHECK_NO_ATTRIBUTE.format(owner=owner, name=name)
                    + self._attribute_suggestions(cls, name),
                )
                return None
            if kind == "method":
                return Value("method", getattr(cls, name), owner=cls, label=name)
            if kind == "unknown":
                return None
            resolved = single_class(annotation)
            item = element_class(annotation)
            if resolved is not None and _is_qagents(resolved):
                return Value("instance", resolved, label=name)
            if item is not None and _is_qagents(item):
                return Value("instance", list, item=item, label=name)
            return None
        return None

    def _attribute_suggestions(self, cls: type, name: str) -> str:
        names = _member_names(cls)
        close = _close(name, names)
        parts = [prompts.DID_YOU_MEAN.format(names=_names(close))] if close else []
        hint = prompts.ATTRIBUTE_HINTS.get(name)
        if hint:
            parts.append(hint)
        return (" " + " ".join(parts)) if parts else ""

    # -- calls -----------------------------------------------------------------------

    def call(self, function: Value, node: ast.Call) -> Value | None:
        if function.kind == "instance" and function.obj is _ToolDecorator:
            from qagents.tools import Tool

            return Value("instance", Tool)  # @tool(...)(function) or tool(...)(function)
        if function.kind == "instance":
            call = getattr(function.obj, "__call__", None)  # noqa: B004
            if call is None or not _is_qagents(function.obj):
                return None
            function = Value("method", call, owner=function.obj, label=function.label)
        target = function.obj
        if function.kind == "module" or not callable(target):
            return None
        if not _is_qagents(target) and function.kind != "method":
            return None
        self.report.counts["calls"] += 1
        name = self.callable_name(function)
        self.check_arguments(target, function, node, name)
        if self._async_depth and getattr(target, "__name__", "") == "run_sync":
            self.problem(
                node,
                prompts.CHECK_RUN_SYNC_IN_ASYNC.format(
                    callable=name, replacement=_await_form(node)
                ),
                severity="warning",
            )
        if inspect.isclass(target):
            return Value("instance", target)
        if _is_tool_function(target):
            return Value("instance", _ToolDecorator) if not node.args else None
        returned = return_type(target)
        if returned is not None and _is_qagents(returned):
            return Value("instance", returned)
        return None

    def callable_name(self, function: Value) -> str:
        target = function.obj
        if function.kind == "method" and function.owner is not None:
            return f"{function.owner.__name__}.{getattr(target, '__name__', '?')}"
        public = self.lookup.public_path(target)
        return (public or getattr(target, "__name__", "?")).removeprefix("qagents.")

    def check_arguments(self, target: Any, function: Value, node: ast.Call, name: str) -> None:
        try:
            # A qagents pydantic model with its own __init__ (ExternalApproval) is checked
            # against that __init__; its **data set the fields, none of them required here.
            signature = call_signature(target)
        except (TypeError, ValueError):
            return
        parameters = list(signature.parameters.values())
        if function.kind == "method" and parameters and parameters[0].name in ("self", "cls"):
            if not inspect.ismethod(target):
                parameters = parameters[1:]
        has_star = any(isinstance(a, ast.Starred) for a in node.args)
        has_double_star = any(k.arg is None for k in node.keywords)
        var_positional = any(p.kind is p.VAR_POSITIONAL for p in parameters)
        var_keyword = any(p.kind is p.VAR_KEYWORD for p in parameters)
        positional = [
            p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        by_keyword = {
            p.name: p for p in parameters if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        accepted = [name for name in by_keyword if not name.startswith("_")]
        forwarded_names: set[str] = set()
        # Unknown keywords are reported unless they may land in a **kwargs nobody checks.
        unchecked_kwargs = var_keyword
        if var_keyword:
            receiving = forwarded_parameters(
                target, function.owner if function.kind == "method" else None
            )
            if receiving is not None:
                forwarded_names = {p.name for p in receiving}
                unchecked_kwargs = False
            accepted += sorted(forwarded_names - set(accepted))

        count = len(node.args)
        if not has_star and not var_positional and count > len(positional):
            self.problem(
                node,
                prompts.CHECK_POSITIONAL.format(
                    callable=name, count=len(positional),
                    names=", ".join(p.name for p in positional) or "none",
                ),
            )  # fmt: skip
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            known = keyword.arg in by_keyword or keyword.arg in forwarded_names
            if known or unchecked_kwargs:
                continue
            message = prompts.CHECK_UNKNOWN_KEYWORD.format(callable=name, name=keyword.arg)
            close = _close(keyword.arg, accepted)
            if close:
                message += " " + prompts.DID_YOU_MEAN.format(names=_names(close))
            hint = prompts.KEYWORD_HINTS.get(keyword.arg)
            if hint:
                message += " " + hint
            message += " " + prompts.CHECK_ACCEPTS.format(names=", ".join(accepted) or "none")
            self.problem(keyword, message)
        if has_star or has_double_star:
            return
        given = {k.arg for k in node.keywords if k.arg}
        missing = [
            p.name
            for index, p in enumerate(positional)
            if p.default is p.empty and index >= count and p.name not in given
        ]
        missing += [
            p.name
            for p in parameters
            if p.kind is p.KEYWORD_ONLY and p.default is p.empty and p.name not in given
        ]
        if missing:
            self.problem(
                node, prompts.CHECK_MISSING.format(callable=name, names=", ".join(missing))
            )

    def not_awaited(self, node: ast.expr | None, scope: dict[str, Value]) -> None:
        """Warn about ``x = agent.run(...)`` and a bare ``agent.run(...)`` statement."""
        if not isinstance(node, ast.Call) or id(node) in self._awaited:
            return
        function = self._done.get(id(node.func))
        if function is None or function.kind not in ("object", "method"):
            return
        target = function.obj
        if inspect.iscoroutinefunction(getattr(target, "__func__", target)):
            name = self.callable_name(function)
            self.problem(
                node,
                prompts.CHECK_NOT_AWAITED.format(callable=name, call=_short_source(node)),
                severity="warning",
            )

    # -- reporting -------------------------------------------------------------------

    def problem(self, node: ast.AST, message: str, *, severity: str = "error") -> None:
        line = getattr(node, "lineno", 1)
        column = getattr(node, "col_offset", 0)
        problem = Problem(line, column, message, severity)
        if problem not in self.report.problems:
            self.report.problems.append(problem)


class _ToolDecorator:
    """Marks the value of ``tool(...)`` with options: a decorator that makes a Tool."""


def _is_tool_function(target: Any) -> bool:
    try:
        from qagents.tools import tool
    except ImportError:  # pragma: no cover
        return False
    return target is tool


def check_code(code: str, lookup: ApiLookup) -> Report:
    """Check ``code`` against the installed qagents. See the module docstring."""
    return Checker(lookup).check(code)


# --- introspection helpers --------------------------------------------------------------


def _version() -> str:
    import qagents

    return qagents.__version__


def _safe_getattr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name, _MISSING)
    except Exception:  # noqa: BLE001 - lazy attributes may fail to import
        return _MISSING


def _is_qagents(obj: Any) -> bool:
    module = getattr(obj, "__module__", None)
    if inspect.ismethod(obj):
        module = getattr(obj.__func__, "__module__", None)
    return isinstance(module, str) and (module == "qagents" or module.startswith("qagents."))


def _label(cls: type) -> str:
    return cls.__name__


def _annotations(cls: type) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
        found.update(getattr(klass, "__annotations__", {}) or {})
    return found


@lru_cache(maxsize=512)
def _self_assignments(cls: type) -> tuple[frozenset[str], dict[str, tuple[str, str]]]:
    """``self.<name> = ...`` in the methods of the qagents classes of ``cls``'s MRO.

    Returns every assigned name, and for the names assigned with an annotation
    (``self.result: RunResult | None = None``) the annotation's source and the module it is
    read in: the first one found, from ``cls`` itself up through its bases.
    """
    names: set[str] = set()
    annotated: dict[str, tuple[str, str]] = {}
    for klass in cls.__mro__:
        if not _is_qagents(klass):
            continue
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(klass)))
        except (OSError, TypeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign | ast.AugAssign):
                targets = [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    names.add(target.attr)
                    if isinstance(node, ast.AnnAssign) and target.attr not in annotated:
                        annotated[target.attr] = (klass.__module__, ast.unparse(node.annotation))
    return frozenset(names), annotated


def _self_assigned(cls: type) -> frozenset[str]:
    """Attribute names assigned as ``self.<name> = ...`` in the methods of qagents classes."""
    return _self_assignments(cls)[0]


def _self_annotation(cls: type, name: str) -> Any:
    """The annotation of ``self.<name>: T = ...`` in a qagents class, resolved; else None."""
    found = _self_assignments(cls)[1].get(name)
    if found is None:
        return None
    module_name, source = found
    try:
        module = importlib.import_module(module_name)
    except ImportError:  # pragma: no cover - the class was imported from it
        return None
    return resolve_annotation(_Namespace(vars(module)), source)


def _instance_member(cls: type, name: str) -> tuple[bool, str, Any]:
    """Whether instances of ``cls`` have ``name``, what it is, and its annotation."""
    for klass in cls.__mro__:
        if "__getattr__" in vars(klass) and _is_qagents(klass):
            return True, "unknown", None
    config = getattr(cls, "model_config", None)
    if isinstance(config, dict) and config.get("extra") == "allow":
        return True, "unknown", None
    fields = getattr(cls, "model_fields", None)
    if isinstance(fields, dict) and name in fields:
        return True, "field", fields[name].annotation
    for klass in cls.__mro__:
        if name in vars(klass):
            raw = vars(klass)[name]
            if isinstance(raw, property):
                getter = raw.fget
                annotation = None
                if getter is not None:
                    try:
                        annotation = resolve_annotation(
                            getter, inspect.signature(getter).return_annotation
                        )
                    except (TypeError, ValueError):
                        annotation = None
                return True, "property", annotation
            if inspect.isfunction(raw) or isinstance(raw, staticmethod | classmethod):
                return True, "method", None
            return True, "attribute", None
    annotations = _annotations(cls)
    if name in annotations:
        holder = next((k for k in cls.__mro__ if name in getattr(k, "__annotations__", {})), cls)
        annotation = annotations[name]
        if isinstance(annotation, str):
            module = importlib.import_module(holder.__module__)
            annotation = resolve_annotation(_Namespace(vars(module)), annotation)
        return True, "attribute", annotation
    if name in _self_assigned(cls):
        annotation = _self_annotation(cls, name)
        if annotation is not None:
            return True, "attribute", annotation  # self.result: RunResult | None = None
        return True, "unknown", None
    for klass in cls.__mro__:
        slots = getattr(klass, "__slots__", ())
        if name in (slots if isinstance(slots, tuple | list) else (slots,)):
            return True, "unknown", None
    if not all(
        _is_qagents(k) or k.__module__ in ("builtins", "abc", "typing") for k in cls.__mro__
    ):
        # Inherits from a class outside qagents (pydantic, Exception, ...): trust hasattr only
        # for names those classes define, and leave the rest unknown rather than wrong.
        if hasattr(cls, name):
            return True, "unknown", None
    return False, "", None


class _Namespace:
    """Gives `resolve_annotation` a module's globals for a class attribute's annotation."""

    def __init__(self, namespace: dict[str, Any]) -> None:
        self.__globals__ = namespace


def _member_names(cls: type) -> list[str]:
    names = {name for name in dir(cls) if not name.startswith("_")}
    names |= set(getattr(cls, "model_fields", {}) or {})
    names |= {name for name in _annotations(cls) if not name.startswith("_")}
    names |= {name for name in _self_assigned(cls) if not name.startswith("_")}
    pydantic = {name for name in dir(_pydantic_base()) if not name.startswith("_")}
    return sorted(names - pydantic) if _pydantic_base() in cls.__mro__ else sorted(names)


def _pydantic_base() -> type:
    from pydantic import BaseModel

    return BaseModel


def _public_names(module: Any) -> list[str]:
    names = {name for name in dir(module) if not name.startswith("_")}
    lazy = getattr(module, "_LAZY", None)
    if isinstance(lazy, dict):
        names |= set(lazy)
    return sorted(names)


def _close(name: str, candidates: list[str] | set[str]) -> list[str]:
    candidates = list(candidates)
    found = difflib.get_close_matches(name, candidates, n=3, cutoff=0.6)
    lowered = {c.lower(): c for c in candidates}
    if name.lower() in lowered and lowered[name.lower()] not in found:
        found.insert(0, lowered[name.lower()])
    for candidate in candidates:
        if len(found) >= 4:
            break
        if len(name) >= 4 and (name.lower() in candidate.lower()) and candidate not in found:
            found.append(candidate)
    return found[:4]


def _names(names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in names)


def _short_source(node: ast.AST, limit: int = 60) -> str:
    text = ast.unparse(node)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _await_form(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return f"{ast.unparse(func.value)}.run"
    return "agent.run"
