"""Live introspection of the installed qagents: what ``get_api`` answers.

A name is resolved by importing the module that defines it and walking its attributes, so the
signature and docstring are those of the code that will run. Optional extras are imported only
when a name from them is asked for; when one is missing, the answer falls back to the static
index (read from the source) and says which extra to install.
"""

from __future__ import annotations

import difflib
import importlib
import inspect
import os
import re
import sys
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from . import prompts
from .apiindex import ApiEntry

# Optional extras must not phone home when the docs server imports them to read a docstring.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")

EXTRAS: dict[str, str] = {
    "qagents.models.anthropic": "anthropic",
    "qagents.models.qcloud": "qcloud",
    "qagents.toolkits.mcp": "mcp",
    "qagents.integrations.mlflow": "mlflow",
    "qagents.stores.postgres": "postgres",
    "qagents.tracing": "otel",
}
"""Modules that need an optional extra, and the extra's name (``pip install 'qagents[...]'``).
Importing ``qagents.models.qcloud`` works without the extra; using the model needs it."""

_MISSING = object()
_TYPING_PREFIX = re.compile(r"\b(?:typing|typing_extensions|collections\.abc)\.")
_QAGENTS_PREFIX = re.compile(r"\bqagents(?:\.\w+)*\.(?=[A-Za-z_])")
_CLASS_REPR = re.compile(r"<class '(?:[\w.]+\.)?(\w+)'>")
_FUNCTION_REPR = re.compile(r"<function ([\w.<>]+) at 0x[0-9a-fA-F]+>")


@dataclass
class Target:
    """A resolved public qagents object."""

    path: str
    """The name to show, for example ``qagents.Agent.run``."""
    kind: str
    """``module``, ``class``, ``function``, ``method``, ``property``, ``field`` or
    ``attribute``."""
    obj: Any = None
    owner: type | None = None
    """The class a member belongs to."""
    member: str | None = None
    entry: ApiEntry | None = None
    """The static index entry, when the source documents it."""
    missing: str | None = None
    """The package that could not be imported, when only static information is available."""
    field_info: Any = None
    """A pydantic ``FieldInfo``, for a model field."""


@dataclass
class Resolution:
    target: Target | None = None
    candidates: list[str] = field(default_factory=list)
    """Several objects have this name; ask again with one of these."""
    suggestions: list[str] = field(default_factory=list)
    """Nothing has this name; these are close."""
    also: list[str] = field(default_factory=list)
    """Other objects with the same name, less likely to be meant."""


class ApiLookup:
    """Resolves and renders public qagents names against the installed package."""

    def __init__(self, api: list[ApiEntry]) -> None:
        self._api = api
        self._by_path = {entry.path: entry for entry in api}
        self._by_public = {entry.public: entry for entry in api}
        self._names = sorted({entry.public for entry in api} | {entry.path for entry in api})

    @property
    def entries(self) -> list[ApiEntry]:
        """Every entry of the static API index."""
        return self._api

    def entry(self, path: str) -> ApiEntry | None:
        """The static index entry defined at ``path`` (``qagents.agent.Agent``), if any."""
        return self._by_path.get(path)

    # -- resolution ----------------------------------------------------------------------

    def resolve(self, symbol: str) -> Resolution:
        """Find the object a loosely written name means.

        Accepts ``Agent``, ``qagents.Agent``, ``Agent.run``, ``agent.run``,
        ``qagents.toolkits.files``, ``qagents.toolkits.files:file_tools`` and
        ``ScriptedModel`` (found in ``qagents.models.testing``).
        """
        text = normalize_symbol(symbol)
        if not text:
            return Resolution()
        full = text if text == "qagents" or text.startswith("qagents.") else f"qagents.{text}"
        target = self.walk(full)
        if target is not None:
            return Resolution(target)
        matches = self._static_matches(text)
        also: list[str] = []
        if len(matches) > 1:
            ranked = sorted(matches, key=self._tier)
            best = self._tier(ranked[0])
            top = [m for m in ranked if self._tier(m) == best]
            if len({m.public for m in top}) > 1:
                return Resolution(candidates=sorted({m.public for m in matches}))
            also = sorted({m.public for m in ranked if m.public != top[0].public})
            matches = top[:1]
        if matches:
            entry = matches[0]
            target = self.walk(entry.public) or self.walk(entry.path)
            if target is None:
                target = self._static_target(entry)
            return Resolution(target, also=also)
        return Resolution(suggestions=self.suggest(text))

    def _tier(self, entry: ApiEntry) -> tuple[int, int]:
        """Lower is more likely meant: exported by ``qagents`` first, then module-level."""
        exported = 0 if entry.public.count(".") == 1 else 1
        parent = self._by_path.get(entry.path.rsplit(".", 1)[0])
        module_level = 0 if parent is None or parent.kind == "module" else 1
        return exported, module_level

    def suggest(self, text: str, n: int = 6) -> list[str]:
        """Public names close to ``text``."""
        text = text.removeprefix("qagents.")
        shorts = {name.removeprefix("qagents."): name for name in self._names}
        last = {name.rsplit(".", 1)[-1]: name for name in self._names}
        found = difflib.get_close_matches(text, list(shorts), n=n, cutoff=0.6)
        found += difflib.get_close_matches(text.rsplit(".", 1)[-1], list(last), n=n, cutoff=0.7)
        lowered = text.lower()
        found += [s for s in shorts if lowered in s.lower()][:n]
        result: list[str] = []
        for name in found:
            full = shorts.get(name) or last.get(name) or name
            public = self._by_path[full].public if full in self._by_path else full
            if public not in result:
                result.append(public)
        return result[:n]

    def _static_matches(self, text: str) -> list[ApiEntry]:
        text = text.removeprefix("qagents.")
        exact = [
            e
            for e in self._api
            if e.public.endswith(f".{text}") or e.path.endswith(f".{text}") or e.public == text
        ]
        if exact:
            return _unique_entries(exact)
        lowered = text.lower()
        loose = [
            e
            for e in self._api
            if e.public.lower().endswith(f".{lowered}") or e.path.lower().endswith(f".{lowered}")
        ]
        return _unique_entries(loose)

    def walk(self, path: str) -> Target | None:
        """Import and walk a dotted path that starts with ``qagents``; None when it fails."""
        parts = path.split(".")
        if not parts or parts[0] != "qagents" or not safe_to_import(path):
            return None
        try:
            obj: Any = importlib.import_module("qagents")
        except ImportError:  # pragma: no cover - the package is running this code
            return None
        owner: type | None = None
        container: ModuleType | None = None
        for index, part in enumerate(parts[1:], start=1):
            so_far = ".".join(parts[: index + 1])
            if inspect.isclass(obj):
                member = self._class_member(obj, part, so_far, last=index == len(parts) - 1)
                if member is None:
                    return None
                if member.kind in ("field", "attribute") or index == len(parts) - 1:
                    member.path = so_far
                    return member
                owner, obj = obj, member.obj
                continue
            try:
                value = getattr(obj, part)
            except AttributeError as error:
                # A lazy name whose extra is not installed: qagents.__getattr__ raises an
                # AttributeError chained from the ImportError (getattr with a default would
                # hide it, and the lookup would fall through to the module's source).
                if isinstance(error.__cause__, ImportError):
                    return self._missing_target(so_far, error.__cause__)
                value = _MISSING
            except ImportError as error:  # a lazy name whose extra is not installed
                return self._missing_target(so_far, error)
            if value is _MISSING and inspect.ismodule(obj):
                try:
                    value = importlib.import_module(f"{obj.__name__}.{part}")
                except ModuleNotFoundError as error:
                    if error.name and error.name.startswith(f"{obj.__name__}.{part}"):
                        return None
                    return self._missing_target(so_far, error)
                except ImportError as error:
                    return self._missing_target(so_far, error)
            if value is _MISSING:
                return None
            owner = obj if inspect.isclass(obj) else None
            container = obj if inspect.ismodule(obj) else None
            obj = value
        return self._target(path, obj, owner, parts[-1], module=container)

    def _class_member(self, cls: type, name: str, path: str, *, last: bool) -> Target | None:
        fields = getattr(cls, "model_fields", None)
        if isinstance(fields, dict) and name in fields and _defined_in_qagents(cls, name):
            entry = self._member_entry(cls, name)
            return Target(
                path, "field", owner=cls, member=name, entry=entry, field_info=fields[name]
            )
        raw = _static_lookup(cls, name)
        if raw is _MISSING:
            annotations = _class_annotations(cls)
            if name in annotations:
                entry = self._member_entry(cls, name)
                return Target(path, "attribute", owner=cls, member=name, entry=entry)
            return None
        value = getattr(cls, name, raw)
        return self._target(path, value, cls, name, raw=raw)

    def _target(
        self,
        path: str,
        obj: Any,
        owner: Any,
        name: str,
        *,
        raw: Any = _MISSING,
        module: ModuleType | None = None,
    ) -> Target:
        static = raw if raw is not _MISSING else None
        if inspect.ismodule(obj):
            kind = "module"
        elif inspect.isclass(obj):
            kind = "class"
        elif isinstance(static, property) or isinstance(obj, property):
            kind = "property"
        elif inspect.isroutine(obj) or isinstance(static, staticmethod | classmethod):
            kind = "method" if inspect.isclass(owner) else "function"
        else:
            kind = "attribute"
        entry = self.entry_for(obj, owner if inspect.isclass(owner) else None, name, kind)
        if entry is None and kind == "attribute" and module is not None:
            entry = self.module_attribute_entry(module, name, obj)
        return Target(
            path, kind, obj=obj, owner=owner if inspect.isclass(owner) else None,
            member=name if inspect.isclass(owner) else None, entry=entry,
        )  # fmt: skip

    def _missing_target(self, path: str, error: ImportError) -> Target:
        entry = self._by_public.get(path) or self._by_path.get(path)
        missing = getattr(error, "name", None) or str(error)
        if entry is None:
            return Target(path, "attribute", missing=missing)
        return self._static_target(entry, missing=missing)

    def _static_target(self, entry: ApiEntry, missing: str | None = None) -> Target:
        return Target(entry.public, entry.kind, entry=entry, missing=missing or "an optional extra")

    def entry_for(self, obj: Any, owner: type | None, name: str, kind: str) -> ApiEntry | None:
        if kind == "module" and inspect.ismodule(obj):
            return self._by_path.get(obj.__name__)
        if owner is not None:
            return self._member_entry(owner, name)
        module = getattr(obj, "__module__", None)
        qualname = getattr(obj, "__qualname__", None)
        if isinstance(module, str) and isinstance(qualname, str):
            return self._by_path.get(f"{module}.{qualname}")
        return None

    def module_attribute_entry(self, module: ModuleType, name: str, value: Any) -> ApiEntry | None:
        """The static entry of a module attribute (a constant, a type alias), which has no
        ``__module__`` of its own to find it by.

        ``qagents.external.APPROVAL_KEY`` is defined in ``qagents.external.permissions``
        and re-exported: the entry is the one, defined in this module or a module below it
        (then in any module), whose module holds this very object under that name.
        """
        entry = self._by_path.get(f"{module.__name__}.{name}")
        if entry is not None and entry.kind == "attribute":
            return entry
        prefix = f"{module.__name__}."
        candidates = [
            e for e in self._api if e.kind == "attribute" and e.path.rsplit(".", 1)[-1] == name
        ]
        candidates.sort(key=lambda e: not e.path.startswith(prefix))  # this package's first
        for candidate in candidates:
            defining = sys.modules.get(candidate.path.rsplit(".", 1)[0])
            if defining is not None and vars(defining).get(name, _MISSING) is value:
                return candidate
        return None

    def _member_entry(self, cls: type, name: str) -> ApiEntry | None:
        for klass in cls.__mro__:
            entry = self._by_path.get(f"{klass.__module__}.{klass.__qualname__}.{name}")
            if entry is not None:
                return entry
        return None

    def public_path(self, obj: Any) -> str | None:
        """The public spelling of a class or function, for example ``qagents.Agent``."""
        module = getattr(obj, "__module__", None)
        qualname = getattr(obj, "__qualname__", None)
        if not isinstance(module, str) or not isinstance(qualname, str):
            return None
        entry = self._by_path.get(f"{module}.{qualname}")
        return entry.public if entry else f"{module}.{qualname}"

    # -- rendering -----------------------------------------------------------------------

    def render(self, target: Target, version: str) -> str:
        """Markdown for one object: signature, docstring, members and where it is defined."""
        lines = [f"# {target.path} ({target.kind})", ""]
        location = self.location(target)
        if location:
            lines.append(f"Source: `{location}` | qagents {version}")
        importing = import_line(target.path, target.kind)
        if importing:
            lines.append(f"Import: `{importing}`")
        module_name = _module_of(target)
        extra = _extra_for(module_name)
        if target.missing:
            lines.append(prompts.API_MISSING_EXTRA.format(missing=target.missing, extra=extra))
        elif extra:
            lines.append(prompts.API_NEEDS_EXTRA.format(extra=extra))
        lines.append("")

        if target.obj is None and target.entry is not None:
            entry = target.entry
            if entry.signature:
                lines += ["```python", f"{entry.name}{entry.signature}", "```", ""]
            if entry.value:
                lines += ["```python", f"{entry.name}{entry.value}", "```", ""]
            lines.append(entry.doc or prompts.API_NO_DOCSTRING)
            return "\n".join(lines).rstrip() + "\n"

        if target.kind == "module":
            lines += self._render_module(target)
        elif target.kind == "class":
            lines += self._render_class(target)
        elif target.kind in ("function", "method"):
            lines += self._render_function(target)
        elif target.kind == "property":
            lines += self._render_property(target)
        else:
            lines += self._render_attribute(target)
        return "\n".join(lines).rstrip() + "\n"

    def location(self, target: Target) -> str | None:
        if target.entry is not None:
            return f"{target.entry.file}:{target.entry.line}"
        obj = target.obj
        try:
            file = inspect.getsourcefile(obj)
            line = inspect.getsourcelines(obj)[1] if file else None
        except (OSError, TypeError):
            return None
        if not file:
            return None
        return f"{_display_file(Path(file))}:{line or 1}"

    def _render_module(self, target: Target) -> list[str]:
        module: ModuleType = target.obj
        lines = [own_doc(module) or prompts.API_NO_DOCSTRING, ""]
        members = self.module_members(module)
        if not members:
            return lines
        lines.append("## Members")
        for kind in ("module", "class", "function", "attribute"):
            for name, value, member_kind in members:
                if member_kind != kind:
                    continue
                summary = _first_sentence(_doc_of(value, self, module, name))
                label = f"{name}{compact_signature(value)}" if kind == "function" else name
                lines.append(f"- `{label}` ({kind}){': ' + summary if summary else ''}")
        return lines

    def module_members(self, module: ModuleType) -> list[tuple[str, Any, str]]:
        names: list[str]
        exported = getattr(module, "__all__", None)
        if isinstance(exported, list | tuple):
            names = [str(name) for name in exported]
            lazy = getattr(module, "_LAZY", None)
            if isinstance(lazy, dict):
                names += [name for name in lazy if name not in names]
        else:
            names = [name for name in vars(module) if not name.startswith("_")]
            lazy = getattr(module, "_LAZY", None)
            if isinstance(lazy, dict):
                names += [name for name in lazy if name not in names]
        prefix = f"{module.__name__}."
        for entry in self._api:
            if entry.kind == "module" and entry.path.startswith(prefix):
                child = entry.path[len(prefix) :]
                if "." not in child and child not in names:
                    names.append(child)
        members: list[tuple[str, Any, str]] = []
        for name in names:
            if name.startswith("_") or name == "annotations":
                continue
            value = vars(module).get(name, _MISSING)
            if value is _MISSING:
                child = self._by_path.get(f"{module.__name__}.{name}")
                if child is not None and child.kind == "module":
                    members.append((name, None, "module"))
                elif isinstance(getattr(module, "_LAZY", None), dict) and name in module._LAZY:
                    members.append((name, None, "class"))
                continue
            if inspect.ismodule(value):
                if value.__name__.startswith("qagents."):
                    members.append((name, value, "module"))
                continue
            defined = getattr(value, "__module__", module.__name__)
            if exported is None and isinstance(defined, str) and defined != module.__name__:
                if inspect.isclass(value) or inspect.isroutine(value):
                    continue  # imported from elsewhere, not part of this module's surface
            if inspect.isclass(value):
                members.append((name, value, "class"))
            elif inspect.isroutine(value):
                members.append((name, value, "function"))
            elif exported is not None or f"{module.__name__}.{name}" in self._by_path:
                members.append((name, value, "attribute"))
        return members

    def _render_class(self, target: Target) -> list[str]:
        cls: type = target.obj
        name = target.path.rsplit(".", 1)[-1]
        pydantic = _is_pydantic_model(cls)
        if pydantic:  # the fields below say the rest
            signature = f"{name}{compact_signature(cls, width=100)}"
        else:
            signature = format_signature(cls, name)
        lines = ["```python", f"class {signature}", "```", ""]
        bases = [
            base.__name__
            for base in cls.__bases__
            if base is not object and base.__name__ not in ("Generic",)
        ]
        if bases:
            lines += [f"Bases: {', '.join(f'`{b}`' for b in bases)}", ""]
        doc = own_doc(cls)
        module = inspect.getmodule(cls)
        module_doc = own_doc(module) if module is not None else ""
        if doc:
            lines += [doc, ""]
            if len(doc.splitlines()) <= 2 and name in module_doc and _has_example(module_doc):
                # A one-line class docstring whose module explains it with an example.
                notes = prompts.API_MODULE_NOTES.format(module=cls.__module__)
                lines += [notes, "", module_doc, ""]
        elif module_doc:
            first = " ".join(module_doc.split("\n\n", 1)[0].split())
            lines += [f"From `{cls.__module__}`: {first}", ""]
        else:
            lines += [prompts.API_NO_DOCSTRING, ""]
        note = self._forward_note(cls, None)
        if note:
            lines += [note, ""]
        fields, methods, properties, attributes = self.class_members(cls)
        if fields:
            lines.append("## Fields")
            for field_name, info, doc, origin in fields:
                lines.append(f"- `{field_name}: {_field_type(info)}{_field_default(info)}`"
                             f"{_from(origin, cls)}{': ' + doc if doc else ''}")  # fmt: skip
            lines.append("")
        if properties:
            lines.append("## Properties")
            for prop_name, doc, origin in properties:
                getter = getattr(_static_lookup(cls, prop_name), "fget", None)
                hint = _return_hint(getter) if getter is not None else ""
                typed = f"{prop_name} -> {hint}" if hint else prop_name
                lines.append(f"- `{typed}`{_from(origin, cls)}{': ' + doc if doc else ''}")
            lines.append("")
        if methods:
            lines.append("## Methods")
            for method_name, function, doc, origin in methods:
                prefix = "async " if _is_async(function) else ""
                signature = compact_signature(function, skip_self=True)
                lines.append(f"- `{prefix}{method_name}{signature}`{_from(origin, cls)}"
                             f"{': ' + doc if doc else ''}")  # fmt: skip
            lines.append("")
        if attributes:
            lines.append("## Attributes")
            for attr_name, doc, origin in attributes:
                lines.append(f"- `{attr_name}`{_from(origin, cls)}{': ' + doc if doc else ''}")
            lines.append("")
        lines.append(prompts.API_MEMBER_HINT.format(path=target.path))
        return lines

    def class_members(
        self, cls: type
    ) -> tuple[
        list[tuple[str, Any, str, type]],
        list[tuple[str, Any, str, type]],
        list[tuple[str, str, type]],
        list[tuple[str, str, type]],
    ]:
        """Fields, methods, properties and documented attributes defined in qagents classes."""
        fields: list[tuple[str, Any, str, type]] = []
        methods: list[tuple[str, Any, str, type]] = []
        properties: list[tuple[str, str, type]] = []
        attributes: list[tuple[str, str, type]] = []
        seen: set[str] = set()
        model_fields = getattr(cls, "model_fields", None)
        if isinstance(model_fields, dict):
            for name, info in model_fields.items():
                origin = _field_origin(cls, name)
                if origin is None or name.startswith("_"):
                    continue
                entry = self._member_entry(cls, name)
                doc = _first_sentence(info.description or (entry.doc if entry else ""))
                fields.append((name, info, doc, origin))
                seen.add(name)
        pydantic = _is_pydantic_model(cls)
        for klass in cls.__mro__:
            if not klass.__module__.startswith("qagents"):
                continue
            for name, raw in vars(klass).items():
                if name.startswith("_") or name in seen or name == "model_config":
                    continue
                if pydantic and name.startswith("model_"):
                    continue  # pydantic's own hooks
                seen.add(name)
                entry = self._by_path.get(f"{klass.__module__}.{klass.__qualname__}.{name}")
                if isinstance(raw, property):
                    doc = _first_sentence(inspect.getdoc(raw) or (entry.doc if entry else ""))
                    properties.append((name, doc, klass))
                elif isinstance(raw, staticmethod | classmethod) or inspect.isfunction(raw):
                    function = getattr(cls, name)
                    methods.append(
                        (name, function, _first_sentence(inspect.getdoc(function) or ""), klass)
                    )
                elif entry is not None and entry.doc:
                    attributes.append((name, _first_sentence(entry.doc), klass))
            for name in getattr(klass, "__annotations__", {}):
                if name.startswith("_") or name in seen:
                    continue
                entry = self._by_path.get(f"{klass.__module__}.{klass.__qualname__}.{name}")
                if entry is not None and entry.doc:
                    seen.add(name)
                    attributes.append((name, _first_sentence(entry.doc), klass))
            for name, annotation, doc in instance_attributes(klass):
                if name in seen:
                    continue
                seen.add(name)
                label = f"{name}: {annotation}" if annotation else name
                attributes.append((label, _first_sentence(doc), klass))
        return fields, methods, properties, attributes

    def _render_function(self, target: Target) -> list[str]:
        function = target.obj
        name = target.path.rsplit(".", 1)[-1]
        prefix = "async def " if _is_async(function) else "def "
        signature = format_signature(function, name, skip_self=target.owner is not None)
        lines = ["```python", f"{prefix}{signature}", "```", ""]
        if _is_async(function):
            lines += [prompts.API_ASYNC_NOTE, ""]
        lines.append(own_doc(function) or prompts.API_NO_DOCSTRING)
        note = self._forward_note(function, target.owner)
        if note:
            lines += ["", note]
        return lines

    def _forward_note(self, function: Any, owner: type | None) -> str:
        """Where ``function``'s ``**kwargs`` go, as a sentence; empty when nowhere known."""
        found = _forwarding(function, owner)
        if found is None:
            return ""
        receiver, model = found
        if receiver is MODEL_FIELDS:  # _forwarding returns it only with the model as owner
            label = self.public_path(model) or _callable_name(model)
            return prompts.API_FORWARDS_FIELDS.format(model=label)
        return prompts.API_FORWARDS.format(target=self._forward_label(receiver))

    def _forward_label(self, function: Any) -> str:
        """How `prompts.API_FORWARDS` names a receiver: a constructor by its class's public
        name (``qagents.external.ExternalAgent``), anything else by `_callable_name`."""
        if inspect.isclass(function):
            return self.public_path(function) or _callable_name(function)
        qualname = getattr(function, "__qualname__", "") or ""
        if qualname.endswith(".__init__"):
            module = getattr(function, "__module__", "") or ""
            entry = self._by_path.get(f"{module}.{qualname.removesuffix('.__init__')}")
            if entry is not None:
                return entry.public
        return _callable_name(function)

    def _render_property(self, target: Target) -> list[str]:
        raw = (
            _static_lookup(target.owner, target.member) if target.owner and target.member else None
        )
        prop = raw if isinstance(raw, property) else target.obj
        returns = ""
        getter = getattr(prop, "fget", None)
        if getter is not None:
            hint = _return_hint(getter)
            if hint:
                returns = f" -> {hint}"
        doc = inspect.getdoc(prop) if prop is not None else None
        doc = doc or (target.entry.doc if target.entry else "") or prompts.API_NO_DOCSTRING
        name = target.path.rsplit(".", 1)[-1]
        return ["```python", f"@property\ndef {name}(self){returns}", "```", "", doc]

    def _render_attribute(self, target: Target) -> list[str]:
        lines: list[str] = []
        name = target.path.rsplit(".", 1)[-1]
        if target.kind == "field" and target.field_info is not None:
            info = target.field_info
            lines += ["```python", f"{name}: {_field_type(info)}{_field_default(info)}", "```", ""]
            doc = info.description or (target.entry.doc if target.entry else "")
            lines.append(doc or prompts.API_NO_DOCSTRING)
            return lines
        if target.obj is not None:
            value = short_repr(target.obj, limit=1500)
            lines += ["```python", f"{name} = {value}", "```", ""]
        elif target.entry is not None and target.entry.value:
            lines += ["```python", f"{name}{target.entry.value}", "```", ""]
        doc = target.entry.doc if target.entry else ""
        lines.append(doc or prompts.API_NO_DOCSTRING)
        return lines


# --- helpers shared with the code checker -----------------------------------------------


def safe_to_import(module: str) -> bool:
    """Whether importing ``module`` only defines things.

    A ``__main__`` module runs a program when it is imported (``qagents.__main__`` starts the
    command line), so the docs server never imports one, whatever a snippet asks for.
    """
    return "__main__" not in module.split(".")


def normalize_symbol(symbol: str) -> str:
    text = symbol.strip().strip("`").strip()
    text = re.sub(r"\(.*\)$", "", text)
    text = text.replace(":", ".").replace("/", ".")
    text = re.sub(r"\.py$", "", text)
    text = re.sub(r"^src\.", "", text)
    return text.strip(".")


def import_line(path: str, kind: str) -> str | None:
    parts = path.split(".")
    if kind == "module":
        return f"import {path}" if len(parts) > 1 else "import qagents"
    if kind in ("method", "property", "field", "attribute") and len(parts) > 2:
        # a member: import its class (or the module, for a module attribute)
        owner = parts[:-1]
        if owner[-1][:1].isupper():
            return f"from {'.'.join(owner[:-1])} import {owner[-1]}"
        return f"from {'.'.join(owner)} import {parts[-1]}"
    if len(parts) >= 2:
        return f"from {'.'.join(parts[:-1])} import {parts[-1]}"
    return None


def model_init(cls: Any) -> Any:
    """The ``__init__`` a qagents pydantic model defines for itself, or None.

    pydantic gives a model a signature made of its fields, merged with the parameters of
    such an ``__init__``. The merge lists every field without a default as a required
    keyword, even one the ``__init__`` fills in: ``ExternalApproval(approved=True)`` works
    and sets ``tool_call_id``, ``name`` and ``content`` itself. The ``__init__`` is what a
    call runs, so its parameters are the ones to show and check.
    """
    if not inspect.isclass(cls) or not _is_pydantic_model(cls):
        return None
    return _own_init(cls)


def call_signature(obj: Any) -> inspect.Signature:
    """The parameters a call of ``obj`` takes: `inspect.signature`, except for a qagents
    pydantic model with its own ``__init__`` (see `model_init`), whose ``__init__`` without
    ``self`` it returns. Raises what `inspect.signature` raises."""
    init = model_init(obj)
    if init is None:
        return inspect.signature(obj)
    signature = inspect.signature(init)
    parameters = list(signature.parameters.values())[1:]
    return signature.replace(parameters=parameters, return_annotation=inspect.Signature.empty)


def model_keywords(cls: type) -> list[str] | None:
    """The keywords that set a pydantic model's fields: names and aliases, in field order.

    None when the model accepts any keyword (``extra="allow"``).
    """
    config = getattr(cls, "model_config", None)
    if isinstance(config, dict) and config.get("extra") == "allow":
        return None
    names: list[str] = []
    for name, info in (getattr(cls, "model_fields", None) or {}).items():
        for keyword in (name, getattr(info, "alias", None)):
            if isinstance(keyword, str) and keyword and keyword not in names:
                names.append(keyword)
    return names


def format_signature(obj: Any, name: str, *, skip_self: bool = False, width: int = 90) -> str:
    """``name(params) -> returns``, one parameter per line when it is long."""
    try:
        signature = call_signature(obj)
    except (TypeError, ValueError):
        return f"{name}(...)"
    parameters = _public_parameters(signature)
    if skip_self and parameters and parameters[0].name in ("self", "cls"):
        parameters = parameters[1:]
    pieces = _parameter_pieces(parameters)
    returns = _annotation_text(signature.return_annotation)
    suffix = f" -> {returns}" if returns and not inspect.isclass(obj) else ""
    one_line = f"{name}({', '.join(pieces)}){suffix}"
    if len(one_line) <= width:
        return one_line
    body = "".join(f"    {piece},\n" for piece in pieces)
    return f"{name}(\n{body}){suffix}"


def compact_signature(obj: Any, *, skip_self: bool = False, width: int = 110) -> str:
    """``(params) -> returns`` on one line, shortened to ``width`` characters."""
    try:
        signature = call_signature(obj)
    except (TypeError, ValueError):
        return "(...)"
    parameters = _public_parameters(signature)
    if skip_self and parameters and parameters[0].name in ("self", "cls"):
        parameters = parameters[1:]
    text = f"({', '.join(_parameter_pieces(parameters))})"
    returns = _annotation_text(signature.return_annotation)
    if returns and not inspect.isclass(obj):
        text += f" -> {returns}"
    if len(text) > width:
        names: list[str] = []
        star = False
        for parameter in parameters:
            if parameter.kind is parameter.KEYWORD_ONLY and not star:
                names.append("*")
                star = True
            if parameter.kind is parameter.VAR_POSITIONAL:
                star = True
            names.append(_param_name(parameter))
        text = f"({', '.join(names)})"
        if returns and not inspect.isclass(obj):
            text += f" -> {returns}"
    if len(text) > width:
        text = text[: width - 4] + "...)"
    return text


def _param_name(parameter: inspect.Parameter) -> str:
    if parameter.kind is parameter.VAR_POSITIONAL:
        return f"*{parameter.name}"
    if parameter.kind is parameter.VAR_KEYWORD:
        return f"**{parameter.name}"
    return parameter.name


def _parameter_pieces(parameters: list[inspect.Parameter]) -> list[str]:
    pieces: list[str] = []
    star_done = False
    for index, parameter in enumerate(parameters):
        kind = parameter.kind
        if kind is parameter.KEYWORD_ONLY and not star_done:
            pieces.append("*")
            star_done = True
        if kind is parameter.VAR_POSITIONAL:
            star_done = True
        text = _param_name(parameter)
        annotation = _annotation_text(parameter.annotation)
        if annotation:
            text += f": {annotation}"
        if parameter.default is not parameter.empty:
            default = short_repr(parameter.default, limit=60)
            text += f" = {default}" if annotation else f"={default}"
        pieces.append(text)
        is_last_positional_only = kind is parameter.POSITIONAL_ONLY and (
            index + 1 == len(parameters)
            or parameters[index + 1].kind is not parameter.POSITIONAL_ONLY
        )
        if is_last_positional_only:
            pieces.append("/")
    return pieces


def _annotation_text(annotation: Any) -> str:
    if annotation is inspect.Signature.empty or annotation is inspect.Parameter.empty:
        return ""
    text = annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)
    text = _TYPING_PREFIX.sub("", text)
    text = _QAGENTS_PREFIX.sub("", text)
    return _drop_annotated(text)


def _drop_annotated(text: str) -> str:
    """``Annotated[X, FieldInfo(...)]`` written as ``X``: the metadata is noise for a reader."""
    while True:
        start = text.find("Annotated[")
        if start < 0:
            return text
        depth = 0
        first_comma = None
        end = None
        for index in range(start + len("Annotated["), len(text)):
            char = text[index]
            if char in "[(":
                depth += 1
            elif char in "])":
                if depth == 0:
                    end = index
                    break
                depth -= 1
            elif char == "," and depth == 0 and first_comma is None:
                first_comma = index
        if end is None:
            return text
        inner = text[start + len("Annotated[") : first_comma if first_comma else end]
        text = text[:start] + inner.strip() + text[end + 1 :]


def short_repr(value: Any, *, limit: int = 200) -> str:
    """``repr`` with class and function addresses removed, shortened to ``limit``."""
    if inspect.isclass(value):
        text = value.__name__
    elif inspect.isroutine(value):
        text = getattr(value, "__qualname__", repr(value))
    else:
        try:
            text = repr(value)
        except Exception:  # noqa: BLE001 - a broken __repr__ must not break the answer
            text = f"<{type(value).__name__}>"
    text = _CLASS_REPR.sub(r"\1", text)
    text = _FUNCTION_REPR.sub(r"\1", text)
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def own_doc(obj: Any) -> str:
    """The docstring of ``obj``, inherited only from qagents classes, never from pydantic."""
    if inspect.isclass(obj):
        for klass in obj.__mro__:
            if not klass.__module__.startswith("qagents"):
                return ""
            doc = klass.__dict__.get("__doc__")
            if isinstance(doc, str) and doc.strip():
                return inspect.cleandoc(doc)
        return ""
    return inspect.getdoc(obj) or ""


def _has_example(doc: str) -> bool:
    """Whether a docstring shows code: a fenced block or an indented block."""
    return "```" in doc or any(
        line.startswith("    ") and line.strip() for line in doc.splitlines()
    )


def _is_pydantic_model(cls: type) -> bool:
    return isinstance(getattr(cls, "model_fields", None), dict) and hasattr(cls, "model_dump")


def _first_sentence(doc: str, limit: int = 160) -> str:
    text = " ".join(doc.strip().split("\n\n", 1)[0].split())
    sentence = re.split(r"(?<=[.!?])\s+(?=[A-Z`])", text, maxsplit=1)[0]
    if len(sentence) > limit:
        sentence = sentence[: limit - 3].rsplit(" ", 1)[0] + "..."
    return sentence


def _doc_of(value: Any, lookup: ApiLookup, module: ModuleType, name: str) -> str:
    if value is None or not (
        inspect.isclass(value) or inspect.isroutine(value) or inspect.ismodule(value)
    ):
        entry = lookup.entry(f"{module.__name__}.{name}")
        if entry is None and value is not None:
            entry = lookup.module_attribute_entry(module, name, value)
        if entry is not None:
            return entry.doc
        if value is None:
            return ""
    if inspect.isclass(value) or inspect.isroutine(value) or inspect.ismodule(value):
        return own_doc(value)
    return ""


def _is_async(function: Any) -> bool:
    return inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function)


def _static_lookup(cls: type | None, name: str | None) -> Any:
    if cls is None or name is None:
        return _MISSING
    for klass in cls.__mro__:
        if name in vars(klass):
            return vars(klass)[name]
    return _MISSING


def _class_annotations(cls: type) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
        found.update(getattr(klass, "__annotations__", {}) or {})
    return found


def _defined_in_qagents(cls: type, name: str) -> bool:
    return _field_origin(cls, name) is not None


def _field_origin(cls: type, name: str) -> type | None:
    """The qagents class in the MRO that declares field ``name``, or None."""
    for klass in cls.__mro__:
        if name in (getattr(klass, "__annotations__", {}) or {}):
            return klass if klass.__module__.startswith("qagents") else None
    return cls if cls.__module__.startswith("qagents") else None


def _from(origin: type, cls: type) -> str:
    return "" if origin is cls else f" (from `{origin.__name__}`)"


def _field_type(info: Any) -> str:
    annotation = getattr(info, "annotation", None)
    return _annotation_text(annotation) if annotation is not None else "Any"


def _field_default(info: Any) -> str:
    factory = getattr(info, "default_factory", None)
    if factory is not None:
        return f" = {short_repr(factory)}()"
    default = getattr(info, "default", None)
    try:
        from pydantic_core import PydanticUndefined
    except ImportError:  # pragma: no cover - pydantic is a dependency
        PydanticUndefined = None  # noqa: N806
    if default is PydanticUndefined:
        return ""
    return f" = {short_repr(default, limit=60)}"


def _return_hint(function: Any) -> str:
    try:
        annotation = inspect.signature(function).return_annotation
    except (TypeError, ValueError):
        return ""
    return _annotation_text(annotation)


def _module_of(target: Target) -> str:
    if target.entry is not None:
        path = target.entry.path
    elif target.obj is not None and inspect.ismodule(target.obj):
        return target.obj.__name__
    else:
        source = target.owner if target.owner is not None else target.obj
        path = getattr(source, "__module__", "") or target.path
    for module in sorted(EXTRAS, key=len, reverse=True):
        if path == module or path.startswith(f"{module}."):
            return module
    return path


def _extra_for(module: str) -> str | None:
    for name, extra in EXTRAS.items():
        if module == name or module.startswith(f"{name}."):
            return extra
    return None


def _display_file(file: Path) -> str:
    parts = file.resolve().parts
    if "qagents" in parts:
        index = len(parts) - 1 - parts[::-1].index("qagents")
        prefix = ("src",) if index > 0 and parts[index - 1] == "src" else ()
        return "/".join([*prefix, *parts[index:]])
    return file.as_posix()


def _callable_name(function: Any) -> str:
    module = getattr(function, "__module__", "") or ""
    qualname = getattr(function, "__qualname__", "") or getattr(function, "__name__", "")
    return f"{module}.{qualname}".removeprefix("qagents.agent.").strip(".")


def _unique_entries(entries: list[ApiEntry]) -> list[ApiEntry]:
    seen: set[str] = set()
    unique: list[ApiEntry] = []
    for entry in entries:
        if entry.path not in seen:
            seen.add(entry.path)
            unique.append(entry)
    return unique


def instance_attributes(cls: type) -> list[tuple[str, str, str]]:
    """Public attributes ``__init__`` of ``cls`` sets on ``self``: name, annotation, docstring.

    A string literal right after the assignment is its docstring, as for class attributes.
    """
    import ast
    import textwrap

    init = cls.__dict__.get("__init__")
    if init is None or not inspect.isfunction(init):
        return []
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(init)))
    except (OSError, TypeError, SyntaxError):
        return []
    found: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    body = tree.body[0].body if tree.body and hasattr(tree.body[0], "body") else []
    for index, node in enumerate(body):
        target = None
        annotation = ""
        if isinstance(node, ast.AnnAssign):
            target = node.target
            annotation = ast.unparse(node.annotation)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and not target.attr.startswith("_")
            and target.attr not in seen
        ):
            continue
        doc = ""
        following = body[index + 1] if index + 1 < len(body) else None
        if (
            isinstance(following, ast.Expr)
            and isinstance(following.value, ast.Constant)
            and isinstance(following.value.value, str)
        ):
            doc = inspect.cleandoc(following.value.value)
        seen.add(target.attr)
        found.append((target.attr, _drop_annotated(annotation), doc))
    return found


def resolve_annotation(function: Any, annotation: Any) -> Any:
    """Evaluate a string annotation of ``function`` in its module; None when it fails."""
    if not isinstance(annotation, str):
        return annotation
    namespace = dict(getattr(function, "__globals__", {}) or {})
    try:
        import qagents

        namespace.setdefault("qagents", qagents)
        for name in getattr(qagents, "__all__", []):
            namespace.setdefault(name, getattr(qagents, name))
    except ImportError:  # pragma: no cover
        pass
    try:
        return eval(annotation, namespace)  # noqa: S307 - annotations of the installed package
    except Exception:  # noqa: BLE001 - forward references that cannot be resolved
        return None


def return_type(function: Any) -> Any:
    """The class a call of ``function`` returns, or None when it is not a single class."""
    try:
        annotation = inspect.signature(function).return_annotation
    except (TypeError, ValueError):
        return None
    if annotation is inspect.Signature.empty:
        return None
    value = resolve_annotation(_unwrap(function), annotation)
    return single_class(value)


def single_class(annotation: Any) -> type | None:
    """The one class an annotation names, ignoring ``None`` in a union; else None."""
    if annotation is None:
        return None
    if inspect.isclass(annotation):
        return annotation
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        classes = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        return single_class(classes[0]) if len(classes) == 1 else None
    if origin is typing.Annotated:
        return single_class(typing.get_args(annotation)[0])
    if inspect.isclass(origin) and origin.__module__.startswith("qagents"):
        return origin
    return None


def element_class(annotation: Any) -> type | None:
    """The item class of ``list[X]``, ``Sequence[X]`` and similar; else None."""
    if annotation is None:
        return None
    args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        for arg in args:
            found = element_class(arg)
            if found is not None:
                return found
        return None
    if origin in (list, tuple, set, frozenset) or (
        origin is not None and getattr(origin, "__module__", "") == "collections.abc"
    ):
        return single_class(args[0]) if args else None
    return None


def _unwrap(function: Any) -> Any:
    return getattr(function, "__func__", function)


_MAX_FORWARDS = 5
"""How many hops of ``**kwargs`` forwarding `forwarded_parameters` follows."""


class _ModelFields:
    """The receiver of ``super().__init__(**data)`` in a pydantic model: its fields."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "MODEL_FIELDS"


MODEL_FIELDS: Any = _ModelFields()
"""What `forwarding_target` returns when ``**kwargs`` reach pydantic's ``BaseModel.__init__``
from a qagents model's own ``__init__``: they set the model's fields by name."""


def forwarding_target(function: Any, owner: type | None) -> Any:
    """The callable that receives ``function``'s ``**kwargs``, when it passes them on.

    ``Agent.run_sync(prompt, **options)`` calls ``self.run(prompt, **options)``, so its
    keyword arguments are those of `Agent.run`. Followed: ``self.method(**kwargs)``,
    ``cls(**kwargs)`` in a classmethod (the constructor), ``super().method(**kwargs)`` (the
    method of the next qagents class in ``owner``'s MRO that defines it),
    ``parameter.method(**kwargs)`` with an annotated parameter, and ``function(**kwargs)`` for
    a function of the same module. For a class, its ``__init__`` is read: ``DevviAgent(...)``
    passes ``**options`` to ``ExternalAgent.__init__``. A pydantic model's own ``__init__``
    that passes ``**data`` to ``super().__init__`` gives `MODEL_FIELDS`.
    """
    found = _forwarding(function, owner)
    return found[0] if found is not None else None


def forwarded_parameters(function: Any, owner: type | None) -> list[inspect.Parameter] | None:
    """The keyword parameters that ``function``'s ``**kwargs`` reach, following every hop.

    ``DevviACPAgent``-style chains (``**options`` passed on to a method that passes its own
    ``**options`` on again) are followed up to `_MAX_FORWARDS` hops. None when ``function``
    takes no ``**kwargs``, or when the chain ends in a ``**kwargs`` that cannot be followed,
    so any keyword may be accepted.
    """
    collected: dict[str, inspect.Parameter] = {}
    for _ in range(_MAX_FORWARDS):
        found = _forwarding(function, owner)
        if found is None:
            return None
        function, owner = found
        if function is MODEL_FIELDS:
            keywords = model_keywords(owner) if owner is not None else None
            if keywords is None:
                return None
            for name in keywords:
                collected.setdefault(
                    name, inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
                )
            return list(collected.values())
        try:
            parameters = list(call_signature(function).parameters.values())
        except (TypeError, ValueError):
            return None
        for parameter in parameters:
            if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
                if parameter.name != "self":
                    collected.setdefault(parameter.name, parameter)
        if not any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters):
            return list(collected.values())
    return None


def _forwarding(function: Any, owner: type | None) -> tuple[Any, type | None] | None:
    """`forwarding_target`, with the class the next hop resolves ``self`` and ``super()``
    against."""
    import ast
    import textwrap

    if inspect.isclass(function):
        owner = function
        function = _own_init(function)
        if function is None:
            return None
    if owner is None and inspect.ismethod(function) and inspect.isclass(function.__self__):
        owner = function.__self__  # a classmethod read from its class
    function = _unwrap(function)
    try:
        signature = inspect.signature(function)
        source = textwrap.dedent(inspect.getsource(function))
        tree = ast.parse(source)
    except (OSError, TypeError, ValueError, SyntaxError):
        return None
    varkw = next((p.name for p in signature.parameters.values() if p.kind is p.VAR_KEYWORD), None)
    if varkw is None:
        return None
    namespace = getattr(function, "__globals__", {}) or {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not any(
            k.arg is None and isinstance(k.value, ast.Name) and k.value.id == varkw
            for k in node.keywords
        ):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            if func.id == "cls" and owner is not None:
                return owner, owner  # the constructor
            value = namespace.get(func.id)
            if callable(value) and getattr(value, "__module__", "").startswith("qagents"):
                return value, None
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call):
            start = _super_start(func.value, namespace)
            if start is not None and owner is not None:
                found = _next_in_mro(owner, function, func.attr, start)
                return (found, owner) if found is not None else None
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            base = func.value.id
            if base in ("self", "cls") and owner is not None:
                method = getattr(owner, func.attr, None)
                return (method, owner) if method is not None else None
            parameter = signature.parameters.get(base)
            if parameter is not None and parameter.annotation is not parameter.empty:
                cls = single_class(resolve_annotation(function, parameter.annotation))
                if cls is not None:
                    method = getattr(cls, func.attr, None)
                    return (method, cls) if method is not None else None
    return None


def _own_init(cls: type) -> Any:
    """The ``__init__`` a call of ``cls`` runs, when a qagents class defines it in Python."""
    for klass in cls.__mro__:
        init = vars(klass).get("__init__")
        if init is None:
            continue
        if klass.__module__.startswith("qagents") and inspect.isfunction(init):
            return init
        return None
    return None


_ZERO_ARGUMENT_SUPER = object()
"""`_super_start` for ``super()``: the class that defines the calling method."""


def _super_start(node: Any, namespace: dict[str, Any]) -> Any:
    """For ``super()`` or ``super(Class, self)``, where the MRO search starts; else None."""
    import ast

    if not (isinstance(node.func, ast.Name) and node.func.id == "super"):
        return None
    if not node.args:
        return _ZERO_ARGUMENT_SUPER
    first = node.args[0]
    if isinstance(first, ast.Name):
        value = namespace.get(first.id)
        if inspect.isclass(value):
            return value
    return None


def _next_in_mro(owner: type, function: Any, name: str, start: Any) -> Any:
    """What ``super().<name>`` is inside ``function`` for an instance of ``owner``.

    Only a qagents class counts: ``**kwargs`` passed on to ``object`` are not followed.
    ``super().__init__(**data)`` that reaches pydantic's ``BaseModel.__init__`` gives
    `MODEL_FIELDS`: the keywords set the model's fields.
    """
    mro = owner.__mro__
    if start is _ZERO_ARGUMENT_SUPER:
        start = next((klass for klass in mro if _defines(klass, function)), None)
    if start is None or start not in mro:
        return None
    for klass in mro[mro.index(start) + 1 :]:
        if name in vars(klass):
            if not klass.__module__.startswith("qagents"):
                if name == "__init__" and _is_pydantic_base(klass) and _is_pydantic_model(owner):
                    return MODEL_FIELDS
                return None
            return getattr(klass, name, None)
    return None


def _is_pydantic_base(klass: type) -> bool:
    try:
        from pydantic import BaseModel
    except ImportError:  # pragma: no cover - pydantic is a dependency
        return False
    return klass is BaseModel


def _defines(klass: type, function: Any) -> bool:
    raw = vars(klass).get(getattr(function, "__name__", ""))
    return raw is not None and getattr(raw, "__func__", raw) is function


def _public_parameters(signature: inspect.Signature) -> list[inspect.Parameter]:
    """The parameters a caller may pass: those whose names do not start with ``_``."""
    return [p for p in signature.parameters.values() if not p.name.startswith("_")]
