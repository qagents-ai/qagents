"""Record a model's responses to a file once, then replay them without the model.


    from qagents.models.recording import cassette

    model = cassette(QCloudModel("<model-id>"), "tests/cassettes/weather.json")
    result = await Agent(model, tools=[get_weather]).run("Weather in Athens?")

The first run sends real requests and writes every response to the cassette.
Every later run finds the file and replays it: no network, no cost, the same
answers streamed in the same chunks. Delete the file, or pass
``mode="record"``, to record it again.

A cassette is one JSON file with sorted keys, so it reads well in a diff:

    {
      "format": "qagents-cassette",
      "interactions": [
        {
          "events": [{"text": "It is ", "type": "text_delta"}, ...],
          "fingerprint": "5f0c...",
          "kind": "response",
          "request": {"messages": [{"digest": "...", "excerpt": "...", "role": "user"}], ...},
          "response": {"finish_reason": "stop", "message": {...}, "usage": {...}, ...}
        }
      ],
      "model": "<model-id>",
      "provider": "qcloud",
      "version": 1
    }

Responses are looked up by the request's fingerprint (see `request_fingerprint`),
so a cassette keeps working when unrelated requests are added or reordered, and
fails loudly when the conversation, the tools, the settings or the output schema
change. The cassette holds what the model received and answered as qagents sees
it; it never sees HTTP headers or the keys given to a provider adapter
(``api_key=``, ``headers=``), which is where credentials belong.

Values in ``ModelSettings.extra`` are stored as ``"<redacted>"`` when their key
looks secret (see `is_secret_setting_key`) or the value itself looks like a
credential. That is a best-effort guess from names: pass ``redact_keys=`` for
the names your deployment uses, and read a new cassette before committing it.

Requests that fail are not recorded: a replay of a run that needed a retry
answers on the first attempt.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter, ValidationError

from ..errors import CassetteMissError, ConfigurationError
from .base import (
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    ModelStreamEvent,
    ReasoningDelta,
    ResponseDone,
    TextDelta,
    ToolCallDelta,
    resolve_model,
)

logger = logging.getLogger("qagents.recording")

CASSETTE_FORMAT = "qagents-cassette"
"""Value of the ``format`` key that marks a JSON file as a cassette."""

CASSETTE_VERSION = 1
"""Version of the cassette layout written by this module. Older versions are read."""

CASSETTE_MODE_ENV = "QAGENTS_CASSETTE_MODE"
"""Environment variable read by `cassette`.

Set it to ``replay`` in CI so that no cassette is recorded: a missing cassette
fails instead of making a paid request, and ``mode="record"`` is refused. Set it
to ``record`` to record every ``mode="auto"`` cassette again. `RecordingModel`
and `ReplayModel` do not read it."""

SECRET_SETTING_KEYS = re.compile(
    r"(^|_)(secrets?|password|passwd|passphrase|pwd|credentials?|token|auth|authorization"
    r"|authentication|bearer|cookies?|signature|private|apikey|headers?|keys?)(_|$)"
)
"""Matched against a key of ``ModelSettings.extra`` (at any depth) in snake_case
(``clientSecret`` as ``client_secret``): any of these words marks it as secret,
so ``key``, ``secret_key``, ``authToken`` and ``aws_secret_access_key`` all do.
Its value is replaced by `REDACTED` before the request is fingerprinted, so a
changed secret does not change the fingerprint. Over-redacting is the safe side;
`PUBLIC_SETTING_KEYS` lists the exceptions."""

PUBLIC_SETTING_KEYS = frozenset({"prompt_cache_key", "cache_key", "idempotency_key"})
"""Keys that `SECRET_SETTING_KEYS` would match but that are documented request
fields, not secrets. They stay readable and part of the fingerprint."""

SECRET_VALUE = re.compile(r'^(?:sk-|AKIA|-----BEGIN|[Bb]earer\s)|"private_key"')
"""String values in ``ModelSettings.extra`` redacted whatever their key: API keys,
AWS key ids, PEM blocks, bearer tokens and service-account JSON."""

REDACTED = "<redacted>"
"""What a cassette stores instead of a secret-looking value."""

EXCERPT_CHARS = 120
"""Length of the text excerpt stored for each message of a recorded request."""

CassetteMode = Literal["auto", "record", "replay"]
"""``"auto"`` replays when the cassette file exists and records otherwise."""

_EventAdapter: TypeAdapter[TextDelta | ReasoningDelta | ToolCallDelta] = TypeAdapter(
    Annotated[TextDelta | ReasoningDelta | ToolCallDelta, Field(discriminator="type")]
)
_EVENT_TYPES = frozenset({"text_delta", "reasoning_delta", "tool_call_delta"})


def request_fingerprint(request: ModelRequest, *, redact_keys: Iterable[str] = ()) -> str:
    """The key a cassette stores a response under.

    It is the SHA-256 of the canonical JSON (sorted keys, no whitespace) of the
    request's messages, tools, settings and output spec, with three
    normalisations:

    * ``metadata`` of messages and tool results is left out, because no
      provider receives it;
    * fields whose value is ``None`` are left out, so a new optional field in a
      later qagents version does not change fingerprints;
    * secret-looking values in ``settings.extra`` (see `is_secret_setting_key`
      and `SECRET_VALUE`) are replaced by `REDACTED`.

    Args:
        request: The request the agent is about to send.
        redact_keys: More keys of ``settings.extra`` to redact, as in `cassette`.

    Returns:
        A 64-character hex digest.
    """
    return _digest(_canonical(request, _names(redact_keys)))


def is_secret_setting_key(key: str, redact_keys: Iterable[str] = ()) -> bool:
    """Whether the value under ``key`` in ``ModelSettings.extra`` is kept out of cassettes.

    Args:
        key: A key at any depth of ``ModelSettings.extra``.
        redact_keys: More keys to treat as secret, compared in snake_case.

    Returns:
        ``True`` when `SECRET_SETTING_KEYS` matches the key in snake_case and it
        is not in `PUBLIC_SETTING_KEYS`, or when it is one of ``redact_keys``.
    """
    name = _snake(key)
    if name in _names(redact_keys):
        return True
    return name not in PUBLIC_SETTING_KEYS and SECRET_SETTING_KEYS.search(name) is not None


class RecordingModel(Model):
    """Passes every request to ``inner`` and appends what it answered to a cassette.

    An interaction is written when the response is complete, before the
    `ResponseDone` event is passed on. Requests that fail or are cancelled
    are not written. Several requests may run at the same time, through one
    recording model or through several that share a path: every write reads
    the file, appends and replaces the file atomically under a per-file lock.

    Args:
        inner: The real model. ``"provider:model-id"`` strings are resolved.
        path: The cassette file. Missing parent directories are created.
        overwrite: Start a new cassette. The file found now is replaced when
            this model records its first response, so a run that fails before
            that leaves it in place. If the file has changed by then (another
            model on the same path wrote first), this model appends to it
            instead. Without it, new interactions are appended.
        redact_keys: More keys of ``ModelSettings.extra`` whose values are
            never written (see `is_secret_setting_key`). Pass the same names
            to the `ReplayModel`, since they are left out of the fingerprint.

    Raises:
        ConfigurationError: ``path`` exists but is not a cassette (with
            ``overwrite``), or not one this version can read (without it).
    """

    def __init__(
        self,
        inner: Model | str,
        path: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        redact_keys: Iterable[str] = (),
    ) -> None:
        self.inner = resolve_model(inner)
        """The model that answers."""
        self.path = Path(path)
        """The cassette file."""
        self.name = self.inner.name
        self.provider = self.inner.provider
        self._lock = _file_lock(self.path)
        self._redact = _names(redact_keys)
        self._capabilities: dict[str, Any] | None = None
        self._replace = overwrite  # this model's first write replaces the cassette it found
        self._found: tuple[int, int, int] | None = None
        with self._lock:
            if overwrite:
                _refuse_foreign(self.path)  # never replace a file that is not a cassette
                self._found = _identity(self.path)
            else:
                _read(self.path)  # an unreadable file fails now, not after a paid request

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        canonical = _canonical(request, self._redact)
        events: list[dict[str, Any]] = []
        async with contextlib.aclosing(self.inner.stream(request)) as source:
            async for event in source:
                if isinstance(event, ResponseDone):
                    response = _with_origin(event.response, self.inner)
                    self._save(
                        {
                            "kind": "response",
                            "fingerprint": _digest(canonical),
                            "request": _describe(canonical),
                            "events": events,
                            "response": response.model_dump(mode="json"),
                        }
                    )
                    yield ResponseDone(response=response)
                    continue
                events.append(event.model_dump(mode="json"))
                yield event

    async def capabilities(self) -> ModelCapabilities:
        """The inner model's capabilities, also stored in the cassette for replay."""
        capabilities = await self.inner.capabilities()
        dumped = capabilities.model_dump(mode="json")
        if dumped != self._capabilities:
            self._capabilities = dumped
            with self._lock:
                # A cassette about to be replaced, or none yet: written with the first response.
                data = None if self._replaces_now() else _read(self.path)
                if data is not None:
                    data["capabilities"] = dumped
                    _write(self.path, data)
        return capabilities

    def validate_settings(self, settings: ModelSettings) -> None:
        self.inner.validate_settings(settings)

    async def aclose(self) -> None:
        await self.inner.aclose()

    def _replaces_now(self) -> bool:
        """Whether the next write starts a new cassette: ``overwrite``, nothing
        written by this model yet, and the file is still the one it found (a
        model sharing the path may have started the new cassette already)."""
        return self._replace and _identity(self.path) == self._found

    def _save(self, interaction: dict[str, Any]) -> None:
        with self._lock:
            data = None if self._replaces_now() else _read(self.path)
            self._replace = False
            data = data or {
                "format": CASSETTE_FORMAT,
                "version": CASSETTE_VERSION,
                "model": self.inner.name,
                "provider": self.inner.provider,
                "interactions": [],
            }
            data["interactions"].append(interaction)
            if self._capabilities is not None:
                data["capabilities"] = self._capabilities
            _write(self.path, data)

    def __repr__(self) -> str:
        return f"RecordingModel({self.inner.provider}:{self.inner.name}, path={str(self.path)!r})"


@dataclass
class _Recorded:
    position: int
    fingerprint: str
    request: dict[str, Any]
    events: list[TextDelta | ReasoningDelta | ToolCallDelta]
    response: ModelResponse


class ReplayModel(Model):
    """Answers from a cassette instead of a provider.

    Each request is matched by `request_fingerprint`. When the cassette holds
    the same request several times, the recordings are served in the order
    they were recorded, each one once. The recorded text, reasoning and
    tool-call deltas are streamed again, then the recorded response. ``name``,
    ``provider`` and every response's ``model`` and ``provider`` are the
    recorded ones.

    Every request is kept in ``requests``, as with `ScriptedModel`.

    Args:
        path: The cassette file, read once now.
        strict: When no recording matches, raise `CassetteMissError`. With
            ``False``, serve the earliest recording not served yet instead,
            and log a warning; use it while a request changes on purpose
            between runs (a timestamp in the instructions, for example).
        redact_keys: The ``redact_keys`` the cassette was recorded with.

    Raises:
        ConfigurationError: The file does not exist or is not a cassette this
            version can read.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        strict: bool = True,
        redact_keys: Iterable[str] = (),
    ) -> None:
        self.path = Path(path)
        """The cassette file."""
        self.strict = strict
        """Whether a request without a matching recording fails."""
        self._redact = _names(redact_keys)
        self.requests: list[ModelRequest] = []
        """Every request received, in order."""
        data = _read(self.path)
        if data is None:
            raise ConfigurationError(
                f"There is no cassette at {self.path}. Record it first with "
                "cassette(model, path, mode='record') or RecordingModel(model, path)."
            )
        self.name = str(data.get("model") or "replay")
        self.provider = str(data.get("provider") or "replay")
        try:
            self._capabilities = ModelCapabilities.model_validate(data.get("capabilities") or {})
        except ValidationError as error:
            raise ConfigurationError(
                f"The capabilities stored in {self.path} are damaged ({error}). "
                "Record the cassette again."
            ) from error
        self._recorded = _parse_interactions(self.path, data["interactions"])
        self._served = [False] * len(self._recorded)
        self._by_fingerprint: dict[str, collections.deque[int]] = collections.defaultdict(
            collections.deque
        )
        for index, recorded in enumerate(self._recorded):
            self._by_fingerprint[recorded.fingerprint].append(index)

    @property
    def remaining(self) -> int:
        """How many recordings have not been served yet."""
        return self._served.count(False)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        recorded = self._take(request)
        for event in recorded.events:
            yield event.model_copy(deep=True)
        yield ResponseDone(response=recorded.response.model_copy(deep=True))

    async def capabilities(self) -> ModelCapabilities:
        """The capabilities stored when the cassette was recorded; unknown if none were."""
        return self._capabilities.model_copy()

    def _take(self, request: ModelRequest) -> _Recorded:
        canonical = _canonical(request, self._redact)
        fingerprint = _digest(canonical)
        queue = self._by_fingerprint.get(fingerprint)
        while queue:
            index = queue.popleft()
            if not self._served[index]:
                self._served[index] = True
                return self._recorded[index]
        if not self.strict:
            index = next((i for i, served in enumerate(self._served) if not served), None)
            if index is not None:
                logger.warning(
                    "No recording in %s matches request %s; serving recording #%d instead "
                    "because strict=False.",
                    self.path,
                    fingerprint[:16],
                    self._recorded[index].position,
                )
                self._served[index] = True
                return self._recorded[index]
        raise self._miss(fingerprint, canonical)

    def _miss(self, fingerprint: str, canonical: dict[str, Any]) -> CassetteMissError:
        details: dict[str, Any] = {
            "fingerprint": fingerprint,
            "path": self.path,
            "provider": self.provider,
            "model": self.name,
        }
        again = (
            "If the change is intended, record the cassette again with "
            "cassette(model, path, mode='record')."
        )
        same = [item for item in self._recorded if item.fingerprint == fingerprint]
        if same:
            return CassetteMissError(
                f"The request {fingerprint} is recorded {len(same)} time(s) in {self.path}, and "
                "every recording of it has been served already: the agent sent it more often "
                f"than when the cassette was recorded. {again}",
                closest_fingerprint=fingerprint,
                **details,
            )
        if not self._recorded:
            return CassetteMissError(
                f"No recording matches the request {fingerprint}: the cassette {self.path} "
                f"has no recorded responses. {again}",
                **details,
            )
        requested = _describe(canonical)
        closest = max(self._recorded, key=lambda item: _closeness(item.request, requested))
        return CassetteMissError(
            f"No recording in {self.path} matches the request {fingerprint}. The closest "
            f"recorded request is #{closest.position} ({closest.fingerprint}), which differs "
            f"from it as follows: {_difference(closest.request, requested)}. {again} "
            "Use ReplayModel(path, strict=False) to serve recordings in order instead.",
            closest_fingerprint=closest.fingerprint,
            **details,
        )

    def __repr__(self) -> str:
        return f"ReplayModel({str(self.path)!r}, strict={self.strict})"


def cassette(
    inner: Model | str | None,
    path: str | os.PathLike[str],
    mode: CassetteMode = "auto",
    *,
    strict: bool = True,
    redact_keys: Iterable[str] = (),
) -> Model:
    """Record ``inner`` to ``path``, or replay ``path`` without calling ``inner``.

        model = cassette(QCloudModel("<model-id>"), "tests/cassettes/refund.json")

    Args:
        inner: The real model, used only when recording. It may be ``None``
            when the cassette is only ever replayed.
        path: The cassette file.
        mode: ``"auto"`` replays when the file exists and records otherwise;
            the `CASSETTE_MODE_ENV` environment variable overrides it.
            ``"record"`` records a new cassette, which replaces the existing
            one when its first response is recorded. ``"replay"`` only replays.
        strict: Passed to `ReplayModel`.
        redact_keys: More keys of ``ModelSettings.extra`` whose values are
            never written to the cassette (see `is_secret_setting_key`).

    Returns:
        A `ReplayModel` or a `RecordingModel`.

    Raises:
        ConfigurationError: The mode is unknown, recording was chosen without a
            model, or ``mode="record"`` while `CASSETTE_MODE_ENV` is ``replay``.
    """
    chosen = _checked_mode(mode, "mode")
    override = os.environ.get(CASSETTE_MODE_ENV, "").strip().lower()
    if override:
        forced = _checked_mode(override, CASSETTE_MODE_ENV)
        if chosen == "auto":
            chosen = forced
        elif chosen == "record" and forced == "replay":
            raise ConfigurationError(
                f"{CASSETTE_MODE_ENV}=replay forbids recording, and cassette(model, "
                f"{str(path)!r}, mode='record') would send real, possibly paid, requests. "
                f"Unset {CASSETTE_MODE_ENV} to record the cassette again."
            )
    if chosen == "auto":
        chosen = "replay" if Path(path).exists() else "record"
    if chosen == "replay":
        return ReplayModel(path, strict=strict, redact_keys=redact_keys)
    if inner is None:
        raise ConfigurationError(
            f"The cassette {path} has to be recorded, which needs the real model: "
            "cassette(model, path). To only replay, record the cassette first."
        )
    return RecordingModel(inner, path, overwrite=True, redact_keys=redact_keys)


# -- request canonical form ------------------------------------------------------------


def _canonical(request: ModelRequest, redact_keys: frozenset[str] = frozenset()) -> dict[str, Any]:
    data = request.model_dump(mode="json")
    messages = []
    for message in data["messages"]:
        message.pop("metadata", None)
        parts = []
        for part in message.get("parts", []):
            if part.get("type") == "tool_result":
                part.pop("metadata", None)
            parts.append(_without_none(part))
        message["parts"] = parts
        messages.append(_without_none(message))
    settings = _without_none(data["settings"])
    if settings.get("extra"):
        settings["extra"] = _redacted(settings["extra"], redact_keys)
    canonical: dict[str, Any] = {
        "messages": messages,
        "tools": [_without_none(spec) for spec in data["tools"]],
        "settings": settings,
    }
    if data.get("output") is not None:
        canonical["output"] = _without_none(data["output"])
    return canonical


def _without_none(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if value is not None}


def _redacted(value: Any, redact_keys: frozenset[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: REDACTED
            if is_secret_setting_key(str(key), redact_keys)
            else _redacted(item, redact_keys)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redacted(item, redact_keys) for item in value]
    if isinstance(value, str) and SECRET_VALUE.search(value):
        return REDACTED
    return value


_WORDS = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+|[A-Z]+|\d+")


def _snake(key: str) -> str:
    """``key`` as lower-case words joined by ``_``: ``Ocp-Apim-Subscription-Key`` is
    ``ocp_apim_subscription_key`` and ``clientSecret`` is ``client_secret``."""
    return "_".join(word.lower() for word in _WORDS.findall(key))


def _names(keys: Iterable[str]) -> frozenset[str]:
    if isinstance(keys, str):
        keys = (keys,)
    return frozenset(_snake(key) for key in keys)


def _compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_compact(value).encode("utf-8")).hexdigest()


def _describe(canonical: dict[str, Any]) -> dict[str, Any]:
    """What a cassette stores about a request: enough to explain a miss, not a full copy."""
    output = canonical.get("output")
    messages = [
        {"role": message["role"], "digest": _digest(message)[:16], "excerpt": _excerpt(message)}
        for message in canonical["messages"]
    ]
    tools = [{"name": spec["name"], "digest": _digest(spec)[:16]} for spec in canonical["tools"]]
    return {
        "messages": messages,
        "tools": tools,
        "settings": canonical["settings"],
        "output": (
            None if output is None else {"name": output["name"], "digest": _digest(output)[:16]}
        ),
    }


def _excerpt(message: dict[str, Any]) -> str:
    pieces: list[str] = []
    for part in message.get("parts", []):
        kind = part.get("type")
        if kind == "text":
            pieces.append(str(part.get("text", "")))
        elif kind == "tool_call":
            pieces.append(f"{part.get('name')}({_compact(part.get('arguments', {}))})")
        elif kind == "tool_result":
            pieces.append(f"{part.get('name')} -> {part.get('content', '')}")
        else:
            pieces.append(f"[{kind}]")
    text = " ".join(" ".join(pieces).split())
    return text if len(text) <= EXCERPT_CHARS else f"{text[: EXCERPT_CHARS - 3]}..."


def _digests(description: dict[str, Any]) -> list[str]:
    return [str(message.get("digest")) for message in description.get("messages") or []]


def _common_prefix(left: list[str], right: list[str]) -> int:
    count = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        count += 1
    return count


def _closeness(recorded: dict[str, Any], requested: dict[str, Any]) -> tuple[int, int, int]:
    recorded_digests, requested_digests = _digests(recorded), _digests(requested)
    alike = sum(recorded.get(key) == requested.get(key) for key in ("tools", "settings", "output"))
    return (
        _common_prefix(recorded_digests, requested_digests),
        alike,
        -abs(len(recorded_digests) - len(requested_digests)),
    )


def _difference(recorded: dict[str, Any], requested: dict[str, Any]) -> str:
    notes: list[str] = []
    old, new = recorded.get("messages") or [], requested.get("messages") or []
    prefix = _common_prefix(_digests(recorded), _digests(requested))
    if prefix < len(old) and prefix < len(new):
        notes.append(
            f"message {prefix + 1} was {_shown(old[prefix])} and is now {_shown(new[prefix])}"
        )
    elif len(new) > len(old):
        notes.append(
            f"the request has {len(new) - len(old)} more message(s), the first of them "
            f"{_shown(new[prefix])}"
        )
    elif len(old) > len(new):
        notes.append(f"the request has {len(old) - len(new)} fewer message(s)")

    old_tools = {tool.get("name"): tool.get("digest") for tool in recorded.get("tools") or []}
    new_tools = {tool.get("name"): tool.get("digest") for tool in requested.get("tools") or []}
    if added := sorted(set(new_tools) - set(old_tools), key=str):
        notes.append(f"tools added: {', '.join(map(str, added))}")
    if removed := sorted(set(old_tools) - set(new_tools), key=str):
        notes.append(f"tools removed: {', '.join(map(str, removed))}")
    if changed := sorted(
        (name for name in set(old_tools) & set(new_tools) if old_tools[name] != new_tools[name]),
        key=str,
    ):
        notes.append(f"tool definitions changed: {', '.join(map(str, changed))}")

    old_settings, new_settings = recorded.get("settings") or {}, requested.get("settings") or {}
    if differing := sorted(
        key
        for key in set(old_settings) | set(new_settings)
        if old_settings.get(key) != new_settings.get(key)
    ):
        notes.append(f"settings changed: {', '.join(differing)}")
    if recorded.get("output") != requested.get("output"):
        notes.append("the output schema changed")
    return "; ".join(notes) or "no difference in messages, tools, settings or output"


def _shown(message: dict[str, Any]) -> str:
    return f"{message.get('role')} {json.dumps(message.get('excerpt', ''), ensure_ascii=False)}"


# -- the file ----------------------------------------------------------------------------

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _identity(path: Path) -> tuple[int, int, int] | None:
    """What tells this file apart from the next version `_write` puts in its place."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _lock_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _file_lock(path: Path) -> threading.Lock:
    """One lock per cassette file in this process, shared by every model using it."""
    key = _lock_key(path)
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def _refuse_foreign(path: Path) -> None:
    """Raise unless ``path`` is missing or marked as a cassette. Its content is not checked."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict) or data.get("format") != CASSETTE_FORMAT:
        raise ConfigurationError(
            f"{path} is not a qagents cassette (its 'format' is not {CASSETTE_FORMAT!r}), so it "
            "is not replaced. Point the model at a different file, or delete this one."
        )


def _read(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        data = json.loads(text)
    except ValueError as error:
        raise ConfigurationError(
            f"{path} is not valid JSON ({error}). Delete it and record the cassette again."
        ) from error
    if not isinstance(data, dict) or data.get("format") != CASSETTE_FORMAT:
        raise ConfigurationError(
            f"{path} is not a qagents cassette (its 'format' is not {CASSETTE_FORMAT!r}). "
            "Point the model at a different file, or delete this one."
        )
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ConfigurationError(f"{path} has no valid cassette version. Record it again.")
    if version > CASSETTE_VERSION:
        raise ConfigurationError(
            f"{path} was written in cassette version {version}, and this qagents reads up to "
            f"version {CASSETTE_VERSION}. Upgrade qagents, or record the cassette again."
        )
    if not isinstance(data.get("interactions"), list):
        raise ConfigurationError(f"{path} has no 'interactions' list. Record it again.")
    return data


def _write(path: Path, data: dict[str, Any]) -> None:
    """Replace ``path`` atomically: a reader sees the old file or the new one, never half."""
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(text, encoding="utf-8")
        try:
            for attempt in range(5):
                try:
                    os.replace(temporary, path)
                    break
                except PermissionError:
                    # Windows refuses the rename while another process (a virus
                    # scanner, an editor) has the file open; that passes quickly.
                    if attempt == 4:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        finally:
            temporary.unlink(missing_ok=True)
    except OSError as error:
        raise ConfigurationError(f"Could not write the cassette {path}: {error}") from error


def _parse_interactions(path: Path, entries: list[Any]) -> list[_Recorded]:
    recorded: list[_Recorded] = []
    for position, entry in enumerate(entries, start=1):
        if isinstance(entry, dict) and entry.get("kind", "response") != "response":
            continue  # a kind written by a later version
        try:
            if not isinstance(entry, dict):
                raise TypeError("an interaction must be a JSON object")
            fingerprint = entry["fingerprint"]
            if not isinstance(fingerprint, str):
                raise TypeError("'fingerprint' must be a string")
            events = [
                _EventAdapter.validate_python(event)
                for event in entry.get("events") or []
                if isinstance(event, dict) and event.get("type") in _EVENT_TYPES
            ]
            response = ModelResponse.model_validate(entry["response"])
        except (KeyError, TypeError, ValidationError) as error:
            raise ConfigurationError(
                f"Interaction {position} of the cassette {path} is damaged ({error}). "
                "Record the cassette again."
            ) from error
        request = entry.get("request")
        recorded.append(
            _Recorded(
                position=position,
                fingerprint=fingerprint,
                request=request if isinstance(request, dict) else {},
                events=events,
                response=response,
            )
        )
    return recorded


def _checked_mode(value: str, source: str) -> CassetteMode:
    if value == "auto" or value == "record" or value == "replay":
        return value
    raise ConfigurationError(f"{source} must be 'auto', 'record' or 'replay', not {value!r}.")


def _with_origin(response: ModelResponse, model: Model) -> ModelResponse:
    """Fill in which model answered, when the provider adapter left it out."""
    if response.model is not None and response.provider is not None:
        return response
    return response.model_copy(
        update={
            "model": response.model or model.name,
            "provider": response.provider or model.provider,
        }
    )
