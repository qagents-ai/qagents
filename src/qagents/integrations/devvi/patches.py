"""Profile patches for the Devvi processes qagents starts, written as JSON in ``.patch.yml`` files.

A Devvi profile is a stack of Cordis patch layers; ``devvi --profile sdk --patch file``
applies one more layer. A patch is a YAML list of rows. YAML 1.2 accepts JSON, so qagents
writes JSON (as Devvi's own ``scripts/smoke-python-runtime.py`` does, ``write_profile_patch``)
and needs no YAML library. A row either changes a row of the profile by ``id`` (``config``
replaces that row's whole config, ``disabled: true`` turns it off) or ``insert``\\ s new rows.

Every launch gets `default_rows`:

* ``session-persistence-jsonl``: session logs under ``<home>/sessions``, uncompressed, so
  `qagents.integrations.devvi.logs` can read them line by line (crash re-attach, exports).
* ``session-log-deepseek`` with ``enabled: false``: Devvi otherwise uploads the session log
  along with official DeepSeek API requests (``DevviAgent(share_session_log=True)`` keeps it).
* ``session-telemetry-otel`` disabled.

Builders for extra rows: `mcp_stdio_row` and `mcp_http_row` (an MCP server for Devvi's
``@deepseek-ai/dsh-mcp-client``; its tools appear as ``mcp__<name>__<tool>``),
`model_route_row` (an OpenAI-compatible model route through ``llm-pi-ai``, such as QCloud) and
`deepseek_models_row` (model ids of the ``deepseek-official`` route).

Credentials never go into a patch: patch files are written to disk. A model route names
its key by environment variable (``apiKeyEnv``), and `write_patch` refuses rows that hold
a value that looks like a credential.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ... import prompts
from ...external import ExternalAgentConfigurationError
from .._process import find_program

PatchRow = dict[str, Any]
"""One row of a Cordis patch list."""

PATCH_SUFFIX = ".patch.yml"

PERSISTENCE_ROW = "session-persistence-jsonl"
SESSION_LOG_ROW = "session-log-deepseek"
TELEMETRY_ROW = "session-telemetry-otel"
MODEL_ROUTES_ROW = "llm-pi-ai"
DEEPSEEK_ROW = "llm-deepseek"
MCP_CLIENT_PACKAGE = "@deepseek-ai/dsh-mcp-client"

SERVER_NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
"""Devvi's ``serverName`` rule (``mcp-client``): tools become ``mcp__<name>__<tool>``."""

SECRET_NAME = re.compile(r"KEY|PASSWORD|PASSWD|SECRET|TOKEN|CREDENTIAL|COOKIE|AUTH", re.IGNORECASE)
"""Names of environment variables and headers whose values are treated as credentials."""

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_KEY = re.compile(
    r"(^|[-_.])(api[-_]?key|apikey|token|access[-_]?token|refresh[-_]?token|secret|"
    r"client[-_]?secret|password|passwd|authorization|bearer|cookie)$",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"^(sk-[A-Za-z0-9_-]{16,}|dvs_[A-Za-z0-9_-]{8,}|bearer\s+\S{8,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"xox[abposr]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16})",
    re.IGNORECASE,
)
_SECRET_MAPS = {"env", "headers", "environment"}


def default_rows(home: str | Path, *, share_session_log: bool = False) -> list[PatchRow]:
    """The rows every Devvi launch gets (see the module docstring)."""
    return [
        {
            "id": PERSISTENCE_ROW,
            "config": {"root": str(Path(home) / "sessions"), "compression": "none"},
        },
        {"id": SESSION_LOG_ROW, "config": {"enabled": bool(share_session_log)}},
        {"id": TELEMETRY_ROW, "disabled": True},
    ]


def devvi_patch(
    home: str | Path, *, share_session_log: bool = False, extra: Iterable[PatchRow] = ()
) -> list[PatchRow]:
    """`default_rows` followed by ``extra``: the whole patch a `DevviAgent` writes."""
    rows = default_rows(home, share_session_log=share_session_log)
    rows.extend(dict(row) for row in extra)
    return rows


def sessions_root(rows: Sequence[Mapping[str, Any]], home: str | Path) -> Path:
    """Where the rows put Devvi's session logs (the last ``session-persistence-jsonl`` row)."""
    root: Path = Path(home) / "sessions"
    for row in rows:
        if row.get("id") == PERSISTENCE_ROW and isinstance(row.get("config"), Mapping):
            configured = row["config"].get("root")
            if isinstance(configured, str) and configured:
                root = Path(configured)
    return root


def compression_of(rows: Sequence[Mapping[str, Any]]) -> str | None:
    """The configured log compression (``None``: Devvi's default, ``zstd``)."""
    found: str | None = None
    for row in rows:
        if row.get("id") == PERSISTENCE_ROW and isinstance(row.get("config"), Mapping):
            value = row["config"].get("compression")
            found = value if isinstance(value, str) else None
    return found


# --- MCP servers ---------------------------------------------------------------------------


def mcp_stdio_row(
    name: str,
    command: str | Path,
    args: Sequence[str] = (),
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    tool_call_timeout_s: float | None = None,
    fail_on_startup_error: bool = True,
    row_id: str | None = None,
) -> PatchRow:
    """An MCP server Devvi starts over stdio; its tools appear as ``mcp__<name>__<tool>``.

    Args:
        name: The server name (``[A-Za-z0-9_-]{1,32}``).
        command: The program; resolved to an absolute path now. A bare name is looked up
            on ``PATH`` (its absolute entries only, never the current directory, which
            Windows would search first); a path with a folder part is relative to the
            current directory.
        args: Its arguments.
        env: Extra variables (Devvi gives MCP children a scrubbed environment). Names that
            look like credentials are refused: their values would be written to disk.
        cwd: Working directory of the server.
        tool_call_timeout_s: Devvi's timeout for one tool call (its default is 60 s).
        fail_on_startup_error: Fail the Devvi start when the server does not connect.
        row_id: Id of the inserted row; defaults to ``mcp-<name>``.

    Raises:
        ExternalAgentConfigurationError: A bad name, a program that is not found, or a
            credential in ``env``.
    """
    _check_server_name(name)
    program = str(command)
    resolved: str | None
    if os.path.isabs(program):
        resolved = program
    elif os.path.dirname(program):  # relative to the current directory: the caller's choice
        candidate = os.path.abspath(program)
        resolved = candidate if os.path.isfile(candidate) else None
    else:
        resolved = find_program(program)
    if resolved is None:
        raise ExternalAgentConfigurationError(
            prompts.DEVVI_MCP_PROGRAM_NOT_FOUND.format(program=program)
        )
    variables = {str(key): str(value) for key, value in (env or {}).items()}
    for key in variables:
        if SECRET_NAME.search(key):
            raise ExternalAgentConfigurationError(
                f"{prompts.DEVVI_MCP_SECRET_ENV.format(name=name, key=key)}\nGive the server "
                "its credential another way (its own configuration file), or use the ACP "
                "transport, which passes env in memory."
            )
    config: dict[str, Any] = {
        "transport": "stdio",
        "serverName": name,
        "command": str(Path(resolved)),
        "args": [str(arg) for arg in args],
        "env": variables,
        "failOnStartupError": bool(fail_on_startup_error),
        "reconnect": {"enabled": False},
    }
    if cwd is not None:
        config["cwd"] = str(Path(cwd).resolve())
    if tool_call_timeout_s is not None:
        config["toolCallTimeoutMs"] = int(tool_call_timeout_s * 1000)
    return {
        "insert": [{"id": row_id or f"mcp-{name}", "name": MCP_CLIENT_PACKAGE, "config": config}]
    }


def mcp_http_row(
    name: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    tool_call_timeout_s: float | None = None,
    fail_on_startup_error: bool = True,
    row_id: str | None = None,
) -> PatchRow:
    """An MCP server Devvi reaches over Streamable HTTP.

    ``url`` must be https, or http on a loopback address. Credential headers
    (``Authorization``, cookies, keys, tokens) are refused: they would be written to disk.

    Raises:
        ExternalAgentConfigurationError: A bad name or URL, or a credential header.
    """
    _check_server_name(name)
    parts = urlsplit(url)
    loopback = parts.hostname in ("127.0.0.1", "localhost", "::1")
    if parts.scheme not in ("http", "https") or (parts.scheme == "http" and not loopback):
        raise ExternalAgentConfigurationError(
            f"The MCP server URL {url!r} must be https, or http on a loopback address."
        )
    values = {str(key): str(value) for key, value in (headers or {}).items()}
    for key in values:
        if SECRET_NAME.search(key):
            raise ExternalAgentConfigurationError(
                f"The MCP server {name!r} would get the {key} header through a patch file, "
                "which is written to disk.\nUse the ACP transport, which passes headers in "
                "memory, or a server that needs no credential on loopback."
            )
    config: dict[str, Any] = {
        "transport": "streamable-http",
        "serverName": name,
        "url": url,
        "headers": values,
        "failOnStartupError": bool(fail_on_startup_error),
        "reconnect": {"enabled": False},
    }
    if tool_call_timeout_s is not None:
        config["toolCallTimeoutMs"] = int(tool_call_timeout_s * 1000)
    return {
        "insert": [{"id": row_id or f"mcp-{name}", "name": MCP_CLIENT_PACKAGE, "config": config}]
    }


# --- model routes ----------------------------------------------------------------------------


def openai_route(
    *,
    base_url: str,
    api_key_env: str,
    models: Sequence[str | Mapping[str, Any]],
    api: str = "openai-completions",
) -> dict[str, Any]:
    """One provider profile for `model_routes_row`: an OpenAI-compatible endpoint.

    Args:
        base_url: The endpoint's base URL.
        api_key_env: The NAME of the environment variable that holds the key. List the
            same name in ``DevviAgent(env_passthrough=...)`` so Devvi gets it.
        models: Model ids, or dicts such as ``{"id": "m", "contextWindow": 131072}``.
        api: The pi-ai protocol (``openai-completions`` for Chat Completions).

    Raises:
        ExternalAgentConfigurationError: ``api_key_env`` is not a variable name.
    """
    if not _ENV_NAME.match(api_key_env or "") or _SECRET_VALUE.match(api_key_env):
        raise ExternalAgentConfigurationError(
            "api_key_env must be the NAME of an environment variable (for example "
            "'QCLOUD_API_KEY'), not a key."
        )
    if not models:
        raise ExternalAgentConfigurationError("A model route needs at least one model.")
    listed = [{"id": model} if isinstance(model, str) else dict(model) for model in models]
    return {"api": api, "baseURL": base_url, "apiKeyEnv": api_key_env, "models": listed}


def model_routes_row(providers: Mapping[str, Mapping[str, Any]]) -> PatchRow:
    """Configure Devvi's dormant ``llm-pi-ai`` row with model routes, keyed by provider.

    ``DevviAgent(model=(provider, model_id))`` then selects one. The row replaces the
    ``llm-pi-ai`` config as a whole, so pass every route in one call.
    """
    if not providers:
        raise ExternalAgentConfigurationError("model_routes_row needs at least one provider.")
    return {
        "id": MODEL_ROUTES_ROW,
        "config": {"providers": {k: dict(v) for k, v in providers.items()}},
    }


def model_route_row(
    provider: str,
    *,
    base_url: str,
    api_key_env: str,
    models: Sequence[str | Mapping[str, Any]],
    api: str = "openai-completions",
) -> PatchRow:
    """One OpenAI-compatible model route (for example QCloud) as an ``llm-pi-ai`` row.

    rows = [model_route_row("qcloud", base_url=url, api_key_env="QCLOUD_API_KEY",
                            models=["<model-id>"])]
    DevviAgent(..., model=("qcloud", "<model-id>"), patches=rows,
               env_passthrough=["QCLOUD_API_KEY"])
    """
    route = openai_route(base_url=base_url, api_key_env=api_key_env, models=models, api=api)
    return model_routes_row({provider: route})


def deepseek_models_row(models: Sequence[str | Mapping[str, Any]]) -> PatchRow:
    """Declare the model ids of the ``deepseek-official`` route (its whole config)."""
    listed = [{"id": model} if isinstance(model, str) else dict(model) for model in models]
    return {"id": DEEPSEEK_ROW, "config": {"models": listed}}


# --- checking and writing ----------------------------------------------------------------------


def find_secrets(value: Any, path: str = "rows") -> list[str]:
    """Paths (``rows[1].config.headers.Authorization``) of values that look like credentials.

    A value counts when its key names a credential (``apiKey``, ``token``, ``password``,
    ``authorization``, ...; keys ending in ``Env`` name a variable and are fine), when it
    sits in an ``env`` or ``headers`` map under a credential-like name, or when it looks
    like a known key format (``sk-...``, ``dvs_...``, ``ghp_...``, ``Bearer ...``).
    """
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            where = f"{path}.{key}"
            name = str(key)
            if isinstance(item, str) and item:
                named = _SECRET_KEY.search(name) and not name.lower().endswith("env")
                if named or _SECRET_VALUE.match(item.strip()):
                    found.append(where)
                    continue
            if name.lower() in _SECRET_MAPS and isinstance(item, Mapping):
                found.extend(
                    f"{where}.{entry}"
                    for entry, text in item.items()
                    if SECRET_NAME.search(str(entry)) and isinstance(text, str) and text
                )
            found.extend(find_secrets(item, where))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            if isinstance(item, str) and _SECRET_VALUE.match(item.strip()):
                found.append(f"{path}[{index}]")
            else:
                found.extend(find_secrets(item, f"{path}[{index}]"))
    return sorted(set(found))


def check_rows(rows: Sequence[Any]) -> list[PatchRow]:
    """Rows as plain dicts, after checking their shape and that they hold no credential.

    Raises:
        ExternalAgentConfigurationError: A row is not a mapping, is not JSON, or holds a
            value that looks like a credential (the message names where, never the value).
    """
    checked: list[PatchRow] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ExternalAgentConfigurationError(
                prompts.DEVVI_PATCH_ROW_TYPE.format(index=index, type=type(row).__name__)
            )
        checked.append(dict(row))
    try:
        json.dumps(checked)
    except (TypeError, ValueError) as error:
        raise ExternalAgentConfigurationError(
            prompts.DEVVI_PATCH_ROWS_JSON.format(error=error)
        ) from error
    secrets = find_secrets(checked)
    if secrets:
        raise ExternalAgentConfigurationError(
            prompts.DEVVI_PATCH_SECRET.format(
                where=prompts.EXTERNAL_SETUP_LIST_SEPARATOR.join(secrets[:5])
            )
            + "\nPatch files are written to disk: reference "
            "credentials by environment variable name (apiKeyEnv) and list the name in "
            "env_passthrough=."
        )
    return checked


def render_patch(rows: Sequence[Any]) -> str:
    """The text of a patch file: the checked rows as indented JSON (valid YAML 1.2)."""
    return json.dumps(check_rows(rows), indent=2, ensure_ascii=True) + "\n"


def write_patch(directory: str | Path, rows: Sequence[Any]) -> Path:
    """Write ``rows`` to ``<directory>/qagents-<hash>.patch.yml`` and return its path.

    The name follows the content, so the same rows reuse one file and a file a running
    Devvi reads is never rewritten. The write is atomic.
    """
    text = render_patch(rows)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    folder = Path(directory)
    path = folder / f"qagents-{digest}{PATCH_SUFFIX}"
    if _same_text(path, text):
        return path
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(text, encoding="utf-8", newline="\n")
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        if not _same_text(path, text):
            raise
    return path


def _same_text(path: Path, text: str) -> bool:
    try:
        return path.read_text(encoding="utf-8") == text
    except OSError:
        return False


def _check_server_name(name: str) -> None:
    if not SERVER_NAME.match(name or ""):
        raise ExternalAgentConfigurationError(prompts.DEVVI_MCP_SERVER_NAME.format(name=name))
