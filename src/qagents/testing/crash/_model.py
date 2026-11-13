"""Frozen crash-harness interfaces; behavior is pending its owner stage."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from qagents.errors import ModelError
from qagents.models.base import ModelRequest


class Cassette:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        raise NotImplementedError("The crash harness is pending its owning stage.")


def when(predicate: Callable[[ModelRequest], bool], reply: Any) -> Any:
    raise NotImplementedError("The crash harness is pending its owning stage.")


class UnscriptedRequest(ModelError):
    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False, ambiguous=False)
