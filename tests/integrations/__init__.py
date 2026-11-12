"""Offline tests of the integrations with external agents (ACP, Devvi) and their foundation.

``tests/`` has no ``__init__.py``, so pytest imports this directory as the package
``integrations``. No project package has that name (the library's is
``qagents.integrations``), so nothing needs to be added to ``__path__``. Helpers for the
tests live in ``integrations.fakes``; the scripts there that run as subprocesses use the
standard library only and are started with ``python -I``.
"""
