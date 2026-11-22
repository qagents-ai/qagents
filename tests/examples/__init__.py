"""Offline tests of the examples in ``examples/``.

``tests/`` has no ``__init__.py``, so pytest imports this directory as the package
``examples``, the same name as the project's ``examples`` package. To keep both importable,
this package also searches the project's ``examples`` directory: ``examples.harness`` and
``examples.patterns.<name>`` resolve to the project's modules, ``examples.test_<name>`` to the
tests here.
"""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "examples"))  # noqa: F821
