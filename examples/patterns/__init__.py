"""One module per agent pattern, each runnable on its own:

    python -m examples.patterns.<name> --provider anthropic

and all together, with a results table:

    python -m examples.run_all --provider anthropic

``examples.run_all`` runs every module of this package whose name does not start with an
underscore. Each one defines ``NAME``, ``DESCRIPTION``, ``INSPIRED_BY`` and
``async def run(ctx: ExampleContext) -> ExampleOutcome``; see ``examples/harness.py`` for how
to write one. Put shared fixtures in a module whose name starts with ``_``.
"""
