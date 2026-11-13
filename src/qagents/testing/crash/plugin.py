"""Runner used by the lightweight pytest fixture."""

from pathlib import Path


class CrashRunner:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    async def run(self, build, prompt=None, **options):
        from . import crash_test

        return await crash_test(build, prompt, directory=self.directory, **options)
