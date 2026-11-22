"""Bootstrap an owned crash worker from a private pipe."""

import pickle
import sys


def main():
    sys.path[:0] = pickle.load(sys.stdin.buffer)
    bootstrap = pickle.load(sys.stdin.buffer)

    from qagents.testing.crash._subprocess import resolve
    from qagents.testing.crash._worker import run_worker

    build = resolve(bootstrap.pop("build"))
    if bootstrap["scenario"] is not None:
        bootstrap["scenario"] = resolve(bootstrap["scenario"])
    run_worker(bootstrap, build)


if __name__ == "__main__":
    main()
