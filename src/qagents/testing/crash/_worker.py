"""Worker-side ports for an owned subprocess crash leg."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import pickle

from qagents.effects import _ports
from qagents.errors import ConfigurationError, HeldEffectsError
from qagents.models.base import Model
from qagents.results import RunResult
from qagents.session import Session, SQLiteStore

from ._env import BuildIsolation, CrashEnv
from ._loop import VirtualLoop
from ._remote import RemoteClient
from ._world import WorldStore


class RemoteModel(Model):
    provider = "test"
    name = "crash-ledger"

    def __init__(self, remote):
        self.remote = remote

    async def stream(self, request):
        stream = await self.remote.call("model.open", (request, _ports.MODEL_TURN.get()))
        try:
            while True:
                done, event = await self.remote.call("model.next", stream)
                if done:
                    return
                yield event
        finally:
            await self.remote.call("model.close", stream)

    async def capabilities(self):
        return await self.remote.call("capabilities")


class RemoteStore(WorldStore):
    """Local SQLite, with each store boundary coordinated by the parent."""

    def __init__(self, remote, backend, mask_features):
        super().__init__(None, 0, backend, mask_features=mask_features)
        self.remote = remote

    async def _call(self, method, *args, write=False, mutating=False, detail="", **kwargs):
        token, execute, result, cancel, stall_s = await self.remote.call(
            "store.before",
            (
                method,
                write,
                detail,
                tuple(entry.model_dump_json() for entry in args[1])
                if method in {"append", "append_leased"}
                else (),
            ),
        )
        if cancel:
            # The parent targets the driving task, including a shielded write.
            self._driver.cancel()
        try:
            if stall_s is not None:
                await asyncio.sleep(stall_s)
            if execute:
                result = await getattr(self._backend, method)(*args, **kwargs)
        except BaseException:
            await self.remote.call("store.aborted", token)
            raise
        await self.remote.call("store.after", token)
        return result


class RemoteTransport:
    def __init__(self, remote, mode, loop):
        self.remote, self.mode, self.loop = remote, mode, loop
        self.services = set()

    def service(self, leg, effect, impl, *, key_arg="idempotency_key"):
        # Use the same client surface as the parent fake. Only the parent instance
        # handles requests. The child instance supplies method signatures.
        local = impl(clock=self.remote.clock, keys=False)
        self.remote.request("service.register", (effect.name, impl, key_arg))
        self.services.add(effect.name)
        return RemoteService(self, effect.name, local, "")

    def client(self, effect):
        registered = effect.name in self.services
        if self.mode == "client" and not registered:
            raise ConfigurationError(f"client transport requires env.service for {effect.name}")
        return registered and self.mode != "fake"

    async def before_second_fence(self, effect):
        await self.remote.call("fence", effect.name)

    async def send(self, effect, ctx, attempt, call_send):
        if not self.client(effect):
            return await self.remote.call("send", (effect.name, attempt))
        await self.remote.call("send.start", (effect.name, attempt))
        try:
            value = await call_send()
        except BaseException:
            await self.remote.call("send.end", (effect.name, attempt, False))
            raise
        await self.remote.call("send.end", (effect.name, attempt, True))
        return value

    async def lookup(self, effect, ctx, inquiry, call_lookup):
        if not self.client(effect):
            return await self.remote.call("lookup", (effect.name, inquiry))
        await self.remote.call("lookup.start", (effect.name, inquiry))
        try:
            result = await call_lookup()
        except BaseException:
            await self.remote.call("lookup.end", (effect.name, inquiry, False))
            raise
        await self.remote.call("lookup.end", (effect.name, inquiry, True))
        return result


class RemoteService:
    def __init__(self, transport, effect, local, prefix):
        self.transport, self.effect, self.local, self.prefix = transport, effect, local, prefix

    def __getattr__(self, name):
        member = getattr(self.local, name)
        path = f"{self.prefix}.{name}" if self.prefix else name

        async def invoke(op, args, kwargs):
            from qagents.effects import current_attempt

            attempt = current_attempt()
            binding = (attempt, attempt is not None and _ports.attempt_live(attempt))
            return await self.transport.remote.call(
                op, (self.effect, path, args, kwargs, binding, _ports.current_inquiry())
            )

        if inspect.isasyncgenfunction(member):

            async def generate(*args, **kwargs):
                remote = self.transport.remote
                stream = await invoke("service.open", args, kwargs)
                try:
                    while True:
                        done, value = await remote.call("service.next", stream)
                        if done:
                            return
                        yield value
                finally:
                    await remote.call("service.close", stream)

            return generate
        if inspect.iscoroutinefunction(member):

            async def call(*args, **kwargs):
                return await invoke("service.call", args, kwargs)

            return call
        if callable(member):

            def call_sync(*args, **kwargs):
                return self.transport.loop.call_from_thread(invoke("service.call", args, kwargs))

            return call_sync
        if member is None or isinstance(member, (str, int, float, bool, bytes)):
            return member
        return RemoteService(self.transport, self.effect, member, path)


async def worker(remote, loop, bootstrap, build):
    from ._driver import _drive_agent

    number = bootstrap["number"]
    backend = SQLiteStore(bootstrap["directory"] / "journal.db", clock=remote.clock.store_time)
    async with backend:
        store = RemoteStore(remote, backend, bootstrap["mask_features"])
        store._driver = remote.driver = asyncio.current_task()
        transport = RemoteTransport(remote, bootstrap["transport"], loop)
        model = RemoteModel(remote)
        env = CrashEnv(
            world=None,
            transport=transport,
            leg=number,
            worker_id=f"leg-{number}",
            recovering=number > 0,
            store=store,
            model=model,
            session_id=bootstrap["session_id"],
            seed=bootstrap["seed"],
            directory=bootstrap["directory"],
            clock=remote.clock,
        )
        session = Session(store, id=env.session_id)
        with contextlib.ExitStack() as stack:
            for var, value in (
                (_ports.TRANSPORT, transport),
                (_ports.REACH_OBSERVER, remote.reached),
                (_ports.FINDING_OBSERVER, lambda k, d: remote.request("finding", (k, d))),
            ):
                token = var.set(value)
                stack.callback(var.reset, token)
            agent, deps = await BuildIsolation().build(number, build, env)
            if agent.model is not model:
                raise ConfigurationError("build(env) must use env.model for offline replay.")
            tools = await agent._collect_tools()
            from ._subprocess import describe

            declarations = [
                describe(effect) for effect in agent._effect_map(tools.values()).values()
            ]
            try:
                pickle.dumps(declarations)
            except (AttributeError, TypeError, pickle.PicklingError) as error:
                raise ConfigurationError(
                    "Subprocess effects require module-level request, receipt and exception types."
                ) from error
            await remote.call(
                "describe",
                (
                    declarations,
                    [
                        (name, bool(tool.effects), tool.read_only, tool.idempotent)
                        for name, tool in tools.items()
                    ],
                    agent.permissions._approval_key,
                    (agent.limits.max_total_tokens, agent.limits.max_cost_usd),
                ),
            )
            try:
                result = await _drive_agent(
                    agent,
                    session,
                    deps,
                    env=env,
                    prompt=bootstrap["prompt"],
                    state=bootstrap["state"],
                    scenario=bootstrap["scenario"],
                    answers=bootstrap["answers"],
                    needs_recheck=bootstrap["needs_recheck"],
                    stale=bootstrap["stale"],
                    operator_answer=lambda item: remote.request("answer", item),
                    has_answer=bootstrap["has_answer"],
                    delivery=bootstrap["delivery"],
                    observe=lambda: remote.request("held_delivery"),
                    hit=lambda plan: remote.request("hit", plan.id),
                    finding=lambda kind, detail: remote.request("finding", (kind, detail)),
                    clock=remote.clock,
                )
            except HeldEffectsError as error:
                await remote.call("result", ("held_error", error.held))
            else:
                if isinstance(result, RunResult):
                    await remote.call("result", ("run", result.model_dump(mode="json")))
                else:
                    await remote.call("result", ("value", result))


def run_worker(bootstrap, build):
    remote = RemoteClient.connect(bootstrap["address"], bootstrap["authkey"])
    try:
        loop = VirtualLoop(remote.clock, thread_timeout_s=45)
        try:
            with loop.observe_tool_threads():
                try:
                    loop.run_until_complete(worker(remote, loop, bootstrap, build))
                except (Exception, asyncio.CancelledError) as error:
                    remote.request("error", (type(error).__name__, str(error)))
        finally:
            loop.close()
    finally:
        remote.close()
