"""Explicit hooks for existing robot executors and chunk interpreters.

There is intentionally no default hold command, queue replacement, observation
key mapping, coordinate transform, or physical-environment reset.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Protocol, Sequence

from .history import Observation


@dataclass(frozen=True)
class DecodedAction:
    """Robot-specific command plus a JSON-compatible record of its real meaning."""
    command: Any
    record: dict


@dataclass(frozen=True)
class ExecutedTick:
    observation: Observation
    action: dict
    reward: float
    info: dict


class ChunkDecoder(Protocol):
    def decode(self, raw_chunk: dict, *, context: dict) -> Sequence[DecodedAction]: ...


class ExecutionBridge(Protocol):
    def describe(self) -> dict: ...
    async def observe(self) -> Observation: ...
    async def sample_tick(self) -> Observation: ...
    async def replace_pending(self, actions: Sequence[DecodedAction], *, epoch: int) -> None: ...
    async def execute_tick(self, *, epoch: int) -> ExecutedTick: ...
    async def clear_pending(self, reason: str) -> None: ...
    async def hold(self, reason: str) -> None: ...
    async def set_phase(self, phase: str) -> None: ...
    async def safe_reset(self) -> bool: ...
    async def arm_reset(self) -> Observation: ...
    async def close(self) -> None: ...


async def invoke(hook, *args, **kwargs):
    result = hook(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


class HookExecutionBridge:
    """Keep an existing async executor; provide its explicit integration hooks.

    Hooks must be nonblocking or async. A blocking SDK needs its own bounded
    worker/stop mechanism; canceling a Python await does not stop physical motion.
    execute_tick must return the action actually applied, not the model proposal.
    """
    def __init__(self, *, describe: Callable, observe: Callable, replace_pending: Callable,
                 execute_tick: Callable, clear_pending: Callable, hold: Callable,
                 sample_tick: Callable, set_phase: Callable, safe_reset: Callable, arm_reset: Callable, close: Callable):
        hooks = locals().copy()
        hooks.pop("self")
        if any(not callable(value) for value in hooks.values()):
            raise TypeError("Every execution bridge hook is required and must be callable")
        self._hooks = hooks

    def describe(self):
        value = self._hooks["describe"]()
        if not isinstance(value, dict):
            raise TypeError("describe must synchronously return a capability dictionary")
        return value

    async def observe(self):
        value = await invoke(self._hooks["observe"])
        if not isinstance(value, Observation):
            raise TypeError("observe must return an explicitly mapped measured Observation")
        return value

    async def replace_pending(self, actions, *, epoch):
        await invoke(self._hooks["replace_pending"], actions, epoch=epoch)

    async def sample_tick(self):
        value = await invoke(self._hooks["sample_tick"])
        if not isinstance(value, Observation):
            raise TypeError("sample_tick must return a new measured control-tick Observation while held")
        return value

    async def execute_tick(self, *, epoch):
        value = await invoke(self._hooks["execute_tick"], epoch=epoch)
        if not isinstance(value, ExecutedTick):
            raise TypeError("execute_tick must return ExecutedTick with the actual applied action")
        return value

    async def clear_pending(self, reason):
        await invoke(self._hooks["clear_pending"], reason)

    async def hold(self, reason):
        await invoke(self._hooks["hold"], reason)

    async def set_phase(self, phase):
        await invoke(self._hooks["set_phase"], phase)

    async def safe_reset(self):
        value = await invoke(self._hooks["safe_reset"])
        if type(value) is not bool:
            raise TypeError("safe_reset must return an independent measured boolean clearance result")
        return value

    async def arm_reset(self):
        value = await invoke(self._hooks["arm_reset"])
        if not isinstance(value, Observation):
            raise TypeError("arm_reset must return the actual measured post-reset Observation")
        return value

    async def close(self):
        await invoke(self._hooks["close"])


class HookChunkDecoder:
    """Adapt an existing ChunkInterpreter without selecting a row in transport."""
    def __init__(self, decode: Callable):
        self._decode = decode

    def decode(self, raw_chunk, *, context):
        decoded = list(self._decode(raw_chunk, context=context))
        if not decoded or not all(isinstance(action, DecodedAction) for action in decoded):
            raise ValueError("Chunk interpreter must return a nonempty sequence of DecodedAction")
        return decoded
