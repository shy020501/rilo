"""The only shared contracts between environments, policies, and the runner.

Actions are adapter-described vectors, not hard-coded robot coordinates. Every
adapter must describe its units, reference frames and gripper convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class ActionSpec:
    names: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    description: str
    control_hz: float
    max_repeat: int = 10
    max_chunk_steps: int = 30

    def __post_init__(self) -> None:
        if not self.names or not (len(self.names) == len(self.low) == len(self.high)):
            raise ValueError("Action dimensions and bounds must have equal nonzero length")
        if any(not math.isfinite(x) for x in (*self.low, *self.high)):
            raise ValueError("Action bounds must be finite")
        if any(a >= b for a, b in zip(self.low, self.high)):
            raise ValueError("Each action lower bound must be below its upper bound")
        if self.control_hz <= 0 or self.max_repeat < 1 or self.max_chunk_steps < 1:
            raise ValueError("Action timing and budgets must be positive")

    def validate(self, values: Sequence[float]) -> list[float]:
        if len(values) != len(self.names):
            raise ValueError(f"Expected {len(self.names)} action values, got {len(values)}")
        result = []
        for name, value, low, high in zip(self.names, values, self.low, self.high):
            if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")
            result.append(float(value))
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"names": list(self.names), "low": list(self.low), "high": list(self.high),
                "description": self.description, "control_hz": self.control_hz,
                "max_repeat": self.max_repeat, "max_chunk_steps": self.max_chunk_steps}


@dataclass
class Observation:
    id: str
    timestamp: float
    task: str
    state: dict[str, Any]
    images: dict[str, np.ndarray]  # RGB uint8, upright; adapter owns camera conversion
    success: bool = False
    terminated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class ActionCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    values: list[float]
    repeat: int = Field(default=1, ge=1, le=30, strict=True)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation_id: str
    status: Literal["continue", "success", "failure", "help"]
    actions: list[ActionCommand] = Field(default_factory=list, max_length=8)
    note: str = Field(default="", max_length=2000)
    usage: dict[str, int] = Field(default_factory=dict)
    response_id: str | None = None


class EnvironmentAdapter(Protocol):
    """Synchronous methods; the runner calls all of them on ONE dedicated thread.

    step executes exactly one bounded control tick. pause MUST prevent further
    motion for physical adapters; a simulator can simply stop stepping. A real
    robot also needs its own controller-side watchdog, independent of Python.
    No reset is allowed to occur implicitly inside step.
    """
    @property
    def action_spec(self) -> ActionSpec: ...
    def describe(self) -> dict[str, Any]: ...
    def reset(self, *, seed: int, task_id: int) -> Observation: ...
    def observe(self) -> Observation: ...
    def step(self, values: Sequence[float]) -> Observation: ...
    def pause(self) -> None: ...
    def close(self) -> None: ...


class Policy(Protocol):
    async def decide(self, observation: Observation, spec: ActionSpec,
                     context: dict[str, Any], history: list[dict[str, Any]]) -> Decision: ...
    async def close(self) -> None: ...


class ReversibleEnvironment(Protocol):
    """Optional capability; phase changes must NEVER teleport or reset physics.

    Observations use success for the CURRENT objective. The adapter verifies
    task success and reset success independently, and describes reset tolerances.
    """
    def begin_reset(self) -> Observation: ...
    def begin_task(self) -> Observation: ...
