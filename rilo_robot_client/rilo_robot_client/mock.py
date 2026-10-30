"""Fully local synthetic robot; imports no robot SDK and never touches hardware."""
from __future__ import annotations

from collections import deque
import numpy as np

from .bridge import ExecutedTick
from .decoder import VectorChunkDecoder
from .history import Observation


class MockBridge:
    def __init__(self, *, control_hz=10):
        self.control_hz, self.tick, self.position = control_hz, 0, 0.0
        self._pending, self._epoch = deque(), None
        self.holds, self.arm_resets = [], 0
        self.phase = "task"

    def describe(self):
        return {"name": "rilo-client-mock", "task": "Move the marker to the right, then restore it to the left",
            "simulation": True,
            "dummy_actions": {"task": [1.0], "reset": [-1.0]},
            "action_spec": {"names": ["mock_velocity"], "low": [-1.0], "high": [1.0],
                "description": "Synthetic one-dimensional velocity per control tick", "control_hz": self.control_hz,
                "max_repeat": 1, "max_chunk_steps": 64},
            "observation_spec": {"cameras": ["overview"], "state_fields": ["position"], "state_sizes": [1],
                "action_names": ["mock_velocity"], "action_low": [-1.0], "action_high": [1.0], "image_size": 32},
            "capabilities": {"execute": True, "hold": True, "arm_reset": True, "safe_reset": True,
                "residual_step": True, "residual_training": True, "simulation": True}}

    async def observe(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :2] = (40, 100, 40)
        column = int(np.clip(self.position, 0, 1) * 29) + 1
        frame[10:22, column:column + 2] = (240, 150, 60)
        success = self.position >= .7 if self.phase == "task" else self.position <= .05
        return Observation.measured(self.tick, {"position": [self.position]}, {"overview": frame}, success=success)

    async def replace_pending(self, actions, *, epoch):
        self._pending.clear()
        self._pending.extend(actions)
        self._epoch = epoch

    async def sample_tick(self):
        if self._pending:
            raise RuntimeError("Cannot sample a held tick while motion is queued")
        self.tick += 1
        return await self.observe()

    async def execute_tick(self, *, epoch):
        if epoch != self._epoch or not self._pending:
            raise RuntimeError("No action from the current epoch is queued")
        action = self._pending.popleft()
        applied = float(np.asarray(action.command)[0])
        if not np.isfinite(applied) or abs(applied) > 1:
            raise ValueError("Mock action outside its explicit bounds")
        self.position = float(np.clip(self.position + applied * .1, 0, 1))
        self.tick += 1
        observation = await self.observe()
        return ExecutedTick(observation, {"names": ["mock_velocity"], "values": [applied],
            "encoding": "environment"}, float(observation.success), {"simulation": True})

    async def clear_pending(self, reason):
        self._pending.clear()

    async def hold(self, reason):
        self.holds.append(reason)

    async def safe_reset(self):
        return self.position <= .05 and not self._pending

    async def arm_reset(self):
        if not await self.safe_reset():
            raise RuntimeError("Mock reset clearance was not verified")
        self.arm_resets += 1
        self.tick += 1
        self.position = 0.0
        return await self.observe()

    async def set_phase(self, phase):
        self.phase = phase

    async def close(self):
        await self.clear_pending("closed")
        await self.hold("closed")


def create_adapter(config=None, env_config=None):
    options = config or {}
    return MockBridge(control_hz=options.get("control_hz", 10)), VectorChunkDecoder()
