"""History indexed by actual robot control ticks, never HTTP request count."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Observation:
    tick: int
    measured_at: float
    state: dict[str, list[float]]
    images: dict[str, np.ndarray]
    success: bool = False
    terminated: bool = False
    truncated: bool = False

    def wire(self, task=""):
        return {"id": f"tick-{self.tick}", "timestamp": self.measured_at, "task": task,
                "state": self.state, "images": self.images, "success": self.success,
                "terminated": self.terminated, "info": {"control_tick": self.tick, "control_step": self.tick,
                "truncated": self.truncated, "time_limit": self.truncated}}

    @classmethod
    def measured(cls, tick, state: Mapping, images: Mapping, *, measured_at=None,
                 success=False, terminated=False, truncated=False):
        if type(tick) is not int or tick < 0:
            raise ValueError("Observation tick must be a nonnegative integer")
        timestamp = time.time() if measured_at is None else float(measured_at)
        if not math.isfinite(timestamp):
            raise ValueError("Observation timestamp must be finite")
        measured = {}
        for name, value in state.items():
            array = np.asarray(value, dtype=np.float64)
            if not name or array.ndim != 1 or not len(array) or not np.isfinite(array).all():
                raise ValueError(f"Measured state field {name!r} must be a finite nonempty vector")
            measured[str(name)] = array.tolist()
        if not measured:
            raise ValueError("An observation must contain measured state; missing fields cannot be padded")
        frames = {}
        for name, value in images.items():
            frame = np.asarray(value)
            if not name or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"Camera {name!r} must supply an RGB uint8 image")
            if frame.shape[0] < 1 or frame.shape[1] < 1:
                raise ValueError("Camera images cannot be empty")
            copied = np.array(frame, copy=True)
            copied.flags.writeable = False
            frames[str(name)] = copied
        return cls(tick, timestamp, measured, frames, bool(success), bool(terminated), bool(truncated))


class ObservationHistory:
    def __init__(self, capacity=64):
        if type(capacity) is not int or capacity < 2:
            raise ValueError("History capacity must be at least two control ticks")
        self._items = deque(maxlen=capacity)

    def append(self, observation: Observation):
        if self._items and observation.tick != self._items[-1].tick + 1:
            raise ValueError("Append one measured observation per consecutive executed control tick")
        self._items.append(observation)

    def reset(self, initial: Observation):
        self._items.clear()
        self._items.append(initial)

    @property
    def latest(self):
        if not self._items:
            raise ValueError("No measured observation is available")
        return self._items[-1]

    def sample(self, tick_offsets=(0,)):
        """Return requested historical ticks in caller order; never fabricate padding."""
        latest = self.latest.tick
        stored = {item.tick: item for item in self._items}
        result = []
        for offset in tick_offsets:
            if type(offset) is not int or offset > 0:
                raise ValueError("History offsets must be nonpositive integer control ticks")
            if latest + offset not in stored:
                raise ValueError(f"Measured history does not contain tick {latest + offset}")
            result.append(stored[latest + offset])
        return result

    def __len__(self):
        return len(self._items)
