"""Explicit, deterministic 2D smoke-test environment; never masquerades as LIBERO."""
from __future__ import annotations

from typing import Sequence
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw

from rilo.contracts import ActionSpec, Observation


class DemoAdapter:
    def __init__(self, image_size: int = 256):
        self.image_size = image_size
        self.position = np.array([0.2, 0.25], dtype=float)
        self.target = np.array([0.75, 0.7], dtype=float)
        self._task_target = self.target.copy()
        self._initial_position = self.position.copy()
        self.stage = "task"
        self.steps = 0
        self._spec = ActionSpec(
            names=("dx", "dy"), low=(-1.0, -1.0), high=(1.0, 1.0),
            description="2D smoke test only. Each tick moves the cyan point by values * 0.025 in canvas coordinates. +x is right, +y is down. Reach the amber target. No gripper.",
            control_hz=20,
        )

    @property
    def action_spec(self) -> ActionSpec:
        return self._spec

    def describe(self) -> dict:
        return {"name": "Demo · 2D point target", "simulation": True,
                "description": "A deterministic UI/runner smoke test, not LIBERO or a robot learning benchmark.",
                "free_space_reset": {"supported": True, "simulation_only": True,
                    "verification": "The demo point must return within 0.045 canvas units of its initial position. "
                                    "The 2D demo has no collision obstacles, gripper, or background dynamics."},
                "action_spec": self.action_spec.to_dict()}

    def list_tasks(self) -> list[dict]:
        return [{"id": 0, "name": "point_target", "language": "Move the cyan point into the amber target (demo only)."}]

    def reset(self, *, seed: int, task_id: int) -> Observation:
        if task_id != 0:
            raise ValueError("Demo environment only has task 0")
        rng = np.random.default_rng(seed)
        self.position = rng.uniform(0.15, 0.3, size=2)
        self._initial_position = self.position.copy()
        self.target = self._task_target.copy()
        self.stage = "task"
        self.steps = 0
        return self._observe()

    def begin_reset(self) -> Observation:
        self.stage = "reset"
        self.target = self._initial_position.copy()
        self.steps = 0
        return self._observe()

    def begin_free_space_reset(self) -> Observation:
        """The deterministic demo's return predicate is its endpoint verifier."""
        return self.begin_reset()

    def begin_task(self) -> Observation:
        self.stage = "task"
        self.target = self._task_target.copy()
        self.steps = 0
        return self._observe()

    def step(self, values: Sequence[float]) -> Observation:
        self.position = np.clip(self.position + np.array(self.action_spec.validate(values)) * 0.025, 0.04, 0.96)
        self.steps += 1
        return self._observe()

    def observe(self) -> Observation:
        return self._observe()

    def _observe(self) -> Observation:
        size = self.image_size
        img = Image.new("RGB", (size, size), "#111d29")
        draw = ImageDraw.Draw(img)
        for i in range(0, size, max(size // 10, 1)):
            draw.line([(i, 0), (i, size)], fill="#22323f")
            draw.line([(0, i), (size, i)], fill="#22323f")
        x, y = (self.target * size).astype(int)
        r = int(size * 0.055)
        draw.ellipse((x-r, y-r, x+r, y+r), outline="#f6bc65", width=3)
        x, y = (self.position * size).astype(int)
        r = int(size * 0.026)
        draw.ellipse((x-r, y-r, x+r, y+r), fill="#7de0ce")
        draw.text((10, 10), "DEMO / NOT LIBERO", fill="#b9c6cf")
        success = bool(np.linalg.norm(self.position - self.target) < 0.045)
        return Observation(id=uuid.uuid4().hex, timestamp=time.time(),
                           task=self.list_tasks()[0]["language"] if self.stage == "task" else "Return the cyan point to its initial position, now marked by the amber target.",
                           state={"position": self.position.tolist(), "target": self.target.tolist()},
                           images={"overview": np.asarray(img)}, success=success, terminated=success,
                           info={"environment_step": self.steps, "stage": self.stage})

    def pause(self) -> None:
        pass  # No simulation tick occurs unless step() is called.

    def close(self) -> None:
        pass
