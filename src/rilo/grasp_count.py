"""Count executed close attempts, not repeated hold commands or GPT proposals."""
from __future__ import annotations
import math


class GraspCounter:
    def __init__(self, control: dict | None):
        self.control = control or {}
        self.index = self.control.get("action_index")
        if (type(self.index) is not int or self.index < 0
                or type(self.control.get("close_sign", 1)) not in (int, float)
                or self.control.get("close_sign", 1) not in (-1, 1)
                or type(self.control.get("deadband", .1)) not in (int, float)
                or not 0 <= self.control.get("deadband", .1) < 1
                or type(self.control.get("reopen_ticks", 2)) is not int
                or self.control.get("reopen_ticks", 2) < 1):
            self.index = None
        self.mode = "open" if self.control.get("initial_open") is True else None
        self.open_ticks = 0
        self.reliable = isinstance(self.index, int) and self.mode is not None
        self.counts = {"task": 0, "reset": 0}

    def applied(self, values, stage: str) -> bool:
        if (not isinstance(self.index, int) or not isinstance(values, (list, tuple))
                or self.index >= len(values) or type(values[self.index]) not in (int, float)
                or not math.isfinite(values[self.index])):
            self.reliable = False
            return False
        value = values[self.index] * self.control.get("close_sign", 1)
        deadband = self.control.get("deadband", .1)
        if value < -deadband:
            self.open_ticks += 1
            if self.open_ticks >= self.control.get("reopen_ticks", 2):
                self.mode = "open"
        elif value > deadband:
            self.open_ticks = 0
            if self.mode != "closed":
                if self.mode is None:
                    self.reliable = False
                self.counts[stage] += 1
                self.mode = "closed"
                return True
        else:
            self.open_ticks = 0
        return False

    def quality(self, stage: str, outcome: str | None, baseline: dict | None):
        expected = (baseline or {}).get("expected_closures") if stage == "task" else None
        actual = self.counts[stage]
        label = "unknown"
        if outcome and outcome != "success":
            label = "not_successful"
        elif outcome == "success" and self.reliable and expected is not None:
            label = "extra_grasp_attempts" if actual > expected else "within_grasp_budget" if actual == expected else "unknown"
        return {"label": label, "actual_closures": actual if isinstance(self.index, int) else None,
                "expected_closures": expected, "baseline_version": (baseline or {}).get("version"),
                "source": "executed_gripper_v1", "count_reliable": self.reliable}
