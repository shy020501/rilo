"""Episode-aware n-step replay and symmetric offline/online sampling.

Replay actions are the full *applied* action, including saturation. Base actions
are separate counterfactual predictions of the same frozen, chunked BC policy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class Transition:
    features: np.ndarray
    base_action: np.ndarray
    action: np.ndarray
    reward: float
    next_features: np.ndarray
    next_base_action: np.ndarray
    terminated: bool = False
    truncated: bool = False
    episode_id: str = ""

    def __post_init__(self):
        for field in ("features", "base_action", "action", "next_features", "next_base_action"):
            value = np.asarray(getattr(self, field), dtype=np.float32)
            if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
                raise ValueError(f"{field} must be a finite, nonempty vector")
            setattr(self, field, value.copy())
        if self.features.shape != self.next_features.shape:
            raise ValueError("Transition feature shapes differ")
        if not (self.base_action.shape == self.action.shape == self.next_base_action.shape):
            raise ValueError("Transition action shapes differ")
        if not np.isfinite(self.reward):
            raise ValueError("Transition reward must be finite")
        self.reward = float(self.reward)


class NStepAccumulator:
    def __init__(self, n_step=3, gamma=.995, *, bootstrap_on_truncation=True):
        if int(n_step) != n_step or n_step < 1 or not 0 < gamma <= 1:
            raise ValueError("Invalid n-step return configuration")
        self.n_step, self.gamma = int(n_step), float(gamma)
        self.bootstrap_on_truncation = bool(bootstrap_on_truncation)
        self.pending: deque[Transition] = deque()

    def append(self, transition: Transition) -> list[dict[str, np.ndarray]]:
        if self.pending and transition.episode_id != self.pending[0].episode_id:
            raise ValueError("Finish the previous episode before adding a new one")
        self.pending.append(transition)
        if transition.terminated or transition.truncated:
            return self.flush()
        if len(self.pending) >= self.n_step:
            row = self._first()
            self.pending.popleft()
            return [row]
        return []

    def _first(self) -> dict[str, np.ndarray]:
        first = self.pending[0]
        reward = 0.
        steps = 0
        last = first
        for step, item in enumerate(self.pending):
            if step >= self.n_step:
                break
            reward += self.gamma ** step * item.reward
            steps += 1
            last = item
            if item.terminated or item.truncated:
                break
        bootstrap = not last.terminated and (not last.truncated or self.bootstrap_on_truncation)
        return {"features": first.features, "base_actions": first.base_action,
                "actions": first.action, "rewards": np.array([reward], dtype=np.float32),
                "next_features": last.next_features, "next_base_actions": last.next_base_action,
                "discounts": np.array([self.gamma ** steps if bootstrap else 0.], dtype=np.float32),
                "n_steps": np.array([steps], dtype=np.float32)}

    def flush(self, *, terminated: bool | None = None) -> list[dict[str, np.ndarray]]:
        """Close a data boundary without ever bootstrapping from a reset state.

        Explicit flush on interruption treats the last observed state as a
        time-limit truncation. Pass terminated=True only for a true terminal.
        """
        if self.pending and not (self.pending[-1].terminated or self.pending[-1].truncated):
            self.pending[-1].terminated = bool(terminated)
            self.pending[-1].truncated = not bool(terminated)
        rows = []
        while self.pending:
            rows.append(self._first())
            self.pending.popleft()
        return rows

    def state_dict(self):
        return {"n_step": self.n_step, "gamma": self.gamma,
                "bootstrap_on_truncation": self.bootstrap_on_truncation,
                "pending": [{key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
                             for key, value in vars(item).items()} for item in self.pending]}

    def load_state_dict(self, state):
        if (state["n_step"], state["gamma"], state["bootstrap_on_truncation"]) != (
                self.n_step, self.gamma, self.bootstrap_on_truncation):
            raise ValueError("n-step replay configuration changed during resume")
        self.pending = deque(Transition(**{key: value.cpu().numpy() if torch.is_tensor(value) else value
                                         for key, value in item.items()}) for item in state["pending"])


class ReplayBuffer:
    """Bounded CPU ring buffer; random sampling with replacement is deliberate."""
    FIELDS = ("features", "base_actions", "actions", "rewards", "next_features",
              "next_base_actions", "discounts", "n_steps")

    def __init__(self, capacity: int):
        if int(capacity) != capacity or capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0
        self.arrays: dict[str, np.ndarray] = {}

    def __len__(self):
        return self.size

    def add(self, row: dict[str, Any]):
        converted = {key: np.asarray(row[key], dtype=np.float32) for key in self.FIELDS}
        if any(value.ndim != 1 or not len(value) or not np.isfinite(value).all() for value in converted.values()):
            raise ValueError("Replay rows must contain finite nonempty vectors")
        if not self.arrays:
            self.arrays = {key: np.empty((self.capacity, *value.shape), dtype=np.float32)
                           for key, value in converted.items()}
        if any(value.shape != self.arrays[key].shape[1:] for key, value in converted.items()):
            raise ValueError("Replay row representation changed")
        for key, value in converted.items():
            self.arrays[key][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, count: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if count < 1 or self.size < 1:
            raise ValueError("Cannot sample an empty replay buffer or empty batch")
        indices = rng.integers(0, self.size, size=count)
        return {key: value[indices].copy() for key, value in self.arrays.items()}

    def state_dict(self):
        return {"capacity": self.capacity, "size": self.size, "position": self.position,
                "arrays": {key: torch.from_numpy(value[:self.size].copy()) for key, value in self.arrays.items()}}

    @classmethod
    def from_state_dict(cls, state):
        buffer = cls(state["capacity"])
        buffer.size, buffer.position = int(state["size"]), int(state["position"])
        if not 0 <= buffer.size <= buffer.capacity or not 0 <= buffer.position < buffer.capacity:
            raise ValueError("Invalid replay checkpoint dimensions")
        for key, value in state["arrays"].items():
            array = value.cpu().numpy()
            if key not in cls.FIELDS or array.shape[0] != buffer.size or not np.isfinite(array).all():
                raise ValueError("Invalid replay checkpoint arrays")
            buffer.arrays[key] = np.empty((buffer.capacity, *array.shape[1:]), dtype=np.float32)
            buffer.arrays[key][:buffer.size] = array
        if buffer.size and set(buffer.arrays) != set(cls.FIELDS):
            raise ValueError("Replay checkpoint is missing fields")
        return buffer


class MixedReplay:
    """Offline and online stores remain separate and their ratio is explicit.

    Before any online transition exists, training does not sample offline-only
    batches silently. If no offline data exists, online-only sampling is allowed
    and its actual fraction is reported, rather than pretending it was 50:50.
    """
    def __init__(self, offline: ReplayBuffer, online: ReplayBuffer, *, offline_fraction=.5, seed=0):
        if not 0 <= offline_fraction <= 1:
            raise ValueError("offline_fraction must lie in [0,1]")
        self.offline, self.online = offline, online
        self.offline_fraction = float(offline_fraction)
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size < 2:
            raise ValueError("Mixed replay batch_size must be at least two")
        if self.offline_fraction < 1 and len(self.online) == 0:
            raise ValueError("Online replay is not ready")
        offline_count = round(batch_size * self.offline_fraction) if len(self.offline) else 0
        online_count = batch_size - offline_count
        if offline_count and not len(self.offline):
            raise ValueError("Offline replay is empty")
        parts = []
        for buffer, count, source in ((self.offline, offline_count, 1.), (self.online, online_count, 0.)):
            if count:
                part = buffer.sample(count, self.rng)
                part["offline"] = np.full((count, 1), source, dtype=np.float32)
                parts.append(part)
        if not parts:
            raise ValueError("No replay data is available")
        batch = {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
        order = self.rng.permutation(batch_size)
        return {key: value[order] for key, value in batch.items()}

    def state_dict(self):
        return {"offline": self.offline.state_dict(), "online": self.online.state_dict(),
                "offline_fraction": self.offline_fraction, "rng_state": self.rng.bit_generator.state}

    @classmethod
    def from_state_dict(cls, state):
        result = cls(ReplayBuffer.from_state_dict(state["offline"]), ReplayBuffer.from_state_dict(state["online"]),
                     offline_fraction=state["offline_fraction"])
        result.rng.bit_generator.state = state["rng_state"]
        return result
