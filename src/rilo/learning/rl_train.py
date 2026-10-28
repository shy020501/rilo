"""Residual learner integration without owning a robot or initiating rollouts.

The orchestrator owns environment access and GPT reset. It supplies one actual
task transition per control tick here; reset motion must never enter RL replay.
This module performs no API requests and never initializes an environment.
"""
from __future__ import annotations

from pathlib import Path
import os
import tempfile
from typing import Any

import numpy as np
import torch

from rilo.learning.replay import MixedReplay, NStepAccumulator, ReplayBuffer, Transition
from rilo.learning.residual import ResidualTD3, TD3Config


class ResidualTrainer:
    def __init__(self, feature_dim: int, action_low, action_high,
                 config: TD3Config | dict | None = None, *, device="cpu"):
        self.config = TD3Config.model_validate(config or {})
        self.agent = ResidualTD3(feature_dim, action_low, action_high, self.config, device=device)
        self.replay = MixedReplay(ReplayBuffer(1), ReplayBuffer(self.config.buffer_capacity),
                                  offline_fraction=self.config.offline_fraction, seed=self.config.seed)
        self.accumulator = NStepAccumulator(self.config.n_step, self.config.gamma,
                                            bootstrap_on_truncation=self.config.bootstrap_on_truncation)
        self.online_steps = 0
        self.episodes = 0
        self._episode_open = False
        self._active_episode = ""
        self.last_metrics: dict[str, Any] = {}
        self.base_fingerprint: str | None = None
        self.dataset_fingerprint: str | None = None
        self.offline_stats: dict[str, Any] = {}

    def act(self, features, base_action, *, explore=True):
        return self.agent.act(features, base_action, explore=explore)

    def add_offline(self, snapshot, dp, *, cancel=None, control=None) -> dict[str, Any]:
        """Reconstruct the frozen BC's queued actions on every recorded episode.

        Includes success and failure data. A successful-only dataset is valid;
        failures are never fabricated. Action labels remain actual demonstrated
        full actions, even when they lie outside the residual policy's envelope.
        """
        if self.online_steps or self.agent.updates:
            raise ValueError("Offline snapshot is immutable after residual learning starts")
        episodes = list(snapshot.episodes)
        total = sum(len(episode.actions) for episode in episodes)
        if total < 1:
            raise ValueError("Residual RL requires recorded task transitions")
        offline = ReplayBuffer(total)
        successes = failures = 0
        for episode in episodes:
            if control is not None:
                control()
            if cancel is not None and cancel():
                raise InterruptedError("Offline replay reconstruction cancelled")
            reconstructed = (dp.reconstruct_episode(episode, control=control) if control is not None
                             else dp.reconstruct_episode(episode))
            features = np.asarray(reconstructed["features"], dtype=np.float32)
            bases = np.asarray(reconstructed["base_actions"], dtype=np.float32)
            actions = np.asarray(episode.actions, dtype=np.float32)
            length = len(actions)
            if length == 0:
                continue
            if features.shape != (length + 1, self.agent.feature_dim) or bases.shape != (length + 1, self.agent.action_dim):
                raise ValueError("Offline frozen BC reconstruction has incompatible shape")
            if actions.shape != (length, self.agent.action_dim):
                raise ValueError("Offline applied actions have incompatible shape")
            low, high = self.agent.low.cpu().numpy(), self.agent.high.cpu().numpy()
            if not np.isfinite(actions).all() or np.any(actions < low - 1e-5) or np.any(actions > high + 1e-5):
                raise ValueError("Offline applied actions exceed the checkpoint's action limits")
            terminated = np.asarray(episode.terminated, dtype=bool)
            truncated = np.asarray(episode.truncated, dtype=bool)
            if terminated.shape != (length,) or truncated.shape != (length,):
                raise ValueError("Offline boundary flags have incompatible shape")
            if np.any(terminated[:-1] | truncated[:-1]):
                raise ValueError("Recorded episode crosses a task terminal/reset boundary")
            accumulator = NStepAccumulator(self.config.n_step, self.config.gamma,
                                            bootstrap_on_truncation=self.config.bootstrap_on_truncation)
            for step in range(length):
                final = step == length - 1
                # Sparse task success; failure demonstrations contribute zero.
                reward = float(final and episode.success)
                terminal = bool(terminated[step] or (final and episode.success))
                timeout = bool(truncated[step] or (final and not terminal))
                item = Transition(features[step], bases[step], actions[step], reward,
                                  features[step + 1], bases[step + 1], terminal, timeout,
                                  f"{episode.run_id}/{episode.episode_id}")
                for row in accumulator.append(item):
                    offline.add(row)
            for row in accumulator.flush():
                offline.add(row)
            successes += int(episode.success)
            failures += int(not episode.success)
        self.replay.offline = offline
        self.dataset_fingerprint = str(snapshot.fingerprint)
        self.base_fingerprint = getattr(dp, "checkpoint_fingerprint", None)
        self.offline_stats = {"offline_transitions": len(offline), "offline_success_episodes": successes,
                              "offline_failure_episodes": failures, "dataset_fingerprint": self.dataset_fingerprint,
                              "base_fingerprint": self.base_fingerprint,
                              "base_action_source": "frozen_dp_reconstruction_with_action_chunking"}
        dp.reset_episode()
        return dict(self.offline_stats)

    def observe(self, features, base_action, action, reward, next_features, next_base_action,
                *, terminated=False, truncated=False, episode_id="") -> int:
        item = Transition(features, base_action, action, reward, next_features, next_base_action,
                          bool(terminated), bool(truncated), str(episode_id))
        if item.features.shape != (self.agent.feature_dim,) or item.action.shape != (self.agent.action_dim,):
            raise ValueError("Online transition representation does not match frozen BC")
        low, high = self.agent.low.cpu().numpy(), self.agent.high.cpu().numpy()
        if np.any(item.action < low - 1e-5) or np.any(item.action > high + 1e-5):
            raise ValueError("Online replay must contain bounded actual applied actions")
        if self._episode_open and self._active_episode != item.episode_id:
            raise ValueError("Finish the current RL episode before starting another")
        rows = self.accumulator.append(item)
        for row in rows:
            self.replay.online.add(row)
        self.online_steps += 1
        self._active_episode = item.episode_id
        self._episode_open = not (terminated or truncated)
        if terminated or truncated:
            self.episodes += 1
        return len(rows)

    def end_episode(self, *, terminated=False) -> int:
        rows = self.accumulator.flush(terminated=terminated)
        for row in rows:
            self.replay.online.add(row)
        if self._episode_open:
            self.episodes += 1
        self._episode_open = False
        return len(rows)

    def update(self, utd: int | None = None) -> dict[str, Any]:
        count = self.config.utd_ratio if utd is None else int(utd)
        if count < 1 or count > 128:
            raise ValueError("UTD must lie in [1,128]")
        basic = {"online_steps": self.online_steps, "online_transitions": len(self.replay.online),
                 "offline_transitions": len(self.replay.offline), "critic_updates": self.agent.updates,
                 "actor_updates": self.agent.actor_updates}
        if len(self.replay.online) < self.config.learning_starts:
            self.last_metrics = {**basic, "warming_up": True, "updates_performed": 0}
            return dict(self.last_metrics)
        collected = []
        for _ in range(count):
            batch = self.replay.sample(self.config.batch_size)
            values = self.agent.update(batch)
            values["sampled_offline_fraction"] = float(batch["offline"].mean())
            collected.append(values)
        averaged = {key: float(np.mean([item[key] for item in collected if item.get(key) is not None]))
                    for key in ("critic_loss", "actor_loss", "q_mean", "target_q_mean",
                                "residual_magnitude", "sampled_offline_fraction")
                    if any(item.get(key) is not None for item in collected)}
        self.last_metrics = {**basic, **averaged, "critic_updates": self.agent.updates,
                             "actor_updates": self.agent.actor_updates,
                             "warming_up": False, "updates_performed": count}
        return dict(self.last_metrics)

    def save_checkpoint(self, path: Path | str, metadata: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "kind": "residual_training", "agent": self.agent.state_dict(),
                   "replay": self.replay.state_dict(), "accumulator": self.accumulator.state_dict(),
                   "online_steps": self.online_steps, "episodes": self.episodes,
                   "episode_open": self._episode_open, "active_episode": self._active_episode,
                   "base_fingerprint": self.base_fingerprint, "dataset_fingerprint": self.dataset_fingerprint,
                   "offline_stats": self.offline_stats, "metrics": self.last_metrics, "metadata": metadata or {}}
        fd, temporary = tempfile.mkstemp(prefix=".rl-training-", suffix=".pt", dir=path.parent)
        os.close(fd)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load_checkpoint(cls, path: Path | str, *, device="cpu", expected_base_fingerprint=None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("kind") != "residual_training" or payload.get("format_version") != 1:
            raise ValueError("Unsupported residual training checkpoint")
        if expected_base_fingerprint is not None and payload.get("base_fingerprint") != expected_base_fingerprint:
            raise ValueError("Residual checkpoint was trained with a different frozen BC checkpoint")
        state = payload["agent"]
        result = cls(state["feature_dim"], state["action_low"], state["action_high"], state["config"], device=device)
        result.agent.load_state_dict(state)
        result.replay = MixedReplay.from_state_dict(payload["replay"])
        result.accumulator.load_state_dict(payload["accumulator"])
        result.online_steps, result.episodes = int(payload["online_steps"]), int(payload["episodes"])
        result._episode_open = bool(payload.get("episode_open", bool(result.accumulator.pending)))
        result._active_episode = str(payload.get("active_episode", ""))
        result.base_fingerprint, result.dataset_fingerprint = payload["base_fingerprint"], payload["dataset_fingerprint"]
        result.offline_stats, result.last_metrics = payload["offline_stats"], payload["metrics"]
        return result
