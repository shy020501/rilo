"""Small TD3 residual learner over frozen BC visual/proprioceptive features.

The algorithm follows ResFiT's full-action critic and RLPD replay recipe. This
implementation is independent of environment, policy architecture, and image
encoder. Features and base actions must come from the frozen checkpoint. The
critic never receives a residual in place of the actual executed action.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import os
import tempfile

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch
from torch import nn


RESFIT_PAPER = "https://arxiv.org/abs/2509.19301"
RESFIT_CODE = "https://github.com/amazon-far/residual-offpolicy-rl"


class TD3Config(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    hidden_dim: int = Field(default=256, ge=16, le=2048)
    hidden_layers: int = Field(default=2, ge=1, le=6)
    num_critics: int = Field(default=4, ge=2, le=16)
    target_subset: int = Field(default=2, ge=2, le=16)
    actor_lr: float = Field(default=1e-5, gt=0, le=.1)
    critic_lr: float = Field(default=1e-4, gt=0, le=.1)
    tau: float = Field(default=.005, gt=0, le=1)
    gamma: float = Field(default=.995, gt=0, le=1)
    n_step: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=256, ge=2, le=8192)
    utd_ratio: int = Field(default=4, ge=1, le=128)
    policy_delay: int = Field(default=2, ge=1, le=128)
    critic_warmup_updates: int = Field(default=1000, ge=0, le=1000000)
    learning_starts: int = Field(default=128, ge=1, le=1000000)
    buffer_capacity: int = Field(default=100000, ge=1, le=2000000)
    offline_fraction: float = Field(default=.5, ge=0, le=1)
    # Fractions of each action dimension's half-range, never physical metres.
    residual_scale: float = Field(default=.1, ge=0, le=1)
    exploration_std: float = Field(default=.025, ge=0, le=1)
    target_noise: float = Field(default=.02, ge=0, le=1)
    target_noise_clip: float = Field(default=.05, ge=0, le=1)
    gradient_clip_norm: float = Field(default=10., gt=0, le=1000)
    bootstrap_on_truncation: bool = True
    seed: int = Field(default=0, ge=0, le=2**31-1)

    @model_validator(mode="after")
    def consistent_limits(self):
        if self.target_subset > self.num_critics:
            raise ValueError("target_subset cannot exceed num_critics")
        if self.buffer_capacity < self.learning_starts:
            raise ValueError("buffer_capacity must cover learning_starts")
        return self


def _mlp(input_dim: int, hidden_dim: int, hidden_layers: int, output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class ResidualActor(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, config: TD3Config):
        super().__init__()
        self.net = _mlp(feature_dim + action_dim, config.hidden_dim, config.hidden_layers, action_dim)
        # Initial evaluation is exactly the original BC policy.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((features, base_action), dim=-1)).tanh()


class QEnsemble(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, config: TD3Config):
        super().__init__()
        self.members = nn.ModuleList(_mlp(feature_dim + action_dim, config.hidden_dim,
                                          config.hidden_layers, 1) for _ in range(config.num_critics))

    def forward(self, features: torch.Tensor, full_action: torch.Tensor) -> torch.Tensor:
        value = torch.cat((features, full_action), dim=-1)
        return torch.cat([member(value) for member in self.members], dim=-1)


class ResidualTD3:
    """TD3 on full actions, with bounded corrections around cached base actions.

    No base-policy module or encoder is registered with the optimizers. Inputs
    are detached even when callers provide tensors requiring gradients.
    """
    def __init__(self, feature_dim: int, action_low, action_high,
                 config: TD3Config | dict | None = None, *, device: str = "cpu"):
        self.config = TD3Config.model_validate(config or {})
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; choose cpu or install CUDA PyTorch")
        self.feature_dim = int(feature_dim)
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self.low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device).reshape(-1)
        self.high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device).reshape(-1)
        if (self.low.numel() == 0 or self.low.shape != self.high.shape or
                not torch.isfinite(self.low).all() or not torch.isfinite(self.high).all() or
                not torch.all(self.high > self.low)):
            raise ValueError("Action bounds must be finite, increasing, matching vectors")
        self.action_dim = self.low.numel()
        self.center, self.half_range = (self.high + self.low) / 2, (self.high - self.low) / 2
        # Initialization does not perturb the caller's torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.actor = ResidualActor(self.feature_dim, self.action_dim, self.config).to(self.device)
            self.critic = QEnsemble(self.feature_dim, self.action_dim, self.config).to(self.device)
        self.target_actor = deepcopy(self.actor).requires_grad_(False).eval()
        self.target_critic = deepcopy(self.critic).requires_grad_(False).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        self.updates = 0
        self.actor_updates = 0

    def _tensor(self, value, *, width: int | None = None) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device).detach()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or (width is not None and tensor.shape[-1] != width) or not torch.isfinite(tensor).all():
            raise ValueError("RL inputs must be finite [batch, feature/action] arrays")
        return tensor

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.center) / self.half_range

    def _full_action(self, actor: nn.Module, features: torch.Tensor, base: torch.Tensor,
                     *, noise_std=0., noise_clip: float | None = None) -> torch.Tensor:
        base = torch.maximum(torch.minimum(base, self.high), self.low)
        residual = self.config.residual_scale * actor(features, self.normalize_action(base))
        if noise_std:
            noise = torch.randn(residual.shape, generator=self.generator, device=self.device) * noise_std
            if noise_clip is not None:
                noise = noise.clamp(-noise_clip, noise_clip)
            residual = residual + noise
        # Exploration and target smoothing obey the same residual safety bound.
        residual = residual.clamp(-self.config.residual_scale, self.config.residual_scale)
        return torch.maximum(torch.minimum(base + self.half_range * residual, self.high), self.low)

    @torch.no_grad()
    def act(self, features, base_action, *, explore=False) -> np.ndarray:
        obs = self._tensor(features, width=self.feature_dim)
        base = self._tensor(base_action, width=self.action_dim)
        if len(obs) != len(base):
            raise ValueError("Feature/action batch sizes differ")
        action = self._full_action(self.actor, obs, base,
                                   noise_std=self.config.exploration_std if explore else 0.)
        result = action.cpu().numpy()
        return result[0] if np.ndim(features) == 1 else result

    @torch.no_grad()
    def bellman_target(self, batch: dict[str, Any]) -> torch.Tensor:
        features = self._tensor(batch["next_features"], width=self.feature_dim)
        base = self._tensor(batch["next_base_actions"], width=self.action_dim)
        reward = self._tensor(batch["rewards"], width=1)
        discount = self._tensor(batch["discounts"], width=1)
        if not (len(features) == len(base) == len(reward) == len(discount)):
            raise ValueError("Replay batch dimensions differ")
        if (discount < 0).any() or (discount > 1).any():
            raise ValueError("Replay bootstrap discounts must lie in [0,1]")
        action = self._full_action(self.target_actor, features, base,
                                   noise_std=self.config.target_noise,
                                   noise_clip=self.config.target_noise_clip)
        q = self.target_critic(features, self.normalize_action(action))
        subset = torch.randperm(self.config.num_critics, generator=self.generator,
                                device=self.device)[:self.config.target_subset]
        return reward + discount * q[:, subset].min(dim=-1, keepdim=True).values

    def update(self, batch: dict[str, Any]) -> dict[str, float | int | None]:
        features = self._tensor(batch["features"], width=self.feature_dim)
        base = self._tensor(batch["base_actions"], width=self.action_dim)
        actions = self._tensor(batch["actions"], width=self.action_dim)
        if not (len(features) == len(base) == len(actions)):
            raise ValueError("Replay batch dimensions differ")
        if ((actions < self.low - 1e-5) | (actions > self.high + 1e-5)).any():
            raise ValueError("Replay full actions exceed environment bounds")
        targets = self.bellman_target(batch)
        q = self.critic(features, self.normalize_action(actions))
        critic_loss = (q - targets).square().mean()
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Nonfinite TD3 critic loss")
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip_norm)
        self.critic_optimizer.step()
        self.updates += 1
        self._polyak(self.critic, self.target_critic)
        actor_loss = None
        if self.updates > self.config.critic_warmup_updates and self.updates % self.config.policy_delay == 0:
            self.critic.requires_grad_(False)
            try:
                proposed = self._full_action(self.actor, features, base)
                loss = -self.critic(features, self.normalize_action(proposed)).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite TD3 actor loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip_norm)
                self.actor_optimizer.step()
                actor_loss = float(loss.detach())
                self.actor_updates += 1
                self._polyak(self.actor, self.target_actor)
            finally:
                self.critic.requires_grad_(True)
        with torch.no_grad():
            final_action = self._full_action(self.actor, features, base)
            magnitude = ((final_action - base) / self.half_range).abs().mean().item()
        return {"critic_loss": float(critic_loss.detach()), "actor_loss": actor_loss,
                "q_mean": float(q.detach().mean()), "target_q_mean": float(targets.mean()),
                "residual_magnitude": magnitude, "critic_updates": self.updates,
                "actor_updates": self.actor_updates}

    @torch.no_grad()
    def _polyak(self, model: nn.Module, target: nn.Module):
        for parameter, target_parameter in zip(model.parameters(), target.parameters()):
            target_parameter.lerp_(parameter, self.config.tau)

    def state_dict(self) -> dict[str, Any]:
        return {"format_version": 1, "kind": "residual_td3", "config": self.config.model_dump(),
                "feature_dim": self.feature_dim, "action_low": self.low.cpu().tolist(),
                "action_high": self.high.cpu().tolist(), "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(), "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "updates": self.updates, "actor_updates": self.actor_updates,
                "rng_state": self.generator.get_state().cpu()}

    def load_state_dict(self, state: dict[str, Any]):
        if state.get("kind") != "residual_td3" or state.get("format_version") != 1:
            raise ValueError("Unsupported residual checkpoint")
        if (state["feature_dim"] != self.feature_dim or state["action_low"] != self.low.cpu().tolist()
                or state["action_high"] != self.high.cpu().tolist()):
            raise ValueError("Residual checkpoint observation/action schema mismatch")
        if TD3Config.model_validate(state["config"]) != self.config:
            raise ValueError("Resume requires the checkpoint's exact TD3 configuration")
        for name in ("actor", "critic", "target_actor", "target_critic"):
            getattr(self, name).load_state_dict(state[name])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.updates, self.actor_updates = int(state["updates"]), int(state["actor_updates"])
        self.generator.set_state(state["rng_state"].cpu())

    def save(self, path: Path | str, *, metadata: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".residual-", suffix=".pt", dir=path.parent)
        os.close(fd)
        try:
            torch.save({**self.state_dict(), "metadata": metadata or {}}, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: Path | str, *, device="cpu") -> "ResidualTD3":
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(state["feature_dim"], state["action_low"], state["action_high"],
                    state["config"], device=device)
        model.load_state_dict(state)
        return model
