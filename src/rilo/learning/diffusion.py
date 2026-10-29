"""Action diffusion with a FiLM-conditioned temporal U-Net and DDIM control.

Independent compact implementation of the Diffusion Policy recipe:
https://diffusion-policy.cs.columbia.edu/ . Trajectories begin at the current
control tick; only action_steps of each prediction are executed before replanning.
"""
from __future__ import annotations

from collections import deque
from functools import wraps
import threading
import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch
from torch import nn
import torch.nn.functional as F

from rilo.learning.data import ObservationSpec, observation_arrays
from rilo.learning.encoders import ObservationEncoder


class DPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_size: int = Field(default=128, ge=16, le=512)
    obs_steps: int = Field(default=2, ge=1, le=8)
    horizon: int = Field(default=16, ge=4, le=64)
    action_steps: int = Field(default=4, ge=1, le=16)
    feature_dim: int = Field(default=128, ge=16, le=512)
    keypoints: int = Field(default=32, ge=4, le=64)
    unet_dims: tuple[int, int, int] = (64, 128, 256)
    diffusion_steps: int = Field(default=100, ge=4, le=1000)
    inference_steps: int = Field(default=10, ge=1, le=100)
    backbone: Literal["resnet18", "tiny"] = "resnet18"
    imagenet_init: bool = True
    steps: int = Field(default=10000, ge=1, le=1000000)
    checkpoint_every: int = Field(default=1000, ge=1, le=1000000)
    batch_size: int = Field(default=32, ge=1, le=1024)
    learning_rate: float = Field(default=1e-4, gt=0, le=.1)
    encoder_learning_rate: float = Field(default=1e-5, gt=0, le=.1)
    weight_decay: float = Field(default=1e-4, ge=0, le=1)
    ema_decay: float = Field(default=.995, ge=0, lt=1)
    validation_fraction: float = Field(default=.2, gt=0, lt=.5)
    eval_every: int = Field(default=250, ge=1)
    validation_batches: int = Field(default=10, ge=1, le=1000)
    patience: int = Field(default=8, ge=1)
    min_steps: int = Field(default=500, ge=0)
    seed: int = Field(default=0, ge=0, le=2**31-1)
    device: str = "auto"
    successful_only: bool = True

    @model_validator(mode="after")
    def valid_shapes(self):
        if self.horizon % 4 or self.action_steps > self.horizon:
            raise ValueError("DP horizon must be divisible by 4 and cover the executed action chunk")
        if any(x < 8 or x % 8 for x in self.unet_dims):
            raise ValueError("U-Net channels must be positive multiples of 8")
        if self.inference_steps > self.diffusion_steps:
            raise ValueError("DDIM inference steps cannot exceed diffusion training steps")
        if self.backbone == "tiny" and self.imagenet_init:
            raise ValueError("Tiny smoke encoder requires imagenet_init=false")
        return self


def device_for(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; choose cpu or install CUDA PyTorch")
    return torch.device(value)


class SinusoidalTime(nn.Module):
    def __init__(self, dimension=128):
        super().__init__()
        self.dimension = dimension

    def forward(self, t):
        frequencies = torch.exp(torch.arange(self.dimension//2, device=t.device)*(-math.log(10000)/(self.dimension//2-1)))
        phase = t.float()[:, None]*frequencies[None]
        return torch.cat((phase.sin(), phase.cos()), -1)


class ConditionedBlock(nn.Module):
    def __init__(self, incoming, outgoing, condition_dim):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(incoming, outgoing, 3, padding=1), nn.GroupNorm(8, outgoing), nn.SiLU())
        self.conv2 = nn.Sequential(nn.Conv1d(outgoing, outgoing, 3, padding=1), nn.GroupNorm(8, outgoing), nn.SiLU())
        self.condition = nn.Linear(condition_dim, outgoing*2)
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv1d(incoming, outgoing, 1)

    def forward(self, x, condition):
        scale, bias = self.condition(F.silu(condition)).chunk(2, -1)
        hidden = self.conv1(x)*(1+scale[..., None])+bias[..., None]
        return self.conv2(hidden)+self.skip(x)


class ConditionalUNet1D(nn.Module):
    def __init__(self, action_dim, observation_dim, widths):
        super().__init__()
        a, b, c = widths
        self.time = nn.Sequential(SinusoidalTime(128), nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 128))
        condition_dim = observation_dim+128
        self.input = nn.Conv1d(action_dim, a, 1)
        self.down0 = nn.ModuleList([ConditionedBlock(a, a, condition_dim) for _ in range(2)])
        self.sample0 = nn.Conv1d(a, b, 4, stride=2, padding=1)
        self.down1 = nn.ModuleList([ConditionedBlock(b, b, condition_dim) for _ in range(2)])
        self.sample1 = nn.Conv1d(b, c, 4, stride=2, padding=1)
        self.middle = nn.ModuleList([ConditionedBlock(c, c, condition_dim) for _ in range(2)])
        self.up1 = nn.ConvTranspose1d(c, b, 4, stride=2, padding=1)
        self.merge1 = ConditionedBlock(b*2, b, condition_dim)
        self.up0 = nn.ConvTranspose1d(b, a, 4, stride=2, padding=1)
        self.merge0 = ConditionedBlock(a*2, a, condition_dim)
        self.output = nn.Sequential(nn.GroupNorm(8, a), nn.SiLU(), nn.Conv1d(a, action_dim, 1))

    def forward(self, actions, timestep, observation):
        condition = torch.cat((self.time(timestep), observation), -1)
        x = self.input(actions.transpose(1, 2))
        for block in self.down0: x = block(x, condition)
        skip0, x = x, self.sample0(x)
        for block in self.down1: x = block(x, condition)
        skip1, x = x, self.sample1(x)
        for block in self.middle: x = block(x, condition)
        x = self.merge1(torch.cat((self.up1(x), skip1), 1), condition)
        x = self.merge0(torch.cat((self.up0(x), skip0), 1), condition)
        return self.output(x).transpose(1, 2)


class DiffusionPolicy(nn.Module):
    def __init__(self, spec: ObservationSpec, config: DPConfig):
        super().__init__()
        self.spec, self.config = spec, config
        self.encoder = ObservationEncoder(spec, config.feature_dim, config.keypoints, config.backbone)
        self.denoiser = ConditionalUNet1D(spec.action_dim, config.feature_dim*config.obs_steps, config.unet_dims)
        # Squared cosine cumulative schedule, with finite beta clipping.
        times = torch.linspace(0, config.diffusion_steps, config.diffusion_steps+1, dtype=torch.float64)
        cumulative = torch.cos(((times/config.diffusion_steps+.008)/1.008)*math.pi/2).square()
        cumulative = cumulative/cumulative[0]
        betas = (1-cumulative[1:]/cumulative[:-1]).clamp(1e-5, .999).float()
        self.register_buffer("alpha_cumulative", (1-betas).cumprod(0))
        self.register_buffer("action_center", torch.tensor(spec.action_low).float().add(torch.tensor(spec.action_high)).div(2))
        self.register_buffer("action_scale", torch.tensor(spec.action_high).float().sub(torch.tensor(spec.action_low)).div(2))

    def encode_observations(self, images, states):
        return self.encoder(images, states)

    def loss(self, batch, generator=None):
        clean = (batch["actions"]-self.action_center)/self.action_scale
        condition = self.encoder(batch["images"], batch["states"]).flatten(1)
        timestep = torch.randint(self.config.diffusion_steps, (len(clean),), device=clean.device, generator=generator)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        alpha = self.alpha_cumulative[timestep, None, None]
        noisy = alpha.sqrt()*clean+(1-alpha).sqrt()*noise
        prediction = self.denoiser(noisy, timestep, condition)
        mask = batch["mask"].to(clean.dtype).unsqueeze(-1)
        return ((prediction-noise).square()*mask).sum()/(mask.sum()*clean.shape[-1]).clamp_min(1)

    @torch.no_grad()
    def predict_chunk(self, images=None, states=None, *, features=None, generator=None):
        if features is None:
            features = self.encoder(images, states)
        condition = features.flatten(1)
        batch = condition.shape[0]
        sample = torch.randn((batch, self.config.horizon, self.spec.action_dim), device=condition.device, generator=generator)
        times = torch.linspace(self.config.diffusion_steps-1, 0, self.config.inference_steps, device=condition.device).round().long()
        for index, timestep in enumerate(times):
            alpha = self.alpha_cumulative[timestep]
            prediction = self.denoiser(sample, timestep.expand(batch), condition)
            clean = ((sample-(1-alpha).sqrt()*prediction)/alpha.sqrt()).clamp(-1, 1)
            previous_alpha = self.alpha_cumulative[times[index+1]] if index+1 < len(times) else torch.ones((), device=sample.device)
            # Deterministic DDIM, eta=0. Initial noise is reproducibly seeded.
            sample = previous_alpha.sqrt()*clean+(1-previous_alpha).sqrt()*prediction
        return sample.clamp(-1, 1)*self.action_scale+self.action_center


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class FrozenDPController:
    """Non-consuming chunk cache: advance() only after an executed control tick."""
    def __init__(self, checkpoint_path: str | Path, device="cpu", seed=0):
        from rilo.learning.checkpoints import load_dp, fingerprint
        self.checkpoint_fingerprint = fingerprint(checkpoint_path)
        self.model, self.metadata = load_dp(checkpoint_path, device=device)
        self.model.eval().requires_grad_(False)
        self.spec, self.config = self.model.spec, self.model.config
        self.device = next(self.model.parameters()).device
        self.seed = seed
        self._lock = threading.RLock()
        self.feature_dim = self.config.feature_dim+1
        self.reset_episode()

    @classmethod
    def from_model(cls, model, *, seed=0):
        """For deterministic local tests; normal deployment loads a registry checkpoint."""
        instance = cls.__new__(cls)
        instance.model = model.eval().requires_grad_(False)
        instance.metadata = {}
        instance.checkpoint_fingerprint = "in-memory-test-model"
        instance.spec, instance.config = model.spec, model.config
        instance.device = next(model.parameters()).device
        instance.seed, instance.feature_dim = seed, model.config.feature_dim+1
        instance._lock = threading.RLock()
        instance.reset_episode()
        return instance

    @_serialized
    def reset_episode(self):
        self.history = deque(maxlen=self.config.obs_steps)
        self.queue = deque()
        self.chunk_phase = 0
        self._new_tick = True
        self._last_id = None
        self._last_result = None
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

    @_serialized
    @torch.no_grad()
    def observe(self, observation):
        if self._last_id == observation.id and self._last_result is not None:
            return tuple(value.copy() for value in self._last_result)
        images, states = observation_arrays(observation, self.spec)
        pixels = torch.from_numpy(images).to(self.device).unsqueeze(0)
        state = torch.from_numpy(states).to(self.device).unsqueeze(0)
        feature = self.model.encoder(pixels, state)[0]
        if not self.history:
            self.history.extend([feature]*self.config.obs_steps)
        elif self._new_tick:
            self.history.append(feature)
        else:
            self.history[-1] = feature
        self._new_tick = False
        if not self.queue:
            features = torch.stack(list(self.history)).unsqueeze(0)
            predicted = self.model.predict_chunk(features=features, generator=self.generator)[0]
            self.queue.extend(predicted[:self.config.action_steps].cpu().numpy())
            self.chunk_phase = 0
        augmented = torch.cat((feature, feature.new_tensor([self.chunk_phase/self.config.action_steps])))
        self._last_result = (augmented.cpu().numpy(), self.queue[0].copy())
        self._last_id = observation.id
        return tuple(value.copy() for value in self._last_result)

    peek = observe

    @_serialized
    def advance(self):
        if not self.queue:
            raise ValueError("Cannot advance DP before observing an action")
        self.queue.popleft()
        self.chunk_phase += 1
        self._new_tick = True
        self._last_id = None
        self._last_result = None

    @_serialized
    def reconstruct_episode(self, episode, *, control=None):
        self.reset_episode()
        features, actions = [], []
        for index in range(len(episode.actions)+1):
            if control: control()
            feature, action = self.observe(episode.observation(index))
            features.append(feature)
            actions.append(action)
            if index < len(episode.actions):
                self.advance()
        return {"features": np.stack(features), "base_actions": np.stack(actions)}


DPController = FrozenDPController
