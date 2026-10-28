"""Weakly supervised progress from successful task trajectories only.

Frozen DP visual/proprioceptive features feed a small temporal GRU. Targets are
normalized positions along successful trajectories, with a temporal ranking
penalty. This is a progress gate for asking the verifier, not a success oracle.
"""
from __future__ import annotations

from collections import deque
import threading
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch
from torch import nn
import torch.nn.functional as F

from rilo.learning.bc_train import _emit
from rilo.learning.checkpoints import atomic_torch_save, fingerprint, load_dp
from rilo.learning.data import ObservationSpec, observation_arrays, split_episodes
from rilo.learning.diffusion import DPConfig, device_for, _serialized
from rilo.learning.encoders import ObservationEncoder


class ProgressConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    history_steps: int = Field(default=4, ge=1, le=32)
    hidden_dim: int = Field(default=128, ge=8, le=512)
    steps: int = Field(default=2000, ge=1, le=100000)
    batch_size: int = Field(default=64, ge=1, le=2048)
    learning_rate: float = Field(default=3e-4, gt=0, le=.1)
    ranking_weight: float = Field(default=.2, ge=0, le=10)
    validation_fraction: float = Field(default=.2, gt=0, lt=.5)
    eval_every: int = Field(default=100, ge=1)
    patience: int = Field(default=10, ge=1)
    seed: int = Field(default=0, ge=0, le=2**31-1)
    device: str = "auto"


class ProgressModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.recurrent = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, features):
        _, hidden = self.recurrent(features)
        return self.head(hidden[-1]).squeeze(-1).sigmoid()


def _feature_episodes(episodes, base, control):
    device = next(base.parameters()).device
    result = []
    with torch.no_grad():
        for episode in episodes:
            frames = []
            for observation in episode.observations:
                if control: control()
                images, state = observation_arrays(observation, base.spec, episode.path)
                features = base.encoder(torch.from_numpy(images).to(device).unsqueeze(0),
                                        torch.from_numpy(state).to(device).unsqueeze(0))
                frames.append(features[0].cpu())
            result.append(torch.stack(frames))
    return result


def _windows(sequence, indices, history):
    return torch.stack([sequence[torch.arange(index-history+1, index+1).clamp_min(0)] for index in indices])


def train_progress(snapshot, dp_checkpoint, outdir, config: ProgressConfig | dict | None = None, *, callback=None, control=None):
    config = ProgressConfig.model_validate(config or {})
    successful = snapshot.successful()
    train_episodes, val_episodes = split_episodes(successful.episodes, config.validation_fraction, config.seed)
    device = device_for(config.device)
    base, _ = load_dp(dp_checkpoint, device=str(device))
    base.eval().requires_grad_(False)
    if (successful.spec.cameras, successful.spec.state_fields, successful.spec.state_sizes) != (base.spec.cameras, base.spec.state_fields, base.spec.state_sizes):
        raise ValueError("Progress data representation differs from the BC checkpoint")
    train_features = _feature_episodes(train_episodes, base, control)
    val_features = _feature_episodes(val_episodes, base, control)
    torch.manual_seed(config.seed)
    generator = np.random.default_rng(config.seed)
    model = ProgressModel(base.config.feature_dim, config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    best, bad, last = float("inf"), 0, {}
    checkpoint = output/"progress.pt"
    for step in range(1, config.steps+1):
        if control: control()
        sequence = train_features[int(generator.integers(len(train_features)))]
        positions = generator.integers(0, len(sequence)-1, size=config.batch_size)
        later = np.minimum(positions+generator.integers(1, max(2, len(sequence)//4), size=config.batch_size), len(sequence)-1)
        first = _windows(sequence, positions, config.history_steps).to(device)
        second = _windows(sequence, later, config.history_steps).to(device)
        targets = torch.as_tensor(positions/(len(sequence)-1), device=device, dtype=torch.float32)
        later_targets = torch.as_tensor(later/(len(sequence)-1), device=device, dtype=torch.float32)
        model.train()
        prediction, later_prediction = model(first), model(second)
        regression = (F.smooth_l1_loss(prediction, targets, beta=.1)+F.smooth_l1_loss(later_prediction, later_targets, beta=.1))/2
        ranking = F.relu(prediction-later_prediction+.1*(later_targets-targets)).mean()
        loss = regression+config.ranking_weight*ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        last = {"step": step, "progress_loss": float(loss.detach().cpu()),
                "progress_ranking_loss": float(ranking.detach().cpu())}
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            model.eval()
            errors = []
            with torch.no_grad():
                for sequence in val_features:
                    for start in range(0, len(sequence), config.batch_size):
                        if control: control()
                        indices = np.arange(start, min(start+config.batch_size, len(sequence)))
                        predicted = model(_windows(sequence, indices, config.history_steps).to(device))
                        targets = torch.as_tensor(indices/(len(sequence)-1), device=device, dtype=torch.float32)
                        errors.extend((predicted-targets).abs().cpu().tolist())
            score = sum(errors)/len(errors)
            if not np.isfinite(score):
                raise ValueError("Nonfinite progress validation loss")
            last["progress_validation_mae"] = score
            if score < best:
                best, bad = score, 0
                atomic_torch_save({"format": "rilo.progress.v1", "spec": base.spec.model_dump(mode="json"),
                    "dp_config": base.config.model_dump(mode="json"), "config": config.model_dump(mode="json"),
                    "encoder_state_dict": base.encoder.state_dict(), "state_dict": model.state_dict(),
                    "metadata": {"dataset": successful.manifest(), "base_checkpoint_sha256": fingerprint(dp_checkpoint),
                        "target": "normalized_successful_trajectory_position", "step": step,
                        "validation_mae": best, "split": {"train": [ep.id for ep in train_episodes],
                                                           "validation": [ep.id for ep in val_episodes]}}}, checkpoint)
            else:
                bad += 1
            _emit(last, callback, output/"metrics.jsonl")
            if bad >= config.patience: break
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": fingerprint(checkpoint),
            "metrics": {**last, "best_validation_mae": best, "steps_completed": step},
            "config": config.model_dump(mode="json"), "dataset": successful.manifest()}


class ProgressPredictor:
    def __init__(self, checkpoint_path, device="cpu"):
        self._lock = threading.RLock()
        self.device = device_for(device)
        payload = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=True)
        if payload.get("format") != "rilo.progress.v1":
            raise ValueError("Unsupported progress checkpoint")
        self.spec = ObservationSpec.model_validate(payload["spec"])
        self.config = ProgressConfig.model_validate(payload["config"])
        base_config = DPConfig.model_validate(payload["dp_config"])
        self.encoder = ObservationEncoder(self.spec, base_config.feature_dim, base_config.keypoints, base_config.backbone).to(self.device)
        self.encoder.load_state_dict(payload["encoder_state_dict"])
        self.encoder.eval().requires_grad_(False)
        self.model = ProgressModel(base_config.feature_dim, self.config.hidden_dim).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval().requires_grad_(False)
        self.metadata = payload.get("metadata", {})
        self.checkpoint_fingerprint = fingerprint(checkpoint_path)
        self.reset_episode()

    @_serialized
    def reset_episode(self):
        self.history = deque(maxlen=self.config.history_steps)
        self._last_id = None
        self._last_value = None

    @_serialized
    @torch.no_grad()
    def observe(self, observation):
        if observation.id == self._last_id:
            return self._last_value
        images, state = observation_arrays(observation, self.spec)
        feature = self.encoder(torch.from_numpy(images).to(self.device).unsqueeze(0),
                               torch.from_numpy(state).to(self.device).unsqueeze(0))[0]
        if not self.history:
            self.history.extend([feature]*self.config.history_steps)
        else:
            self.history.append(feature)
        value = float(self.model(torch.stack(list(self.history)).unsqueeze(0)).cpu()[0])
        self._last_id, self._last_value = observation.id, value
        return value
