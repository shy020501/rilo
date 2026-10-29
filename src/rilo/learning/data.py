"""Read task trajectories from RILO receipts without changing source recordings.

Every action is one executed control tick. Reset trajectories are separate goals
and never enter a task learner. Dataset membership and file digests are frozen
before a job starts; train/validation splitting is by whole episode.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch
from torch.utils.data import Dataset

from rilo.contracts import Observation


# Shared measured proprioception available in both native LIBERO recordings and
# its published demonstrations. Extra measurements remain in the source receipts
# but are not imputed for datasets that did not record velocities or grip-site pose.
LIBERO_STATE_FIELDS = (
    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos",
)


class ObservationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    cameras: tuple[str, ...]
    state_fields: tuple[str, ...]
    state_sizes: tuple[int, ...]
    action_names: tuple[str, ...]
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    image_size: int = Field(default=128, ge=16, le=512)

    @model_validator(mode="after")
    def consistent_dimensions(self):
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError("Camera names must be nonempty and unique")
        if len(self.state_fields) != len(self.state_sizes) or any(size < 1 for size in self.state_sizes):
            raise ValueError("Measured state fields require matching positive sizes")
        if len(set(self.state_fields)) != len(self.state_fields):
            raise ValueError("Measured state field names must be unique")
        if not self.action_names or len(self.action_names) != len(self.action_low) or len(self.action_names) != len(self.action_high):
            raise ValueError("Action names and bounds must have matching nonzero dimensions")
        low, high = np.asarray(self.action_low), np.asarray(self.action_high)
        if not np.isfinite(low).all() or not np.isfinite(high).all() or not (high > low).all():
            raise ValueError("Action bounds must be finite and increasing")
        return self

    @property
    def state_dim(self):
        return sum(self.state_sizes)

    @property
    def action_dim(self):
        return len(self.action_names)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_child(parent: Path, relative: str):
    path = parent / relative
    if not path.resolve().is_relative_to(parent.resolve()):
        raise ValueError("Recorded path escapes its episode")
    return path


def state_vector(state: dict, spec: ObservationSpec) -> np.ndarray:
    pieces = []
    for name, size in zip(spec.state_fields, spec.state_sizes):
        if name not in state:
            raise ValueError(f"Missing required measured state field: {name}")
        value = np.asarray(state[name], dtype=np.float32).reshape(-1)
        if value.size != size or not np.isfinite(value).all():
            raise ValueError(f"State field changed shape or is nonfinite: {name}")
        pieces.append(value)
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)


def observation_arrays(observation: Observation | dict, spec: ObservationSpec,
                       episode_path: Path | None = None):
    state = observation.state if isinstance(observation, Observation) else observation["state"]
    source = observation.images if isinstance(observation, Observation) else observation["images"]
    images = []
    for camera in spec.cameras:
        if camera not in source:
            raise ValueError(f"Missing required camera: {camera}")
        raw = source[camera]
        if isinstance(raw, str):
            if episode_path is None:
                raise ValueError("An episode path is required for recorded images")
            with Image.open(_safe_child(episode_path, raw)) as opened:
                image = opened.convert("RGB").resize((spec.image_size, spec.image_size), Image.Resampling.BILINEAR)
                array = np.array(image, dtype=np.float32)
        else:
            array = np.asarray(raw)
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError("Camera observations must be HWC RGB")
            image = Image.fromarray(array.astype(np.uint8)).resize((spec.image_size, spec.image_size), Image.Resampling.BILINEAR)
            array = np.array(image, dtype=np.float32)
        images.append(array.transpose(2, 0, 1) / 255.)
    return np.stack(images), state_vector(state, spec)


@dataclass
class RecordedEpisode:
    run_id: str
    episode_id: str
    path: Path
    observations: list[dict]
    actions: np.ndarray
    success: bool
    terminated: np.ndarray
    truncated: np.ndarray
    provenance: dict

    @property
    def id(self):
        return f"{self.run_id}/{self.episode_id}"

    @property
    def rewards(self):
        rewards = np.zeros(len(self.actions), dtype=np.float32)
        if self.success and len(rewards):
            rewards[-1] = 1.
        return rewards

    def observation(self, index: int) -> Observation:
        value = self.observations[index]
        images = {}
        for camera, relative in value["images"].items():
            with Image.open(_safe_child(self.path, relative)) as opened:
                images[camera] = np.array(opened.convert("RGB"))
        return Observation(id=value["id"], timestamp=value.get("timestamp", 0),
            task=value.get("task", ""), state=value["state"], images=images,
            success=bool(value.get("success")), terminated=bool(value.get("terminated")),
            info=value.get("info", {}))


@dataclass
class DatasetSnapshot:
    episodes: list[RecordedEpisode]
    spec: ObservationSpec
    fingerprint: str
    project_id: str

    def manifest(self):
        return {"project_id": self.project_id, "fingerprint": self.fingerprint,
                "spec": self.spec.model_dump(mode="json"), "episodes": [ep.provenance for ep in self.episodes],
                "episode_count": len(self.episodes), "transition_count": sum(len(ep.actions) for ep in self.episodes)}

    def successful(self):
        episodes = [ep for ep in self.episodes if ep.success]
        return make_snapshot(episodes, self.spec, self.project_id)


def make_snapshot(episodes, spec, project_id="fixture"):
    content = {"spec": spec.model_dump(mode="json"), "episodes": [ep.provenance for ep in episodes]}
    fingerprint = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return DatasetSnapshot(list(episodes), spec, fingerprint, project_id)


def load_episode(path: Path, summary: dict, manifest: dict) -> RecordedEpisode:
    observations, actions, ends = [], [], []
    current = None
    transitions = path / "transitions.jsonl"
    for line in transitions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        kind, stage = row.get("kind"), row.get("stage", "task")
        if stage != "task":
            continue
        if kind in ("environment_reset", "agent_reset_continuation", "observation"):
            current = row.get("observation", current)
        elif kind == "transition":
            if current is None or current.get("id") != row.get("observation_id"):
                raise ValueError(f"Observation/action alignment is broken in {path.name}")
            if not observations:
                observations.append(current)
            else:
                # A new measurement before inference can have a new ID but the
                # same physical tick. It is the actual state used for this action.
                observations[-1] = current
            action = row.get("applied_action", row.get("requested_action"))
            if action is None:
                raise ValueError("An executed action is required")
            actions.append(action)
            current = row["next_observation"]
            observations.append(current)
            ends.append(bool(current.get("terminated")) and not current.get("info", {}).get("time_limit", False))
        elif kind == "stage_end" and stage == "task":
            break
    if not actions:
        raise ValueError("Episode has no executed task actions")
    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim != 2 or not np.isfinite(action_array).all():
        raise ValueError("Recorded actions must be a finite matrix")
    success = summary.get("outcome") == "success"
    terminated = np.asarray(ends, dtype=bool)
    # A GPT failure verdict ends the attempt, not the physical MDP. Match the
    # online runner: only verified success or an adapter terminal removes the
    # bootstrap; abandonment, time limits and operator stops are truncations.
    terminated[-1] = terminated[-1] or success
    truncated = np.zeros(len(actions), dtype=bool)
    truncated[-1] = not terminated[-1]
    digest = hashlib.sha256(transitions.read_bytes())
    # Images are evidence too; a later changed/replaced frame must change provenance.
    image_paths = sorted({relative for obs in observations for relative in obs["images"].values()})
    for relative in image_paths:
        image_path = _safe_child(path, relative)
        digest.update(relative.encode())
        digest.update(hashlib.sha256(image_path.read_bytes()).digest())
    config = manifest.get("config", {})
    learning = deepcopy(config.get("learning") or {})
    stage = learning.get("pipeline_stage")
    learned_task = stage in ("bc_eval", "rl", "rl_eval")
    source_policy = ("diffusion_policy" if stage == "bc_eval" else "diffusion_policy_residual") if learned_task else config.get("policy")
    source_model = ((learning.get("base_model_id") if stage == "bc_eval" else learning.get("residual_model_id"))
                    if learned_task else config.get("model"))
    provenance = {"run_id": path.parent.name, "episode_id": path.name, "outcome": summary.get("outcome"),
                  "transitions": len(actions), "sha256": digest.hexdigest(),
                  "source_model": source_model, "source_policy": source_policy,
                  "learning": learning}
    source = summary.get("source") or manifest.get("source")
    if source:
        provenance["source"] = deepcopy(source)
    if learned_task:
        # The configured GPT performs reset/judging, not the recorded task
        # actions. Preserve that distinct role plus all learning IDs/hashes.
        provenance["teacher"] = {key: config.get(key) for key in ("policy", "model", "reasoning_effort")}
    return RecordedEpisode(path.parent.name, path.name, path, observations, action_array,
                           success, terminated, truncated, provenance)


def _infer_spec(episode, manifest, image_size, state_fields=None, cameras=None):
    observation = episode.observations[0]
    if state_fields is None and manifest.get("environment", {}).get("name") == "libero":
        state_fields = LIBERO_STATE_FIELDS
        missing = [name for name in state_fields if name not in observation["state"]]
        if missing:
            raise ValueError(f"Missing measured LIBERO proprioception: {', '.join(missing)}")
    if state_fields is None:
        # Flatten only direct numeric measured state entries. Nested metadata,
        # calibration, success labels and adapter info never become model input.
        state_fields = []
        for key, value in observation["state"].items():
            if key.startswith("_") or any(word in key.lower() for word in ("success", "reward", "progress", "ground_truth", "object_pose")):
                continue
            if isinstance(value, (int, float, list, tuple)) and not isinstance(value, bool):
                try:
                    array = np.asarray(value, dtype=np.float32)
                    if array.ndim <= 1 and array.size and np.isfinite(array).all(): state_fields.append(key)
                except (TypeError, ValueError):
                    pass
        state_fields = sorted(state_fields)
    cameras = tuple(cameras or sorted(name for name in observation["images"] if not name.startswith("reset_reference")))
    if not cameras:
        raise ValueError("At least one camera is required")
    action = manifest.get("environment", {}).get("action_spec") or manifest.get("action_spec")
    if not action:
        raise ValueError("Run manifest is missing the adapter action specification")
    return ObservationSpec(cameras=cameras, state_fields=tuple(state_fields),
        state_sizes=tuple(int(np.asarray(observation["state"][key]).size) for key in state_fields),
        action_names=tuple(action["names"]), action_low=tuple(action["low"]), action_high=tuple(action["high"]),
        image_size=image_size)


def snapshot_project(data_dir: Path, project_id: str, *, successful_only=False,
                     state_fields=None, cameras=None, image_size=128) -> DatasetSnapshot:
    from rilo.projects import ProjectStore
    store = ProjectStore(Path(data_dir))
    project = store.get(project_id)
    detail = store.details(project)
    episodes, spec = [], None
    for summary in sorted(detail["episodes"], key=lambda ep: (ep["run_id"], ep["id"])):
        if not summary.get("outcome") or successful_only and summary["outcome"] != "success":
            continue
        path = store._episode_path(summary["run_id"], summary["id"])
        manifest = _json(path.parent / "manifest.json")
        if not path.joinpath("transitions.jsonl").is_file():
            continue
        try:
            episode = load_episode(path, summary, manifest)
        except ValueError as exc:
            if str(exc) == "Episode has no executed task actions":
                continue
            raise
        candidate = _infer_spec(episode, manifest, image_size, state_fields, cameras)
        if spec is not None and candidate != spec:
            raise ValueError("Dataset contains incompatible camera/state/action representations")
        spec = candidate
        for observation in episode.observations:
            state_vector(observation["state"], spec)
            if any(name not in observation["images"] for name in spec.cameras):
                raise ValueError("A required camera is missing within an episode")
        if episode.actions.shape[1] != spec.action_dim:
            raise ValueError("Action dimension does not match the adapter specification")
        if np.any(episode.actions < np.asarray(spec.action_low) - 1e-6) or np.any(episode.actions > np.asarray(spec.action_high) + 1e-6):
            raise ValueError("Recorded executed action is outside adapter bounds")
        episodes.append(episode)
    if not episodes:
        raise ValueError("No eligible completed task trajectories. Collect data before training.")
    return make_snapshot(episodes, spec, project_id)


def split_episodes(episodes, validation_fraction=.2, seed=0):
    if len(episodes) < 2:
        raise ValueError("At least two episodes are required for an independent validation split")
    order = np.random.default_rng(seed).permutation(len(episodes))
    count = min(len(episodes)-1, max(1, round(len(episodes)*validation_fraction)))
    return [episodes[i] for i in order[count:]], [episodes[i] for i in order[:count]]


def state_statistics(episodes, spec):
    states = np.stack([state_vector(obs["state"], spec) for ep in episodes for obs in ep.observations])
    return states.mean(0), np.maximum(states.std(0), 1e-3)


class ActionWindowDataset(Dataset):
    def __init__(self, episodes, spec, obs_steps=2, horizon=16):
        self.episodes, self.spec = episodes, spec
        self.obs_steps, self.horizon = obs_steps, horizon
        self.indices = [(i, t) for i, ep in enumerate(episodes) for t in range(len(ep.actions))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        episode_index, t = self.indices[index]
        episode = self.episodes[episode_index]
        arrays = [observation_arrays(episode.observations[max(0, j)], self.spec, episode.path)
                  for j in range(t-self.obs_steps+1, t+1)]
        actions = np.zeros((self.horizon, self.spec.action_dim), dtype=np.float32)
        valid = min(self.horizon, len(episode.actions)-t)
        actions[:valid] = episode.actions[t:t+valid]
        if valid < self.horizon:
            actions[valid:] = episode.actions[-1]
        mask = np.arange(self.horizon) < valid
        return {"images": torch.from_numpy(np.stack([a[0] for a in arrays])),
                "states": torch.from_numpy(np.stack([a[1] for a in arrays])),
                "actions": torch.from_numpy(actions), "mask": torch.from_numpy(mask)}
