"""Append-only episode receipts and per-control-step observations on local disk."""
from __future__ import annotations

import json
from pathlib import Path
import time
import uuid
import hashlib

import numpy as np
from PIL import Image

from rilo.contracts import Observation


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def dump_json(value) -> str:
    return json.dumps(value, default=json_default, ensure_ascii=False, allow_nan=False)


class Recorder:
    def __init__(self, root: Path, config: dict, environment: dict):
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.path = root / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "manifest.json").write_text(dump_json({
            "schema_version": 1, "run_id": self.run_id, "created_at": time.time(),
            "config": config, "environment": environment,
        }), encoding="utf-8")
        self.episode_path: Path | None = None
        self._file = None
        self._frame_index = 0
        self.summaries: list[dict] = []
        self._excluded_episodes: set[str] = set()

    def exclude_episode(self, episode_id: str):
        """Forget deleted episode payloads without changing controller state.

        The controller may retain its last physical state for a verified-reset
        continuation. Later shutdown/initialization summaries must not put that
        deleted episode's in-memory summary back on disk. Billing receipts have
        an independent lifetime and are intentionally retained.
        """
        self._excluded_episodes.add(episode_id)
        self.summaries = [summary for summary in self.summaries if summary.get("id") != episode_id]
        if self.episode_path and self.episode_path.name == episode_id:
            self.close()

    def update_prepared_config(self, changes: dict):
        """Update an unstarted run's manifest and retain the setup-change receipt."""
        path = self.path / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        previous = {key: manifest["config"].get(key) for key in changes}
        manifest["config"].update(changes)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(dump_json(manifest), encoding="utf-8")
        temporary.replace(path)
        self.write({"kind": "control", "event": "prepared_config_updated",
                    "previous": previous, "changes": changes})

    def begin_episode(self, index: int, observation: Observation, seed: int, *, start_kind: str = "environment_reset"):
        self.episode_path = self.path / f"episode-{index:04d}"
        self.episode_path.mkdir()
        self._file = (self.episode_path / "transitions.jsonl").open("a", encoding="utf-8")
        self._frame_index = 0
        self.write({"kind": start_kind, "seed": seed, "observation": self.save_observation(observation)})

    def write_usage(self, event: dict):
        # Separate lifetime from episode files: cancelled/late requests may
        # report usage after a stage or run has ended.
        stored = dict(event)
        template = stored.pop("prompt_template", None)
        if template is not None:
            digest = hashlib.sha256(template.encode()).hexdigest()
            folder = self.path / "prompts"
            folder.mkdir(exist_ok=True)
            path = folder / (digest + ".txt")
            if not path.exists():
                path.write_text(template, encoding="utf-8")
            stored["prompt_sha256"] = digest
        with (self.path / "api_calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(dump_json(stored) + "\n")

    def write_summary(self, summary: dict):
        summary = dict(summary)
        if summary.get("episode", {}).get("id") in self._excluded_episodes:
            summary["episode"] = {}
        path = self.path / "summary.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(dump_json(summary), encoding="utf-8")
        temporary.replace(path)

    def save_observation(self, observation: Observation) -> dict:
        assert self.episode_path is not None
        paths = {}
        for camera, rgb in observation.images.items():
            # Never use an adapter's camera name as an unchecked path component.
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in camera)
            folder = self.episode_path / "frames" / safe_name
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{self._frame_index:06d}.jpg"
            Image.fromarray(rgb).save(path, quality=90)
            paths[camera] = str(path.relative_to(self.episode_path))
        self._frame_index += 1
        return {"id": observation.id, "timestamp": observation.timestamp,
                "task": observation.task, "state": observation.state, "images": paths,
                "success": observation.success, "terminated": observation.terminated, "info": observation.info}

    def write(self, event: dict):
        if self._file is not None:
            self._file.write(dump_json({"recorded_at": time.time(), **event}) + "\n")
            self._file.flush()

    def finish_episode(self, summary: dict):
        if summary.get("id") in self._excluded_episodes:
            self.close()
            return
        self.write({"kind": "episode_end", **summary})
        if self._file:
            self._file.close()
            self._file = None
        if self.episode_path:
            (self.episode_path / "summary.json").write_text(dump_json(summary), encoding="utf-8")
        self.summaries.append(summary)

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
