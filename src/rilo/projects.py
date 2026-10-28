"""Project metadata and explicit episode lifecycle operations.

Projects own references to runs, never the run files themselves. The registry
can therefore organize historical data without migrating or rewriting evidence.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
from typing import Any
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rilo.runner import RunConfig
from rilo.policy_settings import PolicyConfig


CAPABILITIES = {"collection": True, "bc": True, "progress": True,
                "evaluation": True, "rl": True}


class PipelineGoals(BaseModel):
    model_config = ConfigDict(extra="forbid")
    demo_target: int = Field(default=20, ge=1, le=1000000)
    bc_success_target: float = Field(default=70, gt=0, le=100, allow_inf_nan=False)
    rl_success_target: float = Field(default=90, gt=0, le=100, allow_inf_nan=False)


class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    backend: str = Field(default="libero", min_length=1, max_length=200)
    suite: str = Field(default="libero_90", max_length=100)
    task_id: int = Field(default=7, ge=0, le=10000)
    task_name: str = Field(default="", max_length=300)
    run_config: RunConfig = Field(default_factory=RunConfig)
    pipeline: PipelineGoals = Field(default_factory=PipelineGoals)
    policy_config: PolicyConfig | None = None

    @field_validator("name", "backend")
    @classmethod
    def nonempty(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Must not be blank")
        return value

    @model_validator(mode="after")
    def task_is_authoritative(self):
        self.run_config.task_id = self.task_id
        return self


class ProjectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    run_config: RunConfig | None = None
    pipeline: PipelineGoals | None = None
    policy_config: PolicyConfig | None = None

    @field_validator("name")
    @classmethod
    def nonempty(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Must not be blank")
        return value.strip() if value is not None else value


def empty_stats():
    return {"runs": 0, "episodes": 0, "successes": 0, "reset_successes": 0,
            "total_steps": 0, "total_tokens": 0, "input_tokens": 0,
            "output_tokens": 0, "api_requests": 0, "estimated_cost_usd": 0.0,
            "unknown_usage_requests": 0}


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else 0


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class ProjectStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.path = self.data_dir.parent / "projects.json"
        self.media_dir = self.data_dir.parent / "videos"
        self._lock = threading.RLock()

    def _load(self):
        if not self.path.exists():
            return {"schema_version": 1, "projects": []}
        state = _json(self.path)
        if state.get("schema_version") != 1 or not isinstance(state.get("projects"), list):
            raise ValueError("Unsupported project registry; existing metadata was not changed")
        return state

    def _save(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".projects-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, allow_nan=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _run_path(self, run_id):
        if not isinstance(run_id, str) or run_id in ("", ".", "..") or Path(run_id).name != run_id:
            raise ValueError("Invalid run identifier")
        path = self.data_dir / run_id
        if not path.resolve().is_relative_to(self.data_dir.resolve()):
            raise ValueError("Run path must remain inside the recordings directory")
        return path

    def _episode_path(self, run_id, episode_id):
        if not isinstance(episode_id, str) or not re.fullmatch(r"episode-\d{4,}", episode_id):
            raise ValueError("Invalid episode identifier")
        root = self._run_path(run_id)
        path = root / episode_id
        if root.is_symlink() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Episode path must remain inside its run")
        return path

    def _episode_state(self, run_id, episode_id, *, state=None):
        state = self._load() if state is None else state
        return state.get("episode_state", {}).get(run_id, {}).get(episode_id, {})

    def _require_episode(self, project_id, run_id, episode_id):
        self._episode_path(run_id, episode_id)
        project = self.get(project_id)
        if run_id not in project["run_ids"]:
            raise KeyError("Episode does not belong to this project")
        run = self.read_run(run_id)
        episode = next((ep for ep in run["episodes"] + run["archived_episodes"]
                        if ep["id"] == episode_id and not ep.get("delete_pending")), None)
        if episode is None:
            raise KeyError("Episode does not exist")
        return project, run, episode

    def archive_episode(self, project_id, run_id, episode_id, *, archived):
        with self._lock:
            self._require_episode(project_id, run_id, episode_id)
            state = self._load()
            metadata = state.setdefault("episode_state", {}).setdefault(run_id, {}).setdefault(episode_id, {})
            if archived:
                metadata["archived_at"] = time.time()
            else:
                metadata.pop("archived_at", None)
            self._save(state)

    def _deletion_files(self, run_id, episode_id):
        """Resolve and validate all affected files before changing any data."""
        root = self._run_path(run_id)
        episode = self._episode_path(run_id, episode_id)
        if root.is_symlink() or episode.is_symlink():
            raise ValueError("Cannot delete through a run or episode symlink")
        if episode.exists() and not episode.is_dir():
            raise ValueError("Episode recording must be a directory")
        if (root / "summary.json").is_symlink():
            raise ValueError("Cannot rewrite a summary symlink")
        target_metadata, candidates, shared = [], set(), set()
        for path in self.media_dir.glob("*.json"):
            try:
                metadata = _json(path)
                identity = (metadata.get("run_id"), metadata.get("episode_id"))
                names = {metadata[k] for k in ("video", "poster") if isinstance(metadata.get(k), str)}
            except (OSError, ValueError, AttributeError, TypeError):
                continue
            if identity == (run_id, episode_id):
                if self.media_dir.is_symlink() or path.is_symlink():
                    raise ValueError("Cannot delete media through a symlink")
                target_metadata.append(path)
                for filename in names:
                    if Path(filename).name != filename or Path(filename).suffix.lower() not in (".mp4", ".jpg", ".jpeg", ".png"):
                        raise ValueError("Invalid recorded media filename")
                    media = self.media_dir / filename
                    if media.is_symlink() or not media.resolve().is_relative_to(self.media_dir.resolve()):
                        raise ValueError("Cannot delete media outside the recording directory")
                    if media.exists() and not media.is_file():
                        raise ValueError("Recorded media must be a file")
                    candidates.add(media)
            else:
                shared.update(names)
        return episode, target_metadata, [path for path in candidates if path.name not in shared]

    def delete_episode(self, project_id, run_id, episode_id):
        """Delete payloads; keep usage receipts and a small non-resurrection tombstone.

        A pending tombstone survives partial filesystem failures. Retrying the
        same explicit delete completes cleanup without reporting false success.
        """
        with self._lock:
            self._episode_path(run_id, episode_id)
            project = self.get(project_id)
            if run_id not in project["run_ids"]:
                raise KeyError("Episode does not belong to this project")
            state = self._load()
            existing = self._episode_state(run_id, episode_id, state=state)
            if existing.get("deleted_at") and not existing.get("delete_pending"):
                return
            retained_usage = existing.get("retained_usage", {})
            if not existing.get("deleted_at"):
                _, _, summary = self._require_episode(project_id, run_id, episode_id)
                retained_usage = {key: sum(_number(stage.get(key)) for stage in summary.get("costs", {}).values())
                    for key in ("input_tokens", "output_tokens", "total_tokens", "api_requests",
                                "estimated_cost_usd", "unknown_usage_requests")}
            episode, metadata_paths, media_paths = self._deletion_files(run_id, episode_id)
            metadata = state.setdefault("episode_state", {}).setdefault(run_id, {}).setdefault(episode_id, {})
            metadata.update(deleted_at=existing.get("deleted_at", time.time()), delete_pending=True,
                            retained_usage=retained_usage)
            self._save(state)
            if episode.exists():
                shutil.rmtree(episode)
            for path in media_paths + metadata_paths:
                path.unlink(missing_ok=True)
            # The run summary duplicates its last episode. Remove that copy too;
            # the tombstone also protects against a late legacy summary rewrite.
            live_path = self._run_path(run_id) / "summary.json"
            if live_path.is_file():
                if live_path.is_symlink():
                    raise ValueError("Cannot rewrite a summary symlink")
                live = _json(live_path)
                if live.get("episode", {}).get("id") == episode_id:
                    live["episode"] = {}
                    fd, temporary = tempfile.mkstemp(prefix=".summary-delete-", suffix=".tmp", dir=live_path.parent)
                    try:
                        with os.fdopen(fd, "w", encoding="utf-8") as handle:
                            json.dump(live, handle, ensure_ascii=False, allow_nan=False)
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(temporary, live_path)
                    finally:
                        if os.path.exists(temporary):
                            os.unlink(temporary)
            metadata["delete_pending"] = False
            self._save(state)

    def episode_detail(self, project_id, run_id, episode_id, *, offset=0, limit=50):
        _, run, episode = self._require_episode(project_id, run_id, episode_id)
        records = []
        path = self._episode_path(run_id, episode_id) / "transitions.jsonl"
        if path.is_symlink() or not path.resolve().is_relative_to(self._episode_path(run_id, episode_id).resolve()):
            raise ValueError("Invalid episode recording path")
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    kind = row.get("kind")
                    if kind in ("environment_reset", "agent_reset_continuation"):
                        row = {"kind": "episode_start", "recorded_at": row.get("recorded_at"),
                               "stage": "task", "start_kind": kind, "seed": row.get("seed")}
                    elif kind not in ("decision", "receipt", "stage_start", "stage_end", "episode_end", "control", "policy_transition"):
                        continue
                    # Decisions contain the public action explanation (note),
                    # never the provider's hidden reasoning. Large per-tick
                    # observations remain in the recording, not this endpoint.
                    row = {key: value for key, value in row.items() if key not in ("observation", "next_observation")}
                    records.append({"record_id": len(records), **row})
        selected = records[offset:offset + limit]
        next_offset = offset + limit if offset + limit < len(records) else None
        return {"episode": episode, "config": run["config"], "records": selected,
                "total_records": len(records), "offset": offset, "limit": limit, "next_offset": next_offset}

    def import_history(self):
        """Idempotently link metered legacy runs; ignore unmetered smoke tests."""
        with self._lock:
            state = self._load()
            assigned = {run for p in state["projects"] for run in p["run_ids"]}
            changed = False
            for ledger in sorted(self.data_dir.glob("*/api_calls.jsonl")):
                run_id = ledger.parent.name
                if run_id in assigned:
                    continue
                try:
                    manifest = _json(self._run_path(run_id) / "manifest.json")
                    config = manifest["config"]
                    backend = config.get("backend", manifest.get("environment", {}).get("name", "unknown"))
                    suite, task_id = config.get("suite", ""), config.get("task_id", 0)
                    task = manifest.get("environment", {}).get("task", f"Task {task_id}")
                    key = [backend, suite, task_id]
                    project = next((p for p in state["projects"] if p.get("import_key") == key), None)
                    if project is None:
                        label = ("상단 서랍" if "top drawer" in task else
                                 "중간 서랍" if "middle drawer" in task else task)
                        run_config = RunConfig.model_validate({k: v for k, v in config.items() if k in RunConfig.model_fields})
                        created = ProjectCreate(name=f"[{backend.upper()}] {label}",
                            description="기존 실행 기록을 연결한 프로젝트입니다.", backend=backend,
                            suite=suite, task_id=task_id, task_name=task, run_config=run_config)
                        project = self._new(created, created_at=manifest.get("created_at", time.time()))
                        project["import_key"] = key
                        state["projects"].append(project)
                    project["run_ids"].append(run_id)
                    project["updated_at"] = max(project["updated_at"], manifest.get("created_at", 0))
                    assigned.add(run_id)
                    changed = True
                except (OSError, ValueError, KeyError, TypeError):
                    continue
            if changed:
                self._save(state)

    def _new(self, data: ProjectCreate, *, created_at=None):
        now = time.time() if created_at is None else created_at
        return {"id": uuid.uuid4().hex, **data.model_dump(),
                "created_at": now, "updated_at": now, "run_ids": []}

    def create(self, data: ProjectCreate):
        with self._lock:
            state = self._load()
            project = self._new(data)
            state["projects"].append(project)
            self._save(state)
            return deepcopy(project)

    def list(self):
        with self._lock:
            self.import_history()
            return sorted(deepcopy(self._load()["projects"]), key=lambda p: p["updated_at"], reverse=True)

    def get(self, project_id):
        for project in self.list():
            if project["id"] == project_id:
                return project
        raise KeyError(project_id)

    def update(self, project_id, data: ProjectSettings):
        with self._lock:
            state = self._load()
            project = next((p for p in state["projects"] if p["id"] == project_id), None)
            if project is None:
                raise KeyError(project_id)
            changes = data.model_dump(exclude_none=True)
            if "run_config" in changes:
                changes["run_config"]["task_id"] = project["task_id"]
            project.update(changes, updated_at=time.time())
            self._save(state)
            return deepcopy(project)

    def link_run(self, project_id, run_id):
        self._run_path(run_id)
        with self._lock:
            state = self._load()
            project = next((p for p in state["projects"] if p["id"] == project_id), None)
            if project is None:
                raise KeyError(project_id)
            if any(run_id in p["run_ids"] for p in state["projects"] if p["id"] != project_id):
                raise ValueError("Run already belongs to a different project")
            if run_id not in project["run_ids"]:
                project["run_ids"].append(run_id)
                project["updated_at"] = time.time()
                self._save(state)

    def media_path(self, filename):
        if Path(filename).name != filename or Path(filename).suffix.lower() not in (".mp4", ".jpg", ".jpeg", ".png"):
            raise ValueError("Invalid media filename")
        path = self.media_dir / filename
        if not path.resolve().is_relative_to(self.media_dir.resolve()) or not path.is_file():
            raise FileNotFoundError(filename)
        return path

    def media_index(self):
        result = {}
        for path in self.media_dir.glob("*.json"):
            try:
                metadata = _json(path)
                key = (metadata["run_id"], metadata["episode_id"])
                links = {}
                for name in ("video", "poster"):
                    filename = metadata.get(name)
                    if filename:
                        self.media_path(filename)
                        links[f"{name}_url"] = f"/api/media/{filename}"
                result[key] = links
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return result

    def read_run(self, run_id, *, media=None, episode_state=None):
        root = self._run_path(run_id)
        manifest = _json(root / "manifest.json")
        episode_state = (self._load().get("episode_state", {}).get(run_id, {})
                         if episode_state is None else episode_state)
        summaries = []
        for path in sorted(root.glob("episode-*/summary.json")):
            try:
                summaries.append(_json(path))
            except (OSError, ValueError):
                continue
        # A running/resetting episode may only have a live run summary so far.
        try:
            live = _json(root / "summary.json").get("episode", {})
            if live.get("id") and live["id"] not in {s.get("id") for s in summaries}:
                summaries.append(live)
        except (OSError, ValueError, AttributeError):
            pass
        media = self.media_index() if media is None else media
        episodes = [{**s, "run_id": run_id,
                     "archived": bool(episode_state.get(s.get("id"), {}).get("archived_at")),
                     "archived_at": episode_state.get(s.get("id"), {}).get("archived_at"),
                     "video_url": None, "poster_url": None,
                     **media.get((run_id, s.get("id")), {})} for s in summaries
                    if not episode_state.get(s.get("id"), {}).get("deleted_at")]
        for episode_id, metadata in episode_state.items():
            if metadata.get("delete_pending"):
                episodes.append({"id": episode_id, "index": int(episode_id.split("-")[-1]),
                    "run_id": run_id, "archived": True, "delete_pending": True,
                    "archived_at": metadata.get("archived_at") or metadata.get("deleted_at"),
                    "video_url": None, "poster_url": None})
        stats = empty_stats()
        visible = [episode for episode in episodes if not episode["archived"]]
        archived = [episode for episode in episodes if episode["archived"]]
        stats["runs"] = int(bool(visible))
        stats["episodes"] = len(visible)
        stats["successes"] = sum(s.get("outcome") == "success" for s in visible)
        stats["reset_successes"] = sum(s.get("reset_outcome") == "success" for s in visible)
        stats["total_steps"] = sum(_number(s.get("steps")) + _number(s.get("reset_steps")) for s in visible)
        ledger = root / "api_calls.jsonl"
        if ledger.exists():
            seen = set()
            stage_usage = {}
            usage_fields = ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_tokens",
                            "reasoning_tokens", "total_tokens", "api_requests", "estimated_cost_usd", "unknown_usage_requests")
            with ledger.open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue  # A writer may currently be appending its last line.
                    identity = event.get("attempt_id") or f"line-{index}"
                    if identity in seen:
                        continue
                    seen.add(identity)
                    stats["api_requests"] += 1
                    tokens = event.get("tokens") or {}
                    for key in ("input_tokens", "output_tokens", "total_tokens"):
                        stats[key] += _number(tokens.get(key))
                    cost = event.get("estimated_cost_usd")
                    stats["estimated_cost_usd"] += _number(cost)
                    stats["unknown_usage_requests"] += int(event.get("cost_unknown", cost is None))
                    if event.get("episode_id") and event.get("stage") in ("task", "reset"):
                        key = (event["episode_id"], event["stage"])
                        usage = stage_usage.setdefault(key, {field: 0 for field in usage_fields})
                        for field in usage_fields[:6]:
                            usage[field] += _number(tokens.get(field))
                        usage["api_requests"] += 1
                        usage["estimated_cost_usd"] += _number(cost)
                        usage["unknown_usage_requests"] += int(event.get("cost_unknown", cost is None))
            # API receipts can arrive after episode closure. Apply the same
            # authoritative ledger to episode detail and aggregate totals.
            for episode in episodes:
                if episode.get("delete_pending"):
                    continue
                for stage in ("task", "reset"):
                    usage = stage_usage.get((episode.get("id"), stage))
                    if usage is not None:
                        previous = episode.setdefault("costs", {}).get(stage, {})
                        episode["costs"][stage] = {**previous, **usage}
        else:
            for episode in episodes:
                for stage in episode.get("costs", {}).values():
                    for key in ("input_tokens", "output_tokens", "total_tokens", "api_requests",
                                "estimated_cost_usd", "unknown_usage_requests"):
                        stats[key] += _number(stage.get(key))
            for metadata in episode_state.values():
                if metadata.get("deleted_at"):
                    for key, value in metadata.get("retained_usage", {}).items():
                        if key in stats:
                            stats[key] += _number(value)
        return {"run_id": run_id, "created_at": manifest.get("created_at"),
                "config": manifest.get("config", {}), "episodes": visible,
                "archived_episodes": archived, "stats": stats}

    def details(self, project):
        media = self.media_index()
        episode_state = self._load().get("episode_state", {})
        runs = []
        for run_id in reversed(project["run_ids"]):
            try:
                runs.append(self.read_run(run_id, media=media, episode_state=episode_state.get(run_id, {})))
            except (OSError, ValueError, KeyError):
                continue
        episodes = [episode for run in runs for episode in run["episodes"]]
        archived = [episode for run in runs for episode in run["archived_episodes"]]
        stats = empty_stats()
        for run in runs:
            for key in stats:
                stats[key] += run["stats"][key]
        enriched = {**project, "stats": stats, "archived_count": len(archived),
                    "usage_scope": "all_recorded_episodes", "thumbnail_url": next(
            (s["poster_url"] for s in episodes if s.get("poster_url")), None)}
        return {"project": enriched, "runs": runs, "episodes": episodes, "archived_episodes": archived}
