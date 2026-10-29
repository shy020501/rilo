"""Project-owned background exports of finalized recording episodes."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path

from fastapi import APIRouter, HTTPException

from rilo.video import ExportCancellation, export_episode, find_ffmpeg


class VideoJobs:
    def __init__(self, projects, is_run_busy):
        self.projects = projects
        self.is_run_busy = is_run_busy
        self._status = {}
        self._tasks = {}
        self._cancellations = {}
        self._episode_status = {}
        self._episode_tasks = {}
        self._episode_cancellations = {}
        self._closed = False

    def _run_job_busy(self, run_id):
        task = self._tasks.get(run_id)
        return task is not None and not task.done()

    def is_busy(self, run_id):
        return self._run_job_busy(run_id) or any(
            key[0] == run_id and not task.done() for key, task in self._episode_tasks.items())

    def _episode_job_busy(self, key):
        task = self._episode_tasks.get(key)
        return task is not None and not task.done()

    def _owned_run(self, project_id, run_id):
        try:
            project = self.projects.get(project_id)
            if run_id not in project["run_ids"]:
                raise KeyError(run_id)
            path = self.projects._run_path(run_id)
            if path.is_symlink() or not path.is_dir():
                raise ValueError("Recording directory is missing or is a symlink")
            return path
        except KeyError:
            raise HTTPException(404, "Run does not belong to this project") from None
        except (ValueError, OSError):
            raise HTTPException(409, "Recording path cannot be exported safely") from None

    @staticmethod
    def _source_file(path, root):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Recording source must be a regular file inside its episode")
        return path

    def _episodes(self, root):
        state = self.projects._load()
        result = []
        for path in sorted(root.glob("episode-*")):
            metadata = self.projects._episode_state(root.name, path.name, state=state)
            if metadata.get("deleted_at") or metadata.get("delete_pending"):
                continue
            path = self.projects._episode_path(root.name, path.name)
            if not (path / "summary.json").exists() or not (path / "transitions.jsonl").exists():
                continue
            summary_file = self._source_file(path / "summary.json", path)
            self._source_file(path / "transitions.jsonl", path)
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            if summary.get("id") != path.name or not summary.get("outcome"):
                continue
            result.append(path)
        return result

    def _owned_episode(self, project_id, run_id, episode_id):
        self._owned_run(project_id, run_id)
        try:
            path = self.projects._episode_path(run_id, episode_id)
            metadata = self.projects._episode_state(run_id, episode_id)
            if metadata.get("deleted_at") or not path.is_dir():
                raise HTTPException(404, "Episode recording does not exist")
            if metadata.get("delete_pending"):
                raise HTTPException(409, "Episode deletion is pending")
            return path
        except (ValueError, OSError):
            raise HTTPException(409, "Episode path cannot be exported safely") from None

    def _episode_finalized(self, episode):
        if not (episode / "summary.json").exists() or not (episode / "transitions.jsonl").exists():
            return False
        summary = json.loads(self._source_file(episode / "summary.json", episode).read_text(encoding="utf-8"))
        self._source_file(episode / "transitions.jsonl", episode)
        return summary.get("id") == episode.name and bool(summary.get("outcome"))

    def _has_output(self, episode):
        output = self.projects.media_dir
        if output.is_symlink():
            return False
        for path in output.glob("*.json"):
            try:
                if path.is_symlink():
                    continue
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if (metadata.get("run_id"), metadata.get("episode_id")) != (episode.parent.name, episode.name):
                    continue
                frames = metadata.get("frame_count", 0)
                if (not isinstance(frames, int) or frames < 1
                        or metadata.get("decoded_frame_count") != frames):
                    continue
                video = self.projects.media_path(metadata["video"])
                poster = self.projects.media_path(metadata["poster"])
                if (not video.is_symlink() and not poster.is_symlink() and video.suffix == ".mp4"
                        and video.stat().st_size > 0 and poster.stat().st_size > 0
                        and video.stat().st_size == metadata.get("bytes")):
                    return True
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
        return False

    def _snapshot(self, run_id, state, completed=0, total=0, error=None):
        return {"run_id": run_id, "state": state, "completed_episodes": completed,
                "total_episodes": total, "error": error}

    def status(self, project_id, run_id):
        root = self._owned_run(project_id, run_id)
        if self._run_job_busy(run_id):
            return dict(self._status[run_id])
        try:
            episodes = self._episodes(root)
            complete = sum(self._has_output(episode) for episode in episodes)
        except (OSError, ValueError, KeyError, TypeError):
            return self._snapshot(run_id, "error", error="Recording sources are missing or invalid")
        if self.is_busy(run_id):
            return self._snapshot(run_id, "running", complete, len(episodes))
        if episodes and complete == len(episodes):
            return self._snapshot(run_id, "completed", complete, len(episodes))
        previous = self._status.get(run_id)
        if previous and previous["state"] == "error":
            return {**previous, "completed_episodes": complete, "total_episodes": len(episodes)}
        return self._snapshot(run_id, "idle", complete, len(episodes))

    def start(self, project_id, run_id):
        if self._closed:
            raise HTTPException(503, "Video generation is shutting down")
        root = self._owned_run(project_id, run_id)
        if self._run_job_busy(run_id):
            return dict(self._status[run_id])
        if self.is_busy(run_id):
            raise HTTPException(409, "An episode video is already being generated in this run")
        if self.is_run_busy(run_id):
            raise HTTPException(409, "Stop the active run or pipeline before generating videos")
        try:
            episodes = self._episodes(root)
            if not episodes:
                raise ValueError("No finalized episodes with recordings are available")
            manifest = json.loads(self._source_file(root / "manifest.json", root).read_text(encoding="utf-8"))
            fps = float(manifest["environment"]["action_spec"]["control_hz"])
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Recording has no valid control frame rate")
            if self.projects.media_dir.is_symlink():
                raise ValueError("Video output directory must not be a symlink")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(409, str(exc)[:300]) from None
        missing = [episode for episode in episodes if not self._has_output(episode)]
        status = self._snapshot(run_id, "queued" if missing else "completed", len(episodes) - len(missing), len(episodes))
        self._status[run_id] = status
        if missing:
            cancellation = ExportCancellation()
            self._cancellations[run_id] = cancellation
            self._tasks[run_id] = asyncio.create_task(
                self._run(run_id, missing, fps, cancellation), name=f"rilo-video-{run_id}")
        return dict(status)

    def episode_status(self, project_id, run_id, episode_id):
        episode = self._owned_episode(project_id, run_id, episode_id)
        key = (run_id, episode_id)
        if self._episode_job_busy(key):
            return dict(self._episode_status[key])
        try:
            finalized = self._episode_finalized(episode)
            complete = finalized and self._has_output(episode)
        except (OSError, ValueError, KeyError, TypeError):
            return {**self._snapshot(run_id, "error", error="Episode sources are missing or invalid"),
                    "episode_id": episode_id}
        if complete:
            state = self._snapshot(run_id, "completed", 1, 1)
        elif self._run_job_busy(run_id):
            state = self._snapshot(run_id, "running", 0, int(finalized))
        elif key in self._episode_status and self._episode_status[key]["state"] == "error":
            state = {**self._episode_status[key], "completed_episodes": 0, "total_episodes": int(finalized)}
        else:
            state = self._snapshot(run_id, "idle", 0, int(finalized))
        return {**state, "episode_id": episode_id}

    def start_episode(self, project_id, run_id, episode_id):
        if self._closed:
            raise HTTPException(503, "Video generation is shutting down")
        episode = self._owned_episode(project_id, run_id, episode_id)
        key = (run_id, episode_id)
        if self._episode_job_busy(key):
            return dict(self._episode_status[key])
        if self._run_job_busy(run_id):
            raise HTTPException(409, "A run video export is already generating this episode")
        if self.is_run_busy(run_id):
            raise HTTPException(409, "Stop the active run or pipeline before generating a video")
        try:
            if not self._episode_finalized(episode):
                raise ValueError("This episode has no finalized recording")
            manifest = json.loads(self._source_file(episode.parent / "manifest.json", episode.parent).read_text(encoding="utf-8"))
            fps = float(manifest["environment"]["action_spec"]["control_hz"])
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Recording has no valid control frame rate")
            if self.projects.media_dir.is_symlink():
                raise ValueError("Video output directory must not be a symlink")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(409, str(exc)[:300]) from None
        complete = self._has_output(episode)
        status = {**self._snapshot(run_id, "completed" if complete else "queued", int(complete), 1),
                  "episode_id": episode_id}
        self._episode_status[key] = status
        if not complete:
            cancellation = ExportCancellation()
            self._episode_cancellations[key] = cancellation
            self._episode_tasks[key] = asyncio.create_task(
                self._run_episode(key, episode, fps, cancellation), name=f"rilo-video-{run_id}-{episode_id}")
        return dict(status)

    def delete_episode_video(self, project_id, run_id, episode_id):
        episode = self._owned_episode(project_id, run_id, episode_id)
        if self.is_run_busy(run_id) or self.is_busy(run_id):
            raise HTTPException(409, "Wait for collection and video generation to finish before deleting video files")
        output = self.projects.media_dir
        try:
            if output.is_symlink():
                raise ValueError("Video output directory must not be a symlink")
            metadata_files, candidates, shared = [], set(), set()
            for path in output.glob("*.json"):
                if path.is_symlink():
                    raise ValueError("Cannot delete video metadata through a symlink")
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict):
                    raise ValueError("Invalid video metadata")
                names = {metadata[k] for k in ("video", "poster") if isinstance(metadata.get(k), str)}
                if (metadata.get("run_id"), metadata.get("episode_id")) == (run_id, episode_id):
                    metadata_files.append(path)
                    for name in names:
                        if Path(name).name != name or Path(name).suffix.lower() not in (".mp4", ".jpg", ".jpeg", ".png"):
                            raise ValueError("Invalid generated media filename")
                        candidate = output / name
                        if candidate.is_symlink() or not candidate.resolve().is_relative_to(output.resolve()):
                            raise ValueError("Cannot delete generated media through a symlink")
                        if candidate.exists() and not candidate.is_file():
                            raise ValueError("Generated media must be a regular file")
                        candidates.add(candidate)
                else:
                    shared.update(names)
            # Validate every target before the first mutation. Shared media
            # remains available to its other episode's metadata.
            for candidate in candidates:
                if candidate.name not in shared:
                    candidate.unlink(missing_ok=True)
            for metadata in metadata_files:
                metadata.unlink(missing_ok=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(409, str(exc)[:300]) from None
        key = (run_id, episode_id)
        self._episode_status.pop(key, None)
        self._status.pop(run_id, None)
        finalized_files = (episode / "summary.json").is_file() and (episode / "transitions.jsonl").is_file()
        return {**self._snapshot(run_id, "idle", 0, int(finalized_files)),
                "episode_id": episode_id}

    def _convert(self, episode, fps, cancellation):
        cancellation.check()
        basename = "rilo-" + hashlib.sha256(f"{episode.parent.name}/{episode.name}".encode()).hexdigest()[:24]
        return export_episode(episode, self.projects.media_dir, basename,
                              find_ffmpeg(None), None, fps, cancellation=cancellation)

    async def _run(self, run_id, episodes, fps, cancellation):
        status = self._status[run_id]
        status["state"] = "running"
        try:
            for episode in episodes:
                cancellation.check()
                worker = asyncio.create_task(asyncio.to_thread(self._convert, episode, fps, cancellation))
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    cancellation.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
                    raise
                status["completed_episodes"] += 1
            status["state"] = "completed"
        except asyncio.CancelledError:
            status.update(state="error", error="Video generation interrupted")
        except Exception as exc:
            status.update(state="error", error=str(exc)[:1000] or type(exc).__name__)
        finally:
            self._cancellations.pop(run_id, None)

    async def _run_episode(self, key, episode, fps, cancellation):
        status = self._episode_status[key]
        status["state"] = "running"
        try:
            worker = asyncio.create_task(asyncio.to_thread(self._convert, episode, fps, cancellation))
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancellation.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                raise
            status.update(state="completed", completed_episodes=1)
        except asyncio.CancelledError:
            status.update(state="error", error="Video generation interrupted")
        except Exception as exc:
            status.update(state="error", error=str(exc)[:1000] or type(exc).__name__)
        finally:
            self._episode_cancellations.pop(key, None)

    async def close(self):
        self._closed = True
        for cancellation in list(self._cancellations.values()):
            cancellation.cancel()
        for cancellation in list(self._episode_cancellations.values()):
            cancellation.cancel()
        await asyncio.gather(*self._tasks.values(), *self._episode_tasks.values(), return_exceptions=True)


def create_video_router(projects, is_run_busy):
    router, jobs = APIRouter(), VideoJobs(projects, is_run_busy)

    @router.get("/api/projects/{project_id}/runs/{run_id}/video")
    async def video_status(project_id: str, run_id: str):
        return {"video_job": jobs.status(project_id, run_id)}

    @router.post("/api/projects/{project_id}/runs/{run_id}/video")
    async def generate_video(project_id: str, run_id: str):
        return {"video_job": jobs.start(project_id, run_id)}

    @router.get("/api/projects/{project_id}/episodes/{run_id}/{episode_id}/video")
    async def episode_video_status(project_id: str, run_id: str, episode_id: str):
        return {"video_job": jobs.episode_status(project_id, run_id, episode_id)}

    @router.post("/api/projects/{project_id}/episodes/{run_id}/{episode_id}/video")
    async def generate_episode_video(project_id: str, run_id: str, episode_id: str):
        return {"video_job": jobs.start_episode(project_id, run_id, episode_id)}

    @router.post("/api/projects/{project_id}/episodes/{run_id}/{episode_id}/video/delete")
    async def delete_episode_video(project_id: str, run_id: str, episode_id: str):
        return {"video_job": jobs.delete_episode_video(project_id, run_id, episode_id)}

    return router, jobs
