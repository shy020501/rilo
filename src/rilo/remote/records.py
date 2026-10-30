"""Executed remote transitions in the same format as local RILO episodes."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from io import BytesIO
import time

from PIL import Image

from rilo.grasp_count import GraspCounter
from rilo.recording import Recorder
from rilo.runner import _stats


class RemoteRecording:
    def __init__(self, settings, project, run_config, environment, *, policy_metadata, link_run):
        kind = run_config["policy"]
        learning = ({"pipeline_stage": "bc_eval" if kind == "dp" else "rl" if run_config["training"] else "rl_eval",
                     "base_model_id": run_config.get("model_id"), "residual_model_id": run_config.get("residual_model_id")}
                    if kind in ("dp", "residual") else {})
        self.recorder = Recorder(settings.data_dir, {**run_config, "backend": "remote_robot",
            "suite": project.get("suite", "custom"), "task_id": project["task_id"],
            "policy": "openai" if kind == "gpt" else kind, **(project.get("policy_config") or {}),
            "remote_policy": policy_metadata, "learning": learning}, environment)
        link_run(project["id"], self.recorder.run_id)
        self.environment, self.baseline = environment, project.get("grasp_baseline")
        self.stats, self.episode = _stats(), {}
        self.history, self.frames = deque(maxlen=8), {}
        self.last_seq, self.counter = 0, GraspCounter(environment.get("gripper_control"))
        self.phase = "task"
        self.closed = False

    @property
    def run_id(self):
        return self.recorder.run_id

    def capture(self, observation):
        for camera, pixels in observation.images.items():
            buffer = BytesIO()
            Image.fromarray(pixels).save(buffer, "JPEG", quality=85)
            self.frames[camera] = (observation.id, buffer.getvalue())

    def begin(self, index, observation, seed=0):
        self.phase, self.last_seq = "task", 0
        self.history.clear()
        self.counter = GraspCounter(self.environment.get("gripper_control"))
        self.episode = {"id": f"episode-{index:04d}", "index": index, "task": observation.task,
            "steps": 0, "reset_steps": 0, "decisions": 0, "reset_decisions": 0,
            "outcome": None, "reset_outcome": None, "started_at": time.time(),
            "costs": {"task": _stats(), "reset": _stats()}, "quality": {}}
        self.recorder.begin_episode(index, observation, seed, start_kind="observation")
        self.capture(observation)

    def usage(self, event):
        stage = event.get("stage", self.phase)
        self.recorder.write_usage({**event, "stage": stage, "episode_id": self.episode.get("id")})
        usage = event.get("usage") or {}
        for target in (self.stats, self.episode.get("costs", {}).get(stage, {})):
            target["api_requests"] = target.get("api_requests", 0) + 1
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                target[name] = target.get(name, 0) + int(usage.get(name) or 0)
            target["estimated_cost_usd"] = target.get("estimated_cost_usd", 0.) + float(event.get("estimated_cost_usd") or 0.)
            if not usage:
                target["unknown_usage_requests"] = target.get("unknown_usage_requests", 0) + 1

    def decision(self, metadata):
        key = "decisions" if self.phase == "task" else "reset_decisions"
        self.episode[key] += 1
        self.stats["total_decisions"] += 1
        self.recorder.write({"kind": "remote_decision", "stage": self.phase, **metadata})

    def transition(self, before, action, after, metadata):
        stage = metadata["stage"]
        if metadata["seq"] != self.last_seq + 1:
            raise ValueError("Executed transition sequence has a gap or duplicate")
        stored_before = self.recorder.save_observation(before)
        stored_after = self.recorder.save_observation(after)
        self.recorder.write({"kind": "observation", "stage": stage, "observation": stored_before})
        self.recorder.write({"kind": "transition", "stage": stage, "observation_id": before.id,
            "applied_action": action, "requested_action": action, "next_observation": stored_after,
            "remote": {key: value for key, value in metadata.items() if key not in ("observations", "session_id")}})
        self.last_seq = metadata["seq"]
        self.episode["steps" if stage == "task" else "reset_steps"] += 1
        self.stats["total_steps"] += 1
        self.counter.applied(action, stage)
        self.episode["quality"][stage] = self.counter.quality(stage, None, self.baseline)
        self.history.append({"stage": stage, "observation_id": before.id, "result_observation_id": after.id,
            "executed_steps": 1, "actions": [{"values": action, "repeat": 1}],
            "state_before": before.state, "state_after": after.state, "success": after.success})
        self.capture(after)

    def end_task(self, outcome, source):
        self.episode.update(outcome=outcome, outcome_source=source)
        self.episode["quality"]["task"] = self.counter.quality("task", outcome, self.baseline)
        self.recorder.write({"kind": "stage_end", "stage": "task", "outcome": outcome, "source": source})
        self.phase = "reset"
        self.history.clear()
        self.stats["reset_attempts"] += 1

    def finish(self, reset_outcome, *, arm_reset=False):
        if not self.episode or self.episode.get("finished_at"):
            return
        self.episode.update(reset_outcome=reset_outcome, environment_reset_done=arm_reset, finished_at=time.time())
        self.episode["outcome"] = self.episode["outcome"] or "interrupted"
        self.episode["quality"]["reset"] = self.counter.quality("reset", reset_outcome, self.baseline)
        self.stats["episodes"] += 1
        self.stats["successes"] += self.episode["outcome"] == "success"
        self.stats["reset_successes"] += reset_outcome == "success"
        self.stats["hard_resets"] += bool(arm_reset)
        self.recorder.finish_episode(deepcopy(self.episode))

    def summary(self, state):
        self.recorder.write_summary({"phase": state, "stage": self.phase, "stats": self.stats,
                                     "episode": self.episode, "episodes": self.recorder.summaries})

    def close(self, state="interrupted"):
        if self.closed:
            return
        self.finish("interrupted")
        self.summary(state)
        self.recorder.close()
        self.closed = True
