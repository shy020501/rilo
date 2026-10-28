"""The environment-independent, single-owner observation/action loop."""
from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from io import BytesIO
import time
from typing import Callable, Literal
import hashlib
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from rilo.config import Settings, make_adapter, make_policy
from rilo.contracts import Decision, EnvironmentAdapter, Observation, Policy
from rilo.recording import Recorder, dump_json
from rilo.policy_settings import PolicyConfig


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: int = Field(default=0, ge=0, le=10000)
    seed: int = Field(default=0, ge=0, le=2**31-10000)
    max_steps: int = Field(default=300, ge=1, le=10000)
    max_decisions: int = Field(default=50, ge=1, le=10000)
    episodes: int = Field(default=1, ge=1, le=100)
    reset_mode: Literal["agent", "environment"] = "agent"
    max_reset_steps: int = Field(default=300, ge=1, le=10000)
    max_reset_decisions: int = Field(default=40, ge=1, le=1000)
    budget_usd: float = Field(default=15, gt=0, le=1000, allow_inf_nan=False)


class StateConflict(ValueError):
    pass


def _stats():
    return dict(episodes=0, successes=0, total_steps=0, total_decisions=0,
                input_tokens=0, output_tokens=0, cached_input_tokens=0,
                cache_write_tokens=0, reasoning_tokens=0, total_tokens=0,
                api_requests=0, estimated_cost_usd=0.0, unknown_usage_requests=0,
                reset_attempts=0, reset_successes=0, hard_resets=0)


class Runner:
    def __init__(self, settings: Settings, *, adapter_factory: Callable[[], EnvironmentAdapter] | None = None,
                 policy_factory: Callable[[], Policy] | None = None):
        self.settings = settings
        self._adapter_factory = adapter_factory or partial(make_adapter, settings)
        self._policy_factory = policy_factory
        self._policy_settings = settings
        self.adapter: EnvironmentAdapter | None = None
        self.policy: Policy | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rilo-environment")
        self._control_lock = asyncio.Lock()
        self._permit = asyncio.Event()
        self._epoch = 0
        self._stopping = False
        self._single = False
        self._task: asyncio.Task | None = None
        self._inference: asyncio.Task | None = None
        self._decision_interrupt: asyncio.Event | None = None
        self._retired_requests: set[asyncio.Task] = set()
        self._policy_cleanup_tasks: set[asyncio.Task] = set()
        self.phase = "idle"
        self.stage = "task"
        self.error: str | None = None
        self.config = RunConfig(task_id=settings.task_id)
        self.observation: Observation | None = None
        self.context: dict = {}
        self.episode: dict = {}
        self.stats = _stats()
        self.latest_decision: dict | None = None
        self.latest_receipt: dict | None = None
        self.history: deque = deque(maxlen=8)
        self.events: deque = deque(maxlen=100)
        self._event_id = 0
        self.frames: dict[str, bytes] = {}
        self.recorder: Recorder | None = None
        self._run_policy_override = False
        self._recording_metadata: dict = {}

    async def _env(self, method, *args, **kwargs):
        # MuJoCo's rendering context is thread-affine. Never use asyncio.to_thread
        # (which may pick a different worker) to access the environment.
        return await asyncio.shield(asyncio.get_running_loop().run_in_executor(
            self._executor, partial(method, *args, **kwargs)))

    async def _ensure_adapter(self):
        if self.adapter is None:
            self.adapter = await self._env(self._adapter_factory)

    def _event(self, kind: str, message: str):
        self._event_id += 1
        self.events.append({"id": self._event_id, "time": time.time(), "kind": kind, "message": message,
                            "run_id": self.recorder.run_id if self.recorder else None,
                            "episode_id": self.episode.get("id"), "stage": self.stage})

    @property
    def policy_config(self) -> dict:
        """Resolved, immutable model selection of the prepared/active run."""
        return {"model": self._policy_settings.model,
                "reasoning_effort": self._policy_settings.reasoning_effort}

    def _release_policy(self):
        """Detach a policy without disrupting its retired metering callbacks."""
        policy, self.policy = self.policy, None
        if policy is None:
            return
        pending = tuple(self._retired_requests)

        async def close_after_receipts():
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await policy.close()

        task = asyncio.create_task(close_after_receipts(), name="rilo-policy-cleanup")
        self._policy_cleanup_tasks.add(task)

        def completed(finished):
            self._policy_cleanup_tasks.discard(finished)
            if not finished.cancelled():
                finished.exception()
        task.add_done_callback(completed)

    def _safe_error(self, exc: Exception) -> str:
        text = str(exc) or type(exc).__name__
        if self.settings.api_key:
            text = text.replace(self.settings.api_key, "[redacted]")
        return text[:1500]

    def _capture(self, obs: Observation):
        self.observation = obs
        frames = {}
        for name, rgb in obs.images.items():
            stream = BytesIO()
            Image.fromarray(rgb).save(stream, format="JPEG", quality=85)
            frames[name] = stream.getvalue()
        self.frames = frames

    async def tasks(self) -> list[dict]:
        async with self._control_lock:
            await self._ensure_adapter()
            method = getattr(self.adapter, "list_tasks", None)
            if method is None:
                return [{"id": 0, "name": "default", "language": "Default environment task"}]
            return await self._env(method)

    async def initialize(self, config: RunConfig, *, preserve_environment: bool = False,
                         policy_config: PolicyConfig | None = None, policy_override: Policy | None = None,
                         recording_metadata: dict | None = None):
        async with self._control_lock:
            selected = (replace(self.settings, **PolicyConfig.model_validate(policy_config).model_dump())
                        if policy_config is not None else self.settings)
            if preserve_environment and not (self.stage == "reset" and self.observation and self.observation.success
                    and self.phase in ("completed", "paused") and config.task_id == self.config.task_id
                    and config.seed == self.config.seed):
                raise StateConflict("A verified GPT reset is required before continuing without environment reset.")
            await self._stop_locked("reinitialized")
            if (self._run_policy_override or policy_override is not None or
                    (selected.model, selected.reasoning_effort) != (
                    self._policy_settings.model, self._policy_settings.reasoning_effort)):
                self._release_policy()
            self._run_policy_override = policy_override is not None
            if policy_override is not None:
                self.policy = policy_override
            self._recording_metadata = dict(recording_metadata or {})
            self._policy_settings = selected
            self.phase = "initializing"
            self.error = None
            self.config = config
            self._stopping = False
            self._permit.clear()
            self.history.clear()
            self.latest_decision = self.latest_receipt = None
            self.stats = _stats()
            try:
                await self._ensure_adapter()
                obs = await self._env(self.adapter.begin_task) if preserve_environment else await self._env(self.adapter.reset, seed=config.seed, task_id=config.task_id)
                self.context = await self._env(self.adapter.describe)
                self.context["action_spec"] = self.adapter.action_spec.to_dict()
                if config.reset_mode == "agent" and not all(callable(getattr(self.adapter, name, None)) for name in ("begin_reset", "begin_task")):
                    raise ValueError("This adapter does not implement GPT reset. Choose an adapter with begin_reset/begin_task support.")
                if config.reset_mode == "agent" and self.context.get("agent_reset", {}).get("supported") is False:
                    raise ValueError("This task has no independent GPT reset verifier. Select a supported reversible task (LIBERO-90 task 7 or LIBERO goal task 0).")
                self.recorder = Recorder(self.settings.data_dir, {
                    **config.model_dump(), "backend": self.settings.backend,
                    "suite": self.settings.suite, "policy": self.settings.policy,
                    **self.policy_config,
                    "max_output_tokens": self.settings.max_output_tokens,
                    "source_sha256": self._source_hash(),
                    "learning": self._recording_metadata,
                }, self.context)
                self._begin_episode(obs, 1, start_kind="agent_reset_continuation" if preserve_environment else "environment_reset")
                self.phase = "ready"
                self._event("ready", "Environment initialized. Inspect the cameras, then start or run one decision.")
            except Exception as exc:
                self.phase, self.error = "error", self._safe_error(exc)
                self._event("error", self.error)
                raise

    def _begin_episode(self, obs: Observation, index: int, *, start_kind: str | None = None):
        self.stage = "task"
        self._capture(obs)
        self.history.clear()
        self.latest_receipt = self.latest_decision = None
        self.episode = {"id": f"episode-{index:04d}", "index": index, "steps": 0,
                        "decisions": 0, "outcome": None, "task": obs.task,
                        "stage_steps": 0, "stage_decisions": 0,
                        "reset_steps": 0, "reset_decisions": 0, "reset_outcome": None,
                        "costs": {"task": _stats(), "reset": _stats()},
                        "started_at": time.time()}
        episode_seed = self.config.seed if self.config.reset_mode == "agent" else self.config.seed + index - 1
        self.recorder.begin_episode(index, obs, episode_seed,
                                    start_kind=start_kind or ("agent_reset_continuation" if index > 1 and self.config.reset_mode == "agent" else "environment_reset"))
        if self.policy is not None and hasattr(self.policy, "on_episode_start"):
            self.policy.on_episode_start(self.episode["id"])
        self._event("episode_start", f"Episode {index} started.")

    def _source_hash(self):
        root = Path(__file__).parent
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _cycle_done(self):
        return bool(self.episode.get("outcome") and (self.config.reset_mode == "environment" or self.episode.get("reset_outcome")))

    def _batch_done(self):
        return self._cycle_done() and (self.episode["index"] >= self.config.episodes or
            (self.config.reset_mode == "agent" and self.episode.get("reset_outcome") != "success"))

    def _budget_limits(self):
        return (self.config.max_reset_steps, self.config.max_reset_decisions) if self.stage == "reset" else (self.config.max_steps, self.config.max_decisions)

    def _usage_callback(self):
        recorder, stats, episode, stage = self.recorder, self.stats, self.episode, self.stage
        def record(event):
            event = {**event, "run_id": recorder.run_id, "episode_id": episode["id"], "stage": stage}
            recorder.write_usage(event)
            for counts in (stats, episode["costs"][stage]):
                counts["api_requests"] += 1
                for key, value in (event.get("tokens") or {}).items():
                    if key in counts and isinstance(value, (int, float)):
                        counts[key] += value
                if event.get("estimated_cost_usd") is not None:
                    counts["estimated_cost_usd"] += event["estimated_cost_usd"]
                counts["unknown_usage_requests"] += int(event.get("cost_unknown", True))
            if self.recorder is recorder:
                self._write_summary()
        return record

    def _write_summary(self):
        if self.recorder:
            self.recorder.write_summary({"run_id": self.recorder.run_id, "phase": self.phase,
                "stage": self.stage, "stats": dict(self.stats), "episode": dict(self.episode),
                "source_sha256": self._source_hash(), "updated_at": time.time()})

    async def start(self, *, single: bool = False):
        async with self._control_lock:
            if (self.phase not in ("ready", "paused") or not self.observation
                    or self._batch_done()):
                raise StateConflict("Initialize a new episode before starting; only ready/paused episodes can run.")
            if self.policy is None:
                self.policy = self._policy_factory() if self._policy_factory else make_policy(self._policy_settings)
            self._single, self._stopping = single, False
            self._epoch += 1
            self.phase = "ready"
            self._permit.set()
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._run(), name="rilo-control-loop")
            self._event("control", "Single decision requested." if single else "Execution resumed.")

    async def pause(self):
        async with self._control_lock:
            if self.phase == "paused":
                return
            if self.phase not in ("ready", "thinking", "executing", "pausing", "initializing"):
                raise StateConflict("Only an initialized or active episode can be paused.")
            # Clear permission synchronously, before any await. Once the current
            # adapter tick returns, no next tick or stale model action may start.
            self._permit.clear()
            self._epoch += 1
            self.phase = "pausing"
            self._interrupt_inference()
            await self._env(self.adapter.pause)
            if self.phase == "pausing":
                self.phase = "completed" if self._batch_done() else "paused"
            self._event("control", "Paused. Pending model actions are discarded; resume observes the current state.")

    async def stop(self):
        async with self._control_lock:
            await self._stop_locked("stopped")
            self.phase = "stopped"
            self._event("control", "Execution stopped. Initialize/reset to begin a new run.")

    async def _stop_locked(self, outcome: str):
        self._stopping = True
        self._epoch += 1
        self._interrupt_inference()
        self._permit.set()  # Wake a paused loop so it can exit.
        # Pause the environment before waiting for any model/network cleanup.
        if self.adapter is not None:
            await self._env(self.adapter.pause)
        if self._task is not None:
            await self._task
            self._task = None
        self._permit.clear()
        if self.recorder:
            if self.episode and not self._cycle_done():
                self._finish_stage(outcome)
            self._write_summary()
            self.recorder.close()

    def _interrupt_inference(self):
        if self._decision_interrupt is not None:
            self._decision_interrupt.set()
        if self._inference is not None:
            self._inference.cancel()

    def _retire(self, task: asyncio.Task):
        self._retired_requests.add(task)
        def consume(finished):
            self._retired_requests.discard(finished)
            if not finished.cancelled():
                finished.exception()
        task.add_done_callback(consume)

    async def _request_decision(self, before: Observation) -> Decision | None:
        """Race inference against control interrupts without waiting for SDK cleanup.

        Cancellation is best effort. Even a provider that swallows cancellation
        cannot delay Pause/Stop or dispatch a late result into the action loop.
        """
        if hasattr(self.policy, "usage_callback"):
            self.policy.usage_callback = self._usage_callback()
        request = asyncio.create_task(self.policy.decide(
            before, self.adapter.action_spec, self.context, list(self.history)))
        signal = asyncio.Event()
        self._inference, self._decision_interrupt = request, signal
        interrupted = asyncio.create_task(signal.wait())
        try:
            done, _ = await asyncio.wait((request, interrupted), timeout=self.settings.api_timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
            if signal.is_set():
                request.cancel()
                self._retire(request)
                return None
            if request not in done:
                request.cancel()
                self._retire(request)
                raise TimeoutError("Policy decision timed out. No action was executed.")
            return request.result()
        finally:
            interrupted.cancel()
            self._inference = self._decision_interrupt = None

    async def _pause_after_one(self):
        self._permit.clear()
        self._epoch += 1
        await self._env(self.adapter.pause)
        if not self._stopping:
            complete = self._batch_done()
            self.phase = "completed" if complete else "paused"
            self._event("control", "Single decision complete. Paused before the next request.")

    async def _advance_stage(self):
        """Switch objectives without physics reset in agent mode, including final reset."""
        if not self.episode.get("outcome"):
            return True
        if self.config.reset_mode == "agent" and not self.episode.get("reset_outcome"):
            if self.stage != "reset":
                self.phase = "initializing"
                obs = await self._env(self.adapter.begin_reset)
                self.stage = "reset"
                if self.policy is not None and hasattr(self.policy, "on_stage_start"):
                    self.policy.on_stage_start("reset")
                self._capture(obs)
                self.context = await self._env(self.adapter.describe)
                self.context["action_spec"] = self.adapter.action_spec.to_dict()
                self.episode.update(stage_steps=0, stage_decisions=0)
                self.stats["reset_attempts"] += 1
                self.recorder.write({"kind": "stage_start", "stage": "reset", "observation": self.recorder.save_observation(obs)})
                self._event("reset", "GPT reset started. Physics state is preserved; no environment reset.")
            return True
        if self._batch_done():
            self.phase = "completed"
            self._permit.clear()
            self._write_summary()
            return False
        self.phase = "initializing"
        index = self.episode["index"] + 1
        if self.config.reset_mode == "agent":
            obs = await self._env(self.adapter.begin_task)
        else:
            self.stats["hard_resets"] += 1
            obs = await self._env(self.adapter.reset, seed=self.config.seed + index - 1, task_id=self.config.task_id)
        self.context = await self._env(self.adapter.describe)
        self.context["action_spec"] = self.adapter.action_spec.to_dict()
        self._begin_episode(obs, index)
        return True

    def _validate_decision(self, decision: Decision, obs: Observation):
        spec = self.adapter.action_spec
        if decision.observation_id != obs.id:
            raise ValueError("Rejected stale observation_id in policy action")
        if decision.status == "continue" and not decision.actions:
            raise ValueError("A continuing decision must contain an action")
        if decision.status != "continue" and decision.actions:
            raise ValueError("Terminal decisions cannot also contain movement")
        count = 0
        for action in decision.actions:
            spec.validate(action.values)
            if action.repeat > spec.max_repeat:
                raise ValueError("Action repeat exceeds adapter limit")
            count += action.repeat
        if count > spec.max_chunk_steps:
            raise ValueError("Action chunk exceeds adapter limit")

    async def _run(self):
        try:
            while not self._stopping:
                await self._permit.wait()
                if self._stopping:
                    return
                if not await self._advance_stage():
                    return
                if not self._permit.is_set() or self._stopping:
                    continue
                max_steps, max_decisions = self._budget_limits()
                if self.observation.success:
                    self._finish_stage("success")
                    continue
                if self.episode["stage_steps"] >= max_steps:
                    self._finish_stage("step_budget")
                    continue
                if self.episode["stage_decisions"] >= max_decisions:
                    self._finish_stage("decision_budget")
                    continue
                if self.stats["estimated_cost_usd"] >= self.config.budget_usd:
                    self._permit.clear()
                    self.phase = "paused"
                    self._event("budget", "API cost budget reached. Paused without environment reset.")
                    self._write_summary()
                    continue

                # Pause on a real robot may involve deceleration, and its world
                # can change while waiting. Never reuse the pre-pause image.
                epoch = self._epoch
                current = await self._env(self.adapter.observe)
                self._capture(current)
                if epoch != self._epoch or not self._permit.is_set() or self._stopping:
                    continue
                self.recorder.write({"kind": "observation", "observation": self.recorder.save_observation(current)})
                if current.success or current.terminated:
                    self._finish_stage("success" if current.success else "failure")
                    continue
                before = current
                self.phase = "thinking"
                self.episode["stage_decisions"] += 1
                self.episode["reset_decisions" if self.stage == "reset" else "decisions"] += 1
                self.stats["total_decisions"] += 1
                decision = await self._request_decision(before)
                if epoch != self._epoch or not self._permit.is_set() or self._stopping:
                    self._event("control", "Discarded a response from an interrupted decision.")
                    continue
                if decision is None:
                    continue

                self._validate_decision(decision, before)
                self.latest_decision = decision.model_dump()
                if not hasattr(self.policy, "usage_callback"):
                    for key in ("input_tokens", "output_tokens"):
                        self.stats[key] += decision.usage.get(key, 0)
                self.recorder.write({"kind": "decision", "stage": self.stage, "decision": self.latest_decision})
                if decision.status != "continue":
                    if decision.status == "success":
                        # A language-model claim never becomes a success label.
                        outcome = "success" if self.observation.success else "unverified_success"
                    else:
                        outcome = decision.status
                    self._finish_stage(outcome)
                    if outcome in ("help", "unverified_success"):
                        self.phase = "completed"
                        self._permit.clear()
                        return
                    if self._single:
                        await self._pause_after_one()
                        if self.phase == "completed":
                            return
                    continue

                self.phase = "executing"
                executed = 0
                requested = sum(action.repeat for action in decision.actions)
                for action in decision.actions:
                    for _ in range(action.repeat):
                        if (epoch != self._epoch or not self._permit.is_set() or self._stopping
                                or self.episode["stage_steps"] >= max_steps):
                            break
                        prior = self.observation
                        started = time.time()
                        obs = await self._env(self.adapter.step, action.values)
                        self._capture(obs)
                        executed += 1
                        self.episode["stage_steps"] += 1
                        self.episode["reset_steps" if self.stage == "reset" else "steps"] += 1
                        self.stats["total_steps"] += 1
                        self.recorder.write({"kind": "transition", "stage": self.stage, "step": self.episode["stage_steps"],
                            "observation_id": prior.id, "requested_action": action.values,
                            "applied_action": obs.info.get("applied_action", action.values),
                            "execution_started_at": started, "execution_finished_at": time.time(),
                            "next_observation": self.recorder.save_observation(obs)})
                        if self.policy is not None and hasattr(self.policy, "on_transition"):
                            transition_info = await self.policy.on_transition(
                                prior, obs.info.get("applied_action", action.values), obs,
                                stage=self.stage,
                                truncated=(not obs.success and (
                                    self.episode["stage_steps"] >= max_steps or
                                    self.episode["stage_decisions"] >= max_decisions or
                                    bool(obs.info.get("time_limit")))) )
                            if transition_info:
                                self.recorder.write({"kind": "policy_transition", "stage": self.stage,
                                                     "step": self.episode["stage_steps"], **transition_info})
                        if obs.success or obs.terminated:
                            break
                        # Pace to the adapter's control rate for a watchable UI;
                        # the environment clock does not advance while reasoning.
                        delay = max(0, 1 / self.adapter.action_spec.control_hz - (time.time() - started))
                        if delay:
                            await asyncio.sleep(delay)
                    if (epoch != self._epoch or not self._permit.is_set() or self._stopping
                            or self.observation.success or self.observation.terminated
                            or self.episode["stage_steps"] >= max_steps):
                        break
                self.latest_receipt = {
                    "stage": self.stage, "observation_id": before.id, "result_observation_id": self.observation.id,
                    "requested_steps": requested, "executed_steps": executed,
                    "interrupted": executed < requested,
                    "success": self.observation.success, "state_before": before.state,
                    "state_after": self.observation.state, "actions": [x.model_dump() for x in decision.actions],
                    "note": decision.note,
                }
                self.history.append(self.latest_receipt)
                self.recorder.write({"kind": "receipt", **self.latest_receipt})
                if self.observation.success or self.observation.terminated:
                    outcome = "success" if self.observation.success else (
                        "time_limit" if self.observation.info.get("time_limit") else "failure")
                    self._finish_stage(outcome)
                elif self.episode["stage_steps"] >= max_steps:
                    self._finish_stage("step_budget")
                elif self.episode["stage_decisions"] >= max_decisions:
                    self._finish_stage("decision_budget")
                self._write_summary()
                if self._single and not self._stopping and self._permit.is_set():
                    await self._pause_after_one()
                    if self.phase == "completed":
                        return
        except Exception as exc:
            self._permit.clear()
            self._epoch += 1
            self.error = self._safe_error(exc)
            try:
                await self._env(self.adapter.pause)
            except Exception:
                pass
            if self.recorder and self.episode and not self._cycle_done():
                try:
                    self._finish_stage("error")
                except Exception:
                    pass
            self.phase = "error"
            self._event("error", self.error)
            self._write_summary()

    def _finish_episode(self, outcome: str):
        if self.episode.get("outcome"):
            return
        self.episode["outcome"] = outcome
        self.episode["finished_at"] = time.time()
        self.stats["episodes"] += 1
        self.stats["successes"] += int(outcome == "success")
        self.recorder.write({"kind": "stage_end", "stage": "task", "outcome": outcome})
        if self.config.reset_mode == "environment":
            self.recorder.finish_episode(dict(self.episode))
        self._event("episode", f"Episode {self.episode['index']} ended: {outcome}.")

    def _finish_stage(self, outcome: str):
        already_finished = self.episode.get("outcome" if self.stage == "task" else "reset_outcome")
        if not already_finished and self.policy is not None and hasattr(self.policy, "on_stage_end"):
            self.policy.on_stage_end(self.stage, outcome)
        if self.stage == "task":
            self._finish_episode(outcome)
        elif not self.episode.get("reset_outcome"):
            self.episode["reset_outcome"] = outcome
            self.episode["reset_finished_at"] = time.time()
            self.stats["reset_successes"] += int(outcome == "success")
            self.recorder.write({"kind": "stage_end", "stage": "reset", "outcome": outcome})
            self.recorder.finish_episode(dict(self.episode))
            self._event("reset", f"GPT reset ended: {outcome}." + (" Environment was not reset; operator review needed." if outcome != "success" else ""))
        self._write_summary()

    def status(self) -> dict:
        obs = self.observation
        return {"phase": self.phase, "stage": self.stage, "busy": self.phase in ("initializing", "thinking", "executing", "pausing"),
                "backend": self.settings.backend, "policy": self.settings.policy, **self.policy_config,
                "api_key_configured": bool(self.settings.api_key),
                "config": {"suite": self.settings.suite, **self.config.model_dump()},
                "episode": dict(self.episode), "observation": None if obs is None else {
                    "id": obs.id, "timestamp": obs.timestamp, "task": obs.task, "state": obs.state,
                    "success": obs.success, "cameras": list(obs.images)},
                "stats": dict(self.stats), "latest_decision": self.latest_decision,
                "latest_receipt": self.latest_receipt, "error": self.error,
                "run_id": self.recorder.run_id if self.recorder else None,
                "learning": self._recording_metadata,
                "events": list(self.events), "server_time": time.time(), "environment": self.context}

    async def close(self):
        async with self._control_lock:
            try:
                await self._stop_locked("shutdown")
            finally:
                if self.policy is not None:
                    await self.policy.close()
                if self._policy_cleanup_tasks:
                    # A provider that ignores cancellation must not indefinitely
                    # block shutdown; normal SDK requests finish cleanup here.
                    await asyncio.wait(self._policy_cleanup_tasks, timeout=1.0)
                if self.adapter is not None:
                    await self._env(self.adapter.close)
                self._executor.shutdown(wait=True)
