"""Run ownership and policy RPC for robot-initiated clients.

HTTP requests never call env.step/reset. Only the robot's acknowledged commands
and executed receipts advance a run; old inference generations cannot move it.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import secrets
import tempfile
import time
from typing import Literal
from urllib.parse import quote
import uuid

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from rilo.contracts import ActionSpec, Observation
from rilo.remote.policies import create_policy
from rilo.remote.records import RemoteRecording
from rilo.remote.wire import PROTOCOL, contract_hash, json_bytes, pack_message, unpack_observations


LEASE_SECONDS = 15
ACTIVE = {"preparing", "running", "pausing", "paused", "stopping"}


class RemoteError(ValueError):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


class RunStart(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    policy: Literal["gpt", "dp", "residual", "dummy"] = "gpt"
    model_id: str | None = Field(default=None, max_length=128)
    residual_model_id: str | None = Field(default=None, max_length=128)
    episodes: int = Field(default=1, ge=1, le=100)
    device: str = Field(default="cuda:0", max_length=30)
    max_episode_steps: int = Field(default=300, ge=1, le=10000)
    max_reset_steps: int = Field(default=300, ge=1, le=10000)
    max_task_calls: int = Field(default=50, ge=1, le=10000)
    max_reset_calls: int = Field(default=40, ge=1, le=1000)
    budget_usd: float = Field(default=15., gt=0, le=1000)
    training: bool = False


@dataclass
class ClientSession:
    project_id: str
    client_id: str
    registration_id: str
    environment: dict
    capabilities: dict
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    epoch: int = 0
    last_seen: float = field(default_factory=time.monotonic)
    last_seen_at: float = field(default_factory=time.time)
    phase: str = "idle"
    command_seq: int = 0
    ack_seq: int = 0
    commands: list = field(default_factory=list)
    run: dict | None = None
    policy: object = None
    reset_policy: object = None
    recording: RemoteRecording | None = None
    initial: Observation | None = None
    frames: dict = field(default_factory=dict)
    epoch_stages: dict = field(default_factory=dict)
    cache: OrderedDict = field(default_factory=OrderedDict)
    query_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    prepare_task: asyncio.Task | None = None
    inference_task: asyncio.Task | None = None

    @property
    def connected(self):
        return time.monotonic() - self.last_seen < LEASE_SECONDS


class RemoteService:
    def __init__(self, settings, projects, pipeline, *, policy_factory=create_policy, execution_busy=None):
        self.settings, self.projects, self.pipeline = settings, projects, pipeline
        self.policy_factory, self.execution_busy = policy_factory, execution_busy or (lambda: False)
        self.path = settings.data_dir.parent / "remote_clients.json"
        self.sessions, self.tokens, self.tasks = {}, {}, set()
        self.watchdog = None
        self._closed = False
        if self.path.exists():
            if self.path.is_symlink() or self.path.stat().st_size > 1024**2:
                raise ValueError("Invalid remote client registry")
            value = json.loads(self.path.read_text())
            if value.get("version") != 1 or not isinstance(value.get("tokens"), dict):
                raise ValueError("Invalid remote client registry")
            self.tokens = value["tokens"]

    def project(self, project_id):
        try:
            project = self.projects.get(project_id)
        except KeyError:
            raise RemoteError("Project does not exist", 404) from None
        if project["backend"] != "remote_robot":
            raise RemoteError("This endpoint requires a remote robot project")
        return project

    @property
    def active(self):
        return any(session.run and session.run["status"] in ACTIVE for session in self.sessions.values())

    def owns_run(self, run_id):
        return any(session.recording and session.recording.run_id == run_id
                   and (session.run["status"] in ACTIVE or session.query_lock.locked())
                   for session in self.sessions.values())

    def _save_tokens(self, tokens):
        """Commit hashes before changing authentication or dropping a session."""
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_symlink():
                raise RemoteError("Invalid remote client registry path")
            descriptor, temporary = tempfile.mkstemp(prefix=".remote-clients-", dir=self.path.parent)
            with os.fdopen(descriptor, "w") as handle:
                json.dump({"version": 1, "tokens": tokens}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError:
            raise RemoteError("클라이언트 토큰 설정을 저장하지 못했습니다. 기존 연결 설정은 유지됩니다.", 503) from None
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def token(self, project_id):
        project = self.project(project_id)
        if project.get("archived") or project.get("delete_pending"):
            raise RemoteError("프로젝트를 복원한 뒤 client를 연결하세요.")
        session = self.sessions.get(project_id)
        if session and session.run and session.run["status"] in ACTIVE:
            raise RemoteError("실행을 중지한 뒤 client 토큰을 재발급하세요.")
        token = secrets.token_urlsafe(48)
        tokens = {**self.tokens, project_id: hashlib.sha256(token.encode()).hexdigest()}
        self._save_tokens(tokens)
        self.tokens = tokens
        if session:
            self.sessions.pop(project_id, None)
        return {"token": token, "client_config": {"base_url": "http://127.0.0.1:45587/robot/v1",
            "project_token": token, "client_id": uuid.uuid4().hex, "journal_path": ".rilo/journal.sqlite3",
            "control_hz": 10., "history_capacity": 8, "request_timeout_seconds": 180., "adapter_options": {}}}

    def revoke(self, project_id):
        self.project(project_id)
        session = self.sessions.get(project_id)
        if session and session.run and session.run["status"] in ACTIVE:
            raise RemoteError("실행을 중지한 뒤 client 토큰을 폐기하세요.")
        tokens = {key: value for key, value in self.tokens.items() if key != project_id}
        self._save_tokens(tokens)
        self.tokens = tokens
        self.sessions.pop(project_id, None)
        return {"project_id": project_id, "revoked": True}

    def authenticate(self, token):
        if not isinstance(token, str) or not 20 <= len(token) <= 256:
            raise RemoteError("Invalid client token", 401)
        digest = hashlib.sha256(token.encode()).hexdigest()
        for project_id, stored in self.tokens.items():
            if secrets.compare_digest(digest, stored):
                self.project(project_id)
                return project_id
        raise RemoteError("Invalid or revoked client token", 401)

    def _session(self, project_id, session_id):
        session = self.sessions.get(project_id)
        if session is None or session.id != session_id:
            raise RemoteError("Client session has expired; register again")
        return session

    def models(self, project_id):
        return [{key: model.get(key) for key in ("id", "kind", "name", "base_model_id", "status")}
                for model in self.pipeline._load(project_id)["models"] if model.get("kind") in ("bc", "residual")]

    def snapshot(self, project_id):
        project = self.project(project_id)
        session = self.sessions.get(project_id)
        run = deepcopy(session.run) if session else None
        connected = bool(session and session.connected)
        busy = bool(run and run["status"] in ACTIVE)
        reason = ("프로젝트를 복원한 뒤 실행하세요." if project.get("archived") or project.get("delete_pending") else
                  "로봇 client를 먼저 연결하세요." if not connected else
                  "실행·hold·안전한 팔 reset을 지원하는 로봇 adapter가 필요합니다."
                  if not all(session.capabilities.get(key) is True for key in ("execute", "hold", "arm_reset", "safe_reset")) else
                  "현재 run을 종료한 뒤 새 run을 시작하세요." if busy else
                  "다른 수집·학습 실행이 진행 중입니다." if self.execution_busy() or any(
                      item is not session and item.run and item.run["status"] in ACTIVE for item in self.sessions.values()) else None)
        config = project.get("run_config", {})
        if session and session.recording and run:
            record = session.recording
            run.update(episode_id=record.episode.get("id"), task_steps=record.episode.get("steps", 0),
                reset_steps=record.episode.get("reset_steps", 0),
                task_calls=record.episode.get("costs", {}).get("task", {}).get("api_requests", 0),
                reset_calls=record.episode.get("costs", {}).get("reset", {}).get("api_requests", 0),
                estimated_cost_usd=record.stats["estimated_cost_usd"])
        frames = session.frames if session else {}
        return {"protocol": PROTOCOL, "token_configured": project_id in self.tokens,
            "client_url": "http://127.0.0.1:45587/robot/v1", "client": None if not session else {
                "session_id": session.id, "client_id": session.client_id, "connected": connected,
                "phase": session.phase, "environment": deepcopy(session.environment),
                "capabilities": dict(session.capabilities), "last_seen": session.last_seen_at},
            "run": run, "models": self.models(project_id), "defaults": {"policy": "gpt", "device": "cuda:0",
                "episodes": config.get("episodes", 1), "max_episode_steps": config.get("max_steps", 300),
                "max_reset_steps": config.get("max_reset_steps", 300), "max_task_calls": config.get("max_decisions", 50),
                "max_reset_calls": config.get("max_reset_decisions", 40), "budget_usd": config.get("budget_usd", 15.)},
            "capabilities": {"start": reason is None, "pause": bool(connected and busy and run["status"] == "running"),
                "resume": bool(connected and busy and run["status"] == "paused"), "stop": busy},
            "reasons": {"start": reason}, "frames": [{"camera": name,
                "url": f"/api/projects/{project_id}/remote/frame/{quote(name)}?version={quote(value[0])}"}
                for name, value in frames.items()]}

    def register(self, project_id, data):
        project = self.project(project_id)
        if project.get("archived") or project.get("delete_pending"):
            raise RemoteError("프로젝트를 복원한 뒤 client를 연결하세요.")
        client_id, registration_id = data.get("client_id"), data.get("registration_id", data.get("client_id"))
        if not all(isinstance(value, str) and 1 <= len(value) <= 128 for value in (client_id, registration_id)):
            raise RemoteError("client_id and registration_id are required", 422)
        environment, capabilities = data.get("environment"), data.get("capabilities")
        if not isinstance(environment, dict) or not isinstance(capabilities, dict):
            raise RemoteError("Environment and capabilities are required", 422)
        if any(not isinstance(value, bool) for value in capabilities.values()):
            raise RemoteError("Capabilities must be explicit booleans", 422)
        if capabilities.get("execute"):
            self.action_spec(environment)
        old = self.sessions.get(project_id)
        if old and old.client_id == client_id and old.registration_id == registration_id:
            if environment != old.environment or capabilities != old.capabilities:
                raise RemoteError("The registered environment changed; start a new client session")
            old.last_seen, old.last_seen_at = time.monotonic(), time.time()
            return {"session_id": old.id, "epoch": old.epoch, "lease_seconds": LEASE_SECONDS}
        if old and old.run and old.run["status"] in ACTIVE:
            raise RemoteError("Stop the previous run before replacing its robot client")
        if old and old.connected:
            raise RemoteError("A different robot client is already registered")
        session = ClientSession(project_id, client_id, registration_id, deepcopy(environment), dict(capabilities))
        self.sessions[project_id] = session
        if self.watchdog is None:
            self.watchdog = asyncio.create_task(self._watchdog())
        return {"session_id": session.id, "epoch": session.epoch, "lease_seconds": LEASE_SECONDS}

    @staticmethod
    def action_spec(environment):
        try:
            value = environment["action_spec"]
            result = ActionSpec(names=tuple(value["names"]), low=tuple(value["low"]), high=tuple(value["high"]),
                description=value["description"], control_hz=float(value["control_hz"]),
                max_repeat=value.get("max_repeat", 10), max_chunk_steps=value.get("max_chunk_steps", 30))
            if not 1 <= len(result.names) <= 32 or not math.isfinite(result.control_hz) or not 0 < result.control_hz <= 1000:
                raise ValueError("Invalid action contract")
            return result
        except (KeyError, TypeError, ValueError):
            raise RemoteError("Invalid environment action specification", 422) from None

    def _command(self, session, kind, run=None):
        session.command_seq += 1
        session.commands.append({"id": session.command_seq, "type": kind, "epoch": session.epoch, "run": run})
        session.commands[:] = session.commands[-128:]

    def poll(self, project_id, data):
        session = self._session(project_id, data.get("session_id"))
        status = data.get("status") or {}
        session.last_seen, session.last_seen_at = time.monotonic(), time.time()
        session.phase = str(status.get("phase", "idle"))[:80]
        if session.run and session.run["status"] == "running" and session.phase in ("lease_expired", "error", "held"):
            self.control(project_id, "pause")
        after = data.get("after_command_id", 0)
        if type(after) is not int or not 0 <= after <= session.command_seq:
            raise RemoteError("Invalid command cursor", 422)
        return {"session_id": session.id, "epoch": session.epoch,
                "commands": [deepcopy(command) for command in session.commands if command["id"] > after],
                "lease_seconds": LEASE_SECONDS}

    def ack(self, project_id, data):
        session = self._session(project_id, data.get("session_id"))
        command = next((item for item in session.commands if item["id"] == data.get("command_id")), None)
        if command is None or data.get("epoch") != command["epoch"]:
            raise RemoteError("Unknown or mismatched command acknowledgment")
        if command["id"] <= session.ack_seq:
            return {"ok": True, "epoch": session.epoch}
        if data.get("status") not in ("applied", "rejected"):
            raise RemoteError("Invalid acknowledgment", 422)
        session.ack_seq = command["id"]
        if command["epoch"] != session.epoch:
            return {"ok": True, "epoch": session.epoch, "stale": True}
        run = session.run
        if data["status"] == "rejected":
            if run:
                run.update(status="error", error=str(data.get("detail", "Robot client rejected command"))[:1000])
                self._finish(session, "error")
        elif run:
            if command["type"] in ("start", "resume"):
                run["status"] = "running"
            elif command["type"] == "pause":
                run["status"] = "paused"
            elif command["type"] == "stop":
                self._finish(session, "completed")
        return {"ok": True, "epoch": session.epoch}

    def start(self, project_id, value):
        config = RunStart.model_validate(value)
        snapshot = self.snapshot(project_id)
        if not snapshot["capabilities"]["start"]:
            raise RemoteError(snapshot["reasons"]["start"])
        session = self.sessions[project_id]
        if config.policy == "dummy" and not session.capabilities.get("simulation"):
            raise RemoteError("Dummy actions are only available in a declared simulated environment")
        if config.policy == "residual" and not session.capabilities.get("residual_step"):
            raise RemoteError("Robot executor does not support per-step residual correction")
        if config.training and (config.policy != "residual" or not session.capabilities.get("residual_training")):
            raise RemoteError("This client does not provide the bootstrap receipts required for residual training")
        session.epoch += 1
        session.run = {"id": None, "status": "preparing", "phase": "awaiting_episode", "epoch": session.epoch,
            "policy": config.policy, "config": config.model_dump(), "target_episodes": config.episodes,
            "completed_episodes": 0, "episode_id": None, "error": None}
        session.prepare_task = asyncio.create_task(self._prepare(session, config))
        self.tasks.add(session.prepare_task)
        session.prepare_task.add_done_callback(self.tasks.discard)
        return self.snapshot(project_id)

    async def _prepare(self, session, config):
        generation = session.epoch
        policies = []
        try:
            await self._close_policies(session)
            project = self.project(session.project_id)
            project = {**project, "grasp_baseline": deepcopy(self.projects.grasp_baseline(project))}
            chosen = project.get("policy_config") or {}
            settings = replace(self.settings, model=chosen.get("model", self.settings.model),
                reasoning_effort=chosen.get("reasoning_effort", self.settings.reasoning_effort))
            spec = self.action_spec(session.environment)
            checkpoints = {}
            for argument, model_id, kind in (("checkpoint", config.model_id, "bc"), ("residual_checkpoint", config.residual_model_id, "residual")):
                if model_id:
                    if not any(model["id"] == model_id and model["kind"] == kind for model in self.models(session.project_id)):
                        raise RemoteError("Choose a matching model from this project's registry")
                    checkpoints[argument] = self.pipeline.checkpoint(session.project_id, model_id)
            environment = {**session.environment, "simulation": session.capabilities.get("simulation", False)}
            for kind, training in ((config.policy, config.training), ("dummy" if config.policy == "dummy" else "gpt", False)):
                task = asyncio.create_task(self.policy_factory(kind, settings=settings, action_spec=spec,
                    environment=environment, device=config.device, training=training,
                    model_id=config.model_id, residual_model_id=config.residual_model_id, **checkpoints))
                try:
                    policy = await asyncio.shield(task)
                except asyncio.CancelledError:
                    policy = await task
                    await policy.close()
                    raise
                policies.append(policy)
                if generation != session.epoch or session.run["status"] != "preparing":
                    raise asyncio.CancelledError()
            session.policy, session.reset_policy = policies
            session.recording = RemoteRecording(settings, project, config.model_dump(), environment,
                policy_metadata=session.policy.metadata, link_run=self.projects.link_run)
            for policy in policies:
                policy.usage_callback = session.recording.usage
            session.run["id"] = session.recording.run_id
            contract = {**session.policy.metadata, "hash": contract_hash(session.policy.metadata)}
            reset_contract = {**session.reset_policy.metadata, "hash": contract_hash(session.reset_policy.metadata)}
            session.run.update(contract=contract, reset_contract=reset_contract)
            self._command(session, "start", {"run_id": session.run["id"], "policy": config.policy,
                "episodes": config.episodes, "task": project.get("task_name", ""), "limits": config.model_dump(),
                "contract": contract, "reset_contract": reset_contract,
                "reset_policy": "dummy" if config.policy == "dummy" else "gpt"})
        except asyncio.CancelledError:
            for policy in policies:
                await policy.close()
        except Exception as exc:
            for policy in policies:
                await policy.close()
            if generation == session.epoch:
                session.run.update(status="error", error=str(exc)[:1000])

    def control(self, project_id, command):
        self.project(project_id)
        session = self.sessions.get(project_id)
        if not session or not session.run or session.run["status"] not in ACTIVE:
            raise RemoteError("There is no active remote run")
        run = session.run
        if command == "pause" and run["status"] not in ("running", "pausing"):
            raise RemoteError("Only a running run can be paused")
        if command == "resume" and (run["status"] != "paused" or not session.connected):
            raise RemoteError("Connect the held client before resuming")
        if command not in ("pause", "resume", "stop"):
            raise RemoteError("Unknown run command", 422)
        session.epoch += 1
        run["epoch"] = session.epoch
        session.epoch_stages[session.epoch] = run["phase"]
        run["status"] = {"pause": "pausing", "resume": "preparing", "stop": "stopping"}[command]
        if session.inference_task:
            session.inference_task.cancel()
        if command == "stop" and session.prepare_task and not session.prepare_task.done():
            session.prepare_task.cancel()
        self._command(session, command)
        if command == "stop" and (not session.connected or not session.recording):
            self._finish(session, "completed")
        return self.snapshot(project_id)

    async def _close_policies(self, session):
        policies = (session.policy, session.reset_policy)
        session.policy = session.reset_policy = None
        for policy in policies:
            if policy is not None:
                await policy.close()

    def _finish(self, session, status):
        if session.recording:
            session.recording.close(status)
        if session.run:
            session.run["status"] = status
        policies = (session.policy, session.reset_policy)
        session.policy = session.reset_policy = None
        async def close_owned():
            for policy in policies:
                if policy is not None:
                    await policy.close()
        task = asyncio.create_task(close_owned())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    @staticmethod
    def _capture(session, observation):
        for camera, pixels in observation.images.items():
            stream = BytesIO()
            Image.fromarray(pixels).save(stream, "JPEG", quality=85)
            session.frames[camera] = (observation.id, stream.getvalue())

    def _current(self, session, epoch):
        return bool(session.connected and session.epoch == epoch and session.run and session.run["status"] == "running")

    async def packet(self, project_id, operation, meta, arrays, digest, contract=None):
        if self._closed:
            raise RemoteError("Policy service is shutting down", 503)
        # A disconnected HTTP caller must not interrupt the interval between
        # persisting physical evidence and committing its idempotent reply.
        # Policy cancellation still belongs to Pause/Stop and the session epoch.
        task = asyncio.create_task(self._packet(project_id, operation, meta, arrays, digest, contract))
        self.tasks.add(task)
        def settled(done):
            self.tasks.discard(done)
            if not done.cancelled():
                done.exception()  # Consume failures even if the HTTP caller vanished.
        task.add_done_callback(settled)
        return await asyncio.shield(task)

    async def _packet(self, project_id, operation, meta, arrays, digest, contract=None):
        session = self._session(project_id, meta.get("session_id"))
        request_id = meta.get("request_id")
        if not isinstance(request_id, str) or not request_id.isascii() or not request_id.isalnum() or not 1 <= len(request_id) <= 128:
            raise RemoteError("An alphanumeric request_id is required", 422)
        cache_key = operation, request_id
        async with session.query_lock:
            if cache_key in session.cache:
                cached_digest, result = session.cache[cache_key]
                if digest != cached_digest:
                    raise RemoteError("A request ID was reused for different input")
                # Never hand out a cached motion after a pause/phase change.
                if operation == "infer" and not self._current(session, meta.get("epoch")):
                    raise RemoteError("Stale inference generation")
                return result
            observations = [Observation(**value) for value in unpack_observations(meta.get("observations"), arrays)]
            if any(not math.isfinite(item.timestamp) or not isinstance(item.success, bool)
                   or not isinstance(item.terminated, bool) for item in observations):
                raise RemoteError("Invalid observation metadata", 422)
            epoch = meta.get("epoch")
            if type(epoch) is not int:
                raise RemoteError("An integer epoch is required", 422)
            if operation == "observe":
                if epoch != session.epoch:
                    raise RemoteError("Stale observation generation")
                self._capture(session, observations[-1])
                result = {"ok": True, "epoch": session.epoch}
            else:
                if not session.run or not session.recording:
                    raise RemoteError("No prepared remote run")
                executed_end = (operation == "episode/end" and session.epoch_stages.get(epoch) == "arm_reset"
                                and session.run["status"] in ("pausing", "paused", "stopping"))
                if operation != "transition" and not executed_end and not self._current(session, epoch):
                    raise RemoteError("Run is paused, stopped, disconnected, or from an older generation")
                result = await self._run_packet(session, operation, meta, observations, contract)
            session.cache[cache_key] = digest, result
            while len(session.cache) > 128:
                session.cache.popitem(last=False)
            return result

    async def _run_packet(self, session, operation, meta, observations, contract):
        run, record, epoch = session.run, session.recording, meta["epoch"]
        if operation == "episode/begin":
            if run["phase"] != "awaiting_episode" or meta.get("index") != run["completed_episodes"] + 1:
                raise RemoteError("Unexpected episode begin")
            record.begin(meta["index"], observations[-1], meta.get("seed", 0))
            session.initial = observations[-1]
            await session.policy.begin_episode(record.episode["id"])
            await session.reset_policy.begin_episode(record.episode["id"])
            run.update(phase="task", episode_id=record.episode["id"])
            session.epoch_stages = {epoch: "task"}
            result = {"episode_id": record.episode["id"], "epoch": epoch}
        elif operation == "infer":
            stage = run["phase"]
            if stage not in ("task", "reset"):
                raise RemoteError("Inference is not available in this phase")
            selected = session.policy if stage == "task" else session.reset_policy
            selected_contract = run["contract" if stage == "task" else "reset_contract"]
            if contract != selected_contract["hash"]:
                raise RemoteError("Policy contract does not match this run")
            config = run["config"]
            steps = record.episode["steps" if stage == "task" else "reset_steps"]
            max_steps = config["max_episode_steps" if stage == "task" else "max_reset_steps"]
            calls = record.episode["costs"][stage]["api_requests"]
            max_calls = config["max_task_calls" if stage == "task" else "max_reset_calls"]
            if steps >= max_steps or calls >= max_calls or record.stats["estimated_cost_usd"] >= config["budget_usd"]:
                result = pack_message({"epoch": epoch, "request_id": meta["request_id"], "episode_id": record.episode["id"],
                    "decision": {"status": "failure", "note": "Run execution limit reached"}, "actions_encoding": "environment"})
            else:
                def is_current():
                    return self._current(session, epoch)
                def can_request():
                    return (record.episode["costs"][stage]["api_requests"] < max_calls
                            and record.stats["estimated_cost_usd"] < config["budget_usd"])
                context = {**deepcopy(session.environment), "stage": stage, "phase": stage,
                    "task": self.project(session.project_id).get("task_name", ""),
                    "history": list(record.history), "is_current": is_current, "can_request": can_request,
                    "operation": meta.get("operation", "chunk"), "base_action": meta.get("base_action"),
                    "chunk_phase": meta.get("chunk_phase"), "episode_id": record.episode["id"],
                    "reset_goal": {"instruction": "Restore the task's initial object state, then retreat to free space where the local robot can verify arm reset is safe. Do not invoke arm reset yourself.",
                                   "initial_state": session.initial.state if session.initial else {}}}
                policy_observations = observations
                if stage == "reset":
                    # References describe the target object state, not the
                    # current scene. Keep these policy-only copies out of live
                    # previews and executed transition recordings.
                    reference_images = {f"reset_reference_{name}": pixels.copy()
                        for name, pixels in (session.initial.images if session.initial else {}).items()
                        if not name.startswith("reset_reference_")}
                    context["reset_goal"].update(reference_images=list(reference_images),
                        reference_image_notes="Images named reset_reference_* are saved initial scenes, not live observations. Compare the current scene with these references to restore displaced objects before the local safe arm reset.")
                    policy_observations = [replace(item, success=False, terminated=False)
                                           for item in observations]
                    policy_observations[-1] = replace(policy_observations[-1],
                        images={**observations[-1].images, **reference_images})
                session.inference_task = asyncio.create_task(selected.infer(policy_observations, context))
                try:
                    prediction = await session.inference_task
                except asyncio.CancelledError:
                    raise RemoteError("Inference was interrupted; no action may be executed") from None
                finally:
                    session.inference_task = None
                if not self._current(session, epoch):
                    raise RemoteError("Discarded a stale inference result")
                chunk_id = uuid.uuid4().hex
                encoding = ("environment" if meta.get("operation") == "correct" else selected.metadata["actions"])
                values = prediction.get("actions")
                response = {"epoch": epoch, "request_id": meta["request_id"], "episode_id": record.episode["id"],
                    "chunk_id": chunk_id, "actions_encoding": encoding, "decision": prediction.get("decision"),
                    "outcome": prediction.get("outcome")}
                record.decision({**response, "operation": meta.get("operation", "chunk"), "observation_id": observations[-1].id})
                result = pack_message(response, {"actions": values} if values is not None else {})
        elif operation == "transition":
            if (meta.get("episode_id") != record.episode.get("id") or record.episode.get("finished_at")
                    or session.epoch_stages.get(epoch) != meta.get("stage") or len(observations) != 2):
                raise RemoteError("Executed receipt does not belong to this episode and phase")
            action = self.action_spec(session.environment).validate(meta.get("action", []))
            if type(meta.get("seq")) is not int:
                raise RemoteError("A transition sequence is required", 422)
            record.transition(observations[0], action, observations[1], meta)
            learning = None
            if meta["stage"] == "task" and run["config"]["training"]:
                if meta.get("annotations", {}).get("learning_pending"):
                    learning = {"training": False, "reason": "Actual next base action was unavailable; execution evidence retained"}
                else:
                    try:
                        learning = await session.policy.on_transition(observations[0], action, observations[1], {
                            **meta.get("annotations", {}), "episode_id": record.episode["id"], "transition_id": meta["request_id"],
                            "stage": "task", "reward": meta.get("reward", 0.), "terminated": meta.get("terminated", False),
                            "truncated": meta.get("truncated", False)})
                    except Exception as exc:
                        # The action already happened. Acknowledge its evidence
                        # even when learning fails, then hold further execution.
                        learning = {"training": False, "error": str(exc)[:500]}
                        run["error"] = "Executed receipt saved, but residual learning failed"
                        if run["status"] == "running":
                            self.control(session.project_id, "pause")
            result = {"ok": True, "epoch": session.epoch, "seq": meta["seq"], "learning": learning}
        elif operation == "phase":
            target = meta.get("target")
            if target == "reset" and run["phase"] == "task":
                if meta.get("task_outcome") not in ("success", "failure") or meta.get("outcome_source") not in ("env", "gpt", "operator"):
                    raise RemoteError("A task outcome and its source are required", 422)
                record.end_task(meta["task_outcome"], meta["outcome_source"])
                await session.reset_policy.begin_phase("reset")
            elif target == "arm_reset" and run["phase"] == "reset":
                if meta.get("reset_verified") is not True or session.capabilities.get("safe_reset") is not True:
                    raise RemoteError("Local free-space/reset verification is required before arm reset")
            else:
                raise RemoteError("Invalid phase transition")
            session.epoch += 1
            run.update(phase=target, epoch=session.epoch)
            session.epoch_stages[session.epoch] = target
            result = {"epoch": session.epoch, "phase": target}
        elif operation == "episode/end":
            if run["phase"] != "arm_reset" or meta.get("reset_outcome") not in ("success", "failure"):
                raise RemoteError("Complete task restoration and the arm-reset phase first")
            if meta.get("reset_outcome") == "success" and meta.get("arm_reset_performed") is not True:
                raise RemoteError("An actual local arm-reset receipt is required")
            record.finish(meta["reset_outcome"], arm_reset=meta.get("arm_reset_performed", False))
            run["completed_episodes"] += 1
            finished = run["completed_episodes"] >= run["target_episodes"] or meta["reset_outcome"] != "success"
            session.epoch += 1
            run.update(epoch=session.epoch, phase="awaiting_episode")
            if finished:
                if run["config"]["training"] and hasattr(session.policy, "save_checkpoint"):
                    await session.policy.save_checkpoint(record.recorder.path / "residual-final.pt")
                self._finish(session, "completed" if meta["reset_outcome"] == "success" else "error")
            result = {"epoch": session.epoch, "completed_episodes": run["completed_episodes"], "finished": finished}
        else:
            raise RemoteError("Unknown robot operation", 404)
        self._capture(session, observations[-1])
        record.summary(run["status"])
        return result

    async def _watchdog(self):
        while not self._closed:
            for session in list(self.sessions.values()):
                if session.run and session.run["status"] in ACTIVE and not session.connected:
                    if session.run["status"] == "stopping":
                        self._finish(session, "completed")
                    elif session.run["status"] != "paused":
                        session.epoch += 1
                        session.run.update(epoch=session.epoch, status="paused", error="Client heartbeat expired; explicit resume is required")
                        if session.inference_task:
                            session.inference_task.cancel()
                        self._command(session, "pause")
            await asyncio.sleep(1.)

    async def close(self):
        self._closed = True
        if self.watchdog:
            self.watchdog.cancel()
            await asyncio.gather(self.watchdog, return_exceptions=True)
        for session in list(self.sessions.values()):
            session.epoch += 1
            owned_tasks = []
            for task in (session.prepare_task, session.inference_task):
                if task and not task.done():
                    task.cancel()
                    owned_tasks.append(task)
            if owned_tasks:
                await asyncio.gather(*owned_tasks, return_exceptions=True)
            status = session.run.get("status") if session.run else None
            if status in ACTIVE:
                status = "interrupted"
                session.run.update(status=status, epoch=session.epoch)
            if session.recording:
                session.recording.close(status or "interrupted")
            await self._close_policies(session)
        if self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)
