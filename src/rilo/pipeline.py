"""Explicit, sequential learning jobs over the existing single robot Runner.

Nothing starts on import, application startup, page navigation, or config save.
Neural-network training runs off the HTTP event loop; robot ownership spans the
whole job, so collection and supervised training cannot execute concurrently.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field

from rilo.config import make_policy
from rilo.recording import dump_json
from rilo.runner import StateConflict


STAGES = ("collect", "bc", "bc_eval", "progress", "rl", "rl_eval")


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    collection_episodes: int = Field(default=20, ge=1, le=100)
    evaluation_episodes: int = Field(default=10, ge=1, le=100)
    bc_success_target: float = Field(default=70, gt=0, le=100)
    rl_success_target: float = Field(default=90, gt=0, le=100)
    max_bc_cycles: int = Field(default=10, ge=1, le=1000)
    max_rl_rounds: int = Field(default=20, ge=1, le=1000)
    rl_episodes_per_round: int = Field(default=5, ge=1, le=100)
    progress_gate_threshold: float = Field(default=.8, ge=0, le=1)
    gpt_judge_interval: int = Field(default=20, ge=1, le=10000)
    run_budget_usd: float = Field(default=15, gt=0, le=1000)
    device: str = "auto"
    bc: dict = Field(default_factory=dict)
    progress: dict = Field(default_factory=dict)
    residual: dict = Field(default_factory=dict)


class PipelineStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["auto", "manual"] = "manual"
    stage: Literal["collect", "bc", "bc_eval", "progress", "rl", "rl_eval"] | None = None


class PipelineStopped(RuntimeError):
    pass


def learning_configs(config):
    from rilo.learning.diffusion import DPConfig
    from rilo.learning.progress import ProgressConfig
    from rilo.learning.residual import TD3Config
    value = PipelineConfig.model_validate(config).model_dump()
    if value["device"] not in ("auto", "cpu", "cuda") and not (
            value["device"].startswith("cuda:") and value["device"][5:].isdigit()):
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
    value["bc"] = DPConfig.model_validate({**value["bc"], "device": value["device"]}).model_dump(mode="json")
    value["progress"] = ProgressConfig.model_validate({**value["progress"], "device": value["device"]}).model_dump(mode="json")
    value["residual"] = TD3Config.model_validate(value["residual"]).model_dump(mode="json")
    return value


class PipelineEngine:
    def __init__(self, settings, projects, get_runner, prepare_run, *, on_owner=None):
        self.settings, self.projects = settings, projects
        self.get_runner, self.prepare_run = get_runner, prepare_run
        self.root = settings.data_dir.parent / "learning"
        self._lock = threading.RLock()
        self._task = None
        self._owner = None
        self._state = None
        self._gate = threading.Event()
        self._gate.set()
        self._stop = threading.Event()
        self._trainer = None
        self._saved_rl_steps = 0
        self._last_persist = 0.

    @property
    def active(self):
        return self._task is not None and not self._task.done()

    def runtime(self):
        if self._state is None:
            return None
        return {"project_id": self._owner, **{k: self._state["status"].get(k)
                for k in ("phase", "stage", "mode", "cycle")}}

    def _path(self, project_id):
        self.projects.get(project_id)
        if Path(project_id).name != project_id or project_id in ("", ".", ".."):
            raise ValueError("Invalid project identifier")
        path = self.root / project_id
        if path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Invalid learning artifact directory")
        return path

    def _load(self, project_id):
        if self._owner == project_id and self._state is not None:
            return deepcopy(self._state)
        path = self._path(project_id) / "registry.json"
        if path.exists():
            value = json.loads(path.read_text())
            if value["status"]["phase"] in ("running", "paused"):
                value["status"].update(phase="interrupted", error="서버가 종료되어 중단된 실행입니다. 단계별 실행으로 다시 시작하세요.")
            return value
        goals = self.projects.get(project_id).get("pipeline", {})
        config = PipelineConfig(collection_episodes=min(goals.get("demo_target", 20), 100),
            bc_success_target=goals.get("bc_success_target", 70), rl_success_target=goals.get("rl_success_target", 90))
        return {"config": config.model_dump(), "status": {"phase": "idle", "stage": None,
            "mode": None, "cycle": 0, "metrics": {}, "stage_metrics": {}, "events": [], "error": None},
            "models": [], "evaluations": []}

    def _save(self):
        folder = self._path(self._owner)
        folder.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".registry-", suffix=".tmp", dir=folder)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(dump_json(self._state))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, folder / "registry.json")
        finally:
            if os.path.exists(temp): os.unlink(temp)

    def _event(self, message, kind="pipeline"):
        with self._lock:
            self._state["status"]["events"].append({"time": time.time(), "kind": kind, "message": message})
            self._state["status"]["events"] = self._state["status"]["events"][-100:]
            self._save()

    def _metrics(self, values):
        with self._lock:
            status = self._state["status"]
            status["metrics"].update(values)
            status["stage_metrics"].setdefault(status["stage"], {}).update(values)
            if time.monotonic() - self._last_persist > 1:
                self._save()
                self._last_persist = time.monotonic()

    def _control(self):
        while not self._stop.is_set():
            if self._gate.wait(.1):
                return
        raise PipelineStopped("Pipeline stopped")

    async def _async_control(self):
        while not self._gate.is_set():
            if self._stop.is_set(): raise PipelineStopped("Pipeline stopped")
            await asyncio.sleep(.05)
        if self._stop.is_set(): raise PipelineStopped("Pipeline stopped")

    @staticmethod
    def _latest(state, kind, base_id=None):
        return next((m for m in reversed(state["models"]) if m["kind"] == kind
                     and (base_id is None or m.get("base_model_id") == base_id)), None)

    def view(self, project_id):
        with self._lock:
            state = self._load(project_id)
        installed = importlib.util.find_spec("torch") is not None
        dependencies = {"torch": installed, "cuda": False}
        if installed:
            import torch
            dependencies["cuda"] = torch.cuda.is_available()
            state["config"] = learning_configs(state["config"])
        project = self.projects.get(project_id)
        details = self.projects.details(project)
        successes = sum(ep.get("outcome") == "success" and ep.get("steps", 0) > 0 for ep in details["episodes"])
        bc = self._latest(state, "bc")
        progress = self._latest(state, "progress", bc["id"]) if bc else None
        residual = self._latest(state, "residual", bc["id"]) if bc else None
        evaluations = [e for e in state["evaluations"] if bc and e.get("model_id") == bc["id"] and e["kind"] == "bc"]
        passed = bool(evaluations and evaluations[-1]["success_rate"] >= state["config"]["bc_success_target"])
        teacher_ready = self.settings.policy != "openai" or bool(self.settings.api_key)
        physical = teacher_ready and project["backend"] == self.settings.backend
        checks = {"collect": (physical, "서버 환경과 GPT API key를 확인하세요."),
            "bc": (installed and successes >= 2, "학습/검증 분리를 위해 성공한 task episode가 2개 이상 필요합니다."),
            "bc_eval": (physical and installed and bool(bc), "먼저 BC 모델을 학습해야 합니다."),
            "progress": (installed and bool(bc) and passed and successes >= 2, "현재 BC 모델이 목표 평가 성공률에 도달해야 합니다."),
            "rl": (physical and installed and bool(progress) and passed, "목표에 도달한 BC와 해당 BC의 progress 모델이 필요합니다."),
            "rl_eval": (physical and installed and bool(residual), "먼저 residual RL 모델을 학습해야 합니다."),
            "auto": (physical and installed, "학습 의존성·실행 환경·GPT API key를 확인하세요.")}
        state["readiness"] = {k: {"ready": bool(ok), "reason": "" if ok else reason} for k, (ok, reason) in checks.items()}
        state["dependencies"] = dependencies
        state["capabilities"] = {"bc": installed, "progress": installed, "rl": installed, "evaluation": installed}
        for model in state["models"]:
            model["checkpoint_url"] = f"/api/projects/{project_id}/models/{model['id']}/checkpoint"
        return state

    def configure(self, project_id, value):
        if self.active and self._owner == project_id:
            raise StateConflict("파이프라인을 중지한 뒤 학습 설정을 변경하세요.")
        with self._lock:
            state = self._load(project_id)
            # Nested dicts are complete model configs, not silently partial overrides.
            state["config"] = learning_configs({**state["config"], **value})
            previous_owner, previous_state = self._owner, self._state
            self._owner, self._state = project_id, state
            try: self._save()
            finally:
                if self.active:
                    self._owner, self._state = previous_owner, previous_state
        return self.view(project_id)

    async def start(self, project_id, request: PipelineStart):
        if self.active:
            raise StateConflict("다른 학습/수집 작업이 실행 중입니다. 완료하거나 중지한 뒤 시작하세요.")
        stage = "auto" if request.mode == "auto" else request.stage
        if stage is None:
            raise ValueError("단계별 실행에는 stage가 필요합니다.")
        state = self.view(project_id)
        if not state["readiness"][stage]["ready"]:
            raise StateConflict(state["readiness"][stage]["reason"])
        runner = self.get_runner()
        if runner.phase in ("initializing", "thinking", "executing", "pausing", "paused", "ready"):
            raise StateConflict("기존 수집을 중지한 뒤 파이프라인을 시작하세요.")
        self._owner = project_id
        self._state = {k: state[k] for k in ("config", "status", "models", "evaluations")}
        self._state["status"] = {"phase": "running", "stage": request.stage or "collect", "mode": request.mode,
            "cycle": 0, "metrics": {}, "stage_metrics": {}, "events": [], "error": None,
            "job_id": uuid.uuid4().hex, "started_at": time.time()}
        self._trainer = None
        self._saved_rl_steps = 0
        self._stop.clear(); self._gate.set()
        self._save()
        self._task = asyncio.create_task(self._execute(request), name="rilo-learning-pipeline")
        return self.view(project_id)

    async def pause(self, project_id):
        self._require_active(project_id)
        self._gate.clear()
        self._state["status"]["phase"] = "paused"
        runner = self.get_runner()
        if runner.phase in ("ready", "thinking", "executing", "initializing", "pausing"):
            await runner.pause()
        self._event("파이프라인 일시정지")

    async def resume(self, project_id):
        self._require_active(project_id)
        runner = self.get_runner()
        if runner.phase == "paused" and self._state["status"]["stage"] in ("collect", "bc_eval", "rl", "rl_eval"):
            if runner.stats["estimated_cost_usd"] >= runner.config.budget_usd:
                raise StateConflict("실행 API 예산에 도달했습니다. 중지 후 예산을 조정하세요.")
            await runner.start()
        self._state["status"]["phase"] = "running"
        self._gate.set()
        self._event("파이프라인 재개")

    async def stop(self, project_id):
        self._require_active(project_id)
        self._stop.set(); self._gate.set()
        runner = self.get_runner()
        if runner.phase in ("ready", "thinking", "executing", "initializing", "pausing", "paused"):
            await runner.stop()
        self._event("중지 요청 · 현재 학습 update 종료 후 멈춥니다.")
        # Thread cancellation is cooperative. Ownership stays held until it exits.

    def _require_active(self, project_id):
        if not self.active or self._owner != project_id:
            raise StateConflict("이 프로젝트에서 진행 중인 파이프라인이 없습니다.")

    async def close(self):
        if self.active:
            await self.stop(self._owner)
            await self._task

    async def _execute(self, request):
        try:
            if request.mode == "manual":
                await self._stage(request.stage)
            else:
                cfg = self._state["config"]
                for cycle in range(1, cfg["max_bc_cycles"] + 1):
                    self._state["status"]["cycle"] = cycle
                    for stage in ("collect", "bc", "bc_eval"):
                        result = await self._stage(stage)
                    if result["success_rate"] >= cfg["bc_success_target"]:
                        break
                else:
                    raise StateConflict("BC 최대 반복 횟수에 도달했습니다. 데이터와 평가 결과를 확인하세요.")
                await self._stage("progress")
                for cycle in range(1, cfg["max_rl_rounds"] + 1):
                    self._state["status"]["cycle"] = cycle
                    await self._stage("rl")
                    result = await self._stage("rl_eval")
                    if result["success_rate"] >= cfg["rl_success_target"]:
                        break
                else:
                    raise StateConflict("RL 최대 반복 횟수에 도달했습니다. checkpoint와 평가 결과를 확인하세요.")
            self._state["status"]["phase"] = "completed"
            self._event("요청한 파이프라인 실행 완료")
        except (PipelineStopped, InterruptedError):
            self._state["status"]["phase"] = "stopped"
            self._event("파이프라인 중지 · 기록과 완료된 모델을 유지합니다.")
        except Exception as exc:
            runner = self.get_runner()
            if runner.phase in ("ready", "thinking", "executing", "initializing", "pausing"):
                await runner.pause()
            self._state["status"].update(phase="error", error=self.get_runner()._safe_error(exc))
            self._event(self._state["status"]["error"], "error")
        finally:
            # Stop/pause requests can return control to this task while the
            # Runner is still awaiting an optimizer worker. Join that loop
            # before reading its tensors/replay or releasing robot ownership.
            if self._state["status"]["stage"] in ("collect", "bc_eval", "rl", "rl_eval"):
                await self._quiesce_rollout()
            # Preserve useful online updates when an operator stops mid-round.
            if (self._state["status"]["stage"] == "rl" and self._trainer is not None
                    and self._trainer.online_steps > self._saved_rl_steps):
                try:
                    await self._save_residual({"partial": True,
                        "end_reason": self._state["status"]["phase"], "created_at": time.time()})
                except Exception as exc:
                    self._state["status"]["checkpoint_error"] = self.get_runner()._safe_error(exc)
            self._state["status"]["finished_at"] = time.time()
            self._save()

    async def _quiesce_rollout(self):
        runner = self.get_runner()
        task = getattr(runner, "_task", None)
        if runner.phase == "completed":
            # Keep the verified reset scene eligible for the next stage.
            if task is not None:
                await task
        elif (task is not None and not task.done()) or runner.phase in (
                "ready", "thinking", "executing", "initializing", "pausing", "paused"):
            self._stop.set()
            self._gate.set()  # Wake a learner paused at an update boundary.
            await runner.stop()

    def _snapshot(self, image_size):
        from rilo.learning.data import snapshot_project
        return snapshot_project(self.settings.data_dir, self._owner, image_size=image_size)

    def _artifact_folder(self, kind):
        model_id = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        folder = self._path(self._owner) / "models" / model_id
        folder.mkdir(parents=True, exist_ok=False)
        return model_id, folder

    def _register(self, kind, model_id, result, base_id=None):
        model = {"id": model_id, "name": model_id, "kind": kind, "status": "ready",
            "created_at": time.time(), "base_model_id": base_id, **result}
        self._state["models"].append(model)
        self._save()
        return model

    async def _stage(self, stage):
        await self._async_control()
        self._state["status"].update(stage=stage, metrics={})
        self._event(f"{stage} 단계 시작")
        cfg = self._state["config"]
        if stage == "bc":
            from rilo.learning.bc_train import train_bc
            snapshot = await asyncio.to_thread(self._snapshot, cfg["bc"]["image_size"])
            model_id, folder = self._artifact_folder("bc")
            result = await asyncio.to_thread(train_bc, snapshot, folder, cfg["bc"], callback=self._metrics, control=self._control)
            await self._async_control()
            return self._register("bc", model_id, result)
        if stage == "progress":
            from rilo.learning.progress import train_progress
            bc = self._latest(self._state, "bc")
            snapshot = await asyncio.to_thread(self._snapshot, bc["config"]["image_size"])
            model_id, folder = self._artifact_folder("progress")
            result = await asyncio.to_thread(train_progress, snapshot, bc["checkpoint"], folder,
                                            cfg["progress"], callback=self._metrics, control=self._control)
            await self._async_control()
            return self._register("progress", model_id, result, bc["id"])
        return await self._rollout_stage(stage)

    async def _rollout_stage(self, stage):
        cfg = self._state["config"]
        policy, bc, residual_model = None, None, None
        metadata = {"pipeline_stage": stage, "job_id": self._state["status"]["job_id"]}
        episodes = cfg["collection_episodes"] if stage == "collect" else cfg["rl_episodes_per_round"] if stage == "rl" else cfg["evaluation_episodes"]
        if stage != "collect":
            from rilo.learning.diffusion import FrozenDPController, device_for
            from rilo.learning.progress import ProgressPredictor
            from rilo.learning.rl_train import ResidualTrainer
            from rilo.learning.runtime import LearnedTaskPolicy, make_judge
            bc = self._latest(self._state, "bc")
            device = str(device_for(cfg["device"]))
            base = await asyncio.to_thread(FrozenDPController, bc["checkpoint"], device=device)
            progress, judge = None, None
            residual_model = self._latest(self._state, "residual", bc["id"])
            if stage in ("rl", "rl_eval"):
                progress_model = self._latest(self._state, "progress", bc["id"])
                if progress_model is None: raise StateConflict("Matching progress checkpoint is required")
                progress = await asyncio.to_thread(ProgressPredictor, progress_model["checkpoint"], device=device)
                if self._trainer is None:
                    resume = residual_model and (stage == "rl_eval" or residual_model["config"] == cfg["residual"])
                    if resume:
                        self._trainer = await asyncio.to_thread(ResidualTrainer.load_checkpoint, residual_model["checkpoint"],
                            device=device, expected_base_fingerprint=base.checkpoint_fingerprint)
                        self._saved_rl_steps = self._trainer.online_steps
                    elif stage == "rl":
                        if residual_model:
                            self._event("Residual 설정이 변경되어 새 residual 학습을 시작합니다. 이전 checkpoint는 유지합니다.")
                        residual_model = None
                        self._trainer = ResidualTrainer(base.feature_dim, base.spec.action_low, base.spec.action_high,
                                                       cfg["residual"], device=device)
                        snapshot = await asyncio.to_thread(self._snapshot, base.spec.image_size)
                        self._metrics(await asyncio.to_thread(self._trainer.add_offline, snapshot, base,
                                                              cancel=self._stop.is_set, control=self._control))
                    else: raise StateConflict("Residual checkpoint is required")
            selected = self.projects.get(self._owner).get("policy_config") or {}
            teacher_settings = replace(self.settings, **selected)
            if progress is not None: judge = make_judge(teacher_settings)
            policy = LearnedTaskPolicy(base, make_policy(teacher_settings),
                residual=self._trainer if stage in ("rl", "rl_eval") else None, training=stage == "rl",
                progress=progress, judge=judge, progress_threshold=cfg["progress_gate_threshold"],
                judge_interval=cfg["gpt_judge_interval"], callback=self._metrics, control=self._control)
            metadata.update(base_model_id=bc["id"], residual_model_id=residual_model["id"] if residual_model else None,
                            base_checkpoint_sha256=base.checkpoint_fingerprint,
                            residual_checkpoint_sha256=residual_model.get("checkpoint_sha256") if residual_model else None,
                            evaluation=stage in ("bc_eval", "rl_eval"))
            if progress is not None:
                metadata.update(progress_model_id=progress_model["id"],
                                progress_checkpoint_sha256=progress_model.get("checkpoint_sha256"))
        await self._async_control()
        runner = await self.prepare_run(self._owner, episodes, policy, metadata, cfg["run_budget_usd"])
        await self._async_control()
        if runner.phase in ("ready", "paused"):
            await runner.start()
        while runner.phase not in ("completed", "error", "stopped"):
            if self._stop.is_set(): raise PipelineStopped("Pipeline stopped")
            if runner.phase == "paused" and self._gate.is_set():
                self._gate.clear()
                self._state["status"]["phase"] = "paused"
                self._event("수집이 일시정지되었습니다. 예산과 환경 상태를 확인하세요.")
            await asyncio.sleep(.1)
        if self._stop.is_set(): raise PipelineStopped("Pipeline stopped")
        if runner.phase != "completed": raise StateConflict(runner.error or "수집이 정상 완료되지 않았습니다.")
        await self._quiesce_rollout()
        if runner.stats["reset_successes"] != episodes:
            raise StateConflict("GPT reset이 완료되지 않아 멈췄습니다. 환경을 확인한 뒤 복구하세요.")
        result = {"created_at": time.time(), "kind": "residual" if stage == "rl_eval" else "bc" if stage == "bc_eval" else stage,
            "model_id": bc["id"] if bc else None, "residual_model_id": residual_model["id"] if residual_model else None,
            "episodes": runner.stats["episodes"], "successes": runner.stats["successes"],
            "success_rate": 100 * runner.stats["successes"] / max(1, runner.stats["episodes"]),
            "mean_return": runner.stats["successes"] / max(1, runner.stats["episodes"]),
            "reward_definition": "one sparse reward on verified task success; zero otherwise",
            "run_id": runner.recorder.run_id, "evaluation_protocol": "consecutive GPT resets, same initial seed; no automatic hard resets"}
        self._metrics(result)
        if stage in ("bc_eval", "rl_eval"):
            self._state["evaluations"].append(result)
        if stage == "rl":
            await self._save_residual(result)
        self._save()
        return result

    async def _save_residual(self, result):
        from rilo.learning.checkpoints import fingerprint
        bc = self._latest(self._state, "bc")
        model_id, folder = self._artifact_folder("residual")
        path = folder / "residual.pt"
        await asyncio.to_thread(self._trainer.save_checkpoint, path,
                               metadata={"base_model_id": bc["id"], "rollout": result})
        self._register("residual", model_id, {"checkpoint": str(path.resolve()), "checkpoint_sha256": fingerprint(path),
            "metrics": {**self._trainer.last_metrics, **result}, "config": self._trainer.config.model_dump(mode="json"),
            "dataset": self._trainer.offline_stats}, bc["id"])
        self._saved_rl_steps = self._trainer.online_steps

    def checkpoint(self, project_id, model_id):
        state = self._load(project_id)
        model = next((m for m in state["models"] if m["id"] == model_id), None)
        if model is None: raise FileNotFoundError(model_id)
        path = Path(model["checkpoint"])
        if not path.is_file() or not path.resolve().is_relative_to(self._path(project_id).resolve()):
            raise FileNotFoundError(model_id)
        return path
