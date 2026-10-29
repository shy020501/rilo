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
from rilo.pipeline_plan import PipelinePlanner, evaluation_key, evaluation_signature, evaluation_reset_mode
from rilo.learning_history import bc_history, identifier, read_json, safe_path, training_identity


STAGES = ("collect", "bc", "bc_eval", "progress", "rl", "rl_eval")
MODEL_ALGORITHMS = {"bc": ("dp", "Diffusion Policy"),
                    "residual": ("residual_td3", "Residual TD3"),
                    "progress": ("progress", "Progress model")}


def algorithm_catalog():
    return {"bc": [{"id": "dp", "label": "Diffusion Policy", "available": True},
                   {"id": "pi0", "label": "π0", "available": False, "reason": "아직 구현되지 않았습니다."},
                   {"id": "pi0_5", "label": "π0.5", "available": False, "reason": "아직 구현되지 않았습니다."}],
            "rl": [{"id": "residual_td3", "label": "Residual TD3", "available": True}]}


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    bc_algorithm: Literal["dp"] = "dp"
    rl_algorithm: Literal["residual_td3"] = "residual_td3"
    collection_episodes: int = Field(default=20, ge=1, le=100)
    evaluation_episodes: int = Field(default=10, ge=1, le=100)
    evaluation_reset_mode: Literal["inherit", "agent", "environment", "free_space"] = "inherit"
    bc_success_target: float = Field(default=70, gt=0, le=100)
    rl_success_target: float = Field(default=90, gt=0, le=100)
    max_bc_cycles: int = Field(default=10, ge=1, le=1000)
    max_rl_rounds: int = Field(default=20, ge=1, le=1000)
    rl_episodes_per_round: int = Field(default=5, ge=1, le=100)
    progress_gate_threshold: float = Field(default=.8, ge=0, le=1)
    gpt_judge_interval: int = Field(default=20, ge=1, le=10000)
    run_budget_usd: float = Field(default=15, gt=0, le=1000)
    device: str = "cuda:0"
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
        self._start_lock = asyncio.Lock()
        self._starting_project = None
        self._owner = None
        self._state = None
        self._gate = threading.Event()
        self._gate.set()
        self._stop = threading.Event()
        self._trainer = None
        self._saved_rl_steps = 0
        self._last_persist = 0.
        self._planner = PipelinePlanner(projects, self.root)
        self._trainer_dataset = None
        self._active_bc = self._active_progress = None

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
            status = self._state["status"]
            status["events"].append({"time": time.time(), "kind": kind, "message": message,
                "stage": status.get("stage"), "requested_stage": status.get("requested_stage"),
                "job_id": status.get("job_id"), "stage_started_at": status.get("stage_started_at")})
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

    def view(self, project_id, *, force=False):
        with self._lock:
            state = self._load(project_id)
        installed = importlib.util.find_spec("torch") is not None
        dependencies = {"torch": installed, "cuda": False}
        if installed:
            import torch
            dependencies["cuda"] = torch.cuda.is_available()
            state["config"] = learning_configs(state["config"])
        project = self.projects.get(project_id)
        if installed:
            plan = self._planner.plan(project_id, state, force=force)
            state.update(completion=plan["completion"], auto_plan=plan["auto_plan"])
            successes = plan["completion"]["collect"]["trainable_successes"]
            bc, progress, residual = (plan["models"][kind] for kind in ("bc", "progress", "residual"))
            passed = plan["completion"]["stages"]["bc_eval"]["complete"]
        else:
            successes, bc, progress, residual, passed = 0, None, None, None, False
        teacher_ready = self.settings.policy != "openai" or bool(self.settings.api_key)
        matching_backend = project["backend"] == self.settings.backend
        physical = teacher_ready and matching_backend
        reset_mode = evaluation_reset_mode(project, state["config"])
        simulation = (self.settings.backend in ("demo", "libero")
                      or getattr(self.get_runner(), "context", {}).get("simulation") is True)
        eval_supported = reset_mode != "environment" or simulation
        bc_eval_ready = matching_backend and eval_supported and (reset_mode == "environment" or teacher_ready)
        checks = {"collect": (physical, "서버 환경과 GPT API key를 확인하세요."),
            "bc": (installed and successes >= 2, "학습/검증 분리를 위해 성공한 task episode가 2개 이상 필요합니다."),
            "bc_eval": (bc_eval_ready and installed and bool(bc),
                "현재 BC checkpoint와 평가 환경을 확인하세요. GPT reset은 API key가 필요하고 환경 초기화는 시뮬레이터에서만 지원합니다."),
            "progress": (installed and bool(bc) and passed and successes >= 2, "현재 BC 모델이 목표 평가 성공률에 도달해야 합니다."),
            "rl": (physical and installed and bool(bc) and passed and (bool(progress) or successes >= 2),
                "목표 평가를 통과한 BC와 재사용 가능한 Progress 모델 또는 학습 가능한 성공 episode 2개 이상이 필요합니다."),
            "rl_eval": (physical and eval_supported and installed and bool(residual),
                "Residual checkpoint와 평가 환경을 확인하세요. RL의 GPT 성공 판단에는 API key가 필요합니다."),
            "auto": (physical and installed, "학습 의존성·실행 환경·GPT API key를 확인하세요.")}
        state["readiness"] = {k: {"ready": bool(ok), "reason": "" if ok else reason} for k, (ok, reason) in checks.items()}
        state["dependencies"] = dependencies
        state["capabilities"] = {"bc": installed, "progress": installed, "rl": installed, "evaluation": installed}
        state["algorithm_catalog"] = algorithm_catalog()
        state["rl_preparation"] = {"progress_required": not bool(progress),
                                   "progress_model_id": progress["id"] if progress else None}
        for model in state["models"]:
            model["checkpoint_url"] = f"/api/projects/{project_id}/models/{model['id']}/checkpoint"
            algorithm = MODEL_ALGORITHMS.get(model.get("kind"))
            if algorithm:
                model.setdefault("algorithm", algorithm[0])
                if model["algorithm"] == algorithm[0]:
                    model.setdefault("algorithm_label", algorithm[1])
        identity = training_identity(state, live=self.active and self._owner == project_id)
        state["bc_training"] = ({key: value for key, value in identity.items() if key != "model"}
                                if identity else None)
        if state["bc_training"]:
            state["bc_training"]["history_url"] = (
                f"/api/projects/{project_id}/learning/bc/history?model_id={identity['model_id']}")
        return state

    def configure(self, project_id, value):
        if self.active and self._owner == project_id or self._starting_project == project_id:
            raise StateConflict("파이프라인을 중지한 뒤 학습 설정을 변경하세요.")
        with self._lock:
            state = self._load(project_id)
            # Nested dicts are complete model configs, not silently partial overrides.
            old_target = state["config"]["collection_episodes"]
            state["config"] = learning_configs({**state["config"], **value})
            if state["config"]["collection_episodes"] != old_target:
                # An explicit goal edit supersedes extensions from evaluations
                # already seen; a later failed evaluation may extend it again.
                target = state["config"]["collection_episodes"]
                state["auto_progress"] = {"configured_collection_episodes": target,
                    "collection_goal": target, "goal_override_evaluations": [
                        evaluation_key(item) for item in state.get("evaluations", []) if item.get("kind") == "bc"]}
            previous_owner, previous_state = self._owner, self._state
            self._owner, self._state = project_id, state
            try: self._save()
            finally:
                if self.active:
                    self._owner, self._state = previous_owner, previous_state
        return self.view(project_id)

    async def start(self, project_id, request: PipelineStart):
        async with self._start_lock:
            self._starting_project = project_id
            try:
                return await self._start(project_id, request)
            finally:
                self._starting_project = None

    async def _start(self, project_id, request: PipelineStart):
        if self.active:
            raise StateConflict("다른 학습/수집 작업이 실행 중입니다. 완료하거나 중지한 뒤 시작하세요.")
        stage = "auto" if request.mode == "auto" else request.stage
        if stage is None:
            raise ValueError("단계별 실행에는 stage가 필요합니다.")
        state = await asyncio.to_thread(self.view, project_id, force=True)
        if not state["readiness"][stage]["ready"]:
            raise StateConflict(state["readiness"][stage]["reason"])
        runner = self.get_runner()
        if runner.phase in ("initializing", "thinking", "executing", "pausing", "paused", "ready"):
            raise StateConflict("기존 수집을 중지한 뒤 파이프라인을 시작하세요.")
        self._owner = project_id
        self._state = {k: state[k] for k in ("config", "status", "models", "evaluations", "auto_progress", "bc_trainings") if k in state}
        initial_stage = request.stage or state.get("auto_plan", {}).get("next_stage") or "collect"
        self._state["status"] = {"phase": "running", "stage": initial_stage, "mode": request.mode,
            "requested_stage": request.stage if request.mode == "manual" else "auto",
            "cycle": 0, "metrics": {}, "stage_metrics": {}, "events": [], "error": None,
            "job_id": uuid.uuid4().hex, "started_at": time.time()}
        self._trainer = None
        self._trainer_dataset = None
        self._active_bc = self._active_progress = None
        self._saved_rl_steps = 0
        self._stop.clear(); self._gate.set()
        self._save()
        self._task = asyncio.create_task(self._execute(request), name="rilo-learning-pipeline")
        return await asyncio.to_thread(self.view, project_id)

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
                if request.stage == "rl":
                    await self._prepare_manual_rl()
                await self._stage(request.stage)
            else:
                await self._execute_auto()
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
            for training in self._state.get("bc_trainings", []):
                if (training.get("job_id") == self._state["status"].get("job_id")
                        and training.get("phase") in ("running", "paused")):
                    training.update(phase=self._state["status"]["phase"], finished_at=time.time())
                    self._state["status"]["bc_training"] = deepcopy(training)
            self._save()

    async def _plan(self):
        with self._lock:
            state = deepcopy(self._state)
        return await asyncio.to_thread(self._planner.plan, self._owner, state)

    async def _prepare_manual_rl(self):
        """Prepare the matching Progress dependency under the same RL job owner."""
        await self._async_control()
        plan = await self._plan()
        if not plan["models"]["bc"] or not plan["completion"]["stages"]["bc_eval"]["complete"]:
            raise StateConflict("현재 BC 모델이 목표 평가 성공률을 통과해야 RL을 시작할 수 있습니다.")
        progress = plan["models"]["progress"]
        if progress is None:
            if plan["completion"]["collect"]["trainable_successes"] < 2:
                raise StateConflict("RL 준비용 Progress 학습에는 호환되는 성공 episode가 2개 이상 필요합니다.")
            self._event("RL 준비 · 현재 BC와 데이터에 맞는 Progress 모델을 학습합니다.")
            await self._stage("progress")
            await self._async_control()
            plan = await self._plan()
            if (plan["models"]["progress"] is None
                    or not plan["completion"]["stages"]["bc_eval"]["complete"]):
                raise StateConflict("RL 준비 후 현재 BC 평가와 Progress checkpoint를 확인할 수 없습니다.")
        else:
            self._event(f"RL 준비 · Progress 모델 {progress['id']}을 재사용합니다.")
        # Pause/Stop may arrive while the worker finishes or registers its
        # checkpoint. Never prepare a robot rollout before the gate is open.
        await self._async_control()

    def _persist_collection_goal(self, plan):
        cfg = self._state["config"]
        progress = self._state.setdefault("auto_progress", {})
        if progress.get("configured_collection_episodes") != cfg["collection_episodes"]:
            progress.clear()
            progress.update(configured_collection_episodes=cfg["collection_episodes"],
                            collection_goal=cfg["collection_episodes"])
        if plan["extension"]:
            progress.update(plan["extension"])
            self._event(f"BC 목표 미달 · 누적 성공 수집 목표를 {progress['collection_goal']}개로 늘립니다.")
        self._save()

    async def _execute_auto(self):
        cfg = self._state["config"]
        bc_cycles = rl_rounds = empty_batches = 0
        while True:
            await self._async_control()
            plan = await self._plan()
            self._persist_collection_goal(plan)
            stage = plan["auto_plan"]["next_stage"]
            if stage is None:
                return
            if stage == "collect":
                if bc_cycles >= cfg["max_bc_cycles"]:
                    raise StateConflict("BC 최대 반복 횟수에 도달했습니다. 데이터와 평가 결과를 확인하세요.")
                remaining = plan["completion"]["collect"]["remaining"]
                if remaining <= 0:
                    raise StateConflict("동일 데이터의 반복 학습을 막았습니다. 새 성공 시연을 수집한 뒤 다시 시작하세요.")
                before = plan["completion"]["collect"]["completed"]
                await self._stage("collect", episodes=min(100, remaining))
                after = (await self._plan())["completion"]["collect"]["completed"]
                empty_batches = empty_batches + 1 if after <= before else 0
                if empty_batches >= cfg["max_bc_cycles"]:
                    raise StateConflict("성공 episode가 늘지 않아 자동 수집을 중지했습니다. 환경·정책·예산을 확인하세요.")
                continue
            if stage == "bc":
                if bc_cycles >= cfg["max_bc_cycles"]:
                    raise StateConflict("BC 최대 반복 횟수에 도달했습니다. 데이터와 평가 결과를 확인하세요.")
                bc_cycles += 1
                self._state["status"]["cycle"] = bc_cycles
            elif stage == "rl":
                if rl_rounds >= cfg["max_rl_rounds"]:
                    raise StateConflict("RL 최대 반복 횟수에 도달했습니다. checkpoint와 평가 결과를 확인하세요.")
                rl_rounds += 1
                self._state["status"]["cycle"] = rl_rounds
            await self._stage(stage)
            updated = await self._plan()
            self._persist_collection_goal(updated)
            # Missing receipts/checkpoints must not spin or silently count as done.
            if updated["auto_plan"]["next_stage"] == stage:
                raise StateConflict(f"{stage} 완료 증거를 확인할 수 없어 자동 실행을 중지했습니다.")

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
        return self._planner.snapshot(self._owner, image_size)

    def _artifact_folder(self, kind):
        model_id = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        folder = self._path(self._owner) / "models" / model_id
        folder.mkdir(parents=True, exist_ok=False)
        return model_id, folder

    def _register(self, kind, model_id, result, base_id=None):
        model = {"id": model_id, "name": model_id, "kind": kind, "status": "ready",
            "created_at": time.time(), "base_model_id": base_id, **result}
        if kind in MODEL_ALGORITHMS:
            model["algorithm"], model["algorithm_label"] = MODEL_ALGORITHMS[kind]
        self._state["models"].append(model)
        if kind == "bc":
            for training in self._state.get("bc_trainings", []):
                if training["model_id"] == model_id:
                    training.update(phase="completed", finished_at=time.time())
                    self._state["status"]["bc_training"] = deepcopy(training)
        self._save()
        return model

    async def _stage(self, stage, *, episodes=None):
        await self._async_control()
        self._state["status"].update(stage=stage, metrics={}, stage_started_at=time.time())
        self._state["status"].setdefault("stage_metrics", {})[stage] = {}
        self._event(f"{stage} 단계 시작")
        cfg = self._state["config"]
        if stage == "bc":
            from rilo.learning.bc_train import train_bc
            snapshot = await asyncio.to_thread(self._snapshot, cfg["bc"]["image_size"])
            model_id, folder = self._artifact_folder("bc")
            identity = {"model_id": model_id, "job_id": self._state["status"]["job_id"],
                        "target_steps": cfg["bc"]["steps"], "started_at": time.time(), "phase": "running"}
            self._state.setdefault("bc_trainings", []).append(identity)
            self._state["status"]["bc_training"] = deepcopy(identity)
            self._save()
            result = await asyncio.to_thread(train_bc, snapshot, folder, cfg["bc"], callback=self._metrics, control=self._control)
            await self._async_control()
            return self._register("bc", model_id, result)
        if stage == "progress":
            from rilo.learning.progress import train_progress
            bc = (await self._plan())["models"]["bc"]
            if bc is None:
                raise StateConflict("현재 데이터·설정에 맞는 BC checkpoint가 필요합니다.")
            snapshot = await asyncio.to_thread(self._snapshot, bc["config"]["image_size"])
            model_id, folder = self._artifact_folder("progress")
            result = await asyncio.to_thread(train_progress, snapshot, bc["checkpoint"], folder,
                                            cfg["progress"], callback=self._metrics, control=self._control)
            await self._async_control()
            result["provenance"] = {"base_model_id": bc["id"],
                                    "base_checkpoint_sha256": bc["checkpoint_sha256"]}
            return self._register("progress", model_id, result, bc["id"])
        return await self._rollout_stage(stage, episodes=episodes)

    async def _rollout_stage(self, stage, *, episodes=None):
        cfg = self._state["config"]
        project = self.projects.get(self._owner)
        is_evaluation = stage in ("bc_eval", "rl_eval")
        reset_mode = (evaluation_reset_mode(project, cfg) if is_evaluation else
                      "free_space" if project["run_config"].get("reset_mode") == "free_space" else "agent")
        signature = evaluation_signature(project, cfg, stage) if is_evaluation else None
        policy, bc, residual_model = None, None, None
        metadata = {"pipeline_stage": stage, "job_id": self._state["status"]["job_id"],
                    "bc_algorithm": cfg.get("bc_algorithm", "dp"),
                    "rl_algorithm": cfg.get("rl_algorithm", "residual_td3")}
        if is_evaluation:
            metadata.update(reset_mode=reset_mode, evaluation_reset_mode=cfg["evaluation_reset_mode"])
            selected = replace(self.settings, **(project.get("policy_config") or {}))
            metadata["evaluation_config_snapshot"] = {"run_config": deepcopy(project["run_config"]),
                "policy_config": {"model": selected.model, "reasoning_effort": selected.reasoning_effort}}
        episodes = episodes if episodes is not None else cfg["collection_episodes"] if stage == "collect" else cfg["rl_episodes_per_round"] if stage == "rl" else cfg["evaluation_episodes"]
        if stage != "collect":
            from rilo.learning.diffusion import FrozenDPController, device_for
            from rilo.learning.progress import ProgressPredictor
            from rilo.learning.rl_train import ResidualTrainer
            from rilo.learning.runtime import LearnedTaskPolicy, make_judge
            selected_models = (await self._plan())["models"]
            bc = selected_models["bc"]
            if bc is None:
                raise StateConflict("현재 데이터·설정에 맞는 BC checkpoint가 필요합니다.")
            self._active_bc = bc
            device = str(device_for(cfg["device"]))
            base = await asyncio.to_thread(FrozenDPController, bc["checkpoint"], device=device)
            progress, judge = None, None
            residual_model = selected_models["residual_resume" if stage == "rl" else "residual"]
            if stage in ("rl", "rl_eval"):
                progress_model = selected_models["progress"]
                if progress_model is None: raise StateConflict("Matching progress checkpoint is required")
                self._active_progress = progress_model
                progress = await asyncio.to_thread(ProgressPredictor, progress_model["checkpoint"], device=device)
                if self._trainer is None:
                    resume = residual_model and (stage == "rl_eval" or residual_model["config"] == cfg["residual"])
                    if resume:
                        self._trainer = await asyncio.to_thread(ResidualTrainer.load_checkpoint, residual_model["checkpoint"],
                            device=device, expected_base_fingerprint=base.checkpoint_fingerprint)
                        self._saved_rl_steps = self._trainer.online_steps
                        self._trainer_dataset = residual_model.get("provenance", {}).get("dataset")
                    elif stage == "rl":
                        if residual_model:
                            self._event("Residual 설정이 변경되어 새 residual 학습을 시작합니다. 이전 checkpoint는 유지합니다.")
                        residual_model = None
                        self._trainer = ResidualTrainer(base.feature_dim, base.spec.action_low, base.spec.action_high,
                                                       cfg["residual"], device=device)
                        snapshot = await asyncio.to_thread(self._snapshot, base.spec.image_size)
                        self._trainer_dataset = snapshot.manifest()
                        self._metrics(await asyncio.to_thread(self._trainer.add_offline, snapshot, base,
                                                              cancel=self._stop.is_set, control=self._control))
                    else: raise StateConflict("Residual checkpoint is required")
            selected = project.get("policy_config") or {}
            teacher_settings = replace(self.settings, **selected)
            if progress is not None: judge = make_judge(teacher_settings)
            reset_policy = None if is_evaluation and reset_mode == "environment" else make_policy(teacher_settings)
            policy = LearnedTaskPolicy(base, reset_policy,
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
        if runner.stats["episodes"] != episodes:
            raise StateConflict("요청한 평가/수집 episode 수가 완료되지 않았습니다.")
        if reset_mode == "environment":
            if runner.stats.get("hard_resets", 0) != max(0, episodes - 1):
                raise StateConflict("평가 episode 사이 환경 초기화가 완료되지 않았습니다.")
        elif runner.stats["reset_successes"] != episodes:
            raise StateConflict("GPT reset이 완료되지 않아 멈췄습니다. 환경을 확인한 뒤 복구하세요.")
        initial_seed = signature["run_config"].get("seed", 0) if signature else project["run_config"].get("seed", 0)
        protocol = (f"Simulator reset before the first episode and between episodes; seeds {initial_seed} through "
                    f"{initial_seed + episodes - 1}; no GPT reset and no final reset after the last episode."
                    if reset_mode == "environment" else
                    f"GPT restoration and verified free-space retreat before simulator reset after every episode; seed {initial_seed}."
                    if reset_mode == "free_space" else
                    f"Consecutive GPT physical resets after every episode; initial seed {initial_seed}; no automatic environment resets.")
        result = {"created_at": time.time(), "kind": "residual" if stage == "rl_eval" else "bc" if stage == "bc_eval" else stage,
            "model_id": bc["id"] if bc else None, "residual_model_id": residual_model["id"] if residual_model else None,
            "episodes": runner.stats["episodes"], "successes": runner.stats["successes"],
            "success_rate": 100 * runner.stats["successes"] / max(1, runner.stats["episodes"]),
            "mean_return": runner.stats["successes"] / max(1, runner.stats["episodes"]),
            "reward_definition": "one sparse reward on verified task success; zero otherwise",
            "run_id": runner.recorder.run_id, "reset_mode": reset_mode,
            "reset_successes": runner.stats["reset_successes"], "environment_resets": runner.stats.get("hard_resets", 0),
            "evaluation_protocol": protocol}
        if stage in ("bc_eval", "rl_eval"):
            result["evaluation_reset_mode"] = cfg["evaluation_reset_mode"]
            result["provenance"] = {key: metadata.get(key) for key in (
                "base_model_id", "base_checkpoint_sha256", "residual_model_id", "residual_checkpoint_sha256",
                "progress_model_id", "progress_checkpoint_sha256")}
            result["provenance"]["evaluation"] = signature
            # Freeze this baseline so viewing a failed manual evaluation does
            # not move its proposed collection goal as new data arrives later.
            result["collection_successes"] = (await self._plan())["completion"]["collect"]["completed"]
        self._metrics(result)
        if stage in ("bc_eval", "rl_eval"):
            self._state["evaluations"].append(result)
            if stage == "bc_eval":
                plan = await self._plan()
                if plan["extension"]:
                    self._persist_collection_goal(plan)
        if stage == "rl":
            await self._save_residual(result)
        self._save()
        return result

    async def _save_residual(self, result):
        from rilo.learning.checkpoints import fingerprint
        bc = self._active_bc or (await self._plan())["models"]["bc"]
        progress = self._active_progress or (await self._plan())["models"]["progress"]
        if bc is None or progress is None:
            raise StateConflict("Residual checkpoint의 BC·progress 출처를 확인할 수 없습니다.")
        model_id, folder = self._artifact_folder("residual")
        path = folder / "residual.pt"
        await asyncio.to_thread(self._trainer.save_checkpoint, path,
                               metadata={"base_model_id": bc["id"], "rollout": result})
        self._register("residual", model_id, {"checkpoint": str(path.resolve()), "checkpoint_sha256": fingerprint(path),
            "metrics": {**self._trainer.last_metrics, **result}, "config": self._trainer.config.model_dump(mode="json"),
            "dataset": self._trainer.offline_stats,
            "provenance": {"base_model_id": bc["id"], "base_checkpoint_sha256": bc["checkpoint_sha256"],
                "progress_model_id": progress["id"], "progress_checkpoint_sha256": progress["checkpoint_sha256"],
                "progress_gate_threshold": self._state["config"]["progress_gate_threshold"],
                "gpt_judge_interval": self._state["config"]["gpt_judge_interval"],
                "dataset": self._trainer_dataset}}, bc["id"])
        self._saved_rl_steps = self._trainer.online_steps

    def bc_history(self, project_id, model_id=None):
        """Read only this project's recorded BC logs, without importing torch."""
        identifier(project_id)
        with self.projects._lock:
            if not any(project["id"] == project_id for project in self.projects._load()["projects"]):
                raise FileNotFoundError("Unknown project")
        root = safe_path(self.root, project_id)
        with self._lock:
            live = self._owner == project_id and self.active
            if self._owner == project_id and self._state is not None:
                state = deepcopy(self._state)
            else:
                try:
                    state = read_json(root, "registry.json")
                except FileNotFoundError:
                    state = {}
        return bc_history(root, project_id, state, model_id=model_id, live=live)

    def checkpoint(self, project_id, model_id):
        state = self._load(project_id)
        model = next((m for m in state["models"] if m["id"] == model_id), None)
        if model is None: raise FileNotFoundError(model_id)
        path = Path(model["checkpoint"])
        if not path.is_file() or not path.resolve().is_relative_to(self._path(project_id).resolve()):
            raise FileNotFoundError(model_id)
        return path
