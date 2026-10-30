from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
import asyncio
import json
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from rilo.config import Settings, REMOTE_ROBOT_BACKEND, REMOTE_ROBOT_CONNECTION_ONLY_REASON
from rilo.projects import CAPABILITIES, ProjectCreate, ProjectSettings, ProjectStore, empty_stats
from rilo.policy_settings import PolicyConfig, policy_catalog
from rilo.recording import dump_json
from rilo.runner import Runner, RunConfig, StateConflict
from rilo.pipeline import PipelineEngine, PipelineStart
from rilo.video_jobs import create_video_router
from rilo.robot_connections import create_robot_connection_router
from rilo.remote.api import RemoteAPI, create_rpc_app
from rilo.remote.listener import PolicyRPCListener
from rilo.remote.service import RemoteError, RemoteService


class InitializeProject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preserve_environment: bool = False


class ArchiveEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    archived: bool


class DeleteEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    confirm: bool


class GraspBaselineSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_closures: int | None = Field(ge=0, le=100)


def create_app(settings: Settings | None = None, *, runner: Runner | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    controller = runner or Runner(settings)
    projects = ProjectStore(settings.data_dir)
    active_project_id: str | None = None
    connection_only_mode = False
    ownership_lock = asyncio.Lock()
    retired_controllers: list[Runner] = []

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await remote_api.close()
            await robot_connections.close()
            await video_jobs.close()
            await pipeline.close()
            await controller.close()

    app = FastAPI(title="RILO", lifespan=lifespan)
    app.state.runner = controller
    app.state.projects = projects
    web = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.middleware("http")
    async def local_controls(request: Request, call_next):
        if request.method == "POST" and request.url.path.startswith("/api/"):
            # These are local operator controls. JSON-only + same-origin guards
            # prevent a random website from submitting commands to localhost.
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.url.netloc:
                return Response('{"detail":"Cross-origin control requests are not allowed"}',
                                status_code=403, media_type="application/json")
            if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
                return Response('{"detail":"Controls require application/json"}',
                                status_code=415, media_type="application/json")
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(StateConflict)
    async def state_error(request, exc):
        return Response(dump_json({"detail": str(exc)}), status_code=409, media_type="application/json")

    @app.exception_handler(RemoteError)
    async def remote_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    def snapshot():
        status = controller.status()
        if active_project_id and controller.recorder and status.get("episode", {}).get("id"):
            try:
                number = projects.episode_display_index(active_project_id, controller.recorder.run_id,
                                                       status["episode"]["id"])
                status["episode"] = {**status["episode"], "display_index": number}
            except (KeyError, OSError, ValueError):
                pass
        return Response(dump_json({**status, "active_project_id": active_project_id,
                                   "pipeline_runtime": pipeline.runtime()}), media_type="application/json")

    def require_manual_control():
        require_execution_available(active_project_id)
        if pipeline.active:
            raise StateConflict("파이프라인이 실행 권한을 사용 중입니다. 파이프라인 제어에서 Pause/중지를 사용하세요.")

    def runtime():
        effective = controller.policy_config
        return {"backend": controller.settings.backend, "suite": controller.settings.suite,
                "model": effective["model"], "reasoning_effort": effective["reasoning_effort"],
                "policy": controller.settings.policy, "active_project_id": active_project_id,
                "phase": controller.phase, "run_id": controller.recorder.run_id if controller.recorder else None}

    def policy_options():
        return {"models": policy_catalog(), "defaults": {
            "model": settings.model, "reasoning_effort": settings.reasoning_effort}}

    def project_policy(project):
        return project.get("policy_config") or policy_options()["defaults"]

    def execution_management_reason(run_id, *, ignore_pending_project=None):
        if any(project.get("delete_pending") and project["id"] != ignore_pending_project
               and run_id in project["run_ids"]
               for project in projects._load()["projects"]):
            return "프로젝트를 삭제 중입니다. 데이터 작업을 시작할 수 없습니다."
        if remote_service.owns_run(run_id):
            return "현재 로봇 실행에 속한 데이터입니다. run을 중지한 뒤 변경하세요."
        pipeline_owner = (pipeline.runtime() or {}).get("project_id")
        if pipeline.active and (pipeline_owner is None or run_id in projects.get(pipeline_owner)["run_ids"]):
            return "이 프로젝트의 학습/평가가 실행 중입니다. 파이프라인을 중지한 뒤 데이터를 변경하세요."
        # A retired provider request can still append its usage to an older run.
        # Until it settles, avoid destructive changes to any associated files.
        retired_controllers[:] = [old for old in retired_controllers
                                   if any(not task.done() for task in old._retired_requests)]
        if any(not task.done() for owner in (controller, *retired_controllers)
               for task in owner._retired_requests):
            return "중단된 GPT 요청의 비용 기록이 도착하는 중입니다. 잠시 후 다시 시도하세요."
        if controller.recorder and controller.recorder.run_id == run_id:
            if controller.phase in ("initializing", "ready", "thinking", "executing", "pausing", "paused"):
                return "현재 실행에 속한 데이터입니다. 수집을 중지하거나 실행이 끝난 뒤 변경할 수 있습니다."
            if controller._inference is not None and not controller._inference.done():
                return "GPT 요청이 아직 종료되지 않았습니다. 잠시 후 다시 시도하세요."
        return None

    def management_reason(run_id, *, ignore_pending_project=None):
        if video_jobs.is_busy(run_id):
            return "영상 생성 중입니다. 완료된 뒤 데이터를 변경하세요."
        return execution_management_reason(run_id, ignore_pending_project=ignore_pending_project)

    def episode_controls(episode):
        reason = management_reason(episode["run_id"])
        return {**episode, "can_manage": reason is None, "manage_disabled_reason": reason}

    def get_project(project_id):
        try:
            return projects.get(project_id)
        except KeyError:
            raise HTTPException(404, "Project does not exist") from None

    def require_execution_available(project_id=None):
        if remote_service.active:
            raise StateConflict("원격 로봇 run이 진행 중입니다. 먼저 해당 run을 중지하세요.")
        if project_id is not None:
            project = get_project(project_id)
            if project.get("archived") or project.get("delete_pending"):
                raise StateConflict("보관된 프로젝트를 복원한 뒤 실행하세요.")
        if (settings.backend == REMOTE_ROBOT_BACKEND or project_id is not None
                and get_project(project_id)["backend"] == REMOTE_ROBOT_BACKEND):
            raise StateConflict(REMOTE_ROBOT_CONNECTION_ONLY_REASON)

    def availability(project):
        if project.get("archived") or project.get("delete_pending"):
            reason = "삭제를 마무리하거나 보관된 프로젝트를 복원한 뒤 실행하세요."
            return {"runnable": False, "connectable": False,
                    "connection_only": project["backend"] == REMOTE_ROBOT_BACKEND,
                    "runnable_reason": reason, "unavailable_reason": reason}
        if project["backend"] == REMOTE_ROBOT_BACKEND:
            return {"runnable": False, "connectable": True, "connection_only": True,
                    "runnable_reason": REMOTE_ROBOT_CONNECTION_ONLY_REASON,
                    "unavailable_reason": REMOTE_ROBOT_CONNECTION_ONLY_REASON}
        reason = None
        if project["backend"] != settings.backend:
            reason = f"This server uses {settings.backend}; launch a server configured for {project['backend']} to collect this project."
        return {"runnable": reason is None, "runnable_reason": reason, "unavailable_reason": reason}

    def project_management_reason(project):
        project_id = project["id"]
        pipeline_owner = (pipeline.runtime() or {}).get("project_id")
        if (pipeline._starting_project == project_id
                or pipeline.active and pipeline_owner in (None, project_id)):
            return "이 프로젝트의 학습·평가를 중지한 뒤 프로젝트를 관리하세요."
        for run_id in project["run_ids"]:
            reason = management_reason(run_id, ignore_pending_project=project_id)
            if reason:
                return reason
        if (active_project_id == project_id
                and controller.phase in ("initializing", "ready", "thinking", "executing", "pausing", "paused")):
            return "현재 수집을 중지한 뒤 프로젝트를 관리하세요."
        if project["backend"] == REMOTE_ROBOT_BACKEND:
            session = remote_service.sessions.get(project_id)
            if session and (session.connected or session.run and session.run["status"] in
                            ("preparing", "running", "pausing", "paused", "stopping")):
                return "로봇 run을 중지하고 환경 연결을 해제한 뒤 프로젝트를 관리하세요."
            if project_id in remote_api.jobs or project_id in remote_api.disconnecting:
                return "환경 연결·해제가 끝난 뒤 프로젝트를 관리하세요."
            launcher = remote_api.launcher.snapshot(project_id).get("active_launcher")
            connection = robot_connections.snapshot(project_id).get("active_connection")
            if any(item and item["project_id"] == project_id for item in (launcher, connection)):
                return "SSH·로봇 client 연결을 해제한 뒤 프로젝트를 관리하세요."
        return None

    def project_management(project):
        reason = project_management_reason(project)
        return {"archived": bool(project.get("archived")), "archived_at": project.get("archived_at"),
                "delete_pending": bool(project.get("delete_pending")),
                "can_archive": reason is None and not project.get("delete_pending"),
                "can_delete": reason is None, "manage_disabled_reason": reason}

    def project_detail(project_id):
        result = projects.details(get_project(project_id))
        status = availability(result["project"])
        result["project"].update(status)
        result["project"]["policy_config"] = project_policy(result["project"])
        result["project"].update(project_management(result["project"]))
        result["project"]["pipeline"] = {**result["project"].get("pipeline", {}),
                                          **pipeline.project_goals(project_id)}
        result["episodes"] = [episode_controls(ep) for ep in result["episodes"]]
        result["archived_episodes"] = [episode_controls(ep) for ep in result["archived_episodes"]]
        capabilities = ({**dict.fromkeys(CAPABILITIES, False), "ssh": True}
                        if status.get("connection_only") else CAPABILITIES)
        return {**result, **status, "runtime": runtime(), "capabilities": capabilities,
                "policy_options": policy_options()}

    def require_owner(project_id):
        get_project(project_id)
        require_execution_available(project_id)
        if project_id != active_project_id:
            raise StateConflict("This project does not own the active collection. Connect its environment first.")

    def link_current(project_id):
        if project_id and controller.recorder:
            projects.link_run(project_id, controller.recorder.run_id)

    @app.get("/")
    async def index():
        return FileResponse(web / "workspace.html")

    @app.get("/console")
    async def console():
        return FileResponse(web / "index.html")

    @app.get("/api/status")
    async def status():
        return snapshot()

    @app.get("/api/health")
    async def health():
        return {"ok": True, "backend": controller.settings.backend}

    @app.get("/api/tasks")
    async def tasks(suite: str | None = None):
        if settings.backend == REMOTE_ROBOT_BACKEND:
            return {"suite": "custom", "tasks": [], "connection_only": True}
        try:
            selected = suite or controller.settings.suite
            if selected == controller.settings.suite or controller.settings.backend != "libero":
                result = await controller.tasks()
            else:
                from rilo.adapters.libero import LiberoAdapter
                def catalog():
                    adapter = LiberoAdapter(suite=selected, image_size=settings.image_size)
                    try:
                        return adapter.list_tasks()
                    finally:
                        adapter.close()
                result = await controller._env(catalog)
            if controller.settings.backend == "libero":
                from rilo.adapters.libero import RESET_DRAWER_TASKS
                result = [{**task, "agent_reset_supported": task.get("name") in RESET_DRAWER_TASKS} for task in result]
            return {"suite": selected, "tasks": result}
        except Exception as exc:
            raise HTTPException(503, controller._safe_error(exc)) from None

    @app.get("/api/workspace")
    async def workspace():
        cards, archived_cards = [], []
        totals = {"projects": 0, **empty_stats()}
        for project in projects.list():
            card = projects.details(project)["project"]
            card.update(availability(card))
            card["policy_config"] = project_policy(card)
            card.update(project_management(card))
            card["pipeline"] = {**card.get("pipeline", {}), **pipeline.project_goals(project["id"])}
            if card["archived"]:
                archived_cards.append(card)
            else:
                cards.append(card)
                for key in card["stats"]:
                    totals[key] += card["stats"][key]
        totals["projects"] = len(cards)
        return {"projects": cards, "archived_projects": archived_cards,
                "totals": totals, "runtime": runtime(), "capabilities": CAPABILITIES,
                "pipeline_runtime": pipeline.runtime(),
                "policy_options": policy_options()}

    @app.get("/api/policy-options")
    async def policies():
        return policy_options()

    @app.post("/api/projects", status_code=201)
    async def create_project(data: ProjectCreate):
        project = projects.create(data)
        return {"project": project_detail(project["id"])["project"]}

    @app.get("/api/projects/{project_id}")
    async def project(project_id: str):
        return project_detail(project_id)

    @app.post("/api/projects/{project_id}/archive")
    async def archive_project(project_id: str, data: ArchiveEpisode):
        nonlocal active_project_id, connection_only_mode
        async with ownership_lock:
            project = get_project(project_id)
            reason = project_management_reason(project)
            if reason:
                raise StateConflict(reason)
            try:
                projects.archive(project_id, data.archived)
            except ValueError as exc:
                raise HTTPException(409, controller._safe_error(exc)) from None
            if data.archived and active_project_id == project_id:
                active_project_id, connection_only_mode = None, False
            return project_detail(project_id)

    @app.post("/api/projects/{project_id}/delete")
    async def delete_project(project_id: str, data: DeleteEpisode):
        nonlocal active_project_id, connection_only_mode
        if data.confirm is not True:
            raise HTTPException(400, "Permanent project deletion requires confirm=true")
        async with ownership_lock:
            project = get_project(project_id)
            reason = project_management_reason(project)
            if reason:
                raise StateConflict(reason)
            try:
                projects.begin_delete(project_id)
                if project["backend"] == REMOTE_ROBOT_BACKEND:
                    await remote_api.launcher.forget_project(project_id)
                    await robot_connections.forget_project(project_id)
                    remote_service.revoke(project_id)
                # Completed recorders must not rewrite a deleted run on a later
                # shutdown. Detaching metadata does not reset the environment.
                other_runs = {run_id for other in projects.list() if other["id"] != project_id
                              for run_id in other["run_ids"]}
                deleted_runs = set(project["run_ids"]) - other_runs
                if controller.recorder and controller.recorder.run_id in deleted_runs:
                    controller.recorder.close()
                    controller.recorder = None
                if active_project_id == project_id:
                    active_project_id, connection_only_mode = None, False
                result = await asyncio.to_thread(projects.delete, project_id)
                if pipeline._owner == project_id and not pipeline.active:
                    pipeline._owner = pipeline._state = None
                return result
            except (KeyError, FileNotFoundError):
                raise HTTPException(404, "Project does not exist") from None
            except ValueError as exc:
                raise HTTPException(400, controller._safe_error(exc)) from None
            except OSError:
                raise HTTPException(500, "프로젝트 삭제가 완료되지 않았습니다. 보관함에서 삭제를 다시 시도하세요.") from None

    @app.post("/api/projects/{project_id}/settings")
    async def project_settings(project_id: str, data: ProjectSettings):
        async with ownership_lock:
            if get_project(project_id).get("delete_pending"):
                raise StateConflict("삭제 중인 프로젝트의 설정은 변경할 수 없습니다.")
            saved = projects.update(project_id, data)
            episode_goal_applied = False
            if (data.run_config is not None and active_project_id == project_id
                    and not pipeline.active):
                episode_goal_applied = await controller.update_prepared_episode_goal(
                    saved["run_config"]["episodes"])
            return {"project": project_detail(project_id)["project"],
                    "episode_goal_applied": episode_goal_applied}

    @app.post("/api/projects/{project_id}/grasp-baseline")
    async def grasp_baseline_settings(project_id: str, data: GraspBaselineSettings):
        async with ownership_lock:
            get_project(project_id)
            baseline = projects.set_grasp_baseline(project_id,
                {"expected_closures": data.expected_closures} if data.expected_closures is not None else None)
            return {"project": project_detail(project_id)["project"], "grasp_baseline": baseline}

    def configure_baseline(project, *, allow_estimation=True):
        baseline = projects.grasp_baseline(project)
        provider = save = None
        if (allow_estimation and baseline is None and controller.settings.policy == "openai"
                and type(controller.context.get("gripper_control", {}).get("action_index")) is int):
            from rilo.grasp_baseline import estimate_grasp_baseline
            selected = replace(controller.settings, **controller.policy_config)
            async def provider(obs, spec, context, callback):
                return await estimate_grasp_baseline(selected, obs, spec, context, usage_callback=callback)
            def save(value):
                # A manual override made during inference wins for future runs;
                # this run still uses its own accepted estimate snapshot.
                current = projects.grasp_baseline(project)
                if current is not None:
                    return {**value, "source": "gpt", "version": None}
                return projects.set_grasp_baseline(project["id"], value, source="gpt")
        controller.configure_grasp_baseline(baseline, provider=provider, save=save)

    @app.get("/api/projects/{project_id}/episodes/{run_id}/{episode_id}/detail")
    async def episode_detail(project_id: str, run_id: str, episode_id: str,
                             offset: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=100)):
        try:
            result = projects.episode_detail(project_id, run_id, episode_id, offset=offset, limit=limit)
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "Episode does not exist in this project") from None
        except ValueError as exc:
            raise HTTPException(400, controller._safe_error(exc)) from None
        result["episode"] = episode_controls(result["episode"])
        return result

    @app.post("/api/projects/{project_id}/episodes/{run_id}/{episode_id}/archive")
    async def archive_episode(project_id: str, run_id: str, episode_id: str, data: ArchiveEpisode):
        async with ownership_lock:
            reason = management_reason(run_id)
            if reason:
                raise StateConflict(reason)
            try:
                projects.archive_episode(project_id, run_id, episode_id, archived=data.archived)
            except (KeyError, FileNotFoundError):
                raise HTTPException(404, "Episode does not exist in this project") from None
            except ValueError as exc:
                raise HTTPException(400, controller._safe_error(exc)) from None
            return project_detail(project_id)

    @app.post("/api/projects/{project_id}/episodes/{run_id}/{episode_id}/delete")
    async def delete_episode(project_id: str, run_id: str, episode_id: str, data: DeleteEpisode):
        if not data.confirm:
            raise HTTPException(400, "Permanent deletion requires confirm=true")
        async with ownership_lock:
            reason = management_reason(run_id)
            if reason:
                raise StateConflict(reason)
            try:
                projects.delete_episode(project_id, run_id, episode_id)
            except (KeyError, FileNotFoundError):
                raise HTTPException(404, "Episode does not exist in this project") from None
            except ValueError as exc:
                raise HTTPException(400, controller._safe_error(exc)) from None
            except OSError:
                raise HTTPException(500, "Deletion could not finish. Some files may remain; retry this deletion to complete cleanup.") from None
            if controller.recorder and controller.recorder.run_id == run_id:
                controller.recorder.exclude_episode(episode_id)
            return {**project_detail(project_id), "deleted": True}

    @app.post("/api/projects/{project_id}/initialize")
    async def initialize_project(project_id: str, request: InitializeProject):
        require_manual_control()
        return await initialize_owned_project(project_id, request)

    @app.post("/api/projects/{project_id}/connect")
    async def connect_project(project_id: str):
        nonlocal controller, active_project_id, connection_only_mode
        async with ownership_lock:
            require_manual_control()
            project = get_project(project_id)
            require_execution_available(project_id)
            if not availability(project)["runnable"]:
                raise StateConflict(availability(project)["unavailable_reason"])
            if controller.phase in ("thinking", "executing", "initializing", "pausing", "paused") and controller.status()["run_started"]:
                raise StateConflict("Stop the current run before reconnecting.")
            if active_project_id != project_id and controller.phase == "ready":
                raise StateConflict("Stop the prepared run before connecting another project.")
            try:
                if project["suite"] != controller.settings.suite:
                    if runner is not None:
                        raise StateConflict("Injected controller cannot change suite.")
                    await controller.close()
                    if any(not task.done() for task in controller._retired_requests):
                        retired_controllers.append(controller)
                    controller = Runner(replace(settings, suite=project["suite"], task_id=project["task_id"]))
                    app.state.runner = controller
                await controller.connect_environment(task_id=project["task_id"], seed=project["run_config"]["seed"])
                active_project_id, connection_only_mode = project_id, True
            except StateConflict:
                raise
            except Exception as exc:
                raise HTTPException(503, controller._safe_error(exc)) from None
            return snapshot()

    async def initialize_owned_project(project_id, request, *, config_override=None,
                                       policy_override=None, recording_metadata=None, pipeline_owned=False,
                                       reuse_scene=False, policy_config_snapshot=None):
        nonlocal controller, active_project_id, connection_only_mode
        async with ownership_lock:
            if not pipeline_owned:
                require_manual_control()
            project = get_project(project_id)
            require_execution_available(project_id)
            state = availability(project)
            if not state["runnable"]:
                raise StateConflict(state["unavailable_reason"])
            active = controller.phase in ("initializing", "ready", "thinking", "executing", "pausing", "paused")
            if active and active_project_id != project_id:
                raise StateConflict("Stop the current collection before initializing a different project.")
            if controller.phase in ("initializing", "thinking", "executing", "pausing"):
                raise StateConflict("Pause or stop the current collection before initializing an environment.")
            if request.preserve_environment and active_project_id != project_id:
                raise StateConflict("Only this project's verified GPT reset can be continued.")
            config = RunConfig.model_validate({**project["run_config"], **(config_override or {}), "task_id": project["task_id"]})
            change_suite = project["suite"] != controller.settings.suite
            if change_suite and request.preserve_environment:
                raise StateConflict("Continuing a GPT reset cannot change the environment suite.")
            try:
                if change_suite:
                    if runner is not None:
                        raise StateConflict("The injected controller does not support changing its environment suite.")
                    await controller.close()
                    if any(not task.done() for task in controller._retired_requests):
                        retired_controllers.append(controller)
                    controller = Runner(replace(settings, suite=project["suite"], task_id=project["task_id"]))
                    app.state.runner = controller
                if not request.preserve_environment:
                    active_project_id = None
                chosen_policy = policy_config_snapshot if policy_config_snapshot is not None else project.get("policy_config")
                policy_config = PolicyConfig.model_validate(chosen_policy) if chosen_policy else None
                await controller.initialize(config, preserve_environment=request.preserve_environment,
                                            policy_config=policy_config, policy_override=policy_override,
                                            recording_metadata=recording_metadata, from_connected=reuse_scene)
                active_project_id = project_id
                connection_only_mode = False
                link_current(project_id)
                pure_bc_evaluation = bool(policy_override is not None
                    and (recording_metadata or {}).get("pipeline_stage") == "bc_eval"
                    and config.reset_mode == "environment")
                configure_baseline(project, allow_estimation=not pure_bc_evaluation)
            except StateConflict:
                raise
            except Exception as exc:
                raise HTTPException(503, controller._safe_error(exc)) from None
            return snapshot()

    @app.post("/api/projects/{project_id}/start")
    @app.post("/api/projects/{project_id}/resume")
    async def start_project(project_id: str):
        async with ownership_lock:
            require_manual_control()
            require_owner(project_id)
            if connection_only_mode and not (controller.phase == "paused" and controller.status()["run_started"]):
                await prepare_manual_run(project_id)
            return await begin()

    @app.post("/api/projects/{project_id}/step")
    async def step_project(project_id: str):
        async with ownership_lock:
            require_manual_control()
            require_owner(project_id)
            if controller.status()["run_started"] and controller.phase not in ("completed", "stopped", "connected"):
                raise StateConflict("Resume or stop the current run before starting one episode.")
            await prepare_manual_run(project_id, episodes=1)
            return await begin()

    async def prepare_manual_run(project_id, *, episodes=None):
        require_execution_available(project_id)
        project = get_project(project_id)
        config = RunConfig.model_validate({**project["run_config"], "task_id": project["task_id"],
            **({"episodes": episodes} if episodes is not None else {})})
        policy = PolicyConfig.model_validate(project["policy_config"]) if project.get("policy_config") else None
        await controller.initialize(config, policy_config=policy, from_connected=True)
        configure_baseline(project)
        link_current(project_id)

    @app.post("/api/projects/{project_id}/pause")
    async def pause_project(project_id: str):
        if pipeline.active:
            await pipeline.pause(project_id)
            return snapshot()
        async with ownership_lock:
            require_owner(project_id)
            await controller.pause()
            return snapshot()

    @app.post("/api/projects/{project_id}/stop")
    async def stop_project(project_id: str):
        if pipeline.active:
            await pipeline.stop(project_id)
            return snapshot()
        async with ownership_lock:
            require_owner(project_id)
            await controller.stop()
            return snapshot()

    @app.get("/api/media/{filename}")
    async def media(filename: str):
        try:
            path = projects.media_path(filename)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "Recorded media not found") from None
        return FileResponse(path)

    @app.get("/api/frame/{camera}")
    async def frame(camera: str, project_id: str | None = None, run_id: str | None = None,
                    observation_id: str | None = None):
        if (project_id is not None and project_id != active_project_id
                or run_id is not None and (controller.recorder is None or run_id != controller.recorder.run_id)
                or observation_id is not None and (controller.observation is None or observation_id != controller.observation.id)):
            raise HTTPException(404, "The requested project, run, or observation is no longer active")
        data = controller.frames.get(camera)
        if data is None:
            raise HTTPException(404, "Camera is not available. Initialize the environment first.")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/initialize")
    async def initialize(config: RunConfig):
        nonlocal active_project_id
        async with ownership_lock:
            require_manual_control()
            try:
                active_project_id = None
                await controller.initialize(config)
            except StateConflict:
                raise
            except Exception as exc:
                raise HTTPException(503, controller._safe_error(exc)) from None
            return snapshot()

    async def begin(single=False):
        require_manual_control()
        try:
            await controller.start(single=single)
        except StateConflict:
            raise
        except Exception as exc:
            raise HTTPException(400, controller._safe_error(exc)) from None
        return snapshot()

    @app.post("/api/start")
    async def start():
        async with ownership_lock:
            return await begin()

    @app.post("/api/continue")
    async def continue_run(config: RunConfig):
        async with ownership_lock:
            require_manual_control()
            project_config = get_project(active_project_id).get("policy_config") if active_project_id else None
            await controller.initialize(config, preserve_environment=True,
                policy_config=PolicyConfig.model_validate(project_config) if project_config else None)
            link_current(active_project_id)
            return snapshot()

    @app.post("/api/resume")
    async def resume():
        async with ownership_lock:
            return await begin()

    @app.post("/api/step")
    async def step():
        async with ownership_lock:
            require_manual_control()
            if controller.status()["run_started"] and controller.phase not in ("completed", "stopped", "connected"):
                raise StateConflict("Resume or stop the current run before starting one episode.")
            if active_project_id:
                await prepare_manual_run(active_project_id, episodes=1)
            else:
                config = controller.config.model_copy(update={"episodes": 1})
                await controller.initialize(config, from_connected=True)
            return await begin()

    @app.post("/api/pause")
    async def pause():
        if pipeline.active:
            await pipeline.pause(pipeline.runtime()["project_id"])
            return snapshot()
        await controller.pause()
        return snapshot()

    @app.post("/api/stop")
    async def stop():
        if pipeline.active:
            await pipeline.stop(pipeline.runtime()["project_id"])
            return snapshot()
        async with ownership_lock:
            await controller.stop()
            return snapshot()

    @app.post("/api/reset")
    async def reset():
        async with ownership_lock:
            require_manual_control()
            try:
                project_config = get_project(active_project_id).get("policy_config") if active_project_id else None
                await controller.initialize(controller.config,
                    policy_config=PolicyConfig.model_validate(project_config) if project_config else None)
                link_current(active_project_id)
            except Exception as exc:
                raise HTTPException(503, controller._safe_error(exc)) from None
            return snapshot()

    @app.get("/api/runs")
    async def runs():
        results = []
        if settings.data_dir.exists():
            for path in sorted(settings.data_dir.glob("*/manifest.json"), reverse=True)[:100]:
                try:
                    results.append(projects.read_run(path.parent.name))
                except (OSError, ValueError, KeyError):
                    continue
        return {"runs": results}

    async def prepare_pipeline_run(project_id, episodes, task_policy, metadata, budget):
        require_execution_available(project_id)
        project = get_project(project_id)
        is_evaluation = metadata.get("pipeline_stage") in ("bc_eval", "rl_eval")
        frozen = metadata.get("evaluation_config_snapshot", {}) if is_evaluation else {}
        run_config = frozen.get("run_config", project["run_config"])
        reset_mode = (metadata.get("reset_mode", "agent") if is_evaluation else
                      "free_space" if project["run_config"].get("reset_mode") == "free_space" else "agent")
        if reset_mode not in ("agent", "environment", "free_space"):
            raise StateConflict("Unsupported evaluation reset mode")
        environment_evaluation = is_evaluation and reset_mode == "environment"
        if environment_evaluation:
            await controller._ensure_adapter()
            context = await controller._env(controller.adapter.describe)
            if context.get("simulation") is not True:
                raise StateConflict("Environment reset evaluation requires a simulator; physical robots must use verified reset.")
        preserve = bool(not environment_evaluation and active_project_id == project_id and controller.stage == "reset"
                        and controller.observation and controller.observation.success
                        and controller.phase in ("completed", "paused")
                        and controller.config.task_id == project["task_id"]
                        and controller.config.seed == run_config["seed"]
                        and controller.config.reset_mode != "free_space")
        # Initial setup is allowed only before this project has performed any
        # movement. Every subsequent pipeline rollout must use a verified reset.
        same_scene = active_project_id == project_id and controller.observation is not None
        moved = controller.stats.get("total_steps", 0) > 0
        reuse_scene = same_scene and controller.can_start_new_run and not preserve and not environment_evaluation
        if same_scene and moved and not preserve and not reuse_scene and not environment_evaluation:
            raise StateConflict("검증된 GPT reset이 없어 자동 실행을 멈췄습니다. 환경을 수동 복구한 뒤 다시 시작하세요.")
        overrides = {**(run_config if is_evaluation else {}),
                     "episodes": episodes, "reset_mode": reset_mode, "budget_usd": budget}
        if task_policy is not None:
            overrides["max_decisions"] = run_config["max_steps"]
        await initialize_owned_project(project_id, InitializeProject(preserve_environment=preserve),
            config_override=overrides, policy_override=task_policy, recording_metadata=metadata,
            pipeline_owned=True, reuse_scene=reuse_scene, policy_config_snapshot=frozen.get("policy_config"))
        return controller

    pipeline = PipelineEngine(settings, projects, lambda: controller, prepare_pipeline_run)
    app.state.pipeline = pipeline
    video_router, video_jobs = create_video_router(projects, execution_management_reason)
    app.include_router(video_router)
    app.state.video_jobs = video_jobs
    robot_router, robot_connections = create_robot_connection_router(settings.data_dir, get_project)
    app.include_router(robot_router)
    app.state.robot_connections = robot_connections
    remote_service = RemoteService(settings, projects, pipeline, execution_busy=lambda:
        pipeline.active or pipeline._starting_project is not None
        or controller.phase in ("initializing", "thinking", "executing", "pausing")
        or bool(controller.phase in ("ready", "paused") and controller.status().get("run_started"))
        or bool(controller._inference and not controller._inference.done()))
    remote_api = RemoteAPI(remote_service, robot_connections, ownership_lock=ownership_lock)
    # The tunnel exposes policy RPC only, never the browser's operator controls.
    # Its private loopback listener starts lazily with an explicit SSH connect.
    rpc_listener = PolicyRPCListener(remote_service)
    robot_connections.forwarding_provider = rpc_listener.ensure
    remote_api.rpc_listener = rpc_listener
    app.state.remote_service = remote_service
    app.state.remote_api = remote_api
    app.state.remote_rpc_listener = rpc_listener
    app.include_router(remote_api.router)
    app.mount("/robot/v1", create_rpc_app(remote_service))

    @app.get("/api/projects/{project_id}/pipeline")
    async def pipeline_view(project_id: str):
        get_project(project_id)
        return await asyncio.to_thread(pipeline.view, project_id)

    @app.get("/api/projects/{project_id}/learning/bc/history")
    async def bc_training_history(project_id: str, model_id: str | None = None):
        try:
            history = await asyncio.to_thread(pipeline.bc_history, project_id, model_id)
        except (KeyError, FileNotFoundError, ValueError, OSError):
            raise HTTPException(404, "BC training history not found") from None
        return JSONResponse(history, headers={"Cache-Control": "no-store"})

    @app.post("/api/projects/{project_id}/pipeline/config")
    async def pipeline_config(project_id: str, data: dict):
        get_project(project_id)
        async with ownership_lock:
            try:
                return await asyncio.to_thread(pipeline.configure, project_id, data)
            except StateConflict:
                raise
            except (ValueError, ImportError) as exc:
                raise HTTPException(422, controller._safe_error(exc)) from None

    @app.post("/api/projects/{project_id}/pipeline/start")
    async def pipeline_start(project_id: str, data: PipelineStart):
        get_project(project_id)
        async with ownership_lock:
            if remote_service.active:
                raise StateConflict("원격 로봇 run을 중지한 뒤 학습·평가를 시작하세요.")
            try:
                return await pipeline.start(project_id, data)
            except StateConflict:
                raise
            except (ValueError, ImportError) as exc:
                raise HTTPException(422, controller._safe_error(exc)) from None

    @app.post("/api/projects/{project_id}/pipeline/pause")
    async def pipeline_pause(project_id: str):
        await pipeline.pause(project_id)
        return pipeline.view(project_id)

    @app.post("/api/projects/{project_id}/pipeline/resume")
    async def pipeline_resume(project_id: str):
        await pipeline.resume(project_id)
        return pipeline.view(project_id)

    @app.post("/api/projects/{project_id}/pipeline/stop")
    async def pipeline_stop(project_id: str):
        await pipeline.stop(project_id)
        return pipeline.view(project_id)

    @app.get("/api/projects/{project_id}/models/{model_id}/checkpoint")
    async def model_checkpoint(project_id: str, model_id: str):
        get_project(project_id)
        try:
            path = pipeline.checkpoint(project_id, model_id)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "Model checkpoint not found") from None
        return FileResponse(path, filename=f"{model_id}.pt", media_type="application/octet-stream")

    return app
