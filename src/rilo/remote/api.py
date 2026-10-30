"""Browser controls and authenticated robot RPC are separate interfaces."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from rilo.remote.launcher import LaunchConfig, RemoteLauncher
from rilo.remote.service import ACTIVE, RemoteError, RunStart
from rilo.remote.wire import (MAX_JSON_BYTES, MAX_PAYLOAD_BYTES, MEDIA_TYPE,
                              PROTOCOL, parse_json, unpack_message)


def create_rpc_app(service):
    """Mount at /robot/v1; no browser cookies or operator control endpoints."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(RemoteError)
    async def remote_error(request, error):
        return JSONResponse({"detail": str(error)}, status_code=error.status)

    @app.get("/health")
    async def health():
        return {"ok": True, "protocol": PROTOCOL, "transport": "http-npz",
                "execution": "robot-client", "control_hz": 10}

    async def body(request, media_type, limit):
        if request.headers.get("origin"):
            raise RemoteError("Robot RPC does not accept browser-origin requests", 403)
        if request.headers.get("x-rilo-protocol") != PROTOCOL:
            raise RemoteError("Incompatible X-Rilo-Protocol", 409)
        authorization = request.headers.get("authorization", "")
        if not authorization.startswith("Bearer "):
            raise RemoteError("Client token is required", 401)
        project_id = service.authenticate(authorization[7:])
        if request.headers.get("content-type", "").split(";", 1)[0] != media_type:
            raise RemoteError(f"Expected {media_type}", 415)
        declared = request.headers.get("content-length")
        if declared is not None and (not declared.isascii() or not declared.isdigit()):
            raise RemoteError("Invalid Content-Length", 400)
        if declared is not None and int(declared) > limit:
            raise RemoteError("RPC payload is too large", 413)
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > limit:
                raise RemoteError("RPC payload is too large", 413)
            data.extend(chunk)
        return project_id, bytes(data)

    @app.post("/{operation:path}")
    async def operation(operation: str, request: Request):
        json_operations = {"register", "poll", "ack"}
        packet_operations = {"observe", "episode/begin", "infer", "transition", "phase", "episode/end"}
        if operation not in json_operations | packet_operations:
            raise RemoteError("Unknown robot operation", 404)
        try:
            if operation in json_operations:
                project_id, payload = await body(request, "application/json", MAX_JSON_BYTES)
                result = getattr(service, operation)(project_id, parse_json(payload))
            else:
                project_id, payload = await body(request, MEDIA_TYPE, MAX_PAYLOAD_BYTES)
                meta, arrays = unpack_message(payload)
                result = await service.packet(project_id, operation, meta, arrays,
                    hashlib.sha256(payload).hexdigest(), request.headers.get("x-rilo-contract"))
            headers = {"Cache-Control": "no-store", "X-Rilo-Protocol": PROTOCOL}
            if isinstance(result, bytes):
                return Response(result, media_type=MEDIA_TYPE, headers=headers)
            return JSONResponse(result, headers=headers)
        except RemoteError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError):
            raise RemoteError("Invalid robot request or environment contract", 422) from None
        except Exception:
            # Request headers and bodies can contain secrets or camera images.
            logging.getLogger(__name__).error("Robot RPC %s failed", operation)
            raise RemoteError("Robot operation failed; motion must remain held", 500) from None

    return app


class RemoteAPI:
    """One environment button owns SSH, its reverse tunnel, and its client."""
    def __init__(self, service, connections, *, launcher=None, connect_timeout=25., ownership_lock=None):
        self.service, self.connections = service, connections
        self.launcher = launcher or RemoteLauncher(service.settings.data_dir, connections)
        self.connect_timeout = connect_timeout
        self.ownership_lock = ownership_lock or asyncio.Lock()
        self.jobs, self.errors = {}, {}
        self.disconnecting = set()
        self.closed = False
        self.rpc_listener = None
        connections.disconnect_guard = self.disconnect_reason
        self.router = self._router()

    def disconnect_reason(self, project_id):
        session = self.service.sessions.get(project_id)
        if session and session.run and session.run["status"] in ACTIVE:
            return "현재 run을 먼저 중지한 뒤 환경 연결을 해제하세요."
        owner = self.launcher.snapshot(project_id).get("active_launcher")
        if owner and owner["project_id"] == project_id or project_id in self.jobs:
            return "로봇 client가 연결 중입니다. 환경 연결 해제 버튼을 사용하세요."
        return None

    def snapshot(self, project_id):
        value = {**self.service.snapshot(project_id), **self.launcher.snapshot(project_id)}
        project = self.service.project(project_id)
        ssh = self.connections.snapshot(project_id)
        own = value.get("active_launcher") or {}
        ssh_owner = ssh.get("active_connection") or {}
        busy = bool(value["run"] and value["run"]["status"] in ACTIVE)
        pending = project_id in self.jobs
        stopping = project_id in self.disconnecting
        other = any(item.get("project_id") not in (None, project_id) for item in (own, ssh_owner))
        connect_reason = ("프로젝트를 복원한 뒤 연결하세요." if project.get("archived") or project.get("delete_pending") else
            "환경 연결을 해제하는 중입니다." if stopping else
            "현재 run을 먼저 중지하세요." if busy else
            "다른 프로젝트의 환경 연결을 먼저 해제하세요." if other else
            "SSH 접속 설정을 먼저 저장하세요." if not ssh.get("config") else
            "환경을 연결하는 중입니다." if pending else
            "이미 연결된 client를 먼저 해제하세요." if value["client"] and value["client"]["connected"] else
            "이미 연결된 환경입니다." if own else None)
        disconnect_reason = ("현재 run을 먼저 중지하세요." if busy else
            "다른 프로젝트의 환경은 여기서 해제할 수 없습니다." if other else None)
        configure_reason = ("프로젝트를 복원한 뒤 실행 설정을 변경하세요."
            if project.get("archived") or project.get("delete_pending") else
            "환경 연결을 해제한 뒤 실행 설정을 변경하세요." if pending or stopping or own or busy else None)
        value["capabilities"].update(environment_connect=connect_reason is None,
            environment_disconnect=disconnect_reason is None and bool(pending or own or ssh_owner or value["client"]),
            configure_launch=configure_reason is None)
        value["reasons"].update(environment_connect=connect_reason,
            environment_disconnect=disconnect_reason, configure_launch=configure_reason)
        if pending or stopping:
            value["capabilities"]["start"] = False
            value["reasons"]["start"] = "환경 연결·해제 처리가 끝난 뒤 실행하세요."
        if pending and value["launcher"]["phase"] == "idle":
            value["launcher"].update(phase="starting", ready=False)
        if stopping:
            value["launcher"].update(phase="stopping", ready=False)
        if self.errors.get(project_id):
            value["launcher"].update(phase="error", error=self.errors[project_id], error_code="environment")
        return value

    async def connect(self, project_id):
        value = self.snapshot(project_id)
        if self.closed:
            raise RemoteError("서버가 종료 중입니다.", 503)
        if project_id in self.disconnecting:
            raise RemoteError("환경 연결 해제가 완료된 뒤 다시 연결하세요.")
        if project_id in self.jobs or value["launcher"]["ready"]:
            return value
        if not value["capabilities"]["environment_connect"]:
            raise RemoteError(value["reasons"]["environment_connect"])
        config = value["launch_config"]
        if config["mode"] == "robot" and (not config["adapter"] or not config["env_config"]):
            raise RemoteError("로봇 adapter와 환경 설정 JSON 경로를 먼저 입력하세요.", 422)
        self.errors.pop(project_id, None)
        self.jobs[project_id] = asyncio.create_task(self._connect(project_id))
        return self.snapshot(project_id)

    async def _connect(self, project_id):
        try:
            await self.connections.connect(project_id)
            deadline = time.monotonic() + self.connect_timeout
            while True:
                state = self.connections.snapshot(project_id)["status"]
                if state["phase"] == "connected":
                    break
                if state["phase"] in ("error", "disconnected", "idle"):
                    raise RemoteError(state.get("error") or "SSH 연결을 완료하지 못했습니다.")
                if time.monotonic() > deadline:
                    raise RemoteError("SSH 연결 시간이 초과되었습니다.")
                await asyncio.sleep(.1)
            credentials = self.service.token(project_id)
            await self.launcher.start(project_id, client_config=credentials["client_config"])
        except asyncio.CancelledError:
            raise
        except (RemoteError, HTTPException) as error:
            self.errors[project_id] = str(getattr(error, "detail", error))[:1000]
        except Exception:
            self.errors[project_id] = "원격 client 실행을 준비하지 못했습니다. 연결 설정을 확인하세요."
        finally:
            if self.jobs.get(project_id) is asyncio.current_task():
                self.jobs.pop(project_id, None)

    async def disconnect(self, project_id):
        value = self.snapshot(project_id)
        if value["run"] and value["run"]["status"] in ACTIVE:
            raise RemoteError("현재 run을 먼저 중지한 뒤 환경 연결을 해제하세요.")
        owner = value.get("active_launcher") or {}
        ssh_owner = self.connections.snapshot(project_id).get("active_connection") or {}
        if any(item.get("project_id") not in (None, project_id) for item in (owner, ssh_owner)):
            raise RemoteError("다른 프로젝트의 환경은 여기서 해제할 수 없습니다.")
        if project_id in self.disconnecting:
            return value
        self.disconnecting.add(project_id)
        try:
            task = self.jobs.get(project_id)
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                # A task canceled before its first instruction never enters
                # _connect's finally block. Clear only this exact owner.
                if self.jobs.get(project_id) is task:
                    self.jobs.pop(project_id, None)
            await self.launcher.stop(project_id)
            if self.launcher.snapshot(project_id).get("active_launcher"):
                raise RemoteError("원격 client가 종료 중입니다. 종료를 확인한 뒤 다시 해제하세요.")
            self.service.revoke(project_id)
            await self.connections.disconnect(project_id)
            self.errors.pop(project_id, None)
        finally:
            self.disconnecting.discard(project_id)
        return self.snapshot(project_id)

    async def close(self):
        self.closed = True
        tasks = list(self.jobs.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.jobs.clear()
        await self.launcher.close()
        if self.rpc_listener is not None:
            await self.rpc_listener.close()
        await self.service.close()

    def _router(self):
        router = APIRouter(prefix="/api/projects/{project_id}/remote")

        @router.get("")
        async def snapshot(project_id: str):
            return self.snapshot(project_id)

        @router.post("/token")
        async def token(project_id: str):
            value = self.snapshot(project_id)
            if project_id in self.jobs or project_id in self.disconnecting or value["active_launcher"]:
                raise RemoteError("환경 연결을 해제한 뒤 client 토큰을 재발급하세요.")
            return self.service.token(project_id)

        @router.post("/launch-config")
        async def configure(project_id: str, data: LaunchConfig):
            value = self.snapshot(project_id)
            if not value["capabilities"]["configure_launch"]:
                raise RemoteError(value["reasons"]["configure_launch"])
            await self.launcher.configure(project_id, data.model_dump())
            return self.snapshot(project_id)

        @router.post("/environment/connect")
        async def connect(project_id: str):
            async with self.ownership_lock:
                return await self.connect(project_id)

        @router.post("/environment/disconnect")
        async def disconnect(project_id: str):
            return await self.disconnect(project_id)

        @router.post("/start")
        async def start(project_id: str, data: RunStart):
            async with self.ownership_lock:
                if project_id in self.jobs or project_id in self.disconnecting:
                    raise RemoteError("환경 연결이 완료된 뒤 실행하세요.")
                self.service.start(project_id, data.model_dump())
                return self.snapshot(project_id)

        @router.post("/{command}")
        async def control(project_id: str, command: str):
            if command not in ("pause", "resume", "stop"):
                raise RemoteError("Unknown remote control", 404)
            self.service.control(project_id, command)
            return self.snapshot(project_id)

        @router.get("/frame/{camera}")
        async def frame(project_id: str, camera: str):
            self.service.project(project_id)
            session = self.service.sessions.get(project_id)
            if session is None or camera not in session.frames:
                raise RemoteError("Camera frame not found", 404)
            return Response(session.frames[camera][1], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

        return router
