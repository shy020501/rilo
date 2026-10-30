"""Explicit foreground robot-client launch over an already verified SSH connection.

The connection manager owns its tunnel. This manager owns only its separate SSH
child and never installs software, adopts a remote PID, or executes robot actions.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import tempfile
import time
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rilo.robot_connections import ConnectionFailure, RobotConnectionManager, ssh_arguments
from rilo.ssh_credentials import AskpassSession


class LaunchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    client_directory: str = Field(default="~/rilo_robot_client", min_length=1, max_length=4096)
    conda_env: str = Field(default="hilserl", min_length=1, max_length=128)
    python_executable: str | None = Field(default=None, max_length=4096)
    env_config: str | None = Field(default=None, max_length=4096)
    adapter: str | None = Field(default=None, max_length=256)
    mode: Literal["probe", "mock", "robot"] = "probe"

    @field_validator("client_directory", "conda_env", "python_executable", "env_config", "adapter", mode="before")
    @classmethod
    def literal_text(cls, value, info):
        if isinstance(value, str):
            if any(not ch.isprintable() for ch in value):
                raise ValueError("Launch fields must not contain control characters")
            value = value.strip()
            if not value and info.field_name not in ("client_directory", "conda_env"):
                return None
        return value

    @field_validator("client_directory", "python_executable")
    @classmethod
    def remote_path(cls, value):
        if value is not None and not (value.startswith("/") or value.startswith("~/")):
            raise ValueError("Remote paths must be absolute or start with ~/")
        return value

    @field_validator("conda_env")
    @classmethod
    def conda_name(cls, value):
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value):
            raise ValueError("Enter a Conda environment name, without options or a path")
        return value

    @field_validator("adapter")
    @classmethod
    def adapter_name(cls, value):
        if value is not None and not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", value, re.ASCII):
            raise ValueError("Adapter must be a Python module:factory")
        return value


# User paths are positional shell arguments, never interpolated into this script.
# Only a leading ~/ is expanded; shell expressions inside paths remain literal.
BOOTSTRAP_SCRIPT = r'''set -f
client_dir=$1
conda_env=$2
python_path=$3
case "$client_dir" in '~/'*) client_dir="$HOME/${client_dir#\~/}";; esac
case "$python_path" in '~/'*) python_path="$HOME/${python_path#\~/}";; esac
cd "$client_dir" || exit 72
if [ -n "$python_path" ]; then
    [ -x "$python_path" ] || { printf 'RILO_CLIENT_PYTHON_MISSING\n' >&2; exit 73; }
    exec "$python_path" -u -m rilo_robot_client --agent-stdin
fi
conda_path=$(command -v conda 2>/dev/null || true)
if [ ! -x "$conda_path" ]; then
    conda_path=''
    for candidate in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
        if [ -x "$candidate" ]; then conda_path=$candidate; break; fi
    done
fi
[ -n "$conda_path" ] || { printf 'RILO_CLIENT_CONDA_MISSING\n' >&2; exit 74; }
exec "$conda_path" run --no-capture-output -n "$conda_env" python -u -m rilo_robot_client --agent-stdin'''


def bootstrap_command(config):
    config = LaunchConfig.model_validate(config)
    return shlex.join(["sh", "-c", BOOTSTRAP_SCRIPT, "rilo-client-launch",
                       config.client_directory, config.conda_env, config.python_executable or ""])


def _status():
    return {"phase": "idle", "ready": False, "started_at": None, "last_event_at": None,
            "error": None, "error_code": None, "logs": []}


class RemoteLauncher:
    def __init__(self, data_dir, connection_manager, *, process_factory=None,
                 startup_timeout=45., cleanup_timeout=3.):
        self.path = Path(data_dir).parent / "robot_client_launchers.json"
        self.connections = connection_manager
        self.process_factory = process_factory or asyncio.create_subprocess_exec
        self.startup_timeout, self.cleanup_timeout = startup_timeout, cleanup_timeout
        self._configs, self._statuses = {}, {}
        self._owner = self._process = self._task = None
        self._generation = 0
        self._task_started = False
        self._closed = False
        self._lock = asyncio.Lock()
        self._storage_error = None
        try:
            if self.path.is_symlink():
                raise ValueError("Symbolic launcher registry")
            if self.path.exists():
                if self.path.stat().st_size > 1024 * 1024:
                    raise ValueError("Oversized launcher registry")
                value = json.loads(self.path.read_text(encoding="utf-8"))
                if value.get("version") != 1 or not isinstance(value.get("projects"), dict):
                    raise ValueError("Invalid launcher registry")
                self._configs = {key: LaunchConfig.model_validate(config).model_dump()
                                 for key, config in value["projects"].items()}
        except (OSError, ValueError, TypeError, AttributeError):
            self._storage_error = "로봇 클라이언트 실행 설정을 읽을 수 없습니다."

    def snapshot(self, project_id):
        self.connections.snapshot(project_id)  # Reuse project/backend authorization.
        state = deepcopy(self._statuses.get(project_id, _status()))
        if self._storage_error:
            state.update(phase="error", error=self._storage_error, error_code="storage")
        return {"launch_config": deepcopy(self._configs.get(project_id, LaunchConfig().model_dump())),
                "launcher": state, "active_launcher": None if self._owner is None else {
                    "project_id": self._owner, "phase": self._statuses[self._owner]["phase"]}}

    def _persist(self, configs):
        if self.path.is_symlink():
            raise OSError("Symbolic launcher registry")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".robot-client-launchers-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "projects": configs}, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    async def configure(self, project_id, config):
        self.connections._project(project_id, active=True)
        value = LaunchConfig.model_validate(config).model_dump()
        async with self._lock:
            self.connections._project(project_id, active=True)
            if self._closed or self._storage_error:
                raise HTTPException(503, self._storage_error or "서버가 종료 중입니다.")
            if self._owner == project_id:
                raise HTTPException(409, "환경 연결을 해제한 뒤 실행 설정을 변경하세요.")
            configs = {**self._configs, project_id: value}
            try:
                # Small bounded local file; no await between commit and in-memory
                # update, so canceled HTTP requests cannot leave split state.
                self._persist(configs)
            except OSError:
                raise HTTPException(503, "클라이언트 실행 설정을 저장하지 못했습니다.") from None
            self._configs = configs
        return self.snapshot(project_id)

    async def forget_project(self, project_id):
        """Remove saved launch settings only after this client's exit is confirmed."""
        self.connections._project(project_id)
        async with self._lock:
            self.connections._project(project_id)
            if self._storage_error:
                raise HTTPException(503, self._storage_error)
            if self._owner == project_id:
                raise HTTPException(409, "로봇 client를 완전히 종료한 뒤 프로젝트 실행 설정을 삭제하세요.")
            configs = {key: value for key, value in self._configs.items() if key != project_id}
            try:
                if configs != self._configs:
                    self._persist(configs)
            except OSError:
                raise HTTPException(503, "클라이언트 실행 설정을 삭제하지 못했습니다.") from None
            self._configs = configs
            self._statuses.pop(project_id, None)
        return {"project_id": project_id, "forgotten": True}

    async def start(self, project_id, launch_config=None, client_config=None):
        self.connections._project(project_id, active=True)
        connection = self.connections.snapshot(project_id)
        config = LaunchConfig.model_validate(launch_config or self._configs.get(project_id, {})).model_dump()
        if not isinstance(client_config, dict):
            raise HTTPException(422, "클라이언트 연결 구성이 필요합니다.")
        # Only the caller's in-memory envelope contains credentials. Neither the
        # launcher registry, argv, nor subprocess environment receives this JSON.
        try:
            envelope = json.dumps({"client_config": client_config, "mode": config["mode"],
                "env_config": config["env_config"], "adapter": config["adapter"]},
                ensure_ascii=False, allow_nan=False).encode() + b"\n"
        except (TypeError, ValueError):
            raise HTTPException(422, "클라이언트 연결 구성이 올바르지 않습니다.") from None
        if len(envelope) > 65536:
            raise HTTPException(422, "클라이언트 연결 구성이 너무 큽니다.")
        async with self._lock:
            self.connections._project(project_id, active=True)
            if self._closed or self._storage_error:
                raise HTTPException(503, self._storage_error or "서버가 종료 중입니다.")
            if self._owner is not None:
                if self._owner == project_id and self._statuses[project_id]["phase"] in ("starting", "running"):
                    return self.snapshot(project_id)
                raise HTTPException(409, "실행 중인 로봇 클라이언트를 먼저 종료하세요.")
            connection = self.connections.snapshot(project_id)
            owner = connection.get("active_connection") or {}
            if connection["status"]["phase"] != "connected" or owner.get("project_id") != project_id:
                raise HTTPException(409, "이 프로젝트의 SSH 연결을 먼저 완료하세요.")
            self._generation += 1
            self._owner, self._task_started = project_id, False
            self._statuses[project_id] = {**_status(), "phase": "starting", "started_at": time.time()}
            self._task = asyncio.create_task(self._run(project_id, config, connection["config"],
                envelope, deepcopy(client_config), self._generation))
        return self.snapshot(project_id)

    async def stop(self, project_id):
        self.connections.snapshot(project_id)
        await self._stop(project_id)
        return self.snapshot(project_id)

    async def _stop(self, project_id):
        async with self._lock:
            if self._owner is not None and self._owner != project_id:
                raise HTTPException(409, "다른 프로젝트의 클라이언트는 여기서 종료할 수 없습니다.")
            if self._owner is None:
                return
            task = self._task
            if task is None:
                self._generation += 1
                self._statuses[project_id].update(phase="stopping", ready=False)
                task = self._task = asyncio.create_task(self._finalize(project_id, self._process,
                    self._generation, None))
            if self._statuses[project_id]["phase"] != "stopping":
                self._generation += 1
                self._statuses[project_id].update(phase="stopping", ready=False)
                if self._task_started:
                    task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), self.cleanup_timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    async def close(self):
        self._closed = True
        if self._owner is not None:
            await self._stop(self._owner)
            if self._task is not None:
                await RobotConnectionManager._finish_owned(self._task)

    def _log(self, project_id, stream, message, secrets):
        for secret in secrets:
            if secret:
                message = message.replace(secret, "[REDACTED]")
        message = "".join(ch if ch.isprintable() else " " for ch in message)[:4096]
        state = self._statuses[project_id]
        now = time.time()
        state["last_event_at"] = now
        state["logs"].append({"stream": stream, "message": message, "time": now})
        del state["logs"][:-100]

    async def _read_stream(self, project_id, stream, name, ready, secrets):
        # Bound incomplete lines too, and discard their text instead of exposing
        # a credential split across chunks or truncated in the middle of a line.
        buffer = bytearray()
        oversized = False
        while chunk := await stream.read(1024):
            for part_index, part in enumerate(chunk.split(b"\n")):
                if part_index:
                    if oversized:
                        self._log(project_id, name, "[oversized output omitted]", secrets)
                    else:
                        line = bytes(buffer).decode("utf-8", errors="replace").rstrip("\r")
                        if name == "stdout" and line == "RILO_CLIENT_READY":
                            ready.set()
                        elif line:
                            self._log(project_id, name, line, secrets)
                    buffer.clear()
                    oversized = False
                if not oversized:
                    buffer.extend(part)
                    if len(buffer) > 4096:
                        buffer.clear()
                        oversized = True
        if buffer and not oversized:
            self._log(project_id, name, bytes(buffer).decode("utf-8", errors="replace"), secrets)

    async def _cleanup(self, process):
        if process is None:
            return
        if process.stdin:
            try:
                process.stdin.close()  # Agent observes EOF and holds before exit.
            except OSError:
                pass
        try:
            await asyncio.wait_for(process.wait(), .5)
            return
        except asyncio.TimeoutError:
            pass
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), 1.)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await asyncio.wait_for(process.wait(), 1.)

    async def _run(self, project_id, config, ssh_config, envelope, client_config, generation):
        process = askpass = None
        readers = []
        failure = None
        secrets = [value for key, value in client_config.items()
                   if isinstance(value, str) and any(word in key.lower() for word in ("token", "password", "secret"))]
        try:
            self._task_started = True
            if generation != self._generation or self._closed:
                return
            executable = shutil.which("ssh")
            if executable is None:
                raise ConnectionFailure("ssh_unavailable", "OpenSSH 클라이언트를 찾을 수 없습니다.")
            identity = ssh_config.get("identity_file")
            if ssh_config.get("auth_method", "key") == "key" and identity and not Path(identity).expanduser().is_file():
                raise ConnectionFailure("identity_file", "입력한 인증 키 파일을 찾을 수 없습니다.")
            options = {}
            if ssh_config.get("auth_method", "key") == "password":
                password = self.connections.credentials.get(project_id, ssh_config)
                if password is None:
                    raise ConnectionFailure("authentication", "SSH 비밀번호를 입력하고 저장하세요.")
                secrets.append(password.get_secret_value())
                askpass = AskpassSession(password)
                options["env"] = askpass.environment()
            secrets = sorted(set(secrets + [json.dumps(value)[1:-1] for value in secrets]), key=len, reverse=True)
            args = ssh_arguments(executable, ssh_config,
                known_hosts_file=self.connections.host_keys.known_hosts_file(ssh_config))
            args[-1] = bootstrap_command(config)
            deadline = asyncio.get_running_loop().time() + self.startup_timeout
            spawn = asyncio.create_task(self.process_factory(*args, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=8192, **options))
            try:
                process = await asyncio.wait_for(asyncio.shield(spawn), self.startup_timeout)
            except asyncio.TimeoutError:
                if generation == self._generation:
                    self._statuses[project_id].update(phase="error", error_code="timeout",
                        error="SSH 클라이언트 프로세스 시작 시간이 초과되어 종료를 확인하고 있습니다.")
                process = await RobotConnectionManager._finish_owned(spawn)
                raise
            except asyncio.CancelledError:
                process = await RobotConnectionManager._finish_owned(spawn)
                raise
            self._process = process
            if generation != self._generation or self._closed:
                return
            ready = asyncio.Event()
            readers = [asyncio.create_task(self._read_stream(project_id, stream, name, ready, secrets))
                       for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr"))]
            process.stdin.write(envelope)
            await asyncio.wait_for(process.stdin.drain(), max(.001, deadline - asyncio.get_running_loop().time()))
            async def startup():
                while not ready.is_set():
                    if process.returncode is not None or readers[0].done():
                        raise ConnectionFailure("process_exit", "클라이언트가 준비되기 전에 종료되었습니다. 실행 경로와 환경을 확인하세요.")
                    await asyncio.sleep(.01)
            await asyncio.wait_for(startup(), max(.001, deadline - asyncio.get_running_loop().time()))
            if askpass is not None:
                askpass.close()
            if generation != self._generation or self._closed:
                return
            self._statuses[project_id].update(phase="running", ready=True, last_event_at=time.time())
            while generation == self._generation and not self._closed:
                if process.returncode is not None or readers[0].done():
                    raise ConnectionFailure("process_exit", "로봇 클라이언트가 종료되었습니다.")
                connection = self.connections.snapshot(project_id)
                if connection["status"]["phase"] != "connected":
                    raise ConnectionFailure("ssh_disconnected", "SSH 연결이 종료되어 클라이언트를 정리했습니다.")
                await asyncio.sleep(.2)
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            failure = ("timeout", "로봇 클라이언트 준비 시간이 초과되었습니다. 실행 경로와 환경을 확인하세요.")
        except ConnectionFailure as exc:
            failure = (exc.code, str(exc))
        except Exception:
            failure = ("launch_failed", "로봇 클라이언트를 실행하지 못했습니다. SSH 인증과 실행 설정을 확인하세요.")
        finally:
            if askpass is not None:
                try:
                    askpass.close()
                except OSError:
                    failure = ("cleanup", "임시 SSH 인증 파일을 정리하지 못했습니다.")
            for task in readers:
                task.cancel()
            if readers:
                await RobotConnectionManager._finish_owned(asyncio.gather(*readers, return_exceptions=True))
            await RobotConnectionManager._finish_owned(asyncio.create_task(self._finalize(project_id,
                process, generation, failure, owner_task=asyncio.current_task())))

    async def _finalize(self, project_id, process, generation, failure, *, owner_task=None):
        owner_task = owner_task or asyncio.current_task()
        reaped = True
        try:
            await RobotConnectionManager._finish_owned(asyncio.create_task(self._cleanup(process)))
        except (OSError, asyncio.TimeoutError):
            reaped = False
            failure = ("cleanup", "클라이언트 프로세스 종료를 확인하지 못했습니다. 환경 연결 해제를 다시 시도하세요.")
        async with self._lock:
            if self._task is owner_task:
                state = self._statuses[project_id]
                state.update(ready=False, last_event_at=time.time())
                if failure and (generation == self._generation or not reaped):
                    state.update(phase="error", error_code=failure[0], error=failure[1])
                else:
                    state.update(phase="idle", error=None, error_code=None)
                self._task = None
                if reaped:
                    self._owner = self._process = None
                else:
                    self._process = process
