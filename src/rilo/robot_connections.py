"""Explicit SSH connectivity checks; never creates or controls a robot environment."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import tempfile
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from rilo.ssh_credentials import AskpassSession, CredentialStore, validate_password
from rilo.ssh_host_keys import HostKeyError, HostKeyManager


CAPABILITIES = {"ssh": True, "gym": False, "robot_control": False, "policy_inference": False}
PROBE_SCRIPT = r'''set -f
LC_ALL=C
export LC_ALL
os=$(uname -s) || exit 70
[ "$os" = Linux ] || { printf 'RILO1\tUNSUPPORTED\n'; exit 71; }
host=$(hostname) || exit 70
user=$(id -un) || exit 70
arch=$(uname -m) || exit 70
printf 'RILO1\tREADY\t%s\t%s\t%s\t%s\n' "$host" "$user" "$os" "$arch"
while IFS= read -r request; do
    case "$request" in
        'PING '*) nonce=${request#PING }
            case "$nonce" in ''|*[!0-9a-f]*) exit 72;; esac
            printf 'RILO1\tPONG\t%s\n' "$nonce";;
        QUIT) exit 0;;
        *) exit 72;;
    esac
done'''


class ConnectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    name: str = Field(min_length=1, max_length=120)
    username: str = Field(min_length=1, max_length=64)
    host: str = Field(min_length=1, max_length=253)
    port: int = Field(default=22, ge=1, le=65535)
    identity_file: str | None = Field(default=None, max_length=4096)
    auth_method: Literal["key", "password"] = "key"
    remember_password: bool = False
    password: SecretStr | None = Field(default=None, exclude=True, repr=False)

    @field_validator("password", mode="before")
    @classmethod
    def password_format(cls, value):
        return validate_password(value)

    @field_validator("name", "username", "host", "identity_file", mode="before")
    @classmethod
    def trim_fields(cls, value, info):
        if isinstance(value, str):
            if any(not character.isprintable() for character in value):
                raise ValueError("SSH fields must not contain control characters")
            value = value.strip()
            if info.field_name == "identity_file" and not value:
                return None
        return value

    @field_validator("username")
    @classmethod
    def valid_username(cls, value):
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\$?", value):
            raise ValueError("Enter only the SSH user name, without a host, password, or options")
        return value

    @field_validator("host")
    @classmethod
    def valid_host(cls, value):
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,252}", value):
            return value
        if re.fullmatch(r"[0-9a-fA-F:.]+", value):
            try:
                if ipaddress.ip_address(value).version == 6:
                    return value
            except ValueError:
                pass
        raise ValueError("Enter an IP address or hostname; enter the user and port in their separate fields")

    @field_validator("identity_file")
    @classmethod
    def valid_identity_file(cls, value):
        if value is not None and not (value.startswith("/") or value.startswith("~/")):
            raise ValueError("The identity file must be an absolute path or start with ~/ on the RILO computer")
        # OpenSSH expands percent tokens and ${VAR} itself; accept literal paths
        # so it uses the same file whose existence we check before connecting.
        if value is not None and ("%" in value or "${" in value):
            raise ValueError("The identity file must be a literal path without SSH or environment variable tokens")
        return value

    @property
    def destination(self):
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{self.username}@{host}:{self.port}"


class EmptyCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrustHostKeyCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    verification_id: str = Field(min_length=1, max_length=128)


class ConnectionFailure(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def ssh_arguments(executable, config, *, known_hosts_file=None, reverse_forward=None):
    # The sole remote command is a constant. User input is only a validated
    # argv entry; neither a local shell nor user shell text is used. Ignore
    # both user and system ssh_config so display names cannot resolve aliases.
    config = ConnectionConfig.model_validate(config)
    options = ["BatchMode=no" if config.auth_method == "password" else "BatchMode=yes",
        "StrictHostKeyChecking=yes", "UpdateHostKeys=no",
        "ConnectTimeout=8", "ConnectionAttempts=1", "ServerAliveInterval=5",
        "ServerAliveCountMax=2", "ForwardAgent=no", "ForwardX11=no", "ForwardX11Trusted=no",
        "Tunnel=no", "ClearAllForwardings=no" if reverse_forward else "ClearAllForwardings=yes",
        "ForkAfterAuthentication=no", "StdinNull=no",
        "PermitLocalCommand=no", "ControlMaster=no", "ControlPath=none",
        "RequestTTY=no", "RemoteCommand=none", "SessionType=default", "LogLevel=ERROR"]
    if config.auth_method == "password":
        options.extend(["PreferredAuthentications=password", "PasswordAuthentication=yes",
                        "PubkeyAuthentication=no", "KbdInteractiveAuthentication=no", "NumberOfPasswordPrompts=1"])
    if known_hosts_file is not None:
        path = str(known_hosts_file)
        if not path.isprintable() or "${" in path:
            raise ValueError("Invalid known-hosts path")
        quoted = path.replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"')
        options.extend([f'UserKnownHostsFile="{quoted}"', "GlobalKnownHostsFile=/dev/null"])
    identity = (["-i", str(Path(config.identity_file).expanduser()), "-o", "IdentitiesOnly=yes"]
                if config.auth_method == "key" and config.identity_file else [])
    forwarding = []
    if reverse_forward:
        remote_port, local_host, local_port = reverse_forward
        if any(type(port) is not int or not 1 <= port <= 65535 for port in (remote_port, local_port)):
            raise ValueError("Invalid policy tunnel ports")
        address = ipaddress.ip_address(local_host)
        target = f"[{address}]" if address.version == 6 else str(address)
        forwarding = ["-o", "ExitOnForwardFailure=yes", "-R", f"127.0.0.1:{remote_port}:{target}:{local_port}"]
    return [executable, "-F", "/dev/null", "-T", "-l", config.username, "-p", str(config.port),
            *(part for option in options for part in ("-o", option)), *identity, *forwarding,
            "--", config.host, "sh -c " + shlex.quote(PROBE_SCRIPT)]


def _empty_status(config=None):
    profile = ConnectionConfig.model_validate(config) if config is not None else None
    return {"phase": "disconnected", "name": profile.name if profile else None,
            "destination": profile.destination if profile else None, "connected_at": None,
            "last_heartbeat_at": None, "host_info": None, "error": None, "error_code": None}


class RobotConnectionManager:
    def __init__(self, data_dir, get_project, *, process_factory=None,
                 startup_timeout=15., heartbeat_interval=5., heartbeat_timeout=10., cleanup_timeout=3.):
        self.path = Path(data_dir).parent / "robot_connections.json"
        self.get_project = get_project
        self.process_factory = process_factory or asyncio.create_subprocess_exec
        self.startup_timeout, self.heartbeat_interval = startup_timeout, heartbeat_interval
        self.heartbeat_timeout, self.cleanup_timeout = heartbeat_timeout, cleanup_timeout
        self._configs, self._statuses, self._legacy_names = {}, {}, {}
        self._owner = self._process = self._task = None
        self._generation = 0
        self._task_started = False
        self._lock = asyncio.Lock()
        self._closed = False
        self._storage_error = None
        self.policy_forward = None
        self.forwarding_provider = None
        self.disconnect_guard = None
        self.credentials = CredentialStore(self.path.parent / "robot_ssh")
        self.host_keys = HostKeyManager(self.path.parent / "robot_ssh")
        try:
            if self.path.is_symlink():
                raise ValueError("Symbolic connection registry")
            if self.path.exists():
                if self.path.stat().st_size > 1024 * 1024:
                    raise ValueError("Oversized connection registry")
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                if saved.get("version") not in (1, 2) or not isinstance(saved.get("projects"), dict):
                    raise ValueError("Invalid connection registry")
                if saved["version"] == 1:
                    # Keep the old display names, never resolve ssh_config or
                    # guess a username/port for an incomplete legacy profile.
                    legacy = {key: value["ssh_alias"] for key, value in saved["projects"].items()}
                else:
                    self._configs = {key: ConnectionConfig.model_validate(value).model_dump()
                                     for key, value in saved["projects"].items()}
                    legacy = saved.get("legacy_names", {})
                if not isinstance(legacy, dict) or any(
                        not isinstance(value, str) or not value or len(value) > 319
                        or not value.isprintable() for value in legacy.values()):
                    raise ValueError("Invalid legacy connection names")
                self._legacy_names = {key: value for key, value in legacy.items() if key not in self._configs}
        except (OSError, ValueError, AttributeError, KeyError, TypeError):
            self._storage_error = "SSH 연결 설정 파일을 읽을 수 없습니다. 기존 파일은 변경하지 않았습니다."

    def _project(self, project_id, *, active=False):
        try:
            project = self.get_project(project_id)
        except KeyError:
            raise HTTPException(404, "Project does not exist") from None
        if project.get("backend") != "remote_robot":
            raise HTTPException(409, "SSH 연결은 remote_robot 프로젝트에서만 지원합니다.")
        if active and (project.get("archived") or project.get("delete_pending")):
            raise HTTPException(409, "보관되었거나 삭제 중인 프로젝트입니다. 복원한 뒤 연결 설정을 변경하거나 실행하세요.")
        return project

    def snapshot(self, project_id):
        self._project(project_id)
        config = self._configs.get(project_id)
        status = deepcopy(self._statuses.get(project_id, _empty_status(config)))
        if self._storage_error:
            status.update(phase="error", error=self._storage_error, error_code="internal")
        active = None
        if self._owner is not None:
            current = self._statuses[self._owner]
            active = {"project_id": self._owner, "name": current["name"],
                      "destination": current["destination"], "phase": current["phase"]}
        return {"project_id": project_id, "config": deepcopy(config), "status": status,
                "legacy_name": self._legacy_names.get(project_id),
                "active_connection": active, "capabilities": dict(CAPABILITIES),
                "credentials": self.credentials.flags(project_id, config)}

    def _write_configs(self, configs, *, legacy_names=None):
        if self.path.is_symlink():
            raise OSError("Connection registry must not be a symbolic link")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".robot-connections-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                legacy = self._legacy_names if legacy_names is None else legacy_names
                retained = {key: value for key, value in legacy.items() if key not in configs}
                json.dump({"version": 2, "projects": configs, "legacy_names": retained},
                          handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    async def configure(self, project_id, config):
        self._project(project_id, active=True)
        selected = ConnectionConfig.model_validate(config)
        value = selected.model_dump()
        async with self._lock:
            self._project(project_id, active=True)
            if self._storage_error:
                raise HTTPException(503, self._storage_error)
            if self._owner == project_id:
                raise HTTPException(409, "SSH 연결을 해제한 뒤 접속 설정을 변경하세요.")
            configs = {**self._configs, project_id: value}
            try:
                plan = self.credentials.plan(project_id, self._configs.get(project_id), value, selected.password)
                def persist():
                    self._write_configs(configs)
                    try:
                        self.credentials.commit(plan)
                    except Exception:
                        self._write_configs(self._configs)
                        raise
                await asyncio.to_thread(persist)
            except (OSError, ValueError):
                raise HTTPException(503, "SSH 연결 설정 또는 비밀번호 저장 파일·권한을 확인하세요.") from None
            self._configs = configs
            self._legacy_names.pop(project_id, None)
            self._statuses[project_id] = _empty_status(value)
        return self.snapshot(project_id)

    async def forget_project(self, project_id):
        """Forget inactive project settings and all of its password bindings."""
        self._project(project_id)
        async with self._lock:
            self._project(project_id)
            if self._storage_error:
                raise HTTPException(503, self._storage_error)
            if self._owner == project_id:
                raise HTTPException(409, "SSH 연결을 완전히 해제한 뒤 프로젝트 접속 정보를 삭제하세요.")
            configs = {key: value for key, value in self._configs.items() if key != project_id}
            legacy = {key: value for key, value in self._legacy_names.items() if key != project_id}
            changed = configs != self._configs or legacy != self._legacy_names
            try:
                plan = self.credentials.plan(project_id, self._configs.get(project_id), {"auth_method": "key"})
                # These bounded local registries commit without an await between
                # disk and memory updates; cancellation cannot split cleanup state.
                if changed:
                    self._write_configs(configs, legacy_names=legacy)
                try:
                    self.credentials.commit(plan)
                except Exception:
                    if changed:
                        self._write_configs(self._configs, legacy_names=self._legacy_names)
                    raise
            except (OSError, ValueError):
                raise HTTPException(503, "SSH 접속 정보 또는 비밀번호를 삭제하지 못했습니다. 저장 파일·권한을 확인하세요.") from None
            self._configs, self._legacy_names = configs, legacy
            self._statuses.pop(project_id, None)
        return {"project_id": project_id, "forgotten": True}

    async def connect(self, project_id):
        self._project(project_id, active=True)
        async with self._lock:
            self._project(project_id, active=True)
            if self._closed:
                raise HTTPException(503, "서버가 종료 중입니다.")
            if self._storage_error:
                raise HTTPException(503, self._storage_error)
            if self._owner is not None:
                if self._owner == project_id and self._statuses[project_id]["phase"] in ("connecting", "connected"):
                    return self.snapshot(project_id)
                raise HTTPException(409, "진행 중인 SSH 연결을 먼저 해제하세요.")
            config = self._configs.get(project_id)
            if config is None:
                raise HTTPException(409, "서버 별명, 사용자, IP·호스트와 포트를 먼저 저장하세요.")
            if config.get("auth_method", "key") == "password":
                if self.credentials.error:
                    raise HTTPException(503, self.credentials.error)
                if self.credentials.get(project_id, config) is None:
                    raise HTTPException(409, "SSH 비밀번호를 입력하고 설정을 저장한 뒤 연결하세요.")
            if self.forwarding_provider is not None:
                try:
                    self.policy_forward = await self.forwarding_provider()
                except (OSError, RuntimeError, asyncio.TimeoutError):
                    raise HTTPException(503, "정책 통신 전용 연결을 준비하지 못했습니다. RILO 서버 로그를 확인하세요.") from None
            self._project(project_id, active=True)
            self._generation += 1
            self._owner = project_id
            self._task_started = False
            self._statuses[project_id] = {**_empty_status(config), "phase": "connecting"}
            self._task = asyncio.create_task(self._run(project_id, config, self._generation),
                                             name="rilo-ssh-connectivity")
        return self.snapshot(project_id)

    def _host_key_config(self, project_id):
        self._project(project_id)
        if self._closed:
            raise HTTPException(503, "서버가 종료 중입니다.")
        if self._storage_error:
            raise HTTPException(503, self._storage_error)
        if self._owner is not None:
            raise HTTPException(409, "진행 중인 SSH 연결을 해제한 뒤 호스트 키를 확인하세요.")
        config = self._configs.get(project_id)
        if config is None:
            raise HTTPException(409, "접속 정보를 먼저 저장하세요.")
        return config

    async def inspect_host_key(self, project_id):
        # Hold the ownership lock so a connection or settings change cannot race
        # the preview. Scanning only reads public keys and never authenticates.
        async with self._lock:
            config = self._host_key_config(project_id)
            try:
                return await self.host_keys.inspect(project_id, config)
            except HostKeyError as exc:
                raise HTTPException(exc.status_code, str(exc)) from None
            except OSError:
                raise HTTPException(503, "호스트 키 정보를 읽을 수 없습니다. 파일 권한을 확인하세요.") from None

    async def trust_host_key(self, project_id, verification_id):
        async with self._lock:
            config = self._host_key_config(project_id)
            try:
                await asyncio.to_thread(self.host_keys.trust, project_id, config, verification_id)
            except HostKeyError as exc:
                raise HTTPException(exc.status_code, str(exc)) from None
            except OSError:
                raise HTTPException(503, "호스트 키를 저장하지 못했습니다. 파일 권한을 확인하세요.") from None
            # An old host-key error is resolved by the explicit confirmation;
            # establishing a new SSH connection still requires its own button.
            if self._statuses.get(project_id, {}).get("error_code") == "host_key":
                self._statuses[project_id] = _empty_status(config)
        return self.snapshot(project_id)

    async def disconnect(self, project_id):
        self._project(project_id)
        reason = self.disconnect_guard(project_id) if self.disconnect_guard else None
        if reason:
            raise HTTPException(409, reason)
        await self._disconnect_session(project_id)
        return self.snapshot(project_id)

    async def _disconnect_session(self, project_id):
        async with self._lock:
            if self._owner is not None and self._owner != project_id:
                raise HTTPException(409, "다른 프로젝트의 SSH 연결은 여기서 해제할 수 없습니다.")
            if self._owner is None:
                self._statuses[project_id] = _empty_status(self._configs.get(project_id))
                return
            task = self._task
            if task is None:
                # A previous kill/wait could not confirm exit. Retry cleanup of
                # that exact process; never replace it with another SSH child.
                self._generation += 1
                self._statuses[project_id]["phase"] = "disconnecting"
                task = self._task = asyncio.create_task(self._finalize(project_id,
                    self._configs[project_id], self._process, self._generation, None))
            if self._statuses[project_id]["phase"] != "disconnecting":
                self._statuses[project_id]["phase"] = "disconnecting"
                self._generation += 1
                if self._task_started:
                    task.cancel()
        # Do not hold the ownership lock while the child is reaped. If process
        # creation itself is delayed, retain ownership until its late child is
        # cleaned up instead of admitting another SSH connection.
        try:
            await asyncio.wait_for(asyncio.shield(task), self.cleanup_timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    @staticmethod
    async def _finish_owned(task):
        # Repeated cancellation must not orphan a child whose launch/reap is
        # already in progress. Only these owned cleanup operations use this.
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    raise

    async def close(self):
        self._closed = True
        await self.host_keys.close()
        owner = self._owner
        if owner is not None:
            await self._disconnect_session(owner)
            task = self._task
            if task is not None:
                # HTTP disconnect is bounded; application shutdown additionally
                # waits for a delayed local spawn to yield its child for reaping.
                await self._finish_owned(task)
        self.credentials.clear_memory()

    async def _cleanup(self, process):
        if process is None:
            return
        if process.stdin:
            try:
                process.stdin.close()
            except OSError:
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

    async def _stderr(self, stream, captured):
        while chunk := await stream.read(1024):
            captured.extend(chunk)
            del captured[:-8192]

    async def _line(self, stream):
        try:
            line = await stream.readline()
        except (ValueError, asyncio.LimitOverrunError):
            raise ConnectionFailure("protocol", "SSH 연결 확인 응답이 올바르지 않습니다.") from None
        if not line:
            raise ConnectionFailure("process_exit", "SSH 연결이 종료되었습니다.")
        if len(line) > 4096:
            raise ConnectionFailure("protocol", "SSH 연결 확인 응답이 너무 깁니다.")
        try:
            return line.decode("utf-8").rstrip("\r\n")
        except UnicodeError:
            raise ConnectionFailure("protocol", "SSH 연결 확인 응답을 읽을 수 없습니다.") from None

    async def _handshake(self, process):
        ignored = 0
        while True:
            line = await self._line(process.stdout)
            if line == "RILO1\tUNSUPPORTED":
                raise ConnectionFailure("unsupported_os", "이 연결 기능은 Linux 로봇 PC를 지원합니다.")
            if line.startswith("RILO1\t"):
                parts = line.split("\t")
                if (len(parts) != 6 or parts[1] != "READY" or parts[4] != "Linux"
                        or any(not value or len(value) > 256 or not value.isprintable() for value in parts[2:])):
                    raise ConnectionFailure("protocol", "SSH 연결 확인 응답이 올바르지 않습니다.")
                return dict(zip(("hostname", "user", "os", "arch"), parts[2:]))
            ignored += len(line)
            if ignored > 8192:
                raise ConnectionFailure("protocol", "SSH 연결 확인 메시지를 찾을 수 없습니다.")

    async def _ping(self, process):
        nonce = secrets.token_hex(16)
        process.stdin.write(f"PING {nonce}\n".encode())
        await process.stdin.drain()
        if await self._line(process.stdout) != f"RILO1\tPONG\t{nonce}":
            raise ConnectionFailure("protocol", "SSH heartbeat 응답이 일치하지 않습니다.")

    @staticmethod
    def _failure(exc, stderr):
        text = bytes(stderr).decode("utf-8", errors="replace").lower()
        if any(term in text for term in ("host key verification failed", "remote host identification has changed", "host key is known")):
            return "host_key", "SSH 호스트 키를 확인하지 못했습니다. 호스트 키 확인 버튼에서 서버 지문을 확인하고 등록하세요."
        if any(term in text for term in ("permission denied", "authentication failed", "too many authentication failures")):
            return "authentication", "SSH 인증에 실패했습니다. 입력한 사용자, 비밀번호·인증 키와 원격 계정의 접속 권한을 확인하세요."
        if any(term in text for term in ("could not resolve hostname", "connection refused", "no route to host", "network is unreachable")):
            return "unreachable", "SSH 대상 PC에 연결할 수 없습니다. 주소와 네트워크를 확인하세요."
        if "remote port forwarding failed" in text or "administratively prohibited" in text:
            return "policy_tunnel", "정책 통신 터널을 열지 못했습니다. 로봇 PC의 45587 포트와 SSH 포트 전달 허용 설정을 확인하세요."
        if isinstance(exc, asyncio.TimeoutError) or "connection timed out" in text:
            return "timeout", "SSH 연결 확인 시간이 초과되었습니다. 주소와 PC 상태를 확인하세요."
        if isinstance(exc, ConnectionFailure):
            return exc.code, str(exc)
        return "internal", "SSH 연결을 유지하지 못했습니다. PC 상태를 확인한 뒤 다시 연결하세요."

    async def _run(self, project_id, config, generation):
        process = stderr_task = None
        askpass = None
        stderr = bytearray()
        failure = None
        try:
            self._task_started = True
            if generation != self._generation or self._closed:
                return
            executable = shutil.which("ssh")
            if executable is None:
                raise ConnectionFailure("ssh_unavailable", "서버에서 OpenSSH 클라이언트를 찾을 수 없습니다.")
            identity_file = config.get("identity_file")
            if config.get("auth_method", "key") == "key" and identity_file and not Path(identity_file).expanduser().is_file():
                raise ConnectionFailure("identity_file", "입력한 인증 키 파일을 RILO 컴퓨터에서 찾을 수 없습니다. 키 경로를 확인하세요.")
            try:
                known_hosts_file = self.host_keys.known_hosts_file(config)
            except HostKeyError as exc:
                raise ConnectionFailure("host_key", str(exc)) from None
            process_options = {}
            if config.get("auth_method", "key") == "password":
                password = self.credentials.get(project_id, config)
                if password is None:
                    raise ConnectionFailure("authentication", "SSH 비밀번호를 입력하고 설정을 저장하세요.")
                askpass = AskpassSession(password)
                process_options["env"] = askpass.environment()
            spawn = asyncio.create_task(self.process_factory(*ssh_arguments(executable, config,
                known_hosts_file=known_hosts_file, reverse_forward=self.policy_forward),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, limit=8192, **process_options))
            deadline = asyncio.get_running_loop().time() + self.startup_timeout
            try:
                process = await asyncio.wait_for(asyncio.shield(spawn), self.startup_timeout)
            except asyncio.CancelledError:
                # A canceled launch may still create a child. Keep the launch
                # alive only to obtain and reap that exact child, never reconnect.
                process = await self._finish_owned(spawn)
                raise
            except asyncio.TimeoutError:
                if generation == self._generation:
                    self._statuses[project_id].update(phase="error", error_code="timeout",
                        error="SSH 프로세스 시작 시간이 초과되었습니다. 종료를 확인하고 있습니다.")
                process = await self._finish_owned(spawn)
                raise
            self._process = process
            stderr_task = asyncio.create_task(self._stderr(process.stderr, stderr))
            async def startup():
                info = await self._handshake(process)
                await self._ping(process)
                return info
            info = await asyncio.wait_for(startup(), max(.001, deadline - asyncio.get_running_loop().time()))
            if askpass is not None:
                askpass.close()
            while generation == self._generation and not self._closed:
                now = time.time()
                status = self._statuses[project_id]
                status.update(phase="connected", host_info=info,
                              connected_at=status["connected_at"] or now, last_heartbeat_at=now)
                await asyncio.sleep(self.heartbeat_interval)
                await asyncio.wait_for(self._ping(process), self.heartbeat_timeout)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            # Let an EOF from the same process drain its bounded diagnostic pipe.
            if stderr_task and process.returncode is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(stderr_task), .1)
                except Exception:
                    pass
            failure = self._failure(exc, stderr)
        finally:
            if askpass is not None:
                try:
                    askpass.close()
                except OSError:
                    failure = ("internal", "임시 SSH 인증 파일을 정리하지 못했습니다.")
            if stderr_task:
                stderr_task.cancel()
                await self._finish_owned(asyncio.gather(stderr_task, return_exceptions=True))
            await self._finish_owned(asyncio.create_task(self._finalize(
                project_id, config, process, generation, failure, owner_task=asyncio.current_task())))

    async def _finalize(self, project_id, config, process, generation, failure, *, owner_task=None):
        owner_task = owner_task or asyncio.current_task()
        reaped = True
        try:
            await self._finish_owned(asyncio.create_task(self._cleanup(process)))
        except (OSError, asyncio.TimeoutError):
            reaped = False
            failure = ("internal", "SSH 프로세스 종료를 확인하지 못했습니다. 연결 해제를 다시 시도하세요.")
        async with self._lock:
            if self._task is owner_task:
                status = self._statuses[project_id]
                if failure and (generation == self._generation or not reaped):
                    status.update(phase="error", error_code=failure[0], error=failure[1])
                else:
                    self._statuses[project_id] = _empty_status(config)
                self._task = None
                if reaped:
                    self._owner = self._process = None
                else:
                    self._process = process


async def parse_connection_request(request, model):
    """Never reflect request values or validation contexts containing secrets."""
    try:
        value = await request.json()
    except (ValueError, UnicodeError):
        raise HTTPException(422, "올바른 JSON 형식으로 SSH 설정을 입력하세요.") from None
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        errors = [{"loc": ["body", *item["loc"]], "msg": item["msg"], "type": item["type"]}
                  for item in exc.errors(include_input=False, include_context=False)]
        raise HTTPException(422, errors) from None


def create_robot_connection_router(data_dir, get_project):
    manager = RobotConnectionManager(data_dir, get_project)
    router = APIRouter(prefix="/api/projects/{project_id}/robot-connection")

    @router.get("")
    async def status(project_id: str):
        return manager.snapshot(project_id)

    @router.post("/config")
    async def configure(project_id: str, request: Request):
        data = await parse_connection_request(request, ConnectionConfig)
        return await manager.configure(project_id, data)

    @router.post("/connect")
    async def connect(project_id: str, request: Request):
        await parse_connection_request(request, EmptyCommand)
        return await manager.connect(project_id)

    @router.post("/disconnect")
    async def disconnect(project_id: str, request: Request):
        await parse_connection_request(request, EmptyCommand)
        return await manager.disconnect(project_id)

    @router.post("/host-key")
    async def inspect_host_key(project_id: str, request: Request):
        await parse_connection_request(request, EmptyCommand)
        return await manager.inspect_host_key(project_id)

    @router.post("/host-key/trust")
    async def trust_host_key(project_id: str, request: Request):
        data = await parse_connection_request(request, TrustHostKeyCommand)
        return await manager.trust_host_key(project_id, data.verification_id)

    return router, manager
