"""Inspect SSH public host keys and pin only keys explicitly confirmed in the UI."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import os
from pathlib import Path
import secrets
import shutil
import struct
import tempfile
import time


MAX_OUTPUT = 128 * 1024
ALGORITHMS = {"ssh-ed25519", "ssh-rsa", "ecdsa-sha2-nistp256", "ecdsa-sha2-nistp384", "ecdsa-sha2-nistp521"}


class HostKeyError(Exception):
    def __init__(self, message, *, status_code=409):
        super().__init__(message)
        self.status_code = status_code


def _key(algorithm, encoded):
    if algorithm not in ALGORITHMS or len(encoded) > 16384:
        raise ValueError("Unsupported host key")
    blob = base64.b64decode(encoded, validate=True)
    offset = 0

    def field():
        nonlocal offset
        if offset + 4 > len(blob):
            raise ValueError("Truncated key")
        length = struct.unpack_from(">I", blob, offset)[0]
        offset += 4
        if length > len(blob) - offset:
            raise ValueError("Truncated key field")
        result = blob[offset:offset + length]
        offset += length
        return result

    if field().decode("ascii") != algorithm:
        raise ValueError("Mismatched key algorithm")
    if algorithm == "ssh-ed25519":
        if len(field()) != 32:
            raise ValueError("Invalid Ed25519 key")
    elif algorithm == "ssh-rsa":
        if not field() or not field():
            raise ValueError("Invalid RSA key")
    elif field().decode("ascii") != algorithm.removeprefix("ecdsa-sha2-") or not field():
        raise ValueError("Invalid ECDSA key")
    if offset != len(blob):
        raise ValueError("Trailing key data")
    return {"algorithm": algorithm, "encoded": base64.b64encode(blob).decode("ascii"),
            "fingerprint": "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode("ascii").rstrip("=")}


def _endpoint(config):
    host = config["host"].lower()
    return host if config["port"] == 22 else f"[{host}]:{config['port']}"


def _binding(project_id, config):
    return project_id, config["username"], config["host"].lower(), config["port"]


def _matches(names, endpoint):
    for name in names.split(","):
        if name == endpoint:
            return True
        if name.startswith("|1|"):
            try:
                _, version, salt, digest = name.split("|")
                expected = hmac.new(base64.b64decode(salt, validate=True), endpoint.encode(), hashlib.sha1).digest()
                if hmac.compare_digest(expected, base64.b64decode(digest, validate=True)):
                    return True
            except (ValueError, TypeError):
                pass
    return False


class HostKeyManager:
    def __init__(self, directory, *, process_factory=None, known_hosts_paths=None,
                 timeout=10., verification_ttl=180.):
        self.directory = Path(directory)
        self.process_factory = process_factory or asyncio.create_subprocess_exec
        self.known_hosts_paths = tuple(known_hosts_paths) if known_hosts_paths is not None else (
            Path.home() / ".ssh/known_hosts", Path.home() / ".ssh/known_hosts2",
            Path("/etc/ssh/ssh_known_hosts"), Path("/etc/ssh/ssh_known_hosts2"))
        self.timeout, self.verification_ttl = timeout, verification_ttl
        self._pending, self._tasks = {}, set()
        self._closed = False

    def _path(self, config):
        return self.directory / "known_hosts" / hashlib.sha256(_endpoint(config).encode()).hexdigest()

    def known_hosts_file(self, config):
        path = self._path(config)
        if self.directory.is_symlink() or path.parent.is_symlink() or path.is_symlink():
            raise HostKeyError("호스트 키 저장 경로가 심볼릭 링크입니다. 연결 설정을 확인하세요.")
        return path if path.is_file() else None

    def _existing(self, config):
        own = self.known_hosts_file(config)
        paths = (own,) if own else self.known_hosts_paths
        found = set()
        for path in paths:
            try:
                with Path(path).open("rb") as handle:
                    content = handle.read(2 * 1024 * 1024 + 1)
                if len(content) > 2 * 1024 * 1024:
                    continue
                for line in content.decode("utf-8", errors="replace").splitlines():
                    parts = line.split()
                    if len(parts) < 3 or parts[0].startswith(("#", "@")) or not _matches(parts[0], _endpoint(config)):
                        continue
                    try:
                        key = _key(parts[1], parts[2])
                        found.add((key["algorithm"], key["fingerprint"]))
                    except (ValueError, UnicodeError):
                        continue
            except (FileNotFoundError, PermissionError, IsADirectoryError):
                continue
        return found

    @staticmethod
    async def _finish(task):
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    raise

    @staticmethod
    async def _cleanup(process):
        if process is None:
            return
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

    @staticmethod
    async def _read(stream):
        result = bytearray()
        while chunk := await stream.read(4096):
            result.extend(chunk)
            if len(result) > MAX_OUTPUT:
                raise HostKeyError("호스트 키 응답이 너무 큽니다.", status_code=502)
        return bytes(result)

    async def _scan(self, config):
        executable = shutil.which("ssh-keyscan")
        if not executable:
            raise HostKeyError("RILO 컴퓨터에서 ssh-keyscan 명령을 찾을 수 없습니다.", status_code=503)
        process = None
        readers = []
        spawn = asyncio.create_task(self.process_factory(executable, "-T", "5", "-p", str(config["port"]),
            "-t", "ed25519,ecdsa,rsa", "--", config["host"], stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=MAX_OUTPUT))
        deadline = asyncio.get_running_loop().time() + self.timeout
        try:
            try:
                process = await asyncio.wait_for(asyncio.shield(spawn), self.timeout)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                process = await self._finish(spawn)
                raise
            readers = [asyncio.create_task(self._read(process.stdout)), asyncio.create_task(self._read(process.stderr)),
                       asyncio.create_task(process.wait())]
            stdout, _, code = await asyncio.wait_for(asyncio.gather(*readers),
                max(.001, deadline - asyncio.get_running_loop().time()))
            if code not in (0, 1) or not stdout.strip():
                raise HostKeyError("서버의 호스트 키를 가져오지 못했습니다. IP·포트와 네트워크를 확인하세요.", status_code=502)
            return stdout
        except asyncio.TimeoutError:
            raise HostKeyError("호스트 키 확인 시간이 초과되었습니다.", status_code=504) from None
        except OSError:
            raise HostKeyError("호스트 키 확인 명령을 실행하지 못했습니다.", status_code=503) from None
        finally:
            for task in readers:
                task.cancel()
            if readers:
                await self._finish(asyncio.gather(*readers, return_exceptions=True))
            await self._finish(asyncio.create_task(self._cleanup(process)))

    async def inspect(self, project_id, config):
        if self._closed:
            raise HostKeyError("서버가 종료 중입니다.", status_code=503)
        task = asyncio.current_task()
        self._tasks.add(task)
        try:
            output = await self._scan(config)
            keys = {}
            try:
                for line in output.decode("ascii").splitlines():
                    if not line.strip() or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        raise ValueError("Malformed key scan")
                    key = _key(parts[1], parts[2])
                    if key["algorithm"] in keys and key != keys[key["algorithm"]]:
                        raise ValueError("Conflicting host keys")
                    keys[key["algorithm"]] = key
                if not keys:
                    raise ValueError("No host keys")
            except (ValueError, UnicodeError):
                raise HostKeyError("서버의 호스트 키 응답이 올바르지 않습니다.", status_code=502) from None
            now = time.monotonic()
            self._pending = {token: value for token, value in self._pending.items()
                             if value["expires"] > now and value["binding"][0] != project_id}
            token = secrets.token_urlsafe(24)
            existing = self._existing(config)
            current = {(key["algorithm"], key["fingerprint"]) for key in keys.values()}
            self._pending[token] = {"binding": _binding(project_id, config), "expires": now + self.verification_ttl,
                                    "keys": list(keys.values())}
            host = f"[{config['host']}]" if ":" in config["host"] else config["host"]
            return {"verification_id": token, "destination": f"{config['username']}@{host}:{config['port']}",
                    "keys": [{"algorithm": key["algorithm"], "fingerprint": key["fingerprint"]} for key in keys.values()],
                    "changed": bool(existing and existing != current)}
        finally:
            self._tasks.discard(task)

    def trust(self, project_id, config, verification_id):
        pending = self._pending.get(verification_id)
        if (self._closed or not pending or pending["expires"] <= time.monotonic()
                or pending["binding"] != _binding(project_id, config)):
            raise HostKeyError("호스트 키 확인이 만료되었거나 접속 정보가 바뀌었습니다. 다시 확인하세요.")
        path = self._path(config)
        self.known_hosts_file(config)  # Reject symbolic storage before creating anything.
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.directory.chmod(0o700)
        path.parent.mkdir(mode=0o700, exist_ok=True)
        path.parent.chmod(0o700)
        descriptor, temporary = tempfile.mkstemp(prefix=".host-key-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                for key in pending["keys"]:
                    handle.write(f"{_endpoint(config)} {key['algorithm']} {key['encoded']}\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self._pending.pop(verification_id, None)

    async def close(self):
        self._closed = True
        tasks = [task for task in self._tasks if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pending.clear()
