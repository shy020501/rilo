"""Project-bound SSH passwords and short-lived OpenSSH askpass files.

The optional .env is a private, dedicated plaintext credential file, never the
application's root .env. Nothing in this module exports passwords to os.environ.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import sys
import tempfile

from pydantic import SecretStr


_KEY = re.compile(r"RILO_SSH_PASSWORD_[0-9a-f]{64}_[0-9a-f]{64}")


def validate_password(value):
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    if value is not None and isinstance(value, str):
        if len(value) > 4096 or any(character in value for character in "\x00\r\n"):
            raise ValueError("Password must be at most 4096 characters without NUL or line breaks")
    return value


def _project_prefix(project_id):
    return "RILO_SSH_PASSWORD_" + hashlib.sha256(project_id.encode()).hexdigest() + "_"


def _binding(config):
    return json.dumps([config["username"], config["host"], config["port"]], separators=(",", ":"))


def _key(project_id, config):
    return _project_prefix(project_id) + hashlib.sha256(_binding(config).encode()).hexdigest()


@dataclass(repr=False)
class CredentialPlan:
    memory: dict[str, SecretStr]
    disk: dict[str, SecretStr]


class CredentialStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / ".env"
        self._memory: dict[str, SecretStr] = {}
        self._disk: dict[str, SecretStr] = {}
        self.error = None
        try:
            if self.directory.is_symlink() or self.path.is_symlink():
                raise ValueError("Invalid credential path")
            if self.path.exists():
                descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                    metadata = os.fstat(handle.fileno())
                    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024
                            or metadata.st_mode & 0o077 or metadata.st_uid != os.getuid()):
                        raise ValueError("Invalid credential file permissions")
                    for line in handle:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        name, separator, encoded = line.partition("=")
                        if not separator or not _KEY.fullmatch(name) or name in self._disk:
                            raise ValueError("Invalid credential entry")
                        password = json.loads(encoded)
                        if not isinstance(password, str) or not password:
                            raise ValueError("Invalid credential value")
                        self._disk[name] = SecretStr(validate_password(password))
        except (OSError, ValueError, UnicodeError):
            self._disk.clear()
            self.error = "SSH 비밀번호 저장 파일이나 권한을 확인하세요. 기존 파일은 변경하지 않았습니다."

    def get(self, project_id, config):
        if not config or config.get("auth_method", "key") != "password":
            return None
        key = _key(project_id, config)
        return self._memory.get(key) or (self._disk.get(key) if config.get("remember_password") else None)

    def flags(self, project_id, config):
        return {"password_available": self.get(project_id, config) is not None,
                "password_saved": bool(config and _key(project_id, config) in self._disk)}

    def plan(self, project_id, previous, config, password=None):
        if self.error:
            raise ValueError(self.error)
        prefix = _project_prefix(project_id)
        memory = {key: value for key, value in self._memory.items() if not key.startswith(prefix)}
        disk = {key: value for key, value in self._disk.items() if not key.startswith(prefix)}
        secret = None
        if config.get("auth_method", "key") == "password":
            raw = validate_password(password)
            if isinstance(raw, str) and raw:
                secret = SecretStr(raw)
            elif previous and _binding(previous) == _binding(config):
                secret = self.get(project_id, previous)
            if secret is not None:
                key = _key(project_id, config)
                memory[key] = secret
                if config.get("remember_password"):
                    disk[key] = secret
        return CredentialPlan(memory, disk)

    def _write(self, disk):
        if self.directory.is_symlink() or self.path.is_symlink():
            raise OSError("Invalid credential path")
        if not disk:
            self.path.unlink(missing_ok=True)
            return
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        descriptor, temporary = tempfile.mkstemp(prefix=".credentials-", dir=self.directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write("# RILO SSH passwords: private plaintext, loaded only by RILO.\n")
                for key, secret in sorted(disk.items()):
                    handle.write(key + "=" + json.dumps(secret.get_secret_value(), ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def commit(self, plan):
        if plan.disk != self._disk:
            self._write(plan.disk)
        self._memory, self._disk = plan.memory, plan.disk

    def clear_memory(self):
        self._memory.clear()


_ASKPASS_CODE = '''import os, stat, sys
fd = os.open(os.environ["RILO_SSH_PASSWORD_FILE"], os.O_RDONLY | os.O_NOFOLLOW)
with os.fdopen(fd, "rb") as handle:
    metadata = os.fstat(handle.fileno())
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077 or metadata.st_size > 16384:
        raise SystemExit(1)
    value = handle.read(16385)
if any(character in value for character in (b"\\x00", b"\\r", b"\\n")):
    raise SystemExit(1)
sys.stdout.buffer.write(value + b"\\n")
'''


class AskpassSession:
    def __init__(self, password):
        value = validate_password(password)
        if not isinstance(value, str) or not value:
            raise ValueError("A password is required")
        self.directory = Path(tempfile.mkdtemp(prefix="rilo-ssh-auth-"))
        self.secret_file = self.directory / "password"
        self.helper_file = self.directory / "askpass"
        try:
            descriptor = os.open(self.secret_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(value.encode("utf-8"))
            self.helper_file.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " -c "
                                        + shlex.quote(_ASKPASS_CODE) + "\n", encoding="utf-8")
            self.helper_file.chmod(0o700)
        except BaseException:
            self.close()
            raise

    def environment(self):
        return {**os.environ, "SSH_ASKPASS": str(self.helper_file), "SSH_ASKPASS_REQUIRE": "force",
                "DISPLAY": "rilo-ssh:0", "RILO_SSH_PASSWORD_FILE": str(self.secret_file)}

    def close(self):
        if self.directory.exists():
            shutil.rmtree(self.directory)
