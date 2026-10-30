from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import stat
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ClientConfig:
    base_url: str
    project_token: str = field(repr=False)
    client_id: str = "robot-pc-1"
    journal_path: Path = Path("state/journal.sqlite3")
    control_hz: float = 10
    history_capacity: int = 64
    request_timeout_seconds: float = 180
    adapter_options: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        path = Path(path)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            metadata = os.fstat(handle.fileno())
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077
                    or metadata.st_uid != os.getuid() or metadata.st_size > 65536):
                raise ValueError("Client config must be your regular private file: chmod 600 config.json")
            raw = json.load(handle)
        return cls.from_dict(raw, base_dir=path.resolve().parent)

    @classmethod
    def from_dict(cls, raw, *, base_dir=None):
        if not isinstance(raw, dict) or set(raw) - set(cls.__dataclass_fields__):
            raise ValueError("Unsupported client configuration fields")
        base = raw.get("base_url", "http://127.0.0.1:45587/robot/v1").rstrip("/")
        url = urlsplit(base)
        if (url.scheme not in ("http", "https") or not url.hostname or url.username or url.password
                or url.query or url.fragment or not url.path.endswith("/robot/v1")):
            raise ValueError("base_url must be an HTTP(S) /robot/v1 endpoint without credentials or query")
        token = raw.get("project_token")
        if not isinstance(token, str) or not token or token.startswith("REPLACE_") or not token.isprintable():
            raise ValueError("Set the project token in the private configuration file")
        identifier = raw.get("client_id", "robot-pc-1")
        if not isinstance(identifier, str) or not identifier or len(identifier) > 120 or not identifier.isprintable():
            raise ValueError("client_id must be a nonempty printable name up to 120 characters")
        for name, default in (("control_hz", 10), ("request_timeout_seconds", 180)):
            value = raw.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        capacity = raw.get("history_capacity", 64)
        if type(capacity) is not int or not 2 <= capacity <= 1000:
            raise ValueError("history_capacity must be between 2 and 1000 control ticks")
        journal = Path(raw.get("journal_path", "state/journal.sqlite3"))
        if not journal.is_absolute():
            journal = Path(base_dir or Path.cwd()) / journal
        if not isinstance(raw.get("adapter_options", {}), dict):
            raise ValueError("adapter_options must be an object")
        return cls(**{**raw, "base_url": base, "journal_path": journal})
