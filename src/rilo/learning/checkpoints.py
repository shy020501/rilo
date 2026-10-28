"""Atomic, versioned local learning checkpoints with immutable provenance."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import torch


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def save_dp(model, path, metadata=None):
    return atomic_torch_save({"format": "rilo.dp.v1", "spec": model.spec.model_dump(mode="json"),
        "config": model.config.model_dump(mode="json"), "state_dict": model.state_dict(),
        "metadata": metadata or {}}, path)


def load_dp(path, device="cpu"):
    from rilo.learning.data import ObservationSpec
    from rilo.learning.diffusion import DPConfig, DiffusionPolicy, device_for
    location = device_for(device)
    payload = torch.load(Path(path), map_location=location, weights_only=True)
    if payload.get("format") != "rilo.dp.v1":
        raise ValueError("Unsupported Diffusion Policy checkpoint format")
    model = DiffusionPolicy(ObservationSpec.model_validate(payload["spec"]), DPConfig.model_validate(payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(location).eval(), payload.get("metadata", {})
