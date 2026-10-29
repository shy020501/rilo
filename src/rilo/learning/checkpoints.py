"""Atomic, versioned local learning checkpoints with immutable provenance."""
from __future__ import annotations

import hashlib
import json
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


def atomic_json_save(payload, path):
    """Publish a complete checkpoint index only after its files are durable."""
    path = Path(path)
    content = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
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


def save_dp_training(model, ema, optimizer, path, *, step, metadata=None, loader_state=None):
    """Save an inference-compatible EMA policy and completed optimizer state.

    This is a training snapshot, not a promise of exact resume. The current
    DataLoader shuffle permutation is not captured and no resume path exists.
    State values are tensors and plain Python containers, so weights_only=True
    can read the payload without permitting arbitrary Python objects.
    """
    device = next(model.parameters()).device
    ema_state = ema.state_dict()
    training_state = {"format": "rilo.bc.training-state.v1", "step": int(step),
        "model_state_dict": model.state_dict(), "ema_state_dict": ema_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "device": str(device), "loader": loader_state or {},
        "resume_supported": False, "exact_resume_supported": False}
    # Both references share one tensor mapping in the serialized file; storing
    # training state must not duplicate the EMA policy's weight tensors.
    return atomic_torch_save({"format": "rilo.dp.v1", "spec": ema.spec.model_dump(mode="json"),
        "config": ema.config.model_dump(mode="json"), "state_dict": ema_state,
        "metadata": metadata or {}, "training_state": training_state}, path)


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
