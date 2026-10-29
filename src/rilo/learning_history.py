"""Read-only BC telemetry; no neural dependencies or checkpoint deserialization."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import stat


MAX_LOG_BYTES = 8 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 16384
TRAIN_POINT_TARGET = 2000


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", value):
        raise FileNotFoundError("Unknown training history")
    return value


def safe_path(root, *parts):
    root = Path(root)
    if root.is_symlink():
        raise FileNotFoundError("Invalid artifact directory")
    path = root
    for part in parts:
        identifier(part)
        path = path / part
        if path.is_symlink():
            raise FileNotFoundError("Invalid artifact directory")
    if not path.resolve().is_relative_to(root.resolve()):
        raise FileNotFoundError("Invalid artifact directory")
    return path


def _open(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise FileNotFoundError("Not a regular history file")
    return os.fdopen(descriptor, "rb")


def read_json(root, *parts):
    path = safe_path(root, *parts)
    with _open(path) as handle:
        if os.fstat(handle.fileno()).st_size > MAX_JSON_BYTES:
            raise ValueError("History metadata exceeds the read limit")
        return json.loads(handle.read(MAX_JSON_BYTES))


def _number(value):
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def training_identity(state, model_id=None, *, live=False):
    """Select only identities authorized by this project's own registry."""
    candidates = {}
    for model in state.get("models", []):
        if model.get("kind") != "bc":
            continue
        candidates[model["id"]] = {"model_id": model["id"], "job_id": None,
            "phase": "completed", "started_at": model.get("created_at", 0),
            "target_steps": model.get("config", {}).get("steps"), "model": model}
    status = state.get("status", {})
    for record in state.get("bc_trainings", []):
        item = {**candidates.get(record["model_id"], {}), **record}
        if item.get("phase") in ("running", "paused"):
            if status.get("job_id") == item.get("job_id"):
                item["phase"] = status.get("phase", "interrupted") if live else "interrupted"
            else:
                item["phase"] = "interrupted"
        candidates[item["model_id"]] = item
    if model_id is not None:
        identifier(model_id)
        if model_id not in candidates:
            raise FileNotFoundError("Unknown BC training")
        return candidates[model_id]
    return max(candidates.values(), key=lambda item: (item.get("started_at", 0), item["model_id"]), default=None)


def _points(root, model_id):
    path = safe_path(root, "models", model_id, "metrics.jsonl")
    try:
        with _open(path) as handle:
            size = os.fstat(handle.fileno()).st_size
            offset = max(0, size - MAX_LOG_BYTES)
            handle.seek(offset)
            raw = handle.read(MAX_LOG_BYTES)
    except FileNotFoundError:
        return [], False, False, 0, "metrics_not_found"
    # A bounded tail is explicit in the response. Never parse a cut first row
    # or a concurrently written final row as an actual metric sample.
    lines = raw.split(b"\n")
    if offset:
        lines = lines[1:]
    lines = lines[:-1]
    points = []
    for line in lines:
        if len(line) > MAX_LINE_BYTES:
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if (not isinstance(row, dict) or not isinstance(row.get("step"), int)
                or isinstance(row["step"], bool) or row["step"] < 0 or not _number(row.get("time"))):
            continue
        point = {"time": row["time"], "step": row["step"]}
        for name in ("train_loss", "validation_loss"):
            if _number(row.get(name)):
                point[name] = row[name]
        if len(point) > 2:
            points.append(point)
    total = len(points)
    # Preserve every validation measurement in the read window. Only ordinary
    # training samples are thinned; the byte limit bounds even dense validation.
    keep = {i for i, point in enumerate(points) if "validation_loss" in point}
    if points:
        keep.update((0, len(points) - 1))
    ordinary = [i for i in range(total) if i not in keep]
    slots = max(0, TRAIN_POINT_TARGET - len(keep))
    if len(ordinary) <= slots:
        keep.update(ordinary)
    elif slots:
        keep.update(ordinary[min(len(ordinary)-1, i * len(ordinary) // slots)] for i in range(slots))
    return [points[i] for i in sorted(keep)], len(keep) < total, bool(offset), total, "" if total else "no_metrics"


def _checkpoints(root, model_id, model):
    entries = []
    try:
        index = read_json(root, "models", model_id, "checkpoints.json")
    except (FileNotFoundError, ValueError, OSError):
        index = {}
    rows = index.get("checkpoints", []) if isinstance(index, dict) else []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or row.get("kind") not in ("periodic", "final"):
            continue
        if not isinstance(row.get("step"), int) or isinstance(row["step"], bool) or row["step"] < 0:
            continue
        try:
            path = safe_path(root, "models", model_id, row.get("file"))
        except (FileNotFoundError, TypeError):
            continue
        if not path.is_file():
            continue
        item = {"step": row["step"], "file": path.name, "kind": row["kind"]}
        if _number(row.get("created_at")):
            item["created_at"] = row["created_at"]
        if isinstance(row.get("checkpoint_sha256"), str):
            item["checkpoint_sha256"] = row["checkpoint_sha256"]
        entries.append(item)
    original = Path(model.get("checkpoint", "")) if model else root / "models" / model_id / "policy.pt"
    try:
        best = safe_path(root, "models", model_id, original.name)
        if original.resolve() == best.resolve() and best.is_file():
            # Neither active nor legacy best checkpoints have a reliable
            # selected step in the registry. Never use the final train step.
            entries.insert(0, {"step": None, "file": best.name, "kind": "best",
                               "checkpoint_sha256": model.get("checkpoint_sha256") if model else None})
    except (FileNotFoundError, TypeError):
        pass
    return entries


def bc_history(root, project_id, state, *, model_id=None, live=False):
    identity = training_identity(state, model_id, live=live)
    result = {"project_id": project_id, "model_id": None, "job_id": None, "phase": "idle",
        "target_steps": None, "points": [], "checkpoints": [], "available": False,
        "reason": "no_training", "sampled": False, "truncated": False, "total_points": 0}
    if identity is None:
        return result
    model_id = identifier(identity["model_id"])
    result.update({key: identity.get(key) for key in ("model_id", "job_id", "phase", "target_steps")})
    points, sampled, truncated, total, reason = _points(root, model_id)
    result.update(points=points, sampled=sampled, truncated=truncated, total_points=total,
                  available=bool(points), reason=reason,
                  checkpoints=_checkpoints(root, model_id, identity.get("model")))
    return result
