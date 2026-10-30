"""RILO robot wire v1. This file is also shipped with the standalone client.

Only standard library and NumPy are required. Numeric NPZ members are never
extracted to disk, unpickled, or allocated before their declared size is checked.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
import math
import re
import zipfile

import numpy as np


PROTOCOL = "rilo-robot-v1"
MEDIA_TYPE = "application/x-npz"
MAX_PAYLOAD_BYTES = 64 * 1024**2
MAX_JSON_BYTES = 1024**2
MAX_MEMBERS = 128
META_KEY = "__meta__"


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Unsupported JSON value")


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=_default).encode("utf-8")


def contract_hash(contract):
    return hashlib.sha256(json_bytes({key: value for key, value in contract.items() if key != "hash"})).hexdigest()


def _invalid_constant(value):
    raise ValueError("Nonfinite JSON number")


def parse_json(data):
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("JSON message exceeds 1 MiB")
    result = json.loads(data, parse_constant=_invalid_constant)
    if not isinstance(result, dict):
        raise ValueError("Message metadata must be a JSON object")
    return result


def pack_message(meta, arrays=None):
    if not isinstance(meta, dict):
        raise ValueError("Message metadata must be a dict")
    encoded = json_bytes(meta)
    if len(encoded) > MAX_JSON_BYTES:
        raise ValueError("JSON message exceeds 1 MiB")
    members = dict(arrays or {})
    if META_KEY in members:
        raise ValueError("Reserved metadata member")
    members[META_KEY] = np.frombuffer(encoded, dtype=np.uint8)
    if len(members) > MAX_MEMBERS:
        raise ValueError("Too many numeric arrays")
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for key, values in members.items():
            if not isinstance(key, str) or not key or len(key) > 256 or "\x00" in key:
                raise ValueError("Invalid array name")
            value = np.asarray(values)
            if value.dtype.kind not in "biuf" or value.ndim > 8:
                raise ValueError("Only numeric arrays of rank <= 8 are supported")
            if value.nbytes + stream.tell() > MAX_PAYLOAD_BYTES:
                raise ValueError("NPZ message exceeds 64 MiB")
            with archive.open(key + ".npy", "w") as member:
                np.lib.format.write_array(member, value, allow_pickle=False)
    result = stream.getvalue()
    if len(result) > MAX_PAYLOAD_BYTES:
        raise ValueError("NPZ message exceeds 64 MiB")
    return result


def unpack_message(data):
    if len(data) > MAX_PAYLOAD_BYTES:
        raise ValueError("NPZ message exceeds 64 MiB")
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()
            if not 1 <= len(entries) <= MAX_MEMBERS or sum(entry.file_size for entry in entries) > MAX_PAYLOAD_BYTES:
                raise ValueError("Invalid expanded NPZ size")
            arrays = {}
            for entry in entries:
                name = entry.filename[:-4]
                if (not entry.filename.endswith(".npy") or not name or len(name) > 256
                        or name in arrays or entry.compress_type != zipfile.ZIP_STORED):
                    raise ValueError("Expected unique uncompressed NPY members")
                contents = archive.read(entry)
                stream = BytesIO(contents)
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported NPY version")
                if (len(shape) > 8 or any(dimension < 0 for dimension in shape) or dtype.kind not in "biuf"
                        or math.prod(shape) * dtype.itemsize != len(contents) - stream.tell()):
                    raise ValueError("Invalid numeric NPY shape, type or byte length")
                stream.seek(0)
                arrays[name] = np.load(stream, allow_pickle=False)
            metadata = arrays.pop(META_KEY, None)
            if metadata is None or metadata.dtype != np.uint8 or metadata.ndim != 1:
                raise ValueError("Missing uint8 metadata")
            return parse_json(metadata.tobytes()), arrays
    except (zipfile.BadZipFile, EOFError, OSError, UnicodeError) as exc:
        raise ValueError("Invalid NPZ message") from exc


def pack_observations(observations):
    """Return JSON observation descriptors and the associated RGB arrays."""
    if not 1 <= len(observations) <= 8:
        raise ValueError("Send between 1 and 8 observations")
    descriptors, arrays = [], {}
    for index, observation in enumerate(observations):
        value = observation if isinstance(observation, dict) else vars(observation)
        images = value.get("images", {})
        if not isinstance(images, dict) or not 1 <= len(images) <= 8:
            raise ValueError("Each observation requires 1..8 cameras")
        names = list(images)
        for name, pixels in images.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name):
                raise ValueError("Invalid camera name")
            array = np.asarray(pixels)
            if (array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3
                    or not 1 <= array.shape[0] <= 2048 or not 1 <= array.shape[1] <= 2048):
                raise ValueError("Camera images must be uint8 HWC RGB, at most 2048x2048")
            arrays[f"image/{index}/{name}"] = array
        descriptor = {key: value.get(key, default) for key, default in (
            ("id", ""), ("timestamp", 0.), ("task", ""), ("state", {}),
            ("success", False), ("terminated", False), ("info", {}))}
        if not isinstance(descriptor["state"], dict) or not isinstance(descriptor["info"], dict):
            raise ValueError("State and info must be JSON objects")
        if not isinstance(descriptor["id"], str) or not 1 <= len(descriptor["id"]) <= 128:
            raise ValueError("An observation ID is required")
        descriptor["cameras"] = names
        descriptors.append(descriptor)
    return descriptors, arrays


def unpack_observations(descriptors, arrays):
    if not isinstance(descriptors, list) or not 1 <= len(descriptors) <= 8:
        raise ValueError("Send between 1 and 8 observations")
    result = []
    expected = set()
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("cameras"), list):
            raise ValueError("Missing camera descriptors")
        names = descriptor["cameras"]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate camera names")
        images = {}
        for name in names:
            key = f"image/{index}/{name}"
            if key not in arrays:
                raise ValueError("Missing camera array")
            images[name] = arrays[key]
            expected.add(key)
        value = {key: item for key, item in descriptor.items() if key != "cameras"}
        value["images"] = images
        result.append(value)
    # Apply the same camera/metadata validation on each side of the wire.
    pack_observations(result)
    if set(arrays) != expected:
        raise ValueError("Unexpected observation arrays")
    return result
