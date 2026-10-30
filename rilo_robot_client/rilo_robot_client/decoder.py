"""Explicit vector decoding; physical coordinate conventions remain adapter-owned."""
from __future__ import annotations

import hashlib
import json
import numpy as np

from .bridge import DecodedAction


def verify_contract(contract, environment):
    if not isinstance(contract, dict) or not isinstance(contract.get("hash"), str):
        raise ValueError("Policy contract with a canonical hash is required")
    metadata = {key: value for key, value in contract.items() if key != "hash"}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if digest != contract["hash"]:
        raise ValueError("Policy contract hash does not match its metadata")
    spec, expected = contract.get("action_spec", {}), environment.get("action_spec", {})
    for key in ("names", "low", "high", "control_hz"):
        if spec.get(key) != expected.get(key):
            raise ValueError(f"Policy and robot action schema disagree on {key}")
    if contract.get("simulation_only") and not environment.get("simulation", False):
        raise ValueError("Simulation-only policies cannot execute on a physical adapter")
    return metadata


class VectorChunkDecoder:
    """Decode the declared action-spec vector without inventing robot transforms."""
    def decode(self, raw_chunk, *, context):
        contract = context["contract"]
        if contract.get("representation") != "action_spec_vector":
            raise ValueError("This decoder only accepts explicitly declared action_spec_vector")
        actions = np.asarray(raw_chunk["actions"], dtype=np.float64)
        names = contract["action_spec"]["names"]
        if actions.ndim != 2 or actions.shape[1] != len(names) or not len(actions) or not np.isfinite(actions).all():
            raise ValueError("Full chunk must be a finite [horizon, action_dimension] array")
        encoding = raw_chunk["actions_encoding"]
        if encoding == "normalized":
            transform = contract.get("action_normalization", {})
            center, scale = np.asarray(transform.get("center")), np.asarray(transform.get("scale"))
            if (center.shape != (len(names),) or scale.shape != center.shape
                    or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
                raise ValueError("Normalized actions require explicit finite center/positive scale")
            if np.any(np.abs(actions) > 1 + 1e-6):
                raise ValueError("Normalized action is outside [-1, 1]")
            actions = actions * scale + center
        elif encoding != "environment":
            raise ValueError("Unknown action encoding")
        low, high = np.asarray(contract["action_spec"]["low"]), np.asarray(contract["action_spec"]["high"])
        if low.shape != (len(names),) or high.shape != low.shape or np.any(actions < low - 1e-6) or np.any(actions > high + 1e-6):
            raise ValueError("Decoded action violates the registered environment bounds")
        decoded = [DecodedAction(row.copy(), {"names": list(names), "values": row.tolist(),
                   "encoding": "environment", "chunk_index": index}) for index, row in enumerate(actions)]
        delay = context.get("observation_delay_ticks", 0)
        if type(delay) is not int or delay < 0 or delay >= len(decoded):
            raise ValueError("The measured inference delay leaves no future action in this chunk")
        # This decoder's rows explicitly represent one registered control tick.
        # Custom interpreters receive the full raw matrix and own their alignment.
        return decoded[delay:]
