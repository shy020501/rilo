"""Authenticated bounded HTTP transport; retries preserve exact NPZ request bytes."""
from __future__ import annotations

import asyncio
import uuid

import httpx

from .wire import MEDIA_TYPE, pack_message, unpack_message, pack_observations


PROTOCOL = "rilo-robot-v1"
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class RemoteError(RuntimeError):
    def __init__(self, status, detail):
        self.status = status
        super().__init__(str(detail)[:500])


class HTTPPolicyClient:
    def __init__(self, config, *, transport=None):
        self.config = config
        self._client = httpx.AsyncClient(base_url=config.base_url.rstrip("/") + "/",
            headers={"Authorization": "Bearer " + config.project_token, "X-Rilo-Protocol": PROTOCOL},
            timeout=config.request_timeout_seconds, follow_redirects=False, trust_env=False, transport=transport)

    async def _request(self, path, *, method="POST", content=None, json=None, binary=False, retry=False, contract_hash=None):
        attempts = 3 if retry else 1
        for attempt in range(attempts):
            try:
                headers = {"Content-Type": MEDIA_TYPE} if content is not None else {}
                if contract_hash:
                    headers["X-Rilo-Contract"] = contract_hash
                async with self._client.stream(method, path.lstrip("/"), content=content, json=json, headers=headers) as response:
                    parts, length = [], 0
                    async for block in response.aiter_bytes():
                        length += len(block)
                        if length > MAX_RESPONSE_BYTES:
                            raise RemoteError(502, "Server response exceeded the protocol size limit")
                        parts.append(block)
                    raw = b"".join(parts)
                    if response.status_code >= 400:
                        import json as json_module
                        try:
                            detail = json_module.loads(raw).get("detail", "Remote request rejected")
                        except (ValueError, AttributeError):
                            detail = "Remote request rejected"
                        raise RemoteError(response.status_code, detail)
                    if binary:
                        return unpack_message(raw)
                    import json as json_module
                    return json_module.loads(raw)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt + 1 == attempts:
                    raise RemoteError(0, "Connection interrupted; motion remains held until explicit resume") from None
                await asyncio.sleep(.1 * (attempt + 1))

    async def health(self):
        return await self._request("health", method="GET")

    async def register(self, client_id, environment, capabilities):
        return await self._request("register", json={"client_id": client_id, "environment": environment,
            "capabilities": capabilities, "client_version": "0.1.0", "registration_id": uuid.uuid4().hex}, retry=True)

    async def poll(self, session_id, after_command_id, status):
        return await self._request("poll", json={"session_id": session_id,
            "after_command_id": after_command_id, "status": status}, retry=True)

    async def acknowledge(self, session_id, command, *, status="applied", detail=None):
        return await self._request("ack", json={"session_id": session_id, "command_id": command["id"],
            "epoch": command["epoch"], "status": status, **({"detail": detail} if detail else {})}, retry=True)

    def encode(self, metadata, observations=(), arrays=None):
        meta_observations, images = pack_observations([
            item.wire(metadata.get("task", "")) if hasattr(item, "wire") else item for item in observations])
        metadata = {**metadata, "request_id": metadata.get("request_id") or uuid.uuid4().hex,
                    "observations": meta_observations}
        return pack_message(metadata, {**images, **(arrays or {})})

    async def send_encoded(self, path, payload, *, binary=False, contract_hash=None):
        return await self._request(path, content=payload, binary=binary, retry=True, contract_hash=contract_hash)

    async def infer(self, metadata, observations, *, contract_hash):
        # Encode once. All transport retries use this same request_id and body.
        return await self.send_encoded("infer", self.encode(metadata, observations), binary=True, contract_hash=contract_hash)

    async def close(self):
        await self._client.aclose()
