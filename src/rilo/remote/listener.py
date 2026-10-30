"""Lazy loopback listener exposing only authenticated robot RPC, never UI controls."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import socket

from fastapi import FastAPI
import uvicorn


class _RPCServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self):
        # The main RILO server owns process signals. A nested listener must not
        # install or restore handlers over the application's handlers.
        yield

    def install_signal_handlers(self):
        pass


def _socket():
    return socket.socket(socket.AF_INET, socket.SOCK_STREAM)


class PolicyRPCListener:
    def __init__(self, service, *, socket_factory=None, server_factory=None,
                 startup_timeout=5., shutdown_timeout=5.):
        self.service = service
        self.socket_factory = socket_factory or _socket
        self.server_factory = server_factory or _RPCServer
        self.startup_timeout, self.shutdown_timeout = startup_timeout, shutdown_timeout
        self._lock = asyncio.Lock()
        self._socket = self._server = self._task = None
        self._closed = False
        self._port = None

    @staticmethod
    async def _finish_owned(task):
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    raise

    async def ensure(self):
        """Return the SSH reverse-forward tuple after the private server starts."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("Policy RPC listener is closing")
            if self._task is not None and not self._task.done() and self._server.started:
                return (45587, "127.0.0.1", self._port)
            if self._task is not None:
                await self._finish_owned(asyncio.create_task(self._stop_owned()))
            candidate = None
            try:
                # Local import avoids service/API/launcher import cycles.
                from rilo.remote.api import create_rpc_app
                app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
                app.mount("/robot/v1", create_rpc_app(self.service))
                candidate = self.socket_factory()
                candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                candidate.setblocking(False)
                candidate.set_inheritable(False)
                candidate.bind(("127.0.0.1", 0))
                candidate.listen(128)
                host, port = candidate.getsockname()[:2]
                if host != "127.0.0.1" or not 1 <= port <= 65535:
                    raise RuntimeError("Policy RPC socket did not bind to loopback")
                config = uvicorn.Config(app, host="127.0.0.1", port=port, loop="asyncio",
                    lifespan="off", access_log=False, log_config=None,
                    timeout_graceful_shutdown=self.shutdown_timeout)
                self._socket, self._port = candidate, port
                self._server = self.server_factory(config)
                self._task = asyncio.create_task(self._server.serve(sockets=[candidate]), name="rilo-private-policy-rpc")
                async def ready():
                    while not self._server.started:
                        if self._closed:
                            raise RuntimeError("Policy RPC listener is closing")
                        if self._task.done():
                            # Consume any failure without exposing network or
                            # request diagnostics through the browser response.
                            await asyncio.gather(self._task, return_exceptions=True)
                            raise RuntimeError("Policy RPC listener failed to start")
                        await asyncio.sleep(.01)
                await asyncio.wait_for(ready(), self.startup_timeout)
                if self._task.done() or self._closed:
                    raise RuntimeError("Policy RPC listener stopped during startup")
                return (45587, "127.0.0.1", port)
            except BaseException:
                if candidate is not None and candidate is not self._socket:
                    candidate.close()
                await self._finish_owned(asyncio.create_task(self._stop_owned()))
                raise

    async def _stop_owned(self):
        server, task, owned_socket = self._server, self._task, self._socket
        if server is not None:
            server.should_exit = True
        stopped = task is None or task.done()
        try:
            if task is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(task), self.shutdown_timeout + .25)
                    stopped = True
                except asyncio.TimeoutError:
                    server.force_exit = True
                    # Cancellation skips uvicorn's ordinary shutdown path. Close
                    # only listeners/connections owned by this nested server.
                    for listener in getattr(server, "servers", ()):
                        listener.close()
                    state = getattr(server, "server_state", None)
                    for connection in list(getattr(state, "connections", ())):
                        connection.shutdown()
                    for request in list(getattr(state, "tasks", ())):
                        request.cancel()
                    task.cancel()
                    try:
                        await asyncio.wait_for(asyncio.shield(task), 1.)
                        stopped = True
                    except asyncio.CancelledError:
                        stopped = task.done()
                    except asyncio.TimeoutError:
                        raise RuntimeError("Policy RPC server did not stop within its deadline") from None
                except asyncio.CancelledError:
                    stopped = task.done()
                except Exception:
                    stopped = task.done()
        finally:
            if owned_socket is not None:
                owned_socket.close()
            if stopped:
                self._socket = self._server = self._task = self._port = None

    async def close(self):
        self._closed = True
        async with self._lock:
            await self._finish_owned(asyncio.create_task(self._stop_owned()))
