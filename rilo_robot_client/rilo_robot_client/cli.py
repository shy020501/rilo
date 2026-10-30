from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
from pathlib import Path
import signal
import sys

from .config import ClientConfig
from .controller import RunController
from .http_client import HTTPPolicyClient
from .journal import DurableJournal


def parser():
    value = argparse.ArgumentParser(description="Independent RILO robot PC client; default is read-only probe")
    value.add_argument("--config", default="config.json")
    mode = value.add_mutually_exclusive_group()
    mode.add_argument("--probe", action="store_true", help="Check HTTP reachability without registration or robot imports")
    mode.add_argument("--mock", action="store_true", help="Register and run only the synthetic robot")
    mode.add_argument("--adapter", help="Explicit module:factory implementing the execution bridge")
    mode.add_argument("--agent-stdin", action="store_true", help="Read private launch envelope from stdin and hold on EOF")
    value.add_argument("--env-config", help="Actual robot environment JSON path; no path is assumed")
    return value


async def adapter_for(mode, adapter, env_config, config):
    if mode == "probe":
        return None, None
    if mode == "mock":
        from .mock import create_adapter
        return create_adapter({**config.adapter_options, "control_hz": config.control_hz})
    if not adapter or ":" not in adapter or not env_config:
        raise ValueError("Robot mode requires --adapter module:factory and --env-config /actual/path.json")
    if not Path(env_config).is_file():
        raise ValueError("The explicitly supplied environment JSON file does not exist")
    module, name = adapter.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    result = factory(config={**config.adapter_options, "control_hz": config.control_hz}, env_config=env_config)
    result = await result if inspect.isawaitable(result) else result
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("Adapter factory must return (ExecutionBridge, ChunkDecoder)")
    return result


async def main_async(arguments):
    reader = pipe = None
    daemon_probe = False
    if arguments.agent_stdin:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader(limit=65536)
        pipe, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
        line = await reader.readline()
        if not line or len(line) > 65536:
            raise ValueError("A bounded private launch envelope is required on stdin")
        envelope = json.loads(line)
        if not isinstance(envelope, dict) or set(envelope) - {"client_config", "mode", "adapter", "env_config"}:
            raise ValueError("Unsupported launch envelope fields")
        mode = envelope.get("mode", "probe")
        if mode not in ("probe", "mock", "robot"):
            raise ValueError("Invalid launch mode")
        config = ClientConfig.from_dict(envelope["client_config"])
        adapter, env_config = envelope.get("adapter"), envelope.get("env_config")
        daemon_probe = mode == "probe"
    else:
        config = ClientConfig.load(arguments.config)
        mode = "mock" if arguments.mock else "robot" if arguments.adapter else "probe"
        adapter, env_config = arguments.adapter, arguments.env_config
    client = HTTPPolicyClient(config)
    journal = controller = eof_task = None
    try:
        if mode == "probe" and not daemon_probe:
            response = await client.health()
            print("RILO endpoint is reachable. No robot was imported or registered.")
            if response.get("protocol"):
                print("Protocol:", response["protocol"])
            return 0
        bridge, decoder = await adapter_for(mode, adapter, env_config, config)
        journal = DurableJournal(config.journal_path)
        controller = RunController(client, journal, bridge=bridge, decoder=decoder,
                                   on_ready=lambda: print("RILO_CLIENT_READY", flush=True))
        loop = asyncio.get_running_loop()
        for name in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(name, lambda: asyncio.create_task(controller.shutdown()))
        if reader is not None:
            async def watch_stdin():
                while await reader.read(1024):
                    pass
                await controller.shutdown()
            eof_task = asyncio.create_task(watch_stdin())
        await controller.serve()
        return 0
    except Exception as exc:
        # Transport/config bodies, token values, and environment dictionaries are never logged.
        message = str(exc).replace(config.project_token, "[redacted]")
        print("Robot client stopped:", message[:500], file=sys.stderr)
        return 1
    finally:
        if eof_task:
            eof_task.cancel()
            await asyncio.gather(eof_task, return_exceptions=True)
        if journal:
            journal.close()
        await client.close()
        if pipe:
            pipe.close()


def main():
    try:
        code = asyncio.run(main_async(parser().parse_args()))
    except (ValueError, KeyError, OSError):
        # Before private config validation succeeds, do not echo its contents.
        print("Robot client configuration is invalid or unavailable.", file=sys.stderr)
        code = 2
    raise SystemExit(code)
