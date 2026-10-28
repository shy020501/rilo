from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import os

from dotenv import load_dotenv

from rilo.config import Settings


def main():
    parser = argparse.ArgumentParser(description="RILO · live robot experimentation console")
    parser.add_argument("command", nargs="?", choices=["serve", "doctor"], default="serve")
    parser.add_argument("--env", help="libero, demo, or module:adapter_factory")
    parser.add_argument("--policy", help="openai, scripted (demo only), or module:policy_factory")
    parser.add_argument("--demo", action="store_true", help="Explicit 2D smoke test; no LIBERO or paid API calls")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    load_dotenv()
    # CPU rendering works on Linux with libOSMesa. Set MUJOCO_GL=egl for GPU rendering.
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    settings = Settings.from_env()
    overrides = {name: getattr(args, name) for name in ("policy", "host", "port") if getattr(args, name) is not None}
    if args.env:
        overrides["backend"] = args.env
    if args.demo:
        overrides.update(backend="demo", policy="scripted")
    settings = replace(settings, **overrides)
    if args.command == "doctor":
        print(f"Environment: {settings.backend}; policy: {settings.policy}; model: {settings.model}")
        print(f"OPENAI_API_KEY: {'configured' if settings.api_key else 'missing (preview still available)'}")
        print(f"MUJOCO_GL: {os.environ['MUJOCO_GL']}")
        for module in ("libero", "robosuite", "mujoco", "torch", "openai"):
            print(f"{module}: {'installed' if importlib.util.find_spec(module) else 'not installed'}")
        return
    import uvicorn
    from rilo.server import create_app
    print(f"RILO Console: http://{settings.host}:{settings.port}")
    if settings.backend == "demo":
        print("DEMO environment: this is a 2D smoke test, not LIBERO.")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
