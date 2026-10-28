#!/usr/bin/env python3
"""Install the pinned upstream LIBERO package and write a local path config.

Run after `python -m pip install -e '.[libero]'`. Demonstration datasets are not
needed for GPT rollouts: tasks, assets, and initial states ship in this checkout.
Existing repositories/configuration are checked rather than overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_URL = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PROJECT_ROOT / "vendor" / "LIBERO")
    parser.add_argument("--config-dir", type=Path, default=Path(os.environ.get(
        "LIBERO_CONFIG_PATH", PROJECT_ROOT / "artifacts" / "libero-config")))
    parser.add_argument("--skip-install", action="store_true", help="Only clone/check assets and configure paths")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    config_dir = args.config_dir.expanduser().resolve()
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--depth", "1", LIBERO_URL, str(repo))
        run("git", "-C", str(repo), "fetch", "--depth", "1", "origin", LIBERO_REVISION)
        run("git", "-C", str(repo), "checkout", "--detach", LIBERO_REVISION)
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != LIBERO_REVISION:
        raise SystemExit(
            f"Existing {repo} is at {revision}, expected {LIBERO_REVISION}. "
            "Use --repo with a new empty location or select the pinned commit yourself."
        )
    root = repo / "libero" / "libero"
    for directory in ("assets", "bddl_files", "init_files"):
        if not (root / directory).is_dir():
            raise SystemExit(f"Missing LIBERO asset directory: {root / directory}")
    if not args.skip_install:
        # Upstream uses find_packages() although its outer libero directory is
        # a namespace package. Modern editable mode produces an empty finder;
        # compat mode exposes the checkout just like the original setup did.
        run(sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(repo),
            "--config-settings", "editable_mode=compat")
    config_dir.mkdir(parents=True, exist_ok=True)
    datasets = config_dir / "datasets"
    datasets.mkdir(exist_ok=True)
    config = {
        "benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"), "assets": str(root / "assets"),
        "datasets": str(datasets),
    }
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        print(f"Preserving existing configuration: {config_file}")
    else:
        with config_file.open("x", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        print(f"Created isolated LIBERO configuration: {config_file}")
    print(f"LIBERO revision: {revision}")
    print(f"export LIBERO_CONFIG_PATH={shlex.quote(str(config_dir))}")
    print("Renderer: MUJOCO_GL=egl (working GPU) or MUJOCO_GL=osmesa (libOSMesa CPU rendering).")


if __name__ == "__main__":
    main()
