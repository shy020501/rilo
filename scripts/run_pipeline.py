#!/usr/bin/env python3
"""Control the existing RILO worker; never create a second robot controller."""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--project", required=True, help="Project ID from the UI URL")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    configure = commands.add_parser("configure")
    configure.add_argument("--file", type=Path, required=True)
    start = commands.add_parser("start")
    mode = start.add_mutually_exclusive_group(required=True)
    mode.add_argument("--auto", action="store_true", help="Explicitly run all stages until targets/bounds")
    mode.add_argument("--stage", choices=["collect", "bc", "bc_eval", "progress", "rl", "rl_eval"])
    for command in ("pause", "resume", "stop"):
        commands.add_parser(command)
    args = parser.parse_args()
    url = args.server.rstrip("/") + "/api/projects/" + quote(args.project, safe="") + "/pipeline"
    payload = None
    if args.command == "configure":
        payload = json.loads(args.file.read_text())
        url += "/config"
    elif args.command == "start":
        payload = {"mode": "auto"} if args.auto else {"mode": "manual", "stage": args.stage}
        url += "/start"
    elif args.command != "status":
        payload, url = {}, url + "/" + args.command
    request = Request(url, data=None if payload is None else json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=120) as response:
            print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
    except HTTPError as exc:
        parser.exit(1, f"RILO returned HTTP {exc.code}: {exc.read().decode()}\n")
    except URLError as exc:
        parser.exit(1, f"Cannot reach RILO: {exc.reason}\n")


if __name__ == "__main__":
    main()
