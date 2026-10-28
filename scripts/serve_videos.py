#!/usr/bin/env python3
"""Serve saved experiment videos locally, including HTTP range requests for seeking."""
from __future__ import annotations

import argparse
from pathlib import Path

from starlette.staticfiles import StaticFiles
import uvicorn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/videos"))
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    app = StaticFiles(directory=args.directory, html=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
