#!/usr/bin/env python3
"""Recompute usage/cost without API calls: python scripts/report_costs.py RUN_DIR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rilo.reporting import markdown_report, report_costs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("artifacts/runs")],
                        help="Run directories or directories containing runs")
    parser.add_argument("--json", dest="json_path", type=Path, help="Write the detailed report as JSON")
    parser.add_argument("--markdown", type=Path, help="Also save the concise table report")
    args = parser.parse_args()
    report = report_costs(args.paths)
    rendered = markdown_report(report)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
