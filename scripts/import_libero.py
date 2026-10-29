#!/usr/bin/env python3
"""Stage published LIBERO demos, or commit while the project is explicitly idle."""
import argparse
import json
from pathlib import Path

from rilo.import_libero import commit_import, stage_import
from rilo.projects import ProjectStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--source-url")
    parser.add_argument("--source-revision")
    parser.add_argument("--demo-ids", nargs="+")
    parser.add_argument("--image-convention", choices=("opengl", "opencv"))
    parser.add_argument("--published-successful", action="store_true", required=True,
                        help="Identify an official published successful-demonstration dataset")
    parser.add_argument("--commit-offline", action="store_true",
                        help="Commit only when the server is stopped; otherwise leave a private staged run")
    args = parser.parse_args()
    projects = ProjectStore(args.data_dir)
    staged = stage_import(args.source, args.data_dir.parent / ".libero-imports", projects.get(args.project_id),
        task_name=args.task_name, published_successful=args.published_successful,
        source_url=args.source_url, source_revision=args.source_revision,
        demo_ids=args.demo_ids, image_convention=args.image_convention)
    if args.commit_offline:
        print(json.dumps(commit_import(staged, projects, args.project_id, ensure_idle=lambda: None)))
    else:
        print(json.dumps(staged))


if __name__ == "__main__":
    main()
