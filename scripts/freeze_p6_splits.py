#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.experiments.p6_split import build_split_manifest, write_split_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze project-isolated P6 splits")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-project", action="append", default=[])
    parser.add_argument("--ood-family", action="append", default=[])
    args = parser.parse_args()

    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise SystemExit("datasets is required; run this script in beliefkv-swe") from error
    rows = load_from_disk(str(args.dataset_dir)).to_list()
    manifest = build_split_manifest(
        rows,
        dataset=args.dataset,
        dataset_revision=args.dataset_revision,
        development_projects=args.development_project,
        ood_workload_families=args.ood_family,
    )
    write_split_manifest(args.output, manifest)
    print(args.output)
    for split, count in sorted(manifest["project_counts"].items()):
        print(f"{split}: projects={count} tasks={manifest['task_counts'][split]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
