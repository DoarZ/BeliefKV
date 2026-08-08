#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SWEBENCH_ROOT = REPOSITORY_ROOT / "third_party/SWE-bench"
for path in (REPOSITORY_ROOT, SWEBENCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from beliefkv.experiments.p6_split import load_split_manifest, resolve_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare repository/version P6 trace batches")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks-per-project", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-versions-per-project", type=int, default=3)
    parser.add_argument("--repeat-fraction", type=float, default=0.2)
    parser.add_argument("--repeat-rollouts", type=int, default=3)
    args = parser.parse_args()
    if (
        args.tasks_per_project <= 0
        or args.batch_size <= 0
        or args.max_versions_per_project <= 0
    ):
        raise SystemExit("task and batch sizes must be positive")
    if not 0 <= args.repeat_fraction <= 1 or args.repeat_rollouts < 1:
        raise SystemExit("repeat settings are invalid")

    try:
        from datasets import load_from_disk
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError as error:
        raise SystemExit("datasets and the local SWE-bench harness are required") from error

    split_manifest = load_split_manifest(args.split_manifest)
    dataset_rows = load_from_disk(str(args.dataset_dir)).to_list()
    selected: list[dict[str, Any]] = []
    by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in dataset_rows:
        project = str(raw["repo"])
        split = resolve_split(
            split_manifest,
            dataset=str(split_manifest["dataset"]),
            project=project,
            instance_id=str(raw["instance_id"]),
            base_commit=str(raw["base_commit"]),
        )
        if split == "development":
            continue
        item = dict(raw)
        item["split"] = split
        by_project[project].append(item)
    for project, rows in sorted(by_project.items()):
        selected.extend(
            _select_project_rows(
                rows, args.tasks_per_project, args.max_versions_per_project
            )
        )

    image_by_instance = {
        str(row["instance_id"]): make_test_spec(
            row,
            namespace="swebench",
            instance_image_tag="latest",
            env_image_tag="latest",
        ).instance_image_key
        for row in selected
    }

    repeat_count = round(len(selected) * args.repeat_fraction)
    repeated_ids = {
        item["instance_id"]
        for item in sorted(selected, key=_stable_row_rank)[:repeat_count]
    }
    rollout_rows: list[dict[str, Any]] = []
    for row in selected:
        count = args.repeat_rollouts if row["instance_id"] in repeated_ids else 1
        for rollout_index in range(count):
            project_slug = str(row["repo"]).replace("/", "__")
            rollout_rows.append(
                {
                    **row,
                    "rollout_index": rollout_index,
                    "source_repo": str((args.source_root / project_slug).resolve()),
                    "docker_image": image_by_instance[str(row["instance_id"])],
                }
            )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        grouped[(str(row["split"]), int(row["rollout_index"]))].append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = args.output_dir / "workload_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    batches: list[dict[str, Any]] = []
    batch_sequence = 0
    for group_key, rows in sorted(grouped.items()):
        split, rollout_index = group_key
        rows.sort(key=lambda item: (str(item["repo"]), str(item["instance_id"])))
        for offset in range(0, len(rows), args.batch_size):
            chunk = rows[offset : offset + args.batch_size]
            batch_sequence += 1
            batch_id = f"p6-{batch_sequence:03d}-{split}-mixed-r{rollout_index}"
            workload_manifest = {
                "schema_version": 2,
                "dataset": split_manifest["dataset"],
                "dataset_revision": split_manifest["dataset_revision"],
                "split_manifest": str(args.split_manifest.resolve()),
                "split": split,
                "selection_policy": (
                    "project-isolated frozen split with per-workload repository/version "
                    "sandbox; deterministic difficulty and version stratification"
                ),
                "rollout_index": rollout_index,
                "workloads": [
                    {
                        "instance_id": item["instance_id"],
                        "repo": item["repo"],
                        "base_commit": item["base_commit"],
                        "problem_statement": item["problem_statement"],
                        "difficulty": item.get("difficulty", "unknown"),
                        "version": item["version"],
                        "rollout_index": rollout_index,
                        "source_repo": item["source_repo"],
                        "docker_image": item["docker_image"],
                        "preflight_command": None,
                    }
                    for item in chunk
                ],
            }
            manifest_path = manifest_dir / f"{batch_id}.json"
            _write_json(manifest_path, workload_manifest)
            batches.append(
                {
                    "batch_id": batch_id,
                    "split": split,
                    "projects": sorted({str(item["repo"]) for item in chunk}),
                    "versions": sorted(
                        {f"{item['repo']}:{item['version']}" for item in chunk}
                    ),
                    "rollout_index": rollout_index,
                    "workflow_count": len(chunk),
                    "instance_ids": [item["instance_id"] for item in chunk],
                    "workload_manifest": str(manifest_path.resolve()),
                    "workload_manifest_sha256": _sha256(manifest_path),
                    "source_repositories": [
                        {
                            "project": project,
                            "path": str(
                                (args.source_root / project.replace("/", "__")).resolve()
                            ),
                            "url": f"https://github.com/{project}.git",
                        }
                        for project in sorted({str(item["repo"]) for item in chunk})
                    ],
                    "docker_images": sorted(
                        {str(item["docker_image"]) for item in chunk}
                    ),
                    "concurrency": len(chunk),
                    "predictive_actions": False,
                    "policy": "frozen_p5_observed",
                    "preflight_command": None,
                }
            )

    split_workflows = Counter(row["split"] for row in rollout_rows)
    split_unique = Counter(row["split"] for row in selected)
    plan = {
        "schema_version": 1,
        "plan_id": "p6-agent-semantics-v1",
        "frozen": True,
        "dataset": split_manifest["dataset"],
        "dataset_revision": split_manifest["dataset_revision"],
        "split_manifest": str(args.split_manifest.resolve()),
        "source_root": str(args.source_root.resolve()),
        "predictor_enabled": False,
        "predictive_actions_enabled": False,
        "runtime_policy": "frozen_p5_observed",
        "tasks_per_project_cap": args.tasks_per_project,
        "repeat_fraction": args.repeat_fraction,
        "repeat_rollouts": args.repeat_rollouts,
        "unique_task_count": len(selected),
        "workflow_count": len(rollout_rows),
        "repository_count": len(by_project),
        "batch_count": len(batches),
        "unique_task_counts_by_split": dict(sorted(split_unique.items())),
        "workflow_counts_by_split": dict(sorted(split_workflows.items())),
        "batches": batches,
    }
    _write_json(args.output_dir / "collection_plan.json", plan)
    print(json.dumps({key: plan[key] for key in (
        "unique_task_count", "workflow_count", "repository_count", "batch_count",
        "unique_task_counts_by_split", "workflow_counts_by_split",
    )}, indent=2, sort_keys=True))
    return 0


def _select_project_rows(
    rows: list[dict[str, Any]], limit: int, max_versions: int
) -> list[dict[str, Any]]:
    by_version: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_version[str(row["version"])].append(row)
    versions = sorted(by_version, key=lambda item: (-len(by_version[item]), item))[
        :max_versions
    ]
    queues = []
    for version in versions:
        queues.append(sorted(by_version[version], key=_selection_rank))
    selected: list[dict[str, Any]] = []
    while len(selected) < min(limit, sum(len(item) for item in queues)):
        progressed = False
        for queue in queues:
            if queue and len(selected) < limit:
                selected.append(queue.pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _selection_rank(row: dict[str, Any]) -> tuple[str, str]:
    difficulty = str(row.get("difficulty") or "unknown")
    digest = hashlib.sha256(str(row["instance_id"]).encode()).hexdigest()
    return difficulty, digest


def _stable_row_rank(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"{row['split']}|{row['repo']}|{row['instance_id']}".encode()
    ).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
