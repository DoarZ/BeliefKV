#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SWEBENCH_ROOT = REPOSITORY_ROOT / "third_party/SWE-bench"
for path in (REPOSITORY_ROOT, SWEBENCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from beliefkv.experiments.decision_characterization import (
    select_confirmation_rows,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a predeclared P6 decision-confirmation batch."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--exclude-workload-manifest", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project", default="pydata/xarray")
    parser.add_argument("--batch-id", default="p6-confirm-xarray-v2-r0")
    parser.add_argument("--study-id", default="p6-decision-confirmation-xarray-v2")
    parser.add_argument("--dataset-revision", default="91aa3ed")
    parser.add_argument("--selection-seed", default="beliefkv-p6-confirm-v2")
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    try:
        from datasets import load_from_disk
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError as error:
        raise SystemExit(
            "datasets and the local SWE-bench harness are required"
        ) from error

    excluded_manifest = json.loads(
        args.exclude_workload_manifest.read_text(encoding="utf-8")
    )
    excluded_ids = [
        str(item["instance_id"])
        for item in excluded_manifest.get("workloads", ())
    ]
    rows = load_from_disk(str(args.dataset_dir)).to_list()
    selected = select_confirmation_rows(
        rows,
        project=args.project,
        excluded_instance_ids=excluded_ids,
        count=args.count,
        seed=args.selection_seed,
    )
    source_repo = args.source_repo.expanduser().resolve()
    if not source_repo.joinpath(".git").exists():
        raise ValueError(f"source repository is unavailable: {source_repo}")

    workloads = []
    for raw in selected:
        image = make_test_spec(
            raw,
            namespace="swebench",
            instance_image_tag="latest",
            env_image_tag="latest",
        ).instance_image_key
        workloads.append(
            {
                "instance_id": str(raw["instance_id"]),
                "repo": str(raw["repo"]),
                "base_commit": str(raw["base_commit"]),
                "problem_statement": str(raw["problem_statement"]),
                "difficulty": str(raw.get("difficulty") or "unknown"),
                "version": str(raw["version"]),
                "rollout_index": 0,
                "source_repo": str(source_repo),
                "docker_image": image,
                "preflight_command": None,
            }
        )

    output_dir = args.output_dir.expanduser().resolve()
    workload_path = output_dir / "workload_manifest.json"
    workload = {
        "schema_version": 2,
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "dataset_revision": args.dataset_revision,
        "split": "train",
        "selection_policy": (
            "predeclared development confirmation; exclude prior batch then "
            "rank by SHA256(seed|project|instance_id); no policy-output filtering"
        ),
        "selection_seed": args.selection_seed,
        "excluded_workload_manifest": str(
            args.exclude_workload_manifest.expanduser().resolve()
        ),
        "rollout_index": 0,
        "workloads": workloads,
    }
    _write_json(workload_path, workload)
    plan = {
        "schema_version": 1,
        "plan_id": args.study_id,
        "frozen": True,
        "dataset": workload["dataset"],
        "dataset_revision": args.dataset_revision,
        "predictor_enabled": False,
        "predictive_actions_enabled": False,
        "runtime_policy": "frozen_p5_observed",
        "unique_task_count": len(workloads),
        "workflow_count": len(workloads),
        "repository_count": 1,
        "batch_count": 1,
        "batches": [
            {
                "batch_id": args.batch_id,
                "split": "train",
                "projects": [args.project],
                "versions": sorted(
                    {f"{args.project}:{item['version']}" for item in workloads}
                ),
                "rollout_index": 0,
                "workflow_count": len(workloads),
                "instance_ids": [item["instance_id"] for item in workloads],
                "workload_manifest": str(workload_path),
                "workload_manifest_sha256": _sha256(workload_path),
                "source_repositories": [
                    {
                        "project": args.project,
                        "path": str(source_repo),
                        "url": f"https://github.com/{args.project}.git",
                    }
                ],
                "docker_images": sorted(
                    {item["docker_image"] for item in workloads}
                ),
                "concurrency": len(workloads),
                "predictive_actions": False,
                "policy": "frozen_p5_observed",
                "preflight_command": None,
            }
        ],
    }
    plan_path = output_dir / "collection_plan.json"
    _write_json(plan_path, plan)
    print(
        json.dumps(
            {
                "study_id": args.study_id,
                "batch_id": args.batch_id,
                "instance_ids": [item["instance_id"] for item in workloads],
                "collection_plan": str(plan_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
