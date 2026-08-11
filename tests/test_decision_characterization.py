import json
from pathlib import Path

import pytest

from beliefkv.experiments.decision_characterization import (
    build_manifest,
    select_confirmation_rows,
    validate_manifest,
)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    repository = tmp_path / "repo"
    (repository / "beliefkv").mkdir(parents=True)
    (repository / "scripts").mkdir()
    (repository / "beliefkv" / "runtime.py").write_text("VALUE = 1\n")
    workload = _write(
        tmp_path / "workload.json",
        {
            "dataset": "fixture",
            "workloads": [
                {
                    "instance_id": "target-1",
                    "repo": "target/project",
                    "base_commit": "abc",
                    "problem_statement": "fix",
                    "source_repo": str(tmp_path / "source"),
                    "docker_image": "fixture:image",
                    "rollout_index": 0,
                }
            ],
        },
    )
    source = tmp_path / "source" / ".git"
    source.mkdir(parents=True)
    plan = _write(
        tmp_path / "plan.json",
        {
            "frozen": True,
            "predictor_enabled": False,
            "predictive_actions_enabled": False,
            "runtime_policy": "frozen_p5_observed",
            "plan_id": "fixture-plan",
            "batches": [
                {
                    "batch_id": "batch-1",
                    "split": "train",
                    "policy": "frozen_p5_observed",
                    "predictive_actions": False,
                    "workload_manifest": str(workload),
                    "workload_manifest_sha256": __import__("hashlib").sha256(
                        workload.read_bytes()
                    ).hexdigest(),
                    "workflow_count": 1,
                    "concurrency": 1,
                    "projects": ["target/project"],
                    "docker_images": ["fixture:image"],
                }
            ],
        },
    )
    artifacts = {
        "predictor": _write(
            tmp_path / "predictor.json",
            {"schema_version": 1, "metadata": {"fit_projects": ["fit/project"]}},
        ),
        "gpu": _write(tmp_path / "gpu.json", {"schema_version": 1}),
        "transfer": _write(tmp_path / "transfer.json", {"schema_version": 1}),
        "plan": plan,
    }
    return repository, artifacts


def test_manifest_freezes_inputs_and_rejects_source_drift(tmp_path: Path) -> None:
    repository, artifacts = _fixture(tmp_path)
    manifest = build_manifest(
        repository_root=repository,
        study_id="fixture",
        collection_plan=artifacts["plan"],
        batch_id="batch-1",
        predictor_model=artifacts["predictor"],
        gpu_service_model=artifacts["gpu"],
        transfer_service_model=artifacts["transfer"],
        gpu_index=0,
        port=18000,
        pool_tokens=1,
        host_cache_gib=1,
        max_running_requests=1,
        context_length=1,
    )

    validate_manifest(manifest, repository_root=repository)
    (repository / "beliefkv" / "runtime.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="runtime source changed"):
        validate_manifest(manifest, repository_root=repository)


def test_manifest_rejects_predictor_fit_project_overlap(tmp_path: Path) -> None:
    repository, artifacts = _fixture(tmp_path)
    artifacts["predictor"].write_text(
        json.dumps(
            {"schema_version": 1, "metadata": {"fit_projects": ["target/project"]}}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlap predictor fit projects"):
        build_manifest(
            repository_root=repository,
            study_id="fixture",
            collection_plan=artifacts["plan"],
            batch_id="batch-1",
            predictor_model=artifacts["predictor"],
            gpu_service_model=artifacts["gpu"],
            transfer_service_model=artifacts["transfer"],
            gpu_index=0,
            port=18000,
            pool_tokens=1,
            host_cache_gib=1,
            max_running_requests=1,
            context_length=1,
        )


def test_confirmation_selection_is_stable_and_excludes_prior_tasks() -> None:
    rows = [
        {"instance_id": f"task-{index}", "repo": "target/project"}
        for index in range(10)
    ]

    first = select_confirmation_rows(
        rows,
        project="target/project",
        excluded_instance_ids=("task-0", "task-1"),
        count=4,
        seed="fixed-seed",
    )
    second = select_confirmation_rows(
        tuple(reversed(rows)),
        project="target/project",
        excluded_instance_ids=("task-1", "task-0"),
        count=4,
        seed="fixed-seed",
    )

    assert [item["instance_id"] for item in first] == [
        item["instance_id"] for item in second
    ]
    assert not {item["instance_id"] for item in first}.intersection(
        {"task-0", "task-1"}
    )
