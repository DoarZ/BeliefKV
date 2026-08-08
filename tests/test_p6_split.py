from __future__ import annotations

import json

import pytest

from beliefkv.experiments.p6_split import (
    P6SplitError,
    build_split_manifest,
    resolve_split,
    validate_split_manifest,
)


def _rows() -> list[dict[str, str]]:
    return [
        {
            "repo": f"org/project-{project}",
            "instance_id": f"project-{project}__task-{task}",
            "base_commit": f"commit-{project}-{task}",
        }
        for project in range(6)
        for task in range(project + 1)
    ]


def test_split_is_project_isolated_and_deterministic() -> None:
    first = build_split_manifest(
        _rows(),
        dataset="dataset",
        dataset_revision="revision",
        development_projects=("org/project-0",),
    )
    second = build_split_manifest(
        reversed(_rows()),
        dataset="dataset",
        dataset_revision="revision",
        development_projects=("org/project-0",),
    )
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["project_counts"]["development"] == 1
    assert set(first["project_counts"]) == {
        "train",
        "calibration",
        "test_id",
        "development",
    }
    assert (
        resolve_split(
            first,
            dataset="dataset",
            project="org/project-0",
            instance_id="project-0__task-0",
            base_commit="commit-0-0",
        )
        == "development"
    )


def test_split_rejects_task_or_commit_not_frozen() -> None:
    manifest = build_split_manifest(
        _rows(), dataset="dataset", dataset_revision="revision"
    )
    with pytest.raises(P6SplitError, match="base commit changed"):
        resolve_split(
            manifest,
            dataset="dataset",
            project="org/project-0",
            instance_id="project-0__task-0",
            base_commit="different",
        )
    broken = dict(manifest)
    broken["frozen"] = False
    with pytest.raises(P6SplitError, match="must be frozen"):
        validate_split_manifest(broken)
