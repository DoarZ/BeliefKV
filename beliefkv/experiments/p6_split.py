from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


P6_SPLIT_SCHEMA_VERSION = 1
FINAL_SPLITS = ("train", "calibration", "test_id")


class P6SplitError(ValueError):
    pass


def build_split_manifest(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    dataset_revision: str,
    development_projects: Iterable[str] = (),
    ood_workload_families: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a deterministic project-level split that is frozen before collection."""

    development = frozenset(str(item) for item in development_projects)
    projects: dict[str, list[dict[str, str]]] = {}
    seen_instances: set[str] = set()
    for raw in rows:
        project = str(raw.get("repo") or raw.get("project") or "")
        instance_id = str(raw.get("instance_id") or "")
        base_commit = str(raw.get("base_commit") or "")
        if not project or not instance_id or not base_commit:
            raise P6SplitError("split rows require repo, instance_id and base_commit")
        if instance_id in seen_instances:
            raise P6SplitError(f"duplicate split instance: {instance_id}")
        seen_instances.add(instance_id)
        projects.setdefault(project, []).append(
            {"instance_id": instance_id, "base_commit": base_commit}
        )

    final_projects = sorted(set(projects).difference(development))
    if len(final_projects) < 5:
        raise P6SplitError("at least five non-development projects are required")
    target_counts = _target_project_counts(len(final_projects))
    assignments = _balanced_project_assignment(projects, final_projects, target_counts)
    project_rows = []
    for project in sorted(projects):
        split = "development" if project in development else assignments[project]
        tasks = sorted(projects[project], key=lambda item: item["instance_id"])
        project_rows.append(
            {
                "dataset": dataset,
                "project": project,
                "split": split,
                "task_count": len(tasks),
                "tasks": tasks,
            }
        )

    split_counts = Counter(item["split"] for item in project_rows)
    task_counts = Counter()
    for item in project_rows:
        task_counts[item["split"]] += int(item["task_count"])
    source_digest = hashlib.sha256(
        json.dumps(project_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = {
        "schema_version": P6_SPLIT_SCHEMA_VERSION,
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        "frozen": True,
        "split_unit": "dataset+repository",
        "grouping_contract": (
            "repository, instance_id, base_commit and every repeated rollout share one split"
        ),
        "development_policy": (
            "development rows may be used for pipeline/model-selection sanity checks but "
            "must not appear in final generalization results"
        ),
        "selection_contract": {
            "model_selection": "train projects with leave-one-project-out",
            "calibration": "probability and interval calibration only",
            "test_id": "sealed unseen projects; no model selection or online update",
            "test_ood": "sealed workload families listed separately",
        },
        "target_project_ratios": {
            "train": 0.6,
            "calibration": 0.2,
            "test_id": 0.2,
        },
        "project_counts": dict(sorted(split_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "test_ood_workload_families": sorted(set(ood_workload_families)),
        "projects": project_rows,
        "source_project_task_digest": source_digest,
    }
    validate_split_manifest(result)
    return result


def validate_split_manifest(raw: Mapping[str, Any]) -> None:
    if int(raw.get("schema_version", -1)) != P6_SPLIT_SCHEMA_VERSION:
        raise P6SplitError("unsupported P6 split schema")
    if not raw.get("frozen"):
        raise P6SplitError("P6 split manifest must be frozen")
    seen_projects: set[tuple[str, str]] = set()
    seen_instances: set[tuple[str, str]] = set()
    for project_row in raw.get("projects", ()):
        dataset = str(project_row.get("dataset") or raw.get("dataset") or "")
        project = str(project_row.get("project") or "")
        split = str(project_row.get("split") or "")
        if not dataset or not project or split not in {*FINAL_SPLITS, "development"}:
            raise P6SplitError("invalid project split row")
        project_key = (dataset, project)
        if project_key in seen_projects:
            raise P6SplitError(f"project appears in multiple splits: {project_key}")
        seen_projects.add(project_key)
        tasks = project_row.get("tasks", ())
        if int(project_row.get("task_count", -1)) != len(tasks):
            raise P6SplitError(f"task count mismatch for project {project}")
        for task in tasks:
            instance_id = str(task.get("instance_id") or "")
            base_commit = str(task.get("base_commit") or "")
            task_key = (dataset, instance_id)
            if not instance_id or not base_commit or task_key in seen_instances:
                raise P6SplitError(f"invalid or duplicate task row: {task_key}")
            seen_instances.add(task_key)


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split_manifest(raw)
    return raw


def resolve_split(
    manifest: Mapping[str, Any],
    *,
    dataset: str,
    project: str,
    instance_id: str,
    base_commit: str | None = None,
) -> str:
    for project_row in manifest.get("projects", ()):
        row_dataset = str(project_row.get("dataset") or manifest.get("dataset") or "")
        if row_dataset != dataset or str(project_row.get("project")) != project:
            continue
        task = next(
            (
                item
                for item in project_row.get("tasks", ())
                if str(item.get("instance_id")) == instance_id
            ),
            None,
        )
        if task is None:
            raise P6SplitError(f"task is absent from frozen split: {instance_id}")
        expected_commit = str(task.get("base_commit") or "")
        if base_commit and expected_commit != base_commit:
            raise P6SplitError(
                f"base commit changed for {instance_id}: {base_commit} != {expected_commit}"
            )
        return str(project_row["split"])
    raise P6SplitError(f"project is absent from frozen split: {dataset}:{project}")


def write_split_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    validate_split_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _target_project_counts(total: int) -> dict[str, int]:
    train = max(1, round(total * 0.6))
    calibration = max(1, round(total * 0.2))
    test = total - train - calibration
    if test < 1:
        train -= 1 - test
        test = 1
    return {"train": train, "calibration": calibration, "test_id": test}


def _balanced_project_assignment(
    projects: Mapping[str, list[dict[str, str]]],
    candidates: list[str],
    target_counts: Mapping[str, int],
) -> dict[str, str]:
    assignment: dict[str, str] = {}
    project_counts = Counter()
    task_counts = Counter()
    total_tasks = sum(len(projects[item]) for item in candidates)
    ratios = {"train": 0.6, "calibration": 0.2, "test_id": 0.2}
    for project in sorted(candidates, key=lambda item: (-len(projects[item]), item)):
        eligible = [
            split
            for split in FINAL_SPLITS
            if project_counts[split] < int(target_counts[split])
        ]
        split = min(
            eligible,
            key=lambda item: (
                (task_counts[item] + len(projects[project]))
                / max(1.0, total_tasks * ratios[item]),
                project_counts[item] / max(1, int(target_counts[item])),
                FINAL_SPLITS.index(item),
            ),
        )
        assignment[project] = split
        project_counts[split] += 1
        task_counts[split] += len(projects[project])
    return assignment
