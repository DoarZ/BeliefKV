from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from beliefkv.experiments.deepagents_swebench import load_workload_bundle


ALLOWED_SPLITS = frozenset({"train", "calibration", "test_id"})


@dataclass(frozen=True)
class P6CollectionBatch:
    plan_path: Path
    plan_id: str
    batch_id: str
    split: str
    workload_manifest: Path
    workflow_count: int
    concurrency: int
    preflight_command: str | None


def load_collection_batch(
    plan_path: Path,
    batch_id: str,
    *,
    allow_calibration: bool = False,
    allow_test: bool = False,
) -> P6CollectionBatch:
    plan_path = plan_path.expanduser().resolve()
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("collection plan must be a JSON object")
    if not bool(raw.get("frozen")):
        raise ValueError("collection plan must be frozen")
    if bool(raw.get("predictor_enabled")) or bool(
        raw.get("predictive_actions_enabled")
    ):
        raise ValueError("training evidence must disable predictive policy")
    if raw.get("runtime_policy") != "frozen_p5_observed":
        raise ValueError("training evidence must use frozen_p5_observed")

    matches = [
        item
        for item in raw.get("batches", [])
        if isinstance(item, dict) and item.get("batch_id") == batch_id
    ]
    if len(matches) != 1:
        raise ValueError(f"batch ID must resolve exactly once: {batch_id}")
    batch = matches[0]
    split = str(batch.get("split"))
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported collection split: {split}")
    if split == "calibration" and not allow_calibration:
        raise PermissionError("calibration collection requires --allow-calibration")
    if split == "test_id" and not allow_test:
        raise PermissionError("sealed test collection requires --allow-test")
    if bool(batch.get("predictive_actions")):
        raise ValueError("batch enables predictive actions")
    if batch.get("policy") != "frozen_p5_observed":
        raise ValueError("batch policy differs from frozen_p5_observed")

    manifest_path = Path(str(batch["workload_manifest"])).expanduser().resolve()
    expected_digest = str(batch.get("workload_manifest_sha256", ""))
    if not expected_digest:
        raise ValueError("batch is missing workload_manifest_sha256")
    if _sha256(manifest_path) != expected_digest:
        raise ValueError(f"workload manifest digest mismatch: {manifest_path}")
    bundle = load_workload_bundle(manifest_path)
    workflow_count = int(batch["workflow_count"])
    if len(bundle.workloads) != workflow_count:
        raise ValueError("batch workflow count differs from workload manifest")
    projects = set(batch.get("projects", []))
    if any(item.repo not in projects for item in bundle.workloads):
        raise ValueError("workload project is absent from batch project set")
    declared_images = set(batch.get("docker_images", []))
    if any(item.docker_image not in declared_images for item in bundle.workloads):
        raise ValueError("workload image is absent from batch image set")

    preflight = batch.get("preflight_command")
    return P6CollectionBatch(
        plan_path=plan_path,
        plan_id=str(raw["plan_id"]),
        batch_id=batch_id,
        split=split,
        workload_manifest=manifest_path,
        workflow_count=workflow_count,
        concurrency=int(batch["concurrency"]),
        preflight_command=str(preflight) if preflight is not None else None,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
