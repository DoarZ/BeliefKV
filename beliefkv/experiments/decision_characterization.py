from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
RISK_POLICY = {
    "particle_count": 128,
    "top_k": 8,
    "max_candidates": 8,
    "min_calibration_coverage": 0.9,
    "commit_guard_ms": 25.0,
    "prefetch_min_hbm_feasibility": 0.95,
    "prefetch_desired_lead_ms": 100.0,
    "prepare_host_enabled": True,
}
SERVER_CONFIG_CONTRACT = {
    "predictive_risk_shadow_enabled": True,
    "predictive_joint_overlay_enabled": False,
    "predictive_prefetch_canary_enabled": False,
    "joint_policy_enabled": True,
    "joint_observed_mode_enabled": True,
    "running_batch_retraction_enabled": True,
    "predictive_transfer_model_mode": "morphology-aware",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(repository_root: Path) -> dict[str, Any]:
    files = sorted(
        {
            *repository_root.joinpath("beliefkv").rglob("*.py"),
            *repository_root.joinpath("scripts").glob("*.py"),
            *repository_root.joinpath("scripts").glob("*.sh"),
        }
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(repository_root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return {
        "algorithm": "sha256(path\\0sha256(content))",
        "digest": digest.hexdigest(),
        "file_count": len(files),
    }


def select_confirmation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    project: str,
    excluded_instance_ids: Sequence[str],
    count: int,
    seed: str,
) -> list[dict[str, Any]]:
    """Select a frozen development batch without consulting policy outcomes."""

    if count <= 0:
        raise ValueError("confirmation batch size must be positive")
    if not seed:
        raise ValueError("confirmation selection seed must be non-empty")
    excluded = set(excluded_instance_ids)
    candidates: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if str(raw.get("repo")) != project:
            continue
        instance_id = str(raw.get("instance_id") or "")
        if not instance_id or instance_id in excluded:
            continue
        if instance_id in candidates:
            raise ValueError(f"duplicate confirmation instance: {instance_id}")
        candidates[instance_id] = dict(raw)
    ranked = sorted(
        candidates.values(),
        key=lambda item: hashlib.sha256(
            f"{seed}|{project}|{item['instance_id']}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ranked) < count:
        raise ValueError(
            f"confirmation pool has {len(ranked)} rows, fewer than {count}"
        )
    return ranked[:count]


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "schema_version": raw.get("schema_version"),
        "model_version": raw.get("model_version"),
    }


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_payload_sha256", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    *,
    repository_root: Path,
    study_id: str,
    collection_plan: Path,
    batch_id: str,
    predictor_model: Path,
    gpu_service_model: Path,
    transfer_service_model: Path,
    gpu_index: int,
    port: int,
    pool_tokens: int,
    host_cache_gib: int,
    max_running_requests: int,
    context_length: int,
) -> dict[str, Any]:
    from beliefkv.experiments.p6_collection import load_collection_batch

    repository_root = repository_root.expanduser().resolve()
    collection_plan = collection_plan.expanduser().resolve()
    batch = load_collection_batch(collection_plan, batch_id)
    workload = json.loads(batch.workload_manifest.read_text(encoding="utf-8"))
    projects = sorted({str(item["repo"]) for item in workload["workloads"]})
    predictor = _artifact(predictor_model)
    predictor_raw = json.loads(Path(predictor["path"]).read_text(encoding="utf-8"))
    fit_projects = sorted(
        str(item)
        for item in predictor_raw.get("metadata", {}).get("fit_projects", ())
    )
    overlap = sorted(set(projects).intersection(fit_projects))
    if overlap:
        raise ValueError(
            "characterization projects overlap predictor fit projects: "
            + ", ".join(overlap)
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "frozen": True,
        "selection_contract": {
            "predeclared_task_batch": True,
            "adaptive_trace_selection": False,
            "sealed_test_id_opened": False,
            "decision_rule": (
                "report every paired candidate; do not add tasks after observing "
                "promotion, veto, or selected-action outcomes"
            ),
        },
        "collection": {
            "plan_path": str(collection_plan),
            "plan_sha256": sha256_file(collection_plan),
            "batch_id": batch.batch_id,
            "split": batch.split,
            "workload_manifest_path": str(batch.workload_manifest),
            "workload_manifest_sha256": sha256_file(batch.workload_manifest),
            "workflow_count": batch.workflow_count,
            "concurrency": batch.concurrency,
            "projects": projects,
            "instance_ids": [
                str(item["instance_id"]) for item in workload["workloads"]
            ],
            "arrival_policy": "single_frozen_batch_concurrency_8",
            "rollout_index": sorted(
                {int(item.get("rollout_index", 0)) for item in workload["workloads"]}
            ),
        },
        "artifacts": {
            "predictor": predictor,
            "gpu_service": _artifact(gpu_service_model),
            "transfer_service": _artifact(transfer_service_model),
        },
        "predictor_fit_projects": fit_projects,
        "target_project_overlap_with_predictor_fit_projects": overlap,
        "risk_policy": dict(RISK_POLICY),
        "runtime_policy": {
            "mode": "p5_observed_plus_p6_read_only_risk_shadow",
            "predictive_actions_enabled": False,
            "gpu_index": gpu_index,
            "port": port,
            "pool_tokens": pool_tokens,
            "host_cache_gib": host_cache_gib,
            "max_running_requests": max_running_requests,
            "context_length": context_length,
            "joint_workflow_active_window": batch.concurrency,
        },
        "runtime_source_fingerprint": source_fingerprint(repository_root),
    }
    payload["manifest_payload_sha256"] = _canonical_digest(payload)
    validate_manifest(payload, repository_root=repository_root)
    return payload


def validate_manifest(
    payload: Mapping[str, Any],
    *,
    repository_root: Path,
    server_config: Mapping[str, Any] | None = None,
) -> None:
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported decision-characterization manifest schema")
    if not payload.get("frozen"):
        raise ValueError("decision-characterization manifest is not frozen")
    if payload.get("manifest_payload_sha256") != _canonical_digest(payload):
        raise ValueError("decision-characterization payload digest mismatch")
    selection = payload.get("selection_contract", {})
    if selection.get("adaptive_trace_selection") is not False:
        raise ValueError("adaptive trace selection must be disabled")
    if selection.get("sealed_test_id_opened") is not False:
        raise ValueError("sealed test-ID must remain closed")
    collection = payload.get("collection", {})
    for path_key, digest_key in (
        ("plan_path", "plan_sha256"),
        ("workload_manifest_path", "workload_manifest_sha256"),
    ):
        path = Path(str(collection[path_key]))
        if sha256_file(path) != collection[digest_key]:
            raise ValueError(f"frozen collection input changed: {path}")
    for artifact in payload.get("artifacts", {}).values():
        path = Path(str(artifact["path"]))
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"frozen model artifact changed: {path}")
    current_source = source_fingerprint(repository_root.expanduser().resolve())
    if current_source != payload.get("runtime_source_fingerprint"):
        raise ValueError("runtime source changed after characterization freeze")
    if payload.get("target_project_overlap_with_predictor_fit_projects"):
        raise ValueError("target project was used to fit the frozen predictor")
    if server_config is None:
        return
    for key, expected in SERVER_CONFIG_CONTRACT.items():
        if server_config.get(key) != expected:
            raise ValueError(f"server config violates {key}={expected!r}")
    artifact_fields = {
        "predictor_model_path": "predictor",
        "gpu_service_model_path": "gpu_service",
        "transfer_service_model_path": "transfer_service",
    }
    for config_key, artifact_key in artifact_fields.items():
        actual = Path(str(server_config.get(config_key))).expanduser().resolve()
        expected = Path(
            str(payload["artifacts"][artifact_key]["path"])
        ).expanduser().resolve()
        if actual != expected:
            raise ValueError(f"server config changed frozen {artifact_key} artifact")
    risk_fields = {
        "predictive_risk_particle_count": "particle_count",
        "predictive_risk_top_k": "top_k",
        "predictive_risk_max_candidates": "max_candidates",
        "predictive_risk_min_calibration_coverage": "min_calibration_coverage",
        "predictive_commit_guard_ms": "commit_guard_ms",
        "predictive_prefetch_min_hbm_feasibility": (
            "prefetch_min_hbm_feasibility"
        ),
        "predictive_prefetch_desired_lead_ms": "prefetch_desired_lead_ms",
        "predictive_prepare_host_enabled": "prepare_host_enabled",
    }
    for config_key, policy_key in risk_fields.items():
        if server_config.get(config_key) != payload["risk_policy"][policy_key]:
            raise ValueError(f"server config changed frozen risk field {config_key}")
    if int(server_config.get("joint_workflow_active_window", -1)) != int(
        payload["runtime_policy"]["joint_workflow_active_window"]
    ):
        raise ValueError("server config changed frozen active workflow window")


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
