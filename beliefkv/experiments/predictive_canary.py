from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from beliefkv.experiments.decision_characterization import (
    sha256_file,
    source_fingerprint,
)


SCHEMA_VERSION = 1
COMMON_SERVER_CONTRACT = {
    "predictive_risk_shadow_enabled": True,
    "predictive_prefetch_canary_enabled": False,
    "joint_policy_enabled": True,
    "joint_observed_mode_enabled": True,
    "running_batch_retraction_enabled": True,
}
ARM_SERVER_CONTRACTS = {
    "p5_observed_control": {
        "predictive_joint_overlay_enabled": False,
        "predictive_transfer_model_mode": "morphology-aware",
        "predictive_prepare_authority_gate": "natural",
        "predictive_prepare_host_canary_limit": 0,
    },
    "byte_only_veto_treatment": {
        "predictive_joint_overlay_enabled": True,
        "predictive_transfer_model_mode": "byte-only",
        "predictive_prepare_authority_gate": "byte-only-veto",
        "predictive_prepare_host_canary_limit": 1,
    },
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


def _frozen_file(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def build_veto_canary_manifest(
    *,
    repository_root: Path,
    study_id: str,
    characterization_manifest: Path,
    replay_comparison: Path,
    harness_profiles: Path,
) -> dict[str, Any]:
    repository_root = repository_root.expanduser().resolve()
    characterization_manifest = characterization_manifest.expanduser().resolve()
    characterization = json.loads(
        characterization_manifest.read_text(encoding="utf-8")
    )
    comparison = json.loads(replay_comparison.read_text(encoding="utf-8"))
    if comparison.get("supported_shape_veto_gate") is not True:
        raise ValueError(
            "paired replay did not establish a shape-supported morphology veto"
        )
    if comparison.get("shape_action_gate") is not False:
        raise ValueError("veto canary cannot reuse a morphology-promotion result")
    if comparison.get("selected_action_gate") is not True:
        raise ValueError("paired replay did not change the selected action")

    collection = characterization["collection"]
    artifacts = characterization["artifacts"]
    runtime_policy = dict(characterization["runtime_policy"])
    runtime_policy.pop("predictive_actions_enabled", None)
    runtime_policy["mode"] = "paired_veto_canary"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "frozen": True,
        "selection_contract": {
            "adaptive_trace_selection": False,
            "same_workload_and_arrival_for_both_arms": True,
            "maximum_predictive_prepare_actions": 1,
            "no_threshold_changes_after_freeze": True,
            "no_forced_or_injected_intent": True,
            "no_predictive_prefetch_or_retraction": True,
        },
        "evidence": {
            "characterization_manifest": _frozen_file(
                characterization_manifest
            ),
            "paired_replay_comparison": _frozen_file(replay_comparison),
            "shape_veto_gate": True,
            "supported_shape_veto_gate": True,
            "shape_action_gate": False,
            "selected_action_gate": True,
            "recommended_validation_arm": "byte_only_veto_treatment",
        },
        "collection": {
            "plan": _frozen_file(Path(str(collection["plan_path"]))),
            "batch_id": str(collection["batch_id"]),
            "workload_manifest": _frozen_file(
                Path(str(collection["workload_manifest_path"]))
            ),
            "harness_profiles": _frozen_file(harness_profiles),
            "workflow_count": int(collection["workflow_count"]),
            "concurrency": int(collection["concurrency"]),
            "instance_ids": list(collection["instance_ids"]),
            "arrival_policy": str(collection["arrival_policy"]),
        },
        "artifacts": artifacts,
        "risk_policy": dict(characterization["risk_policy"]),
        "runtime_policy": runtime_policy,
        "arms": {
            name: dict(contract)
            for name, contract in ARM_SERVER_CONTRACTS.items()
        },
        "runtime_source_fingerprint": source_fingerprint(repository_root),
    }
    payload["manifest_payload_sha256"] = _canonical_digest(payload)
    validate_veto_canary_manifest(payload, repository_root=repository_root)
    return payload


def validate_veto_canary_manifest(
    payload: Mapping[str, Any],
    *,
    repository_root: Path,
    arm: str | None = None,
    server_config: Mapping[str, Any] | None = None,
) -> None:
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported veto-canary manifest schema")
    if payload.get("frozen") is not True:
        raise ValueError("veto-canary manifest is not frozen")
    if payload.get("manifest_payload_sha256") != _canonical_digest(payload):
        raise ValueError("veto-canary manifest digest mismatch")
    selection = payload.get("selection_contract", {})
    required_true = (
        "same_workload_and_arrival_for_both_arms",
        "no_threshold_changes_after_freeze",
        "no_forced_or_injected_intent",
        "no_predictive_prefetch_or_retraction",
    )
    if selection.get("adaptive_trace_selection") is not False:
        raise ValueError("adaptive trace selection must remain disabled")
    if any(selection.get(key) is not True for key in required_true):
        raise ValueError("veto-canary selection contract is incomplete")
    if int(selection.get("maximum_predictive_prepare_actions", -1)) != 1:
        raise ValueError("veto canary must allow at most one predictive prepare")

    evidence = payload.get("evidence", {})
    if not (
        evidence.get("shape_veto_gate") is True
        and evidence.get("supported_shape_veto_gate") is True
        and evidence.get("shape_action_gate") is False
        and evidence.get("selected_action_gate") is True
    ):
        raise ValueError("veto-canary evidence direction changed")
    frozen_files = [
        evidence["characterization_manifest"],
        evidence["paired_replay_comparison"],
        payload["collection"]["plan"],
        payload["collection"]["workload_manifest"],
        payload["collection"]["harness_profiles"],
        *payload.get("artifacts", {}).values(),
    ]
    for item in frozen_files:
        path = Path(str(item["path"]))
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"frozen veto-canary input changed: {path}")
    current_source = source_fingerprint(repository_root.expanduser().resolve())
    if current_source != payload.get("runtime_source_fingerprint"):
        raise ValueError("runtime source changed after veto-canary freeze")

    if server_config is None:
        if arm is not None:
            raise ValueError("arm validation requires a server config")
        return
    if arm not in ARM_SERVER_CONTRACTS:
        raise ValueError(f"unknown veto-canary arm: {arm}")
    expected_contract = {
        **COMMON_SERVER_CONTRACT,
        **ARM_SERVER_CONTRACTS[arm],
    }
    for key, expected in expected_contract.items():
        if server_config.get(key) != expected:
            raise ValueError(
                f"server config violates {arm} contract: {key}={expected!r}"
            )
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


def write_veto_canary_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
