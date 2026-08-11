import hashlib
import json
from pathlib import Path

import pytest

from beliefkv.experiments.predictive_canary import (
    ARM_SERVER_CONTRACTS,
    COMMON_SERVER_CONTRACT,
    build_veto_canary_manifest,
    validate_veto_canary_manifest,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_veto_canary_requires_directional_decision_flip(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    _write_json(
        comparison,
        {
            "shape_veto_gate": False,
            "supported_shape_veto_gate": False,
            "shape_action_gate": False,
            "selected_action_gate": True,
        },
    )
    characterization = tmp_path / "characterization.json"
    _write_json(characterization, {})

    with pytest.raises(ValueError, match="did not establish"):
        build_veto_canary_manifest(
            repository_root=Path(__file__).resolve().parents[1],
            study_id="test",
            characterization_manifest=characterization,
            replay_comparison=comparison,
            harness_profiles=tmp_path / "profiles.json",
        )


def test_arm_server_contracts_keep_veto_treatment_bounded(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    inputs = tmp_path / "inputs"
    plan = inputs / "plan.json"
    workload = inputs / "workload.json"
    profiles = inputs / "profiles.json"
    comparison = inputs / "comparison.json"
    predictor = inputs / "predictor.json"
    gpu = inputs / "gpu.json"
    transfer = inputs / "transfer.json"
    for path in (plan, workload, profiles, predictor, gpu, transfer):
        _write_json(path, {"schema_version": 1})
    _write_json(
        comparison,
        {
            "shape_veto_gate": True,
            "supported_shape_veto_gate": True,
            "shape_action_gate": False,
            "selected_action_gate": True,
        },
    )
    characterization = inputs / "characterization.json"
    _write_json(
        characterization,
        {
            "collection": {
                "plan_path": str(plan),
                "batch_id": "batch",
                "workload_manifest_path": str(workload),
                "workflow_count": 1,
                "concurrency": 1,
                "instance_ids": ["task"],
                "arrival_policy": "single",
            },
            "artifacts": {
                name: {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for name, path in (
                    ("predictor", predictor),
                    ("gpu_service", gpu),
                    ("transfer_service", transfer),
                )
            },
            "risk_policy": {},
            "runtime_policy": {},
        },
    )
    payload = build_veto_canary_manifest(
        repository_root=repository_root,
        study_id="test-veto",
        characterization_manifest=characterization,
        replay_comparison=comparison,
        harness_profiles=profiles,
    )
    treatment = {
        **COMMON_SERVER_CONTRACT,
        **ARM_SERVER_CONTRACTS["byte_only_veto_treatment"],
        "predictor_model_path": str(predictor),
        "gpu_service_model_path": str(gpu),
        "transfer_service_model_path": str(transfer),
    }
    validate_veto_canary_manifest(
        payload,
        repository_root=repository_root,
        arm="byte_only_veto_treatment",
        server_config=treatment,
    )

    treatment["predictive_prepare_host_canary_limit"] = 2
    with pytest.raises(ValueError, match="canary_limit"):
        validate_veto_canary_manifest(
            payload,
            repository_root=repository_root,
            arm="byte_only_veto_treatment",
            server_config=treatment,
        )
