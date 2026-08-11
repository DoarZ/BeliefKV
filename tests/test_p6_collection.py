from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from beliefkv.experiments.p6_collection import load_collection_batch
from scripts.run_p6_collection_batch import (
    _actual_kv_pool_tokens,
    _materialize_runtime_workload_manifest,
)


def test_actual_kv_pool_tokens_uses_server_report(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_urls: list[str] = []

    def fake_urlopen(url: str, *, timeout: float) -> io.BytesIO:
        requested_urls.append(url)
        assert timeout == 3.0
        return io.BytesIO(b'{"max_total_num_tokens": 167816}')

    monkeypatch.setattr(
        "scripts.run_p6_collection_batch.urllib.request.urlopen",
        fake_urlopen,
    )

    assert _actual_kv_pool_tokens("http://127.0.0.1:18000/v1", timeout_s=3.0) == 167816
    assert requested_urls == ["http://127.0.0.1:18000/get_server_info"]


def _write_fixture(tmp_path: Path, *, split: str = "train") -> Path:
    source = tmp_path / "source"
    (source / ".git").mkdir(parents=True)
    manifest = tmp_path / "workloads.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "dataset_revision": "v1",
                "workloads": [
                    {
                        "instance_id": "repo__task-1",
                        "repo": "org/repo",
                        "base_commit": "abc",
                        "problem_statement": "fix it",
                        "source_repo": str(source),
                        "docker_image": "fixture:latest",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "frozen": True,
                "plan_id": "fixture-plan",
                "predictor_enabled": False,
                "predictive_actions_enabled": False,
                "runtime_policy": "frozen_p5_observed",
                "batches": [
                    {
                        "batch_id": "batch-1",
                        "split": split,
                        "policy": "frozen_p5_observed",
                        "predictive_actions": False,
                        "projects": ["org/repo"],
                        "docker_images": ["fixture:latest"],
                        "workload_manifest": str(manifest),
                        "workload_manifest_sha256": digest,
                        "workflow_count": 1,
                        "concurrency": 1,
                        "preflight_command": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return plan


def test_load_collection_train_batch(tmp_path: Path) -> None:
    batch = load_collection_batch(_write_fixture(tmp_path), "batch-1")
    assert batch.split == "train"
    assert batch.workflow_count == 1
    assert batch.preflight_command is None


def test_collection_batch_keeps_calibration_and_test_sealed(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="calibration"):
        load_collection_batch(
            _write_fixture(tmp_path / "cal", split="calibration"), "batch-1"
        )
    with pytest.raises(PermissionError, match="sealed test"):
        load_collection_batch(
            _write_fixture(tmp_path / "test", split="test_id"), "batch-1"
        )


def test_collection_batch_rejects_manifest_mutation(tmp_path: Path) -> None:
    plan = _write_fixture(tmp_path)
    raw = json.loads(plan.read_text(encoding="utf-8"))
    Path(raw["batches"][0]["workload_manifest"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_collection_batch(plan, "batch-1")


def test_runtime_manifest_applies_instance_scoped_harness_profile(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "workloads": [
                    {
                        "instance_id": "psf__requests-5414",
                        "repo": "psf/requests",
                        "docker_image": "source:latest",
                    },
                    {
                        "instance_id": "psf__requests-1142",
                        "repo": "psf/requests",
                        "docker_image": "source-legacy:latest",
                        "preflight_command": "legacy repository-wide check",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "fixture",
                "instances": {
                    "psf__requests-5414": {
                        "repo": "psf/requests",
                        "source_image": "source:latest",
                        "runtime_image": "runtime:harness",
                        "preflight_policy": "psf_requests_pytest_httpbin_v1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    path, applied, count = _materialize_runtime_workload_manifest(
        source_path=source,
        destination=tmp_path / "runtime.json",
        profile_path=profiles,
        selected_instance_ids=None,
    )

    runtime = json.loads(path.read_text(encoding="utf-8"))
    assert count == 2
    assert runtime["workloads"][0]["docker_image"] == "runtime:harness"
    assert "pytest_httpbin" in runtime["workloads"][0]["preflight_command"]
    assert "preflight_command" not in runtime["workloads"][1]
    assert applied[0]["preflight_policy"] == "psf_requests_pytest_httpbin_v1"


def test_runtime_manifest_rejects_unknown_harness_preflight_policy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "workloads": [
                    {
                        "instance_id": "repo__task-1",
                        "repo": "org/repo",
                        "docker_image": "source:latest",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "fixture",
                "instances": {
                    "repo__task-1": {
                        "repo": "org/repo",
                        "source_image": "source:latest",
                        "runtime_image": "runtime:harness",
                        "preflight_policy": "unknown-v1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown harness preflight policy"):
        _materialize_runtime_workload_manifest(
            source_path=source,
            destination=tmp_path / "runtime.json",
            profile_path=profiles,
            selected_instance_ids=None,
        )
