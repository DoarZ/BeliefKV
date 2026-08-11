from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
SOURCE_ROOTS = ("beliefkv", "scripts", "patches")
SOURCE_SUFFIXES = frozenset({".py", ".sh", ".patch"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locked_file(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_head(repository_root: Path) -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        text=True,
    ).strip()


def source_tree_fingerprint(repository_root: Path) -> dict[str, Any]:
    repository_root = repository_root.expanduser().resolve()
    records: list[tuple[str, str]] = []
    for root_name in SOURCE_ROOTS:
        root = repository_root / root_name
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            relative = path.relative_to(repository_root).as_posix()
            records.append((relative, sha256_file(path)))
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "roots": list(SOURCE_ROOTS),
        "suffixes": sorted(SOURCE_SUFFIXES),
        "file_count": len(records),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def validate_source_tree_fingerprint(
    expected: Mapping[str, Any], repository_root: Path
) -> None:
    actual = source_tree_fingerprint(repository_root)
    if dict(expected) != actual:
        raise ValueError(
            "frozen predictive JointPlan source tree changed: "
            f"expected={expected.get('sha256')} actual={actual['sha256']}"
        )


def build_baseline_manifest(
    *,
    repository_root: Path,
    workload_manifest: Path,
    predictor_artifact: Path,
    gpu_service_artifact: Path,
    transfer_service_artifact: Path,
    model_path: Path,
    gpu_name: str,
    gpu_index: int,
    kv_pool_tokens: int,
    host_cache_gib: int,
    max_running_requests: int,
    context_length: int,
    concurrency: int,
    random_seed: int,
    arrival_interval_ms: int,
) -> dict[str, Any]:
    if min(
        kv_pool_tokens,
        host_cache_gib,
        max_running_requests,
        context_length,
        concurrency,
    ) <= 0:
        raise ValueError("baseline capacities must be positive")
    if min(random_seed, arrival_interval_ms, gpu_index) < 0:
        raise ValueError("baseline seed, arrival interval and GPU index are non-negative")
    repository_root = repository_root.expanduser().resolve()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "frozen": True,
        "p5_base_commit": git_head(repository_root),
        "source_tree": source_tree_fingerprint(repository_root),
        "artifacts": {
            "predictor": _locked_file(predictor_artifact),
            "gpu_service": _locked_file(gpu_service_artifact),
            "transfer_service": _locked_file(transfer_service_artifact),
        },
        "workload": {
            "manifest": _locked_file(workload_manifest),
            "fanout_profile": "parallel_analysis_2to3",
            "concurrency": concurrency,
            "arrival_schedule": {
                "policy": "fixed_interval",
                "interval_ms": arrival_interval_ms,
            },
            "random_seed": random_seed,
        },
        "runtime": {
            "gpu_name": gpu_name,
            "gpu_index": gpu_index,
            "model": str(model_path.expanduser().resolve()),
            "kv_pool_tokens": kv_pool_tokens,
            "host_cache_gib": host_cache_gib,
            "max_running_requests": max_running_requests,
            "context_length": context_length,
            "sglang_version": "0.5.2rc1",
        },
        "arms": {
            "p5_observed": {
                "predictive_prepare": False,
                "frontier_retraction": False,
            },
            "predictive_joint": {
                "predictive_prepare": True,
                "predictive_prepare_limit": 1,
                "frontier_retraction": True,
            },
        },
        "selection_contract": {
            "same_tasks_and_arrivals": True,
            "adaptive_trace_selection": False,
            "paired_run_count": 3,
            "paired_order": ["A-B", "B-A", "A-B"],
        },
    }
    payload["manifest_sha256"] = _digest(payload)
    validate_baseline_manifest(payload)
    return payload


def validate_baseline_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("frozen") is not True:
        raise ValueError("unsupported or unfrozen predictive JointPlan baseline")
    if payload.get("manifest_sha256") != _digest(payload):
        raise ValueError("predictive JointPlan baseline digest mismatch")
    source_tree = payload.get("source_tree")
    if not isinstance(source_tree, Mapping):
        raise ValueError("predictive JointPlan baseline omits source tree")
    for artifact in payload["artifacts"].values():
        path = Path(str(artifact["path"]))
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"frozen artifact changed: {path}")
    workload = payload["workload"]["manifest"]
    workload_path = Path(str(workload["path"]))
    if sha256_file(workload_path) != workload["sha256"]:
        raise ValueError(f"frozen workload changed: {workload_path}")
    selection = payload["selection_contract"]
    if selection.get("adaptive_trace_selection") is not False:
        raise ValueError("adaptive trace selection is forbidden")
    if selection.get("same_tasks_and_arrivals") is not True:
        raise ValueError("paired arms must share tasks and arrivals")


def write_baseline_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"baseline manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
