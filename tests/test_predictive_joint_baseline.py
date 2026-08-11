from __future__ import annotations

import json
from pathlib import Path

import pytest

from beliefkv.experiments.predictive_joint_baseline import (
    build_baseline_manifest,
    source_tree_fingerprint,
    validate_baseline_manifest,
    validate_source_tree_fingerprint,
    write_baseline_manifest,
)


def test_baseline_freezes_every_external_input(tmp_path: Path, monkeypatch) -> None:
    files = [tmp_path / name for name in ("workload.json", "p.json", "g.json", "t.json")]
    for index, path in enumerate(files):
        path.write_text(json.dumps({"index": index}), encoding="utf-8")
    monkeypatch.setattr(
        "beliefkv.experiments.predictive_joint_baseline.git_head",
        lambda _root: "a" * 40,
    )
    monkeypatch.setattr(
        "beliefkv.experiments.predictive_joint_baseline.source_tree_fingerprint",
        lambda _root: {
            "roots": ["beliefkv", "scripts", "patches"],
            "suffixes": [".patch", ".py", ".sh"],
            "file_count": 1,
            "sha256": "b" * 64,
        },
    )
    payload = build_baseline_manifest(
        repository_root=tmp_path,
        workload_manifest=files[0],
        predictor_artifact=files[1],
        gpu_service_artifact=files[2],
        transfer_service_artifact=files[3],
        model_path=tmp_path / "model",
        gpu_name="gpu",
        gpu_index=0,
        kv_pool_tokens=10,
        host_cache_gib=96,
        max_running_requests=4,
        context_length=32_768,
        concurrency=2,
        random_seed=7,
        arrival_interval_ms=100,
    )
    validate_baseline_manifest(payload)
    output = tmp_path / "baseline.json"
    write_baseline_manifest(output, payload)
    assert output.is_file()
    with pytest.raises(FileExistsError):
        write_baseline_manifest(output, payload)
    files[1].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen artifact changed"):
        validate_baseline_manifest(payload)


def test_source_tree_fingerprint_detects_code_changes(tmp_path: Path) -> None:
    for root_name in ("beliefkv", "scripts", "patches"):
        (tmp_path / root_name).mkdir()
    source = tmp_path / "beliefkv/runtime.py"
    source.write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "beliefkv/ignored.txt").write_text("a", encoding="utf-8")
    frozen = source_tree_fingerprint(tmp_path)

    validate_source_tree_fingerprint(frozen, tmp_path)
    (tmp_path / "beliefkv/ignored.txt").write_text("b", encoding="utf-8")
    validate_source_tree_fingerprint(frozen, tmp_path)
    source.write_text("value = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source tree changed"):
        validate_source_tree_fingerprint(frozen, tmp_path)
