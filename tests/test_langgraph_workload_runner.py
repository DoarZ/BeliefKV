from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


pytest.importorskip("langchain")

from scripts.run_langgraph_peer_workloads import (
    _collect_workspace_artifacts,
    _initial_spawn_range_valid,
)


def test_spawn_range_uses_initial_batch_not_total_dynamic_children() -> None:
    assert _initial_spawn_range_valid(
        required=True,
        observed_initial_subagents=4,
        minimum=2,
        maximum=4,
    )
    assert not _initial_spawn_range_valid(
        required=True,
        observed_initial_subagents=6,
        minimum=2,
        maximum=4,
    )
    assert _initial_spawn_range_valid(
        required=False,
        observed_initial_subagents=None,
        minimum=0,
        maximum=0,
    )


def test_workspace_artifact_timeout_is_recorded_without_raising(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    destination = tmp_path / "output"
    workspace.mkdir()
    destination.mkdir()

    def fake_runner(command, *, cwd):
        assert cwd == workspace
        if "diff" in command:
            raise subprocess.TimeoutExpired(command, 60.0)
        return " M module.py"

    result = _collect_workspace_artifacts(
        workspace,
        destination,
        command_runner=fake_runner,
    )

    assert not result["artifact_collection_valid"]
    assert result["workspace_modified"]
    assert result["patch_chars"] == 0
    assert "TimeoutExpired" in result["artifact_collection_errors"][0]
    assert (destination / "model.patch").read_text(encoding="utf-8") == ""


def test_workspace_status_failure_preserves_collected_patch(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    destination = tmp_path / "output"
    workspace.mkdir()
    destination.mkdir()

    def fake_runner(command, *, cwd):
        assert cwd == workspace
        if "status" in command:
            raise RuntimeError("status unavailable")
        return "diff --git a/module.py b/module.py"

    result = _collect_workspace_artifacts(
        workspace,
        destination,
        command_runner=fake_runner,
    )

    assert not result["artifact_collection_valid"]
    assert not result["workspace_modified"]
    assert result["patch_chars"] > 0
    assert "status unavailable" in result["artifact_collection_errors"][0]
    assert (destination / "model.patch").read_text(encoding="utf-8").endswith(
        "\n"
    )
