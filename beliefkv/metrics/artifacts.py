from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from beliefkv import __version__
from beliefkv.core.config import BeliefKVConfig
from beliefkv.simulator.page_simulator import SimulationResult


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at_utc: str
    workload: str
    seed: int
    beliefkv_version: str
    git_commit: str
    git_dirty: bool
    python_version: str
    platform: str
    gpu: str
    config_sha256: str
    config: dict[str, Any]
    metadata: dict[str, Any]


class ExperimentArtifactWriter:
    """Write immutable, self-describing replay artifacts."""

    def __init__(self, repository_root: Path, output_root: Path) -> None:
        self.repository_root = repository_root.resolve()
        self.output_root = output_root.resolve()
        self._cached_git_state: tuple[str, bool] | None = None
        self._cached_gpu_name: str | None = None

    def write(
        self,
        *,
        run_id: str,
        workload: str,
        seed: int,
        config: BeliefKVConfig,
        result: SimulationResult,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        if not run_id or any(character in run_id for character in "/\\\0"):
            raise ValueError("run_id must be a non-empty path-safe name")
        self.output_root.mkdir(parents=True, exist_ok=True)
        run_dir = self.output_root / run_id
        staging_dir = self.output_root / f".{run_id}.incomplete"
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        if staging_dir.exists():
            raise FileExistsError(
                f"incomplete run directory already exists: {staging_dir}"
            )
        staging_dir.mkdir()
        config_payload = config.to_dict()
        config_json = json.dumps(config_payload, sort_keys=True, separators=(",", ":"))
        commit, dirty = self._git_state()
        manifest = RunManifest(
            run_id=run_id,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            workload=workload,
            seed=seed,
            beliefkv_version=__version__,
            git_commit=commit,
            git_dirty=dirty,
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            gpu=self._gpu_name(),
            config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
            config=config_payload,
            metadata=metadata or {},
        )
        self._atomic_json(staging_dir / "manifest.json", asdict(manifest))
        self._atomic_json(staging_dir / "summary.json", result.summary)
        self._atomic_json(staging_dir / "final_graph.json", result.final_graph)
        self._atomic_jsonl(staging_dir / "events.jsonl", result.event_log)
        staging_dir.replace(run_dir)
        return run_dir

    def _git_state(self) -> tuple[str, bool]:
        if self._cached_git_state is not None:
            return self._cached_git_state
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            result = (commit, bool(status.strip()))
        except (OSError, subprocess.CalledProcessError):
            result = ("unknown", True)
        self._cached_git_state = result
        return result

    def _gpu_name(self) -> str:
        if self._cached_gpu_name is not None:
            return self._cached_gpu_name
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            result = output.splitlines()[0] if output else "unknown"
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            result = "unknown"
        self._cached_gpu_name = result
        return result

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False))
                stream.write("\n")
        temporary.replace(path)
