from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from deepagents.graph import BASE_AGENT_PROMPT
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from beliefkv.experiments.agent_protocol import (
    AgentLoopGuardMiddleware,
    ChildCompletion,
    LoopGuardPolicy,
    WorkflowCompletion,
    require_structured_completion,
)
from beliefkv.runtime.deepagents_adapter import (
    BeliefKVChatOpenAI,
    DeclaredRuntimeTask,
    DeepAgentsRuntimeAdapter,
)
from beliefkv.runtime.event_channel import (
    JsonlRuntimeEventSink,
    UnixDatagramRuntimeEventSink,
)
from beliefkv.runtime.langchain_tool_safety import (
    ToolCircuitBreakerMiddleware,
    ToolObservationBudgetMiddleware,
    ToolObservationBudgetPolicy,
    ToolOutcomeStatusMiddleware,
)
from beliefkv.runtime.context_lifecycle import (
    CONTEXT_LIFECYCLE_PRIVATE_STATE_KEYS,
    CompletionBudgetMiddleware,
    ContextLifecycleMiddleware,
    ContextLifecyclePolicy,
)
from beliefkv.runtime.sglang_adapter import BeliefKVRequestMetadata
from beliefkv.runtime.subagent_state import PrivateStateIsolatingSubAgentMiddleware


SERVER_ARTIFACT_FILENAMES = {
    "runtime_audit": "runtime_audit.jsonl",
    "runtime_events": "runtime_events.sglang.jsonl",
    "server_log": "server.log",
}

DEFAULT_SANDBOX_TEST_ENV = "/opt/miniconda3/envs/testbed"
DEFAULT_SANDBOX_SUPPORT_DIR = Path(__file__).with_name("sandbox_support")
SANDBOX_PATH_CONTRACT = """
Sandbox path and environment contract:
- The repository checkout root is exactly `/workspace`. Filesystem tools and execute
  share this namespace, and execute starts in `/workspace`.
- Repository paths are relative to `/workspace`. Do not prepend the repository owner,
  repository name, or package name unless that directory actually appears in the
  checkout. Never invent another virtual repository root.
- Before retrying a missing path, inspect `/workspace` or run `pwd` and
  `git rev-parse --show-toplevel`; do not repeatedly guess path prefixes.
- `python`, `pytest`, and other Python entry points already resolve to the image's
  prebuilt test environment. Do not install or upgrade packages and do not use network
  package managers.
- Diagnostic `python -c` probes do not count as tests. Discover and run the focused
  repository-native test command described by the workload-specific contract. Treat
  both the exit status and the executed-test count as authoritative.
"""
TEST_COMMAND_PATTERN = re.compile(
    r"(?:^|[;&|]\s*)(?:python\s+(?:-m\s+pytest|bin/test)\b|pytest\b|"
    r"py\.test\b|tox\b|make\s+(?:test|check)\b)"
)
ZERO_TEST_OUTPUT_PATTERN = re.compile(
    r"(?:\b0\s+(?:tests?\s+(?:collected|executed|run)|passed)\b|"
    r"\bcollected\s+0\s+items?\b|\bno\s+tests?\s+(?:ran|were\s+run)\b)",
    re.IGNORECASE,
)
UNSUPPORTED_SYMPY_TEST_SELECTOR_PATTERN = re.compile(
    r"\bpython\s+bin/test\b[^\n;&|]*::"
)
INCOMPLETE_SUMMARY_PATTERN = re.compile(
    r"\b(?:not implemented|requires additional (?:implementation|work)|"
    r"only addresses?)\b",
    re.IGNORECASE,
)
RUNTIME_VERIFIED_TESTS_KEY = "_beliefkv_runtime_verified_tests"

SYMPY_SANDBOX_PREFLIGHT = "python -c " + shlex.quote(
    "import collections, collections.abc, os, mpmath, sympy; "
    "assert collections.__dict__.get('Mapping') is collections.abc.Mapping; "
    "origin = os.path.realpath(sympy.__file__); "
    "assert os.path.commonpath(('/workspace', origin)) == '/workspace', origin"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, sort_keys=True, allow_nan=False, default=str)
                + "\n"
            )
    temporary.replace(path)


def capture_append_offset(path: Path) -> int:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"server artifact is absent: {path}")
    return path.stat().st_size


def copy_append_window(
    source: Path,
    destination: Path,
    *,
    start_offset: int,
) -> dict[str, Any]:
    """Freeze bytes appended to a line-buffered server artifact during one run."""

    source = source.expanduser().resolve()
    end_offset = source.stat().st_size
    if start_offset < 0 or end_offset < start_offset:
        raise ValueError(
            f"invalid append window for {source}: {start_offset}..{end_offset}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    byte_count = end_offset - start_offset
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        input_stream.seek(start_offset)
        remaining = byte_count
        while remaining:
            chunk = input_stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(
                    f"server artifact truncated while freezing window: {source}"
                )
            output_stream.write(chunk)
            remaining -= len(chunk)
    return {
        "source_path": str(source),
        "path": str(destination),
        "start_offset": start_offset,
        "end_offset": end_offset,
        "byte_count": byte_count,
        "sha256": sha256(destination),
    }


def command_output(command: Sequence[str], *, cwd: Path, timeout: float = 60.0) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"command failed ({result.returncode}): {detail}")
    return result.stdout.strip()


@dataclass(frozen=True)
class SweBenchWorkload:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    difficulty: str
    source_repo: Path | None = None
    docker_image: str | None = None
    preflight_command: str | None = None


def repository_sandbox_contract(workload: SweBenchWorkload) -> str:
    """Describe the mounted checkout without leaking another repository's layout."""

    common = f"""
Workload-specific repository contract:
- Repository identity is `{workload.repo}`, but its checkout root is still exactly
  `/workspace`. Do not append `{workload.repo}` to `/workspace`.
"""
    if workload.repo == "django/django":
        return common + """- Django source paths such as `django/db/backends/base/base.py` map to
  `/workspace/django/db/backends/base/base.py`, not
  `/workspace/django/django/db/backends/base/base.py`.
- Run focused Django tests from `/workspace` with the repository test runner, for
  example `python tests/runtests.py <test_label>`.
"""
    if workload.repo == "sympy/sympy":
        return common + """- SymPy source paths such as `sympy/core/basic.py` map to
  `/workspace/sympy/core/basic.py`, not `/workspace/sympy/sympy/core/basic.py`.
- Run focused SymPy tests with `python bin/test <test-path>`. This runner accepts
  `-k <test-name>` or a complete test-file path; do not use a pytest-style `::` selector.
"""
    return common + """- Inspect the checkout's top-level files before choosing a package path or test
  command. Prefer paths returned by repository search tools over paths inferred from
  the repository slug.
"""


@dataclass(frozen=True)
class WorkloadBundle:
    manifest_path: Path
    manifest_sha256: str
    dataset: str
    dataset_revision: str
    source_repo: Path | None
    workloads: tuple[SweBenchWorkload, ...]


def load_workload_bundle(path: Path) -> WorkloadBundle:
    path = path.expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("workloads"), list):
        raise ValueError("workload manifest must contain a workloads list")
    source_repo = (
        Path(str(raw["source_repo"])).expanduser().resolve()
        if raw.get("source_repo")
        else None
    )
    if source_repo is not None and not (source_repo / ".git").exists():
        raise FileNotFoundError(f"source repository is absent: {source_repo}")
    workloads = tuple(
        SweBenchWorkload(
            instance_id=str(item["instance_id"]),
            repo=str(item["repo"]),
            base_commit=str(item["base_commit"]),
            problem_statement=str(item["problem_statement"]),
            difficulty=str(item.get("difficulty", "unknown")),
            source_repo=(
                Path(str(item["source_repo"])).expanduser().resolve()
                if item.get("source_repo")
                else source_repo
            ),
            docker_image=(
                str(item["docker_image"]) if item.get("docker_image") else None
            ),
            preflight_command=(
                str(item["preflight_command"])
                if item.get("preflight_command")
                else None
            ),
        )
        for item in raw["workloads"]
    )
    if len({item.instance_id for item in workloads}) != len(workloads):
        raise ValueError("workload instance IDs must be unique")
    missing_sources = [
        item.instance_id
        for item in workloads
        if item.source_repo is None or not (item.source_repo / ".git").exists()
    ]
    if missing_sources:
        raise FileNotFoundError(
            f"workload source repositories are absent: {missing_sources}"
        )
    return WorkloadBundle(
        manifest_path=path,
        manifest_sha256=sha256(path),
        dataset=str(raw.get("dataset", "unknown")),
        dataset_revision=str(raw.get("dataset_revision", "unknown")),
        source_repo=source_repo,
        workloads=workloads,
    )


def prepare_workspace(
    source_repo: Path,
    workload: SweBenchWorkload,
    destination: Path,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"workspace already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    clone = subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--shared",
            "--no-checkout",
            str(source_repo),
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180.0,
    )
    if clone.returncode != 0:
        raise RuntimeError(f"workspace clone failed: {clone.stderr.strip()}")
    command_output(
        ["git", "checkout", "--quiet", "--detach", workload.base_commit],
        cwd=destination,
    )
    head = command_output(["git", "rev-parse", "HEAD"], cwd=destination)
    status = command_output(["git", "status", "--porcelain"], cwd=destination)
    if head != workload.base_commit or status:
        raise RuntimeError(
            f"workspace identity mismatch: head={head}, status={status!r}"
        )
    return {
        "source_repo": str(source_repo),
        "base_commit": workload.base_commit,
        "workspace": str(destination),
        "initial_head": head,
        "initial_status": status,
        "isolation": "per-workflow-shared-local-clone",
    }


class JsonlAudit:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("x", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._sequence = 0

    def emit(self, event: str, **fields: Any) -> None:
        with self._lock:
            self._sequence += 1
            payload = {
                "schema_version": 1,
                "sequence": self._sequence,
                "ts_ms": time.monotonic() * 1000.0,
                "event": event,
                **fields,
            }
            self._stream.write(
                json.dumps(payload, sort_keys=True, allow_nan=False, default=str)
                + "\n"
            )

    def close(self) -> None:
        self._stream.close()


class DockerWorkspaceBackend(FilesystemBackend, SandboxBackendProtocol):
    """Host-scoped files plus shell execution in a restricted Docker sandbox."""

    shell_workdir = "/workspace"

    def __init__(
        self,
        workspace: Path,
        *,
        image: str,
        audit: JsonlAudit,
        cpus: float = 2.0,
        memory_gib: float = 6.0,
        default_timeout_s: int = 180,
        max_output_chars: int = 100_000,
        test_env_path: str = DEFAULT_SANDBOX_TEST_ENV,
        preflight_command: str | None = None,
        support_dir: Path | None = DEFAULT_SANDBOX_SUPPORT_DIR,
    ) -> None:
        super().__init__(root_dir=workspace, virtual_mode=True)
        if cpus <= 0 or memory_gib <= 0 or default_timeout_s <= 0:
            raise ValueError("sandbox resource limits must be positive")
        self.workspace = workspace.resolve()
        self.image = image
        self.audit = audit
        self.cpus = cpus
        self.memory_gib = memory_gib
        self.default_timeout_s = default_timeout_s
        self.max_output_chars = max_output_chars
        self.test_env_path = test_env_path.rstrip("/")
        self.preflight_command = preflight_command
        self.support_dir = support_dir.resolve() if support_dir is not None else None
        if not self.test_env_path.startswith("/"):
            raise ValueError("sandbox test environment path must be absolute")
        if self.support_dir is not None and not self.support_dir.is_dir():
            raise FileNotFoundError(
                f"sandbox support directory is absent: {self.support_dir}"
            )
        suffix = re.sub(r"[^a-zA-Z0-9_.-]+", "-", workspace.parent.name)[-32:]
        self._container_name = f"beliefkv-{suffix}-{uuid.uuid4().hex[:8]}"
        self._started = False
        self._closed = False
        self._execute_lock = threading.Lock()
        self._workspace_digest_lock = threading.Lock()
        self._workspace_epoch_lock = threading.Lock()
        self._workspace_epoch = 0
        self._cancel_lock = threading.Lock()

    @property
    def id(self) -> str:
        return self._container_name

    def _resolve_path(self, key: str) -> Path:
        """Accept shell-visible `/workspace` paths in the virtual file backend."""

        prefix = self.shell_workdir
        if key == prefix:
            key = "/"
        elif key.startswith(prefix + "/"):
            key = key[len(prefix) :]
        return super()._resolve_path(key)

    def _to_virtual_path(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.cwd).as_posix()
        if relative == ".":
            return self.shell_workdir
        return f"{self.shell_workdir}/{relative}"

    def _docker_environment_args(self) -> list[str]:
        path = (
            f"{self.test_env_path}/bin:/opt/miniconda3/bin:/usr/local/sbin:"
            "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        python_path = (
            f"/beliefkv-support:{self.shell_workdir}"
            if self.support_dir is not None
            else self.shell_workdir
        )
        environment = (
            "HOME=/tmp",
            f"PATH={path}",
            f"CONDA_PREFIX={self.test_env_path}",
            "CONDA_DEFAULT_ENV=testbed",
            f"PYTHONPATH={python_path}",
            "PYTHONNOUSERSITE=1",
            "PIP_DISABLE_PIP_VERSION_CHECK=1",
            "PIP_NO_INDEX=1",
        )
        return [item for value in environment for item in ("--env", value)]

    def _docker_exec_argv(self, command: str) -> list[str]:
        return [
            "docker",
            "exec",
            "--workdir",
            self.shell_workdir,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            *self._docker_environment_args(),
            self._container_name,
            "/bin/sh",
            "-c",
            command,
        ]

    def _preflight(self) -> None:
        expected_python = f"{self.test_env_path}/bin/python"
        checks = [
            'test "$(pwd -P)" = /workspace',
            'test "$(git rev-parse --show-toplevel)" = /workspace',
            f'test "$(command -v python)" = {shlex.quote(expected_python)}',
            (
                "python -c "
                + shlex.quote(
                    "import sys; "
                    f"assert sys.executable.startswith({self.test_env_path!r})"
                )
            ),
        ]
        if self.preflight_command:
            checks.append(self.preflight_command)
        command = " && ".join(checks)
        started = time.monotonic()
        result = subprocess.run(
            self._docker_exec_argv(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60.0,
        )
        output = result.stdout or ""
        self.audit.emit(
            "sandbox_preflight",
            duration_ms=(time.monotonic() - started) * 1000.0,
            returncode=result.returncode,
            expected_python=expected_python,
            expected_workdir=self.shell_workdir,
            output_chars=len(output),
            output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                "sandbox test environment preflight failed: " + output[-2000:].strip()
            )

    def start(self) -> None:
        if self._started:
            raise RuntimeError("sandbox already started")
        uid = os.getuid()
        gid = os.getgid()
        support_mount = (
            [
                "--mount",
                f"type=bind,source={self.support_dir},target=/beliefkv-support,readonly",
            ]
            if self.support_dir is not None
            else []
        )
        command = [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            self._container_name,
            "--network",
            "none",
            "--read-only",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--pids-limit",
            "512",
            "--cpus",
            str(self.cpus),
            "--memory",
            f"{self.memory_gib:g}g",
            "--user",
            f"{uid}:{gid}",
            *self._docker_environment_args(),
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=2g",
            "--mount",
            f"type=bind,source={self.workspace},target=/workspace",
            *support_mount,
            "--workdir",
            self.shell_workdir,
            "--entrypoint",
            "/bin/sh",
            self.image,
            "-c",
            "while :; do sleep 3600; done",
        ]
        started = time.monotonic()
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120.0,
        )
        self.audit.emit(
            "sandbox_start",
            container_name=self._container_name,
            image=self.image,
            duration_ms=(time.monotonic() - started) * 1000.0,
            returncode=result.returncode,
            stderr=(result.stderr or "")[-2000:],
        )
        if result.returncode != 0:
            raise RuntimeError(f"sandbox start failed: {result.stderr.strip()}")
        self._started = True
        try:
            self._preflight()
        except BaseException:
            self.close()
            raise

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        if not self._started or self._closed:
            raise RuntimeError("sandbox is not running")
        timeout_s = timeout if timeout is not None else self.default_timeout_s
        timeout_s = max(1, min(int(timeout_s), 3600))
        command_sha256 = hashlib.sha256(command.encode("utf-8")).hexdigest()
        wrapped = (
            f"timeout --signal=KILL {timeout_s}s /bin/sh -c "
            f"{shlex.quote(command)}"
        )
        argv = self._docker_exec_argv(wrapped)
        started = time.monotonic()
        with self._execute_lock:
            try:
                result = subprocess.run(
                    argv,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_s + 15.0,
                )
                output = result.stdout or ""
                exit_code: int | None = result.returncode
            except subprocess.TimeoutExpired as error:
                partial = error.stdout or ""
                output = (
                    partial.decode("utf-8", errors="replace")
                    if isinstance(partial, bytes)
                    else partial
                )
                output += f"\nCommand exceeded host timeout ({timeout_s + 15}s)."
                exit_code = 124
        truncated = len(output) > self.max_output_chars
        if truncated:
            output = output[: self.max_output_chars] + "\n... output truncated ..."
        self.audit.emit(
            "sandbox_execute",
            command_chars=len(command),
            command_sha256=command_sha256,
            duration_ms=(time.monotonic() - started) * 1000.0,
            exit_code=exit_code,
            output_chars=len(output),
            output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
            truncated=truncated,
        )
        return ExecuteResponse(
            output=output,
            exit_code=exit_code,
            truncated=truncated,
        )

    def workspace_epoch(self) -> int:
        with self._workspace_epoch_lock:
            return self._workspace_epoch

    def _advance_workspace_epoch(self) -> int:
        with self._workspace_epoch_lock:
            self._workspace_epoch += 1
            return self._workspace_epoch

    def write(self, file_path: str, content: str) -> Any:
        result = super().write(file_path, content)
        if getattr(result, "error", None) is None:
            self._advance_workspace_epoch()
        return result

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> Any:
        result = super().edit(
            file_path,
            old_string,
            new_string,
            replace_all=replace_all,
        )
        if getattr(result, "error", None) is None:
            self._advance_workspace_epoch()
        return result

    def tool_state_digest(
        self,
        tool_name: str,
        payload: Mapping[str, Any],
    ) -> str | None:
        del payload
        if tool_name not in {"apply_patch", "edit_file", "write_file"}:
            return None
        return self.workspace_digest()

    def workspace_digest(self) -> str:
        """Hash tracked changes and untracked contents without changing the tree."""

        with self._workspace_digest_lock:
            diff = subprocess.run(
                ["git", "diff", "--no-ext-diff", "--binary", "HEAD", "--"],
                cwd=self.workspace,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15.0,
            )
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard", "-z"],
                cwd=self.workspace,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15.0,
            )
            if diff.returncode != 0 or untracked.returncode != 0:
                detail = (diff.stderr + untracked.stderr).decode(
                    "utf-8", errors="replace"
                )
                raise RuntimeError("workspace digest failed: " + detail[-1000:])
            digest = hashlib.sha256()
            digest.update(diff.stdout)
            for encoded_path in sorted(
                item for item in untracked.stdout.split(b"\0") if item
            ):
                digest.update(b"\0untracked\0")
                digest.update(encoded_path)
                path = self.workspace / os.fsdecode(encoded_path)
                if not path.is_file() or path.is_symlink():
                    digest.update(b"\0non-regular")
                    continue
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            return digest.hexdigest()

    def cancel_active_commands(self, *, reason: str) -> int:
        """Stop the per-workflow container to unblock any active sandbox command."""

        with self._cancel_lock:
            if not self._started or self._closed:
                return 0
            started = time.monotonic()
            result = subprocess.run(
                ["docker", "kill", self._container_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            self.audit.emit(
                "sandbox_cancel",
                reason=reason,
                duration_ms=(time.monotonic() - started) * 1000.0,
                returncode=result.returncode,
                stderr=(result.stderr or "")[-2000:],
            )
            if result.returncode == 0:
                # The container was started with --rm, so a successful kill also
                # removes it and makes all subsequent tool calls terminal errors.
                self._started = False
                self._closed = True
                return 1
            return 0

    def apply_unified_patch(self, patch: str) -> str:
        encoded = patch.encode("utf-8")
        if not encoded or len(encoded) > 200_000:
            return "Error: unified patch must contain between 1 and 200000 UTF-8 bytes"
        if "diff --git " not in patch and not patch.startswith("--- "):
            return "Error: expected a unified diff with repository-relative paths"

        normalized_patch = patch if patch.endswith("\n") else patch + "\n"

        patch_name = f".beliefkv-patch-{uuid.uuid4().hex}.diff"
        patch_path = self.workspace / patch_name
        patch_path.write_text(normalized_patch, encoding="utf-8")
        quoted = shlex.quote(patch_name)
        try:
            checked = self.execute(f"git apply --check --recount -- {quoted}")
            if checked.exit_code != 0:
                return "Error: git apply check failed\n" + checked.output
            applied = self.execute(
                f"git apply --whitespace=nowarn --recount -- {quoted}"
            )
            if applied.exit_code != 0:
                return "Error: git apply failed\n" + applied.output
            self._advance_workspace_epoch()
            return "Patch applied successfully."
        finally:
            patch_path.unlink(missing_ok=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._started:
            return
        started = time.monotonic()
        result = subprocess.run(
            ["docker", "rm", "--force", self._container_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        self.audit.emit(
            "sandbox_stop",
            duration_ms=(time.monotonic() - started) * 1000.0,
            returncode=result.returncode,
            stderr=(result.stderr or "")[-2000:],
        )


class GPUStatsMonitor:
    def __init__(self, gpu_index: int, output_path: Path) -> None:
        self.gpu_index = gpu_index
        self.output_path = output_path
        self.process: subprocess.Popen[str] | None = None
        self.stream: TextIO | None = None

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.output_path.open("x", encoding="utf-8")
        self.stream.write(
            "timestamp,memory_used_mib,memory_free_mib,gpu_utilization_percent,"
            "memory_utilization_percent,power_watts\n"
        )
        self.stream.flush()
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                str(self.gpu_index),
                "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu,"
                "utilization.memory,power.draw",
                "--format=csv,noheader,nounits",
                "--loop-ms=200",
            ],
            stdout=self.stream,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
            self.process = None
        if self.stream is not None:
            self.stream.close()
            self.stream = None


PROMETHEUS_GAUGE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$"
)


def prometheus_gauge_sum(payload: str, metric_name: str) -> float | None:
    values = [
        float(match.group("value"))
        for line in payload.splitlines()
        if (match := PROMETHEUS_GAUGE.match(line.strip())) is not None
        and match.group("name") == metric_name
    ]
    return sum(values) if values else None


class SGLangMetricsMonitor:
    def __init__(
        self,
        base_url: str,
        output_path: Path,
        *,
        pool_tokens: int,
        poll_interval_s: float = 0.1,
    ) -> None:
        root = base_url.rstrip("/")
        self.root = root[:-3] if root.endswith("/v1") else root
        self.output_path = output_path
        self.pool_tokens = pool_tokens
        self.poll_interval_s = poll_interval_s
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []
        self.error_count = 0

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("x", encoding="utf-8", buffering=1) as stream:
            while not self.stop.is_set():
                sample: dict[str, Any] = {
                    "monotonic_ts_ms": time.monotonic() * 1000.0,
                }
                try:
                    with urllib.request.urlopen(
                        f"{self.root}/metrics", timeout=3.0
                    ) as response:
                        payload = response.read().decode("utf-8")
                    with urllib.request.urlopen(
                        f"{self.root}/get_load", timeout=3.0
                    ) as response:
                        load = json.load(response)
                    for metric in (
                        "sglang:num_used_tokens",
                        "sglang:num_running_reqs",
                        "sglang:num_queue_reqs",
                        "sglang:token_usage",
                    ):
                        value = prometheus_gauge_sum(payload, metric)
                        if value is not None:
                            sample[metric.removeprefix("sglang:")] = value
                    resident = sample.get("num_used_tokens")
                    if resident is not None:
                        sample["resident_pressure"] = resident / self.pool_tokens
                    sample["demand_load"] = int(load.get("load", 0))
                    self.samples.append(sample)
                except Exception as error:
                    self.error_count += 1
                    sample["error"] = f"{type(error).__name__}: {error}"
                stream.write(
                    json.dumps(sample, sort_keys=True, allow_nan=False) + "\n"
                )
                self.stop.wait(self.poll_interval_s)

    def close(self) -> dict[str, Any]:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10.0)
            if self.thread.is_alive():
                raise RuntimeError("SGLang metrics monitor did not stop")
        resident = [
            int(item["num_used_tokens"])
            for item in self.samples
            if "num_used_tokens" in item
        ]
        return {
            "sample_count": len(self.samples),
            "error_count": self.error_count,
            "max_resident_tokens": max(resident, default=0),
            "max_resident_pressure": max(resident, default=0) / self.pool_tokens,
        }


class DelegatedTask(BaseModel):
    role: str = Field(description="Short semantic role for the child agent")
    description: str = Field(description="Self-contained repository analysis task")


class DelegationPlan(BaseModel):
    rationale: str = Field(description="Brief reason for the chosen decomposition")
    tasks: list[str] = Field(
        default_factory=list,
        description=(
            "Zero to two self-contained independent analysis task descriptions "
            "worth parallelizing"
        ),
        max_length=2,
    )


class PartialAgentRunError(RuntimeError):
    def __init__(self, cause: BaseException, partial_result: dict[str, Any]) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.partial_result = partial_result


@dataclass(frozen=True)
class DeepAgentsExperimentConfig:
    mode: str
    base_url: str
    model: str
    output_dir: Path
    workload_manifest: Path
    docker_image: str
    control_socket: Path | None = None
    server_audit_path: Path | None = None
    server_event_path: Path | None = None
    server_log_path: Path | None = None
    instance_ids: tuple[str, ...] = ()
    max_workflows: int = 4
    concurrency: int = 4
    gpu_index: int = 0
    pool_tokens: int = 163_840
    max_completion_tokens: int = 2048
    sampling_seed: int | None = None
    recursion_limit: int = 512
    request_timeout_s: float = 600.0
    sandbox_command_timeout_s: int = 600
    sandbox_test_env_path: str = DEFAULT_SANDBOX_TEST_ENV
    sandbox_preflight_command: str | None = None
    completion_gate_enabled: bool = True
    completion_repair_attempts: int = 2
    runtime_event_ack_timeout_s: float = 10.0
    runtime_event_ack_retries: int = 3
    context_lifecycle: ContextLifecyclePolicy = field(
        default_factory=ContextLifecyclePolicy
    )
    loop_guard: LoopGuardPolicy = field(default_factory=LoopGuardPolicy)
    tool_observation_budget: ToolObservationBudgetPolicy = field(
        default_factory=ToolObservationBudgetPolicy
    )

    def __post_init__(self) -> None:
        if self.mode not in {"autonomous", "planned"}:
            raise ValueError("mode must be autonomous or planned")
        if min(
            self.max_workflows,
            self.concurrency,
            self.pool_tokens,
            self.max_completion_tokens,
            self.recursion_limit,
            self.sandbox_command_timeout_s,
        ) <= 0:
            raise ValueError("experiment limits must be positive")
        if self.completion_repair_attempts < 0:
            raise ValueError("completion_repair_attempts must be non-negative")
        if self.sampling_seed is not None and self.sampling_seed < 0:
            raise ValueError("sampling_seed must be non-negative when configured")
        if self.runtime_event_ack_timeout_s <= 0 or self.runtime_event_ack_retries <= 0:
            raise ValueError("runtime-event ACK policy must be positive")
        if self.context_lifecycle.intermediate_output_tokens > self.max_completion_tokens:
            raise ValueError(
                "context lifecycle intermediate output budget exceeds the model budget"
            )
        if not self.sandbox_test_env_path.startswith("/"):
            raise ValueError("sandbox_test_env_path must be absolute")


AUTONOMOUS_SYSTEM_PROMPT = """You are the supervisor for a real SWE-bench coding task.
Work only in the mounted repository. Use the filesystem and execute tools freely; the
execute tool is already isolated in an offline Docker sandbox. Diagnose, edit, and test
the repository. A workflow is complete only when you return the required
WorkflowCompletion structured response. Do not finish with ordinary prose. Use
status=patched_and_tested only after implementing every requirement, leaving unresolved
empty, and observing a successful focused repository test command.

You may delegate repository work through task. Decide the number of subagents at
runtime: there is no required or preconfigured count. Delegate only when a task has
independent multi-step work or benefits from a separate context. When several tasks are
independent, issue their task calls together so they can run concurrently. Do not
delegate trivial one-step work. Integrate child reports and leave the final patch in the
shared workspace. Never access paths outside the mounted repository.
""" + SANDBOX_PATH_CONTRACT


PLANNER_SYSTEM_PROMPT = """You are a code-orchestrated planner for a SWE-bench task.
Return a structured decomposition containing zero to two independent repository
analysis tasks. Choose the count from task structure, not from a fixed policy. Use two
tasks only when they produce orthogonal evidence, normally one implementation/call-path
analysis and one failure/test analysis. Never create several tasks that merely reproduce
the same bug or inspect adjacent functions in the same call path. Children only report
evidence; a later implementation agent will edit the repository. Make every description
self-contained and require a concrete candidate symbol or invariant, not only a broad
area of the codebase. The tasks field must be a plain list of strings, with one complete
task description per string.
"""


IMPLEMENTER_SYSTEM_PROMPT = """You are the implementation stage of a planned
SWE-bench workflow. Work only in the mounted repository. Use the supplied child reports
as evidence, but verify them. Diagnose, edit, and test the code with filesystem and
execute tools. Use edit_file for focused changes or apply_patch for coherent multi-line
edits. The execute tool runs
in an offline Docker sandbox. Leave the final patch in the workspace. A workflow is
complete only when you return the required
WorkflowCompletion structured response. Do not finish with ordinary prose. You may use
status=patched_and_tested only after implementing every requirement in the issue,
leaving unresolved empty, and observing a successful focused repository test command.
Diagnostic `python -c` commands are not repository tests. If the task cannot be
completed, return status=blocked with concrete unresolved reasons.
Reproduce the issue once, then move from diagnosis to a source change as soon as a
candidate function or invariant is identified. Do not spend the implementation budget
repeating equivalent `python -c` variants.
When the issue includes a proposed diff, use it as a concrete starting point, apply it
with correct surrounding context, and validate it instead of repeatedly re-deriving it.
All file-editing tools are confined to the isolated `/workspace`. Do not rewrite source
files through shell commands.
""" + SANDBOX_PATH_CONTRACT


COMPLETION_REPAIR_SYSTEM_PROMPT = """You are the correctness-repair stage for a real
SWE-bench workflow. A previous implementation attempt was rejected by the runtime gate.
Inspect the current workspace and complete every requirement in the original issue.
Correct or extend the existing patch, add or update regression tests when appropriate,
and run a focused repository test command. Use edit_file for focused changes or the
apply_patch tool for coherent multi-line changes; use repository-relative `a/...` and
`b/...` paths in unified diffs. Do not repeatedly run the same diagnostic probe. `python -c` is useful for
diagnosis but does not count as a repository test.
All file-editing tools are confined to the isolated `/workspace`. Do not rewrite source
files through shell commands.

Return status=patched_and_tested only when the workspace has a substantive patch, every
issue requirement is implemented, unresolved is empty, and at least one actual
repository-native test such as `python bin/test <test-path>` has exited successfully.
Otherwise return an honest non-success status with concrete unresolved items.
""" + SANDBOX_PATH_CONTRACT


CHILD_COMPLETION_INSTRUCTION = (
    "Return a ChildCompletion structured response with a concise summary, concrete "
    "repository evidence, commands and outcomes, files changed, unresolved items, and "
    "confidence."
)
WORKFLOW_COMPLETION_INSTRUCTION = (
    "Return a WorkflowCompletion structured response describing terminal status, the "
    "implementation, changed files, tests, and unresolved items. Use "
    "patched_and_tested only when all issue requirements are implemented, unresolved "
    "is empty, and a real repository test (not python -c) succeeded."
)


def _loop_guard(
    config: DeepAgentsExperimentConfig,
    *,
    completion_schema: type[BaseModel],
    completion_instruction: str,
    audit: JsonlAudit,
    scope: str,
    policy: LoopGuardPolicy | None = None,
) -> AgentLoopGuardMiddleware:
    return AgentLoopGuardMiddleware(
        policy=policy or config.loop_guard,
        completion_schema=completion_schema,
        completion_instruction=completion_instruction,
        audit=audit,
        scope=scope,
        finalization_tool_names=(
            frozenset({"apply_patch", "execute"})
            if completion_schema is WorkflowCompletion
            else frozenset()
        ),
    )


def _tool_circuit(
    backend: DockerWorkspaceBackend,
    *,
    scope: str,
    adapter: DeepAgentsRuntimeAdapter | None = None,
) -> ToolCircuitBreakerMiddleware:
    return ToolCircuitBreakerMiddleware(
        state_epoch=backend.workspace_epoch,
        audit=backend.audit,
        scope=scope,
        censor_observer=(adapter.record_call_censor if adapter is not None else None),
    )


def _tool_observation_budget(
    config: DeepAgentsExperimentConfig,
    *,
    audit: JsonlAudit,
    scope: str,
) -> ToolObservationBudgetMiddleware:
    return ToolObservationBudgetMiddleware(
        policy=config.tool_observation_budget,
        audit=audit,
        scope=scope,
    )


def _planned_child_loop_guard_policy(
    config: DeepAgentsExperimentConfig,
) -> LoopGuardPolicy:
    policy = config.loop_guard
    return replace(
        policy,
        repeated_call_limit=min(policy.repeated_call_limit, 3),
        max_model_calls_without_completion=min(
            policy.max_model_calls_without_completion, 12
        ),
        max_tool_calls_without_completion=min(
            policy.max_tool_calls_without_completion, 16
        ),
    )


def _workspace_patch_tool(backend: DockerWorkspaceBackend) -> BaseTool:
    @tool("apply_patch")
    def apply_patch_tool(patch: str) -> str:
        """Atomically apply a unified diff to repository files in `/workspace`.

        Use repository-relative paths prefixed with `a/` and `b/`. The patch is checked
        before it is applied, and paths outside the repository are rejected.
        """

        return backend.apply_unified_patch(patch)

    return apply_patch_tool


def _filesystem_middleware(
    backend: DockerWorkspaceBackend, *, allow_direct_edits: bool
) -> FilesystemMiddleware:
    middleware = FilesystemMiddleware(backend=backend)
    if not allow_direct_edits:
        middleware.tools = [
            item
            for item in middleware.tools
            if item.name not in {"write_file", "edit_file"}
        ]
    return middleware


def _model(
    config: DeepAgentsExperimentConfig,
    adapter: DeepAgentsRuntimeAdapter,
) -> BeliefKVChatOpenAI:
    return BeliefKVChatOpenAI(
        beliefkv_adapter=adapter,
        model=config.model,
        base_url=config.base_url,
        api_key="EMPTY",
        temperature=0.0,
        seed=config.sampling_seed,
        max_completion_tokens=config.max_completion_tokens,
        timeout=config.request_timeout_s,
        max_retries=0,
        streaming=False,
        disable_streaming="tool_calling",
    )


def _task_prompt(workload: SweBenchWorkload) -> str:
    return (
        f"SWE-bench instance: {workload.instance_id}\n"
        f"Repository: {workload.repo}\n"
        f"Base commit: {workload.base_commit}\n\n"
        f"Problem statement:\n{workload.problem_statement}\n\n"
        f"{repository_sandbox_contract(workload)}\n"
        "Produce a complete working patch for every stated requirement and run a "
        "focused repository test command before reporting success."
    )


def _message_payload(message: BaseMessage) -> dict[str, Any]:
    payload = message.model_dump(mode="json")
    payload["message_type"] = message.type
    return payload


def _result_messages(result: dict[str, Any]) -> list[BaseMessage]:
    messages = result.get("messages", [])
    return [item for item in messages if isinstance(item, BaseMessage)]


def _message_text(message: BaseMessage) -> str:
    try:
        return message.text
    except (AttributeError, TypeError, ValueError):
        return str(message.content)


def observed_successful_test_commands(messages: Sequence[BaseMessage]) -> list[str]:
    execute_calls: dict[str, str] = {}
    successful: list[str] = []
    failure_markers = (
        "[command failed with exit code",
        "command exceeded host timeout",
        "killed by signal",
    )
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call.get("name") != "execute":
                    continue
                command = call.get("args", {}).get("command")
                call_id = str(call.get("id", ""))
                if call_id and isinstance(command, str):
                    execute_calls[call_id] = command
            continue
        if not isinstance(message, ToolMessage) or message.name != "execute":
            continue
        command = execute_calls.get(str(message.tool_call_id))
        if command is None or TEST_COMMAND_PATTERN.search(command) is None:
            continue
        output = _message_text(message).lower()
        if any(marker in output for marker in failure_markers):
            continue
        if ZERO_TEST_OUTPUT_PATTERN.search(output):
            continue
        if UNSUPPORTED_SYMPY_TEST_SELECTOR_PATTERN.search(command):
            continue
        successful.append(command)
    return list(dict.fromkeys(successful))


def validate_workflow_completion(
    completion: WorkflowCompletion | None,
    *,
    patch: str,
    observed_tests: Sequence[str],
) -> dict[str, Any]:
    errors: list[str] = []
    if completion is None:
        errors.append("missing_structured_completion")
    else:
        if completion.status != "patched_and_tested":
            errors.append(f"terminal_status:{completion.status}")
        if not completion.tests:
            errors.append("completion_has_no_test_evidence")
        if completion.unresolved:
            errors.append("completion_has_unresolved_items")
        if INCOMPLETE_SUMMARY_PATTERN.search(completion.summary):
            errors.append("completion_summary_declares_incomplete_work")
    if not patch.strip():
        errors.append("workspace_has_no_patch")
    if not observed_tests:
        errors.append("no_successful_test_command_observed")
    return {
        "passed": not errors,
        "errors": errors,
        "observed_successful_test_commands": list(observed_tests),
    }


def _final_text(result: dict[str, Any]) -> str:
    structured = result.get("structured_response")
    if isinstance(structured, BaseModel):
        return json.dumps(
            structured.model_dump(mode="json"),
            sort_keys=True,
            allow_nan=False,
        )
    if isinstance(structured, dict):
        return json.dumps(structured, sort_keys=True, allow_nan=False, default=str)
    messages = _result_messages(result)
    return messages[-1].text if messages else ""


def _invoke_with_partial_state(
    agent: Any,
    inputs: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    try:
        for state in agent.stream(inputs, config=config, stream_mode="values"):
            if isinstance(state, dict):
                latest = state
    except BaseException as error:
        raise PartialAgentRunError(error, latest) from error
    return latest


AUTONOMOUS_SUBAGENT_SPECS = (
    (
        "repository-explorer",
        "Trace implementation paths and report concrete code evidence.",
        "Investigate the assigned repository question deeply. Use filesystem and "
        "execute tools, avoid broad unrelated edits, and finish with the required "
        "ChildCompletion structured response.",
    ),
    (
        "test-analyst",
        "Reproduce failures and identify focused regression tests.",
        "Analyze or reproduce the assigned failure in the sandbox. Report exact "
        "commands, relevant tests, and likely regression coverage through the "
        "required ChildCompletion structured response.",
    ),
    (
        "implementation-agent",
        "Implement and validate a self-contained part of the fix.",
        "Implement the delegated part in the shared workspace and run focused "
        "tests. Finish with the required ChildCompletion structured response, "
        "including files changed, test results, and unresolved risks.",
    ),
    (
        str(GENERAL_PURPOSE_SUBAGENT["name"]),
        str(GENERAL_PURPOSE_SUBAGENT["description"]),
        str(GENERAL_PURPOSE_SUBAGENT["system_prompt"])
        + "\n\nFinish with the required ChildCompletion structured response.",
    ),
)


def _context_lifecycle_middleware(
    config: DeepAgentsExperimentConfig,
    backend: DockerWorkspaceBackend,
    adapter: DeepAgentsRuntimeAdapter,
    summary_model: Any,
    *,
    persist_cursor_across_invocations: bool = False,
) -> ContextLifecycleMiddleware:
    return ContextLifecycleMiddleware(
        summary_model,
        backend=backend,
        policy=config.context_lifecycle,
        compaction_sink=adapter,
        summary_callbacks=(adapter,),
        persist_cursor_across_invocations=persist_cursor_across_invocations,
    )


def _autonomous_subagents(
    config: DeepAgentsExperimentConfig,
    workload: SweBenchWorkload,
    backend: DockerWorkspaceBackend,
    adapter: DeepAgentsRuntimeAdapter,
    model: Any,
    summary_model: Any,
) -> list[dict[str, Any]]:
    subagents: list[dict[str, Any]] = []
    for name, description, system_prompt in AUTONOMOUS_SUBAGENT_SPECS:
        scope = f"autonomous:{name}"
        subagents.append(
            {
                "name": name,
                "description": description,
                "system_prompt": (
                    system_prompt
                    + SANDBOX_PATH_CONTRACT
                    + repository_sandbox_contract(workload)
                ),
                "model": model,
                "tools": [_workspace_patch_tool(backend)],
                "response_format": ToolStrategy(ChildCompletion),
                "middleware": [
                    TodoListMiddleware(),
                    _filesystem_middleware(backend, allow_direct_edits=True),
                    _context_lifecycle_middleware(
                        config,
                        backend,
                        adapter,
                        summary_model,
                    ),
                    CompletionBudgetMiddleware(
                        intermediate_tokens=(
                            config.context_lifecycle.intermediate_output_tokens
                        ),
                        final_tokens=config.max_completion_tokens,
                    ),
                    PatchToolCallsMiddleware(),
                    _tool_circuit(backend, scope=scope, adapter=adapter),
                    ToolOutcomeStatusMiddleware(),
                    _tool_observation_budget(
                        config,
                        audit=backend.audit,
                        scope=scope,
                    ),
                    _loop_guard(
                        config,
                        completion_schema=ChildCompletion,
                        completion_instruction=CHILD_COMPLETION_INSTRUCTION,
                        audit=backend.audit,
                        scope=scope,
                    ),
                ],
            }
        )
    return subagents


def _build_autonomous_agent(
    config: DeepAgentsExperimentConfig,
    workload: SweBenchWorkload,
    backend: DockerWorkspaceBackend,
    adapter: DeepAgentsRuntimeAdapter,
) -> Any:
    model = _model(config, adapter)
    summary_model = model.model_copy(
        update={"max_tokens": config.context_lifecycle.summary_output_tokens}
    )
    task_middleware = PrivateStateIsolatingSubAgentMiddleware(
        backend=backend,
        private_state_keys=CONTEXT_LIFECYCLE_PRIVATE_STATE_KEYS,
        subagents=_autonomous_subagents(
            config,
            workload,
            backend,
            adapter,
            model,
            summary_model,
        ),
    )
    middleware: list[Any] = [
        TodoListMiddleware(),
        _filesystem_middleware(backend, allow_direct_edits=True),
        task_middleware,
        _context_lifecycle_middleware(
            config,
            backend,
            adapter,
            summary_model,
        ),
        CompletionBudgetMiddleware(
            intermediate_tokens=config.context_lifecycle.intermediate_output_tokens,
            final_tokens=config.max_completion_tokens,
        ),
        PatchToolCallsMiddleware(),
        _tool_circuit(backend, scope="autonomous:supervisor", adapter=adapter),
        ToolOutcomeStatusMiddleware(),
        _tool_observation_budget(
            config,
            audit=backend.audit,
            scope="autonomous:supervisor",
        ),
        _loop_guard(
            config,
            completion_schema=WorkflowCompletion,
            completion_instruction=WORKFLOW_COMPLETION_INSTRUCTION,
            audit=backend.audit,
            scope="autonomous:supervisor",
        ),
    ]
    return create_agent(
        model=model,
        tools=[_workspace_patch_tool(backend)],
        system_prompt=(
            AUTONOMOUS_SYSTEM_PROMPT
            + repository_sandbox_contract(workload)
            + "\n\n"
            + BASE_AGENT_PROMPT
        ),
        middleware=middleware,
        response_format=ToolStrategy(WorkflowCompletion),
        name="beliefkv-swebench-supervisor",
    )


def _run_autonomous(
    config: DeepAgentsExperimentConfig,
    workload: SweBenchWorkload,
    backend: DockerWorkspaceBackend,
    adapter: DeepAgentsRuntimeAdapter,
) -> dict[str, Any]:
    agent = _build_autonomous_agent(config, workload, backend, adapter)
    return _invoke_with_partial_state(
        agent,
        {"messages": [{"role": "user", "content": _task_prompt(workload)}]},
        {
            "callbacks": [adapter],
            "recursion_limit": config.recursion_limit,
            "metadata": {"beliefkv_mode": "autonomous"},
        },
    )


def _run_planned_child(
    config: DeepAgentsExperimentConfig,
    workload: SweBenchWorkload,
    backend: DockerWorkspaceBackend,
    adapter: DeepAgentsRuntimeAdapter,
    handle: DeclaredRuntimeTask,
    task: DelegatedTask,
) -> ChildCompletion:
    role_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", task.role).strip("-")
    child_key = hashlib.sha256(handle.invocation_id.encode()).hexdigest()[:12]
    child_workspace = (
        backend.workspace.parent / "planned_children" / child_key / "workspace"
    )
    prepare_workspace(backend.workspace, workload, child_workspace)
    child_backend = DockerWorkspaceBackend(
        child_workspace,
        image=backend.image,
        audit=backend.audit,
        cpus=backend.cpus,
        memory_gib=backend.memory_gib,
        default_timeout_s=backend.default_timeout_s,
        max_output_chars=backend.max_output_chars,
        test_env_path=backend.test_env_path,
        preflight_command=backend.preflight_command,
        support_dir=backend.support_dir,
    )
    try:
        child_backend.start()
        child = create_agent(
            model=_model(config, adapter),
            tools=[],
            middleware=[
                _tool_circuit(
                    child_backend,
                    scope=f"planned:child:{handle.invocation_id}",
                    adapter=adapter,
                ),
                ToolOutcomeStatusMiddleware(),
                _tool_observation_budget(
                    config,
                    audit=backend.audit,
                    scope=f"planned:child:{handle.invocation_id}",
                ),
                _filesystem_middleware(child_backend, allow_direct_edits=False),
                _loop_guard(
                    config,
                    completion_schema=ChildCompletion,
                    completion_instruction=CHILD_COMPLETION_INSTRUCTION,
                    audit=backend.audit,
                    scope=f"planned:child:{handle.invocation_id}",
                    policy=_planned_child_loop_guard_policy(config),
                ),
            ],
            response_format=ToolStrategy(ChildCompletion),
            system_prompt=(
                "You are an analysis child in a code-planned SWE-bench workflow. "
                "Inspect and test the mounted repository. Do not edit files. Return "
                "concrete evidence only; a later implementation stage owns the patch. "
                "Use at most a few targeted tool calls: reproduce once, inspect the "
                "relevant implementation and tests, then report the most likely symbol "
                "and invariant to change. Do not enumerate many equivalent python -c "
                "probes. "
                "You complete the task only by returning the required ChildCompletion "
                "structured response. Do not finish with ordinary prose."
            ) + SANDBOX_PATH_CONTRACT + repository_sandbox_contract(workload),
            name=f"beliefkv-planned-{role_name or 'analyst'}",
        )
        result = _invoke_with_partial_state(
            child,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Issue:\n{workload.problem_statement}\n\n"
                            f"Assigned analysis task:\n{task.description}"
                        ),
                    }
                ]
            },
            {
                "callbacks": [adapter],
                "recursion_limit": config.recursion_limit,
                "metadata": {
                    **adapter.invocation_scope(handle),
                    "beliefkv_mode": "planned_child",
                },
            },
        )
        completion = require_structured_completion(result, ChildCompletion)
        assert isinstance(completion, ChildCompletion)
        return completion
    finally:
        child_backend.close()


def _run_planned(
    config: DeepAgentsExperimentConfig,
    workload: SweBenchWorkload,
    backend: DockerWorkspaceBackend,
    adapter: DeepAgentsRuntimeAdapter,
    artifact_dir: Path,
) -> tuple[dict[str, Any], DelegationPlan, list[dict[str, Any]]]:
    planner_model = _model(config, adapter)
    planner = planner_model.with_structured_output(
        DelegationPlan,
        method="function_calling",
        strict=False,
    )
    plan = planner.invoke(
        [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": _task_prompt(workload)},
        ],
        config={
            "callbacks": [adapter],
            "metadata": {"beliefkv_mode": "planned_planner"},
        },
    )
    if not isinstance(plan, DelegationPlan):
        plan = DelegationPlan.model_validate(plan)
    write_json(artifact_dir / "plan.json", plan.model_dump(mode="json"))
    planned_tasks = [
        DelegatedTask(role=f"analysis-{index + 1}", description=description)
        for index, description in enumerate(plan.tasks)
    ]
    handles = adapter.declare_runtime_tasks(
        [(item.role, item.description) for item in planned_tasks],
        group_id=f"planned:{workload.instance_id}",
    )
    reports: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(handles))) as executor:
        futures = {
            executor.submit(
                _run_planned_child,
                config,
                workload,
                backend,
                adapter,
                handle,
                task,
            ): (handle, task)
            for handle, task in zip(handles, planned_tasks)
        }
        for future in as_completed(futures):
            handle, task = futures[future]
            error: BaseException | None = None
            report = ""
            completion_payload: dict[str, Any] | None = None
            try:
                completion = future.result()
                completion_payload = completion.model_dump(mode="json")
                report = json.dumps(
                    completion_payload,
                    sort_keys=True,
                    allow_nan=False,
                )
            except BaseException as caught:
                error = caught
                partial = (
                    _final_text(caught.partial_result)
                    if isinstance(caught, PartialAgentRunError)
                    else ""
                )
                report = partial or f"Child failed: {type(caught).__name__}: {caught}"
            finally:
                adapter.complete_runtime_task(handle, error=error)
            reports.append(
                {
                    "role": task.role,
                    "description": task.description,
                    "invocation_id": handle.invocation_id,
                    "report": report,
                    "semantic_completion": completion_payload,
                    "error": (
                        (
                            f"{type(error.cause).__name__}: {error.cause}"
                            if isinstance(error, PartialAgentRunError)
                            else f"{type(error).__name__}: {error}"
                        )
                        if error is not None
                        else None
                    ),
                }
            )
            write_json(artifact_dir / "child_reports.json", reports)

    evidence = "\n\n".join(
        f"[{item['role']}]\n{str(item['report'])[:24000]}" for item in reports
    )
    implementer = create_agent(
        model=_model(config, adapter),
        tools=[_workspace_patch_tool(backend)],
        middleware=[
            _tool_circuit(
                backend, scope="planned:implementer", adapter=adapter
            ),
            ToolOutcomeStatusMiddleware(),
            _tool_observation_budget(
                config,
                audit=backend.audit,
                scope="planned:implementer",
            ),
            _filesystem_middleware(backend, allow_direct_edits=True),
            _loop_guard(
                config,
                completion_schema=WorkflowCompletion,
                completion_instruction=WORKFLOW_COMPLETION_INSTRUCTION,
                audit=backend.audit,
                scope="planned:implementer",
            ),
        ],
        response_format=ToolStrategy(WorkflowCompletion),
        system_prompt=(
            IMPLEMENTER_SYSTEM_PROMPT + repository_sandbox_contract(workload)
        ),
        name="beliefkv-planned-implementer",
    )
    result = _invoke_with_partial_state(
        implementer,
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{_task_prompt(workload)}\n\n"
                        f"Planner rationale:\n{plan.rationale}\n\n"
                        f"Child reports:\n{evidence or '(no delegated tasks)'}"
                    ),
                }
            ]
        },
        {
            "callbacks": [adapter],
            "recursion_limit": config.recursion_limit,
            "metadata": {"beliefkv_mode": "planned_implementer"},
        },
    )
    return result, plan, reports


def _completion_gate_for_result(
    result: dict[str, Any], workspace: Path, *, runtime_tests: Sequence[str] = ()
) -> dict[str, Any]:
    try:
        parsed = require_structured_completion(result, WorkflowCompletion)
        completion = parsed if isinstance(parsed, WorkflowCompletion) else None
    except (RuntimeError, ValueError):
        completion = None
    patch = command_output(["git", "diff", "--binary", "HEAD"], cwd=workspace)
    observed = [
        *observed_successful_test_commands(_result_messages(result)),
        *runtime_tests,
    ]
    return validate_workflow_completion(
        completion,
        patch=patch,
        observed_tests=list(dict.fromkeys(observed)),
    )


def _runtime_verify_changed_tests(
    result: dict[str, Any], backend: DockerWorkspaceBackend
) -> list[str]:
    patch = command_output(["git", "diff", "--binary", "HEAD"], cwd=backend.workspace)
    patch_sha256 = hashlib.sha256(patch.encode("utf-8")).hexdigest()
    cached = result.get(RUNTIME_VERIFIED_TESTS_KEY)
    if isinstance(cached, dict) and cached.get("patch_sha256") == patch_sha256:
        return [str(item) for item in cached.get("commands", [])]

    changed = command_output(
        ["git", "diff", "--name-only", "--diff-filter=ACMRT", "HEAD", "--"],
        cwd=backend.workspace,
    ).splitlines()
    test_files = [
        name
        for name in changed
        if name.endswith(".py")
        and any(part == "tests" for part in Path(name).parts)
        and Path(name).name.startswith("test_")
        and (backend.workspace / name).is_file()
    ][:8]
    commands: list[str] = []
    returncode: int | None = None
    if test_files:
        quoted_files = " ".join(shlex.quote(name) for name in test_files)
        if (backend.workspace / "bin/test").is_file():
            command = f"python bin/test {quoted_files}"
        else:
            command = f"pytest {quoted_files}"
        response = backend.execute(command, timeout=600)
        returncode = response.exit_code
        if response.exit_code == 0:
            commands.append(command)
    backend.audit.emit(
        "workflow_test_verifier",
        patch_sha256=patch_sha256,
        test_file_count=len(test_files),
        returncode=returncode,
        passed=bool(commands),
    )
    result[RUNTIME_VERIFIED_TESTS_KEY] = {
        "patch_sha256": patch_sha256,
        "commands": commands,
    }
    return commands


def _repair_incomplete_workflow(
    config: DeepAgentsExperimentConfig,
    workload: SweBenchWorkload,
    backend: DockerWorkspaceBackend,
    adapter: DeepAgentsRuntimeAdapter,
    initial_result: dict[str, Any],
) -> dict[str, Any]:
    result = initial_result
    for attempt in range(config.completion_repair_attempts + 1):
        runtime_tests = _runtime_verify_changed_tests(result, backend)
        gate = _completion_gate_for_result(
            result, backend.workspace, runtime_tests=runtime_tests
        )
        backend.audit.emit(
            "workflow_completion_gate",
            attempt=attempt,
            passed=gate["passed"],
            errors=gate["errors"],
            observed_test_count=len(gate["observed_successful_test_commands"]),
        )
        if gate["passed"] or attempt == config.completion_repair_attempts:
            return result

        patch = command_output(
            ["git", "diff", "--binary", "HEAD"], cwd=backend.workspace
        )
        repair = create_agent(
            model=_model(config, adapter),
            tools=[_workspace_patch_tool(backend)],
            middleware=[
                _tool_circuit(
                    backend,
                    scope=f"completion-repair:{attempt + 1}",
                    adapter=adapter,
                ),
                ToolOutcomeStatusMiddleware(),
                _tool_observation_budget(
                    config,
                    audit=backend.audit,
                    scope=f"completion-repair:{attempt + 1}",
                ),
                _filesystem_middleware(backend, allow_direct_edits=True),
                _loop_guard(
                    config,
                    completion_schema=WorkflowCompletion,
                    completion_instruction=WORKFLOW_COMPLETION_INSTRUCTION,
                    audit=backend.audit,
                    scope=f"completion-repair:{attempt + 1}",
                ),
            ],
            response_format=ToolStrategy(WorkflowCompletion),
            system_prompt=COMPLETION_REPAIR_SYSTEM_PROMPT,
            name=f"beliefkv-completion-repair-{attempt + 1}",
        )
        result = _invoke_with_partial_state(
            repair,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{_task_prompt(workload)}\n\n"
                            f"Runtime gate rejection reasons: {gate['errors']}\n\n"
                            f"Runtime-verified passing tests: {runtime_tests or '(none)'}\n\n"
                            f"Previous completion:\n{_final_text(result)[:12000]}\n\n"
                            f"Current patch:\n{patch[:24000] or '(empty)'}"
                        ),
                    }
                ]
            },
            {
                "callbacks": [adapter],
                "recursion_limit": config.recursion_limit,
                "metadata": {
                    "beliefkv_mode": "completion_repair",
                    "beliefkv_repair_attempt": attempt + 1,
                },
            },
        )
    return result


def _trace_summary(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts = Counter(str(item.get("kind")) for item in records)
    children = {
        str(item["invocation_id"])
        for item in records
        if item.get("kind") == "invocation_create"
        and item.get("invocation_id") is not None
        and item.get("parent_invocation_id") is not None
        and not bool((item.get("attributes") or {}).get("runtime_internal"))
    }
    returned_children = {
        str(item["invocation_id"])
        for item in records
        if item.get("kind") == "return"
        and item.get("invocation_id") in children
    }
    tool_status_observations = 0
    mutating_tool_ends = 0
    workspace_digest_observations = 0
    external_llm_submits = 0
    external_llm_results = 0
    for item in records:
        kind = item.get("kind")
        attributes = item.get("attributes") or {}
        if kind == "llm_submit" and not bool(attributes.get("runtime_internal")):
            external_llm_submits += 1
        elif kind == "llm_result" and not bool(attributes.get("runtime_internal")):
            external_llm_results += 1
        elif kind == "tool_end":
            tool_status_observations += int(attributes.get("status") is not None)
            if attributes.get("tool_name") in {
                "apply_patch",
                "edit_file",
                "write_file",
            }:
                mutating_tool_ends += 1
                workspace_digest_observations += int(
                    attributes.get("workspace_digest_before") is not None
                    and attributes.get("workspace_digest_after") is not None
                )
    join_create_count = counts["join_create"]
    join_satisfied_count = counts["join_satisfied"]
    return {
        "event_count": len(records),
        "event_counts": dict(sorted(counts.items())),
        "dynamic_subagent_count": counts["spawn"],
        "llm_request_count": external_llm_submits,
        "llm_result_count": external_llm_results,
        "llm_pairing_valid": external_llm_submits == external_llm_results,
        "tool_call_count": counts["tool_start"],
        "tool_result_count": counts["tool_end"],
        "tool_pairing_valid": counts["tool_start"] == counts["tool_end"],
        "tool_status_coverage": (
            tool_status_observations / counts["tool_end"]
            if counts["tool_end"]
            else 1.0
        ),
        "mutating_tool_end_count": mutating_tool_ends,
        "workspace_digest_observation_count": workspace_digest_observations,
        "workspace_digest_coverage": (
            workspace_digest_observations / mutating_tool_ends
            if mutating_tool_ends
            else 1.0
        ),
        "all_subagents_returned": bool(children)
        and returned_children == children,
        "all_joins_satisfied": (
            join_create_count == join_satisfied_count
            and counts["join_timeout"] == 0
        ),
        "workflow_lifecycle_valid": (
            counts["workflow_start"] == 1 and counts["workflow_end"] == 1
        ),
    }


def classify_workflow_measurement(
    *,
    outcome: str,
    error: str | None,
    semantic_completion: Mapping[str, Any] | None,
    agent_control: Mapping[str, Any],
    control_delivery: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate system-measurement validity from native agent quality."""

    system_reasons: list[str] = []
    if outcome != "completed":
        system_reasons.append(f"outcome:{outcome}")
    if error is not None:
        system_reasons.append("workflow_error")
    if semantic_completion is None:
        system_reasons.append("missing_semantic_completion")
    if bool(control_delivery.get("degraded")):
        system_reasons.append("runtime_control_delivery_degraded")
    if int(agent_control.get("protocol_repair_failures", 0)):
        system_reasons.append("protocol_repair_failed")
    for field in (
        "workflow_lifecycle_valid",
        "llm_pairing_valid",
        "tool_pairing_valid",
    ):
        if not bool(trace.get(field)):
            system_reasons.append(field)
    if float(trace.get("tool_status_coverage", 0.0)) != 1.0:
        system_reasons.append("incomplete_tool_status_coverage")
    if float(trace.get("workspace_digest_coverage", 0.0)) != 1.0:
        system_reasons.append("incomplete_workspace_digest_coverage")
    if int(trace.get("dynamic_subagent_count", 0)) > 0:
        if not bool(trace.get("all_subagents_returned")):
            system_reasons.append("subagent_not_returned")
        if not bool(trace.get("all_joins_satisfied")):
            system_reasons.append("join_not_satisfied")

    native_reasons = list(system_reasons)
    if agent_control.get("stuck_reasons"):
        native_reasons.append("guard_detected_stuck_execution")
    if int(agent_control.get("forced_semantic_completions", 0)):
        native_reasons.append("forced_semantic_completion")
    if int(agent_control.get("guard_intervened_completions", 0)):
        native_reasons.append("guard_intervened_completion")
    if int(agent_control.get("protocol_repaired_completions", 0)):
        native_reasons.append("protocol_repaired_completion")
    if int(agent_control.get("protocol_normalized_completions", 0)):
        native_reasons.append("protocol_normalized_completion")

    return {
        "system_jct_eligible": not system_reasons,
        "system_jct_exclusion_reasons": system_reasons,
        "native_agent_jct_eligible": not native_reasons,
        "native_agent_jct_exclusion_reasons": native_reasons,
    }


def summarize_agent_control(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    control = [
        item
        for item in records
        if str(item.get("event", "")).startswith("agent_")
    ]
    event_counts = Counter(str(item["event"]) for item in control)
    stuck_reasons = Counter(
        str(item.get("reason", "unknown"))
        for item in control
        if item.get("event") == "agent_stuck_detected"
    )
    semantic = [
        item for item in control if item.get("event") == "agent_semantic_completion"
    ]
    return {
        "event_counts": dict(sorted(event_counts.items())),
        "stuck_reasons": dict(sorted(stuck_reasons.items())),
        "semantic_completions": len(semantic),
        "natural_semantic_completions": sum(
            not bool(item.get("forced", False))
            and not bool(item.get("guard_intervened", False))
            and not bool(item.get("protocol_repaired", False))
            and not bool(item.get("protocol_normalized", False))
            for item in semantic
        ),
        "forced_semantic_completions": sum(
            bool(item.get("forced", False)) for item in semantic
        ),
        "guard_intervened_completions": sum(
            bool(item.get("guard_intervened", False)) for item in semantic
        ),
        "protocol_repaired_completions": sum(
            bool(item.get("protocol_repaired", False)) for item in semantic
        ),
        "protocol_normalized_completions": sum(
            bool(item.get("protocol_normalized", False)) for item in semantic
        ),
        "protocol_repair_failures": event_counts["agent_protocol_repair_failed"],
        "duplicate_tool_calls_suppressed": event_counts[
            "agent_tool_duplicate_suppressed"
        ],
    }


def _run_workflow(
    config: DeepAgentsExperimentConfig,
    bundle: WorkloadBundle,
    workload: SweBenchWorkload,
) -> dict[str, Any]:
    workflow_dir = config.output_dir / "workflows" / workload.instance_id
    workflow_dir.mkdir(parents=True, exist_ok=False)
    workspace = workflow_dir / "workspace"
    if workload.source_repo is None:
        raise RuntimeError(f"workload has no source repository: {workload.instance_id}")
    workspace_metadata = prepare_workspace(workload.source_repo, workload, workspace)
    write_json(workflow_dir / "workspace.json", workspace_metadata)
    trace_path = workflow_dir / "runtime_events.deepagents.jsonl"
    sandbox_audit_path = workflow_dir / "sandbox_audit.jsonl"
    sandbox_audit = JsonlAudit(sandbox_audit_path)
    backend = DockerWorkspaceBackend(
        workspace,
        image=workload.docker_image or config.docker_image,
        audit=sandbox_audit,
        default_timeout_s=config.sandbox_command_timeout_s,
        test_env_path=config.sandbox_test_env_path,
        preflight_command=(
            workload.preflight_command or config.sandbox_preflight_command
        ),
    )
    workflow_token = hashlib.sha256(
        f"{config.output_dir.name}:{config.mode}:{workload.instance_id}".encode()
    ).hexdigest()[:12]
    workflow_id = f"deepagents:{config.mode}:{workload.instance_id}:{workflow_token}"
    root_metadata = BeliefKVRequestMetadata(
        root_workflow_id=workflow_id,
        invocation_id=f"{workflow_id}:root",
        context_id=f"{workflow_id}:context:root",
        context_epoch=0,
        agent_definition_id=(
            "autonomous-supervisor" if config.mode == "autonomous" else "planned-orchestrator"
        ),
        agent_instance_id=f"{workflow_id}:supervisor",
    )
    trace_sink = JsonlRuntimeEventSink(trace_path)
    control_sink = (
        UnixDatagramRuntimeEventSink(
            config.control_socket,
            ack_timeout_s=config.runtime_event_ack_timeout_s,
            retries=config.runtime_event_ack_retries,
        )
        if config.control_socket is not None
        else None
    )
    adapter = DeepAgentsRuntimeAdapter(
        trace_sink,
        root_metadata,
        control_sink=control_sink,
        workspace_digest_provider=backend.tool_state_digest,
    )
    started = time.monotonic()
    outcome = "error"
    error_text: str | None = None
    result: dict[str, Any] = {}
    plan_payload: dict[str, Any] | None = None
    child_reports: list[dict[str, Any]] = []
    semantic_completion: dict[str, Any] | None = None
    completion: WorkflowCompletion | None = None
    try:
        backend.start()
        adapter.start()
        if config.mode == "autonomous":
            result = _run_autonomous(config, workload, backend, adapter)
        else:
            result, plan, child_reports = _run_planned(
                config, workload, backend, adapter, workflow_dir
            )
            plan_payload = plan.model_dump(mode="json")
        if config.completion_gate_enabled:
            result = _repair_incomplete_workflow(
                config, workload, backend, adapter, result
            )
        completion = require_structured_completion(result, WorkflowCompletion)
        semantic_completion = completion.model_dump(mode="json")
        outcome = "completed"
    except BaseException as error:
        if isinstance(error, PartialAgentRunError):
            result = error.partial_result
            error_text = f"{type(error.cause).__name__}: {error.cause}"
        else:
            error_text = f"{type(error).__name__}: {error}"
    finally:
        try:
            adapter.finish(outcome=outcome)
        except BaseException as finish_error:
            if error_text is None:
                error_text = f"{type(finish_error).__name__}: {finish_error}"
                outcome = "error"
        if control_sink is not None:
            control_sink.close()
        trace_sink.close()
        backend.close()
        sandbox_audit.close()

    duration_s = time.monotonic() - started
    messages = _result_messages(result)
    write_json(
        workflow_dir / "trajectory.json",
        [_message_payload(message) for message in messages],
    )
    if plan_payload is not None:
        write_json(workflow_dir / "plan.json", plan_payload)
        write_json(workflow_dir / "child_reports.json", child_reports)
    patch = command_output(["git", "diff", "--binary", "HEAD"], cwd=workspace)
    (workflow_dir / "model.patch").write_text(
        patch + ("\n" if patch else ""), encoding="utf-8"
    )
    final_status = command_output(["git", "status", "--porcelain"], cwd=workspace)
    runtime_verification = result.get(RUNTIME_VERIFIED_TESTS_KEY, {})
    runtime_tests = (
        [str(item) for item in runtime_verification.get("commands", [])]
        if isinstance(runtime_verification, dict)
        else []
    )
    observed_tests = list(
        dict.fromkeys([*observed_successful_test_commands(messages), *runtime_tests])
    )
    correctness_gate = validate_workflow_completion(
        completion,
        patch=patch,
        observed_tests=observed_tests,
    )
    control_delivery = adapter.control_delivery_summary()
    agent_control = summarize_agent_control(sandbox_audit_path)
    trace = _trace_summary(trace_path)
    eligibility = classify_workflow_measurement(
        outcome=outcome,
        error=error_text,
        semantic_completion=semantic_completion,
        agent_control=agent_control,
        control_delivery=control_delivery,
        trace=trace,
    )
    task_correctness_valid = bool(correctness_gate.get("passed")) and not bool(
        control_delivery.get("degraded")
    )
    summary = {
        "schema_version": 1,
        "instance_id": workload.instance_id,
        "repo": workload.repo,
        "base_commit": workload.base_commit,
        "source_repo": str(workload.source_repo) if workload.source_repo else None,
        "docker_image": workload.docker_image or config.docker_image,
        "mode": config.mode,
        "workflow_id": workflow_id,
        "outcome": outcome,
        "error": error_text,
        "duration_seconds": duration_s,
        "final_text": _final_text(result),
        "patch_chars": len(patch),
        "workspace_modified": bool(final_status),
        "final_status": final_status,
        "semantic_completion": semantic_completion,
        "correctness_gate": correctness_gate,
        "task_correctness_valid": task_correctness_valid,
        # Compatibility alias for pre-P5G experiment readers.
        "measurement_valid": task_correctness_valid,
        **eligibility,
        "agent_control": agent_control,
        "runtime_control_delivery": control_delivery,
        "trace": trace,
    }
    write_json(workflow_dir / "result.json", summary)
    return summary


def server_alive(base_url: str, timeout_s: float = 5.0) -> bool:
    root = base_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        with urllib.request.urlopen(f"{root}/get_model_info", timeout=timeout_s):
            return True
    except (OSError, urllib.error.URLError):
        return False


def run_experiment(config: DeepAgentsExperimentConfig) -> dict[str, Any]:
    output_dir = config.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"experiment output already exists: {output_dir}")
    if not server_alive(config.base_url):
        raise RuntimeError(f"SGLang server is not reachable: {config.base_url}")
    if config.control_socket is not None and not config.control_socket.exists():
        raise FileNotFoundError(
            f"BeliefKV control socket is absent: {config.control_socket}"
        )
    server_sources = {
        "runtime_audit": config.server_audit_path,
        "runtime_events": config.server_event_path,
        "server_log": config.server_log_path,
    }
    server_offsets = {
        name: capture_append_offset(path)
        for name, path in server_sources.items()
        if path is not None
    }
    config = replace(config, output_dir=output_dir)
    bundle = load_workload_bundle(config.workload_manifest)
    if config.instance_ids:
        indexed = {item.instance_id: item for item in bundle.workloads}
        unknown = set(config.instance_ids) - set(indexed)
        if unknown:
            raise ValueError(f"unknown SWE-bench instances: {sorted(unknown)}")
        workloads = tuple(indexed[item] for item in config.instance_ids)
    else:
        workloads = bundle.workloads[: config.max_workflows]
    output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "config": {
            **asdict(config),
            "output_dir": str(output_dir),
            "workload_manifest": str(config.workload_manifest),
            "control_socket": (
                str(config.control_socket) if config.control_socket else None
            ),
            "server_audit_path": (
                str(config.server_audit_path) if config.server_audit_path else None
            ),
            "server_event_path": (
                str(config.server_event_path) if config.server_event_path else None
            ),
            "server_log_path": (
                str(config.server_log_path) if config.server_log_path else None
            ),
        },
        "dataset": bundle.dataset,
        "dataset_revision": bundle.dataset_revision,
        "workload_manifest_sha256": bundle.manifest_sha256,
        "instance_ids": [item.instance_id for item in workloads],
        "dynamic_subagent_policy": "runtime-selected; no fixed count",
        "evaluation_scope": (
            "load_and_kv_migration_measurement; official correctness requires "
            "SWE-bench harness"
        ),
        "server_artifact_start_offsets": server_offsets,
    }
    write_json(output_dir / "manifest.json", manifest)
    gpu_monitor = GPUStatsMonitor(config.gpu_index, output_dir / "gpu_samples.csv")
    sglang_monitor = SGLangMetricsMonitor(
        config.base_url,
        output_dir / "sglang_metrics.jsonl",
        pool_tokens=config.pool_tokens,
    )
    started = time.monotonic()
    gpu_monitor.start()
    sglang_monitor.start()
    results: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            futures = {
                executor.submit(_run_workflow, config, bundle, workload): workload
                for workload in workloads
            }
            for future in as_completed(futures):
                workload = futures[future]
                try:
                    results.append(future.result())
                except BaseException as error:
                    results.append(
                        {
                            "instance_id": workload.instance_id,
                            "mode": config.mode,
                            "outcome": "runner_error",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
    finally:
        metrics = sglang_monitor.close()
        gpu_monitor.close()
    # The event socket is acknowledged synchronously, while the audit and event
    # files are line-buffered. A short grace period captures the final scheduler
    # safe point without including a later experiment.
    time.sleep(0.2)
    server_artifacts: dict[str, dict[str, Any]] = {}
    for name, start_offset in server_offsets.items():
        source = server_sources[name]
        assert source is not None
        destination = output_dir / "server" / SERVER_ARTIFACT_FILENAMES[name]
        server_artifacts[name] = copy_append_window(
            source,
            destination,
            start_offset=start_offset,
        )
    server_summary: dict[str, Any] = {}
    if "runtime_audit" in server_artifacts:
        from beliefkv.experiments.codex_ab import summarize_reactive_audit

        server_summary["runtime_audit"] = summarize_reactive_audit(
            Path(server_artifacts["runtime_audit"]["path"])
        )
    if "runtime_events" in server_artifacts:
        from beliefkv.experiments.codex_ab import summarize_runtime_events

        server_summary["runtime_events"] = summarize_runtime_events(
            Path(server_artifacts["runtime_events"]["path"])
        )
    if "server_log" in server_artifacts:
        from beliefkv.experiments.codex_ab import summarize_server_log

        server_summary["server_log"] = summarize_server_log(
            Path(server_artifacts["server_log"]["path"])
        )
    elapsed = time.monotonic() - started
    aggregate_stuck_reasons: Counter[str] = Counter()
    for item in results:
        aggregate_stuck_reasons.update(
            {
                str(reason): int(count)
                for reason, count in item.get("agent_control", {})
                .get("stuck_reasons", {})
                .items()
            }
        )
    summary = {
        "schema_version": 1,
        "mode": config.mode,
        "duration_seconds": elapsed,
        "workflow_count": len(results),
        "completed_workflows": sum(
            item.get("outcome") == "completed" for item in results
        ),
        "successful_workflows": sum(
            bool(item.get("correctness_gate", {}).get("passed")) for item in results
        ),
        "measurement_valid_workflows": sum(
            bool(item.get("measurement_valid")) for item in results
        ),
        "system_jct_eligible_workflows": sum(
            bool(item.get("system_jct_eligible")) for item in results
        ),
        "native_agent_jct_eligible_workflows": sum(
            bool(item.get("native_agent_jct_eligible")) for item in results
        ),
        "dynamic_subagent_count": sum(
            int(item.get("trace", {}).get("dynamic_subagent_count", 0))
            for item in results
        ),
        "llm_request_count": sum(
            int(item.get("trace", {}).get("llm_request_count", 0))
            for item in results
        ),
        "tool_call_count": sum(
            int(item.get("trace", {}).get("tool_call_count", 0))
            for item in results
        ),
        "agent_control": {
            "semantic_completions": sum(
                int(item.get("agent_control", {}).get("semantic_completions", 0))
                for item in results
            ),
            "natural_semantic_completions": sum(
                int(
                    item.get("agent_control", {}).get(
                        "natural_semantic_completions", 0
                    )
                )
                for item in results
            ),
            "forced_semantic_completions": sum(
                int(
                    item.get("agent_control", {}).get(
                        "forced_semantic_completions", 0
                    )
                )
                for item in results
            ),
            "guard_intervened_completions": sum(
                int(
                    item.get("agent_control", {}).get(
                        "guard_intervened_completions", 0
                    )
                )
                for item in results
            ),
            "protocol_repaired_completions": sum(
                int(
                    item.get("agent_control", {}).get(
                        "protocol_repaired_completions", 0
                    )
                )
                for item in results
            ),
            "protocol_normalized_completions": sum(
                int(
                    item.get("agent_control", {}).get(
                        "protocol_normalized_completions", 0
                    )
                )
                for item in results
            ),
            "protocol_repair_failures": sum(
                int(
                    item.get("agent_control", {}).get(
                        "protocol_repair_failures", 0
                    )
                )
                for item in results
            ),
            "duplicate_tool_calls_suppressed": sum(
                int(
                    item.get("agent_control", {}).get(
                        "duplicate_tool_calls_suppressed", 0
                    )
                )
                for item in results
            ),
            "stuck_reasons": dict(sorted(aggregate_stuck_reasons.items())),
        },
        "runtime_control_delivery": {
            "degraded_workflows": sum(
                bool(item.get("runtime_control_delivery", {}).get("degraded"))
                for item in results
            ),
            "failure_count": sum(
                int(
                    item.get("runtime_control_delivery", {}).get(
                        "failure_count", 0
                    )
                )
                for item in results
            ),
        },
        "sglang": metrics,
        "server": server_summary,
        "server_artifacts": server_artifacts,
        "workflows": sorted(results, key=lambda item: str(item["instance_id"])),
    }
    write_json(output_dir / "summary.json", summary)
    return summary
