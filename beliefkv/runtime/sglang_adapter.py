from __future__ import annotations

import ast
import importlib.metadata
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from beliefkv.control.controller import BeliefKVController, ControllerTickResult
from beliefkv.runtime.protocol import CommandAck, PageHandle, ResolvedCommand


BASE_SGLANG_VERSION = "0.5.2rc1"
BASE_SGLANG_GIT_COMMIT = "18f91eb639084825717c0e3c3c7273492812ab71"


@dataclass(frozen=True)
class RuntimeHook:
    name: str
    purpose: str
    expected_file: str


def required_hooks() -> list[RuntimeHook]:
    return [
        RuntimeHook(
            name="request_metadata",
            purpose="carry root workflow, invocation, context, and epoch fields",
            expected_file="python/sglang/srt/managers/io_struct.py",
        ),
        RuntimeHook(
            name="openai_request_metadata",
            purpose="carry causal identity through the OpenAI chat endpoint",
            expected_file="python/sglang/srt/entrypoints/openai/protocol.py",
        ),
        RuntimeHook(
            name="scheduler_safe_point",
            purpose="drain BeliefKV commands before selecting the next batch",
            expected_file="python/sglang/srt/managers/scheduler.py",
        ),
        RuntimeHook(
            name="radix_ownership_events",
            purpose="report node insert, split, delete, lock, and physical index changes",
            expected_file="python/sglang/srt/mem_cache/radix_cache.py",
        ),
        RuntimeHook(
            name="targeted_hicache_actions",
            purpose="prepare, commit, prefetch, and cancel selected sealed extents",
            expected_file="python/sglang/srt/mem_cache/hiradix_cache.py",
        ),
        RuntimeHook(
            name="runtime_flags",
            purpose="enable BeliefKV and configure its control endpoint",
            expected_file="python/sglang/srt/server_args.py",
        ),
    ]


@dataclass(frozen=True)
class BeliefKVRequestMetadata:
    root_workflow_id: str
    invocation_id: str
    context_id: str
    context_epoch: int
    agent_definition_id: str = "unknown"
    agent_instance_id: str = "unknown"
    parent_invocation_id: str | None = None
    parent_context_id: str | None = None
    relation_type: str = "root"
    context_mode: str = "fresh"
    execution_mode: str = "foreground"
    return_target_id: str | None = None
    join_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("root_workflow_id", "invocation_id", "context_id"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")
        if self.context_epoch < 0:
            raise ValueError("context_epoch must be non-negative")
        if self.relation_type not in {"root", "call", "spawn", "message", "handoff"}:
            raise ValueError("unsupported relation_type")
        if self.context_mode not in {"fresh", "fork", "resume"}:
            raise ValueError("unsupported context_mode")
        if self.execution_mode not in {"foreground", "background"}:
            raise ValueError("unsupported execution_mode")

    def to_wire(self) -> dict[str, object]:
        return {
            "root_workflow_id": self.root_workflow_id,
            "invocation_id": self.invocation_id,
            "context_id": self.context_id,
            "context_epoch": self.context_epoch,
            "agent_definition_id": self.agent_definition_id,
            "agent_instance_id": self.agent_instance_id,
            "parent_invocation_id": self.parent_invocation_id,
            "parent_context_id": self.parent_context_id,
            "relation_type": self.relation_type,
            "context_mode": self.context_mode,
            "execution_mode": self.execution_mode,
            "return_target_id": self.return_target_id,
            "join_id": self.join_id,
        }

    @classmethod
    def from_wire(cls, raw: dict[str, object]) -> "BeliefKVRequestMetadata":
        return cls(
            root_workflow_id=str(raw["root_workflow_id"]),
            invocation_id=str(raw["invocation_id"]),
            context_id=str(raw["context_id"]),
            context_epoch=int(raw["context_epoch"]),
            agent_definition_id=str(raw.get("agent_definition_id", "unknown")),
            agent_instance_id=str(raw.get("agent_instance_id", "unknown")),
            parent_invocation_id=(
                str(raw["parent_invocation_id"])
                if raw.get("parent_invocation_id") is not None
                else None
            ),
            parent_context_id=(
                str(raw["parent_context_id"])
                if raw.get("parent_context_id") is not None
                else None
            ),
            relation_type=str(raw.get("relation_type", "root")),
            context_mode=str(raw.get("context_mode", "fresh")),
            execution_mode=str(raw.get("execution_mode", "foreground")),
            return_target_id=(
                str(raw["return_target_id"])
                if raw.get("return_target_id") is not None
                else None
            ),
            join_id=str(raw["join_id"]) if raw.get("join_id") is not None else None,
        )


@dataclass(frozen=True)
class BackendSubmission:
    command_id: str
    started_handles: tuple[PageHandle, ...]


class SGLangCommandBackend(Protocol):
    """Interface implemented inside the patched SGLang scheduler process."""

    def submit(self, command: ResolvedCommand) -> BackendSubmission:
        ...

    def cancel(self, command_id: str) -> None:
        ...

    def poll_acks(self) -> list[CommandAck]:
        ...


class SGLangSchedulerBridge:
    """Drive controller commands through a scheduler-owned backend.

    This object must be called from a scheduler safe point, never from a CUDA
    callback or BeliefKV background thread.
    """

    def __init__(
        self,
        controller: BeliefKVController,
        backend: SGLangCommandBackend,
    ) -> None:
        self.controller = controller
        self.backend = backend

    def scheduler_step(
        self, now_ms: float, *, drain_acks: bool = True
    ) -> ControllerTickResult:
        if drain_acks:
            self.drain_acks()
        tick = self.controller.tick(now_ms)
        for command_id in tick.cancel_command_ids:
            self.backend.cancel(command_id)
        if tick.transfer is not None:
            submission = self.backend.submit(tick.transfer)
            if submission.command_id != tick.transfer.command.command_id:
                raise RuntimeError("backend returned a mismatched command id")
            self.controller.mark_command_started(
                submission.command_id, submission.started_handles
            )
        return tick

    def drain_acks(self) -> tuple[CommandAck, ...]:
        acks = tuple(self.backend.poll_acks())
        for ack in acks:
            self.controller.acknowledge_command(ack)
        return acks


@dataclass(frozen=True)
class SourceContractFailure:
    file: str
    symbol: str
    reason: str


@dataclass(frozen=True)
class SourceContractReport:
    compatible: bool
    failures: tuple[SourceContractFailure, ...]


class SGLangSourceContract:
    """AST-level guard for the exact upstream integration surface."""

    REQUIRED_SYMBOLS: dict[str, dict[str, tuple[str, ...]]] = {
        "python/sglang/srt/managers/io_struct.py": {
            "GenerateReqInput": (),
            "TokenizedGenerateReqInput": (),
        },
        "python/sglang/srt/entrypoints/openai/protocol.py": {
            "ChatCompletionRequest": (),
        },
        "python/sglang/srt/entrypoints/openai/serving_chat.py": {
            "OpenAIServingChat": ("_convert_to_internal_request",),
        },
        "python/sglang/srt/managers/scheduler.py": {
            "Scheduler": (
                "handle_generate_request",
                "get_next_batch_to_run",
                "get_new_batch_prefill",
                "_add_admitted_beliefkv_request",
            ),
        },
        "python/sglang/srt/managers/schedule_batch.py": {
            "Req": ("__init__",),
        },
        "python/sglang/srt/mem_cache/radix_cache.py": {
            "TreeNode": ("__init__",),
            "RadixCache": (
                "_split_node",
                "_insert_helper",
                "_beliefkv_notify",
                "inc_lock_ref",
                "dec_lock_ref",
                "take_events",
            ),
        },
        "python/sglang/srt/mem_cache/hiradix_cache.py": {
            "HiRadixCache": (
                "write_backup",
                "writing_check",
                "_evict_backuped",
                "load_back",
                "check_hicache_events",
            ),
        },
        "python/sglang/srt/server_args.py": {"ServerArgs": ()},
    }
    REQUIRED_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
        "python/sglang/srt/managers/io_struct.py": {
            "GenerateReqInput": ("beliefkv_metadata",),
            "TokenizedGenerateReqInput": ("beliefkv_metadata",),
        },
        "python/sglang/srt/entrypoints/openai/protocol.py": {
            "ChatCompletionRequest": ("beliefkv_metadata",),
        },
        "python/sglang/srt/server_args.py": {
            "ServerArgs": ("enable_beliefkv", "beliefkv_config"),
        },
    }
    REQUIRED_SNIPPETS: dict[str, tuple[str, ...]] = {
        "python/sglang/srt/entrypoints/openai/serving_chat.py": (
            "beliefkv_metadata=request.beliefkv_metadata",
        ),
        "python/sglang/srt/managers/scheduler.py": (
            "beliefkv_runtime.close()",
        ),
    }

    def check(self, source_root: Path) -> SourceContractReport:
        failures: list[SourceContractFailure] = []
        self._check_version(source_root, failures)
        self._check_git_commit(source_root, failures)
        for relative_path, classes in self.REQUIRED_SYMBOLS.items():
            path = source_root / relative_path
            if not path.is_file():
                failures.append(
                    SourceContractFailure(relative_path, "<file>", "missing file")
                )
                continue
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except (OSError, SyntaxError) as exc:
                failures.append(
                    SourceContractFailure(relative_path, "<module>", str(exc))
                )
                continue
            for snippet in self.REQUIRED_SNIPPETS.get(relative_path, ()):
                if snippet not in source:
                    failures.append(
                        SourceContractFailure(
                            relative_path,
                            "<source>",
                            f"missing patched snippet: {snippet}",
                        )
                    )
            class_nodes = {
                node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
            }
            for class_name, methods in classes.items():
                node = class_nodes.get(class_name)
                if node is None:
                    failures.append(
                        SourceContractFailure(relative_path, class_name, "missing class")
                    )
                    continue
                available = {
                    child.name
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                for method in methods:
                    if method not in available:
                        failures.append(
                            SourceContractFailure(
                                relative_path,
                                f"{class_name}.{method}",
                                "missing method",
                            )
                        )
                fields = self.REQUIRED_FIELDS.get(relative_path, {}).get(
                    class_name, ()
                )
                available_fields: set[str] = set()
                for child in node.body:
                    if isinstance(child, ast.AnnAssign) and isinstance(
                        child.target, ast.Name
                    ):
                        available_fields.add(child.target.id)
                    elif isinstance(child, ast.Assign):
                        available_fields.update(
                            target.id
                            for target in child.targets
                            if isinstance(target, ast.Name)
                        )
                for field_name in fields:
                    if field_name not in available_fields:
                        failures.append(
                            SourceContractFailure(
                                relative_path,
                                f"{class_name}.{field_name}",
                                "missing patched field",
                            )
                        )
        return SourceContractReport(not failures, tuple(failures))

    @staticmethod
    def _check_version(
        source_root: Path, failures: list[SourceContractFailure]
    ) -> None:
        relative_path = "python/sglang/version.py"
        path = source_root / relative_path
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            version = next(
                ast.literal_eval(node.value)
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "__version__"
                    for target in node.targets
                )
            )
        except (OSError, SyntaxError, StopIteration, ValueError):
            failures.append(
                SourceContractFailure(relative_path, "__version__", "missing version")
            )
            return
        if version != BASE_SGLANG_VERSION:
            failures.append(
                SourceContractFailure(
                    relative_path,
                    "__version__",
                    f"expected {BASE_SGLANG_VERSION}, found {version}",
                )
            )

    @staticmethod
    def _check_git_commit(
        source_root: Path, failures: list[SourceContractFailure]
    ) -> None:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=source_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return
        if commit != BASE_SGLANG_GIT_COMMIT:
            failures.append(
                SourceContractFailure(
                    ".git",
                    "HEAD",
                    f"expected {BASE_SGLANG_GIT_COMMIT}, found {commit}",
                )
            )


def installed_sglang_version() -> str | None:
    try:
        return importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        return None


def assert_supported_sglang_version(version: str | None = None) -> None:
    actual = installed_sglang_version() if version is None else version
    if actual != BASE_SGLANG_VERSION:
        raise RuntimeError(
            f"BeliefKV requires SGLang {BASE_SGLANG_VERSION}; found {actual or 'not installed'}"
        )
