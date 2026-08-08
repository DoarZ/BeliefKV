from __future__ import annotations

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import PrivateAttr


pytest.importorskip("deepagents")

from deepagents.backends.protocol import ExecuteResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest
from langchain.agents.structured_output import ToolStrategy

from beliefkv.experiments.agent_protocol import (
    ActivationDeadline,
    ActivationDeadlineExceeded,
    AgentLoopGuardMiddleware,
    ChildCompletion,
    LoopGuardPolicy,
    TerminalProtocolError,
    WorkflowCompletion,
    analyze_agent_history,
    classify_tool_outcome,
    require_structured_completion,
)
from beliefkv.experiments.deepagents_swebench import (
    DeepAgentsExperimentConfig,
    DockerWorkspaceBackend,
    DelegationPlan,
    JsonlAudit,
    PartialAgentRunError,
    SANDBOX_PATH_CONTRACT,
    SYMPY_SANDBOX_PREFLIGHT,
    SweBenchWorkload,
    capture_append_offset,
    classify_workflow_measurement,
    copy_append_window,
    load_workload_bundle,
    observed_successful_test_commands,
    prepare_workspace,
    prometheus_gauge_sum,
    repository_sandbox_contract,
    summarize_agent_control,
    validate_workflow_completion,
    _trace_summary,
    _invoke_with_partial_state,
    _filesystem_middleware,
    _autonomous_subagents,
    _runtime_verify_changed_tests,
    _planned_child_loop_guard_policy,
    _workspace_patch_tool,
)
from beliefkv.runtime.langchain_tool_safety import (
    ToolCircuitBreakerMiddleware,
    ToolObservationBudgetMiddleware,
    ToolObservationBudgetPolicy,
    ToolOutcomeStatusMiddleware,
)
from beliefkv.runtime.context_lifecycle import (
    CompletionBudgetMiddleware,
    ContextLifecycleMiddleware,
    ContextLifecyclePolicy,
)
from langchain.agents.middleware.types import ToolCallRequest


def test_prometheus_gauge_sum_aggregates_labeled_series() -> None:
    payload = """\
sglang:num_used_tokens{tp_rank="0"} 123
sglang:num_used_tokens{tp_rank="1"} 45
sglang:num_running_reqs{tp_rank="0"} 2
"""
    assert prometheus_gauge_sum(payload, "sglang:num_used_tokens") == 168
    assert prometheus_gauge_sum(payload, "missing") is None


def test_agent_control_summary_separates_protocol_and_guard_outcomes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.jsonl"
    records = [
        {"event": "agent_semantic_completion", "forced": False},
        {
            "event": "agent_semantic_completion",
            "forced": False,
            "protocol_repaired": True,
        },
        {
            "event": "agent_semantic_completion",
            "forced": False,
            "guard_intervened": True,
        },
        {"event": "agent_protocol_repair_failed"},
        {"event": "agent_tool_duplicate_suppressed"},
    ]
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )

    summary = summarize_agent_control(path)

    assert summary["semantic_completions"] == 3
    assert summary["natural_semantic_completions"] == 1
    assert summary["protocol_repaired_completions"] == 1
    assert summary["guard_intervened_completions"] == 1
    assert summary["protocol_repair_failures"] == 1
    assert summary["duplicate_tool_calls_suppressed"] == 1


def test_direct_runtime_trace_reports_pairing_and_subagent_lifecycle(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    records = [
        {"kind": "workflow_start"},
        {
            "kind": "invocation_create",
            "invocation_id": "child",
            "parent_invocation_id": "root",
        },
        {"kind": "spawn", "target_invocation_id": "child"},
        {"kind": "join_create"},
        {"kind": "llm_submit", "invocation_id": "child"},
        {"kind": "llm_result", "invocation_id": "child"},
        {
            "kind": "tool_start",
            "attributes": {"tool_name": "edit_file"},
        },
        {
            "kind": "tool_end",
            "attributes": {
                "tool_name": "edit_file",
                "status": "success",
                "workspace_digest_before": "before",
                "workspace_digest_after": "after",
            },
        },
        {"kind": "return", "invocation_id": "child"},
        {"kind": "join_satisfied"},
        {"kind": "workflow_end"},
    ]
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )

    summary = _trace_summary(path)

    assert summary["llm_pairing_valid"]
    assert summary["tool_pairing_valid"]
    assert summary["tool_status_coverage"] == 1.0
    assert summary["workspace_digest_coverage"] == 1.0
    assert summary["all_subagents_returned"]
    assert summary["all_joins_satisfied"]
    assert summary["workflow_lifecycle_valid"]


def test_system_jct_allows_recovered_protocol_but_native_jct_does_not() -> None:
    result = classify_workflow_measurement(
        outcome="completed",
        error=None,
        semantic_completion={"status": "blocked"},
        agent_control={
            "protocol_repair_failures": 0,
            "protocol_repaired_completions": 1,
            "protocol_normalized_completions": 0,
            "guard_intervened_completions": 0,
            "forced_semantic_completions": 0,
            "stuck_reasons": {},
        },
        control_delivery={"degraded": False},
        trace={
            "workflow_lifecycle_valid": True,
            "llm_pairing_valid": True,
            "tool_pairing_valid": True,
            "tool_status_coverage": 1.0,
            "workspace_digest_coverage": 1.0,
            "dynamic_subagent_count": 1,
            "all_subagents_returned": True,
            "all_joins_satisfied": True,
        },
    )

    assert result["system_jct_eligible"]
    assert not result["native_agent_jct_eligible"]
    assert "protocol_repaired_completion" in result[
        "native_agent_jct_exclusion_reasons"
    ]


def test_copy_append_window_freezes_only_new_bytes(tmp_path: Path) -> None:
    source = tmp_path / "server.jsonl"
    source.write_bytes(b'{"old":1}\n')
    offset = capture_append_offset(source)
    with source.open("ab") as stream:
        stream.write(b'{"new":2}\n')
    destination = tmp_path / "run" / "server.jsonl"

    metadata = copy_append_window(
        source,
        destination,
        start_offset=offset,
    )

    assert destination.read_bytes() == b'{"new":2}\n'
    assert metadata["start_offset"] == len(b'{"old":1}\n')
    assert metadata["end_offset"] == source.stat().st_size
    assert metadata["byte_count"] == len(b'{"new":2}\n')
    assert metadata["sha256"]


def test_load_bundle_and_prepare_exact_commit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=source, check=True)
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "module.py"], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=BeliefKV Test",
            "-c",
            "user.email=beliefkv@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "test fixture",
        ],
        cwd=source,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = tmp_path / "workloads.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "fixture",
                "dataset_revision": "r1",
                "source_repo": str(source),
                "workloads": [
                    {
                        "instance_id": "fixture-1",
                        "repo": "fixture/repo",
                        "base_commit": commit,
                        "problem_statement": "Fix VALUE.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    bundle = load_workload_bundle(manifest)
    destination = tmp_path / "workspace"
    metadata = prepare_workspace(source, bundle.workloads[0], destination)
    assert metadata["initial_head"] == commit
    assert (destination / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_docker_backend_hashes_commands_and_truncates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = JsonlAudit(tmp_path / "audit.jsonl")
    backend = DockerWorkspaceBackend(
        tmp_path,
        image="fixture:latest",
        audit=audit,
        max_output_chars=4,
    )
    backend._started = True

    invocations = []

    def fake_run(*args, **kwargs):
        invocations.append((args, kwargs))
        return subprocess.CompletedProcess([], 0, stdout="abcdef", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    response = backend.execute("printf abcdef")
    backend.close()
    audit.close()
    assert response.truncated
    records = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    execute = next(item for item in records if item["event"] == "sandbox_execute")
    assert execute["command_chars"] == len("printf abcdef")
    assert "command" not in execute
    execute_argv = invocations[0][0][0]
    assert execute_argv[:4] == ["docker", "exec", "--workdir", "/workspace"]
    assert "PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:" in " ".join(
        execute_argv
    )
    assert execute_argv[-3:-1] == ["/bin/sh", "-c"]


def test_tool_observation_budget_bounds_parallel_turn_deterministically(
    tmp_path: Path,
) -> None:
    audit = JsonlAudit(tmp_path / "audit.jsonl")
    policy = ToolObservationBudgetPolicy(
        total_chars_per_turn=65_536,
        max_chars_per_result=16_384,
    )
    middleware = ToolObservationBudgetMiddleware(
        policy=policy,
        audit=audit,
        scope="observation-budget-test",
    )
    calls = [
        {"name": "execute", "args": {"command": f"probe {index}"}, "id": f"c{index}"}
        for index in range(50)
    ]
    state = {"messages": [AIMessage(content="", tool_calls=calls)]}
    visible: list[str] = []
    for call in calls:
        request = ToolCallRequest(
            tool_call=call,
            tool=None,
            state=state,
            runtime=SimpleNamespace(config={}),
        )
        result = middleware.wrap_tool_call(
            request,
            lambda current: ToolMessage(
                content=("head-" + "x" * 70_000 + "-tail"),
                tool_call_id=str(current.tool_call["id"]),
                name="execute",
                status="success",
            ),
        )
        assert isinstance(result, ToolMessage)
        assert result.status == "success"
        assert result.additional_kwargs["beliefkv_observation_truncated"] is True
        visible.append(str(result.content))
    audit.close()

    assert sum(map(len, visible)) <= policy.total_chars_per_turn
    assert all(len(item) == policy.total_chars_per_turn // len(calls) for item in visible)
    assert all("BeliefKV observation truncated" in item for item in visible)
    assert all(item.endswith("-tail") for item in visible)
    records = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == len(calls)
    assert {item["turn_tool_fanout"] for item in records} == {len(calls)}
    assert {item["per_result_budget_chars"] for item in records} == {1310}


def test_docker_backend_deadline_cancel_stops_workflow_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = JsonlAudit(tmp_path / "audit.jsonl")
    backend = DockerWorkspaceBackend(tmp_path, image="fixture:latest", audit=audit)
    backend._started = True
    invocations: list[list[str]] = []

    def fake_run(argv, **kwargs):
        del kwargs
        invocations.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="fixture", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert backend.cancel_active_commands(reason="workflow_timeout") == 1
    assert backend.cancel_active_commands(reason="workflow_timeout") == 0
    audit.close()

    assert invocations == [["docker", "kill", backend.id]]
    records = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cancel = next(item for item in records if item["event"] == "sandbox_cancel")
    assert cancel["reason"] == "workflow_timeout"
    assert cancel["returncode"] == 0


def test_workspace_digest_detects_mutating_tool_changes(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "module.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=BeliefKV Test",
            "-c",
            "user.email=beliefkv@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    audit = JsonlAudit(tmp_path / "audit.jsonl")
    backend = DockerWorkspaceBackend(tmp_path, image="fixture:latest", audit=audit)

    before = backend.tool_state_digest("edit_file", {"path": "module.py"})
    source.write_text("VALUE = 2\n", encoding="utf-8")
    after = backend.tool_state_digest("edit_file", {"path": "module.py"})

    assert before is not None
    assert after is not None
    assert before != after
    assert backend.tool_state_digest("read_file", {"path": "module.py"}) is None
    audit.close()


def test_workspace_epoch_advances_only_after_successful_mutations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    audit = JsonlAudit(tmp_path / "audit.jsonl")
    backend = DockerWorkspaceBackend(tmp_path, image="fixture:latest", audit=audit)

    assert backend.workspace_epoch() == 0
    edited = backend.edit("/workspace/module.py", "VALUE = 1", "VALUE = 2")
    assert getattr(edited, "error", None) is None
    assert backend.workspace_epoch() == 1

    failed = backend.edit("/workspace/module.py", "MISSING", "VALUE = 3")
    assert getattr(failed, "error", None) is not None
    assert backend.workspace_epoch() == 1

    written = backend.write("/workspace/new.py", "NEW = True\n")
    assert getattr(written, "error", None) is None
    assert backend.workspace_epoch() == 2
    audit.close()


def test_docker_backend_preflights_test_environment_before_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = JsonlAudit(tmp_path / "audit.jsonl")
    backend = DockerWorkspaceBackend(
        tmp_path,
        image="fixture:latest",
        audit=audit,
        preflight_command=SYMPY_SANDBOX_PREFLIGHT,
    )
    invocations: list[list[str]] = []

    def fake_run(argv, **kwargs):
        del kwargs
        invocations.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend.start()
    backend.close()
    audit.close()

    assert invocations[0][:2] == ["docker", "run"]
    assert invocations[1][:4] == ["docker", "exec", "--workdir", "/workspace"]
    assert SYMPY_SANDBOX_PREFLIGHT in invocations[1][-1]
    assert invocations[2][:3] == ["docker", "rm", "--force"]
    records = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    preflight = next(item for item in records if item["event"] == "sandbox_preflight")
    assert preflight["returncode"] == 0
    assert preflight["expected_python"] == "/opt/miniconda3/envs/testbed/bin/python"


def test_docker_backend_uses_workspace_paths_for_file_and_shell_tools(
    tmp_path: Path,
) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    audit = JsonlAudit(tmp_path / "audit.jsonl")
    backend = DockerWorkspaceBackend(tmp_path, image="fixture:latest", audit=audit)

    listing = backend.ls("/")
    assert listing.entries is not None
    paths = [item["path"] for item in listing.entries]
    assert "/workspace/module.py" in paths
    read_result = backend.read("/workspace/module.py")
    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "VALUE = 1\n"
    assert backend._resolve_path("/workspace/module.py") == tmp_path / "module.py"
    audit.close()


def test_workspace_patch_tool_checks_then_applies_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = JsonlAudit(tmp_path / "audit.jsonl")
    backend = DockerWorkspaceBackend(tmp_path, image="fixture:latest", audit=audit)
    commands: list[str] = []

    def fake_execute(command: str, *, timeout: int | None = None):
        del timeout
        commands.append(command)
        patch_files = list(tmp_path.glob(".beliefkv-patch-*.diff"))
        assert len(patch_files) == 1
        normalized = patch_files[0].read_text(encoding="utf-8")
        assert "diff --git a/module.py b/module.py" in normalized
        assert normalized.endswith("\n")
        return ExecuteResponse(output="", exit_code=0, truncated=False)

    monkeypatch.setattr(backend, "execute", fake_execute)
    patch_tool = _workspace_patch_tool(backend)
    result = patch_tool.invoke(
        {
            "patch": (
                "diff --git a/module.py b/module.py\n"
                "--- a/module.py\n"
                "+++ b/module.py\n"
                "@@ -1 +1 @@\n"
                "-OLD\n"
                "+NEW"
            )
        }
    )

    assert result == "Patch applied successfully."
    assert len(commands) == 2
    assert "git apply --check" in commands[0]
    assert not list(tmp_path.glob(".beliefkv-patch-*.diff"))
    audit.close()


def test_patch_only_filesystem_middleware_removes_direct_edit_tools(
    tmp_path: Path,
) -> None:
    audit = JsonlAudit(tmp_path / "audit.jsonl")
    backend = DockerWorkspaceBackend(tmp_path, image="fixture:latest", audit=audit)
    middleware = _filesystem_middleware(backend, allow_direct_edits=False)

    tool_names = {item.name for item in middleware.tools}
    assert {"ls", "read_file", "glob", "grep", "execute"} <= tool_names
    assert "write_file" not in tool_names
    assert "edit_file" not in tool_names
    audit.close()


def test_writable_filesystem_middleware_exposes_sandboxed_edit_tools(
    tmp_path: Path,
) -> None:
    audit = JsonlAudit(tmp_path / "audit.jsonl")
    backend = DockerWorkspaceBackend(tmp_path, image="fixture:latest", audit=audit)
    middleware = _filesystem_middleware(backend, allow_direct_edits=True)

    tool_names = {item.name for item in middleware.tools}
    assert {"write_file", "edit_file", "execute"} <= tool_names
    audit.close()


def test_runtime_verifier_runs_changed_repository_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_file = tmp_path / "package/tests/test_feature.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_feature(): pass\n", encoding="utf-8")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/test").write_text("", encoding="utf-8")
    audit = JsonlAudit(tmp_path / "audit.jsonl")
    backend = DockerWorkspaceBackend(tmp_path, image="fixture:latest", audit=audit)
    commands: list[str] = []

    def fake_command_output(command, *, cwd, timeout=60.0):
        del cwd, timeout
        if "--name-only" in command:
            return "package/tests/test_feature.py"
        return "diff --git a/package/tests/test_feature.py b/package/tests/test_feature.py"

    def fake_execute(command: str, *, timeout: int | None = None):
        assert timeout == 600
        commands.append(command)
        return ExecuteResponse(output="1 passed", exit_code=0, truncated=False)

    monkeypatch.setattr(
        "beliefkv.experiments.deepagents_swebench.command_output",
        fake_command_output,
    )
    monkeypatch.setattr(backend, "execute", fake_execute)
    result: dict[str, object] = {}
    observed = _runtime_verify_changed_tests(result, backend)

    assert observed == [
        "python bin/test package/tests/test_feature.py"
    ]
    assert commands == observed
    assert _runtime_verify_changed_tests(result, backend) == observed
    assert commands == observed
    audit.close()


def test_workflow_success_requires_patch_and_observed_passing_test() -> None:
    passing_command = "python bin/test sympy/core/tests/test_basic.py -k test_args"
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute",
                    "args": {"command": passing_command},
                    "id": "test-pass",
                },
                {
                    "name": "execute",
                    "args": {"command": "pytest failing_test.py"},
                    "id": "test-fail",
                },
            ],
        ),
        ToolMessage(
            content="3 passed in 0.4 seconds",
            tool_call_id="test-pass",
            name="execute",
        ),
        ToolMessage(
            content="1 failed\n[Command failed with exit code 1]",
            tool_call_id="test-fail",
            name="execute",
        ),
    ]
    observed = observed_successful_test_commands(messages)
    assert observed == [passing_command]

    completion = WorkflowCompletion(
        status="patched_and_tested",
        summary="Fixed the issue",
        files_changed=["sympy/core/basic.py"],
        tests=[f"{passing_command}: passed"],
        unresolved=[],
    )
    accepted = validate_workflow_completion(
        completion,
        patch="diff --git a/sympy/core/basic.py b/sympy/core/basic.py",
        observed_tests=observed,
    )
    assert accepted["passed"]

    unresolved = validate_workflow_completion(
        completion.model_copy(update={"unresolved": ["second requirement missing"]}),
        patch="diff --git a/sympy/core/basic.py b/sympy/core/basic.py",
        observed_tests=observed,
    )
    assert not unresolved["passed"]
    assert "completion_has_unresolved_items" in unresolved["errors"]

    self_declared_incomplete = validate_workflow_completion(
        completion.model_copy(
            update={
                "summary": "The second requirement requires additional implementation."
            }
        ),
        patch="diff --git a/sympy/core/basic.py b/sympy/core/basic.py",
        observed_tests=observed,
    )
    assert not self_declared_incomplete["passed"]
    assert "completion_summary_declares_incomplete_work" in self_declared_incomplete[
        "errors"
    ]

    rejected = validate_workflow_completion(
        completion.model_copy(update={"status": "blocked"}),
        patch="",
        observed_tests=[],
    )
    assert not rejected["passed"]
    assert "terminal_status:blocked" in rejected["errors"]
    assert "workspace_has_no_patch" in rejected["errors"]


@pytest.mark.parametrize(
    ("command", "output"),
    [
        (
            "python bin/test sympy/core/tests/test_arit.py::test_Mod",
            "tests finished: 0 passed, in 0.00 seconds\n"
            "[Command succeeded with exit code 0]",
        ),
        (
            "pytest sympy/core/tests/test_arit.py -k does_not_exist",
            "collected 0 items\n[Command succeeded with exit code 0]",
        ),
        (
            "python -m pytest sympy/core/tests/test_arit.py -k does_not_exist",
            "no tests ran in 0.01s\n[Command succeeded with exit code 0]",
        ),
    ],
)
def test_workflow_gate_rejects_successful_exit_with_zero_tests(
    command: str, output: str
) -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute",
                    "args": {"command": command},
                    "id": "zero-tests",
                }
            ],
        ),
        ToolMessage(
            content=output,
            tool_call_id="zero-tests",
            name="execute",
        ),
    ]

    assert observed_successful_test_commands(messages) == []


def test_experiment_config_rejects_unsupported_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode"):
        DeepAgentsExperimentConfig(
            mode="hybrid",
            base_url="http://localhost:18000/v1",
            model="model",
            output_dir=tmp_path,
            workload_manifest=tmp_path / "workloads.json",
            docker_image="fixture:latest",
        )


def test_experiment_config_uses_hard_fuse_as_langgraph_limit(tmp_path: Path) -> None:
    config = DeepAgentsExperimentConfig(
        mode="planned",
        base_url="http://localhost:18000/v1",
        model="model",
        output_dir=tmp_path,
        workload_manifest=tmp_path / "workloads.json",
        docker_image="fixture:latest",
    )
    assert config.recursion_limit == 512
    assert config.sampling_seed is None
    assert config.loop_guard.enabled
    assert config.completion_gate_enabled is True
    assert config.completion_repair_attempts == 2
    assert config.runtime_event_ack_timeout_s == 10.0
    assert config.runtime_event_ack_retries == 3
    assert config.context_lifecycle.window_tokens == 32_768
    assert config.context_lifecycle.keep_tokens == 8_192
    assert config.sandbox_preflight_command is None
    assert "/workspace" in SANDBOX_PATH_CONTRACT
    assert "sympy/core/basic.py" not in SANDBOX_PATH_CONTRACT


def test_experiment_config_rejects_negative_sampling_seed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sampling_seed"):
        DeepAgentsExperimentConfig(
            mode="planned",
            base_url="http://localhost:18000/v1",
            model="model",
            output_dir=tmp_path,
            workload_manifest=tmp_path / "workloads.json",
            docker_image="fixture:latest",
            sampling_seed=-1,
        )


def test_repository_contract_does_not_duplicate_django_checkout_root() -> None:
    workload = SweBenchWorkload(
        instance_id="django__django-11138",
        repo="django/django",
        base_commit="deadbeef",
        problem_statement="Fix the database backend.",
        difficulty="unknown",
    )

    contract = repository_sandbox_contract(workload)

    assert "checkout root is still exactly\n  `/workspace`" in contract
    assert "/workspace/django/db/backends/base/base.py" in contract
    assert "/workspace/django/django/db/backends/base/base.py" in contract
    assert "not\n  `/workspace/django/django" in contract
    assert "python tests/runtests.py" in contract


def test_autonomous_subagents_have_independent_context_lifecycles(
    tmp_path: Path,
) -> None:
    config = DeepAgentsExperimentConfig(
        mode="autonomous",
        base_url="http://localhost:18000/v1",
        model="model",
        output_dir=tmp_path / "output",
        workload_manifest=tmp_path / "workloads.json",
        docker_image="fixture:latest",
        max_completion_tokens=4_096,
        context_lifecycle=ContextLifecyclePolicy(
            window_tokens=32_768,
            keep_tokens=8_192,
            intermediate_output_tokens=4_096,
            summary_output_tokens=2_048,
        ),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    audit = JsonlAudit(tmp_path / "sandbox.jsonl")
    backend = DockerWorkspaceBackend(
        workspace,
        image="fixture:latest",
        audit=audit,
        support_dir=None,
    )
    adapter = SimpleNamespace(record_call_censor=lambda _record: None)
    model = FakeMessagesListChatModel(responses=[AIMessage(content="done")])
    try:
        subagents = _autonomous_subagents(
            config,
            SweBenchWorkload(
                instance_id="django__django-11138",
                repo="django/django",
                base_commit="deadbeef",
                problem_statement="Fix the database backend.",
                difficulty="unknown",
            ),
            backend,
            adapter,
            model,
            model,
        )
    finally:
        audit.close()

    lifecycles = [
        next(
            item
            for item in spec["middleware"]
            if isinstance(item, ContextLifecycleMiddleware)
        )
        for spec in subagents
    ]
    assert len(lifecycles) == 4
    assert len({id(item) for item in lifecycles}) == len(lifecycles)
    assert all(not item.persist_cursor_across_invocations for item in lifecycles)
    assert all(
        "/workspace/django/db/backends/base/base.py" in spec["system_prompt"]
        for spec in subagents
    )
    assert all(item.policy.window_tokens == 32_768 for item in lifecycles)
    assert all(
        any(
            isinstance(item, CompletionBudgetMiddleware)
            and item.intermediate_tokens == 4_096
            and item.final_tokens == 4_096
            for item in spec["middleware"]
        )
        for spec in subagents
    )


@pytest.mark.parametrize(
    ("ack_timeout_s", "ack_retries"),
    [(0.0, 3), (10.0, 0)],
)
def test_experiment_config_rejects_invalid_runtime_event_ack_policy(
    tmp_path: Path,
    ack_timeout_s: float,
    ack_retries: int,
) -> None:
    with pytest.raises(ValueError, match="runtime-event ACK"):
        DeepAgentsExperimentConfig(
            mode="planned",
            base_url="http://localhost:18000/v1",
            model="model",
            output_dir=tmp_path,
            workload_manifest=tmp_path / "workloads.json",
            docker_image="fixture:latest",
            runtime_event_ack_timeout_s=ack_timeout_s,
            runtime_event_ack_retries=ack_retries,
        )


def test_planner_schema_uses_qwen_compatible_flat_tasks() -> None:
    plan = DelegationPlan.model_validate(
        {"rationale": "Parallel repository evidence", "tasks": ["Inspect parser"]}
    )
    assert plan.tasks == ["Inspect parser"]

    with pytest.raises(ValueError):
        DelegationPlan.model_validate(
            {
                "rationale": "Too many overlapping tasks",
                "tasks": ["one", "two", "three"],
            }
        )


def test_planned_children_use_a_shorter_analysis_budget(tmp_path: Path) -> None:
    config = DeepAgentsExperimentConfig(
        mode="planned",
        base_url="http://localhost:18000/v1",
        model="model",
        output_dir=tmp_path,
        workload_manifest=tmp_path / "workloads.json",
        docker_image="fixture:latest",
        loop_guard=LoopGuardPolicy(
            repeated_call_limit=5,
            max_model_calls_without_completion=32,
            max_tool_calls_without_completion=64,
        ),
    )

    policy = _planned_child_loop_guard_policy(config)
    assert policy.repeated_call_limit == 3
    assert policy.max_model_calls_without_completion == 12
    assert policy.max_tool_calls_without_completion == 16


def test_agent_stream_failure_preserves_latest_state() -> None:
    class FailingAgent:
        def stream(self, inputs, *, config, stream_mode):
            assert inputs == {"messages": []}
            assert config == {"recursion_limit": 2}
            assert stream_mode == "values"
            yield {"messages": ["partial"]}
            raise RuntimeError("limit")

    with pytest.raises(PartialAgentRunError) as caught:
        _invoke_with_partial_state(
            FailingAgent(), {"messages": []}, {"recursion_limit": 2}
        )
    assert caught.value.partial_result == {"messages": ["partial"]}


def _tool_exchange(
    name: str,
    args: dict[str, object],
    call_id: str,
    output: str,
) -> list[object]:
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": call_id}],
        ),
        ToolMessage(content=output, tool_call_id=call_id, name=name),
    ]


def test_loop_guard_detects_repeated_and_alternating_calls() -> None:
    policy = LoopGuardPolicy()
    repeated = []
    for index in range(3):
        repeated.extend(_tool_exchange("ls", {"path": "/src"}, str(index), "same"))
    assert analyze_agent_history(repeated, policy).reason == "repeated_tool_call"

    alternating = []
    for index, path in enumerate(("/a", "/b", "/a", "/b", "/a", "/b")):
        alternating.extend(_tool_exchange("read_file", {"path": path}, str(index), path))
    assert analyze_agent_history(alternating, policy).reason == "alternating_tool_cycle"


def test_loop_guard_resets_after_a_persistent_thread_completion() -> None:
    old_activation = []
    for index in range(4):
        old_activation.extend(
            _tool_exchange("read_file", {"path": f"/old-{index}"}, str(index), "old")
        )
    old_activation.extend(
        _tool_exchange(
            "AgenticPeerDecision",
            {"complete": False, "next_role": "reviewer"},
            "peer-decision",
            "accepted",
        )
    )
    current_activation = _tool_exchange(
        "read_file", {"path": "/current"}, "current", "new evidence"
    )

    snapshot = analyze_agent_history(
        [*old_activation, HumanMessage(content="Resume"), *current_activation],
        LoopGuardPolicy(max_model_calls_without_completion=4),
        completion_tool_names=frozenset({"AgenticPeerDecision"}),
    )

    assert snapshot.model_calls == 1
    assert snapshot.tool_calls == 1
    assert snapshot.reason is None


def test_loop_guard_detects_errors_no_progress_and_completion_budget() -> None:
    errors = []
    for index in range(3):
        errors.extend(
            _tool_exchange(
                "read_file",
                {"path": f"/missing-{index}"},
                str(index),
                "Error: path_not_found",
            )
        )
    assert analyze_agent_history(errors, LoopGuardPolicy()).reason == (
        "consecutive_tool_errors"
    )

    source_reads = []
    for index in range(3):
        source_reads.extend(
            _tool_exchange(
                "read_file",
                {"path": f"/source-{index}"},
                str(index),
                "def timeout_handler():\n    return 'error: documented value'",
            )
        )
    source_snapshot = analyze_agent_history(source_reads, LoopGuardPolicy())
    assert source_snapshot.consecutive_errors == 0

    no_progress_policy = LoopGuardPolicy(
        repeated_call_limit=99,
        consecutive_no_progress_limit=3,
    )
    no_progress = []
    for index in range(4):
        no_progress.extend(
            _tool_exchange("grep", {"pattern": "same"}, str(index), "same output")
        )
    assert analyze_agent_history(no_progress, no_progress_policy).reason == (
        "no_observable_progress"
    )

    budget_policy = LoopGuardPolicy(
        max_model_calls_without_completion=4,
        enforce_call_budgets=True,
    )
    exploring = []
    for index in range(4):
        exploring.extend(
            _tool_exchange(
                "read_file",
                {"path": f"/new-{index}"},
                str(index),
                f"unique output {index}",
            )
        )
    assert analyze_agent_history(exploring, budget_policy).reason == (
        "completion_budget_exhausted"
    )

    tool_budget_policy = LoopGuardPolicy(
        max_model_calls_without_completion=99,
        max_tool_calls_without_completion=4,
        enforce_call_budgets=True,
    )
    multi_tool = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "probe",
                    "args": {"path": f"/{index}"},
                    "id": f"probe-{index}",
                }
                for index in range(4)
            ],
        )
    ]
    assert analyze_agent_history(multi_tool, tool_budget_policy).reason == (
        "tool_call_budget_exhausted"
    )

    assert analyze_agent_history(
        exploring,
        LoopGuardPolicy(max_model_calls_without_completion=4),
    ).reason is None


def test_structured_tool_error_triggers_failed_call_circuit_breaker() -> None:
    messages = []
    for index in range(2):
        call_id = f"edit-{index}"
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "edit_file",
                            "args": {
                                "file_path": "/workspace/module.py",
                                "old_string": "old",
                                "new_string": "new",
                            },
                            "id": call_id,
                        }
                    ],
                ),
                ToolMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Error: String not found in file: 'old'",
                        }
                    ],
                    tool_call_id=call_id,
                    name="edit_file",
                    status="error",
                ),
            ]
        )

    outcome = classify_tool_outcome(messages[-1], tool_name="edit_file")
    snapshot = analyze_agent_history(messages, LoopGuardPolicy())

    assert outcome.status == "error"
    assert outcome.error_class == "string_not_found"
    assert snapshot.repeated_failed_calls == 2
    assert snapshot.reason == "repeated_failed_tool_call"


def test_suppressed_retry_is_not_counted_as_second_physical_failure() -> None:
    args = {"pattern": 401, "path": "/workspace"}
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"name": "grep", "args": args, "id": "grep-1"}],
        ),
        ToolMessage(
            content="Input should be a valid string",
            tool_call_id="grep-1",
            name="grep",
            status="error",
            additional_kwargs={
                "beliefkv_error_class": "validation_error",
                "beliefkv_physical_execution": True,
                "beliefkv_suppressed_repeat_intent": False,
                "beliefkv_failure_episode_id": "failure-1",
            },
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "grep", "args": args, "id": "grep-2"}],
        ),
        ToolMessage(
            content="duplicate_suppressed",
            tool_call_id="grep-2",
            name="grep",
            status="error",
            additional_kwargs={
                "beliefkv_error_class": "duplicate_suppressed",
                "beliefkv_physical_execution": False,
                "beliefkv_suppressed_repeat_intent": True,
                "beliefkv_failure_episode_id": "failure-1",
            },
        ),
    ]

    snapshot = analyze_agent_history(
        messages,
        LoopGuardPolicy(
            repeated_call_limit=99,
            consecutive_error_limit=99,
            consecutive_no_progress_limit=99,
        ),
    )

    assert snapshot.physical_failure_count == 1
    assert snapshot.suppressed_repeat_intent_count == 1
    assert snapshot.repeated_failed_calls == 1
    assert snapshot.reason is None


def test_tool_status_middleware_preserves_content_and_marks_semantic_error() -> None:
    original = ToolMessage(
        content="output\n[Command failed with exit code 1]",
        tool_call_id="execute-1",
        name="execute",
        status="success",
    )
    request = type(
        "FixtureRequest",
        (),
        {"tool_call": {"name": "execute", "id": "execute-1"}},
    )()

    normalized = ToolOutcomeStatusMiddleware().wrap_tool_call(
        request,
        lambda unused: original,
    )

    assert isinstance(normalized, ToolMessage)
    assert normalized.status == "error"
    assert normalized.content == original.content
    assert original.status == "success"


def _circuit_request(
    *,
    call_id: str,
    command: str,
    conversation: str = "conversation-1",
    tool_name: str = "execute",
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": {"command": command},
            "id": call_id,
        },
        tool=None,
        state={"messages": [HumanMessage(content=conversation, id=conversation)]},
        runtime=SimpleNamespace(config={}),
    )


def test_tool_circuit_does_not_suppress_successful_repeated_calls() -> None:
    epoch = [0]
    executed: list[str] = []
    censored: list[dict[str, Any]] = []
    circuit = ToolCircuitBreakerMiddleware(
        state_epoch=lambda: epoch[0],
        audit=None,
        scope="circuit-test",
        censor_observer=lambda fields: censored.append(dict(fields)),
    )

    def handler(request: ToolCallRequest) -> ToolMessage:
        executed.append(str(request.tool_call["id"]))
        return ToolMessage(
            content="1 passed",
            tool_call_id=str(request.tool_call["id"]),
            name="execute",
        )

    first = circuit.wrap_tool_call(
        _circuit_request(call_id="call-1", command="pytest test_a.py"),
        handler,
    )
    confirmation = circuit.wrap_tool_call(
        _circuit_request(call_id="call-2", command="pytest test_a.py"),
        handler,
    )
    repeated = circuit.wrap_tool_call(
        _circuit_request(call_id="call-3", command="pytest test_a.py"),
        handler,
    )
    assert first.status == "success"
    assert confirmation.status == "success"
    assert repeated.status == "success"
    assert first.additional_kwargs["beliefkv_physical_execution"] is True
    assert repeated.additional_kwargs["beliefkv_physical_execution"] is True
    assert executed == ["call-1", "call-2", "call-3"]
    assert censored == []

    epoch[0] += 1
    rerun = circuit.wrap_tool_call(
        _circuit_request(call_id="call-4", command="pytest test_a.py"),
        handler,
    )
    assert rerun.status == "success"
    assert executed == ["call-1", "call-2", "call-3", "call-4"]


def test_tool_circuit_suppresses_repeated_success_without_effect() -> None:
    epoch = [17]
    executed: list[str] = []
    censored: list[dict[str, Any]] = []
    circuit = ToolCircuitBreakerMiddleware(
        state_epoch=lambda: epoch[0],
        audit=None,
        scope="successful-no-effect-test",
        censor_observer=lambda fields: censored.append(dict(fields)),
    )

    def handler(request: ToolCallRequest) -> ToolMessage:
        executed.append(str(request.tool_call["id"]))
        return ToolMessage(
            content="\n[Command succeeded with exit code 0]",
            tool_call_id=str(request.tool_call["id"]),
            name="execute",
        )

    results = [
        circuit.wrap_tool_call(
            _circuit_request(
                call_id=f"no-effect-{index}",
                command="python -c '# only a comment'",
            ),
            handler,
        )
        for index in range(3)
    ]

    assert executed == ["no-effect-0", "no-effect-1"]
    assert [result.status for result in results] == ["success", "success", "error"]
    assert results[0].additional_kwargs["beliefkv_no_observable_effect"] is True
    assert results[2].additional_kwargs["beliefkv_physical_execution"] is False
    assert results[2].additional_kwargs["beliefkv_duplicate_reason"] == (
        "successful_no_effect"
    )
    assert len(censored) == 1

    epoch[0] += 1
    rerun = circuit.wrap_tool_call(
        _circuit_request(
            call_id="no-effect-new-epoch",
            command="python -c '# only a comment'",
        ),
        handler,
    )
    assert rerun.status == "success"
    assert executed[-1] == "no-effect-new-epoch"


def test_tool_circuit_does_not_share_observation_across_conversations() -> None:
    executed: list[str] = []
    circuit = ToolCircuitBreakerMiddleware(
        state_epoch=lambda: 0,
        audit=None,
        scope="conversation-isolation-test",
    )

    def handler(request: ToolCallRequest) -> ToolMessage:
        executed.append(str(request.tool_call["id"]))
        return ToolMessage(
            content="result",
            tool_call_id=str(request.tool_call["id"]),
            name="execute",
        )

    for call_id, conversation in (("call-a", "a"), ("call-b", "b")):
        circuit.wrap_tool_call(
            _circuit_request(
                call_id=call_id,
                command="pytest test_a.py",
                conversation=conversation,
            ),
            handler,
        )
    assert executed == ["call-a", "call-b"]


def test_tool_circuit_does_not_suppress_concurrent_successful_calls() -> None:
    entered = threading.Event()
    release = threading.Event()
    executed: list[str] = []
    circuit = ToolCircuitBreakerMiddleware(
        state_epoch=lambda: 0,
        audit=None,
        scope="concurrent-circuit-test",
    )

    def handler(request: ToolCallRequest) -> ToolMessage:
        executed.append(str(request.tool_call["id"]))
        entered.set()
        assert release.wait(timeout=5.0)
        return ToolMessage(
            content="1 passed",
            tool_call_id=str(request.tool_call["id"]),
            name="execute",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        first = executor.submit(
            circuit.wrap_tool_call,
            _circuit_request(call_id="call-0", command="pytest test_a.py"),
            handler,
        )
        assert entered.wait(timeout=5.0)
        duplicates = [
            executor.submit(
                circuit.wrap_tool_call,
                _circuit_request(
                    call_id=f"call-{index}",
                    command="pytest test_a.py",
                ),
                handler,
            )
            for index in range(1, 8)
        ]
        release.set()
        duplicate_results = [future.result(timeout=5.0) for future in duplicates]
        first_result = first.result(timeout=5.0)

    assert first_result.status == "success"
    assert sorted(executed) == [f"call-{index}" for index in range(8)]
    assert all(result.status == "success" for result in duplicate_results)


def test_tool_circuit_allows_one_transient_retry_then_suppresses() -> None:
    executed: list[str] = []
    circuit = ToolCircuitBreakerMiddleware(
        state_epoch=lambda: 0,
        audit=None,
        scope="transient-retry-test",
        transient_retry_limit=1,
    )

    def handler(request: ToolCallRequest) -> ToolMessage:
        executed.append(str(request.tool_call["id"]))
        return ToolMessage(
            content="Error: command timed out",
            tool_call_id=str(request.tool_call["id"]),
            name="execute",
            status="error",
        )

    for index in range(3):
        result = circuit.wrap_tool_call(
            _circuit_request(
                call_id=f"timeout-{index}",
                command="pytest test_slow.py",
            ),
            handler,
        )
    assert executed == ["timeout-0", "timeout-1"]
    assert result.status == "error"
    assert "duplicate_suppressed" in str(result.content)


def test_tool_circuit_suppresses_deterministic_failure_without_retry() -> None:
    executed: list[str] = []
    circuit = ToolCircuitBreakerMiddleware(
        state_epoch=lambda: 0,
        audit=None,
        scope="deterministic-failure-test",
    )

    def handler(request: ToolCallRequest) -> ToolMessage:
        executed.append(str(request.tool_call["id"]))
        return ToolMessage(
            content="Error: String not found in file: old text",
            tool_call_id=str(request.tool_call["id"]),
            name="edit_file",
            status="error",
        )

    for index in range(2):
        result = circuit.wrap_tool_call(
            _circuit_request(
                call_id=f"edit-{index}",
                command="replace old text",
                tool_name="edit_file",
            ),
            handler,
        )
    assert executed == ["edit-0"]
    assert result.status == "error"
    assert "duplicate_suppressed" in str(result.content)


def test_loop_guard_counts_distinct_structured_tool_outputs_as_progress() -> None:
    messages = [
        ToolMessage(
            content=[{"type": "json", "value": index}],
            tool_call_id=f"unmatched-{index}",
            name="execute",
        )
        for index in range(8)
    ]

    snapshot = analyze_agent_history(
        messages,
        LoopGuardPolicy(
            repeated_call_limit=99,
            consecutive_no_progress_limit=3,
        ),
    )

    assert snapshot.reason is None
    assert snapshot.consecutive_no_progress == 1


def test_loop_guard_counts_parallel_failures_as_one_decision_batch() -> None:
    calls = [
        {
            "name": "read_file",
            "args": {"path": f"/missing-{index}"},
            "id": f"parallel-{index}",
        }
        for index in range(44)
    ]
    messages = [AIMessage(content="", tool_calls=calls)]
    messages.extend(
        ToolMessage(
            content="Error: path_not_found",
            tool_call_id=str(call["id"]),
            name="read_file",
        )
        for call in calls
    )

    snapshot = analyze_agent_history(
        messages,
        LoopGuardPolicy(
            repeated_call_limit=6,
            consecutive_error_limit=6,
            consecutive_no_progress_limit=8,
        ),
    )

    assert snapshot.reason is None
    assert snapshot.tool_calls == 44
    assert snapshot.completed_tool_calls == 44
    assert snapshot.consecutive_errors == 1
    assert snapshot.consecutive_no_progress == 0


def test_loop_guard_does_not_treat_distinct_probe_strings_as_progress() -> None:
    messages = []
    for index in range(8):
        messages.extend(
            _tool_exchange(
                "execute",
                {"command": f'python -c "print({index})"'},
                str(index),
                f"diagnostic result {index}",
            )
        )

    snapshot = analyze_agent_history(messages, LoopGuardPolicy())
    assert snapshot.reason == "no_observable_progress"
    assert snapshot.consecutive_no_progress == 8


def test_loop_guard_observes_semantic_patterns_without_intervening() -> None:
    messages = []
    for index in range(3):
        messages.extend(_tool_exchange("ls", {"path": "/src"}, str(index), "same"))
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(),
        completion_schema=ChildCompletion,
        completion_instruction="Return ChildCompletion.",
        audit=None,
        scope="observe-only-test",
    )

    update = guard.before_model({"messages": messages}, runtime=None)

    assert update is not None
    assert update["guard_observed_patterns"] == ("repeated_tool_call",)
    assert "guard_phase" not in update
    assert "guard_forcing_completion" not in update


def test_loop_guard_clears_legacy_semantic_finalization_state() -> None:
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(),
        completion_schema=ChildCompletion,
        completion_instruction="Return ChildCompletion.",
        audit=None,
        scope="legacy-state-test",
    )

    update = guard.before_model(
        {
            "messages": [],
            "guard_phase": "FINALIZE",
            "guard_forcing_completion": True,
            "guard_reason": "no_observable_progress",
            "guard_recovery_attempt": 4,
        },
        runtime=None,
    )

    assert update is not None
    assert update["guard_phase"] == "NORMAL"
    assert update["guard_forcing_completion"] is False
    assert update["guard_reason"] == ""


def test_loop_guard_enters_suspect_before_bounded_recovery() -> None:
    messages = []
    for index in range(3):
        messages.extend(_tool_exchange("ls", {"path": "/src"}, str(index), "same"))
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(enforce_semantic_guard=True),
        completion_schema=ChildCompletion,
        completion_instruction="Return ChildCompletion.",
        audit=None,
        scope="test",
    )
    update = guard.before_model({"messages": messages}, runtime=None)
    assert update is not None
    assert update["guard_phase"] == "SUSPECT"
    assert update["guard_forcing_completion"] is False
    assert update["guard_reason"] == "repeated_tool_call"
    assert update["guard_trigger_model_calls"] == 3

    recovery = guard.before_model({**update, "messages": messages}, runtime=None)
    assert recovery is not None
    assert recovery["guard_phase"] == "RECOVERY"
    assert recovery["guard_forcing_completion"] is True
    assert recovery["guard_recovery_attempt"] == 1


def test_loop_guard_returns_to_normal_after_credible_recovery_progress() -> None:
    messages = []
    for index in range(3):
        messages.extend(_tool_exchange("ls", {"path": "/src"}, str(index), "same"))
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(enforce_semantic_guard=True),
        completion_schema=ChildCompletion,
        completion_instruction="Return ChildCompletion.",
        audit=None,
        scope="recoverable-test",
    )
    state: dict[str, Any] = {"messages": messages}
    suspect = guard.before_model(state, runtime=None)
    assert suspect is not None
    state.update(suspect)
    recovery = guard.before_model(state, runtime=None)
    assert recovery is not None
    state.update(recovery)

    messages.extend(
        _tool_exchange(
            "grep",
            {"pattern": "PoolManager", "path": "/workspace/requests"},
            "new-evidence",
            "requests/pools.py:PoolManager",
        )
    )
    state["messages"] = messages
    recovered = guard.before_model(state, runtime=None)

    assert recovered is not None
    assert recovered["guard_phase"] == "NORMAL"
    assert recovered["guard_forcing_completion"] is False
    assert recovered["guard_recovery_attempt"] == 0


def test_loop_guard_extends_soft_graph_budget_only_with_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "beliefkv.experiments.agent_protocol.get_config",
        lambda: {
            "recursion_limit": 512,
            "metadata": {"langgraph_step": 384},
        },
    )
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(enforce_soft_graph_budget=True),
        completion_schema=WorkflowCompletion,
        completion_instruction="Return WorkflowCompletion.",
        audit=None,
        scope="soft-lease-test",
    )
    messages = _tool_exchange(
        "read_file",
        {"path": "/workspace/requests/models.py"},
        "read-1",
        "class Request:",
    )

    extended = guard.before_model(
        {
            "messages": messages,
            "guard_graph_progress_keys": (),
            "guard_graph_lease_until": 384,
        },
        runtime=None,
    )
    assert extended is not None
    assert extended["guard_graph_lease_until"] == 480

    stalled = guard.before_model(
        {
            "messages": [],
            "guard_graph_progress_keys": (),
            "guard_graph_lease_until": 384,
        },
        runtime=None,
    )
    assert stalled is not None
    assert stalled["guard_phase"] == "SUSPECT"
    assert stalled["guard_reason"] == "graph_soft_budget_without_progress"


def test_loop_guard_observes_soft_graph_budget_without_intervening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "beliefkv.experiments.agent_protocol.get_config",
        lambda: {
            "recursion_limit": 512,
            "metadata": {"langgraph_step": 384},
        },
    )
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(),
        completion_schema=WorkflowCompletion,
        completion_instruction="Return WorkflowCompletion.",
        audit=None,
        scope="soft-observe-only-test",
    )

    update = guard.before_model({"messages": []}, runtime=None)

    assert update is not None
    assert update["guard_soft_budget_observed"] is True
    assert "guard_phase" not in update
    assert "guard_forcing_completion" not in update


def test_loop_guard_hard_graph_limit_reserves_terminal_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "beliefkv.experiments.agent_protocol.get_config",
        lambda: {
            "recursion_limit": 512,
            "metadata": {"langgraph_step": 481},
        },
    )
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(graph_step_reserve=32),
        completion_schema=WorkflowCompletion,
        completion_instruction="Return WorkflowCompletion.",
        audit=None,
        scope="graph-budget-test",
    )

    update = guard.before_model({"messages": []}, runtime=None)

    assert update is not None
    assert update["guard_phase"] == "FINALIZE"
    assert update["guard_forcing_completion"] is True
    assert update["guard_reason"] == "graph_step_hard_limit_low"
    assert update["guard_ever_intervened"] is True


def test_graph_budget_finalization_retains_only_completion_tool() -> None:
    @tool
    def execute(command: str) -> str:
        """Execute one command."""

        return command

    @tool("WorkflowCompletion")
    def completion(status: str) -> str:
        """Return terminal workflow status."""

        return status

    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(enforce_semantic_guard=True),
        completion_schema=WorkflowCompletion,
        completion_instruction="Return WorkflowCompletion.",
        audit=None,
        scope="graph-budget-tools-test",
    )
    model = FakeMessagesListChatModel(responses=[AIMessage(content="")])
    bounded = guard._guard_recovery_request(
        ModelRequest(
            model=model,
            messages=[],
            system_message=SystemMessage(content="base"),
            tools=[execute, completion],
            state={
                "guard_reason": "graph_step_budget_low",
                "guard_recovery_attempt": 1,
            },
        )
    )

    assert [item.name for item in bounded.tools] == ["WorkflowCompletion"]
    assert "RUNTIME COMPLETION DIRECTIVE" in bounded.system_message.text


def test_loop_guard_tracks_and_enforces_activation_wall_clock() -> None:
    now = [10.0]
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(activation_wall_clock_s=5.0),
        completion_schema=ChildCompletion,
        completion_instruction="Return ChildCompletion.",
        audit=None,
        scope="wall-clock-test",
        clock=lambda: now[0],
    )

    initial = guard.before_model({"messages": []}, runtime=None)
    assert initial is not None
    assert initial["guard_activation_started_monotonic"] == 10.0
    now[0] = 14.9
    assert (
        guard.before_model(
            {
                "messages": [],
                "guard_activation_started_monotonic": 10.0,
            },
            runtime=None,
        )
        is None
    )
    now[0] = 15.0
    exhausted = guard.before_model(
        {
            "messages": [],
            "guard_activation_started_monotonic": 10.0,
        },
        runtime=None,
    )
    assert exhausted is not None
    assert exhausted["guard_phase"] == "FINALIZE"
    assert exhausted["guard_forcing_completion"] is True
    assert exhausted["guard_reason"] == "activation_wall_clock_exhausted"


def test_activation_deadline_caps_late_requests_by_remaining_budget() -> None:
    now = [100.0]
    deadline = ActivationDeadline(clock=lambda: now[0])

    assert deadline.request_timeout_s(900.0) == 900.0
    deadline.start(600.0)
    assert deadline.request_timeout_s(900.0) == 600.0
    now[0] = 275.0
    assert deadline.request_timeout_s(900.0) == 425.0
    assert deadline.request_timeout_s(30.0) == 30.0
    now[0] = 700.0
    with pytest.raises(ActivationDeadlineExceeded):
        deadline.request_timeout_s(900.0)

    deadline.clear()
    assert deadline.request_timeout_s(900.0) == 900.0


def test_loop_guard_uses_parent_activation_deadline_for_child_scope() -> None:
    now = [10.0]
    deadline = ActivationDeadline(clock=lambda: now[0])
    deadline.start(5.0)
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(activation_wall_clock_s=5.0),
        completion_schema=ChildCompletion,
        completion_instruction="Return ChildCompletion.",
        audit=None,
        scope="shared-wall-clock-test",
        activation_deadline=deadline,
    )

    now[0] = 15.0
    exhausted = guard.before_model({"messages": []}, runtime=None)
    assert exhausted is not None
    assert exhausted["guard_phase"] == "FINALIZE"
    assert exhausted["guard_forcing_completion"] is True
    assert exhausted["guard_reason"] == "activation_wall_clock_exhausted"


def test_loop_guard_keeps_alternate_tools_during_bounded_recovery() -> None:
    @tool
    def apply_patch(patch: str) -> str:
        """Apply one synthetic patch."""

        return patch

    @tool
    def read_file(path: str) -> str:
        """Read one synthetic file."""

        return path

    @tool
    def execute(command: str) -> str:
        """Execute one synthetic command."""

        return command

    messages = []
    for index in range(3):
        messages.extend(_tool_exchange("probe", {"path": "/same"}, str(index), "same"))
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(recovery_model_call_limit=3),
        completion_schema=WorkflowCompletion,
        completion_instruction="Return WorkflowCompletion.",
        audit=None,
        scope="recovery-test",
        finalization_tool_names=frozenset({"apply_patch", "execute"}),
    )
    model = FakeMessagesListChatModel(responses=[AIMessage(content="")])

    recovering = guard._guard_recovery_request(
        ModelRequest(
            model=model,
            messages=messages,
            system_message=SystemMessage(content="base"),
            tools=[apply_patch, read_file],
            state={
                "guard_reason": "repeated_tool_call",
                "guard_recovery_attempt": 1,
            },
        )
    )
    assert [item.name for item in recovering.tools] == ["apply_patch", "read_file"]
    assert recovering.tool_choice is None
    assert "RUNTIME RECOVERY DIRECTIVE" in recovering.system_message.text

    exhausted = guard._guard_recovery_request(
        ModelRequest(
            model=model,
            messages=messages,
            system_message=SystemMessage(content="base"),
            tools=[apply_patch, read_file, execute],
            state={
                "guard_reason": "repeated_tool_call",
                "guard_recovery_attempt": 4,
            },
        )
    )
    assert exhausted.tools == []
    assert "RUNTIME COMPLETION DIRECTIVE" in exhausted.system_message.text


def test_loop_guard_recovery_attempt_is_monotonic_across_compaction() -> None:
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(
            recovery_model_call_limit=3,
            enforce_semantic_guard=True,
        ),
        completion_schema=ChildCompletion,
        completion_instruction="Return ChildCompletion.",
        audit=None,
        scope="compaction-safe-recovery-test",
    )
    state = {
        "messages": [],
        "guard_phase": "RECOVERY",
        "guard_forcing_completion": True,
        "guard_reason": "consecutive_tool_errors",
        "guard_recovery_attempt": 0,
        "guard_recovery_baseline_keys": (),
    }

    for expected in (1, 2, 3, 4):
        update = guard.before_model(state, runtime=None)
        assert update is not None
        assert update["guard_recovery_attempt"] == expected
        assert update["guard_phase"] == (
            "FINALIZE" if expected == 4 else "RECOVERY"
        )
        state = {**state, **update, "messages": []}

    assert state["guard_recovery_attempt"] == 4


def test_unstructured_guard_output_repairs_format_without_synthesizing_blocked() -> None:
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(),
        completion_schema=ChildCompletion,
        completion_instruction="Return ChildCompletion.",
        audit=None,
        scope="forced-terminal-test",
    )

    first = guard.after_model(
        {
            "messages": [AIMessage(content="still no structured result")],
            "guard_forcing_completion": True,
            "guard_reason": "repeated_tool_call",
            "guard_trigger_model_calls": 3,
        },
        runtime=None,
    )

    assert first is not None
    assert first["jump_to"] == "model"
    assert first["protocol_repair_active"] is True
    assert "structured_response" not in first

    second = guard.after_model(
        {
            "messages": [AIMessage(content="still no structured result")],
            "guard_forcing_completion": True,
            "guard_reason": "repeated_tool_call",
            **first,
        },
        runtime=None,
    )
    assert second is not None
    assert second["jump_to"] == "end"
    assert second["protocol_repair_failed"] is True
    assert "structured_response" not in second
    with pytest.raises(TerminalProtocolError):
        require_structured_completion(second, ChildCompletion)


def test_loop_guard_exhaustion_requests_honest_completion_without_tools() -> None:
    @tool
    def execute(command: str) -> str:
        """Execute one synthetic command."""

        return command

    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(recovery_model_call_limit=3),
        completion_schema=WorkflowCompletion,
        completion_instruction="Return WorkflowCompletion.",
        audit=None,
        scope="bounded-test-recovery",
        finalization_tool_names=frozenset({"execute"}),
    )
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "edit_file", "args": {"path": "/a"}, "id": "edit"}
            ],
        ),
        ToolMessage(
            content="updated",
            tool_call_id="edit",
            name="edit_file",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "execute", "args": {"command": "test"}, "id": "test"}
            ],
        ),
        ToolMessage(
            content="passed",
            tool_call_id="test",
            name="execute",
        ),
    ]
    model = FakeMessagesListChatModel(responses=[AIMessage(content="")])

    request = guard._guard_recovery_request(
        ModelRequest(
            model=model,
            messages=messages,
            system_message=SystemMessage(content="base"),
            tools=[execute],
            state={
                "guard_reason": "repeated_tool_call",
                "guard_recovery_attempt": 4,
            },
        )
    )

    assert request.tools == []
    assert "RUNTIME COMPLETION DIRECTIVE" in request.system_message.text
    assert "Select the terminal status honestly" in request.system_message.text


def test_structured_completion_is_required() -> None:
    completion = WorkflowCompletion(
        status="blocked",
        summary="Insufficient evidence",
        unresolved=["missing reproduction"],
    )
    assert require_structured_completion(
        {"structured_response": completion}, WorkflowCompletion
    ) is completion
    with pytest.raises(RuntimeError, match="WorkflowCompletion"):
        require_structured_completion({"messages": []}, WorkflowCompletion)


def test_loop_guard_observes_real_agent_graph_without_blocking_model_completion() -> None:
    class ToolCallingFakeModel(FakeMessagesListChatModel):
        _bound_tool_names: list[list[str]] = PrivateAttr(default_factory=list)

        def bind_tools(self, tools, **kwargs):
            del kwargs
            self._bound_tool_names.append(
                [
                    str(tool.name if hasattr(tool, "name") else tool.get("name"))
                    for tool in tools
                ]
            )
            return self

    @tool
    def probe(path: str) -> str:
        """Read one synthetic path."""

        return f"content from {path}"

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "probe", "args": {"path": "/same"}, "id": f"probe-{i}"}
            ],
        )
        for i in range(3)
    ]
    responses.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "ChildCompletion",
                    "args": {
                        "status": "blocked",
                        "summary": "Stopped after repeated probes",
                        "evidence": ["/same was probed"],
                        "tests": [],
                        "files_changed": [],
                        "unresolved": ["no additional evidence"],
                        "confidence": "low",
                    },
                    "id": "completion-1",
                }
            ],
        )
    )
    model = ToolCallingFakeModel(responses=responses)
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(),
        completion_schema=ChildCompletion,
        completion_instruction="Return ChildCompletion.",
        audit=None,
        scope="integration-test",
    )
    agent = create_agent(
        model=model,
        tools=[probe],
        middleware=[guard],
        response_format=ToolStrategy(ChildCompletion),
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Inspect the path."}]},
        config={"recursion_limit": 200},
    )

    completion = require_structured_completion(result, ChildCompletion)
    assert completion.status == "blocked"
    assert model.i == 0
    assert "probe" in model._bound_tool_names[0]
    assert model._bound_tool_names[-1] == ["probe", "ChildCompletion"]


def test_tool_circuit_preserves_successful_repeats_in_real_agent_graph() -> None:
    class ToolCallingFakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            del tools, kwargs
            return self

    executions: list[str] = []

    @tool
    def probe(path: str) -> str:
        """Read one synthetic path."""

        executions.append(path)
        return f"content from {path}"

    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "probe", "args": {"path": "/same"}, "id": "probe-1"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "probe", "args": {"path": "/same"}, "id": "probe-2"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ChildCompletion",
                        "args": {
                            "status": "complete",
                            "summary": "Used the existing observation",
                            "evidence": ["/same"],
                            "tests": [],
                            "files_changed": [],
                            "unresolved": [],
                            "confidence": "high",
                        },
                        "id": "completion",
                    }
                ],
            ),
        ]
    )
    circuit = ToolCircuitBreakerMiddleware(
        state_epoch=lambda: 0,
        audit=None,
        scope="graph-circuit-test",
    )
    agent = create_agent(
        model=model,
        tools=[probe],
        middleware=[circuit, ToolOutcomeStatusMiddleware()],
        response_format=ToolStrategy(ChildCompletion),
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Inspect the path."}]},
        config={"recursion_limit": 40},
    )

    completion = require_structured_completion(result, ChildCompletion)
    assert completion.status == "complete"
    assert executions == ["/same", "/same"]


def test_semantic_protocol_repairs_unstructured_model_stop() -> None:
    class ToolCallingFakeModel(FakeMessagesListChatModel):
        _bound_tool_names: list[list[str]] = PrivateAttr(default_factory=list)

        def bind_tools(self, tools, **kwargs):
            del kwargs
            self._bound_tool_names.append(
                [
                    str(tool.name if hasattr(tool, "name") else tool.get("name"))
                    for tool in tools
                ]
            )
            return self

    @tool
    def probe(path: str) -> str:
        """Read one synthetic path."""

        return path

    model = ToolCallingFakeModel(
        responses=[
            AIMessage(content="ordinary prose without a completion tool"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ChildCompletion",
                        "args": {
                            "status": "blocked",
                            "summary": "Converted to the required protocol",
                            "evidence": [],
                            "tests": [],
                            "files_changed": [],
                            "unresolved": ["initial response was unstructured"],
                            "confidence": "low",
                        },
                        "id": "completion-2",
                    }
                ],
            ),
        ]
    )
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(enabled=False),
        completion_schema=ChildCompletion,
        completion_instruction="Return ChildCompletion.",
        audit=None,
        scope="semantic-repair-test",
    )
    agent = create_agent(
        model=model,
        tools=[probe],
        middleware=[guard],
        response_format=ToolStrategy(ChildCompletion),
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Finish semantically."}]},
        config={"recursion_limit": 200},
    )

    completion = require_structured_completion(result, ChildCompletion)
    assert completion.status == "blocked"
    assert len(model._bound_tool_names) == 2
    assert "probe" in model._bound_tool_names[0]
    assert model._bound_tool_names[1] == ["ChildCompletion"]


def test_semantic_protocol_normalizes_valid_json_without_another_model_call() -> None:
    completion = ChildCompletion(
        status="complete",
        summary="Repository evidence collected",
        evidence=["sympy/core/basic.py"],
        tests=[],
        files_changed=[],
        unresolved=[],
        confidence="high",
    )
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(enabled=False),
        completion_schema=ChildCompletion,
        completion_instruction="Return ChildCompletion.",
        audit=None,
        scope="json-normalization-test",
    )

    update = guard.after_model(
        {"messages": [AIMessage(content=completion.model_dump_json())]},
        runtime=None,
    )

    assert update is not None
    assert update["jump_to"] == "end"
    assert update["protocol_normalized"] is True
    assert update["structured_response"] == completion
