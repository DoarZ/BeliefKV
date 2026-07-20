from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import PrivateAttr


pytest.importorskip("deepagents")

from deepagents.backends.protocol import ExecuteResponse
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest
from langchain.agents.structured_output import ToolStrategy

from beliefkv.experiments.agent_protocol import (
    AgentLoopGuardMiddleware,
    ChildCompletion,
    LoopGuardPolicy,
    WorkflowCompletion,
    analyze_agent_history,
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
    copy_append_window,
    load_workload_bundle,
    observed_successful_test_commands,
    prepare_workspace,
    prometheus_gauge_sum,
    validate_workflow_completion,
    _invoke_with_partial_state,
    _filesystem_middleware,
    _runtime_verify_changed_tests,
    _planned_child_loop_guard_policy,
    _workspace_patch_tool,
)


def test_prometheus_gauge_sum_aggregates_labeled_series() -> None:
    payload = """\
sglang:num_used_tokens{tp_rank="0"} 123
sglang:num_used_tokens{tp_rank="1"} 45
sglang:num_running_reqs{tp_rank="0"} 2
"""
    assert prometheus_gauge_sum(payload, "sglang:num_used_tokens") == 168
    assert prometheus_gauge_sum(payload, "missing") is None


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


def test_experiment_config_defaults_to_200_graph_steps(tmp_path: Path) -> None:
    config = DeepAgentsExperimentConfig(
        mode="planned",
        base_url="http://localhost:18000/v1",
        model="model",
        output_dir=tmp_path,
        workload_manifest=tmp_path / "workloads.json",
        docker_image="fixture:latest",
    )
    assert config.recursion_limit == 200
    assert config.loop_guard.enabled
    assert config.sandbox_preflight_command == SYMPY_SANDBOX_PREFLIGHT
    assert "/workspace" in SANDBOX_PATH_CONTRACT


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
            consecutive_diagnostic_probe_limit=8,
            max_model_calls_without_completion=32,
            max_tool_calls_without_completion=64,
        ),
    )

    policy = _planned_child_loop_guard_policy(config)
    assert policy.repeated_call_limit == 3
    assert policy.consecutive_diagnostic_probe_limit == 4
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

    budget_policy = LoopGuardPolicy(max_model_calls_without_completion=4)
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


def test_loop_guard_stops_consecutive_diagnostic_python_probes() -> None:
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
    assert snapshot.reason == "diagnostic_probe_loop"
    assert snapshot.consecutive_no_progress == 0


def test_loop_guard_switches_to_forced_completion_state() -> None:
    messages = []
    for index in range(3):
        messages.extend(_tool_exchange("ls", {"path": "/src"}, str(index), "same"))
    guard = AgentLoopGuardMiddleware(
        policy=LoopGuardPolicy(),
        completion_schema=ChildCompletion,
        completion_instruction="Return ChildCompletion.",
        audit=None,
        scope="test",
    )
    update = guard.before_model({"messages": messages}, runtime=None)
    assert update == {
        "guard_forcing_completion": True,
        "guard_reason": "repeated_tool_call",
        "guard_trigger_model_calls": 3,
    }


def test_loop_guard_retains_only_bounded_recovery_tools() -> None:
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

    recovering = guard._force_completion_request(
        ModelRequest(
            model=model,
            messages=messages,
            system_message=SystemMessage(content="base"),
            tools=[apply_patch, read_file],
            state={
                "guard_reason": "repeated_tool_call",
                "guard_trigger_model_calls": 3,
            },
        )
    )
    assert [item.name for item in recovering.tools] == ["apply_patch"]
    assert recovering.tool_choice == "required"
    assert "RUNTIME RECOVERY DIRECTIVE" in recovering.system_message.text

    patched_messages = [
        *messages,
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "apply_patch",
                    "args": {"patch": "diff --git ..."},
                    "id": "patch-1",
                }
            ],
        ),
        ToolMessage(
            content="Patch applied successfully.",
            tool_call_id="patch-1",
            name="apply_patch",
        ),
    ]
    testing = guard._force_completion_request(
        ModelRequest(
            model=model,
            messages=patched_messages,
            system_message=SystemMessage(content="base"),
            tools=[apply_patch, read_file, execute],
            state={
                "guard_reason": "repeated_tool_call",
                "guard_trigger_model_calls": 3,
            },
        )
    )
    assert [item.name for item in testing.tools] == ["execute"]
    assert testing.tool_choice == "required"
    assert "The patch was applied" in testing.system_message.text

    exhausted = guard._force_completion_request(
        ModelRequest(
            model=model,
            messages=messages,
            system_message=SystemMessage(content="base"),
            tools=[apply_patch, read_file, execute],
            state={
                "guard_reason": "repeated_tool_call",
                "guard_trigger_model_calls": 0,
            },
        )
    )
    assert exhausted.tools == []
    assert "RUNTIME COMPLETION DIRECTIVE" in exhausted.system_message.text


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


def test_loop_guard_terminates_real_agent_graph_with_structured_result() -> None:
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
    assert model._bound_tool_names[-1] == ["ChildCompletion"]


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
