#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.runtime.codex_adapter import (
    CodexRuntimeAdapter,
    CodexThreadRegistry,
    MultiplexRuntimeEventSink,
)
from beliefkv.runtime.codex_app_server import CodexAppServerClient
from beliefkv.runtime.event_channel import (
    JsonlRuntimeEventSink,
    UnixDatagramRuntimeEventSink,
)
from beliefkv.runtime.responses_server import ResponsesChatBridgeServer


DEFAULT_MANIFEST = (
    REPOSITORY_ROOT / "configs/workloads/codex_swebench_sympy_subagents.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class AppServerAudit:
    def __init__(self, path: Path) -> None:
        self.stream = path.open("x", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()
        self.sequence = 0

    def emit(self, message: dict[str, Any]) -> None:
        sanitized = copy.deepcopy(message)
        params = sanitized.get("params")
        item = params.get("item") if isinstance(params, dict) else None
        if isinstance(item, dict) and "aggregatedOutput" in item:
            output = str(item.pop("aggregatedOutput") or "")
            item["aggregatedOutputChars"] = len(output)
            item["aggregatedOutputSha256"] = hashlib.sha256(
                output.encode("utf-8")
            ).hexdigest()
        with self.lock:
            self.sequence += 1
            payload = {
                "schema_version": 1,
                "sequence": self.sequence,
                "observed_ts_ms": time.monotonic() * 1000.0,
                **sanitized,
            }
            self.stream.write(
                json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
            )

    def close(self) -> None:
        self.stream.close()


class GPUStatsMonitor:
    def __init__(self, gpu_index: int, output_path: Path) -> None:
        self.gpu_index = gpu_index
        self.output_path = output_path
        self.process: subprocess.Popen[str] | None = None
        self.stream = None

    def start(self) -> None:
        self.stream = self.output_path.open("x", encoding="utf-8")
        command = [
            "nvidia-smi",
            "-i",
            str(self.gpu_index),
            "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw",
            "--format=csv,noheader,nounits",
            "--loop-ms=250",
        ]
        self.process = subprocess.Popen(
            command,
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


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_prompt(workload: dict[str, Any]) -> str:
    return f"""Read-only SWE-bench diagnosis for {workload['instance_id']}.

You must exercise the Codex subagent runtime exactly as follows:
1. Call spawn_agent exactly twice before doing the final synthesis.
2. Ask one subagent to trace the likely implementation path and relevant symbols.
3. Ask the other subagent to inspect tests, reproduction paths, and regression risks.
4. After both children are spawned, call wait with a list containing only the first
   child ID and timeout_ms=120000. Once it is terminal, call wait with a list
   containing only the second child ID and timeout_ms=120000. Never put both IDs in
   one wait call, and retry the same child if a wait returns without a terminal state.
5. Produce a concise diagnosis that cites concrete files and symbols.

Do not edit files. Do not apply a patch. Read-only shell inspection is allowed. Do not
replace subagents with your own sequential analysis.

SWE-bench problem statement:
{workload['problem_statement']}
"""


def codex_command(codex: str, bridge_url: str, model: str) -> list[str]:
    return [
        codex,
        "app-server",
        "--stdio",
        "--enable",
        "multi_agent",
        "--disable",
        "enable_request_compression",
        "--disable",
        "image_generation",
        "--disable",
        "browser_use",
        "--disable",
        "plugins",
        "--disable",
        "remote_plugin",
        "--disable",
        "plugin_sharing",
        "--disable",
        "apps",
        "-c",
        'model_provider="sglang"',
        "-c",
        f'model="{model}"',
        "-c",
        'model_providers.sglang.name="SGLang"',
        "-c",
        f'model_providers.sglang.base_url="{bridge_url}"',
        "-c",
        'model_providers.sglang.wire_api="responses"',
        "-c",
        "model_providers.sglang.requires_openai_auth=false",
        "-c",
        "model_context_window=32768",
        "-c",
        'approval_policy="never"',
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run concurrent real SWE-bench workflows with Codex subagents."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sglang-base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--bridge-port", type=int, default=18080)
    parser.add_argument("--event-socket", type=Path)
    parser.add_argument("--model", default="Qwen2.5-14B-Instruct")
    parser.add_argument("--codex", default=shutil.which("codex") or "codex")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=768)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--min-subagents-per-workflow", type=int, default=2)
    parser.add_argument(
        "--enforce-child-join-guard",
        action="store_true",
        help="Inject Codex wait_agent calls until every dynamically spawned child is joined.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workloads = list(manifest["workloads"][: args.concurrency])
    if len(workloads) != args.concurrency:
        raise ValueError("manifest has fewer workloads than requested concurrency")
    for workload in workloads:
        worktree = Path(workload["worktree"]).resolve()
        if git_head(worktree) != workload["base_commit"]:
            raise RuntimeError(f"worktree commit mismatch: {workload['instance_id']}")
        if subprocess.run(
            ["git", "-C", str(worktree), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout:
            raise RuntimeError(f"worktree is dirty: {workload['instance_id']}")

    run_dir = args.run_dir.resolve()
    staging = run_dir.with_name(run_dir.name + ".incomplete")
    if run_dir.exists() or staging.exists():
        raise FileExistsError(f"run output already exists: {run_dir}")
    staging.mkdir(parents=True)
    event_socket = args.event_socket.resolve() if args.event_socket else None
    if event_socket is not None and not event_socket.is_socket():
        raise RuntimeError(f"BeliefKV event socket is not ready: {event_socket}")

    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    registry = CodexThreadRegistry()
    app_audit = AppServerAudit(staging / "codex_app_server_events.jsonl")
    root_done: dict[str, threading.Event] = {}
    root_results: dict[str, dict[str, Any]] = {}
    thread_results: dict[str, dict[str, Any]] = {}
    root_by_thread: dict[str, str] = {}
    notification_lock = threading.Lock()
    thread_completion = threading.Condition(notification_lock)

    with contextlib.ExitStack() as stack:
        stack.callback(app_audit.close)
        local_sink = stack.enter_context(
            JsonlRuntimeEventSink(staging / "runtime_events.codex.jsonl")
        )
        sinks = [local_sink]
        if event_socket is not None:
            sinks.append(
                stack.enter_context(UnixDatagramRuntimeEventSink(event_socket))
            )
        adapter = CodexRuntimeAdapter(MultiplexRuntimeEventSink(*sinks), registry)

        def on_notification(message: dict[str, Any]) -> None:
            app_audit.emit(message)
            adapter.handle_notification(message)
            if message.get("method") != "turn/completed":
                return
            params = message.get("params") or {}
            thread_id = str(params.get("threadId", ""))
            with thread_completion:
                thread_results[thread_id] = params.get("turn") or {}
                instance_id = root_by_thread.get(thread_id)
                if instance_id is not None:
                    root_results[instance_id] = params.get("turn") or {}
                    root_done[instance_id].set()
                thread_completion.notify_all()

        bridge = ResponsesChatBridgeServer(
            registry,
            upstream_base_url=args.sglang_base_url,
            port=args.bridge_port,
            max_completion_tokens=args.max_completion_tokens,
            audit_path=staging / "responses_bridge.jsonl",
            enforce_child_join_guard=args.enforce_child_join_guard,
            join_guard_min_children=args.min_subagents_per_workflow,
        )
        bridge.start()
        stack.callback(bridge.close)

        command = codex_command(args.codex, bridge.base_url, args.model)
        client = CodexAppServerClient(
            command,
            codex_home=staging / "codex_home",
            notification_handler=on_notification,
            stderr_path=staging / "codex_app_server.stderr.log",
            environment={
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            },
        )
        client.start()
        stack.callback(client.close)

        for workload in workloads:
            instance_id = workload["instance_id"]
            result = client.request(
                "thread/start",
                {
                    "model": args.model,
                    "modelProvider": "sglang",
                    "cwd": str(Path(workload["worktree"]).resolve()),
                    "approvalPolicy": "never",
                    "sandbox": "read-only",
                    "ephemeral": True,
                    "experimentalRawEvents": False,
                    "developerInstructions": (
                        "This benchmark requires exactly two spawn_agent calls. After "
                        "both spawns, wait for child 1 alone with timeout_ms=120000, then "
                        "child 2 alone with timeout_ms=120000. Retry any timeout; do not "
                        "synthesize until both are terminal. Never edit files."
                    ),
                },
                timeout_s=60.0,
            )
            thread = result["thread"]
            adapter.register_root(thread, workload_id=instance_id)
            root_by_thread[thread["id"]] = instance_id
            root_done[instance_id] = threading.Event()

        gpu_monitor = GPUStatsMonitor(args.gpu_index, staging / "gpu_samples.csv")
        gpu_monitor.start()
        stack.callback(gpu_monitor.close)

        def start_turn(workload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            instance_id = workload["instance_id"]
            thread_id = next(
                value for value, item in root_by_thread.items() if item == instance_id
            )
            result = client.request(
                "turn/start",
                {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": build_prompt(workload)}],
                    "approvalPolicy": "never",
                    "cwd": str(Path(workload["worktree"]).resolve()),
                    "model": args.model,
                    "responsesapiClientMetadata": {
                        "beliefkv_workload_id": instance_id,
                        "beliefkv_run_id": run_dir.name,
                    },
                },
                timeout_s=60.0,
            )
            return instance_id, result

        turn_start_results = {}
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(start_turn, workload) for workload in workloads]
            for future in as_completed(futures):
                instance_id, result = future.result()
                turn_start_results[instance_id] = result

        deadline = time.monotonic() + args.timeout_seconds

        def wait_for_root(done: threading.Event) -> bool:
            while True:
                client.raise_if_reader_failed()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                if done.wait(min(remaining, 0.5)):
                    client.raise_if_reader_failed()
                    return True

        for workload in workloads:
            instance_id = workload["instance_id"]
            if not wait_for_root(root_done[instance_id]):
                adapter.finish_incomplete_workflows(outcome="benchmark_timeout")
                raise TimeoutError(f"Codex workflow timed out: {instance_id}")

        with thread_completion:
            while True:
                client.raise_if_reader_failed()
                expected_thread_ids = {
                    identity.thread_id for identity in registry.identities()
                }
                unfinished = expected_thread_ids - set(thread_results)
                if not unfinished:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    adapter.finish_incomplete_workflows(outcome="benchmark_timeout")
                    raise TimeoutError(
                        "Codex subagents timed out: " + ", ".join(sorted(unfinished))
                    )
                thread_completion.wait(min(remaining, 0.5))

    finished_at = datetime.now(timezone.utc)
    identity_records = [
        {
            "thread_id": item.thread_id,
            "workflow_id": item.workflow_id,
            "invocation_id": item.invocation_id,
            "context_id": item.context_id,
            "parent_thread_id": item.parent_thread_id,
            "agent_role": item.agent_role,
            "agent_nickname": item.agent_nickname,
        }
        for item in registry.identities()
    ]
    runtime_event_records = [
        json.loads(line)
        for line in (staging / "runtime_events.codex.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    event_counts_by_workflow: dict[str, Counter[str]] = defaultdict(Counter)
    for record in runtime_event_records:
        event_counts_by_workflow[str(record["workflow_id"])][str(record["kind"])] += 1
    child_counts_by_workflow = Counter(
        item["workflow_id"]
        for item in identity_records
        if item["parent_thread_id"] is not None
    )
    return_sequence_by_invocation = {
        str(record["invocation_id"]): int(record["sequence"])
        for record in runtime_event_records
        if record["kind"] == "return" and record.get("invocation_id")
    }
    join_members_by_workflow: dict[str, set[str]] = defaultdict(set)
    for record in runtime_event_records:
        if record["kind"] == "join_wait":
            join_members_by_workflow[str(record["workflow_id"])].update(
                str(item) for item in record.get("member_invocation_ids", ())
            )
    gate_failures: list[str] = []
    root_workflow_ids = {
        item["workflow_id"]
        for item in identity_records
        if item["parent_thread_id"] is None
    }
    required_event_kinds = ("spawn", "join_wait", "join_satisfied")
    for workflow_id in sorted(root_workflow_ids):
        child_count = child_counts_by_workflow[workflow_id]
        if child_count < args.min_subagents_per_workflow:
            gate_failures.append(
                f"{workflow_id}: observed {child_count} subagents, expected at least "
                f"{args.min_subagents_per_workflow}"
            )
        workflow_children = [
            item
            for item in identity_records
            if item["workflow_id"] == workflow_id
            and item["parent_thread_id"] is not None
        ]
        workflow_roots = [
            item
            for item in identity_records
            if item["workflow_id"] == workflow_id
            and item["parent_thread_id"] is None
        ]
        root_return_sequence = (
            return_sequence_by_invocation.get(workflow_roots[0]["invocation_id"])
            if len(workflow_roots) == 1
            else None
        )
        for child in workflow_children:
            if child["invocation_id"] not in join_members_by_workflow[workflow_id]:
                gate_failures.append(
                    f"{workflow_id}: child was never joined: {child['thread_id']}"
                )
            child_return_sequence = return_sequence_by_invocation.get(
                child["invocation_id"]
            )
            if child_return_sequence is None:
                gate_failures.append(
                    f"{workflow_id}: child did not return: {child['thread_id']}"
                )
            elif (
                root_return_sequence is not None
                and child_return_sequence > root_return_sequence
            ):
                gate_failures.append(
                    f"{workflow_id}: root returned before child {child['thread_id']}"
                )
        for kind in required_event_kinds:
            if event_counts_by_workflow[workflow_id][kind] == 0:
                gate_failures.append(f"{workflow_id}: missing {kind} event")
    subagent_gate = {
        "passed": not gate_failures,
        "min_subagents_per_workflow": args.min_subagents_per_workflow,
        "required_event_kinds": list(required_event_kinds),
        "child_counts_by_workflow": dict(sorted(child_counts_by_workflow.items())),
        "event_counts_by_workflow": {
            workflow_id: dict(sorted(counts.items()))
            for workflow_id, counts in sorted(event_counts_by_workflow.items())
        },
        "failures": gate_failures,
    }
    run_manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": time.monotonic() - started_monotonic,
        "runtime": "codex-app-server",
        "codex_version": subprocess.run(
            [args.codex, "--version"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "model": args.model,
        "sglang_base_url": args.sglang_base_url,
        "bridge_url": f"http://127.0.0.1:{args.bridge_port}/v1",
        "beliefkv_event_socket": str(event_socket) if event_socket else None,
        "concurrency": args.concurrency,
        "max_completion_tokens": args.max_completion_tokens,
        "enforce_child_join_guard": args.enforce_child_join_guard,
        "workload_manifest": str(manifest_path),
        "workload_manifest_sha256": sha256(manifest_path),
        "instance_ids": [item["instance_id"] for item in workloads],
        "thread_identities": identity_records,
        "root_turn_start_results": turn_start_results,
        "root_turn_results": root_results,
        "thread_turn_results": thread_results,
        "root_workflow_count": len(workloads),
        "subagent_count": sum(
            item["parent_thread_id"] is not None for item in identity_records
        ),
        "subagent_gate": subagent_gate,
        "artifacts": {
            name: sha256(staging / name)
            for name in (
                "runtime_events.codex.jsonl",
                "codex_app_server_events.jsonl",
                "responses_bridge.jsonl",
                "gpu_samples.csv",
                "codex_app_server.stderr.log",
            )
        },
    }
    write_json(staging / "manifest.json", run_manifest)
    if gate_failures:
        raise RuntimeError("subagent behavior gate failed: " + "; ".join(gate_failures))
    staging.replace(run_dir)
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Codex subagent benchmark failed: {error}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2) from error
