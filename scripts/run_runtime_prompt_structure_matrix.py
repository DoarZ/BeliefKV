#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.experiments.runtime_policy_matrix import (
    MatrixCondition,
    MatrixSpec,
    aggregate_rows,
    build_conditions,
    decode_qwen_records,
    evaluate_markers,
    sha256,
    summarize_qwen_records,
    text_sha256,
)
from beliefkv.runtime.codex_adapter import CodexRuntimeAdapter, CodexThreadRegistry
from beliefkv.runtime.codex_app_server import CodexAppServerClient
from beliefkv.runtime.event_channel import JsonlRuntimeEventSink
from beliefkv.runtime.responses_server import ResponsesChatBridgeServer


DEFAULT_SPEC = (
    REPOSITORY_ROOT
    / "configs/workloads/runtime_prompt_structure_matrix.json"
)
DEFAULT_QWEN_SETTINGS = (
    REPOSITORY_ROOT
    / "configs/qwen_code/qwen3_coder_30b_a3b_fp8_matrix.json"
)
DEFAULT_QWEN_RUNNER = REPOSITORY_ROOT / "scripts/run_qwen_code_local.sh"
DEFAULT_QWEN_PROXY = REPOSITORY_ROOT / "scripts/qwen_sandbox_allowlist_proxy.mjs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def command_output(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    return (result.stdout or result.stderr).strip()


@contextlib.contextmanager
def disposable_git_workspace(repository: Path, evidence_dir: Path):
    source_head = command_output(["git", "rev-parse", "HEAD"], cwd=repository)
    if not source_head:
        raise RuntimeError(f"cannot resolve source commit: {repository}")
    with tempfile.TemporaryDirectory(
        prefix="beliefkv-matrix-workspace-"
    ) as temporary:
        workspace = Path(temporary) / "workspace"
        clone = subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--no-local",
                "--no-checkout",
                str(repository),
                str(workspace),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120.0,
        )
        if clone.returncode != 0:
            raise RuntimeError(f"disposable clone failed: {clone.stderr.strip()}")
        checkout = subprocess.run(
            ["git", "checkout", "--quiet", "--detach", source_head],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
        if checkout.returncode != 0:
            raise RuntimeError(
                f"disposable checkout failed: {checkout.stderr.strip()}"
            )
        marker = workspace / ".beliefkv-disposable-workspace"
        marker.write_text(
            f"source_commit={source_head}\n", encoding="utf-8"
        )
        exclude = workspace / ".git/info/exclude"
        with exclude.open("a", encoding="utf-8") as stream:
            stream.write(
                "\n.beliefkv-disposable-workspace\n"
                ".beliefkv-qwen-allowlist-proxy.mjs\n"
            )
        initial_status = command_output(
            ["git", "status", "--short"], cwd=workspace
        )
        if initial_status:
            raise RuntimeError(
                f"disposable workspace is not clean: {initial_status}"
            )
        metadata = {
            "schema_version": 1,
            "isolation": "per-condition-disposable-git-clone",
            "source_repository": str(repository),
            "source_commit": source_head,
            "workspace_path": str(workspace),
            "initial_status": initial_status,
        }
        try:
            yield workspace
        finally:
            try:
                final_status = command_output(
                    ["git", "status", "--short"], cwd=workspace
                )
                subprocess.run(
                    ["git", "add", "-N", "--", "."],
                    cwd=workspace,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30.0,
                )
                workspace_diff = command_output(
                    ["git", "diff", "--binary", "HEAD"], cwd=workspace
                )
                (evidence_dir / "workspace.patch").write_text(
                    workspace_diff + ("\n" if workspace_diff else ""),
                    encoding="utf-8",
                )
                metadata.update(
                    {
                        "final_status": final_status,
                        "workspace_modified": bool(final_status),
                        "workspace_patch_sha256": sha256(
                            evidence_dir / "workspace.patch"
                        ),
                    }
                )
            except Exception as error:
                metadata["evidence_error"] = f"{type(error).__name__}: {error}"
            write_json(evidence_dir / "workspace.json", metadata)


class NotificationAudit:
    def __init__(self, path: Path) -> None:
        self._stream = path.open("x", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._sequence = 0

    def emit(self, message: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            payload = {
                "schema_version": 1,
                "sequence": self._sequence,
                "observed_ts_ms": time.monotonic() * 1000.0,
                **message,
            }
            self._stream.write(
                json.dumps(payload, sort_keys=True, allow_nan=False, default=str)
                + "\n"
            )

    def close(self) -> None:
        self._stream.close()


class GPUStatsMonitor:
    def __init__(self, gpu_index: int, output_path: Path) -> None:
        self.gpu_index = gpu_index
        self.output_path = output_path
        self.process: subprocess.Popen[str] | None = None
        self.stream: TextIO | None = None

    def start(self) -> None:
        self.stream = self.output_path.open("x", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                str(self.gpu_index),
                "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw",
                "--format=csv,noheader,nounits",
                "--loop-ms=500",
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


def prometheus_gauge_sum(payload: str, metric_name: str) -> float:
    total = 0.0
    found = False
    for line in payload.splitlines():
        match = PROMETHEUS_GAUGE.match(line.strip())
        if match is None or match.group("name") != metric_name:
            continue
        total += float(match.group("value"))
        found = True
    if not found:
        raise ValueError(f"Prometheus metric is missing: {metric_name}")
    return total


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
        if root.endswith("/v1"):
            root = root[:-3]
        self.metrics_url = f"{root}/metrics"
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
        with self.output_path.open("x", encoding="utf-8", buffering=1) as stream:
            while not self.stop.is_set():
                sample: dict[str, Any] = {
                    "monotonic_ts_ms": time.monotonic() * 1000.0,
                }
                try:
                    with urllib.request.urlopen(
                        self.metrics_url, timeout=5.0
                    ) as response:
                        payload = response.read().decode("utf-8")
                    used = int(
                        prometheus_gauge_sum(payload, "sglang:num_used_tokens")
                    )
                    sample.update(
                        {
                            "resident_tokens": used,
                            "resident_pressure": used / self.pool_tokens,
                        }
                    )
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
        max_resident = max(
            (int(sample["resident_tokens"]) for sample in self.samples),
            default=0,
        )
        return {
            "metrics_sample_count": len(self.samples),
            "metrics_error_count": self.error_count,
            "max_resident_tokens": max_resident,
            "max_resident_pressure": max_resident / self.pool_tokens,
        }


def http_json(url: str, *, method: str = "GET", timeout: float = 30.0) -> Any:
    data = b"" if method == "POST" else None
    request = urllib.request.Request(url, data=data, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        if not body:
            return {"status_code": response.status}
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"status_code": response.status, "body": body}


def flush_sglang_cache(base_url: str) -> dict[str, Any]:
    root_url = base_url.rstrip("/")
    if root_url.endswith("/v1"):
        root_url = root_url[:-3]
    last_error = ""
    for attempt in range(1, 11):
        started = time.monotonic()
        try:
            response = http_json(
                f"{root_url}/flush_cache", method="POST", timeout=30.0
            )
            return {
                "success": True,
                "attempt": attempt,
                "duration_ms": (time.monotonic() - started) * 1000.0,
                "response": response,
            }
        except (OSError, urllib.error.URLError) as error:
            last_error = f"{type(error).__name__}: {error}"
            time.sleep(1.0)
    raise RuntimeError(f"SGLang cache flush failed: {last_error}")


def terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10.0)


def run_captured_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.monotonic()
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            return_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process)
            return_code = process.returncode
    return {
        "command": command,
        "return_code": return_code,
        "timed_out": timed_out,
        "wall_duration_ms": (time.monotonic() - started) * 1000.0,
    }


def run_qwen_condition(
    condition_dir: Path,
    prompt: str,
    *,
    repository: Path,
    qwen_runner: Path,
    qwen_settings: Path,
    qwen_proxy: Path,
    qwen_proxy_allow: str,
    sglang_base_url: str,
    timeout_s: float,
    max_turns: int,
    max_tool_calls: int,
) -> dict[str, Any]:
    output_path = condition_dir / "qwen_output.jsonl"
    stderr_path = condition_dir / "qwen.stderr.log"
    environment = os.environ.copy()
    proxy_name = ".beliefkv-qwen-allowlist-proxy.mjs"
    shutil.copy2(qwen_proxy, repository / proxy_name)
    environment.update(
        {
            "QWEN_HOME": str(condition_dir / "qwen_home"),
            "QWEN_RUNTIME_DIR": str(condition_dir / "qwen_runtime"),
            "OPENAI_LOG_DIR": str(condition_dir / "openai_logs"),
            "QWEN_SETTINGS_TEMPLATE": str(qwen_settings),
            "OPENAI_BASE_URL": sglang_base_url,
            "QWEN_WORKSPACE_DISPOSABLE": "1",
            "QWEN_SANDBOX_PROXY_COMMAND": (
                f"node {proxy_name} --allow {qwen_proxy_allow}"
            ),
        }
    )
    wall_minutes = max(1, math.ceil(timeout_s / 60.0))
    command = [
        str(qwen_runner),
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--max-wall-time",
        f"{wall_minutes}m",
        "--max-session-turns",
        str(max_turns),
        "--max-tool-calls",
        str(max_tool_calls),
    ]
    process = run_captured_process(
        command,
        cwd=repository,
        environment=environment,
        stdout_path=output_path,
        stderr_path=stderr_path,
        timeout_s=timeout_s + 30.0,
    )
    try:
        records = decode_qwen_records(output_path.read_text(encoding="utf-8"))
        summary = summarize_qwen_records(records)
    except Exception as error:
        summary = {
            "runtime_success": False,
            "runtime_status": "unparseable_output",
            "runtime_error": f"{type(error).__name__}: {error}",
            "runtime_reported_duration_ms": 0.0,
            "turn_count": 0,
            "request_count": len(list((condition_dir / "openai_logs").glob("*.json"))),
            "tool_call_count": 0,
            "tool_failure_count": 0,
            "permission_rejection_count": 0,
            "spawn_count": 0,
            "spawn_attempt_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "final_text": "",
        }
    else:
        summary["runtime_reported_duration_ms"] = summary.pop("duration_ms")
        summary["request_count"] = len(
            list((condition_dir / "openai_logs").glob("*.json"))
        )
    summary.update(
        {
            "duration_ms": process["wall_duration_ms"],
            "process_return_code": process["return_code"],
            "process_timed_out": process["timed_out"],
            "process_command": process["command"],
        }
    )
    if process["timed_out"]:
        summary["runtime_success"] = False
        summary["runtime_status"] = "process_timeout"
        summary["runtime_error"] = "Qwen Code exceeded the external wall limit"
    return summary


def codex_command(
    codex: str, bridge_url: str, model: str, context_window: int
) -> list[str]:
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
        f"model_context_window={context_window}",
        "-c",
        'approval_policy="never"',
        "-c",
        "agents.max_threads=4",
        "-c",
        "agents.max_depth=1",
    ]


def summarize_bridge(path: Path, root_thread_id: str) -> dict[str, Any]:
    records = load_jsonl(path)
    completed = [item for item in records if item.get("event") == "request_completed"]
    root = [item for item in completed if item.get("thread_id") == root_thread_id]
    return {
        "request_count": len(completed),
        "root_request_count": len(root),
        "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in completed),
        "completion_tokens": sum(
            int(item.get("completion_tokens") or 0) for item in completed
        ),
        "cached_tokens": 0,
        "event_counts": dict(
            sorted(Counter(str(item.get("event")) for item in records).items())
        ),
        "bridge_error_count": sum(item.get("event") == "bridge_error" for item in records),
        "upstream_error_count": sum(
            item.get("event") == "upstream_error" for item in records
        ),
    }


def run_codex_condition(
    condition_dir: Path,
    prompt: str,
    *,
    condition_id: str,
    repository: Path,
    codex: str,
    model: str,
    sglang_base_url: str,
    bridge_port: int,
    max_completion_tokens: int,
    client_context_window: int,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.monotonic()
    registry = CodexThreadRegistry()
    app_audit = NotificationAudit(condition_dir / "codex_app_server_events.jsonl")
    root_done = threading.Event()
    root_thread_id = ""
    turn_results: dict[str, dict[str, Any]] = {}
    agent_messages: dict[str, list[str]] = {}
    state_lock = threading.Lock()

    with contextlib.ExitStack() as stack:
        stack.callback(app_audit.close)
        sink = stack.enter_context(
            JsonlRuntimeEventSink(condition_dir / "runtime_events.codex.jsonl")
        )
        adapter = CodexRuntimeAdapter(sink, registry)

        def on_notification(message: dict[str, Any]) -> None:
            app_audit.emit(message)
            adapter.handle_notification(message)
            params = message.get("params") or {}
            thread_id = str(params.get("threadId") or "")
            if message.get("method") == "item/completed":
                item = params.get("item") or {}
                if item.get("type") == "agentMessage" and item.get("text"):
                    with state_lock:
                        agent_messages.setdefault(thread_id, []).append(str(item["text"]))
            elif message.get("method") == "turn/completed":
                with state_lock:
                    turn_results[thread_id] = params.get("turn") or {}
                if thread_id == root_thread_id:
                    root_done.set()

        bridge = ResponsesChatBridgeServer(
            registry,
            upstream_base_url=sglang_base_url,
            port=bridge_port,
            max_completion_tokens=max_completion_tokens,
            audit_path=condition_dir / "responses_bridge.jsonl",
            enforce_child_join_guard=False,
        )
        bridge.start()
        stack.callback(bridge.close)
        command = codex_command(
            codex, bridge.base_url, model, client_context_window
        )
        client = CodexAppServerClient(
            command,
            codex_home=condition_dir / "codex_home",
            notification_handler=on_notification,
            stderr_path=condition_dir / "codex_app_server.stderr.log",
            environment={
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            },
        )
        client.start()
        stack.callback(client.close)
        thread_result = client.request(
            "thread/start",
            {
                "model": model,
                "modelProvider": "sglang",
                "cwd": str(repository),
                "approvalPolicy": "never",
                "sandbox": "workspace-write",
                "ephemeral": True,
                "experimentalRawEvents": False,
                "developerInstructions": (
                    "This is a controlled experiment in a disposable repository clone. "
                    "Use repository tools, edits, and tests when useful, but do not access "
                    "paths outside the workspace or external networks. Follow the user task "
                    "and do not exceed 350 words."
                ),
            },
            timeout_s=60.0,
        )
        root_thread_id = str(thread_result["thread"]["id"])
        adapter.register_root(thread_result["thread"], workload_id=condition_id)
        client.request(
            "turn/start",
            {
                "threadId": root_thread_id,
                "input": [{"type": "text", "text": prompt}],
                "approvalPolicy": "never",
                "cwd": str(repository),
                "model": model,
                "responsesapiClientMetadata": {
                    "beliefkv_workload_id": condition_id,
                    "beliefkv_run_id": condition_dir.parent.name,
                },
            },
            timeout_s=60.0,
        )
        deadline = time.monotonic() + timeout_s
        while not root_done.wait(0.5):
            client.raise_if_reader_failed()
            if time.monotonic() >= deadline:
                adapter.finish_incomplete_workflows(outcome="matrix_timeout")
                break
        if root_done.is_set():
            quiet_deadline = min(deadline, time.monotonic() + 10.0)
            while time.monotonic() < quiet_deadline:
                client.raise_if_reader_failed()
                identities = registry.identities()
                with state_lock:
                    completed_threads = set(turn_results)
                if all(item.thread_id in completed_threads for item in identities):
                    break
                time.sleep(0.25)

    wall_duration_ms = (time.monotonic() - started) * 1000.0
    identities = registry.identities()
    children = [item for item in identities if item.parent_thread_id is not None]
    root_turn = turn_results.get(root_thread_id, {})
    final_messages = agent_messages.get(root_thread_id, [])
    bridge_summary = summarize_bridge(
        condition_dir / "responses_bridge.jsonl", root_thread_id
    )
    runtime_events = load_jsonl(condition_dir / "runtime_events.codex.jsonl")
    app_events = load_jsonl(condition_dir / "codex_app_server_events.jsonl")
    runtime_status = str(root_turn.get("status") or "timeout")
    runtime_success = root_done.is_set() and runtime_status == "completed"
    runtime_event_counts = Counter(
        str(item.get("kind")) for item in runtime_events
    )
    failed_tools = sum(
        item.get("kind") == "tool_end"
        and int((item.get("attributes") or {}).get("returncode") or 0) != 0
        for item in runtime_events
    )
    return {
        "runtime_success": runtime_success,
        "runtime_status": runtime_status,
        "runtime_error": str(root_turn.get("error") or ""),
        "duration_ms": wall_duration_ms,
        "runtime_reported_duration_ms": float(root_turn.get("durationMs") or 0.0),
        "turn_count": bridge_summary["root_request_count"],
        "request_count": bridge_summary["request_count"],
        "tool_call_count": runtime_event_counts["tool_start"],
        "tool_failure_count": failed_tools,
        "permission_rejection_count": 0,
        "spawn_count": len(children),
        "spawn_attempt_count": len(children),
        "prompt_tokens": bridge_summary["prompt_tokens"],
        "completion_tokens": bridge_summary["completion_tokens"],
        "cached_tokens": bridge_summary["cached_tokens"],
        "final_text": final_messages[-1] if final_messages else "",
        "root_thread_id": root_thread_id,
        "child_thread_ids": [item.thread_id for item in children],
        "runtime_event_counts": dict(sorted(runtime_event_counts.items())),
        "bridge": bridge_summary,
        "process_command": command,
        "process_timed_out": not root_done.is_set(),
        "process_return_code": 0 if runtime_success else 1,
    }


def artifact_hashes(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        records.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def prepare_staging(
    run_dir: Path, spec_path: Path, *, resume: bool
) -> tuple[Path, dict[str, Any]]:
    staging = run_dir.with_name(run_dir.name + ".incomplete")
    if run_dir.exists():
        raise FileExistsError(f"completed run already exists: {run_dir}")
    spec_digest = sha256(spec_path)
    if staging.exists():
        if not resume:
            raise FileExistsError(
                f"incomplete run exists; pass --resume: {staging}"
            )
        manifest = json.loads(
            (staging / "run_manifest.partial.json").read_text(encoding="utf-8")
        )
        if manifest["spec_sha256"] != spec_digest:
            raise ValueError("cannot resume with a different matrix spec")
        return staging, manifest
    staging.mkdir(parents=True)
    shutil.copy2(spec_path, staging / "matrix_spec.json")
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "status": "running",
        "started_at": utc_now(),
        "spec_path": str(spec_path),
        "spec_sha256": spec_digest,
    }
    write_json(staging / "run_manifest.partial.json", manifest)
    return staging, manifest


def preserve_interrupted_condition(path: Path) -> None:
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(path.name + f".interrupted-{suffix}")
    path.rename(target)


def run_condition(
    args: argparse.Namespace,
    spec: MatrixSpec,
    condition: MatrixCondition,
    staging: Path,
) -> dict[str, Any]:
    runs_root = staging / "runs"
    runs_root.mkdir(exist_ok=True)
    final_dir = runs_root / f"{condition.order:02d}__{condition.condition_id}"
    partial_dir = final_dir.with_name(final_dir.name + ".incomplete")
    if final_dir.is_dir():
        return json.loads((final_dir / "result.json").read_text(encoding="utf-8"))
    if partial_dir.exists():
        preserve_interrupted_condition(partial_dir)
    partial_dir.mkdir()
    (partial_dir / "prompt.txt").write_text(condition.prompt, encoding="utf-8")
    cache_flush = flush_sglang_cache(args.sglang_base_url)
    started_at = utc_now()
    monitor = GPUStatsMonitor(args.gpu_index, partial_dir / "gpu_samples.csv")
    resident_monitor = SGLangMetricsMonitor(
        args.sglang_base_url,
        partial_dir / "sglang_metrics.jsonl",
        pool_tokens=args.sglang_pool_tokens,
    )
    monitor.start()
    resident_monitor.start()
    metrics_summary: dict[str, Any] = {}
    try:
        with disposable_git_workspace(spec.repository, partial_dir) as workspace:
            if condition.runtime == "qwen_code":
                runtime_result = run_qwen_condition(
                    partial_dir,
                    condition.prompt,
                    repository=workspace,
                    qwen_runner=args.qwen_runner,
                    qwen_settings=args.qwen_settings,
                    qwen_proxy=args.qwen_proxy,
                    qwen_proxy_allow=args.qwen_proxy_allow,
                    sglang_base_url=args.qwen_sglang_base_url,
                    timeout_s=args.timeout_seconds,
                    max_turns=args.max_turns,
                    max_tool_calls=args.max_tool_calls,
                )
            else:
                runtime_result = run_codex_condition(
                    partial_dir,
                    condition.prompt,
                    condition_id=condition.condition_id,
                    repository=workspace,
                    codex=args.codex,
                    model=spec.model,
                    sglang_base_url=args.sglang_base_url,
                    bridge_port=args.bridge_port,
                    max_completion_tokens=args.max_completion_tokens,
                    client_context_window=args.client_context_window,
                    timeout_s=args.timeout_seconds,
                )
    except Exception as error:
        (partial_dir / "runner_exception.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        runtime_result = {
            "runtime_success": False,
            "runtime_status": "runner_exception",
            "runtime_error": f"{type(error).__name__}: {error}",
            "duration_ms": 0.0,
            "runtime_reported_duration_ms": 0.0,
            "turn_count": 0,
            "request_count": 0,
            "tool_call_count": 0,
            "tool_failure_count": 0,
            "permission_rejection_count": 0,
            "spawn_count": 0,
            "spawn_attempt_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "final_text": "",
            "process_timed_out": False,
            "process_return_code": 1,
        }
    finally:
        metrics_summary = resident_monitor.close()
        monitor.close()
    marker_result = evaluate_markers(
        runtime_result["final_text"], condition.task.required_marker_groups
    )
    result = {
        "schema_version": 1,
        "condition_id": condition.condition_id,
        "order": condition.order,
        "runtime": condition.runtime,
        "policy": condition.policy,
        "structure": condition.task.structure,
        "block": condition.task.block,
        "task_id": condition.task.task_id,
        "model": spec.model,
        "started_at": started_at,
        "finished_at": utc_now(),
        "prompt_sha256": text_sha256(condition.prompt),
        "cache_flush": cache_flush,
        **metrics_summary,
        **runtime_result,
        **marker_result,
    }
    result["sample_valid"] = bool(
        result["runtime_success"]
        and int(result["permission_rejection_count"]) == 0
    )
    result["task_success"] = bool(
        result["sample_valid"] and result["content_gate_passed"]
    )
    write_json(partial_dir / "result.json", result)
    write_json(
        partial_dir / "artifact_manifest.json",
        {"schema_version": 1, "files": artifact_hashes(partial_dir)},
    )
    partial_dir.rename(final_dir)
    return result


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "order",
        "condition_id",
        "runtime",
        "policy",
        "structure",
        "block",
        "task_id",
        "runtime_success",
        "task_success",
        "runtime_status",
        "sample_valid",
        "spawn_count",
        "spawn_attempt_count",
        "duration_ms",
        "turn_count",
        "request_count",
        "tool_call_count",
        "tool_failure_count",
        "permission_rejection_count",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "max_resident_tokens",
        "max_resident_pressure",
        "metrics_sample_count",
        "metrics_error_count",
        "marker_coverage",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the paired runtime x prompt-policy x task-structure matrix."
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sglang-base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument(
        "--qwen-sglang-base-url", default="http://172.20.0.1:18000/v1"
    )
    parser.add_argument("--qwen-proxy", type=Path, default=DEFAULT_QWEN_PROXY)
    parser.add_argument("--qwen-proxy-allow", default="172.20.0.1:18000")
    parser.add_argument(
        "--condition-id",
        action="append",
        default=[],
        help="Run only the named condition; repeat for multiple conditions.",
    )
    parser.add_argument("--bridge-port", type=int, default=18080)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--max-completion-tokens", type=int, default=32768)
    parser.add_argument("--client-context-window", type=int, default=163840)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=24)
    parser.add_argument(
        "--codex",
        default=shutil.which("codex") or "codex",
    )
    parser.add_argument("--qwen-runner", type=Path, default=DEFAULT_QWEN_RUNNER)
    parser.add_argument("--qwen-settings", type=Path, default=DEFAULT_QWEN_SETTINGS)
    args = parser.parse_args()

    args.spec = args.spec.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.qwen_runner = args.qwen_runner.expanduser().resolve()
    args.qwen_settings = args.qwen_settings.expanduser().resolve()
    args.qwen_proxy = args.qwen_proxy.expanduser().resolve()
    spec = MatrixSpec.load(args.spec)
    if not args.qwen_runner.is_file() or not os.access(args.qwen_runner, os.X_OK):
        raise ValueError(f"Qwen runner is not executable: {args.qwen_runner}")
    if not args.qwen_settings.is_file():
        raise ValueError(f"Qwen settings do not exist: {args.qwen_settings}")
    if not args.qwen_proxy.is_file():
        raise ValueError(f"Qwen sandbox proxy does not exist: {args.qwen_proxy}")
    health = http_json(args.sglang_base_url.rstrip("/")[:-3] + "/health")
    models = http_json(args.sglang_base_url.rstrip("/") + "/models")
    server_info = http_json(
        args.sglang_base_url.rstrip("/")[:-3] + "/get_server_info"
    )
    args.sglang_pool_tokens = int(server_info["max_total_num_tokens"])
    model_ids = [item.get("id") for item in models.get("data", ())]
    if spec.model not in model_ids:
        raise RuntimeError(f"fixed model is not served: {spec.model}; got {model_ids}")

    staging, partial_manifest = prepare_staging(
        args.run_dir, args.spec, resume=args.resume
    )
    all_conditions = build_conditions(spec)
    condition_ids = {condition.condition_id for condition in all_conditions}
    unknown_conditions = sorted(set(args.condition_id) - condition_ids)
    if unknown_conditions:
        raise ValueError(f"unknown condition ids: {unknown_conditions}")
    conditions = tuple(
        condition
        for condition in all_conditions
        if not args.condition_id or condition.condition_id in args.condition_id
    )
    qwen_version_environment = os.environ.copy()
    qwen_version_environment.update(
        {"QWEN_WORKSPACE_DISPOSABLE": "1", "QWEN_SANDBOX": "false"}
    )
    environment = {
        "model": spec.model,
        "repository": str(spec.repository),
        "repository_head": command_output(
            ["git", "rev-parse", "HEAD"], cwd=spec.repository
        ),
        "repository_status": command_output(
            ["git", "status", "--short"], cwd=spec.repository
        ),
        "codex_version": command_output([args.codex, "--version"], cwd=spec.repository),
        "qwen_version": command_output(
            [str(args.qwen_runner), "--version"],
            cwd=spec.repository,
            environment=qwen_version_environment,
        ),
        "sglang_health": health,
        "served_models": model_ids,
        "sglang_base_url": args.sglang_base_url,
        "qwen_sglang_base_url": args.qwen_sglang_base_url,
        "qwen_proxy_allow": args.qwen_proxy_allow,
        "sglang_pool_tokens": args.sglang_pool_tokens,
        "selected_condition_ids": [
            condition.condition_id for condition in conditions
        ],
        "max_completion_tokens": args.max_completion_tokens,
        "client_context_window": args.client_context_window,
        "timeout_seconds": args.timeout_seconds,
        "max_turns": args.max_turns,
        "max_tool_calls": args.max_tool_calls,
    }
    write_json(staging / "environment.json", environment)

    rows = []
    total = len(conditions)
    for index, condition in enumerate(conditions, 1):
        print(
            f"[{index}/{total}] {condition.runtime} {condition.policy} "
            f"{condition.task.task_id}",
            flush=True,
        )
        result = run_condition(args, spec, condition, staging)
        rows.append(result)
        print(
            "  status="
            f"{result['runtime_status']} spawn={result['spawn_count']} "
            f"task_success={result['task_success']} "
            f"duration={result['duration_ms'] / 1000.0:.1f}s",
            flush=True,
        )

    rows.sort(key=lambda item: int(item["order"]))
    with (staging / "runs.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    write_rows_csv(staging / "runs.csv", rows)
    summary = aggregate_rows(rows)
    write_json(staging / "summary.json", summary)
    final_manifest = {
        **partial_manifest,
        "status": (
            "complete"
            if all(bool(row["runtime_success"]) for row in rows)
            else "complete_with_runtime_failures"
        ),
        "finished_at": utc_now(),
        "condition_count": len(rows),
        "runtime_success_count": sum(bool(row["runtime_success"]) for row in rows),
        "task_success_count": sum(bool(row["task_success"]) for row in rows),
        "environment_sha256": sha256(staging / "environment.json"),
        "summary_sha256": sha256(staging / "summary.json"),
        "runs_sha256": sha256(staging / "runs.jsonl"),
    }
    write_json(staging / "run_manifest.json", final_manifest)
    (staging / "run_manifest.partial.json").unlink()
    staging.rename(args.run_dir)
    print(f"Matrix complete: {args.run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
