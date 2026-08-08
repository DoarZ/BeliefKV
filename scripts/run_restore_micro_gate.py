#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic P5 D2H/restore/H2D correctness probe."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument(
        "--model", default="Qwen3-Coder-30B-A3B-Instruct-FP8"
    )
    parser.add_argument("--victim-prompt-words", type=int, default=48_000)
    parser.add_argument("--anchor-prompt-words", type=int, default=48_000)
    parser.add_argument("--replacement-prompt-words", type=int, default=64_000)
    parser.add_argument("--holder-output-tokens", type=int, default=1024)
    parser.add_argument("--replacement-output-tokens", type=int, default=256)
    parser.add_argument(
        "--required-max-running-requests",
        type=int,
        default=2,
        help=(
            "Required server running-slot count. Two holder requests must occupy "
            "all slots so the replacement remains physically waiting."
        ),
    )
    parser.add_argument("--service-wait-seconds", type=float, default=300.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=1800.0)
    return parser.parse_args()


def _metadata(workflow_id: str, role: str) -> dict[str, object]:
    return {
        "root_workflow_id": workflow_id,
        "invocation_id": f"{workflow_id}:invocation",
        "context_id": f"{workflow_id}:context",
        "context_epoch": 0,
        "agent_definition_id": "restore-micro-gate",
        "agent_instance_id": f"restore-micro-gate:{role}",
        "parent_invocation_id": None,
        "parent_context_id": None,
        "relation_type": "root",
        "context_mode": "fresh",
        "execution_mode": "foreground",
        "return_target_id": None,
        "join_id": None,
    }


def _payload(
    model: str,
    workflow_id: str,
    role: str,
    prompt_words: int,
    output_tokens: int,
) -> dict[str, object]:
    unique = f"restore-micro-gate-{role}"
    content = f"{unique} " + f"{role} " * prompt_words
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Emit deterministic plain text continuously until the token "
                    "limit. Do not call tools and do not stop early."
                ),
            },
            {"role": "user", "content": content},
        ],
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
        "beliefkv_metadata": _metadata(workflow_id, role),
    }


def _post(endpoint: str, payload: dict[str, object], timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict) or value.get("error"):
        raise RuntimeError(f"invalid model response: {value!r}")
    choice = (value.get("choices") or [{}])[0]
    return {
        "elapsed_seconds": time.monotonic() - started,
        "finish_reason": choice.get("finish_reason"),
        "usage": value.get("usage") or {},
    }


def _server_info(base_url: str, timeout: float) -> dict[str, Any]:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    with urllib.request.urlopen(f"{root}/get_server_info", timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict) or value.get("status") != "ready":
        raise RuntimeError(f"model server is not ready: {value!r}")
    return value


def _served_workflows(path: Path) -> set[str]:
    workflows: set[str] = set()
    if not path.is_file():
        return workflows
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("event") != "gpu_service_sample":
                continue
            for sample in value.get("request_samples", ()):
                if isinstance(sample, dict) and sample.get("token_delta", 0) > 0:
                    workflows.add(str(sample.get("workflow_id", "")))
    return workflows


def _wait_for_initial_service(
    audit: Path,
    futures: tuple[Future[dict[str, Any]], ...],
    timeout: float,
) -> None:
    expected = {"restore-micro-gate:victim", "restore-micro-gate:anchor"}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for future in futures:
            if future.done() and future.exception() is not None:
                future.result()
        if expected.issubset(_served_workflows(audit)):
            return
        time.sleep(0.5)
    observed = sorted(_served_workflows(audit))
    raise TimeoutError(
        f"initial GPU service was not observed for {sorted(expected)}; "
        f"observed={observed}"
    )


def main() -> int:
    args = _parse_args()
    pause_file = Path(
        os.environ.get(
            "BELIEFKV_EXPERIMENT_PAUSE_FILE",
            "/tmp/beliefkv-experiments.paused",
        )
    ).expanduser()
    if pause_file.exists():
        print(f"BeliefKV experiments are paused: {pause_file}", file=sys.stderr)
        return 75
    positive = (
        args.victim_prompt_words,
        args.anchor_prompt_words,
        args.replacement_prompt_words,
        args.holder_output_tokens,
        args.replacement_output_tokens,
        args.required_max_running_requests,
        args.service_wait_seconds,
        args.request_timeout_seconds,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("micro-gate sizes and timeouts must be positive")
    server_info = _server_info(args.base_url, min(10.0, args.request_timeout_seconds))
    actual_max_running = int(server_info.get("max_running_requests", 0) or 0)
    if actual_max_running != args.required_max_running_requests:
        raise RuntimeError(
            "deterministic restore micro-gate requires "
            f"max_running_requests={args.required_max_running_requests}, got "
            f"{actual_max_running}; no workload was submitted"
        )
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    started_at = datetime.now(timezone.utc)
    specs = {
        "victim": (
            "restore-micro-gate:victim",
            args.victim_prompt_words,
            args.holder_output_tokens,
        ),
        "anchor": (
            "restore-micro-gate:anchor",
            args.anchor_prompt_words,
            args.holder_output_tokens,
        ),
        "replacement": (
            "restore-micro-gate:replacement",
            args.replacement_prompt_words,
            args.replacement_output_tokens,
        ),
    }
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        first = {
            role: executor.submit(
                _post,
                endpoint,
                _payload(args.model, workflow_id, role, words, output_tokens),
                args.request_timeout_seconds,
            )
            for role, (workflow_id, words, output_tokens) in specs.items()
            if role != "replacement"
        }
        _wait_for_initial_service(
            args.runtime_audit.expanduser().resolve(),
            tuple(first.values()),
            args.service_wait_seconds,
        )
        replacement_spec = specs["replacement"]
        replacement = executor.submit(
            _post,
            endpoint,
            _payload(
                args.model,
                replacement_spec[0],
                "replacement",
                replacement_spec[1],
                replacement_spec[2],
            ),
            args.request_timeout_seconds,
        )
        for role, future in {**first, "replacement": replacement}.items():
            results[role] = future.result()
    finished_at = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 1,
        "kind": "deterministic_restore_micro_gate",
        "gate_id": "p5g-restore-v1",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "model": args.model,
        "server_preflight": {
            "max_running_requests": actual_max_running,
            "max_total_num_tokens": server_info.get("max_total_num_tokens"),
            "status": server_info.get("status"),
        },
        "request_specs": {
            role: {
                "workflow_id": workflow_id,
                "prompt_words": words,
                "max_tokens": output_tokens,
            }
            for role, (workflow_id, words, output_tokens) in specs.items()
        },
        "results": results,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    staging.replace(output)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
