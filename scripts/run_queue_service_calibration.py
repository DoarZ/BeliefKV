#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tagged prefill/decode service calibration requests."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument(
        "--model", default="Qwen3-Coder-30B-A3B-Instruct-FP8"
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--decode-tokens", type=int, default=64)
    return parser.parse_args()


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("model server response must be an object")
    if value.get("error"):
        raise RuntimeError(f"model server error: {value['error']}")
    return value


def _metadata(split: str, case_id: str) -> dict[str, object]:
    workflow_id = f"service-calibration:{split}:{case_id}"
    return {
        "root_workflow_id": workflow_id,
        "invocation_id": f"{workflow_id}:invocation",
        "context_id": f"{workflow_id}:context",
        "context_epoch": 0,
        "agent_definition_id": "service-calibration",
        "agent_instance_id": f"service-calibration:{case_id}",
        "parent_invocation_id": None,
        "parent_context_id": None,
        "relation_type": "root",
        "context_mode": "fresh",
        "execution_mode": "foreground",
        "return_target_id": None,
        "join_id": None,
    }


def _request(
    model: str,
    split: str,
    case_id: str,
    content: str,
    *,
    max_tokens: int,
) -> dict[str, object]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Continue deterministically until the output token limit.",
            },
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
        "beliefkv_metadata": _metadata(split, case_id),
    }


def _summary(
    *,
    split: str,
    case_id: str,
    kind: str,
    requested_batch_size: int,
    elapsed_ms: float,
    response: dict[str, Any],
) -> dict[str, object]:
    choice = (response.get("choices") or [{}])[0]
    return {
        "split": split,
        "case_id": case_id,
        "kind": kind,
        "requested_batch_size": requested_batch_size,
        "client_elapsed_ms": elapsed_ms,
        "usage": response.get("usage") or {},
        "finish_reason": choice.get("finish_reason"),
    }


def _run_one(
    endpoint: str,
    payload: dict[str, object],
    *,
    timeout: float,
    split: str,
    case_id: str,
    kind: str,
    requested_batch_size: int,
    barrier: threading.Barrier | None = None,
) -> dict[str, object]:
    if barrier is not None:
        barrier.wait(timeout=timeout)
    started = time.perf_counter_ns()
    response = _post_json(endpoint, payload, timeout)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return _summary(
        split=split,
        case_id=case_id,
        kind=kind,
        requested_batch_size=requested_batch_size,
        elapsed_ms=elapsed_ms,
        response=response,
    )


def main() -> int:
    args = _parse_args()
    if args.timeout <= 0 or args.decode_tokens <= 0:
        raise ValueError("timeout and decode token count must be positive")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    results: list[dict[str, object]] = []
    started_at = datetime.now(timezone.utc)

    prefill_cases = {
        "train": (512, 1024, 2048, 3072, 6144, 10240),
        "holdout": (768, 1536, 2560, 5120, 8192),
    }
    for split, lengths in prefill_cases.items():
        for index, word_count in enumerate(lengths):
            case_id = f"prefill-{word_count}-{index}"
            content = f"unique-{split}-{case_id} " + "alpha " * word_count + "finish"
            payload = _request(
                args.model,
                split,
                case_id,
                content,
                max_tokens=1,
            )
            results.append(
                _run_one(
                    endpoint,
                    payload,
                    timeout=args.timeout,
                    split=split,
                    case_id=case_id,
                    kind="prefill",
                    requested_batch_size=1,
                )
            )

    for split in ("train", "holdout"):
        for batch_size in (1, 2, 4):
            for repeat in range(2):
                barrier = threading.Barrier(batch_size)
                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    futures = []
                    for index in range(batch_size):
                        case_id = f"decode-b{batch_size}-r{repeat}-i{index}"
                        payload = _request(
                            args.model,
                            split,
                            case_id,
                            f"unique-{split}-{case_id} Emit an infinite deterministic "
                            "numbered sequence: 1 2 3 4",
                            max_tokens=args.decode_tokens,
                        )
                        futures.append(
                            executor.submit(
                                _run_one,
                                endpoint,
                                payload,
                                timeout=args.timeout,
                                split=split,
                                case_id=case_id,
                                kind="decode",
                                requested_batch_size=batch_size,
                                barrier=barrier,
                            )
                        )
                    results.extend(future.result() for future in futures)

    manifest = {
        "schema_version": 1,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "decode_tokens": args.decode_tokens,
        "request_count": len(results),
        "split_counts": {
            split: sum(item["split"] == split for item in results)
            for split in ("train", "holdout")
        },
        "results": sorted(results, key=lambda item: str(item["case_id"])),
        "timing_scope": (
            "client timing is diagnostic only; calibration uses runtime "
            "gpu_service_sample events"
        ),
    }
    (staging / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    staging.replace(output)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
