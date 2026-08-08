#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    parser.add_argument(
        "--phases",
        default="prefill,decode",
        help="Comma-separated subset of prefill,decode.",
    )
    parser.add_argument(
        "--prefill-token-targets",
        default="1024,4096,16384,30720",
        help="Comma-separated approximate prompt-token targets; 30720 is the safe 32K-window case.",
    )
    parser.add_argument("--cache-hit-ratios", default="0,0.5,0.9")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument(
        "--decode-context-token-targets",
        default="1024,4096,16384,30720",
        help="Approximate decode sequence lengths; runtime sequence labels are authoritative.",
    )
    parser.add_argument(
        "--max-batch-token-budget",
        type=int,
        default=143_360,
        help=(
            "Skip decode compositions whose aggregate context plus requested "
            "decode tokens exceed this budget. Keep it below max-total-tokens."
        ),
    )
    parser.add_argument("--repeats", type=int, default=1)
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
    profile_id: str,
    kind: str,
    requested_batch_size: int,
    target_prompt_tokens: int,
    target_cache_hit_ratio: float,
    warmup: bool,
    elapsed_ms: float,
    response: dict[str, Any],
) -> dict[str, object]:
    choice = (response.get("choices") or [{}])[0]
    return {
        "split": split,
        "case_id": case_id,
        "profile_id": profile_id,
        "kind": kind,
        "requested_batch_size": requested_batch_size,
        "target_prompt_tokens": target_prompt_tokens,
        "target_cache_hit_ratio": target_cache_hit_ratio,
        "warmup": warmup,
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
    profile_id: str,
    kind: str,
    requested_batch_size: int,
    target_prompt_tokens: int,
    target_cache_hit_ratio: float,
    warmup: bool = False,
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
        profile_id=profile_id,
        kind=kind,
        requested_batch_size=requested_batch_size,
        target_prompt_tokens=target_prompt_tokens,
        target_cache_hit_ratio=target_cache_hit_ratio,
        warmup=warmup,
        elapsed_ms=elapsed_ms,
        response=response,
    )


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("integer calibration dimensions must be positive")
    return tuple(dict.fromkeys(values))


def _parse_float_csv(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("floating-point calibration dimensions are required")
    return tuple(dict.fromkeys(values))


def _record_result(
    staging: Path,
    results: list[dict[str, object]],
    result: dict[str, object],
) -> None:
    results.append(result)
    with (staging / "completed_requests.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")


def _run_prefill_case(
    *,
    args: argparse.Namespace,
    endpoint: str,
    staging: Path,
    results: list[dict[str, object]],
    split: str,
    case_id: str,
    profile_id: str,
    target_tokens: int,
    hit_ratio: float,
) -> None:
    shared_words = int(target_tokens * hit_ratio)
    unique_prefix = f"case-{split}-{case_id} "
    shared = unique_prefix + "alpha " * shared_words
    if shared_words:
        warmup_id = f"{case_id}-warmup"
        warmup = _request(
            args.model,
            split,
            warmup_id,
            shared + "warmup-finish",
            max_tokens=1,
        )
        _record_result(
            staging,
            results,
            _run_one(
                endpoint,
                warmup,
                timeout=args.timeout,
                split=split,
                case_id=warmup_id,
                profile_id=profile_id,
                kind="cache_warmup",
                requested_batch_size=1,
                target_prompt_tokens=shared_words,
                target_cache_hit_ratio=hit_ratio,
                warmup=True,
            ),
        )
    suffix_words = max(1, target_tokens - shared_words)
    content = shared + "beta " * suffix_words + "measured-finish"
    payload = _request(
        args.model,
        split,
        case_id,
        content,
        max_tokens=1,
    )
    _record_result(
        staging,
        results,
        _run_one(
            endpoint,
            payload,
            timeout=args.timeout,
            split=split,
            case_id=case_id,
            profile_id=profile_id,
            kind="prefill",
            requested_batch_size=1,
            target_prompt_tokens=target_tokens,
            target_cache_hit_ratio=hit_ratio,
        ),
    )


def main() -> int:
    args = _parse_args()
    prompt_targets = _parse_int_csv(args.prefill_token_targets)
    hit_ratios = _parse_float_csv(args.cache_hit_ratios)
    batch_sizes = _parse_int_csv(args.batch_sizes)
    decode_context_targets = _parse_int_csv(args.decode_context_token_targets)
    phases = tuple(
        dict.fromkeys(item.strip() for item in args.phases.split(",") if item.strip())
    )
    if (
        args.timeout <= 0
        or args.decode_tokens <= 0
        or args.repeats <= 0
        or args.max_batch_token_budget <= 0
    ):
        raise ValueError("timeout and decode token count must be positive")
    if any(not 0 <= value < 1 for value in hit_ratios):
        raise ValueError("cache-hit ratios must be in [0, 1)")
    if not phases or not set(phases).issubset({"prefill", "decode"}):
        raise ValueError("phases must be a non-empty subset of prefill,decode")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    results: list[dict[str, object]] = []
    skipped_decode_profiles: list[dict[str, object]] = []
    started_at = datetime.now(timezone.utc)

    prefill_cases = {
        "train": (prompt_targets, hit_ratios),
        "holdout": (
            tuple(sorted({2048, 8192, 24576}.difference(prompt_targets))),
            (0.25, 0.75),
        ),
    }
    if "prefill" in phases:
        for split, (lengths, ratios) in prefill_cases.items():
            for target_tokens in lengths:
                for hit_ratio in ratios:
                    ratio_tag = int(round(hit_ratio * 100))
                    profile_id = f"prefill-t{target_tokens}-h{ratio_tag}"
                    for repeat in range(args.repeats):
                        case_id = f"{profile_id}-r{repeat}"
                        _run_prefill_case(
                            args=args,
                            endpoint=endpoint,
                            staging=staging,
                            results=results,
                            split=split,
                            case_id=case_id,
                            profile_id=profile_id,
                            target_tokens=target_tokens,
                            hit_ratio=hit_ratio,
                        )

    decode_contexts = {
        "train": decode_context_targets,
        "holdout": tuple(
            sorted({2048, 8192, 24576}.difference(decode_context_targets))
        ),
    }
    if "decode" in phases:
        for split in ("train", "holdout"):
            for batch_size in batch_sizes:
                for repeat in range(args.repeats):
                    homogeneous = [
                        (f"c{target}", (target,) * batch_size)
                        for target in decode_contexts[split]
                    ]
                    mixed = []
                    if batch_size > 1 and len(decode_contexts[split]) > 1:
                        low = min(decode_contexts[split])
                        high = max(decode_contexts[split])
                        mixed.append(
                            (
                                f"mixed-{low}-{high}",
                                tuple(
                                    low if index % 2 == 0 else high
                                    for index in range(batch_size)
                                ),
                            )
                        )
                    for profile, context_targets in (*homogeneous, *mixed):
                        aggregate_tokens = sum(context_targets) + (
                            len(context_targets) * args.decode_tokens
                        )
                        if aggregate_tokens > args.max_batch_token_budget:
                            skipped_decode_profiles.append(
                                {
                                    "split": split,
                                    "batch_size": batch_size,
                                    "profile": profile,
                                    "context_targets": list(context_targets),
                                    "aggregate_token_budget": aggregate_tokens,
                                    "reason": "exceeds_max_batch_token_budget",
                                }
                            )
                            continue
                        barrier = threading.Barrier(batch_size)
                        with ThreadPoolExecutor(max_workers=batch_size) as executor:
                            futures = []
                            profile_id = f"decode-b{batch_size}-{profile}"
                            for index, context_target in enumerate(context_targets):
                                case_id = f"{profile_id}-r{repeat}-i{index}"
                                content = (
                                    f"unique-{split}-{case_id} "
                                    + "context " * max(1, context_target)
                                    + "Emit an infinite deterministic numbered sequence: 1 2 3 4"
                                )
                                payload = _request(
                                    args.model,
                                    split,
                                    case_id,
                                    content,
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
                                        profile_id=profile_id,
                                        kind="decode",
                                        requested_batch_size=batch_size,
                                        target_prompt_tokens=context_target,
                                        target_cache_hit_ratio=0.0,
                                        barrier=barrier,
                                    )
                                )
                            failures: list[Exception] = []
                            for future in as_completed(futures):
                                try:
                                    result = future.result()
                                except Exception as error:
                                    failures.append(error)
                                else:
                                    _record_result(staging, results, result)
                            if failures:
                                raise failures[0]

    manifest = {
        "schema_version": 1,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "decode_tokens": args.decode_tokens,
        "design": {
            "prefill_token_targets": list(prompt_targets),
            "cache_hit_ratios": list(hit_ratios),
            "decode_batch_sizes": list(batch_sizes),
            "decode_context_token_targets": list(decode_context_targets),
            "holdout_decode_context_token_targets": list(
                decode_contexts["holdout"]
            ),
            "mixed_decode_sequence_batches": True,
            "max_batch_token_budget": args.max_batch_token_budget,
            "skipped_decode_profiles": skipped_decode_profiles,
            "repeats": args.repeats,
            "phases": list(phases),
            "holdout_prefill_token_targets": list(prefill_cases["holdout"][0]),
            "holdout_cache_hit_ratios": list(prefill_cases["holdout"][1]),
            "prompt_target_semantics": (
                "approximate repeated-token target; runtime prompt_tokens and "
                "cache_hit_tokens are authoritative labels"
            ),
        },
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
