#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress a running SGLang server with full Qwen Code requests."
    )
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--concurrency", type=int, default=7)
    parser.add_argument("--pool-tokens", type=int, required=True)
    parser.add_argument("--protocol-max-tokens", type=int, default=32768)
    parser.add_argument("--stress-max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--min-free-mib", type=float, default=1024.0)
    parser.add_argument("--load-poll-ms", type=float, default=50.0)
    return parser.parse_args()


def load_request(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    request = payload.get("request", payload)
    if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
        raise ValueError(f"{path} does not contain an OpenAI chat request")
    return request


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def flush_cache(base_url: str, timeout: float) -> None:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    request = urllib.request.Request(
        f"{root}/flush_cache", data=b"", method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def server_alive(base_url: str, timeout: float) -> bool:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        with urllib.request.urlopen(f"{root}/get_model_info", timeout=timeout):
            return True
    except (OSError, urllib.error.URLError):
        return False


def prepare_request(
    template: dict[str, Any], *, max_tokens: int, marker: str | None = None
) -> dict[str, Any]:
    request = copy.deepcopy(template)
    request["max_tokens"] = max_tokens
    request["stream"] = False
    request.pop("stream_options", None)
    if marker is not None:
        messages = request["messages"]
        system = next(
            (message for message in messages if message.get("role") == "system"),
            None,
        )
        if system is None or not isinstance(system.get("content"), str):
            raise ValueError("request must have a string system message")
        system["content"] = f"{marker}\n{system['content']}"
    return request


def response_summary(index: int, response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or [{}]
    return {
        "index": index,
        "usage": response.get("usage") or {},
        "finish_reason": choices[0].get("finish_reason"),
        "error": None,
    }


class NvidiaMonitor:
    def __init__(self, gpu: int) -> None:
        self.gpu = gpu
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.samples: list[dict[str, float | str]] = []

    def __enter__(self) -> "NvidiaMonitor":
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                str(self.gpu),
                "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
                "--loop-ms=200",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                continue
            try:
                self.samples.append(
                    {
                        "timestamp": fields[0],
                        "memory_used_mib": float(fields[1]),
                        "memory_free_mib": float(fields[2]),
                        "gpu_utilization_percent": float(fields[3]),
                    }
                )
            except ValueError:
                continue

    def __exit__(self, *_: object) -> None:
        assert self.process is not None
        self.process.terminate()
        try:
            self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5.0)
        if self.thread is not None:
            self.thread.join(timeout=5.0)

    def summary(self) -> dict[str, float | int | None]:
        if not self.samples:
            return {"sample_count": 0, "max_used_mib": None, "min_free_mib": None}
        return {
            "sample_count": len(self.samples),
            "max_used_mib": max(
                float(sample["memory_used_mib"]) for sample in self.samples
            ),
            "min_free_mib": min(
                float(sample["memory_free_mib"]) for sample in self.samples
            ),
            "max_gpu_utilization_percent": max(
                float(sample["gpu_utilization_percent"]) for sample in self.samples
            ),
        }


class SGLangLoadMonitor:
    def __init__(self, base_url: str, pool_tokens: int, poll_ms: float) -> None:
        root = base_url.rstrip("/")
        self.root = root[:-3] if root.endswith("/v1") else root
        self.pool_tokens = pool_tokens
        self.poll_seconds = poll_ms / 1000.0
        self.samples: list[dict[str, float | int]] = []
        self.errors: list[str] = []
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "SGLangLoadMonitor":
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def _poll(self) -> None:
        while not self.stop.is_set():
            try:
                with urllib.request.urlopen(
                    f"{self.root}/get_load", timeout=2.0
                ) as response:
                    demand_load = int(json.load(response)["load"])
                with urllib.request.urlopen(
                    f"{self.root}/metrics", timeout=2.0
                ) as response:
                    metrics = response.read().decode("utf-8")
                matches = re.findall(
                    r"^sglang:num_used_tokens(?:\{[^}]*\})?\s+([0-9.eE+-]+)$",
                    metrics,
                    flags=re.MULTILINE,
                )
                if not matches:
                    raise ValueError("sglang:num_used_tokens metric is absent")
                resident_tokens = max(int(float(value)) for value in matches)
                self.samples.append(
                    {
                        "monotonic_seconds": time.monotonic(),
                        "demand_load_tokens": demand_load,
                        "resident_tokens": resident_tokens,
                    }
                )
            except Exception as error:
                self.errors.append(f"{type(error).__name__}: {error}")
            self.stop.wait(self.poll_seconds)

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)

    def summary(self) -> dict[str, float | int | None]:
        if not self.samples:
            return {
                "sample_count": 0,
                "max_demand_load_tokens": None,
                "max_demand_pressure": None,
                "max_resident_tokens": None,
                "max_resident_pressure": None,
                "error_count": len(self.errors),
            }
        max_demand = max(
            int(sample["demand_load_tokens"]) for sample in self.samples
        )
        max_resident = max(int(sample["resident_tokens"]) for sample in self.samples)
        return {
            "sample_count": len(self.samples),
            "max_demand_load_tokens": max_demand,
            "max_demand_pressure": max_demand / self.pool_tokens,
            "max_resident_tokens": max_resident,
            "max_resident_pressure": max_resident / self.pool_tokens,
            "error_count": len(self.errors),
        }


def main() -> int:
    args = parse_args()
    if args.concurrency <= 0 or args.pool_tokens <= 0:
        raise ValueError("concurrency and pool-tokens must be positive")

    template = load_request(args.request_log)
    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    started = time.monotonic()

    protocol_request = prepare_request(
        template, max_tokens=args.protocol_max_tokens
    )
    protocol_response = post_json(endpoint, protocol_request, args.timeout)
    protocol = response_summary(-1, protocol_response)
    flush_cache(args.base_url, min(args.timeout, 30.0))

    results: list[dict[str, Any]] = []
    with NvidiaMonitor(args.gpu) as gpu_monitor, SGLangLoadMonitor(
        args.base_url, args.pool_tokens, args.load_poll_ms
    ) as load_monitor:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    post_json,
                    endpoint,
                    prepare_request(
                        template,
                        max_tokens=args.stress_max_tokens,
                        marker=f"BELIEFKV_UNIQUE_BRANCH_{index:04d}",
                    ),
                    args.timeout,
                ): index
                for index in range(args.concurrency)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results.append(response_summary(index, future.result()))
                except Exception as error:
                    results.append(
                        {
                            "index": index,
                            "usage": {},
                            "finish_reason": None,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

    results.sort(key=lambda item: int(item["index"]))
    aggregate_request_tokens = sum(
        int(item["usage"].get("prompt_tokens") or 0)
        + int(item["usage"].get("completion_tokens") or 0)
        for item in results
    )
    failures = sum(item["error"] is not None for item in results)
    payload = {
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request_log": str(args.request_log.resolve()),
        "base_url": args.base_url,
        "pool_tokens": args.pool_tokens,
        "protocol_max_tokens": args.protocol_max_tokens,
        "protocol": protocol,
        "concurrency": args.concurrency,
        "stress_max_tokens": args.stress_max_tokens,
        "stress_results": results,
        "aggregate_request_tokens": aggregate_request_tokens,
        "nominal_request_coverage": aggregate_request_tokens / args.pool_tokens,
        "failure_count": failures,
        "server_alive_after_stress": server_alive(args.base_url, 10.0),
        "gpu": gpu_monitor.summary(),
        "sglang_load": load_monitor.summary(),
        "required_min_free_mib": args.min_free_mib,
        "duration_seconds": time.monotonic() - started,
    }
    observed_min_free = payload["gpu"]["min_free_mib"]
    observed_pressure = payload["sglang_load"]["max_resident_pressure"]
    payload["passed"] = (
        failures == 0
        and payload["server_alive_after_stress"]
        and observed_pressure is not None
        and observed_pressure >= 0.8
        and observed_min_free is not None
        and observed_min_free >= args.min_free_mib
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
