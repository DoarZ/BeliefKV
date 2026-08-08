from __future__ import annotations

import json
import gzip
import base64
import hashlib
import math
import os
import queue
import secrets
import struct
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, TextIO

if TYPE_CHECKING:
    from beliefkv.policy.reference.base import PolicyInput


class AuditLevel(str, Enum):
    CORRECTNESS = "correctness"
    METRICS = "metrics"
    DEBUG = "debug"


class RuntimeAuditLog:
    """Optional append-only audit log for scheduler/control-plane validation.

    The disabled path performs no file-system work. When enabled, a bounded
    single-consumer writer persists JSON records carrying a run identifier.
    Correctness and metrics records apply backpressure; sampled debug records
    may be dropped when the queue is full.
    """

    SCHEMA_VERSION = 2

    def __init__(
        self,
        path: str | Path | None,
        *,
        run_id: str | None = None,
        max_pending: int = 8192,
        debug_sample_rate: float = 1.0,
        max_debug_event_bytes: int = 16 * 1024,
        flush_interval_s: float = 1.0,
    ) -> None:
        if max_pending <= 0:
            raise ValueError("runtime audit queue capacity must be positive")
        if not 0 < debug_sample_rate <= 1:
            raise ValueError("runtime audit debug sample rate must be in (0, 1]")
        if max_debug_event_bytes <= 0:
            raise ValueError("runtime audit debug event limit must be positive")
        if not math.isfinite(flush_interval_s) or flush_interval_s <= 0:
            raise ValueError("runtime audit flush interval must be positive")
        self.path = Path(path).expanduser().resolve() if path is not None else None
        self.run_id = run_id or uuid.uuid4().hex
        self._sequence = 0
        self._written_count = 0
        self._dropped_debug_count = 0
        self._sampled_debug_count = 0
        self._oversize_debug_count = 0
        self._debug_seen = 0
        self._debug_sample_stride = max(1, math.ceil(1.0 / debug_sample_rate))
        self._max_debug_event_bytes = max_debug_event_bytes
        self._flush_interval_s = flush_interval_s
        self._stream: TextIO | None = None
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[dict[str, Any], AuditLevel] | None] = (
            queue.Queue(maxsize=max_pending)
        )
        self._writer_error: BaseException | None = None
        self._writer_thread: threading.Thread | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("a", encoding="utf-8", buffering=64 * 1024)
            self._writer_thread = threading.Thread(
                target=self._writer_main,
                name="beliefkv-runtime-audit-writer",
                daemon=True,
            )
            self._writer_thread.start()

    @property
    def enabled(self) -> bool:
        return self._stream is not None

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    def summary(self) -> dict[str, int]:
        return {
            "accepted_count": self._sequence,
            "written_count": self._written_count,
            "pending_count": self.pending_count,
            "sampled_debug_count": self._sampled_debug_count,
            "dropped_debug_count": self._dropped_debug_count,
            "oversize_debug_count": self._oversize_debug_count,
        }

    @staticmethod
    def _default_level(event: str) -> AuditLevel:
        if event in {"physical_bundle_preview", "bundle_lease_aggregated"}:
            return AuditLevel.DEBUG
        if (
            event in {
                "transfer_dispatched",
                "transfer_acknowledged",
                "runtime_initialized",
                "runtime_shutdown",
            }
            or event.endswith("_terminal")
            or event.endswith("_summary")
        ):
            return AuditLevel.CORRECTNESS
        return AuditLevel.METRICS

    def emit(
        self,
        event: str,
        ts_ms: float,
        *,
        audit_level: AuditLevel | str | None = None,
        **fields: Any,
    ) -> None:
        if self._stream is None:
            return
        if not event:
            raise ValueError("audit event must be non-empty")
        if not math.isfinite(ts_ms) or ts_ms < 0:
            raise ValueError("audit timestamp must be finite and non-negative")
        level = (
            self._default_level(event)
            if audit_level is None
            else AuditLevel(audit_level)
        )
        with self._lock:
            if self._writer_error is not None:
                raise RuntimeError("runtime audit writer failed") from self._writer_error
            if level == AuditLevel.DEBUG:
                self._debug_seen += 1
                if (self._debug_seen - 1) % self._debug_sample_stride:
                    self._sampled_debug_count += 1
                    return
            self._sequence += 1
            payload = {
                "schema_version": self.SCHEMA_VERSION,
                "run_id": self.run_id,
                "sequence": self._sequence,
                "pid": os.getpid(),
                "event": event,
                "ts_ms": float(ts_ms),
                "audit_level": level.value,
                **fields,
            }
            if level == AuditLevel.DEBUG:
                try:
                    self._queue.put_nowait((payload, level))
                except queue.Full:
                    self._sequence -= 1
                    self._dropped_debug_count += 1
                return
            while True:
                if self._writer_error is not None:
                    raise RuntimeError("runtime audit writer failed") from self._writer_error
                try:
                    self._queue.put((payload, level), timeout=0.1)
                    return
                except queue.Full:
                    continue

    def _writer_main(self) -> None:
        assert self._stream is not None
        last_flush = time.monotonic()
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    self._stream.flush()
                    return
                payload, level = item
                encoded = json.dumps(
                    payload, sort_keys=True, allow_nan=False
                ).encode("utf-8")
                if (
                    level == AuditLevel.DEBUG
                    and len(encoded) > self._max_debug_event_bytes
                ):
                    self._oversize_debug_count += 1
                    encoded = json.dumps(
                        {
                            "schema_version": self.SCHEMA_VERSION,
                            "run_id": self.run_id,
                            "sequence": payload["sequence"],
                            "pid": payload["pid"],
                            "event": "audit_debug_event_oversize",
                            "ts_ms": payload["ts_ms"],
                            "audit_level": AuditLevel.DEBUG.value,
                            "original_event": payload["event"],
                            "original_size_bytes": len(encoded),
                            "field_names": sorted(payload),
                            "payload_digest": hashlib.blake2b(
                                encoded,
                                digest_size=16,
                                person=b"bkv-audit-event",
                            ).hexdigest(),
                        },
                        sort_keys=True,
                        allow_nan=False,
                    ).encode("utf-8")
                self._stream.write(encoded.decode("utf-8") + "\n")
                self._written_count += 1
                now = time.monotonic()
                if now - last_flush >= self._flush_interval_s:
                    self._stream.flush()
                    last_flush = now
        except BaseException as error:
            self._writer_error = error

    def close(self) -> None:
        thread = self._writer_thread
        if thread is not None:
            while thread.is_alive():
                if self._writer_error is not None:
                    break
                try:
                    self._queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            thread.join()
            self._writer_thread = None
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            error = self._writer_error
        if error is not None:
            raise RuntimeError("runtime audit writer failed") from error

    def __enter__(self) -> "RuntimeAuditLog":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class PolicySnapshotLog:
    """Compressed replay snapshots kept separate from the high-rate audit log."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str | Path | None,
        *,
        trace_id: str,
        trace_sensitivity: str,
        max_pending: int = 8,
    ) -> None:
        if not trace_id:
            raise ValueError("trace_id must be non-empty")
        if trace_sensitivity not in {
            "schedule_invariant",
            "timing_sensitive",
            "semantic_race_sensitive",
        }:
            raise ValueError("unsupported policy trace sensitivity")
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self.path = Path(path).expanduser().resolve() if path is not None else None
        self.trace_id = trace_id
        self.trace_sensitivity = trace_sensitivity
        self._sequence = 0
        self._written_count = 0
        self._dropped_count = 0
        self._stream: TextIO | None = None
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[int, "PolicyInput", str] | None] = (
            queue.Queue(maxsize=max_pending)
        )
        self._writer_error: BaseException | None = None
        self._writer_thread: threading.Thread | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.suffix == ".gz":
                self._stream = gzip.open(
                    self.path,
                    mode="xt",
                    encoding="utf-8",
                )
            else:
                self._stream = self.path.open(
                    "x", encoding="utf-8", buffering=1
                )
            self._writer_thread = threading.Thread(
                target=self._writer_main,
                name="beliefkv-policy-snapshot-writer",
                daemon=True,
            )
            self._writer_thread.start()

    @property
    def enabled(self) -> bool:
        return self._stream is not None

    @property
    def count(self) -> int:
        return self._sequence

    @property
    def written_count(self) -> int:
        return self._written_count

    @property
    def pending_count(self) -> int:
        return max(0, self._sequence - self._written_count)

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    def emit(self, policy_input: "PolicyInput", *, trigger: str) -> int:
        if self._stream is None:
            return 0
        if not trigger:
            raise ValueError("policy snapshot trigger must be non-empty")
        with self._lock:
            if self._writer_error is not None:
                raise RuntimeError("policy snapshot writer failed") from self._writer_error
            sequence = self._sequence + 1
            try:
                self._queue.put_nowait((sequence, policy_input, trigger))
            except queue.Full:
                self._dropped_count += 1
                return 0
            self._sequence = sequence
        return sequence

    def _writer_main(self) -> None:
        assert self._stream is not None
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                sequence, policy_input, trigger = item
                payload = {
                    "schema_version": self.SCHEMA_VERSION,
                    "sequence": sequence,
                    "trace_id": self.trace_id,
                    "trace_sensitivity": self.trace_sensitivity,
                    "trigger": trigger,
                    "policy_input": policy_input.to_dict(),
                }
                self._stream.write(
                    json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
                )
                self._stream.flush()
                self._written_count += 1
        except BaseException as error:
            self._writer_error = error

    def close(self) -> None:
        thread = self._writer_thread
        if thread is not None:
            self._queue.put(None)
            thread.join()
            self._writer_thread = None
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            error = self._writer_error
        if error is not None:
            raise RuntimeError("policy snapshot writer failed") from error

    def __enter__(self) -> "PolicySnapshotLog":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class RequestTokenTraceLog:
    """Asynchronous exact-prefix trace using irreversible run-local symbols."""

    SCHEMA_VERSION = 1
    ENCODING = "run_local_random_u64_bijection+uint64_le_base64"
    VALID_EVENTS = frozenset(
        {"request_prompt", "cache_partial_commit", "cache_final_commit", "cache_reset"}
    )

    def __init__(
        self,
        path: str | Path | None,
        *,
        run_id: str,
    ) -> None:
        if not run_id:
            raise ValueError("token trace run_id must be non-empty")
        self.path = Path(path).expanduser().resolve() if path is not None else None
        self.run_id = run_id
        self._sequence = 0
        self._written_count = 0
        self._stream: TextIO | None = None
        self._lock = threading.Lock()
        self._queue: queue.Queue[
            tuple[int, str, float, tuple[int, ...], dict[str, Any]] | None
        ] = queue.Queue()
        self._writer_error: BaseException | None = None
        self._writer_thread: threading.Thread | None = None
        self._symbol_by_token_id: dict[int, int] = {}
        self._used_symbols: set[int] = set()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.suffix == ".gz":
                self._stream = gzip.open(
                    self.path,
                    mode="xt",
                    encoding="utf-8",
                )
            else:
                self._stream = self.path.open("x", encoding="utf-8", buffering=1)
            self._writer_thread = threading.Thread(
                target=self._writer_main,
                name="beliefkv-request-token-writer",
                daemon=True,
            )
            self._writer_thread.start()

    @property
    def enabled(self) -> bool:
        return self._stream is not None

    @property
    def count(self) -> int:
        return self._sequence

    @property
    def written_count(self) -> int:
        return self._written_count

    @property
    def pending_count(self) -> int:
        return max(0, self._sequence - self._written_count)

    def emit(
        self,
        event: str,
        ts_ms: float,
        token_ids: Sequence[int] = (),
        **fields: Any,
    ) -> int:
        if self._stream is None:
            return 0
        if event not in self.VALID_EVENTS:
            raise ValueError(f"unsupported request token trace event: {event}")
        if ts_ms < 0:
            raise ValueError("token trace timestamp must be non-negative")
        with self._lock:
            if self._writer_error is not None:
                raise RuntimeError("request token trace writer failed") from self._writer_error
            symbols = tuple(self._symbol(int(token_id)) for token_id in token_ids)
            self._sequence += 1
            sequence = self._sequence
        self._queue.put((sequence, event, float(ts_ms), symbols, dict(fields)))
        return sequence

    def _symbol(self, token_id: int) -> int:
        if token_id < 0:
            raise ValueError("token IDs must be non-negative")
        existing = self._symbol_by_token_id.get(token_id)
        if existing is not None:
            return existing
        symbol = secrets.randbits(64)
        while symbol in self._used_symbols:
            symbol = secrets.randbits(64)
        self._symbol_by_token_id[token_id] = symbol
        self._used_symbols.add(symbol)
        return symbol

    def _writer_main(self) -> None:
        assert self._stream is not None
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                sequence, event, ts_ms, symbols, fields = item
                packed = (
                    struct.pack(f"<{len(symbols)}Q", *symbols) if symbols else b""
                )
                payload = {
                    "schema_version": self.SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "sequence": sequence,
                    "event": event,
                    "ts_ms": ts_ms,
                    "token_count": len(symbols),
                    "token_encoding": self.ENCODING,
                    "token_symbols_b64": base64.b64encode(packed).decode("ascii"),
                    "token_symbols_blake2b": hashlib.blake2b(
                        packed,
                        digest_size=16,
                        person=b"bk-token-trace",
                    ).hexdigest(),
                    **fields,
                }
                self._stream.write(
                    json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
                )
                self._stream.flush()
                self._written_count += 1
        except BaseException as error:
            self._writer_error = error

    def close(self) -> None:
        thread = self._writer_thread
        if thread is not None:
            self._queue.put(None)
            thread.join()
            self._writer_thread = None
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            error = self._writer_error
        if error is not None:
            raise RuntimeError("request token trace writer failed") from error

    def __enter__(self) -> "RequestTokenTraceLog":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
