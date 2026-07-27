from __future__ import annotations

import json
import gzip
import base64
import hashlib
import os
import queue
import secrets
import struct
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, TextIO

if TYPE_CHECKING:
    from beliefkv.policy.reference.base import PolicyInput


class RuntimeAuditLog:
    """Optional append-only audit log for scheduler/control-plane validation.

    The disabled path performs no file-system work. When enabled, every record
    is line-buffered JSON and carries a run identifier so multiple server runs
    can safely append to the same experiment artifact.
    """

    SCHEMA_VERSION = 2

    def __init__(
        self,
        path: str | Path | None,
        *,
        run_id: str | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve() if path is not None else None
        self.run_id = run_id or uuid.uuid4().hex
        self._sequence = 0
        self._stream: TextIO | None = None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("a", encoding="utf-8", buffering=1)

    @property
    def enabled(self) -> bool:
        return self._stream is not None

    def emit(self, event: str, ts_ms: float, **fields: Any) -> None:
        if self._stream is None:
            return
        if not event:
            raise ValueError("audit event must be non-empty")
        if ts_ms < 0:
            raise ValueError("audit timestamp must be non-negative")
        with self._lock:
            self._sequence += 1
            payload = {
                "schema_version": self.SCHEMA_VERSION,
                "run_id": self.run_id,
                "sequence": self._sequence,
                "pid": os.getpid(),
                "event": event,
                "ts_ms": float(ts_ms),
                **fields,
            }
            self._stream.write(
                json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
            )

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None

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
