from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, TextIO


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
