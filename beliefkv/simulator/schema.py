from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class SimulationEventKind(str, Enum):
    RUNTIME = "runtime"
    CACHE_INSERT = "cache_insert"
    CACHE_BIND = "cache_bind"
    PAGE_FREE = "page_free"
    LOCK_CHANGE = "lock_change"
    READER_CHANGE = "reader_change"
    REQUEST_SUBMIT = "request_submit"
    ADMISSION_ACK = "admission_ack"
    SERVICE_CHARGE = "service_charge"
    SIGNAL = "signal"
    TICK = "tick"


@dataclass(frozen=True)
class SimulationEvent:
    ts_ms: float
    kind: SimulationEventKind
    payload: Mapping[str, Any] = field(default_factory=dict)
    sequence: int = 0

    def __post_init__(self) -> None:
        if self.ts_ms < 0:
            raise ValueError("simulation event timestamp must be non-negative")
        if self.sequence < 0:
            raise ValueError("simulation event sequence must be non-negative")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, sequence: int = 0) -> "SimulationEvent":
        if "kind" not in raw or "ts_ms" not in raw:
            raise ValueError("simulation event requires kind and ts_ms")
        payload = dict(raw.get("payload", {}))
        payload.update(
            {
                key: value
                for key, value in raw.items()
                if key not in {"kind", "ts_ms", "payload", "sequence"}
            }
        )
        return cls(
            ts_ms=float(raw["ts_ms"]),
            kind=SimulationEventKind(raw["kind"]),
            payload=payload,
            sequence=int(raw.get("sequence", sequence)),
        )


@dataclass(frozen=True)
class SimulationScenario:
    name: str
    events: tuple[SimulationEvent, ...]
    seed: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SimulationScenario":
        events = tuple(
            SimulationEvent.from_dict(item, sequence=index)
            for index, item in enumerate(raw.get("events", []))
        )
        if not events:
            raise ValueError("simulation scenario must contain at least one event")
        return cls(
            name=str(raw.get("name", "unnamed")),
            events=events,
            seed=int(raw.get("seed", 0)),
            metadata=dict(raw.get("metadata", {})),
        )
