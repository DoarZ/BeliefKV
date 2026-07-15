from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PCIeCostModel:
    """Fixed-overhead plus bandwidth model using decimal GB/s."""

    bandwidth_gbps: float = 24.0
    overhead_ms: float = 0.08

    def __post_init__(self) -> None:
        if self.bandwidth_gbps <= 0:
            raise ValueError("bandwidth_gbps must be positive")
        if self.overhead_ms < 0:
            raise ValueError("overhead_ms must be non-negative")

    def transfer_ms(self, size_bytes: int) -> float:
        if size_bytes <= 0:
            return 0.0
        bytes_per_ms = self.bandwidth_gbps * 1_000_000.0
        return self.overhead_ms + size_bytes / bytes_per_ms
