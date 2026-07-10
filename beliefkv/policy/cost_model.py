from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PCIeCostModel:
    """Simple transfer model used by the planner.

    bandwidth_gbps uses decimal GB/s, matching common PCIe bandwidth reporting.
    """

    bandwidth_gbps: float = 24.0
    overhead_ms: float = 0.08

    def transfer_ms(self, size_bytes: int) -> float:
        if size_bytes <= 0:
            return 0.0
        bytes_per_ms = self.bandwidth_gbps * 1_000_000_000 / 1000.0
        return self.overhead_ms + size_bytes / bytes_per_ms


def estimate_kv_bytes_per_token(
    *,
    num_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    dtype_bytes: int = 2,
) -> int:
    if num_attention_heads <= 0:
        raise ValueError("num_attention_heads must be positive")
    head_dim = hidden_size // num_attention_heads
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def estimate_recompute_ms(token_count: int, prefill_tokens_per_ms: float) -> float:
    if token_count <= 0:
        return 0.0
    if prefill_tokens_per_ms <= 0:
        raise ValueError("prefill_tokens_per_ms must be positive")
    return token_count / prefill_tokens_per_ms
