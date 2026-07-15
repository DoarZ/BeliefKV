from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LLMServiceSample:
    model: str
    prompt_tokens: int
    cache_hit_tokens: int
    output_tokens: int
    batch_size: int
    context_tokens: int
    queue_ms: float
    prefill_ms: float
    decode_ms: float


@dataclass(frozen=True)
class LLMServiceEstimate:
    queue_ms: float
    prefill_ms: float
    decode_ms: float

    @property
    def total_ms(self) -> float:
        return self.queue_ms + self.prefill_ms + self.decode_ms


@dataclass
class _BucketProfile:
    queue_ms: float
    prefill_tokens_per_ms: float
    decode_ms_per_token: float
    observations: int = 0


class LLMServiceCostModel:
    """Online bucketed service model that separates queue/prefill/decode."""

    def __init__(
        self,
        *,
        default_prefill_tokens_per_ms: float = 80.0,
        default_decode_ms_per_token: float = 10.0,
        ewma_alpha: float = 0.2,
    ) -> None:
        if default_prefill_tokens_per_ms <= 0:
            raise ValueError("default prefill rate must be positive")
        if default_decode_ms_per_token < 0:
            raise ValueError("default decode cost must be non-negative")
        if not 0 < ewma_alpha <= 1:
            raise ValueError("ewma_alpha must be in (0, 1]")
        self.default_prefill_tokens_per_ms = default_prefill_tokens_per_ms
        self.default_decode_ms_per_token = default_decode_ms_per_token
        self.ewma_alpha = ewma_alpha
        self._profiles: dict[tuple[str, int, int], _BucketProfile] = {}

    def observe(self, sample: LLMServiceSample) -> None:
        if min(
            sample.prompt_tokens,
            sample.cache_hit_tokens,
            sample.output_tokens,
            sample.batch_size,
            sample.context_tokens,
        ) < 0:
            raise ValueError("token and batch fields must be non-negative")
        key = self._bucket_key(sample.model, sample.batch_size, sample.context_tokens)
        uncached = max(0, sample.prompt_tokens - sample.cache_hit_tokens)
        prefill_rate = (
            uncached / sample.prefill_ms
            if uncached > 0 and sample.prefill_ms > 0
            else self.default_prefill_tokens_per_ms
        )
        decode_rate = (
            sample.decode_ms / sample.output_tokens
            if sample.output_tokens > 0
            else self.default_decode_ms_per_token
        )
        profile = self._profiles.get(key)
        if profile is None:
            self._profiles[key] = _BucketProfile(
                queue_ms=max(0.0, sample.queue_ms),
                prefill_tokens_per_ms=max(1e-6, prefill_rate),
                decode_ms_per_token=max(0.0, decode_rate),
                observations=1,
            )
            return
        alpha = self.ewma_alpha
        profile.queue_ms = (1 - alpha) * profile.queue_ms + alpha * max(
            0.0, sample.queue_ms
        )
        profile.prefill_tokens_per_ms = (
            (1 - alpha) * profile.prefill_tokens_per_ms
            + alpha * max(1e-6, prefill_rate)
        )
        profile.decode_ms_per_token = (
            (1 - alpha) * profile.decode_ms_per_token
            + alpha * max(0.0, decode_rate)
        )
        profile.observations += 1

    def estimate(
        self,
        *,
        model: str,
        prompt_tokens: int,
        cache_hit_tokens: int,
        expected_output_tokens: int,
        batch_size: int,
        context_tokens: int,
    ) -> LLMServiceEstimate:
        key = self._bucket_key(model, batch_size, context_tokens)
        profile = self._profiles.get(key)
        if profile is None:
            profile = self._nearest_profile(model, batch_size, context_tokens)
        if profile is None:
            profile = _BucketProfile(
                queue_ms=0.0,
                prefill_tokens_per_ms=self.default_prefill_tokens_per_ms,
                decode_ms_per_token=self.default_decode_ms_per_token,
            )
        uncached = max(0, prompt_tokens - cache_hit_tokens)
        return LLMServiceEstimate(
            queue_ms=profile.queue_ms,
            prefill_ms=uncached / max(1e-6, profile.prefill_tokens_per_ms),
            decode_ms=max(0, expected_output_tokens) * profile.decode_ms_per_token,
        )

    @staticmethod
    def _bucket_key(model: str, batch_size: int, context_tokens: int) -> tuple[str, int, int]:
        batch_bucket = 1
        while batch_bucket < max(1, batch_size):
            batch_bucket *= 2
        context_bucket = 1
        while context_bucket < max(1, context_tokens):
            context_bucket *= 2
        return model, batch_bucket, context_bucket

    def _nearest_profile(
        self, model: str, batch_size: int, context_tokens: int
    ) -> _BucketProfile | None:
        target = self._bucket_key(model, batch_size, context_tokens)
        candidates = [
            (abs(key[1] - target[1]) + abs(key[2] - target[2]), profile)
            for key, profile in self._profiles.items()
            if key[0] == model
        ]
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_prefill_tokens_per_ms": self.default_prefill_tokens_per_ms,
            "default_decode_ms_per_token": self.default_decode_ms_per_token,
            "ewma_alpha": self.ewma_alpha,
            "profiles": [
                {
                    "model": key[0],
                    "batch_bucket": key[1],
                    "context_bucket": key[2],
                    "queue_ms": profile.queue_ms,
                    "prefill_tokens_per_ms": profile.prefill_tokens_per_ms,
                    "decode_ms_per_token": profile.decode_ms_per_token,
                    "observations": profile.observations,
                }
                for key, profile in sorted(self._profiles.items())
            ],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LLMServiceCostModel":
        model = cls(
            default_prefill_tokens_per_ms=float(
                raw.get("default_prefill_tokens_per_ms", 80.0)
            ),
            default_decode_ms_per_token=float(
                raw.get("default_decode_ms_per_token", 10.0)
            ),
            ewma_alpha=float(raw.get("ewma_alpha", 0.2)),
        )
        for item in raw.get("profiles", []):
            key = (
                str(item["model"]),
                int(item["batch_bucket"]),
                int(item["context_bucket"]),
            )
            model._profiles[key] = _BucketProfile(
                queue_ms=float(item["queue_ms"]),
                prefill_tokens_per_ms=float(item["prefill_tokens_per_ms"]),
                decode_ms_per_token=float(item["decode_ms_per_token"]),
                observations=int(item.get("observations", 0)),
            )
        return model
