from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from hashlib import blake2b
from pathlib import Path
from typing import Iterable, Mapping

from beliefkv.metrics.summary import percentile
from beliefkv.simulator.queue_service import QueueServiceModel


CALIBRATION_SCHEMA_VERSION = 2
CALIBRATION_ALGORITHM = "episode_piecewise_isotonic_v1"
SAMPLE_TIMING_SEMANTICS = "gpu_service_interval_v1"


@dataclass(frozen=True)
class QueueServiceSample:
    phase: str
    tokens: int
    batch_size: int
    elapsed_ms: float
    split: str
    sample_id: str
    episode_id: str | None = None
    prefill_chunk_index: int = 0
    prefix_tokens_before: int = 0

    def __post_init__(self) -> None:
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("service sample phase must be prefill or decode")
        if self.split not in {"train", "holdout"}:
            raise ValueError("service sample split must be train or holdout")
        if self.tokens <= 0 or self.batch_size <= 0 or self.elapsed_ms <= 0:
            raise ValueError("service sample demand and elapsed time must be positive")
        if not math.isfinite(self.elapsed_ms) or not self.sample_id:
            raise ValueError("service sample must be finite and identified")
        if self.episode_id is None:
            object.__setattr__(self, "episode_id", self.sample_id)
        elif not self.episode_id:
            raise ValueError("service sample episode must be identified")
        if self.prefill_chunk_index < 0 or self.prefix_tokens_before < 0:
            raise ValueError("prefill position must be non-negative")

    @classmethod
    def from_dict(cls, raw: Mapping[str, object], *, line_number: int) -> "QueueServiceSample":
        elapsed_ms = (
            raw["service_elapsed_ms"]
            if "service_elapsed_ms" in raw
            else raw["elapsed_ms"]
        )
        return cls(
            phase=str(raw["phase"]),
            tokens=int(raw["tokens"]),
            batch_size=int(raw.get("batch_size", 1)),
            elapsed_ms=float(elapsed_ms),
            split=str(raw["split"]),
            sample_id=str(raw.get("sample_id", f"line-{line_number}")),
            episode_id=str(
                raw.get("episode_id")
                or _infer_episode_id(raw)
                or raw.get("sample_id", f"line-{line_number}")
            ),
            prefill_chunk_index=int(raw.get("prefill_chunk_index", 0) or 0),
            prefix_tokens_before=int(raw.get("prefix_tokens_before", 0) or 0),
        )


@dataclass(frozen=True)
class QueueServiceCalibrationResult:
    model: QueueServiceModel
    train_sample_count: int
    holdout_sample_count: int
    train_episode_count: int
    holdout_episode_count: int
    sample_counts: Mapping[str, int]
    episode_counts: Mapping[str, int]
    holdout_relative_error_p50: float | None
    holdout_relative_error_p95: float | None
    holdout_absolute_error_p95_ms: float | None
    holdout_phase_relative_error_p95: Mapping[str, float | None]
    calibration_coverage: Mapping[str, object]
    max_allowed_relative_error_p95: float
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "calibration_algorithm": CALIBRATION_ALGORITHM,
            "sample_timing_semantics": SAMPLE_TIMING_SEMANTICS,
            "model": self.model.to_dict(),
            "train_sample_count": self.train_sample_count,
            "holdout_sample_count": self.holdout_sample_count,
            "train_episode_count": self.train_episode_count,
            "holdout_episode_count": self.holdout_episode_count,
            "sample_counts": dict(self.sample_counts),
            "episode_counts": dict(self.episode_counts),
            "holdout_relative_error_p50": self.holdout_relative_error_p50,
            "holdout_relative_error_p95": self.holdout_relative_error_p95,
            "holdout_absolute_error_p95_ms": self.holdout_absolute_error_p95_ms,
            "holdout_phase_relative_error_p95": dict(
                self.holdout_phase_relative_error_p95
            ),
            "calibration_coverage": dict(self.calibration_coverage),
            "max_allowed_relative_error_p95": (
                self.max_allowed_relative_error_p95
            ),
            "rejection_reasons": list(self.rejection_reasons),
        }


class QueueServiceCalibrator:
    """Fit a service model only from explicitly split microbenchmark samples."""

    def __init__(
        self,
        *,
        min_train_samples_per_phase: int = 3,
        min_holdout_samples_per_phase: int = 2,
        max_relative_error_p95: float = 0.25,
        prefill_chunk_tokens: int = 4_096,
        decode_quantum_tokens: int = 8,
        require_multichunk_prefill: bool = False,
    ) -> None:
        if min_train_samples_per_phase <= 0 or min_holdout_samples_per_phase <= 0:
            raise ValueError("service calibration sample gates must be positive")
        if not 0 < max_relative_error_p95 < 1:
            raise ValueError("service calibration error gate must be in (0, 1)")
        self.min_train_samples_per_phase = min_train_samples_per_phase
        self.min_holdout_samples_per_phase = min_holdout_samples_per_phase
        self.max_relative_error_p95 = max_relative_error_p95
        self.prefill_chunk_tokens = prefill_chunk_tokens
        self.decode_quantum_tokens = decode_quantum_tokens
        self.require_multichunk_prefill = require_multichunk_prefill

    def fit(
        self,
        samples: Iterable[QueueServiceSample],
        *,
        calibration_source: str,
    ) -> QueueServiceCalibrationResult:
        samples = tuple(samples)
        if not samples:
            raise ValueError("service calibration requires samples")
        if not calibration_source:
            raise ValueError("calibration_source must be non-empty")
        train = tuple(item for item in samples if item.split == "train")
        holdout = tuple(item for item in samples if item.split == "holdout")
        counts = Counter(f"{item.split}:{item.phase}" for item in samples)
        episodes = _group_episodes(samples)
        episode_counts = Counter(
            f"{group[0].split}:{group[0].phase}" for group in episodes.values()
        )
        rejection: list[str] = []
        for phase in ("prefill", "decode"):
            if (
                episode_counts[f"train:{phase}"]
                < self.min_train_samples_per_phase
            ):
                rejection.append(f"insufficient_train_{phase}")
            if (
                episode_counts[f"holdout:{phase}"]
                < self.min_holdout_samples_per_phase
            ):
                rejection.append(f"insufficient_holdout_{phase}")

        multichunk_prefill_episodes = Counter(
            group[0].split
            for group in episodes.values()
            if group[0].phase == "prefill" and len(group) > 1
        )
        if self.require_multichunk_prefill:
            for split in ("train", "holdout"):
                if multichunk_prefill_episodes[split] == 0:
                    rejection.append(f"missing_{split}_multichunk_prefill")

        prefill = tuple(item for item in train if item.phase == "prefill")
        decode_by_batch: dict[int, list[QueueServiceSample]] = defaultdict(list)
        for item in train:
            if item.phase == "decode":
                decode_by_batch[item.batch_size].append(item)
        if not prefill:
            rejection.append("missing_prefill_rate")
        if 1 not in decode_by_batch:
            rejection.append("missing_single_request_decode_rate")
        if prefill:
            prefill_rate, prefill_launch_ms, fit_rejection = _fit_prefill_affine(
                prefill
            )
            if fit_rejection is not None:
                rejection.append(fit_rejection)
        else:
            prefill_rate = 1e-12
            prefill_launch_ms = 0.0
        first_prefill_curve = _fit_service_curve(
            item for item in prefill if item.prefill_chunk_index == 0
        )
        continuation_prefill_curve = _fit_service_curve(
            item for item in prefill if item.prefill_chunk_index > 0
        )
        if prefill and not first_prefill_curve:
            rejection.append("missing_first_prefill_service_curve")
        if self.require_multichunk_prefill and not continuation_prefill_curve:
            rejection.append("missing_continuation_prefill_service_curve")
        base_decode_rate = (
            _aggregate_rate(decode_by_batch[1])
            if 1 in decode_by_batch
            else 1e-12
        )
        max_batch = max(decode_by_batch, default=1)
        efficiencies = []
        last_rate = base_decode_rate
        for batch_size in range(1, max_batch + 1):
            group = decode_by_batch.get(batch_size)
            if group:
                last_rate = _aggregate_rate(group)
            efficiencies.append(last_rate / base_decode_rate)

        provisional = QueueServiceModel(
            model_id="pending-calibration",
            prefill_tokens_per_ms=prefill_rate,
            decode_tokens_per_ms=base_decode_rate,
            decode_batch_efficiency=tuple(efficiencies),
            max_decode_batch=max_batch,
            prefill_chunk_tokens=self.prefill_chunk_tokens,
            decode_quantum_tokens=self.decode_quantum_tokens,
            prefill_launch_ms=prefill_launch_ms,
            prefill_first_chunk_curve=first_prefill_curve,
            prefill_continuation_chunk_curve=continuation_prefill_curve,
            calibrated=False,
            calibration_source=calibration_source,
        )
        relative_errors = []
        absolute_errors = []
        phase_relative_errors: dict[str, list[float]] = defaultdict(list)
        for episode in episodes.values():
            if episode[0].split != "holdout":
                continue
            observed = sum(item.elapsed_ms for item in episode)
            predicted = sum(
                _predict_elapsed_ms(provisional, item) for item in episode
            )
            absolute = abs(predicted - observed)
            absolute_errors.append(absolute)
            relative = absolute / observed
            relative_errors.append(relative)
            phase_relative_errors[episode[0].phase].append(relative)
        relative_p95 = (
            percentile(relative_errors, 95) if relative_errors else None
        )
        if relative_p95 is None or relative_p95 > self.max_relative_error_p95:
            rejection.append("holdout_relative_error_p95_exceeds_gate")
        phase_p95 = {
            phase: (
                percentile(phase_relative_errors[phase], 95)
                if phase_relative_errors[phase]
                else None
            )
            for phase in ("prefill", "decode")
        }
        for phase, value in phase_p95.items():
            if value is None or value > self.max_relative_error_p95:
                rejection.append(
                    f"holdout_{phase}_relative_error_p95_exceeds_gate"
                )
        rejection = sorted(set(rejection))
        payload = {
            "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
            "calibration_algorithm": CALIBRATION_ALGORITHM,
            "sample_timing_semantics": SAMPLE_TIMING_SEMANTICS,
            "calibration_source": calibration_source,
            "rates": {
                "prefill": prefill_rate,
                "prefill_launch_ms": prefill_launch_ms,
                "prefill_first_chunk_curve": first_prefill_curve,
                "prefill_continuation_chunk_curve": continuation_prefill_curve,
                "decode": base_decode_rate,
                "decode_batch_efficiency": efficiencies,
            },
            "sample_ids": sorted(item.sample_id for item in samples),
            "gate": self.max_relative_error_p95,
        }
        model_id = "queue-service-" + blake2b(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
            digest_size=12,
            person=b"bk-service",
        ).hexdigest()
        model = replace(
            provisional,
            model_id=model_id,
            calibrated=not rejection,
        )
        return QueueServiceCalibrationResult(
            model=model,
            train_sample_count=len(train),
            holdout_sample_count=len(holdout),
            train_episode_count=sum(
                group[0].split == "train" for group in episodes.values()
            ),
            holdout_episode_count=sum(
                group[0].split == "holdout" for group in episodes.values()
            ),
            sample_counts=dict(sorted(counts.items())),
            episode_counts=dict(sorted(episode_counts.items())),
            holdout_relative_error_p50=(
                percentile(relative_errors, 50) if relative_errors else None
            ),
            holdout_relative_error_p95=relative_p95,
            holdout_absolute_error_p95_ms=(
                percentile(absolute_errors, 95) if absolute_errors else None
            ),
            holdout_phase_relative_error_p95=phase_p95,
            calibration_coverage={
                "prefill_chunk_tokens": self.prefill_chunk_tokens,
                "train_multichunk_prefill_episodes": (
                    multichunk_prefill_episodes["train"]
                ),
                "holdout_multichunk_prefill_episodes": (
                    multichunk_prefill_episodes["holdout"]
                ),
                "max_train_prefill_service_tokens": max(
                    (item.tokens for item in prefill), default=0
                ),
                "train_decode_batch_sizes": sorted(decode_by_batch),
                "first_prefill_curve_points": len(first_prefill_curve),
                "continuation_prefill_curve_points": len(
                    continuation_prefill_curve
                ),
            },
            max_allowed_relative_error_p95=self.max_relative_error_p95,
            rejection_reasons=tuple(rejection),
        )


def _aggregate_rate(samples: Iterable[QueueServiceSample]) -> float:
    samples = tuple(samples)
    tokens = sum(item.tokens for item in samples)
    elapsed = sum(item.elapsed_ms for item in samples)
    if tokens <= 0 or elapsed <= 0:
        raise ValueError("cannot fit a service rate from empty demand")
    return tokens / elapsed


def _fit_prefill_affine(
    samples: Iterable[QueueServiceSample],
) -> tuple[float, float, str | None]:
    samples = tuple(samples)
    distinct_tokens = sorted({item.tokens for item in samples})
    if len(distinct_tokens) < 2:
        return _aggregate_rate(samples), 0.0, "insufficient_prefill_token_diversity"
    slopes = [
        (right.elapsed_ms - left.elapsed_ms) / (right.tokens - left.tokens)
        for index, left in enumerate(samples)
        for right in samples[index + 1 :]
        if right.tokens != left.tokens
    ]
    slope = percentile(slopes, 50)
    if not math.isfinite(slope) or slope <= 0:
        return _aggregate_rate(samples), 0.0, "non_positive_prefill_slope"
    intercept = percentile(
        [item.elapsed_ms - slope * item.tokens for item in samples], 50
    )
    if not math.isfinite(intercept):
        return _aggregate_rate(samples), 0.0, "non_finite_prefill_intercept"
    if intercept < 0:
        intercept = 0.0
        denominator = sum(item.tokens * item.tokens for item in samples)
        slope = sum(item.tokens * item.elapsed_ms for item in samples) / denominator
    if slope <= 0 or not math.isfinite(slope):
        return _aggregate_rate(samples), 0.0, "non_positive_prefill_slope"
    return 1.0 / slope, intercept, None


def _fit_service_curve(
    samples: Iterable[QueueServiceSample],
    *,
    merge_token_distance: int = 64,
) -> tuple[tuple[int, float], ...]:
    ordered = sorted(samples, key=lambda item: (item.tokens, item.sample_id))
    if not ordered:
        return ()
    groups: list[list[QueueServiceSample]] = []
    for item in ordered:
        if not groups or item.tokens - groups[-1][-1].tokens > merge_token_distance:
            groups.append([item])
        else:
            groups[-1].append(item)
    points = tuple(
        (
            int(round(percentile([item.tokens for item in group], 50))),
            percentile([item.elapsed_ms for item in group], 50),
            len(group),
        )
        for group in groups
    )
    fitted = _isotonic_non_decreasing(
        tuple(item[1] for item in points),
        tuple(item[2] for item in points),
    )
    return tuple((item[0], elapsed_ms) for item, elapsed_ms in zip(points, fitted))


def _isotonic_non_decreasing(
    values: tuple[float, ...], weights: tuple[int, ...]
) -> tuple[float, ...]:
    if len(values) != len(weights) or any(weight <= 0 for weight in weights):
        raise ValueError("isotonic values and positive weights must align")
    blocks: list[dict[str, float | int]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append(
            {
                "start": index,
                "end": index + 1,
                "weight": weight,
                "weighted_sum": value * weight,
            }
        )
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = float(left["weighted_sum"]) / int(left["weight"])
            right_mean = float(right["weighted_sum"]) / int(right["weight"])
            if left_mean <= right_mean:
                break
            blocks[-2:] = [
                {
                    "start": left["start"],
                    "end": right["end"],
                    "weight": int(left["weight"]) + int(right["weight"]),
                    "weighted_sum": float(left["weighted_sum"])
                    + float(right["weighted_sum"]),
                }
            ]
    result = [0.0] * len(values)
    for block in blocks:
        mean = float(block["weighted_sum"]) / int(block["weight"])
        for index in range(int(block["start"]), int(block["end"])):
            result[index] = mean
    return tuple(result)


def _group_episodes(
    samples: Iterable[QueueServiceSample],
) -> dict[tuple[str, str, str, int], tuple[QueueServiceSample, ...]]:
    grouped: dict[
        tuple[str, str, str, int], list[QueueServiceSample]
    ] = defaultdict(list)
    for item in samples:
        grouped[
            (item.split, item.phase, str(item.episode_id), item.batch_size)
        ].append(item)
    return {key: tuple(value) for key, value in grouped.items()}


def _infer_episode_id(raw: Mapping[str, object]) -> str | None:
    workflow_ids = raw.get("workflow_ids")
    if not isinstance(workflow_ids, list) or not workflow_ids:
        return None
    parsed = []
    for value in workflow_ids:
        workflow_id = str(value)
        prefix = "service-calibration:"
        if not workflow_id.startswith(prefix):
            return None
        suffix = workflow_id[len(prefix) :]
        split, separator, case_id = suffix.partition(":")
        if not separator or split not in {"train", "holdout"}:
            return None
        if case_id.startswith("decode-"):
            episode, separator, request_index = case_id.rpartition("-i")
            if not separator or not request_index.isdigit():
                return None
        else:
            episode = case_id
        parsed.append(f"{split}:{episode}")
    unique = set(parsed)
    return next(iter(unique)) if len(unique) == 1 else None


def _infer_calibration_kind(raw: Mapping[str, object]) -> str | None:
    workflow_ids = raw.get("workflow_ids")
    if not isinstance(workflow_ids, list) or not workflow_ids:
        return None
    kinds = set()
    for value in workflow_ids:
        workflow_id = str(value)
        if ":prefill-" in workflow_id:
            kinds.add("prefill")
        elif ":decode-" in workflow_id:
            kinds.add("decode")
        else:
            return None
    return next(iter(kinds)) if len(kinds) == 1 else None


def _predict_elapsed_ms(
    model: QueueServiceModel, sample: QueueServiceSample
) -> float:
    if sample.phase == "prefill":
        return model.prefill_elapsed_ms(
            sample.tokens,
            chunk_index=sample.prefill_chunk_index,
        )
    return model.decode_launch_ms + sample.tokens / model.decode_rate(
        sample.batch_size
    )


def read_service_samples(
    path: Path, *, min_prefill_tokens: int = 1
) -> tuple[QueueServiceSample, ...]:
    if min_prefill_tokens <= 0:
        raise ValueError("min_prefill_tokens must be positive")
    result = []
    prefill_chunk_count: Counter[str] = Counter()
    prefill_token_count: Counter[str] = Counter()
    with path.expanduser().resolve().open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"service sample line {line_number} is not an object")
            if "event" in raw and raw.get("event") != "gpu_service_sample":
                continue
            calibration_kind = raw.get("calibration_kind") or _infer_calibration_kind(
                raw
            )
            if calibration_kind is not None and raw.get("phase") != calibration_kind:
                continue
            normalized = dict(raw)
            if normalized.get("phase") == "prefill":
                episode_id = str(
                    normalized.get("episode_id")
                    or _infer_episode_id(normalized)
                    or normalized.get("sample_id", f"line-{line_number}")
                )
                normalized.setdefault(
                    "prefill_chunk_index", prefill_chunk_count[episode_id]
                )
                normalized.setdefault(
                    "prefix_tokens_before", prefill_token_count[episode_id]
                )
                prefill_chunk_count[episode_id] += 1
                prefill_token_count[episode_id] += int(normalized["tokens"])
            sample = QueueServiceSample.from_dict(
                normalized, line_number=line_number
            )
            if sample.phase == "prefill" and sample.tokens < min_prefill_tokens:
                continue
            result.append(sample)
    return tuple(result)


def load_calibrated_service_model(path: Path) -> QueueServiceModel:
    source = path.expanduser().resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("queue service calibration artifact must be an object")
    if int(raw.get("schema_version", 0)) != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported queue service calibration schema")
    if raw.get("calibration_algorithm") != CALIBRATION_ALGORITHM:
        raise ValueError("unsupported queue service calibration algorithm")
    if raw.get("sample_timing_semantics") != SAMPLE_TIMING_SEMANTICS:
        raise ValueError("unsupported queue service timing semantics")
    if raw.get("rejection_reasons"):
        raise ValueError("queue service calibration did not pass its holdout gate")
    model_raw = raw.get("model")
    if not isinstance(model_raw, Mapping):
        raise ValueError("queue service calibration lacks a model object")
    model = QueueServiceModel.from_dict(model_raw)
    if not model.calibrated:
        raise ValueError("queue service model is not calibrated")
    coverage = raw.get("calibration_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("queue service calibration lacks coverage evidence")
    if int(coverage.get("prefill_chunk_tokens", 0)) != model.prefill_chunk_tokens:
        raise ValueError("prefill chunk coverage disagrees with the service model")
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and holdout-check a BeliefKV queue service model."
    )
    parser.add_argument("samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--max-relative-error-p95", type=float, default=0.25)
    parser.add_argument("--min-prefill-tokens", type=int, default=256)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=4096)
    parser.add_argument("--decode-quantum-tokens", type=int, default=8)
    parser.add_argument("--require-multichunk-prefill", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = QueueServiceCalibrator(
        max_relative_error_p95=args.max_relative_error_p95,
        prefill_chunk_tokens=args.prefill_chunk_tokens,
        decode_quantum_tokens=args.decode_quantum_tokens,
        require_multichunk_prefill=args.require_multichunk_prefill,
    ).fit(
        read_service_samples(
            args.samples, min_prefill_tokens=args.min_prefill_tokens
        ),
        calibration_source=args.source,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.model.calibrated else 2


if __name__ == "__main__":
    raise SystemExit(main())
