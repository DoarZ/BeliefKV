from __future__ import annotations

import json
from dataclasses import replace

import pytest

from beliefkv.experiments.service_calibration import (
    QueueServiceCalibrator,
    QueueServiceSample,
    load_calibrated_service_model,
)
from beliefkv.simulator.queue_service import QueueServiceModel


def _sample(
    sample_id: str,
    phase: str,
    tokens: int,
    elapsed_ms: float,
    split: str,
    *,
    batch_size: int = 1,
) -> QueueServiceSample:
    return QueueServiceSample(
        phase=phase,
        tokens=tokens,
        batch_size=batch_size,
        elapsed_ms=elapsed_ms,
        split=split,
        sample_id=sample_id,
    )


def _samples() -> tuple[QueueServiceSample, ...]:
    return (
        _sample("p1", "prefill", 100, 10, "train"),
        _sample("p2", "prefill", 200, 20, "train"),
        _sample("p3", "prefill", 300, 30, "train"),
        _sample("d1", "decode", 10, 2, "train"),
        _sample("d2", "decode", 20, 4, "train"),
        _sample("d3", "decode", 20, 2, "train", batch_size=2),
        _sample("ph1", "prefill", 150, 15, "holdout"),
        _sample("ph2", "prefill", 250, 25, "holdout"),
        _sample("dh1", "decode", 15, 3, "holdout"),
        _sample("dh2", "decode", 30, 3, "holdout", batch_size=2),
    )


def test_calibrator_requires_explicit_holdout_and_emits_auditable_model() -> None:
    result = QueueServiceCalibrator().fit(
        _samples(), calibration_source="gpu0-held-out-microbenchmark"
    )

    assert result.model.calibrated
    assert result.model.prefill_tokens_per_ms == pytest.approx(10)
    assert result.model.prefill_launch_ms == pytest.approx(0)
    assert result.model.decode_tokens_per_ms == pytest.approx(5)
    assert result.model.decode_batch_efficiency == pytest.approx((1, 2))
    assert result.holdout_relative_error_p95 == pytest.approx(0)
    assert result.rejection_reasons == ()
    assert (
        result.to_dict()["calibration_algorithm"]
        == "episode_piecewise_isotonic_v1"
    )
    assert (
        result.to_dict()["sample_timing_semantics"]
        == "gpu_service_interval_v1"
    )
    assert result.train_episode_count == 6
    assert result.holdout_episode_count == 4
    assert QueueServiceModel.from_dict(result.model.to_dict()) == result.model


def test_calibrator_fails_closed_on_holdout_error() -> None:
    samples = list(_samples())
    samples[-1] = replace(samples[-1], elapsed_ms=30)

    result = QueueServiceCalibrator().fit(
        samples, calibration_source="bad-held-out-run"
    )

    assert not result.model.calibrated
    assert "holdout_relative_error_p95_exceeds_gate" in result.rejection_reasons


def test_calibrator_fails_closed_without_phase_coverage() -> None:
    result = QueueServiceCalibrator().fit(
        tuple(item for item in _samples() if item.phase == "prefill"),
        calibration_source="prefill-only",
    )

    assert not result.model.calibrated
    assert "missing_single_request_decode_rate" in result.rejection_reasons
    assert "insufficient_train_decode" in result.rejection_reasons


def test_calibrator_can_require_independent_multichunk_prefill_coverage() -> None:
    result = QueueServiceCalibrator(require_multichunk_prefill=True).fit(
        _samples(), calibration_source="single-chunk-only"
    )

    assert not result.model.calibrated
    assert "missing_train_multichunk_prefill" in result.rejection_reasons
    assert "missing_holdout_multichunk_prefill" in result.rejection_reasons


def test_calibrator_fits_prefill_launch_cost_and_validates_episode_totals() -> None:
    samples = (
        _sample("p1", "prefill", 100, 12, "train"),
        _sample("p2", "prefill", 200, 22, "train"),
        _sample("p3", "prefill", 300, 32, "train"),
        _sample("d1", "decode", 1, 5, "train", batch_size=1),
        _sample("d2", "decode", 1, 5, "train", batch_size=1),
        _sample("d3", "decode", 2, 5, "train", batch_size=2),
        _sample("ph1", "prefill", 150, 17, "holdout"),
        _sample("ph2", "prefill", 250, 27, "holdout"),
        QueueServiceSample(
            phase="decode",
            tokens=1,
            batch_size=1,
            elapsed_ms=4,
            split="holdout",
            sample_id="dh1-1",
            episode_id="dh1",
        ),
        QueueServiceSample(
            phase="decode",
            tokens=1,
            batch_size=1,
            elapsed_ms=6,
            split="holdout",
            sample_id="dh1-2",
            episode_id="dh1",
        ),
        QueueServiceSample(
            phase="decode",
            tokens=2,
            batch_size=2,
            elapsed_ms=4,
            split="holdout",
            sample_id="dh2-1",
            episode_id="dh2",
        ),
        QueueServiceSample(
            phase="decode",
            tokens=2,
            batch_size=2,
            elapsed_ms=6,
            split="holdout",
            sample_id="dh2-2",
            episode_id="dh2",
        ),
    )

    result = QueueServiceCalibrator().fit(
        samples, calibration_source="affine-and-episode-test"
    )

    assert result.model.calibrated
    assert result.model.prefill_tokens_per_ms == pytest.approx(10)
    assert result.model.prefill_launch_ms == pytest.approx(2)
    assert result.holdout_episode_count == 4
    assert result.holdout_phase_relative_error_p95["prefill"] == pytest.approx(0)
    assert result.holdout_phase_relative_error_p95["decode"] == pytest.approx(0)


def test_prefill_service_curve_is_monotonic_under_noisy_samples() -> None:
    samples = list(_samples())
    samples[0] = replace(samples[0], elapsed_ms=20)
    samples[1] = replace(samples[1], elapsed_ms=10)

    result = QueueServiceCalibrator().fit(
        samples, calibration_source="noisy-prefill-curve"
    )
    curve = result.model.prefill_first_chunk_curve

    assert all(left[1] <= right[1] for left, right in zip(curve, curve[1:]))


def test_calibrated_model_loader_rejects_failed_artifact(tmp_path) -> None:
    result = QueueServiceCalibrator().fit(
        _samples(), calibration_source="loader-round-trip"
    )
    path = tmp_path / "service-model.json"
    path.write_text(json.dumps(result.to_dict()), encoding="utf-8")

    assert load_calibrated_service_model(path) == result.model

    failed = result.to_dict()
    failed["rejection_reasons"] = ["forced-test-failure"]
    path.write_text(json.dumps(failed), encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        load_calibrated_service_model(path)
