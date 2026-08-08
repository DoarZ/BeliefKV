from __future__ import annotations

import pytest

from beliefkv.predictor.hardware_service import (
    GPURequestServiceDemand,
    GPUServiceCurveModel,
    GPUServiceFeatures,
)


def _rows(
    split: str = "train", *, evidence_role: str = "controlled_microbenchmark"
) -> list[dict[str, object]]:
    return [
        {
            "row_type": "gpu_batch_service_interval",
            "split": split,
            "sample_id": f"sample-{sample}",
            "case_ids": ["decode-profile"],
            "phase": "decode",
            "batch_size": 2,
            "request_samples": [
                {
                    "request_id": f"request-{sample}-a",
                    "sequence_tokens_before": 4096,
                    "token_delta": 1,
                    "cache_hit_ratio": 0.0,
                },
                {
                    "request_id": f"request-{sample}-b",
                    "sequence_tokens_before": 8192,
                    "token_delta": 1,
                    "cache_hit_ratio": 0.0,
                },
            ],
            "chunk_position": "first",
            "prefill_decode_mixed": False,
            "pcie_contention_state": "idle",
            "hicache_inflight_bytes": 0,
            "service_elapsed_ms": value,
            "warmup": False,
            "evidence_role": evidence_role,
        }
        for sample, value in enumerate((4.0, 6.0, 8.0, 10.0))
    ]


def _features() -> GPUServiceFeatures:
    return GPUServiceFeatures(
        phase="decode",
        request_demands=(
            GPURequestServiceDemand(4096, 1),
            GPURequestServiceDemand(8192, 1),
        ),
        chunk_position="first",
        pcie_contention_state="idle",
    )


def test_gpu_service_curve_uses_unique_complete_batches(tmp_path) -> None:
    model = GPUServiceCurveModel(minimum_support=2.0)
    summary = model.fit(_rows())
    assert summary["batch_sample_count"] == 4
    assert summary["unique_sample_count"] == 4
    estimate = model.predict(_features())
    assert estimate.support == 4.0
    assert estimate.p95_ms >= estimate.p90_ms >= estimate.p50_ms > 0

    path = tmp_path / "gpu-service.json"
    model.save(path)
    assert GPUServiceCurveModel.load(path).predict(_features()) == estimate


def test_gpu_service_curve_rejects_holdout_during_fit() -> None:
    with pytest.raises(ValueError, match="only train"):
        GPUServiceCurveModel().fit(_rows("holdout"))


def test_gpu_service_curve_rejects_runtime_overlap_as_training() -> None:
    with pytest.raises(ValueError, match="controlled microbenchmarks only"):
        GPUServiceCurveModel().fit(
            _rows(evidence_role="runtime_validation")
        )


def test_gpu_service_curve_rejects_request_expanded_or_duplicate_batches() -> None:
    request_row = dict(_rows()[0], row_type="gpu_service_interval")
    with pytest.raises(ValueError, match="one row per complete batch"):
        GPUServiceCurveModel(minimum_support=1).fit([request_row])

    duplicate = _rows()[:2]
    duplicate[1]["sample_id"] = duplicate[0]["sample_id"]
    with pytest.raises(ValueError, match="unique batch sample_id"):
        GPUServiceCurveModel().fit(duplicate)


def test_runtime_rows_are_validation_only() -> None:
    model = GPUServiceCurveModel(minimum_support=2.0)
    model.fit(_rows())
    runtime = _rows(evidence_role="runtime_validation")
    report = model.validate_runtime_rows(runtime)
    assert report["sample_count"] == 4
    assert report["model_updated"] is False


def _profile_rows(split: str = "train") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sample = 0
    for profile, (sequence_tokens, elapsed_ms) in enumerate(
        ((1024, 2.0), (4096, 4.0), (16384, 8.0), (32768, 12.0))
    ):
        for repetition in range(3):
            rows.append(
                {
                    "row_type": "gpu_batch_service_interval",
                    "split": split,
                    "sample_id": f"{split}-{sample}",
                    "case_ids": [f"decode-c{sequence_tokens}"],
                    "phase": "decode",
                    "batch_size": 1,
                    "request_samples": [
                        {
                            "request_id": f"request-{sample}",
                            "workflow_id": f"profile-{profile}",
                            "sequence_tokens_before": sequence_tokens,
                            "token_delta": 1,
                            "cache_hit_ratio": 0.0,
                        }
                    ],
                    "chunk_position": "continuation",
                    "prefill_decode_mixed": False,
                    "pcie_contention_state": "idle",
                    "hicache_inflight_bytes": 0,
                    "service_elapsed_ms": elapsed_ms + repetition * 0.1,
                    "warmup": False,
                    "evidence_role": "controlled_microbenchmark",
                }
            )
            sample += 1
    return rows


def test_gpu_service_curve_interpolates_sequence_neighborhood() -> None:
    model = GPUServiceCurveModel(minimum_support=2, neighbor_count=6)
    model.fit(_profile_rows())
    estimate = model.predict(
        GPUServiceFeatures(
            phase="decode",
            request_demands=(GPURequestServiceDemand(8192, 1),),
            chunk_position="continuation",
            pcie_contention_state="idle",
        )
    )
    assert estimate.source == "neighbor_interpolation"
    assert 4.0 <= estimate.p50_ms <= 8.2
    assert estimate.neighbor_count == 6
    assert estimate.nearest_distance is not None


def test_profile_grouped_cross_calibration_keeps_holdout_sealed(tmp_path) -> None:
    model = GPUServiceCurveModel(minimum_support=2, neighbor_count=6)
    summary = model.fit_cross_calibrated(
        _profile_rows(), folds=2, minimum_phase_calibration=2
    )
    assert summary["calibrated"] is True
    assert summary["calibration"]["holdout_consumed"] is False
    estimate = model.predict(
        GPUServiceFeatures(
            phase="decode",
            request_demands=(GPURequestServiceDemand(8192, 1),),
            chunk_position="continuation",
            pcie_contention_state="idle",
        )
    )
    assert estimate.calibrated is True

    holdout = _profile_rows("holdout")
    evaluation = model.evaluate_controlled_rows(holdout)
    assert evaluation["scored_count"] == len(holdout)
    assert evaluation["holdout_used_for_fit_or_calibration"] is False

    path = tmp_path / "calibrated.json"
    model.save(path)
    restored = GPUServiceCurveModel.load(path)
    assert restored.predict(
        GPUServiceFeatures(
            phase="decode",
            request_demands=(GPURequestServiceDemand(8192, 1),),
            chunk_position="continuation",
            pcie_contention_state="idle",
        )
    ) == estimate
