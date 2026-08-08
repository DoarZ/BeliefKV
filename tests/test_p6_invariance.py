from __future__ import annotations

import json
from pathlib import Path

from beliefkv.experiments.p6_invariance import audit_paired_load_invariance


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _cohort(
    root: Path,
    *,
    wall_clock_ms: tuple[float, ...],
    seed: int | None,
) -> Path:
    decisions = []
    calls = []
    for ordinal, elapsed in enumerate(wall_clock_ms):
        request_id = f"request-{ordinal}"
        decisions.append(
            {
                "schema_version": 2,
                "trigger_kind": "llm_submit",
                "trigger_request_id": request_id,
                "trigger_attributes": {
                    "prompt_semantic_sha256": "same-semantic-prompt",
                    "sampling_seed": seed,
                },
                "timestamp_ms": float(ordinal),
                "project": "org/project",
                "instance_id": "project__task",
                "base_commit": "abc123",
                "invocations": [
                    {
                        "invocation_id": "worker",
                        "request_id": request_id,
                    }
                ],
                "labels": [
                    {
                        "invocation_id": "worker",
                        "next_boundary_kind": "function_call",
                        "remaining_output_tokens": 128 + ordinal,
                    }
                ],
            }
        )
        calls.append(
            {
                "request_id": request_id,
                "ordinal": ordinal,
                "wall_clock_ms": elapsed,
            }
        )
    _write_jsonl(root / "frontier_decision_points.jsonl", decisions)
    _write_jsonl(root / "request_calls.jsonl", calls)
    return root


def test_paired_invariance_audit_preserves_repeated_prompt_occurrences(
    tmp_path: Path,
) -> None:
    report = audit_paired_load_invariance(
        {
            "w1": _cohort(tmp_path / "w1", wall_clock_ms=(10.0, 20.0), seed=7),
            "w8": _cohort(tmp_path / "w8", wall_clock_ms=(30.0, 80.0), seed=7),
        }
    )

    assert report["paired_semantic_key_count"] == 1
    assert report["paired_occurrence_count"] == 2
    assert report["controlled_pair_count"] == 2
    assert report["controlled_audit_eligible"] is True
    assert report["controlled_metrics"]["action_exact_agreement"] == 1.0
    assert (
        report["controlled_metrics"]["remaining_decode_relative_range_p95"]
        == 0.0
    )
    assert (
        report["controlled_metrics"]["request_wall_clock_relative_range_p50"]
        > 0.0
    )


def test_unseeded_pairs_are_diagnostic_only(tmp_path: Path) -> None:
    report = audit_paired_load_invariance(
        {
            "w1": _cohort(tmp_path / "w1", wall_clock_ms=(10.0,), seed=None),
            "w4": _cohort(tmp_path / "w4", wall_clock_ms=(20.0,), seed=None),
        }
    )

    assert report["paired_occurrence_count"] == 1
    assert report["controlled_pair_count"] == 0
    assert report["controlled_audit_eligible"] is False
    assert report["controlled_metrics"]["action_exact_agreement"] is None
    assert report["diagnostic_metrics"]["action_exact_agreement"] == 1.0
