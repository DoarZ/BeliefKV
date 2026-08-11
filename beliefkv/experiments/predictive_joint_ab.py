from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from beliefkv.metrics.summary import percentile


@dataclass(frozen=True)
class PredictiveJointABRun:
    pair_id: str
    arm: str
    summary: Mapping[str, object]
    audit_records: tuple[Mapping[str, object], ...] = ()
    gpu_utilization: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.pair_id or self.arm not in {"A", "B"}:
            raise ValueError("paired run requires a pair ID and arm A/B")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_ab_run_plan(baseline_manifest: Mapping[str, object]) -> dict[str, object]:
    if baseline_manifest.get("frozen") is not True:
        raise ValueError("R5 requires a frozen R0 baseline manifest")
    selection = baseline_manifest.get("selection_contract")
    if not isinstance(selection, Mapping):
        raise ValueError("baseline manifest omits selection contract")
    orders = tuple(str(item) for item in selection.get("paired_order", ()))
    if orders != ("A-B", "B-A", "A-B"):
        raise ValueError("R5 order must be the frozen A-B/B-A/A-B sequence")

    arm_contracts = {
        "A": {
            "name": "p5_observed_jointplan",
            "predictive_prepare": False,
            "frontier_retraction_shadow": False,
            "frontier_retraction_canary_limit": 0,
            "server_flags": [
                "--enable-observed-admission",
                "--enable-online-joint",
                "--enable-running-retraction",
            ],
        },
        "B": {
            "name": "predictive_jointplan",
            "predictive_prepare": True,
            "predictive_prepare_limit": 1,
            "frontier_retraction_shadow": True,
            "frontier_retraction_canary_limit": 1,
            "server_flags": [
                "--enable-observed-admission",
                "--enable-online-joint",
                "--enable-running-retraction",
                "--enable-predictive-risk-shadow",
                "--enable-predictive-joint-overlay",
                "--enable-frontier-retraction-shadow",
                "--predictive-prepare-canary-limit=1",
                "--frontier-retraction-canary-limit=1",
            ],
        },
    }
    runs = []
    for index, order in enumerate(orders, start=1):
        for sequence, arm in enumerate(order.split("-"), start=1):
            runs.append(
                {
                    "run_id": f"pair-{index}-{sequence}-{arm.lower()}",
                    "pair_id": f"pair-{index}",
                    "sequence": sequence,
                    "arm": arm,
                    "arm_contract": arm_contracts[arm],
                }
            )
    return {
        "schema_version": 1,
        "frozen": True,
        "baseline_manifest_sha256": baseline_manifest.get("manifest_sha256"),
        "source_tree": baseline_manifest.get("source_tree"),
        "workload": baseline_manifest.get("workload"),
        "runtime": baseline_manifest.get("runtime"),
        "artifacts": baseline_manifest.get("artifacts"),
        "arm_contracts": arm_contracts,
        "runs": runs,
        "stopping_rule": (
            "run each listed item once; do not repeat or select traces based on action"
        ),
    }


def write_immutable_ab_run_plan(
    baseline_path: Path,
    output_path: Path,
) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen A/B plan: {output_path}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    payload = build_ab_run_plan(baseline)
    payload["baseline_manifest_path"] = str(baseline_path.resolve())
    payload["baseline_file_sha256"] = _sha256(baseline_path)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encoded, encoding="utf-8")
    return payload


def summarize_ab_run(run: PredictiveJointABRun) -> dict[str, object]:
    summary = run.summary
    duration_s = float(summary.get("duration_seconds") or 0.0)
    clean_workflows = int(summary.get("system_jct_eligible_workflows") or 0)
    workflow_count = int(summary.get("workflow_count") or 0)
    event_counts = Counter(
        str(item.get("event") or "") for item in run.audit_records
    )
    runtime_events = (
        summary.get("server", {}).get("runtime_events", {}).get("event_counts", {})
        if isinstance(summary.get("server"), Mapping)
        else {}
    )
    predictive_outcomes = Counter(
        str(item.get("state") or "unknown")
        for item in run.audit_records
        if item.get("event") == "predictive_action_outcome"
        and str(item.get("state") or "")
        in {"useful", "wasted", "too_late", "censored", "failed"}
    )
    admission_wait = (
        summary.get("server", {}).get("runtime_audit", {}).get(
            "admission_queue_wait_ms", {}
        )
        if isinstance(summary.get("server"), Mapping)
        else {}
    )
    gpu_busy = (
        sum(value > 5.0 for value in run.gpu_utilization)
        / len(run.gpu_utilization)
        if run.gpu_utilization
        else None
    )
    return {
        "pair_id": run.pair_id,
        "arm": run.arm,
        "duration_seconds": duration_s,
        "workflow_count": workflow_count,
        "clean_completed_workflows": clean_workflows,
        "successful_workflows_per_hour": (
            clean_workflows * 3600.0 / duration_s if duration_s > 0 else 0.0
        ),
        "task_success_count": int(summary.get("successful_workflows") or 0),
        "runtime_failure_count": int(
            summary.get("runtime_control_delivery", {}).get("failure_count", 0)
            if isinstance(summary.get("runtime_control_delivery"), Mapping)
            else 0
        ),
        "workflow_end_count": int(runtime_events.get("workflow_end", 0)),
        "tool_starts_per_hour": (
            int(runtime_events.get("tool_start", 0)) * 3600.0 / duration_s
            if duration_s > 0
            else 0.0
        ),
        "join_completions_per_hour": (
            int(runtime_events.get("join_satisfied", 0)) * 3600.0 / duration_s
            if duration_s > 0
            else 0.0
        ),
        "gpu_busy_fraction": gpu_busy,
        "peak_running_requests": int(
            summary.get("server", {}).get("server_log", {}).get(
                "peak_running_requests", 0
            )
            if isinstance(summary.get("server"), Mapping)
            else 0
        ),
        "prefill_batch_count": int(
            summary.get("server", {}).get("server_log", {}).get(
                "prefill_batch_count", 0
            )
            if isinstance(summary.get("server"), Mapping)
            else 0
        ),
        "admission_wait_p50_ms": float(admission_wait.get("p50", 0.0)),
        "admission_wait_p95_ms": float(admission_wait.get("p95", 0.0)),
        "predictive_action_outcomes": dict(sorted(predictive_outcomes.items())),
        "frontier_retraction_canary_count": event_counts[
            "frontier_retraction_canary_bound"
        ],
        "retraction_completed_count": event_counts[
            "running_retraction_transaction_completed"
        ],
        "retraction_failed_count": event_counts[
            "running_retraction_transaction_failed"
        ],
        "restore_failed_count": sum(
            1
            for item in run.audit_records
            if item.get("event") == "restore_obligation_terminal"
            and item.get("state") != "satisfied"
        ),
        "stale_intent_count": event_counts["predictive_semantic_intent_rejected"],
        "predictive_planning_p95_ms": percentile(
            (
                float(item.get("compute_ms") or 0.0)
                for item in run.audit_records
                if item.get("event") == "predictive_risk_shadow"
            ),
            95,
        ),
    }


def compare_predictive_joint_ab(
    runs: Iterable[PredictiveJointABRun],
) -> dict[str, object]:
    summarized = [summarize_ab_run(run) for run in runs]
    by_pair: dict[str, dict[str, Mapping[str, object]]] = {}
    for item in summarized:
        pair = by_pair.setdefault(str(item["pair_id"]), {})
        arm = str(item["arm"])
        if arm in pair:
            raise ValueError(f"duplicate arm {arm} for {item['pair_id']}")
        pair[arm] = item
    if len(by_pair) != 3 or any(set(pair) != {"A", "B"} for pair in by_pair.values()):
        raise ValueError("R5 requires exactly three complete A/B pairs")

    pair_rows = []
    for pair_id, pair in sorted(by_pair.items()):
        baseline = pair["A"]
        treatment = pair["B"]
        baseline_rate = float(baseline["successful_workflows_per_hour"])
        treatment_rate = float(treatment["successful_workflows_per_hour"])
        pair_rows.append(
            {
                "pair_id": pair_id,
                "A": baseline,
                "B": treatment,
                "throughput_ratio": (
                    treatment_rate / baseline_rate if baseline_rate > 0 else None
                ),
                "throughput_improved": treatment_rate > baseline_rate,
                "task_success_not_lower": int(treatment["task_success_count"])
                >= int(baseline["task_success_count"]),
                "liveness_clean": (
                    int(treatment["runtime_failure_count"]) == 0
                    and int(treatment["retraction_failed_count"]) == 0
                    and int(treatment["restore_failed_count"]) == 0
                ),
            }
        )
    improved = sum(bool(item["throughput_improved"]) for item in pair_rows)
    total_task_a = sum(int(item["A"]["task_success_count"]) for item in pair_rows)
    total_task_b = sum(int(item["B"]["task_success_count"]) for item in pair_rows)
    clean = all(bool(item["liveness_clean"]) for item in pair_rows)
    return {
        "schema_version": 1,
        "pair_count": len(pair_rows),
        "pairs_with_throughput_improvement": improved,
        "aggregate_task_success_A": total_task_a,
        "aggregate_task_success_B": total_task_b,
        "aggregate_task_success_not_lower": total_task_b >= total_task_a,
        "no_new_liveness_failure": clean,
        "development_continue_gate": (
            improved >= 2 and total_task_b >= total_task_a and clean
        ),
        "pairs": pair_rows,
    }
