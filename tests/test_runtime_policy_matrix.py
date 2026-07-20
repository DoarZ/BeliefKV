import json
import tempfile
import unittest
from pathlib import Path

from beliefkv.experiments.runtime_policy_matrix import (
    MatrixSpec,
    aggregate_rows,
    build_conditions,
    decode_json_document,
    decode_qwen_records,
    evaluate_markers,
    summarize_qwen_records,
)
from scripts.run_runtime_prompt_structure_matrix import prometheus_gauge_sum


class RuntimePolicyMatrixTest(unittest.TestCase):
    def _spec(self, root: Path) -> MatrixSpec:
        raw = {
            "schema_version": 1,
            "name": "test",
            "model": "model",
            "repository": str(root),
            "runtimes": ["qwen_code", "codex"],
            "policies": {
                "natural": "",
                "policy_guided": "Delegate independent branches dynamically.",
            },
            "common_suffix": "Read only.",
            "tasks": [
                {
                    "id": "seq",
                    "block": "one",
                    "structure": "sequential",
                    "prompt": "Trace A then B.",
                    "required_marker_groups": [["A"], ["B"]],
                },
                {
                    "id": "par",
                    "block": "one",
                    "structure": "parallelizable",
                    "prompt": "Audit A and B independently.",
                    "required_marker_groups": [["A"], ["B"]],
                },
            ],
        }
        return MatrixSpec.from_dict(raw, base_directory=root)

    def test_conditions_are_balanced_and_runtime_prompts_are_paired(self):
        with tempfile.TemporaryDirectory() as temporary:
            conditions = build_conditions(self._spec(Path(temporary)))
        self.assertEqual(len(conditions), 8)
        prompts = {}
        for condition in conditions:
            key = (condition.task.task_id, condition.policy)
            prompts.setdefault(key, set()).add(condition.prompt)
        self.assertTrue(all(len(values) == 1 for values in prompts.values()))
        guided = [item for item in conditions if item.policy == "policy_guided"]
        natural = [item for item in conditions if item.policy == "natural"]
        self.assertTrue(all("Delegate independent" in item.prompt for item in guided))
        self.assertTrue(all("Delegate independent" not in item.prompt for item in natural))

    def test_qwen_summary_and_marker_gate(self):
        payload = [
            {
                "type": "result",
                "subtype": "success",
                "duration_ms": 123,
                "num_turns": 2,
                "result": "Found Alpha and Beta",
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 4,
                    "cache_read_input_tokens": 10,
                },
                "stats": {
                    "models": {"m": {"api": {"totalRequests": 3}}},
                    "tools": {
                        "totalCalls": 2,
                        "totalFail": 1,
                        "totalDecisions": {"auto_accept": 1, "reject": 1},
                        "byName": {"agent": {"count": 1, "success": 1}},
                    },
                },
            }
        ]
        decoded = decode_json_document("warning\n" + json.dumps(payload))
        summary = summarize_qwen_records(decoded)
        self.assertTrue(summary["runtime_success"])
        self.assertEqual(summary["spawn_count"], 1)
        self.assertEqual(summary["request_count"], 3)
        self.assertEqual(summary["tool_failure_count"], 1)
        self.assertEqual(summary["permission_rejection_count"], 1)
        gate = evaluate_markers(summary["final_text"], (("alpha",), ("beta",)))
        self.assertTrue(gate["content_gate_passed"])

    def test_qwen_stream_json_records_are_decoded(self):
        payload = "\n".join(
            (
                json.dumps({"type": "system", "subtype": "init"}),
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "result": "done",
                        "stats": {},
                    }
                ),
            )
        )
        records = decode_qwen_records(payload)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[-1]["result"], "done")

    def test_qwen_stream_summary_recovers_tools_without_stats(self):
        records = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "spawn-1",
                            "name": "agent",
                            "input": {},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "spawn-1",
                            "is_error": False,
                            "content": "done",
                        }
                    ]
                },
            },
            {
                "type": "result",
                "subtype": "success",
                "duration_ms": 10,
                "num_turns": 2,
                "result": "done",
                "usage": {},
                "permission_denials": [],
            },
        ]
        summary = summarize_qwen_records(records)
        self.assertEqual(summary["tool_call_count"], 1)
        self.assertEqual(summary["spawn_attempt_count"], 1)
        self.assertEqual(summary["spawn_count"], 1)

    def test_aggregate_reports_factor_effects(self):
        rows = []
        for runtime in ("qwen_code", "codex"):
            for policy in ("natural", "policy_guided"):
                for structure in ("sequential", "parallelizable"):
                    for block in ("a", "b"):
                        spawn = int(
                            policy == "policy_guided"
                            and structure == "parallelizable"
                        )
                        rows.append(
                            {
                                "runtime": runtime,
                                "policy": policy,
                                "structure": structure,
                                "block": block,
                                "spawn_count": spawn,
                                "runtime_success": True,
                                "task_success": True,
                                "marker_coverage": 1.0,
                                "duration_ms": 10,
                                "turn_count": 1,
                                "request_count": 1,
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                            }
                        )
        summary = aggregate_rows(rows)
        self.assertEqual(summary["run_count"], 16)
        self.assertEqual(summary["replicates_per_cell"], [2])
        guided_parallel = next(
            item
            for item in summary["cells"]
            if item["runtime"] == "codex"
            and item["policy"] == "policy_guided"
            and item["structure"] == "parallelizable"
        )
        self.assertEqual(guided_parallel["spawn_rate"], 1.0)

    def test_prometheus_gauge_sum_handles_labeled_scheduler_metrics(self):
        payload = """\
# HELP sglang:num_used_tokens The number of used tokens.
# TYPE sglang:num_used_tokens gauge
sglang:num_used_tokens{tp_rank="0"} 123.0
sglang:num_used_tokens{tp_rank="1"} 45.0
sglang:num_running_reqs{tp_rank="0"} 2.0
"""
        self.assertEqual(
            prometheus_gauge_sum(payload, "sglang:num_used_tokens"), 168.0
        )
        with self.assertRaisesRegex(ValueError, "metric is missing"):
            prometheus_gauge_sum(payload, "sglang:num_queue_reqs")


if __name__ == "__main__":
    unittest.main()
