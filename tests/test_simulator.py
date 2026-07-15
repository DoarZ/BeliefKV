import json
import tempfile
import unittest
from pathlib import Path

from beliefkv.core.config import BeliefKVConfig
from beliefkv.metrics.artifacts import ExperimentArtifactWriter
from beliefkv.simulator.page_simulator import PageLevelSimulator
from beliefkv.simulator.schema import SimulationScenario


def runtime(ts_ms, event_id, kind, **kwargs):
    event = {
        "event_id": event_id,
        "ts_ms": ts_ms,
        "kind": kind,
        "workflow_id": "wf",
    }
    event.update(kwargs)
    return {"ts_ms": ts_ms, "kind": "runtime", "event": event}


class PageSimulatorTest(unittest.TestCase):
    def config(self) -> BeliefKVConfig:
        return BeliefKVConfig(
            hbm_capacity_bytes=1000,
            host_capacity_bytes=10000,
            reserve_hbm_bytes=300,
            urgent_chunk_bytes=1000,
            shadow_chunk_bytes=500,
            shadow_min_parked_ms=0,
            predictor_enabled=False,
        )

    def base_events(self):
        return [
            runtime(0, "start", "workflow_start"),
            runtime(
                1,
                "create",
                "invocation_create",
                invocation_id="parent",
                context_id="ctx",
                context_epoch=0,
            ),
            runtime(
                2,
                "tool",
                "tool_start",
                invocation_id="parent",
                tool_family="search",
            ),
            {
                "ts_ms": 3,
                "kind": "cache_insert",
                "page_id": 1,
                "generation": 0,
                "size_bytes": 400,
                "bindings": [{"context_id": "ctx", "context_epoch": 0}],
            },
        ]

    def test_prepare_then_pressure_commit_is_counted_as_useful_shadow(self):
        events = self.base_events()
        events.extend(
            [
                {
                    "ts_ms": 100,
                    "kind": "request_submit",
                    "request_id": "request",
                    "workflow_id": "wf",
                    "invocation_id": "parent",
                    "context_id": "ctx",
                    "context_epoch": 0,
                    "submitted_ts_ms": 100,
                    "uncached_prompt_tokens": 4,
                    "expected_output_tokens": 1,
                    "kv_bytes_per_token": 100,
                },
                {"ts_ms": 110, "kind": "admission_ack", "request_id": "request"},
                runtime(200, "end", "workflow_end"),
            ]
        )
        scenario = SimulationScenario.from_dict({"name": "useful-shadow", "events": events})
        result = PageLevelSimulator(self.config()).run(scenario)
        self.assertEqual(result.summary["shadow_prepared_bytes"], 400)
        self.assertEqual(result.summary["useful_shadow_bytes"], 400)
        self.assertEqual(result.summary["admission_stall_ms"]["request"], 0.0)
        self.assertEqual(result.summary["workflow_completion_ms"]["wf"], 200)

    def test_wakeup_stops_future_shadow_but_current_dma_chunk_completes(self):
        events = self.base_events()
        events.append(
            runtime(3.01, "tool-end", "tool_end", invocation_id="parent")
        )
        scenario = SimulationScenario.from_dict({"name": "cancel-shadow", "events": events})
        result = PageLevelSimulator(self.config()).run(scenario)
        statuses = [
            record["payload"]["status"]
            for record in result.event_log
            if record["kind"] == "command_ack"
        ]
        self.assertIn("completed", statuses)
        reasons = [
            record["payload"]["reason"]
            for record in result.event_log
            if record["kind"] == "command_ack"
        ]
        self.assertIn("current_nonpreemptible_chunk_completed_after_cancel", reasons)
        self.assertEqual(result.summary["final_gpu_bytes"], 400)
        self.assertEqual(result.summary["final_cpu_bytes"], 400)

    def test_experiment_writer_creates_self_describing_artifacts(self):
        scenario = SimulationScenario.from_dict(
            {"name": "artifact", "seed": 7, "events": self.base_events()}
        )
        result = PageLevelSimulator(self.config()).run(scenario)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            writer = ExperimentArtifactWriter(Path.cwd(), output)
            run_dir = writer.write(
                run_id="run-1",
                workload=scenario.name,
                seed=scenario.seed,
                config=self.config(),
                result=result,
            )
            manifest = json.loads((run_dir / "manifest.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(manifest["seed"], 7)
            self.assertEqual(manifest["workload"], "artifact")
            self.assertIn("config_sha256", manifest)
            self.assertEqual(summary["shadow_prepared_bytes"], 400)
            self.assertTrue((run_dir / "events.jsonl").stat().st_size > 0)

    def test_simulator_rejects_accidental_stateful_reuse(self):
        scenario = SimulationScenario.from_dict(
            {"name": "single-use", "events": self.base_events()}
        )
        simulator = PageLevelSimulator(self.config())
        simulator.run(scenario)
        with self.assertRaises(RuntimeError):
            simulator.run(scenario)


if __name__ == "__main__":
    unittest.main()
