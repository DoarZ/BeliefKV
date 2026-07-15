import tempfile
import unittest
from pathlib import Path

from beliefkv.experiments.codex_ab import (
    distribution,
    summarize_reactive_audit,
    summarize_gpu_samples,
    summarize_server_log,
)


class CodexABAnalysisTest(unittest.TestCase):
    def test_distribution_is_deterministic(self):
        self.assertEqual(
            distribution([1, 2, 3, 4]),
            {"count": 4, "mean": 2.5, "p50": 2.5, "p95": 3.85, "max": 4.0},
        )

    def test_server_and_gpu_parsers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = root / "server.log"
            server.write_text(
                "Prefill batch. #new-seq: 1, #new-token: 128, "
                "#cached-token: 256, token usage: 0.50, #running-req: 0, "
                "#queue-req: 0\n"
                "Decode batch. #running-req: 3, token usage: 0.75, "
                "gen throughput (token/s): 42.5, #queue-req: 2\n",
                encoding="utf-8",
            )
            gpu = root / "gpu.csv"
            gpu.write_text(
                "2026/07/15 20:00:00, 100, 200, 75, 25, 150\n",
                encoding="utf-8",
            )
            server_summary = summarize_server_log(server)
            gpu_summary = summarize_gpu_samples(gpu)

        self.assertEqual(server_summary["token_usage"]["max"], 0.75)
        self.assertEqual(server_summary["peak_running_requests"], 3)
        self.assertEqual(server_summary["peak_queued_requests"], 2)
        self.assertEqual(server_summary["prefill_batch_count"], 1)
        self.assertEqual(server_summary["prefill_computed_tokens"], 128)
        self.assertEqual(server_summary["prefill_cached_token_observations"], 256)
        self.assertEqual(gpu_summary["memory_used_mib"]["max"], 100.0)
        self.assertEqual(gpu_summary["power_watts"]["mean"], 150.0)

    def test_reactive_audit_requires_acknowledged_bytes_for_physical_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit = Path(temporary) / "audit.jsonl"
            audit.write_text(
                "\n".join(
                    (
                        '{"event":"runtime_initialized"}',
                        '{"event":"runtime_event_delivery","accepted":true}',
                        '{"event":"admission_decision","admitted":false}',
                        '{"event":"transfer_dispatched","command_id":"c1",'
                        '"kind":"offload_context","selected_bytes":1024,"ts_ms":10}',
                        '{"event":"transfer_acknowledged","command_id":"c1",'
                        '"status":"completed","actual_bytes":768,"ts_ms":14}',
                        '{"event":"request_deferred","request_id":"r1","ts_ms":20}',
                        '{"event":"request_admitted","request_id":"r1","ts_ms":23}',
                        '{"event":"runtime_shutdown"}',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            summary = summarize_reactive_audit(audit)

        self.assertTrue(summary["physical_kv_transfer_observed"])
        self.assertEqual(
            summary["selected_transfer_bytes_by_kind"], {"offload_context": 1024}
        )
        self.assertEqual(summary["acknowledged_transfer_bytes"], 768)
        self.assertEqual(
            summary["transfer_acknowledgement_statuses"], {"completed": 1}
        )
        self.assertEqual(summary["transfer_callback_latency_ms"]["max"], 4.0)
        self.assertEqual(summary["admission_queue_wait_ms"]["max"], 3.0)
        self.assertEqual(summary["transfer_watchdog_expirations"], 0)

    def test_drop_unowned_is_reclamation_not_a_kv_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit = Path(temporary) / "audit.jsonl"
            audit.write_text(
                "\n".join(
                    (
                        '{"event":"runtime_initialized"}',
                        '{"event":"transfer_dispatched","command_id":"c1",'
                        '"kind":"drop_unowned","selected_bytes":512}',
                        '{"event":"transfer_acknowledged","command_id":"c1",'
                        '"status":"completed","actual_bytes":512}',
                        '{"event":"runtime_shutdown"}',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            summary = summarize_reactive_audit(audit)

        self.assertFalse(summary["physical_kv_transfer_observed"])
        self.assertEqual(summary["acknowledged_kv_transfer_bytes"], 0)
        self.assertEqual(summary["acknowledged_reclamation_bytes"], 512)


if __name__ == "__main__":
    unittest.main()
