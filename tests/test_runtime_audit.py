import base64
import json
import gzip
import queue
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from beliefkv.runtime.audit import (
    AuditLevel,
    PolicySnapshotLog,
    RequestTokenTraceLog,
    RuntimeAuditLog,
)


class RuntimeAuditLogTest(unittest.TestCase):
    def test_disabled_log_performs_no_io(self):
        audit = RuntimeAuditLog(None, run_id="disabled")
        self.assertFalse(audit.enabled)
        audit.emit("ignored", 1.0, request_id="req")
        audit.close()

    def test_records_are_append_only_and_run_scoped(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "runtime.jsonl"
            with RuntimeAuditLog(path, run_id="run-a") as audit:
                audit.emit("request_deferred", 1.5, request_id="req-1")
                audit.emit("request_admitted", 2.0, request_id="req-1")
            with RuntimeAuditLog(path, run_id="run-b") as audit:
                audit.emit("runtime_initialized", 3.0)

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 3)
            self.assertEqual([item["sequence"] for item in records], [1, 2, 1])
            self.assertEqual([item["run_id"] for item in records], ["run-a", "run-a", "run-b"])
            self.assertEqual(records[0]["request_id"], "req-1")
            self.assertEqual(records[0]["schema_version"], 2)

    def test_rejects_non_finite_json_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            with RuntimeAuditLog(Path(temporary) / "audit.jsonl") as audit:
                with self.assertRaises(ValueError):
                    audit.emit("bad", float("nan"))

    def test_debug_events_are_sampled_and_oversize_payloads_are_compacted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            with RuntimeAuditLog(
                path,
                run_id="bounded",
                debug_sample_rate=0.5,
                max_debug_event_bytes=256,
            ) as audit:
                for sequence in range(4):
                    audit.emit(
                        "physical_bundle_preview",
                        float(sequence),
                        payload="x" * 2048,
                    )
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(
                all(item["event"] == "audit_debug_event_oversize" for item in records)
            )
            self.assertEqual(audit.summary()["sampled_debug_count"], 2)
            self.assertEqual(audit.summary()["oversize_debug_count"], 2)

    def test_correctness_event_applies_backpressure_instead_of_dropping(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit = RuntimeAuditLog(Path(temporary) / "audit.jsonl")
            with mock.patch.object(audit._queue, "put", wraps=audit._queue.put) as put:
                audit.emit(
                    "transfer_acknowledged",
                    1.0,
                    audit_level=AuditLevel.CORRECTNESS,
                )
            audit.close()
            self.assertGreaterEqual(put.call_count, 1)
            self.assertEqual(audit.summary()["dropped_debug_count"], 0)
            self.assertEqual(audit.summary()["written_count"], 1)

    def test_policy_snapshot_log_is_exclusive_and_gzip_compressed(self):
        class _Input:
            def to_dict(self):
                return {"snapshot": "state"}

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.jsonl.gz"
            with PolicySnapshotLog(
                path,
                trace_id="trace-a",
                trace_sensitivity="timing_sensitive",
            ) as snapshots:
                sequence = snapshots.emit(_Input(), trigger="graph_or_queue")
                self.assertEqual(sequence, 1)
            with gzip.open(path, mode="rt", encoding="utf-8") as stream:
                record = json.loads(stream.readline())
            self.assertEqual(record["sequence"], 1)
            self.assertEqual(record["trace_id"], "trace-a")
            self.assertEqual(record["trigger"], "graph_or_queue")
            self.assertEqual(record["policy_input"], {"snapshot": "state"})
            with self.assertRaises(FileExistsError):
                PolicySnapshotLog(
                    path,
                    trace_id="trace-b",
                    trace_sensitivity="timing_sensitive",
                )

    def test_policy_snapshot_log_drops_instead_of_blocking_when_full(self):
        class _Input:
            def to_dict(self):
                return {"snapshot": "state"}

        with tempfile.TemporaryDirectory() as temporary:
            with PolicySnapshotLog(
                Path(temporary) / "bounded.jsonl.gz",
                trace_id="trace-bounded",
                trace_sensitivity="timing_sensitive",
                max_pending=1,
            ) as snapshots:
                with mock.patch.object(
                    snapshots._queue,
                    "put_nowait",
                    side_effect=queue.Full,
                ):
                    self.assertEqual(
                        snapshots.emit(_Input(), trigger="bounded"),
                        0,
                    )
                self.assertEqual(snapshots.count, 0)
                self.assertEqual(snapshots.dropped_count, 1)

    def test_request_token_trace_preserves_equality_without_raw_token_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "request_tokens.jsonl.gz"
            with RequestTokenTraceLog(path, run_id="run-a") as trace:
                trace.emit(
                    "request_prompt",
                    1.0,
                    [11, 22, 11],
                    request_id="request-a",
                )
                trace.emit(
                    "cache_final_commit",
                    2.0,
                    [11, 22, 11, 33],
                    request_id="request-a",
                )
                self.assertEqual(trace.count, 2)
            self.assertEqual(trace.written_count, 2)

            with gzip.open(path, mode="rt", encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream]
            decoded = []
            for record in records:
                packed = base64.b64decode(record["token_symbols_b64"])
                decoded.append(
                    struct.unpack(f"<{record['token_count']}Q", packed)
                )
                self.assertEqual(
                    record["token_encoding"], RequestTokenTraceLog.ENCODING
                )
            self.assertEqual(decoded[0], decoded[1][:3])
            self.assertEqual(decoded[0][0], decoded[0][2])
            self.assertNotEqual(decoded[0][0], decoded[0][1])


if __name__ == "__main__":
    unittest.main()
