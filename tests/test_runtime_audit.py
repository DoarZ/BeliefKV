import json
import tempfile
import unittest
from pathlib import Path

from beliefkv.runtime.audit import RuntimeAuditLog


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
            self.assertEqual(records[0]["schema_version"], 1)

    def test_rejects_non_finite_json_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            with RuntimeAuditLog(Path(temporary) / "audit.jsonl") as audit:
                with self.assertRaises(ValueError):
                    audit.emit("bad", float("nan"))


if __name__ == "__main__":
    unittest.main()
