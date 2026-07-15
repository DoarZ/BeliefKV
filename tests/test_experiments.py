import csv
import json
import tempfile
import unittest
from pathlib import Path

from beliefkv.experiments.matrix import ExperimentMatrix, ExperimentMatrixRunner


class ExperimentMatrixTest(unittest.TestCase):
    def _scenario(self, root: Path) -> Path:
        path = root / "scenario.json"
        path.write_text(
            json.dumps(
                {
                    "name": "tiny-workflow",
                    "seed": 7,
                    "config": {
                        "hbm_capacity_bytes": 1000,
                        "host_capacity_bytes": 2000,
                        "reserve_hbm_bytes": 100,
                        "urgent_chunk_bytes": 100,
                        "shadow_chunk_bytes": 100,
                    },
                    "events": [
                        {
                            "ts_ms": 0,
                            "kind": "runtime",
                            "event": {
                                "event_id": "start",
                                "ts_ms": 0,
                                "kind": "workflow_start",
                                "workflow_id": "wf",
                            },
                        },
                        {
                            "ts_ms": 10,
                            "kind": "runtime",
                            "event": {
                                "event_id": "end",
                                "ts_ms": 10,
                                "kind": "workflow_end",
                                "workflow_id": "wf",
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_matrix_publishes_complete_artifacts_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenario = self._scenario(root)
            matrix = ExperimentMatrix.from_dict(
                {
                    "name": "smoke",
                    "scenarios": [str(scenario)],
                    "variants": [
                        {
                            "name": "reactive",
                            "config": {
                                "predictor_enabled": False,
                                "shadow_enabled": False,
                            },
                        },
                        {
                            "name": "full",
                            "config": {
                                "predictor_enabled": True,
                                "shadow_enabled": True,
                            },
                        },
                    ],
                },
                base_directory=root,
            )
            output = root / "results"
            runner = ExperimentMatrixRunner(
                Path.cwd(), output, bootstrap_resamples=20
            )
            result = runner.run(matrix, run_id="matrix-1")
            self.assertEqual(len(result.run_rows), 2)
            self.assertTrue((result.matrix_dir / "matrix_manifest.json").is_file())
            self.assertTrue((result.matrix_dir / "summary.json").is_file())
            self.assertFalse((output / ".matrix-1.incomplete").exists())
            with (result.matrix_dir / "runs.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["variant"] for row in rows}, {"reactive", "full"})
            for row in result.run_rows:
                self.assertTrue(Path(row["run_dir"]).is_dir())
            with self.assertRaises(FileExistsError):
                runner.run(matrix, run_id="matrix-1")

    def test_duplicate_scenario_labels_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenario = self._scenario(root)
            matrix = ExperimentMatrix.from_dict(
                {
                    "name": "duplicate",
                    "scenarios": [
                        {"path": str(scenario), "label": "same"},
                        {"path": str(scenario), "label": "same"},
                    ],
                    "variants": [{"name": "reactive"}],
                },
                base_directory=root,
            )
            output = root / "results"
            with self.assertRaises(ValueError):
                ExperimentMatrixRunner(Path.cwd(), output).run(
                    matrix, run_id="invalid"
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
