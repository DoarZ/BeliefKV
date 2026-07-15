import json
import tempfile
import unittest
from pathlib import Path

from beliefkv.control.controller import BeliefKVController
from beliefkv.core.config import BeliefKVConfig
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.predictor.composer import RemainingTimePredictor
from beliefkv.predictor.training import extract_training_corpus, train_predictor


def event(event_id, ts_ms, kind, workflow_id, invocation_id=None, **attributes):
    return RuntimeEvent(
        event_id=event_id,
        ts_ms=ts_ms,
        kind=kind,
        workflow_id=workflow_id,
        invocation_id=invocation_id,
        attributes=attributes,
    )


class PredictorTrainingTest(unittest.TestCase):
    def events(self):
        return [
            event("w1-start", 0, RuntimeEventKind.WORKFLOW_START, "wf-1"),
            RuntimeEvent(
                event_id="w1-create",
                ts_ms=1,
                kind=RuntimeEventKind.INVOCATION_CREATE,
                workflow_id="wf-1",
                invocation_id="inv-1",
                context_id="ctx-1",
                context_epoch=0,
            ),
            event(
                "llm-start",
                2,
                RuntimeEventKind.LLM_SUBMIT,
                "wf-1",
                "inv-1",
                request_id="r1",
                model="qwen",
                prompt_tokens=100,
                cache_hit_tokens=20,
                context_tokens=100,
            ),
            event(
                "llm-end",
                8,
                RuntimeEventKind.LLM_RESULT,
                "wf-1",
                "inv-1",
                request_id="r1",
                output_tokens=4,
                queue_ms=1,
                prefill_ms=2,
                decode_ms=3,
            ),
            event(
                "tool-start",
                10,
                RuntimeEventKind.TOOL_START,
                "wf-1",
                "inv-1",
                tool_call_id="t1",
                tool_name="web_search",
                tool_family="search",
                backend_class="example.com",
            ),
            event(
                "tool-end",
                30,
                RuntimeEventKind.TOOL_END,
                "wf-1",
                "inv-1",
                tool_call_id="t1",
            ),
            event("return", 35, RuntimeEventKind.RETURN, "wf-1", "inv-1"),
            event("w1-end", 40, RuntimeEventKind.WORKFLOW_END, "wf-1"),
            event("w2-start", 0, RuntimeEventKind.WORKFLOW_START, "wf-2"),
            RuntimeEvent(
                event_id="w2-create",
                ts_ms=1,
                kind=RuntimeEventKind.INVOCATION_CREATE,
                workflow_id="wf-2",
                invocation_id="inv-2",
                context_id="ctx-2",
                context_epoch=0,
            ),
            event(
                "tool-censored",
                5,
                RuntimeEventKind.TOOL_START,
                "wf-2",
                "inv-2",
                tool_call_id="t2",
                tool_family="search",
                backend_class="example.com",
            ),
            event("w2-end", 55, RuntimeEventKind.WORKFLOW_END, "wf-2"),
        ]

    def test_extracts_censoring_actions_and_service_components(self):
        corpus = extract_training_corpus(self.events())
        self.assertEqual(corpus.summary.tool_samples, 2)
        self.assertEqual(corpus.summary.censored_tool_samples, 1)
        self.assertEqual(corpus.summary.action_trajectories, 2)
        self.assertEqual(corpus.summary.llm_service_samples, 1)
        self.assertEqual(corpus.service_samples[0].prefill_ms, 2)

    def test_artifact_roundtrip_preserves_predictions(self):
        corpus = extract_training_corpus(self.events())
        predictor = train_predictor(
            corpus,
            min_family_samples=1,
            min_backend_samples=1,
            min_context_count=1,
        )
        before = predictor.tool_model.predict(
            context_id="ctx",
            now_ms=100,
            elapsed_ms=0,
            family="search",
            backend_class="example.com",
            transfer_window_ms=25,
        )
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "predictor.json"
            predictor.save(artifact, metadata={"dataset": "unit-test"})
            raw = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(raw["schema_version"], 1)
            loaded = RemainingTimePredictor.load(artifact)
            after = loaded.tool_model.predict(
                context_id="ctx",
                now_ms=100,
                elapsed_ms=0,
                family="search",
                backend_class="example.com",
                transfer_window_ms=25,
            )
        self.assertEqual(before, after)

    def test_controller_loads_artifact_and_updates_online_features(self):
        predictor = train_predictor(
            extract_training_corpus(self.events()),
            min_family_samples=1,
            min_backend_samples=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "predictor.json"
            predictor.save(artifact)
            controller = BeliefKVController(
                BeliefKVConfig(predictor_model_path=str(artifact))
            )
            controller.process_runtime_event(self.events()[0])
            controller.process_runtime_event(self.events()[1])
            controller.process_runtime_event(self.events()[4])
            features = controller.predictor.features["inv-1"]
            self.assertEqual(features.tool_backend_class, "example.com")
            self.assertEqual(features.action_history[-1].value, "tool_search")

    def test_unknown_artifact_schema_fails_closed(self):
        with self.assertRaises(ValueError):
            RemainingTimePredictor.from_dict({"schema_version": 99, "models": {}})


if __name__ == "__main__":
    unittest.main()
