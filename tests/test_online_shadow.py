import unittest

from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.predictor.composer import (
    InvocationPredictionFeatures,
    RemainingTimePredictor,
    observed_boundary_action,
)
from beliefkv.predictor.online_shadow import (
    _features_for_invocation,
    build_frontier_shadow_records,
)
from beliefkv.predictor.structured_frontier import (
    EmpiricalDistribution,
    LocalFrontierPrediction,
)


class FakeFrontierModel:
    def __init__(self) -> None:
        self.predict_calls = 0
        self.boundary = {"tool_call": 0.7, "final_answer": 0.3}

    def predict(self, features: LocalFrontierPrediction) -> LocalFrontierPrediction:
        self.predict_calls += 1
        return LocalFrontierPrediction(
            invocation_id=features.invocation_id,
            boundary_distribution=dict(self.boundary),
            current_sequence_tokens=features.current_sequence_tokens,
            remaining_decode_tokens=EmpiricalDistribution(
                (10.0, 20.0), (0.5, 0.5), 1.0
            ),
            remaining_external_wait=EmpiricalDistribution(
                (100.0,), (1.0,), 1.0
            ),
            tool_terminal_distribution={"success": 0.8, "error": 0.2},
            prompt_growth_tokens=EmpiricalDistribution((50.0,), (1.0,), 1.0),
            next_output_tokens=EmpiricalDistribution((30.0,), (1.0,), 1.0),
            support_level="exact",
            calibration_coverage=0.9,
        )


def build_graph() -> RuntimeCausalContextGraph:
    graph = RuntimeCausalContextGraph()
    graph.apply(
        RuntimeEvent(
            event_id="s",
            ts_ms=1.0,
            kind=RuntimeEventKind.WORKFLOW_START,
            workflow_id="wf",
        )
    )
    graph.apply(
        RuntimeEvent(
            event_id="c",
            ts_ms=2.0,
            kind=RuntimeEventKind.INVOCATION_CREATE,
            workflow_id="wf",
            invocation_id="root",
            context_id="ctx",
            context_epoch=0,
        )
    )
    graph.apply(
        RuntimeEvent(
            event_id="l",
            ts_ms=3.0,
            kind=RuntimeEventKind.LLM_SUBMIT,
            workflow_id="wf",
            invocation_id="root",
            attributes={"prompt_tokens": 100, "context_tokens": 150},
        )
    )
    return graph


class OnlineShadowTest(unittest.TestCase):
    def test_no_frontier_model_emits_nothing(self) -> None:
        predictor = RemainingTimePredictor()
        records, signatures = build_frontier_shadow_records(
            build_graph(), predictor, now_ms=100.0
        )
        self.assertEqual(records, ())
        self.assertEqual(signatures, {})

    def test_emits_only_on_signature_change_after_interval(self) -> None:
        graph = build_graph()
        model = FakeFrontierModel()
        predictor = RemainingTimePredictor(_frontier=model)
        predictor.features["root"] = InvocationPredictionFeatures(
            context_tokens=150,
            generated_tokens=5,
            action_history=[],
        )

        records, signatures = build_frontier_shadow_records(
            graph, predictor, now_ms=1_000.0
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].invocation_id, "root")
        self.assertEqual(records[0].support_level, "exact")
        self.assertEqual(records[0].feature_source, "online_approx")
        self.assertEqual(records[0].remaining_decode_tokens_p50, 10.0)
        self.assertIn("root", signatures)

        unchanged, signatures = build_frontier_shadow_records(
            graph,
            predictor,
            now_ms=1_100.0,
            last_signatures=signatures,
        )
        self.assertEqual(unchanged, ())

        model.boundary = {"final_answer": 1.0}
        within_interval, signatures = build_frontier_shadow_records(
            graph,
            predictor,
            now_ms=1_400.0,
            last_signatures=signatures,
        )
        self.assertEqual(within_interval, ())

        after_interval, _ = build_frontier_shadow_records(
            graph,
            predictor,
            now_ms=2_400.0,
            last_signatures=signatures,
        )
        self.assertEqual(len(after_interval), 1)
        self.assertEqual(after_interval[0].boundary_top, "final_answer")

    def test_terminal_invocations_are_excluded(self) -> None:
        graph = build_graph()
        graph.apply(
            RuntimeEvent(
                event_id="t",
                ts_ms=4.0,
                kind=RuntimeEventKind.LLM_RESULT,
                workflow_id="wf",
                invocation_id="root",
                attributes={"terminal": True},
            )
        )
        model = FakeFrontierModel()
        predictor = RemainingTimePredictor(_frontier=model)
        records, _ = build_frontier_shadow_records(
            graph, predictor, now_ms=1_000.0
        )
        self.assertEqual(records, ())

    def test_frontier_mode_legacy_predictions_are_unusable(self) -> None:
        graph = build_graph()
        model = FakeFrontierModel()
        predictor = RemainingTimePredictor(_frontier=model)
        predictions = predictor.predict_all(graph, now_ms=100.0)
        prediction = predictions["ctx"]
        self.assertFalse(prediction.usable)
        self.assertEqual(prediction.backoff_level, "frontier_shadow")

    def test_observe_event_builds_training_vocabulary_history(self) -> None:
        predictor = RemainingTimePredictor()
        predictor.observe_event(
            RuntimeEvent(
                event_id="r1",
                ts_ms=10.0,
                kind=RuntimeEventKind.LLM_RESULT,
                workflow_id="wf",
                invocation_id="root",
                attributes={"structured_action_kinds": ("function_call",)},
            )
        )
        predictor.observe_event(
            RuntimeEvent(
                event_id="r2",
                ts_ms=11.0,
                kind=RuntimeEventKind.TOOL_END,
                workflow_id="wf",
                invocation_id="root",
            )
        )
        predictor.observe_event(
            RuntimeEvent(
                event_id="r3",
                ts_ms=12.0,
                kind=RuntimeEventKind.SPAWN,
                workflow_id="wf",
                invocation_id="root",
            )
        )
        self.assertEqual(
            predictor.features["root"].boundary_history,
            ["function_call", "tool_end", "spawn"],
        )
        self.assertEqual(
            observed_boundary_action(
                RuntimeEvent(
                    event_id="r4",
                    ts_ms=13.0,
                    kind=RuntimeEventKind.LLM_RESULT,
                    workflow_id="wf",
                    invocation_id="root",
                )
            ),
            "final",
        )

    def test_feature_mapping_matches_training_format(self) -> None:
        graph = build_graph()
        graph.apply(
            RuntimeEvent(
                event_id="t1",
                ts_ms=4.0,
                kind=RuntimeEventKind.INVOCATION_CREATE,
                workflow_id="wf",
                invocation_id="tool-a",
                context_id="ctx2",
                context_epoch=0,
            )
        )
        graph.apply(
            RuntimeEvent(
                event_id="t2",
                ts_ms=5.0,
                kind=RuntimeEventKind.TOOL_START,
                workflow_id="wf",
                invocation_id="tool-a",
                attributes={"tool_family": "file", "backend_class": "py"},
            )
        )
        graph.apply(
            RuntimeEvent(
                event_id="t3",
                ts_ms=6.0,
                kind=RuntimeEventKind.INVOCATION_CREATE,
                workflow_id="wf",
                invocation_id="tool-b",
                context_id="ctx3",
                context_epoch=0,
            )
        )
        graph.apply(
            RuntimeEvent(
                event_id="t4",
                ts_ms=7.0,
                kind=RuntimeEventKind.TOOL_START,
                workflow_id="wf",
                invocation_id="tool-b",
                attributes={"tool_family": "file", "backend_class": "py"},
            )
        )
        predictor = RemainingTimePredictor()
        for event_id in ("t2", "t4"):
            predictor.observe_event(
                RuntimeEvent(
                    event_id=event_id,
                    ts_ms=1.0,
                    kind=RuntimeEventKind.TOOL_START,
                    workflow_id="wf",
                    invocation_id="tool-a" if event_id == "t2" else "tool-b",
                    attributes={
                        "tool_family": "file",
                        "backend_class": "py",
                    },
                )
            )
        features = _features_for_invocation(
            graph,
            "tool-a",
            predictor,
            now_ms=100.0,
            active_tool_count=2,
            family_counts={"file": 2},
        )
        self.assertEqual(features.tool_family, "file")
        self.assertEqual(features.backend_class, "py")
        self.assertEqual(features.backend_pressure, "active_family:2")
        self.assertEqual(features.active_tool_count, 2)

        unknown = _features_for_invocation(
            graph,
            "root",
            predictor,
            now_ms=100.0,
            active_tool_count=2,
            family_counts={"file": 2},
        )
        self.assertEqual(unknown.tool_family, "unknown")
        self.assertEqual(unknown.backend_pressure, "unknown")


if __name__ == "__main__":
    unittest.main()
