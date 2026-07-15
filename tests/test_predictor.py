import math
import unittest

from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.predictor.action_context_tree import ActionObservation, SemiMarkovContextTree
from beliefkv.predictor.calibration import RollingIntervalCalibrator
from beliefkv.predictor.composer import InvocationPredictionFeatures, RemainingTimePredictor
from beliefkv.predictor.service_cost import LLMServiceCostModel, LLMServiceSample
from beliefkv.predictor.taxonomy import ActionKind, ToolTaxonomy
from beliefkv.predictor.tool_survival import (
    HierarchicalToolSurvivalModel,
    KaplanMeierCurve,
    ToolDurationSample,
)
from beliefkv.predictor.types import RemainingTimePrediction


class SurvivalModelTest(unittest.TestCase):
    def test_kaplan_meier_accounts_for_right_censoring(self):
        curve = KaplanMeierCurve().fit(
            [(10, True), (20, False), (30, True), (40, True)]
        )
        self.assertAlmostEqual(curve.survival(10), 0.75)
        self.assertAlmostEqual(curve.conditional_survival(10, 20), (0.75 * 0.5) / 0.75)
        self.assertEqual(curve.remaining_quantile(0, 0.5), 30)

    def test_hierarchical_model_uses_backend_then_global_backoff(self):
        samples = [
            ToolDurationSample(100 + index, True, "search", "google")
            for index in range(6)
        ] + [
            ToolDurationSample(200 + index, True, "shell", "bash")
            for index in range(6)
        ]
        model = HierarchicalToolSurvivalModel(
            min_backend_samples=5, min_family_samples=8
        )
        model.fit(samples)
        exact = model.predict(
            context_id="ctx",
            now_ms=0,
            elapsed_ms=0,
            family="search",
            backend_class="google",
            transfer_window_ms=50,
        )
        unknown = model.predict(
            context_id="other",
            now_ms=0,
            elapsed_ms=0,
            family="unseen",
            transfer_window_ms=50,
        )
        self.assertEqual(exact.backoff_level, "backend")
        self.assertEqual(unknown.backoff_level, "global")
        self.assertGreater(unknown.ood_score, exact.ood_score)


class ContextTreeTest(unittest.TestCase):
    def test_longest_supported_context_changes_next_action(self):
        model = SemiMarkovContextTree(max_order=2, min_context_count=2)
        model.fit(
            [
                [
                    ActionObservation(ActionKind.LLM_TOOL_CALL, 5),
                    ActionObservation(ActionKind.TOOL_SEARCH, 100),
                    ActionObservation(ActionKind.LLM_TEXT, 20),
                ],
                [
                    ActionObservation(ActionKind.LLM_TOOL_CALL, 6),
                    ActionObservation(ActionKind.TOOL_SEARCH, 110),
                    ActionObservation(ActionKind.LLM_TEXT, 21),
                ],
                [
                    ActionObservation(ActionKind.LLM_TEXT, 10),
                    ActionObservation(ActionKind.RETURN, 1),
                ],
            ]
        )
        prediction = model.predict(
            [ActionKind.LLM_TOOL_CALL, ActionKind.TOOL_SEARCH],
            current_action=ActionKind.TOOL_SEARCH,
        )
        self.assertEqual(prediction.selected_order, 2)
        self.assertGreater(
            prediction.next_distribution[ActionKind.LLM_TEXT],
            prediction.next_distribution[ActionKind.RETURN],
        )
        self.assertTrue(math.isfinite(prediction.current_remaining_p50_ms))


class ServiceCostTest(unittest.TestCase):
    def test_cache_hits_reduce_only_prefill_component(self):
        model = LLMServiceCostModel()
        model.observe(
            LLMServiceSample(
                model="qwen",
                prompt_tokens=1000,
                cache_hit_tokens=0,
                output_tokens=100,
                batch_size=1,
                context_tokens=1000,
                queue_ms=2,
                prefill_ms=10,
                decode_ms=500,
            )
        )
        cold = model.estimate(
            model="qwen",
            prompt_tokens=1000,
            cache_hit_tokens=0,
            expected_output_tokens=100,
            batch_size=1,
            context_tokens=1000,
        )
        warm = model.estimate(
            model="qwen",
            prompt_tokens=1000,
            cache_hit_tokens=900,
            expected_output_tokens=100,
            batch_size=1,
            context_tokens=1000,
        )
        self.assertLess(warm.prefill_ms, cold.prefill_ms)
        self.assertEqual(warm.decode_ms, cold.decode_ms)


class CalibrationTest(unittest.TestCase):
    def test_bad_coverage_lowers_confidence_and_widens_interval(self):
        calibrator = RollingIntervalCalibrator(
            min_observations=3, target_p95_coverage=0.9
        )
        prediction = RemainingTimePrediction(
            context_id="ctx",
            generated_ts_ms=0,
            p50_ms=10,
            p90_ms=15,
            p95_ms=20,
            confidence=1,
            ood_score=0,
        )
        for actual in (40, 50, 60):
            calibrator.observe(prediction, actual)
        adjusted = calibrator.adjust(prediction)
        self.assertGreater(adjusted.p95_ms, prediction.p95_ms)
        self.assertLess(adjusted.confidence, prediction.confidence)


class ComposerTest(unittest.TestCase):
    def test_parent_wait_time_composes_blocking_child_service(self):
        graph = RuntimeCausalContextGraph()
        sequence = 0

        def emit(kind, **kwargs):
            nonlocal sequence
            sequence += 1
            graph.apply(
                RuntimeEvent(
                    event_id=f"e{sequence}",
                    ts_ms=float(sequence),
                    kind=kind,
                    workflow_id="wf",
                    **kwargs,
                )
            )

        emit(RuntimeEventKind.WORKFLOW_START)
        emit(
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id="parent",
            context_id="ctx-parent",
        )
        emit(
            RuntimeEventKind.INVOCATION_CREATE,
            invocation_id="child",
            context_id="ctx-child",
        )
        emit(
            RuntimeEventKind.CALL,
            invocation_id="parent",
            target_invocation_id="child",
        )
        predictor = RemainingTimePredictor()
        predictor.set_features(
            "child",
            InvocationPredictionFeatures(
                model="qwen",
                prompt_tokens=800,
                expected_output_tokens=10,
                context_tokens=800,
            ),
        )
        child = predictor.predict_context(
            graph,
            context_id="ctx-child",
            now_ms=10,
            transfer_window_ms=20,
        )
        parent = predictor.predict_context(
            graph,
            context_id="ctx-parent",
            now_ms=10,
            transfer_window_ms=20,
        )
        self.assertEqual(parent.backoff_level, "child_composition")
        self.assertEqual(parent.p50_ms, child.p50_ms)

    def test_tool_start_at_zero_keeps_elapsed_time(self):
        graph = RuntimeCausalContextGraph()
        graph.apply(
            RuntimeEvent("start", 0, RuntimeEventKind.WORKFLOW_START, "wf")
        )
        graph.apply(
            RuntimeEvent(
                "create",
                0,
                RuntimeEventKind.INVOCATION_CREATE,
                "wf",
                invocation_id="agent",
                context_id="ctx",
                context_epoch=0,
            )
        )
        graph.apply(
            RuntimeEvent(
                "tool",
                0,
                RuntimeEventKind.TOOL_START,
                "wf",
                invocation_id="agent",
                attributes={"tool_family": "search"},
            )
        )
        model = HierarchicalToolSurvivalModel(
            min_family_samples=1, min_backend_samples=1
        )
        model.fit([ToolDurationSample(10, True, "search")])
        prediction = RemainingTimePredictor(tool_model=model).predict_context(
            graph,
            context_id="ctx",
            now_ms=5,
            transfer_window_ms=1,
        )
        self.assertEqual(prediction.p50_ms, 5)


class TaxonomyTest(unittest.TestCase):
    def test_runtime_specific_tool_name_maps_to_stable_family(self):
        taxonomy = ToolTaxonomy()
        normalized = taxonomy.normalize("web_search_v2", "https://api.example.com/v1")
        self.assertEqual(normalized.family, "search")
        self.assertEqual(normalized.backend_class, "example.com")


if __name__ == "__main__":
    unittest.main()
