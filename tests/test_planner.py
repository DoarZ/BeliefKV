import unittest

from beliefkv.core.types import (
    ContinuationBelief,
    DeviceState,
    KVAction,
    KVObjectMeta,
    PlannerConfig,
    RuntimeSnapshot,
)
from beliefkv.policy.planner import BeliefKVPlanner


class PlannerTest(unittest.TestCase):
    def test_active_decode_kv_is_protected(self):
        planner = BeliefKVPlanner(PlannerConfig(reserve_hbm_bytes=0))
        decisions = planner.plan(
            kv_objects=[
                KVObjectMeta(
                    object_id="active",
                    workflow_ids=frozenset({"wf"}),
                    agent_ids=frozenset({"coder"}),
                    branch_ids=frozenset({"main"}),
                    token_count=1024,
                    size_bytes=64 * 1024 * 1024,
                    device_state=DeviceState.GPU,
                    is_active_decode=True,
                )
            ],
            continuations=[],
            snapshot=RuntimeSnapshot(
                now_ms=0,
                hbm_capacity_bytes=128 * 1024 * 1024,
                hbm_used_bytes=120 * 1024 * 1024,
            ),
        )

        self.assertEqual(decisions[0].action, KVAction.KEEP_GPU)

    def test_far_future_gpu_object_can_be_offloaded_under_pressure(self):
        planner = BeliefKVPlanner(
            PlannerConfig(
                reserve_hbm_bytes=64 * 1024 * 1024,
                prefill_tokens_per_ms=20.0,
                high_hbm_pressure=0.7,
                offload_min_benefit_ms=1.0,
            )
        )
        decisions = planner.plan(
            kv_objects=[
                KVObjectMeta(
                    object_id="cold",
                    workflow_ids=frozenset({"wf-cold"}),
                    agent_ids=frozenset({"searcher"}),
                    branch_ids=frozenset({"b1"}),
                    token_count=8192,
                    size_bytes=96 * 1024 * 1024,
                    device_state=DeviceState.GPU,
                )
            ],
            continuations=[
                ContinuationBelief(
                    workflow_id="wf-cold",
                    agent_id="searcher",
                    branch_id="b1",
                    probability=0.8,
                    ready_time_p50_ms=1000.0,
                    ready_time_p95_ms=2000.0,
                )
            ],
            snapshot=RuntimeSnapshot(
                now_ms=0,
                hbm_capacity_bytes=128 * 1024 * 1024,
                hbm_used_bytes=120 * 1024 * 1024,
            ),
        )

        self.assertEqual(decisions[0].action, KVAction.OFFLOAD_CPU)

    def test_cpu_object_prefetches_when_next_use_is_imminent(self):
        planner = BeliefKVPlanner(
            PlannerConfig(reserve_hbm_bytes=0, prefetch_slack_ms=100.0)
        )
        decisions = planner.plan(
            kv_objects=[
                KVObjectMeta(
                    object_id="soon",
                    workflow_ids=frozenset({"wf"}),
                    agent_ids=frozenset({"coder"}),
                    branch_ids=frozenset({"main"}),
                    token_count=512,
                    size_bytes=16 * 1024 * 1024,
                    device_state=DeviceState.CPU,
                )
            ],
            continuations=[
                ContinuationBelief(
                    workflow_id="wf",
                    agent_id="coder",
                    branch_id="main",
                    probability=1.0,
                    ready_time_p50_ms=20.0,
                    ready_time_p95_ms=40.0,
                )
            ],
            snapshot=RuntimeSnapshot(
                now_ms=0,
                hbm_capacity_bytes=256 * 1024 * 1024,
                hbm_used_bytes=32 * 1024 * 1024,
            ),
        )

        self.assertEqual(decisions[0].action, KVAction.PREFETCH_GPU)


if __name__ == "__main__":
    unittest.main()
