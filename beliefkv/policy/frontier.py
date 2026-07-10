from __future__ import annotations

from collections import defaultdict
from math import inf

from beliefkv.core.types import ContinuationBelief, KVObjectMeta


class BeliefFrontier:
    def __init__(self, beliefs: list[ContinuationBelief]):
        self._beliefs = beliefs
        self._by_workflow: dict[str, list[ContinuationBelief]] = defaultdict(list)
        for belief in beliefs:
            self._by_workflow[belief.workflow_id].append(belief)

    def match(self, kv: KVObjectMeta) -> tuple[float, float, float]:
        """Return reuse probability, next-use p50, and confidence.

        Matching is intentionally conservative: workflow id is required, while
        agent and branch ids refine the probability when available.
        """

        probability = 0.0
        next_use_p50 = inf
        confidence_mass = 0.0

        for workflow_id in kv.workflow_ids:
            for belief in self._by_workflow.get(workflow_id, []):
                if kv.agent_ids and belief.agent_id not in kv.agent_ids:
                    continue
                if kv.branch_ids and belief.branch_id not in kv.branch_ids:
                    continue
                p = belief.effective_probability
                probability += p
                next_use_p50 = min(next_use_p50, belief.ready_time_p50_ms)
                confidence_mass += p * belief.confidence

        probability = max(0.0, min(1.0, probability))
        confidence = 0.0 if probability == 0 else confidence_mass / probability
        return probability, next_use_p50, confidence
