from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.runtime.page_index import PageOwnershipIndex
from beliefkv.runtime.protocol import PhysicalResidency


class PrefixAffinityKind(str, Enum):
    PARENT_CHILD = "parent_child"
    SIBLING_TEMPLATE = "sibling_template"
    TEMPLATE = "template"
    PEER = "peer"
    CROSS_WORKFLOW = "cross_workflow"


@dataclass(frozen=True)
class ContextPrefixAffinity:
    context_a: str
    context_b: str
    workflow_a: str
    workflow_b: str
    kind: PrefixAffinityKind
    shared_physical_bytes: int
    union_physical_bytes: int
    shared_page_count: int
    byte_jaccard: float
    observed_ts_ms: float

    def __post_init__(self) -> None:
        if not self.context_a or not self.context_b or self.context_a >= self.context_b:
            raise ValueError("affinity contexts must be a sorted, distinct pair")
        if not self.workflow_a or not self.workflow_b:
            raise ValueError("affinity workflow IDs must be non-empty")
        if self.shared_physical_bytes <= 0:
            raise ValueError("affinity requires positive shared physical bytes")
        if self.union_physical_bytes < self.shared_physical_bytes:
            raise ValueError("union bytes cannot be smaller than shared bytes")
        if self.shared_page_count <= 0:
            raise ValueError("shared_page_count must be positive")
        if not 0.0 < self.byte_jaccard <= 1.0:
            raise ValueError("byte_jaccard must be in (0, 1]")
        if self.observed_ts_ms < 0:
            raise ValueError("observed_ts_ms must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "context_a": self.context_a,
            "context_b": self.context_b,
            "workflow_a": self.workflow_a,
            "workflow_b": self.workflow_b,
            "kind": self.kind.value,
            "shared_physical_bytes": self.shared_physical_bytes,
            "union_physical_bytes": self.union_physical_bytes,
            "shared_page_count": self.shared_page_count,
            "byte_jaccard": self.byte_jaccard,
            "observed_ts_ms": self.observed_ts_ms,
        }


class ContextPrefixAffinityIndex:
    """Build physical prefix affinities without inferring KV reuse from causality."""

    def __init__(
        self,
        graph: RuntimeCausalContextGraph,
        page_index: PageOwnershipIndex,
    ) -> None:
        self.graph = graph
        self.page_index = page_index

    def snapshot(self, *, now_ms: float) -> tuple[ContextPrefixAffinity, ...]:
        if now_ms < 0:
            raise ValueError("now_ms must be non-negative")
        context_bytes: dict[str, int] = {}
        shared_bytes: dict[tuple[str, str], int] = {}
        shared_pages: dict[tuple[str, str], int] = {}
        for page in self.page_index.pages.values():
            if page.residency == PhysicalResidency.DEAD:
                continue
            owners = tuple(
                sorted(
                    context_id
                    for context_id in page.owner_contexts
                    if context_id in self.graph.contexts
                )
            )
            for context_id in owners:
                context_bytes[context_id] = context_bytes.get(context_id, 0) + page.size_bytes
            for pair in combinations(owners, 2):
                shared_bytes[pair] = shared_bytes.get(pair, 0) + page.size_bytes
                shared_pages[pair] = shared_pages.get(pair, 0) + 1

        affinities: list[ContextPrefixAffinity] = []
        for (context_a, context_b), byte_count in sorted(shared_bytes.items()):
            record_a = self.graph.contexts[context_a]
            record_b = self.graph.contexts[context_b]
            union_bytes = (
                context_bytes[context_a] + context_bytes[context_b] - byte_count
            )
            affinities.append(
                ContextPrefixAffinity(
                    context_a=context_a,
                    context_b=context_b,
                    workflow_a=record_a.workflow_id,
                    workflow_b=record_b.workflow_id,
                    kind=self._kind(context_a, context_b),
                    shared_physical_bytes=byte_count,
                    union_physical_bytes=union_bytes,
                    shared_page_count=shared_pages[(context_a, context_b)],
                    byte_jaccard=byte_count / union_bytes,
                    observed_ts_ms=now_ms,
                )
            )
        return tuple(affinities)

    def _kind(self, context_a: str, context_b: str) -> PrefixAffinityKind:
        record_a = self.graph.contexts[context_a]
        record_b = self.graph.contexts[context_b]
        if record_a.workflow_id != record_b.workflow_id:
            return PrefixAffinityKind.CROSS_WORKFLOW
        if (
            record_a.parent_context_id == context_b
            or record_b.parent_context_id == context_a
        ):
            return PrefixAffinityKind.PARENT_CHILD
        if (
            record_a.parent_context_id is not None
            and record_a.parent_context_id == record_b.parent_context_id
        ):
            return PrefixAffinityKind.SIBLING_TEMPLATE
        definitions_a = {
            self.graph.invocations[item].agent_definition_id
            for item in record_a.invocation_ids
        }
        definitions_b = {
            self.graph.invocations[item].agent_definition_id
            for item in record_b.invocation_ids
        }
        if definitions_a & definitions_b:
            return PrefixAffinityKind.TEMPLATE
        return PrefixAffinityKind.PEER
