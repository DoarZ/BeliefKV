from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable

from beliefkv.control.causal_graph import RuntimeCausalContextGraph
from beliefkv.core.events import EventConfidence, RuntimeEvent, RuntimeEventKind


class ConsumerRelation(str, Enum):
    RETURN = "return"
    MESSAGE = "message"
    BROADCAST = "broadcast"
    WORKSPACE = "workspace"
    HANDOFF = "handoff"


@dataclass(frozen=True)
class ConsumerEdge:
    producer_invocation_id: str
    consumer_invocation_id: str
    relation: ConsumerRelation
    observed: bool
    confidence: float
    last_observed_ts_ms: float
    workflow_id: str
    observation_count: int = 1

    def __post_init__(self) -> None:
        if not self.producer_invocation_id or not self.consumer_invocation_id:
            raise ValueError("consumer edge invocation IDs must be non-empty")
        if self.producer_invocation_id == self.consumer_invocation_id:
            raise ValueError("a producer cannot consume its own output")
        if not self.workflow_id:
            raise ValueError("consumer edge workflow_id must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("consumer edge confidence must be in [0, 1]")
        if self.last_observed_ts_ms < 0:
            raise ValueError("consumer edge timestamp must be non-negative")
        if self.observation_count <= 0:
            raise ValueError("consumer edge observation_count must be positive")
        if not self.observed:
            raise ValueError("predicted consumers do not belong in the observed index")


@dataclass(frozen=True)
class ConsumerIndexDelta:
    event_id: str
    changed_edges: tuple[ConsumerEdge, ...] = ()
    reactivated_invocation_ids: frozenset[str] = frozenset()
    index_version: int = 0


class ObservedDataConsumerIndex:
    """Observed result/message consumers, separate from causality and KV ownership."""

    def __init__(self, graph: RuntimeCausalContextGraph) -> None:
        self.graph = graph
        self._edges: dict[
            tuple[str, str, ConsumerRelation], ConsumerEdge
        ] = {}
        self._processed_event_ids: set[str] = set()
        self._delta_by_event_id: dict[str, ConsumerIndexDelta] = {}
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def apply(self, event: RuntimeEvent) -> ConsumerIndexDelta:
        if event.event_id in self._processed_event_ids:
            return ConsumerIndexDelta(
                event_id=event.event_id,
                index_version=self._version,
            )

        observations = self._observations(event)
        changed = tuple(
            self.observe(
                producer_invocation_id=producer_id,
                consumer_invocation_id=consumer_id,
                relation=relation,
                workflow_id=event.workflow_id,
                ts_ms=event.ts_ms,
                confidence=_confidence(event.confidence),
            )
            for producer_id, consumer_id, relation in observations
        )
        reactivated = (
            frozenset({event.invocation_id})
            if event.kind == RuntimeEventKind.REACTIVATE
            and event.invocation_id is not None
            else frozenset()
        )
        self._processed_event_ids.add(event.event_id)
        if changed or reactivated:
            self._version += 1
        delta = ConsumerIndexDelta(
            event_id=event.event_id,
            changed_edges=changed,
            reactivated_invocation_ids=reactivated,
            index_version=self._version,
        )
        self._delta_by_event_id[event.event_id] = delta
        return delta

    def apply_batch(
        self, events: Iterable[RuntimeEvent], *, atomic: bool = True
    ) -> tuple[ConsumerIndexDelta, ...]:
        if not atomic:
            return tuple(self.apply(event) for event in events)
        snapshot = (
            deepcopy(self._edges),
            set(self._processed_event_ids),
            deepcopy(self._delta_by_event_id),
            self._version,
        )
        try:
            return tuple(self.apply(event) for event in events)
        except Exception:
            (
                self._edges,
                self._processed_event_ids,
                self._delta_by_event_id,
                self._version,
            ) = snapshot
            raise

    def observe(
        self,
        *,
        producer_invocation_id: str,
        consumer_invocation_id: str,
        relation: ConsumerRelation,
        workflow_id: str,
        ts_ms: float,
        confidence: float,
    ) -> ConsumerEdge:
        producer = self.graph.invocations.get(producer_invocation_id)
        consumer = self.graph.invocations.get(consumer_invocation_id)
        if producer is None or consumer is None:
            raise ValueError("consumer edge refers to an unknown invocation")
        if producer.workflow_id != workflow_id or consumer.workflow_id != workflow_id:
            raise ValueError("consumer edge cannot cross root workflows")
        key = (producer_invocation_id, consumer_invocation_id, relation)
        previous = self._edges.get(key)
        if previous is None:
            edge = ConsumerEdge(
                producer_invocation_id=producer_invocation_id,
                consumer_invocation_id=consumer_invocation_id,
                relation=relation,
                observed=True,
                confidence=confidence,
                last_observed_ts_ms=ts_ms,
                workflow_id=workflow_id,
            )
        else:
            if ts_ms < previous.last_observed_ts_ms:
                raise ValueError("consumer observations must be timestamp-monotonic")
            edge = replace(
                previous,
                confidence=max(previous.confidence, confidence),
                last_observed_ts_ms=ts_ms,
                observation_count=previous.observation_count + 1,
            )
        self._edges[key] = edge
        return edge

    def delta_for_event(self, event_id: str) -> ConsumerIndexDelta | None:
        return self._delta_by_event_id.get(event_id)

    def consumers_for(self, producer_invocation_id: str) -> tuple[ConsumerEdge, ...]:
        return tuple(
            sorted(
                (
                    edge
                    for edge in self._edges.values()
                    if edge.producer_invocation_id == producer_invocation_id
                ),
                key=_edge_sort_key,
            )
        )

    def producers_for(self, consumer_invocation_id: str) -> tuple[ConsumerEdge, ...]:
        return tuple(
            sorted(
                (
                    edge
                    for edge in self._edges.values()
                    if edge.consumer_invocation_id == consumer_invocation_id
                ),
                key=_edge_sort_key,
            )
        )

    def consumer_fanout(self, producer_invocation_id: str) -> int:
        return len(
            {
                edge.consumer_invocation_id
                for edge in self.consumers_for(producer_invocation_id)
            }
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "index_version": self._version,
            "edges": [
                {
                    "workflow_id": edge.workflow_id,
                    "producer_invocation_id": edge.producer_invocation_id,
                    "consumer_invocation_id": edge.consumer_invocation_id,
                    "relation": edge.relation.value,
                    "observed": edge.observed,
                    "confidence": edge.confidence,
                    "last_observed_ts_ms": edge.last_observed_ts_ms,
                    "observation_count": edge.observation_count,
                }
                for edge in sorted(self._edges.values(), key=_edge_sort_key)
            ],
        }

    def _observations(
        self, event: RuntimeEvent
    ) -> tuple[tuple[str, str, ConsumerRelation], ...]:
        observations: set[tuple[str, str, ConsumerRelation]] = set()
        if (
            event.kind in {RuntimeEventKind.MESSAGE, RuntimeEventKind.HANDOFF}
            and event.invocation_id is not None
            and event.target_invocation_id is not None
        ):
            relation = (
                ConsumerRelation.MESSAGE
                if event.kind == RuntimeEventKind.MESSAGE
                else ConsumerRelation.HANDOFF
            )
            observations.add(
                (event.invocation_id, event.target_invocation_id, relation)
            )
        elif event.kind == RuntimeEventKind.RETURN and event.invocation_id is not None:
            invocation = self.graph.invocations.get(event.invocation_id)
            target_id = event.return_target_id or (
                invocation.return_target_id if invocation is not None else None
            )
            if target_id is not None and target_id != event.invocation_id:
                observations.add(
                    (event.invocation_id, target_id, ConsumerRelation.RETURN)
                )

        explicit = event.attributes.get("consumer_invocation_ids", ())
        if isinstance(explicit, str):
            explicit = (explicit,)
        if isinstance(explicit, (list, tuple)) and event.invocation_id is not None:
            relation_raw = str(
                event.attributes.get("consumer_relation", ConsumerRelation.BROADCAST.value)
            )
            try:
                relation = ConsumerRelation(relation_raw)
            except ValueError:
                relation = ConsumerRelation.BROADCAST
            for consumer_id in explicit:
                if (
                    isinstance(consumer_id, str)
                    and consumer_id != event.invocation_id
                    and consumer_id in self.graph.invocations
                ):
                    observations.add((event.invocation_id, consumer_id, relation))
        return tuple(sorted(observations, key=lambda item: (item[0], item[1], item[2].value)))


def _confidence(confidence: EventConfidence) -> float:
    return {
        EventConfidence.OBSERVED_EXACT: 1.0,
        EventConfidence.DECLARED_RUNTIME: 0.9,
        EventConfidence.INFERRED: 0.5,
    }[confidence]


def _edge_sort_key(edge: ConsumerEdge) -> tuple[str, str, str, str]:
    return (
        edge.workflow_id,
        edge.producer_invocation_id,
        edge.consumer_invocation_id,
        edge.relation.value,
    )
