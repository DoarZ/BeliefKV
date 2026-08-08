from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import blake2b
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


CONTRACT_SCHEMA_VERSION = 1


class PolicyContractError(ValueError):
    """Raised when a policy crosses the common comparison contract."""


def _require_nonempty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def _require_nonnegative(value: int | float, field_name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _freeze_json(value: object, *, path: str = "metadata") -> object:
    """Copy JSON-compatible data into an immutable representation."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key")
            frozen[key] = _freeze_json(value[key], path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        _thaw_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _fingerprint(value: Mapping[str, object], *, person: bytes) -> str:
    return blake2b(
        _canonical_json(value).encode("utf-8"),
        digest_size=16,
        person=person,
    ).hexdigest()


class MetadataSource(str, Enum):
    OBSERVED = "observed"
    PREDICTED = "predicted"
    APPLICATION_PROVIDED = "application_provided"
    HINDSIGHT = "hindsight"


class MetadataMode(str, Enum):
    ONLINE = "online"
    ORACLE = "oracle"


class EvaluationMode(str, Enum):
    SHADOW = "shadow"
    REPLAY = "replay"


class ResidencyAction(str, Enum):
    KEEP = "keep"
    PREPARE_HOST = "prepare_host"
    COMMIT_CPU = "commit_cpu"
    PREFETCH_GPU = "prefetch_gpu"
    RECOMPUTE = "recompute"
    DROP = "drop"


class AdmissionAction(str, Enum):
    ADMIT = "admit"
    DEFER = "defer"
    RESTORE_THEN_ADMIT = "restore_then_admit"
    PARKED = "parked"
    PAUSE = "pause"


class UnsupportedKind(str, Enum):
    ACTION = "action"
    METADATA = "metadata"
    CAPABILITY = "capability"
    IDENTITY = "identity"


@dataclass(frozen=True)
class MetadataValue:
    source: MetadataSource
    value: object
    producer: str = "unknown"

    def __post_init__(self) -> None:
        _require_nonempty(self.producer, "metadata producer")
        object.__setattr__(self, "value", _freeze_json(self.value))

    @property
    def assumption_prefix(self) -> str:
        return self.source.value

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source.value,
            "value": _thaw_json(self.value),
            "producer": self.producer,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "MetadataValue":
        return cls(
            source=MetadataSource(str(raw["source"])),
            value=raw.get("value"),
            producer=str(raw.get("producer", "unknown")),
        )


@dataclass(frozen=True)
class MetadataRequirement:
    name: str
    allowed_sources: frozenset[MetadataSource]
    required: bool = True

    def __post_init__(self) -> None:
        _require_nonempty(self.name, "metadata requirement name")
        if not self.allowed_sources:
            raise ValueError("metadata requirement must allow at least one source")


@dataclass(frozen=True)
class RuntimeGraphSnapshot:
    snapshot_id: str
    graph_version: int
    observed_ts_ms: float
    state: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.snapshot_id, "snapshot_id")
        if self.graph_version < 0:
            raise ValueError("graph_version must be non-negative")
        _require_nonnegative(self.observed_ts_ms, "observed_ts_ms")
        frozen = _freeze_json(self.state, path="runtime_graph.state")
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "state", frozen)

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "graph_version": self.graph_version,
            "observed_ts_ms": self.observed_ts_ms,
            "state": _thaw_json(self.state),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "RuntimeGraphSnapshot":
        return cls(
            snapshot_id=str(raw["snapshot_id"]),
            graph_version=int(raw["graph_version"]),
            observed_ts_ms=float(raw["observed_ts_ms"]),
            state=_mapping(raw.get("state", {}), "runtime_graph.state"),
        )


@dataclass(frozen=True)
class RunnableInvocation:
    request_id: str
    workflow_id: str
    invocation_id: str
    context_id: str
    context_epoch: int
    submitted_ts_ms: float
    startup_bytes: int
    causal_class: str = "foreground"
    program_id: str | None = None
    predicted_remaining_decode_tokens: float | None = None
    predicted_external_wait_ms: float | None = None
    predicted_next_output_tokens: float | None = None
    prediction_support_level: str = ""
    prediction_ood_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "request_id",
            "workflow_id",
            "invocation_id",
            "context_id",
            "causal_class",
        ):
            _require_nonempty(str(getattr(self, field_name)), field_name)
        if self.context_epoch < 0:
            raise ValueError("context_epoch must be non-negative")
        _require_nonnegative(self.submitted_ts_ms, "submitted_ts_ms")
        if self.startup_bytes < 0:
            raise ValueError("startup_bytes must be non-negative")
        for field_name in (
            "predicted_remaining_decode_tokens",
            "predicted_external_wait_ms",
            "predicted_next_output_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.prediction_support_level not in {
            "",
            "exact",
            "backoff",
            "unavailable",
        }:
            raise ValueError(
                "prediction_support_level must be one of "
                "{'', 'exact', 'backoff', 'unavailable'}"
            )
        object.__setattr__(
            self,
            "prediction_ood_reasons",
            tuple(sorted(set(self.prediction_ood_reasons))),
        )
        if any(not reason for reason in self.prediction_ood_reasons):
            raise ValueError("prediction OOD reasons must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "workflow_id": self.workflow_id,
            "invocation_id": self.invocation_id,
            "context_id": self.context_id,
            "context_epoch": self.context_epoch,
            "submitted_ts_ms": self.submitted_ts_ms,
            "startup_bytes": self.startup_bytes,
            "causal_class": self.causal_class,
            "program_id": self.program_id,
            "predicted_remaining_decode_tokens": (
                self.predicted_remaining_decode_tokens
            ),
            "predicted_external_wait_ms": self.predicted_external_wait_ms,
            "predicted_next_output_tokens": self.predicted_next_output_tokens,
            "prediction_support_level": self.prediction_support_level,
            "prediction_ood_reasons": list(self.prediction_ood_reasons),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "RunnableInvocation":
        ood = raw.get("prediction_ood_reasons", ())
        return cls(
            request_id=str(raw["request_id"]),
            workflow_id=str(raw["workflow_id"]),
            invocation_id=str(raw["invocation_id"]),
            context_id=str(raw["context_id"]),
            context_epoch=int(raw["context_epoch"]),
            submitted_ts_ms=float(raw["submitted_ts_ms"]),
            startup_bytes=int(raw["startup_bytes"]),
            causal_class=str(raw.get("causal_class", "foreground")),
            program_id=(
                str(raw["program_id"]) if raw.get("program_id") is not None else None
            ),
            predicted_remaining_decode_tokens=(
                float(raw["predicted_remaining_decode_tokens"])
                if raw.get("predicted_remaining_decode_tokens") is not None
                else None
            ),
            predicted_external_wait_ms=(
                float(raw["predicted_external_wait_ms"])
                if raw.get("predicted_external_wait_ms") is not None
                else None
            ),
            predicted_next_output_tokens=(
                float(raw["predicted_next_output_tokens"])
                if raw.get("predicted_next_output_tokens") is not None
                else None
            ),
            prediction_support_level=str(
                raw.get("prediction_support_level", "")
            ),
            prediction_ood_reasons=tuple(str(item) for item in ood),
        )


@dataclass(frozen=True)
class PhysicalBundleSnapshot:
    bundle_id: str
    owner_context_ids: tuple[str, ...]
    scope: str
    physical_unique_bytes: int
    gpu_bytes: int
    cpu_bytes: int
    marginal_reclaimable_bytes: int
    closure_bytes: int
    locked_bytes: int
    residency: str
    generation_fingerprint: str
    last_access_ms: float = 0.0
    extent_ids: tuple[str, ...] = ()
    lease_kind: str = "dead"
    actionable: bool = True
    blocker_codes: tuple[str, ...] = ()
    parent_extent_id: str | None = None
    child_extent_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.bundle_id, "bundle_id")
        _require_nonempty(self.scope, "bundle scope")
        _require_nonempty(self.residency, "bundle residency")
        _require_nonempty(self.generation_fingerprint, "generation_fingerprint")
        _require_nonempty(self.lease_kind, "lease_kind")
        if len(set(self.owner_context_ids)) != len(self.owner_context_ids):
            raise ValueError("owner_context_ids must be unique")
        for field_name in (
            "physical_unique_bytes",
            "gpu_bytes",
            "cpu_bytes",
            "marginal_reclaimable_bytes",
            "closure_bytes",
            "locked_bytes",
        ):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.gpu_bytes > self.physical_unique_bytes:
            raise ValueError("gpu_bytes cannot exceed physical_unique_bytes")
        if self.cpu_bytes > self.physical_unique_bytes:
            raise ValueError("cpu_bytes cannot exceed physical_unique_bytes")
        if self.marginal_reclaimable_bytes > self.gpu_bytes:
            raise ValueError("marginal_reclaimable_bytes cannot exceed gpu_bytes")
        if self.locked_bytes > self.gpu_bytes:
            raise ValueError("locked_bytes cannot exceed gpu_bytes")
        _require_nonnegative(self.last_access_ms, "last_access_ms")
        object.__setattr__(self, "owner_context_ids", tuple(sorted(self.owner_context_ids)))
        if len(set(self.extent_ids)) != len(self.extent_ids):
            raise ValueError("extent_ids must be unique")
        for extent_id in self.extent_ids:
            _require_nonempty(extent_id, "extent_id")
        object.__setattr__(self, "extent_ids", tuple(sorted(self.extent_ids)))
        if len(set(self.blocker_codes)) != len(self.blocker_codes):
            raise ValueError("blocker_codes must be unique")
        for blocker_code in self.blocker_codes:
            _require_nonempty(blocker_code, "blocker_code")
        object.__setattr__(
            self, "blocker_codes", tuple(sorted(self.blocker_codes))
        )
        if self.parent_extent_id is not None:
            _require_nonempty(self.parent_extent_id, "parent_extent_id")
        if len(set(self.child_extent_ids)) != len(self.child_extent_ids):
            raise ValueError("child_extent_ids must be unique")
        for extent_id in self.child_extent_ids:
            _require_nonempty(extent_id, "child_extent_id")
        object.__setattr__(
            self, "child_extent_ids", tuple(sorted(self.child_extent_ids))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "owner_context_ids": list(self.owner_context_ids),
            "scope": self.scope,
            "physical_unique_bytes": self.physical_unique_bytes,
            "gpu_bytes": self.gpu_bytes,
            "cpu_bytes": self.cpu_bytes,
            "marginal_reclaimable_bytes": self.marginal_reclaimable_bytes,
            "closure_bytes": self.closure_bytes,
            "locked_bytes": self.locked_bytes,
            "residency": self.residency,
            "generation_fingerprint": self.generation_fingerprint,
            "last_access_ms": self.last_access_ms,
            "extent_ids": list(self.extent_ids),
            "lease_kind": self.lease_kind,
            "actionable": self.actionable,
            "blocker_codes": list(self.blocker_codes),
            "parent_extent_id": self.parent_extent_id,
            "child_extent_ids": list(self.child_extent_ids),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "PhysicalBundleSnapshot":
        return cls(
            bundle_id=str(raw["bundle_id"]),
            owner_context_ids=tuple(str(item) for item in raw.get("owner_context_ids", ())),
            scope=str(raw["scope"]),
            physical_unique_bytes=int(raw["physical_unique_bytes"]),
            gpu_bytes=int(raw["gpu_bytes"]),
            cpu_bytes=int(raw["cpu_bytes"]),
            marginal_reclaimable_bytes=int(raw["marginal_reclaimable_bytes"]),
            closure_bytes=int(raw["closure_bytes"]),
            locked_bytes=int(raw["locked_bytes"]),
            residency=str(raw["residency"]),
            generation_fingerprint=str(raw["generation_fingerprint"]),
            last_access_ms=float(raw.get("last_access_ms", 0.0)),
            extent_ids=tuple(
                str(item)
                for item in _sequence(raw.get("extent_ids", ()), "extent_ids")
            ),
            lease_kind=str(raw.get("lease_kind", "dead")),
            actionable=bool(raw.get("actionable", True)),
            blocker_codes=tuple(
                str(item)
                for item in _sequence(
                    raw.get("blocker_codes", ()), "blocker_codes"
                )
            ),
            parent_extent_id=(
                str(raw["parent_extent_id"])
                if raw.get("parent_extent_id") is not None
                else None
            ),
            child_extent_ids=tuple(
                str(item)
                for item in _sequence(
                    raw.get("child_extent_ids", ()), "child_extent_ids"
                )
            ),
        )


@dataclass(frozen=True)
class PhysicalKVSnapshot:
    snapshot_id: str
    topology_version: int
    allocator_version: int
    gpu_bytes: int
    cpu_bytes: int
    bundles: tuple[PhysicalBundleSnapshot, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.snapshot_id, "snapshot_id")
        if self.topology_version < 0 or self.allocator_version < 0:
            raise ValueError("physical snapshot versions must be non-negative")
        if self.gpu_bytes < 0 or self.cpu_bytes < 0:
            raise ValueError("physical snapshot bytes must be non-negative")
        bundle_ids = [item.bundle_id for item in self.bundles]
        if len(set(bundle_ids)) != len(bundle_ids):
            raise ValueError("physical bundle IDs must be unique")
        object.__setattr__(
            self, "bundles", tuple(sorted(self.bundles, key=lambda item: item.bundle_id))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "topology_version": self.topology_version,
            "allocator_version": self.allocator_version,
            "gpu_bytes": self.gpu_bytes,
            "cpu_bytes": self.cpu_bytes,
            "bundles": [item.to_dict() for item in self.bundles],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "PhysicalKVSnapshot":
        return cls(
            snapshot_id=str(raw["snapshot_id"]),
            topology_version=int(raw["topology_version"]),
            allocator_version=int(raw["allocator_version"]),
            gpu_bytes=int(raw["gpu_bytes"]),
            cpu_bytes=int(raw["cpu_bytes"]),
            bundles=tuple(
                PhysicalBundleSnapshot.from_dict(_mapping(item, "physical bundle"))
                for item in _sequence(raw.get("bundles", ()), "physical bundles")
            ),
        )


@dataclass(frozen=True)
class ResourceSnapshot:
    snapshot_id: str
    ts_ms: float
    hbm_capacity_bytes: int
    hbm_used_bytes: int
    hbm_reserved_bytes: int
    host_free_bytes: int
    urgent_d2h_bytes: int
    urgent_h2d_bytes: int
    pcie_utilization: float
    gpu_compute_utilization: float
    recent_kv_growth_bytes_per_ms: float
    h2d_service_bytes_per_ms: float
    d2h_service_bytes_per_ms: float
    transfer_setup_p50_ms: float
    unhidden_stall_per_byte: float

    def __post_init__(self) -> None:
        _require_nonempty(self.snapshot_id, "snapshot_id")
        _require_nonnegative(self.ts_ms, "resource timestamp")
        for field_name in (
            "hbm_capacity_bytes",
            "hbm_used_bytes",
            "hbm_reserved_bytes",
            "host_free_bytes",
            "urgent_d2h_bytes",
            "urgent_h2d_bytes",
            "recent_kv_growth_bytes_per_ms",
            "h2d_service_bytes_per_ms",
            "d2h_service_bytes_per_ms",
            "transfer_setup_p50_ms",
            "unhidden_stall_per_byte",
        ):
            _require_nonnegative(getattr(self, field_name), field_name)
        if self.hbm_capacity_bytes <= 0:
            raise ValueError("hbm_capacity_bytes must be positive")
        if self.hbm_used_bytes + self.hbm_reserved_bytes > self.hbm_capacity_bytes:
            raise ValueError("HBM used plus reserved bytes exceed capacity")
        for field_name in ("pcie_utilization", "gpu_compute_utilization"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and in [0, 1]")

    @property
    def hbm_available_bytes(self) -> int:
        return self.hbm_capacity_bytes - self.hbm_used_bytes - self.hbm_reserved_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ResourceSnapshot":
        return cls(**{name: raw[name] for name in cls.__dataclass_fields__})  # type: ignore[arg-type]


@dataclass(frozen=True)
class IdentityMapping:
    request_id: str
    workflow_id: str
    invocation_id: str
    context_id: str
    context_epoch: int
    program_id: str | None = None
    native_request_id: str | None = None
    native_program_id: str | None = None
    native_context_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("request_id", "workflow_id", "invocation_id", "context_id"):
            _require_nonempty(str(getattr(self, field_name)), field_name)
        if self.context_epoch < 0:
            raise ValueError("context_epoch must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "IdentityMapping":
        return cls(
            request_id=str(raw["request_id"]),
            workflow_id=str(raw["workflow_id"]),
            invocation_id=str(raw["invocation_id"]),
            context_id=str(raw["context_id"]),
            context_epoch=int(raw["context_epoch"]),
            program_id=str(raw["program_id"]) if raw.get("program_id") is not None else None,
            native_request_id=(
                str(raw["native_request_id"])
                if raw.get("native_request_id") is not None
                else None
            ),
            native_program_id=(
                str(raw["native_program_id"])
                if raw.get("native_program_id") is not None
                else None
            ),
            native_context_id=(
                str(raw["native_context_id"])
                if raw.get("native_context_id") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class CapabilityReport:
    runtime_name: str
    runtime_version: str
    supported_residency_actions: frozenset[ResidencyAction]
    execution_order_control: bool
    admission_control: bool
    transfer_dependencies: bool
    native_identity_mapping: bool
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.runtime_name, "runtime_name")
        _require_nonempty(self.runtime_version, "runtime_version")
        object.__setattr__(self, "limitations", tuple(sorted(set(self.limitations))))

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_name": self.runtime_name,
            "runtime_version": self.runtime_version,
            "supported_residency_actions": sorted(
                item.value for item in self.supported_residency_actions
            ),
            "execution_order_control": self.execution_order_control,
            "admission_control": self.admission_control,
            "transfer_dependencies": self.transfer_dependencies,
            "native_identity_mapping": self.native_identity_mapping,
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "CapabilityReport":
        return cls(
            runtime_name=str(raw["runtime_name"]),
            runtime_version=str(raw["runtime_version"]),
            supported_residency_actions=frozenset(
                ResidencyAction(str(item))
                for item in _sequence(
                    raw.get("supported_residency_actions", ()),
                    "supported_residency_actions",
                )
            ),
            execution_order_control=bool(raw.get("execution_order_control", False)),
            admission_control=bool(raw.get("admission_control", False)),
            transfer_dependencies=bool(raw.get("transfer_dependencies", False)),
            native_identity_mapping=bool(raw.get("native_identity_mapping", False)),
            limitations=tuple(
                str(item)
                for item in _sequence(raw.get("limitations", ()), "limitations")
            ),
        )


@dataclass(frozen=True)
class PolicyInput:
    runtime_graph: RuntimeGraphSnapshot
    runnable_frontier: tuple[RunnableInvocation, ...]
    physical_kv: PhysicalKVSnapshot
    resources: ResourceSnapshot
    optional_metadata: Mapping[str, MetadataValue] = field(default_factory=dict)
    identity_mappings: tuple[IdentityMapping, ...] = ()
    capabilities: CapabilityReport = field(
        default_factory=lambda: CapabilityReport(
            runtime_name="unknown",
            runtime_version="unknown",
            supported_residency_actions=frozenset(),
            execution_order_control=False,
            admission_control=False,
            transfer_dependencies=False,
            native_identity_mapping=False,
            limitations=("capability report unavailable",),
        )
    )
    metadata_mode: MetadataMode = MetadataMode.ONLINE

    def __post_init__(self) -> None:
        snapshot_ids = {
            self.runtime_graph.snapshot_id,
            self.physical_kv.snapshot_id,
            self.resources.snapshot_id,
        }
        if len(snapshot_ids) != 1:
            raise ValueError("policy input components must share one snapshot_id")
        if self.physical_kv.gpu_bytes != self.resources.hbm_used_bytes:
            raise ValueError(
                "physical KV GPU bytes must equal resource HBM used bytes"
            )
        request_ids = [item.request_id for item in self.runnable_frontier]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("runnable request IDs must be unique")
        mapping_request_ids = [item.request_id for item in self.identity_mappings]
        if len(set(mapping_request_ids)) != len(mapping_request_ids):
            raise ValueError("identity mappings must have unique request IDs")
        metadata: dict[str, MetadataValue] = {}
        for name, value in sorted(self.optional_metadata.items()):
            _require_nonempty(name, "metadata name")
            if not isinstance(value, MetadataValue):
                raise TypeError("optional_metadata values must be MetadataValue")
            metadata[name] = value
        if self.metadata_mode == MetadataMode.ONLINE and any(
            item.source == MetadataSource.HINDSIGHT for item in metadata.values()
        ):
            raise PolicyContractError(
                "an online PolicyInput cannot expose hindsight metadata"
            )
        object.__setattr__(self, "optional_metadata", MappingProxyType(metadata))
        object.__setattr__(
            self,
            "runnable_frontier",
            tuple(sorted(self.runnable_frontier, key=lambda item: item.request_id)),
        )
        object.__setattr__(
            self,
            "identity_mappings",
            tuple(sorted(self.identity_mappings, key=lambda item: item.request_id)),
        )

    @property
    def snapshot_id(self) -> str:
        return self.runtime_graph.snapshot_id

    def restricted_view(
        self,
        mode: MetadataMode,
        requirements: Sequence[MetadataRequirement],
    ) -> tuple["PolicyInput", tuple["UnsupportedRequirement", ...]]:
        """Return the only input view exposed to a reference policy."""

        selected: dict[str, MetadataValue] = {}
        unsupported: list[UnsupportedRequirement] = []
        original = self.optional_metadata
        for requirement in sorted(requirements, key=lambda item: item.name):
            value = original.get(requirement.name)
            if value is None:
                if requirement.required:
                    unsupported.append(
                        UnsupportedRequirement(
                            kind=UnsupportedKind.METADATA,
                            name=requirement.name,
                            reason="required metadata is absent",
                        )
                    )
                continue
            if value.source == MetadataSource.HINDSIGHT and mode == MetadataMode.ONLINE:
                if requirement.required:
                    unsupported.append(
                        UnsupportedRequirement(
                            kind=UnsupportedKind.METADATA,
                            name=requirement.name,
                            reason="hindsight metadata is unavailable in online mode",
                        )
                    )
                continue
            if value.source not in requirement.allowed_sources:
                if requirement.required:
                    unsupported.append(
                        UnsupportedRequirement(
                            kind=UnsupportedKind.METADATA,
                            name=requirement.name,
                            reason=(
                                f"source {value.source.value} is not accepted; expected "
                                + ",".join(
                                    sorted(item.value for item in requirement.allowed_sources)
                                )
                            ),
                        )
                    )
                continue
            selected[requirement.name] = value
        return (
            replace(self, optional_metadata=selected, metadata_mode=mode),
            tuple(sorted(unsupported, key=_unsupported_sort_key)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "runtime_graph": self.runtime_graph.to_dict(),
            "runnable_frontier": [item.to_dict() for item in self.runnable_frontier],
            "physical_kv": self.physical_kv.to_dict(),
            "resources": self.resources.to_dict(),
            "optional_metadata": {
                name: value.to_dict()
                for name, value in sorted(self.optional_metadata.items())
            },
            "identity_mappings": [item.to_dict() for item in self.identity_mappings],
            "capabilities": self.capabilities.to_dict(),
            "metadata_mode": self.metadata_mode.value,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "PolicyInput":
        _check_schema(raw)
        metadata_raw = _mapping(raw.get("optional_metadata", {}), "optional_metadata")
        return cls(
            runtime_graph=RuntimeGraphSnapshot.from_dict(
                _mapping(raw["runtime_graph"], "runtime_graph")
            ),
            runnable_frontier=tuple(
                RunnableInvocation.from_dict(_mapping(item, "runnable invocation"))
                for item in _sequence(raw.get("runnable_frontier", ()), "runnable_frontier")
            ),
            physical_kv=PhysicalKVSnapshot.from_dict(
                _mapping(raw["physical_kv"], "physical_kv")
            ),
            resources=ResourceSnapshot.from_dict(
                _mapping(raw["resources"], "resources")
            ),
            optional_metadata={
                str(name): MetadataValue.from_dict(_mapping(value, f"metadata {name}"))
                for name, value in metadata_raw.items()
            },
            identity_mappings=tuple(
                IdentityMapping.from_dict(_mapping(item, "identity mapping"))
                for item in _sequence(raw.get("identity_mappings", ()), "identity_mappings")
            ),
            capabilities=CapabilityReport.from_dict(
                _mapping(raw["capabilities"], "capabilities")
            ),
            metadata_mode=MetadataMode(str(raw.get("metadata_mode", "online"))),
        )

    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(), person=b"beliefkv-pol-in")


@dataclass(frozen=True)
class ExecutionIntent:
    ordered_request_ids: tuple[str, ...]
    selected_workflow_id: str | None
    selected_invocation_id: str | None
    mode: str
    graph_version: int
    reason: str

    def __post_init__(self) -> None:
        if len(set(self.ordered_request_ids)) != len(self.ordered_request_ids):
            raise ValueError("execution request IDs must be unique")
        _require_nonempty(self.mode, "execution mode")
        _require_nonempty(self.reason, "execution reason")
        if self.graph_version < 0:
            raise ValueError("graph_version must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "ordered_request_ids": list(self.ordered_request_ids),
            "selected_workflow_id": self.selected_workflow_id,
            "selected_invocation_id": self.selected_invocation_id,
            "mode": self.mode,
            "graph_version": self.graph_version,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ExecutionIntent":
        return cls(
            ordered_request_ids=tuple(
                str(item)
                for item in _sequence(raw.get("ordered_request_ids", ()), "ordered_request_ids")
            ),
            selected_workflow_id=(
                str(raw["selected_workflow_id"])
                if raw.get("selected_workflow_id") is not None
                else None
            ),
            selected_invocation_id=(
                str(raw["selected_invocation_id"])
                if raw.get("selected_invocation_id") is not None
                else None
            ),
            mode=str(raw["mode"]),
            graph_version=int(raw["graph_version"]),
            reason=str(raw["reason"]),
        )


@dataclass(frozen=True)
class AdmissionIntent:
    request_id: str
    action: AdmissionAction
    reserved_bytes: int
    required_bundle_ids: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        _require_nonempty(self.request_id, "admission request_id")
        _require_nonempty(self.reason, "admission reason")
        if self.reserved_bytes < 0:
            raise ValueError("reserved_bytes must be non-negative")
        if len(set(self.required_bundle_ids)) != len(self.required_bundle_ids):
            raise ValueError("required_bundle_ids must be unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "action": self.action.value,
            "reserved_bytes": self.reserved_bytes,
            "required_bundle_ids": list(self.required_bundle_ids),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "AdmissionIntent":
        return cls(
            request_id=str(raw["request_id"]),
            action=AdmissionAction(str(raw["action"])),
            reserved_bytes=int(raw["reserved_bytes"]),
            required_bundle_ids=tuple(
                str(item)
                for item in _sequence(
                    raw.get("required_bundle_ids", ()), "required_bundle_ids"
                )
            ),
            reason=str(raw["reason"]),
        )


@dataclass(frozen=True)
class ResidencyIntent:
    bundle_id: str
    action: ResidencyAction
    target_bytes: int
    deadline_ms: float
    scenario_support: frozenset[str]
    reason: str

    def __post_init__(self) -> None:
        _require_nonempty(self.bundle_id, "residency bundle_id")
        _require_nonempty(self.reason, "residency reason")
        if self.target_bytes < 0:
            raise ValueError("target_bytes must be non-negative")
        _require_nonnegative(self.deadline_ms, "residency deadline_ms")

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "action": self.action.value,
            "target_bytes": self.target_bytes,
            "deadline_ms": self.deadline_ms,
            "scenario_support": sorted(self.scenario_support),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ResidencyIntent":
        return cls(
            bundle_id=str(raw["bundle_id"]),
            action=ResidencyAction(str(raw["action"])),
            target_bytes=int(raw["target_bytes"]),
            deadline_ms=float(raw["deadline_ms"]),
            scenario_support=frozenset(
                str(item)
                for item in _sequence(raw.get("scenario_support", ()), "scenario_support")
            ),
            reason=str(raw["reason"]),
        )


@dataclass(frozen=True)
class TransferDependency:
    before_request_id: str | None
    residency_intent_index: int
    require_ack: bool

    def __post_init__(self) -> None:
        if self.residency_intent_index < 0:
            raise ValueError("residency_intent_index must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "before_request_id": self.before_request_id,
            "residency_intent_index": self.residency_intent_index,
            "require_ack": self.require_ack,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "TransferDependency":
        return cls(
            before_request_id=(
                str(raw["before_request_id"])
                if raw.get("before_request_id") is not None
                else None
            ),
            residency_intent_index=int(raw["residency_intent_index"]),
            require_ack=bool(raw["require_ack"]),
        )


@dataclass(frozen=True)
class UnsupportedRequirement:
    kind: UnsupportedKind
    name: str
    reason: str

    def __post_init__(self) -> None:
        _require_nonempty(self.name, "unsupported requirement name")
        _require_nonempty(self.reason, "unsupported requirement reason")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "name": self.name, "reason": self.reason}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "UnsupportedRequirement":
        return cls(
            kind=UnsupportedKind(str(raw["kind"])),
            name=str(raw["name"]),
            reason=str(raw["reason"]),
        )


def _unsupported_sort_key(item: UnsupportedRequirement) -> tuple[str, str, str]:
    return item.kind.value, item.name, item.reason


@dataclass(frozen=True)
class PolicyOutput:
    execution: ExecutionIntent
    admissions: tuple[AdmissionIntent, ...]
    residency: tuple[ResidencyIntent, ...]
    dependencies: tuple[TransferDependency, ...]
    policy_name: str
    metadata_assumptions: tuple[str, ...]
    policy_state_updates: Mapping[str, object] = field(default_factory=dict)
    unsupported: tuple[UnsupportedRequirement, ...] = ()
    input_snapshot_id: str = ""
    metadata_mode: MetadataMode = MetadataMode.ONLINE
    evaluation_mode: EvaluationMode = EvaluationMode.SHADOW
    shadow_only: bool = True
    decision_id: str = ""

    def __post_init__(self) -> None:
        _require_nonempty(self.policy_name, "policy_name")
        _require_nonempty(self.input_snapshot_id, "input_snapshot_id")
        if not self.shadow_only:
            raise PolicyContractError("P2.5/P3 reference outputs must be shadow-only")
        assumptions = tuple(sorted(set(self.metadata_assumptions)))
        if not assumptions:
            raise PolicyContractError("metadata_assumptions must declare observed inputs")
        for assumption in assumptions:
            _split_assumption(assumption)
        object.__setattr__(self, "metadata_assumptions", assumptions)
        state_updates = _freeze_json(
            self.policy_state_updates,
            path="policy_output.policy_state_updates",
        )
        assert isinstance(state_updates, Mapping)
        object.__setattr__(self, "policy_state_updates", state_updates)
        object.__setattr__(
            self,
            "unsupported",
            tuple(sorted(set(self.unsupported), key=_unsupported_sort_key)),
        )
        admission_ids = [item.request_id for item in self.admissions]
        if len(set(admission_ids)) != len(admission_ids):
            raise ValueError("a PolicyOutput cannot contain duplicate admission intents")
        residency_ids = [item.bundle_id for item in self.residency]
        if len(set(residency_ids)) != len(residency_ids):
            raise ValueError("a PolicyOutput cannot contain duplicate residency intents")
        for dependency in self.dependencies:
            if dependency.residency_intent_index >= len(self.residency):
                raise ValueError("transfer dependency references a missing residency intent")

    def to_dict(self, *, include_decision_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "execution": self.execution.to_dict(),
            "admissions": [item.to_dict() for item in self.admissions],
            "residency": [item.to_dict() for item in self.residency],
            "dependencies": [item.to_dict() for item in self.dependencies],
            "policy_name": self.policy_name,
            "metadata_assumptions": list(self.metadata_assumptions),
            "policy_state_updates": _thaw_json(self.policy_state_updates),
            "unsupported": [item.to_dict() for item in self.unsupported],
            "input_snapshot_id": self.input_snapshot_id,
            "metadata_mode": self.metadata_mode.value,
            "evaluation_mode": self.evaluation_mode.value,
            "shadow_only": self.shadow_only,
        }
        if include_decision_id:
            payload["decision_id"] = self.decision_id
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "PolicyOutput":
        _check_schema(raw)
        return cls(
            execution=ExecutionIntent.from_dict(
                _mapping(raw["execution"], "execution")
            ),
            admissions=tuple(
                AdmissionIntent.from_dict(_mapping(item, "admission intent"))
                for item in _sequence(raw.get("admissions", ()), "admissions")
            ),
            residency=tuple(
                ResidencyIntent.from_dict(_mapping(item, "residency intent"))
                for item in _sequence(raw.get("residency", ()), "residency")
            ),
            dependencies=tuple(
                TransferDependency.from_dict(_mapping(item, "transfer dependency"))
                for item in _sequence(raw.get("dependencies", ()), "dependencies")
            ),
            policy_name=str(raw["policy_name"]),
            metadata_assumptions=tuple(
                str(item)
                for item in _sequence(
                    raw.get("metadata_assumptions", ()), "metadata_assumptions"
                )
            ),
            policy_state_updates=_mapping(
                raw.get("policy_state_updates", {}),
                "policy_state_updates",
            ),
            unsupported=tuple(
                UnsupportedRequirement.from_dict(_mapping(item, "unsupported requirement"))
                for item in _sequence(raw.get("unsupported", ()), "unsupported")
            ),
            input_snapshot_id=str(raw["input_snapshot_id"]),
            metadata_mode=MetadataMode(str(raw.get("metadata_mode", "online"))),
            evaluation_mode=EvaluationMode(str(raw.get("evaluation_mode", "shadow"))),
            shadow_only=bool(raw.get("shadow_only", True)),
            decision_id=str(raw.get("decision_id", "")),
        )


@dataclass(frozen=True)
class PolicyDecisionRecord:
    input_snapshot_id: str
    input_fingerprint: str
    output: PolicyOutput

    def __post_init__(self) -> None:
        _require_nonempty(self.input_snapshot_id, "input_snapshot_id")
        _require_nonempty(self.input_fingerprint, "input_fingerprint")
        if self.output.input_snapshot_id != self.input_snapshot_id:
            raise PolicyContractError("decision input and output snapshot IDs differ")
        expected = _decision_id(self.input_fingerprint, self.output)
        if self.output.decision_id != expected:
            raise PolicyContractError("reference policy decision_id is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "input_snapshot_id": self.input_snapshot_id,
            "input_fingerprint": self.input_fingerprint,
            "output": self.output.to_dict(),
        }

    def to_audit_fields(self) -> dict[str, object]:
        return {"decision": self.to_dict()}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "PolicyDecisionRecord":
        _check_schema(raw)
        return cls(
            input_snapshot_id=str(raw["input_snapshot_id"]),
            input_fingerprint=str(raw["input_fingerprint"]),
            output=PolicyOutput.from_dict(_mapping(raw["output"], "policy output")),
        )

    @classmethod
    def from_audit_record(cls, raw: Mapping[str, object]) -> "PolicyDecisionRecord":
        if raw.get("event") != "reference_policy_decision":
            raise PolicyContractError("not a reference_policy_decision audit record")
        return cls.from_dict(_mapping(raw["decision"], "decision"))


@runtime_checkable
class ReferencePolicy(Protocol):
    name: str

    def metadata_requirements(
        self, mode: MetadataMode
    ) -> tuple[MetadataRequirement, ...]:
        ...

    def decide(self, policy_input: PolicyInput) -> PolicyOutput:
        ...


class ReferencePolicyAdapter:
    """Fail-closed wrapper for shadow/replay reference-policy evaluation."""

    def __init__(
        self,
        policy: ReferencePolicy,
        *,
        metadata_mode: MetadataMode = MetadataMode.ONLINE,
        evaluation_mode: EvaluationMode = EvaluationMode.SHADOW,
    ) -> None:
        if metadata_mode == MetadataMode.ORACLE and evaluation_mode != EvaluationMode.REPLAY:
            raise PolicyContractError("oracle metadata is only available in replay mode")
        self.policy = policy
        self.metadata_mode = metadata_mode
        self.evaluation_mode = evaluation_mode

    def evaluate(self, policy_input: PolicyInput) -> PolicyDecisionRecord:
        requirements = self.policy.metadata_requirements(self.metadata_mode)
        _validate_requirements(requirements)
        visible_input, missing = policy_input.restricted_view(
            self.metadata_mode, requirements
        )
        if missing:
            output = self._unsupported_output(visible_input, missing)
        else:
            output = self.policy.decide(visible_input)
            output = self._validate_and_finalize(visible_input, requirements, output)
        input_fingerprint = visible_input.fingerprint()
        output = replace(
            output,
            decision_id=_decision_id(input_fingerprint, output),
        )
        return PolicyDecisionRecord(
            input_snapshot_id=visible_input.snapshot_id,
            input_fingerprint=input_fingerprint,
            output=output,
        )

    def _unsupported_output(
        self,
        policy_input: PolicyInput,
        unsupported: tuple[UnsupportedRequirement, ...],
    ) -> PolicyOutput:
        return PolicyOutput(
            execution=ExecutionIntent(
                ordered_request_ids=(),
                selected_workflow_id=None,
                selected_invocation_id=None,
                mode="reference_unsupported",
                graph_version=policy_input.runtime_graph.graph_version,
                reason="required reference-policy metadata is unavailable",
            ),
            admissions=(),
            residency=(),
            dependencies=(),
            policy_name=self.policy.name,
            metadata_assumptions=("observed:runtime_graph",),
            unsupported=unsupported,
            input_snapshot_id=policy_input.snapshot_id,
            metadata_mode=self.metadata_mode,
            evaluation_mode=self.evaluation_mode,
            shadow_only=True,
        )

    def _validate_and_finalize(
        self,
        policy_input: PolicyInput,
        requirements: Sequence[MetadataRequirement],
        output: PolicyOutput,
    ) -> PolicyOutput:
        if output.policy_name != self.policy.name:
            raise PolicyContractError("policy output name does not match adapter policy")
        if output.input_snapshot_id != policy_input.snapshot_id:
            raise PolicyContractError("policy output refers to a different snapshot")
        if output.execution.graph_version != policy_input.runtime_graph.graph_version:
            raise PolicyContractError("execution intent uses a stale graph version")
        if not output.shadow_only:
            raise PolicyContractError("reference policy attempted an active decision")

        assumptions = set(output.metadata_assumptions)
        visible = policy_input.optional_metadata
        core = {
            "observed:runtime_graph",
            "observed:runnable_frontier",
            "observed:physical_kv",
            "observed:resources",
            "observed:capabilities",
            "observed:identity_mappings",
        }
        for assumption in assumptions:
            source, name = _split_assumption(assumption)
            if assumption in core:
                continue
            value = visible.get(name)
            if value is None or value.source != source:
                raise PolicyContractError(
                    f"policy declared unavailable metadata assumption {assumption}"
                )
            if source == MetadataSource.HINDSIGHT and self.metadata_mode != MetadataMode.ORACLE:
                raise PolicyContractError("hindsight metadata leaked into online output")
        for requirement in requirements:
            if not requirement.required or requirement.name not in visible:
                continue
            expected = (
                f"{visible[requirement.name].source.value}:{requirement.name}"
            )
            if expected not in assumptions:
                raise PolicyContractError(
                    f"policy used required metadata without declaring {expected}"
                )

        unsupported = list(output.unsupported)
        capabilities = policy_input.capabilities
        runnable_request_ids = {
            item.request_id for item in policy_input.runnable_frontier
        }
        unknown_execution = (
            set(output.execution.ordered_request_ids) - runnable_request_ids
        )
        if unknown_execution:
            raise PolicyContractError(
                "execution intent refers to unknown requests: "
                f"{sorted(unknown_execution)}"
            )
        bundle_by_id = {
            item.bundle_id: item for item in policy_input.physical_kv.bundles
        }
        for admission in output.admissions:
            if admission.request_id not in runnable_request_ids:
                raise PolicyContractError(
                    f"admission intent refers to unknown request {admission.request_id}"
                )
            missing = set(admission.required_bundle_ids) - set(bundle_by_id)
            if missing:
                raise PolicyContractError(
                    "admission intent refers to unknown bundles: "
                    f"{sorted(missing)}"
                )
        for dependency in output.dependencies:
            if dependency.before_request_id not in runnable_request_ids:
                raise PolicyContractError(
                    "transfer dependency refers to unknown request "
                    f"{dependency.before_request_id}"
                )
        if output.execution.ordered_request_ids and not capabilities.execution_order_control:
            unsupported.append(
                UnsupportedRequirement(
                    UnsupportedKind.CAPABILITY,
                    "execution_order_control",
                    "runtime cannot apply execution ordering",
                )
            )
        if output.admissions and not capabilities.admission_control:
            unsupported.append(
                UnsupportedRequirement(
                    UnsupportedKind.CAPABILITY,
                    "admission_control",
                    "runtime cannot apply admission intents",
                )
            )
        if output.dependencies and not capabilities.transfer_dependencies:
            unsupported.append(
                UnsupportedRequirement(
                    UnsupportedKind.CAPABILITY,
                    "transfer_dependencies",
                    "runtime cannot enforce transfer ACK dependencies",
                )
            )
        for intent in output.residency:
            bundle = bundle_by_id.get(intent.bundle_id)
            if bundle is None:
                raise PolicyContractError(
                    f"residency intent refers to unknown bundle {intent.bundle_id}"
                )
            if intent.action not in capabilities.supported_residency_actions:
                unsupported.append(
                    UnsupportedRequirement(
                        UnsupportedKind.ACTION,
                        f"residency:{intent.action.value}",
                        "runtime capability report does not support this action",
                    )
                )
            if intent.action != ResidencyAction.KEEP and not bundle.actionable:
                unsupported.append(
                    UnsupportedRequirement(
                        UnsupportedKind.ACTION,
                        f"bundle:{intent.bundle_id}:{intent.action.value}",
                        "physical bundle is not actionable in this snapshot; blockers="
                        + ",".join(bundle.blocker_codes),
                    )
                )
        return replace(
            output,
            unsupported=tuple(unsupported),
            metadata_mode=self.metadata_mode,
            evaluation_mode=self.evaluation_mode,
            shadow_only=True,
            decision_id="",
        )


def _decision_id(input_fingerprint: str, output: PolicyOutput) -> str:
    payload = {
        "input_fingerprint": input_fingerprint,
        "output": output.to_dict(include_decision_id=False),
    }
    return _fingerprint(payload, person=b"beliefkv-pol-out")


def _split_assumption(assumption: str) -> tuple[MetadataSource, str]:
    try:
        raw_source, name = assumption.split(":", 1)
        source = MetadataSource(raw_source)
    except (ValueError, AttributeError) as exc:
        raise PolicyContractError(
            f"invalid metadata assumption {assumption!r}; expected source:name"
        ) from exc
    _require_nonempty(name, "metadata assumption name")
    return source, name


def _validate_requirements(requirements: Sequence[MetadataRequirement]) -> None:
    names = [item.name for item in requirements]
    if len(set(names)) != len(names):
        raise PolicyContractError("metadata requirements must have unique names")


def _check_schema(raw: Mapping[str, object]) -> None:
    version = int(raw.get("schema_version", -1))
    if version != CONTRACT_SCHEMA_VERSION:
        raise PolicyContractError(
            f"unsupported reference-policy contract schema {version}"
        )


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    return value
