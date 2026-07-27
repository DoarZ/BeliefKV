from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

from beliefkv.simulator.token_radix import (
    TieredRadixBundle,
    TieredRadixMatch,
    TieredRadixMutation,
    TieredTokenRadix,
    TieredTokenRadixError,
)


class RollingPhysicalReplayError(RuntimeError):
    """Raised when no legal physical action can make a request runnable."""


class ResidencyReplayMode(str, Enum):
    REACTIVE_LRU = "reactive_lru"
    HINDSIGHT_NEXT_USE = "hindsight_next_use"


@dataclass(frozen=True)
class FutureRadixUse:
    request_id: str
    prompt_token_symbols: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("future request_id must be non-empty")
        if any(item < 0 or item >= 1 << 64 for item in self.prompt_token_symbols):
            raise ValueError("future prompt symbols must be unsigned 64-bit integers")


@dataclass(frozen=True)
class RollingPhysicalEvent:
    sequence: int
    start_ms: float
    end_ms: float
    kind: str
    reason: str
    request_id: str | None
    bundle_id: str | None
    root_path: tuple[int, ...]
    transfer_bytes: int
    hbm_bytes_before: int
    hbm_bytes_after: int
    host_bytes_before: int
    host_bytes_after: int

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.end_ms < self.start_ms:
            raise ValueError("physical event sequence or time interval is invalid")
        for value in (
            self.transfer_bytes,
            self.hbm_bytes_before,
            self.hbm_bytes_after,
            self.host_bytes_before,
            self.host_bytes_after,
        ):
            if value < 0:
                raise ValueError("physical event byte accounting must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "kind": self.kind,
            "reason": self.reason,
            "request_id": self.request_id,
            "bundle_id": self.bundle_id,
            "root_path": list(self.root_path),
            "transfer_bytes": self.transfer_bytes,
            "hbm_bytes_before": self.hbm_bytes_before,
            "hbm_bytes_after": self.hbm_bytes_after,
            "host_bytes_before": self.host_bytes_before,
            "host_bytes_after": self.host_bytes_after,
        }


@dataclass(frozen=True)
class RollingAdmission:
    request_id: str
    match: TieredRadixMatch
    ready_ms: float
    d2h_bytes: int
    h2d_bytes: int


@dataclass(frozen=True)
class RollingMaterialization:
    ready_ms: float
    unique_growth_tokens: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "unique_growth_tokens",
            MappingProxyType(dict(sorted(self.unique_growth_tokens.items()))),
        )


class RollingRadixAllocator:
    """Rebuild Radix residency and allocator state after every candidate action.

    The model starts from an empty, known Radix epoch. Cache nodes are exact
    anonymized token paths; active requests lock their GPU prefix. Capacity
    pressure triggers a fresh closure enumeration, so a stale context-level
    victim can never be reused after topology or lock state changes.
    """

    def __init__(
        self,
        *,
        hbm_capacity_bytes: int,
        hbm_fixed_bytes: int,
        host_capacity_bytes: int,
        bytes_per_token: int,
        d2h_bytes_per_ms: float,
        h2d_bytes_per_ms: float,
        transfer_setup_ms: float = 0.0,
        mode: ResidencyReplayMode = ResidencyReplayMode.REACTIVE_LRU,
    ) -> None:
        if hbm_capacity_bytes <= 0 or bytes_per_token <= 0:
            raise ValueError("HBM capacity and token size must be positive")
        if not 0 <= hbm_fixed_bytes <= hbm_capacity_bytes:
            raise ValueError("fixed HBM bytes must fit within capacity")
        if host_capacity_bytes < 0:
            raise ValueError("Host capacity must be non-negative")
        if d2h_bytes_per_ms <= 0 or h2d_bytes_per_ms <= 0:
            raise ValueError("transfer service rates must be positive")
        if transfer_setup_ms < 0:
            raise ValueError("transfer setup latency must be non-negative")
        self.hbm_capacity_bytes = int(hbm_capacity_bytes)
        self.hbm_fixed_bytes = int(hbm_fixed_bytes)
        self.host_capacity_bytes = int(host_capacity_bytes)
        self.d2h_bytes_per_ms = float(d2h_bytes_per_ms)
        self.h2d_bytes_per_ms = float(h2d_bytes_per_ms)
        self.transfer_setup_ms = float(transfer_setup_ms)
        self.mode = mode
        self.radix = TieredTokenRadix(
            bytes_per_token=bytes_per_token,
            validate_each_mutation=False,
        )
        self.events: list[RollingPhysicalEvent] = []
        self.d2h_bytes = 0
        self.h2d_bytes = 0
        self.pcie_busy_ms = 0.0
        self.hbm_peak_bytes = hbm_fixed_bytes

    @property
    def hbm_occupied_bytes(self) -> int:
        return self.hbm_fixed_bytes + self.radix.gpu_bytes

    @property
    def host_occupied_bytes(self) -> int:
        return self.radix.cpu_bytes

    @property
    def hbm_available_bytes(self) -> int:
        return self.hbm_capacity_bytes - self.hbm_occupied_bytes

    def admit(
        self,
        *,
        request_id: str,
        prompt_token_symbols: Sequence[int],
        now_ms: float,
        future_uses: Sequence[FutureRadixUse] = (),
    ) -> RollingAdmission:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if not math.isfinite(now_ms) or now_ms < 0:
            raise ValueError("admission timestamp must be finite and non-negative")
        prompt = tuple(int(item) for item in prompt_token_symbols)
        match = self.radix.match(prompt, touch=False)
        restore_bytes = match.restore_tokens * self.radix.bytes_per_token
        start_d2h = self.d2h_bytes
        start_h2d = self.h2d_bytes
        ready_ms = self._ensure_capacity(
            restore_bytes,
            now_ms=now_ms,
            request_id=request_id,
            protected_paths=(prompt[: match.logical_hit_tokens],),
            future_uses=future_uses,
            reason="admission_restore",
        )
        if match.restore_tokens:
            before_hbm = self.hbm_occupied_bytes
            before_host = self.host_occupied_bytes
            mutation = self.radix.restore_prefix(prompt, match.logical_hit_tokens)
            transfer = mutation.transfer_tokens * self.radix.bytes_per_token
            end_ms = ready_ms + self._transfer_ms(transfer, self.h2d_bytes_per_ms)
            self.h2d_bytes += transfer
            self.pcie_busy_ms += end_ms - ready_ms
            self._append_event(
                start_ms=ready_ms,
                end_ms=end_ms,
                kind="H2D_RESTORE",
                reason="admission_restore",
                request_id=request_id,
                bundle_id=None,
                mutation=mutation,
                transfer_bytes=transfer,
                hbm_before=before_hbm,
                host_before=before_host,
            )
            ready_ms = end_ms
        self.radix.lock_prefix(
            request_id,
            prompt,
            match.logical_hit_tokens,
        )
        self.radix.match(prompt, touch=True)
        self._assert_capacity()
        return RollingAdmission(
            request_id=request_id,
            match=match,
            ready_ms=ready_ms,
            d2h_bytes=self.d2h_bytes - start_d2h,
            h2d_bytes=self.h2d_bytes - start_h2d,
        )

    def materialize(
        self,
        *,
        request_id: str,
        path: Sequence[int],
        now_ms: float,
        future_uses: Sequence[FutureRadixUse] = (),
        reason: str = "request_progress",
    ) -> float:
        result = self.materialize_batch(
            ((request_id, tuple(int(item) for item in path)),),
            now_ms=now_ms,
            future_uses=future_uses,
            reason=reason,
        )
        return result.ready_ms

    def materialize_batch(
        self,
        request_paths: Sequence[tuple[str, tuple[int, ...]]],
        *,
        now_ms: float,
        future_uses: Sequence[FutureRadixUse] = (),
        reason: str = "request_progress",
    ) -> RollingMaterialization:
        if not request_paths:
            return RollingMaterialization(now_ms, {})
        request_ids = [item[0] for item in request_paths]
        if len(set(request_ids)) != len(request_ids) or any(not item for item in request_ids):
            raise ValueError("materialization batch request IDs must be unique and non-empty")
        normalized = tuple(
            (request_id, tuple(int(item) for item in path))
            for request_id, path in request_paths
        )
        required_tokens = self.radix.gpu_extension_union_tokens(
            normalized,
            validated=True,
        )
        ready_ms = self._ensure_capacity(
            required_tokens * self.radix.bytes_per_token,
            now_ms=now_ms,
            request_id=",".join(request_ids),
            protected_paths=tuple(path for _, path in normalized),
            future_uses=future_uses,
            reason=reason,
        )
        unique_growth: dict[str, int] = {}
        for request_id, path in normalized:
            before_hbm = self.hbm_occupied_bytes
            before_host = self.host_occupied_bytes
            before_unique = self.radix.unique_tokens
            mutation = self.radix.extend_request_gpu(
                request_id,
                path,
                validated=True,
            )
            unique_growth[request_id] = self.radix.unique_tokens - before_unique
            if mutation.gpu_delta_tokens or mutation.recomputed_cpu_tokens:
                self._append_event(
                    start_ms=ready_ms,
                    end_ms=ready_ms,
                    kind="MATERIALIZE",
                    reason=reason,
                    request_id=request_id,
                    bundle_id=None,
                    mutation=mutation,
                    transfer_bytes=0,
                    hbm_before=before_hbm,
                    host_before=before_host,
                )
        self._assert_capacity()
        return RollingMaterialization(ready_ms, unique_growth)

    def finish_request(
        self,
        *,
        request_id: str,
        context_id: str,
        final_path: Sequence[int],
        now_ms: float,
        future_uses: Sequence[FutureRadixUse] = (),
    ) -> float:
        ready_ms = self.materialize(
            request_id=request_id,
            path=final_path,
            now_ms=now_ms,
            future_uses=future_uses,
            reason="request_finish",
        )
        self.radix.bind_context(context_id, final_path)
        self.radix.unlock_request(request_id)
        return ready_ms

    def complete_request(
        self,
        *,
        request_id: str,
        context_id: str,
        final_path: Sequence[int],
    ) -> None:
        path = tuple(int(item) for item in final_path)
        if self.radix.gpu_materialization_tokens(path):
            raise RollingPhysicalReplayError(
                "request cannot complete before its final cache path is materialized"
            )
        self.radix.bind_context(context_id, path)
        self.radix.unlock_request(request_id)

    def prepare_host(
        self,
        root_path: Sequence[int],
        *,
        now_ms: float,
        reason: str = "shadow",
    ) -> float:
        before_hbm = self.hbm_occupied_bytes
        before_host = self.host_occupied_bytes
        required = (
            self.radix.missing_cpu_tokens(root_path)
            * self.radix.bytes_per_token
        )
        if before_host + required > self.host_capacity_bytes:
            raise RollingPhysicalReplayError("shadow copy exceeds Host capacity")
        mutation = self.radix.prepare_host(root_path)
        transfer = mutation.transfer_tokens * self.radix.bytes_per_token
        end_ms = now_ms + self._transfer_ms(transfer, self.d2h_bytes_per_ms)
        self.d2h_bytes += transfer
        self.pcie_busy_ms += end_ms - now_ms
        self._append_event(
            start_ms=now_ms,
            end_ms=end_ms,
            kind="D2H_SHADOW",
            reason=reason,
            request_id=None,
            bundle_id=None,
            mutation=mutation,
            transfer_bytes=transfer,
            hbm_before=before_hbm,
            host_before=before_host,
        )
        return end_ms

    def _ensure_capacity(
        self,
        required_bytes: int,
        *,
        now_ms: float,
        request_id: str,
        protected_paths: Sequence[tuple[int, ...]],
        future_uses: Sequence[FutureRadixUse],
        reason: str,
    ) -> float:
        if required_bytes < 0:
            raise ValueError("required capacity must be non-negative")
        cursor = now_ms
        if required_bytes <= self.hbm_available_bytes:
            return cursor
        initial_candidates = tuple(
            item
            for item in self.radix.evictable_bundles()
            if not any(
                _is_prefix(item.root_path, protected)
                for protected in protected_paths
            )
        )
        reclaimable = _disjoint_reclaimable_bytes(
            initial_candidates, self.radix.bytes_per_token
        )
        if required_bytes > self.hbm_available_bytes + reclaimable:
            shortage = required_bytes - self.hbm_available_bytes
            raise RollingPhysicalReplayError(
                "no closure-safe bundle can satisfy HBM admission; "
                f"shortage={shortage}, request={request_id}"
            )
        while required_bytes > self.hbm_available_bytes:
            candidates = tuple(
                item
                for item in self.radix.evictable_bundles()
                if not any(
                    _is_prefix(item.root_path, protected)
                    for protected in protected_paths
                )
            )
            if not candidates:
                shortage = required_bytes - self.hbm_available_bytes
                raise RollingPhysicalReplayError(
                    "no closure-safe bundle can satisfy HBM admission; "
                    f"shortage={shortage}, request={request_id}"
                )
            bundle, next_use = self._choose_bundle(candidates, future_uses)
            before_hbm = self.hbm_occupied_bytes
            before_host = self.host_occupied_bytes
            missing_host = bundle.missing_cpu_tokens * self.radix.bytes_per_token
            preserve = (
                (
                    self.mode == ResidencyReplayMode.REACTIVE_LRU
                    or next_use is not None
                )
                and before_host + missing_host <= self.host_capacity_bytes
            )
            if preserve:
                mutation = self.radix.commit_cpu(bundle.root_path)
                transfer = mutation.transfer_tokens * self.radix.bytes_per_token
                end_ms = cursor + self._transfer_ms(
                    transfer, self.d2h_bytes_per_ms
                )
                self.d2h_bytes += transfer
                self.pcie_busy_ms += end_ms - cursor
                kind = "D2H_COMMIT"
            else:
                mutation = self.radix.drop_subtree(bundle.root_path)
                transfer = 0
                end_ms = cursor
                kind = "DROP"
            self._append_event(
                start_ms=cursor,
                end_ms=end_ms,
                kind=kind,
                reason=reason,
                request_id=request_id,
                bundle_id=bundle.bundle_id,
                mutation=mutation,
                transfer_bytes=transfer,
                hbm_before=before_hbm,
                host_before=before_host,
            )
            cursor = end_ms
        return cursor

    def _choose_bundle(
        self,
        candidates: Sequence[TieredRadixBundle],
        future_uses: Sequence[FutureRadixUse],
    ) -> tuple[TieredRadixBundle, int | None]:
        if self.mode == ResidencyReplayMode.REACTIVE_LRU:
            selected = min(
                candidates,
                key=lambda item: (
                    item.last_access_clock,
                    -item.gpu_tokens,
                    item.bundle_id,
                ),
            )
            return selected, None
        next_use = self.radix.future_use_indices(
            item.prompt_token_symbols for item in future_uses
        )
        selected = min(
            candidates,
            key=lambda item: (
                0 if item.node_id not in next_use else 1,
                -next_use.get(item.node_id, 0),
                -item.gpu_tokens,
                item.bundle_id,
            ),
        )
        return selected, next_use.get(selected.node_id)

    def _append_event(
        self,
        *,
        start_ms: float,
        end_ms: float,
        kind: str,
        reason: str,
        request_id: str | None,
        bundle_id: str | None,
        mutation: TieredRadixMutation,
        transfer_bytes: int,
        hbm_before: int,
        host_before: int,
    ) -> None:
        self.events.append(
            RollingPhysicalEvent(
                sequence=len(self.events),
                start_ms=start_ms,
                end_ms=end_ms,
                kind=kind,
                reason=reason,
                request_id=request_id,
                bundle_id=bundle_id,
                root_path=mutation.root_path,
                transfer_bytes=transfer_bytes,
                hbm_bytes_before=hbm_before,
                hbm_bytes_after=self.hbm_occupied_bytes,
                host_bytes_before=host_before,
                host_bytes_after=self.host_occupied_bytes,
            )
        )
        self.hbm_peak_bytes = max(self.hbm_peak_bytes, self.hbm_occupied_bytes)

    def _assert_capacity(self) -> None:
        if self.hbm_occupied_bytes > self.hbm_capacity_bytes:
            raise RollingPhysicalReplayError("rolling allocator exceeded HBM capacity")
        if self.host_occupied_bytes > self.host_capacity_bytes:
            raise RollingPhysicalReplayError("rolling allocator exceeded Host capacity")

    def _transfer_ms(self, bytes_: int, rate: float) -> float:
        if bytes_ <= 0:
            return 0.0
        return self.transfer_setup_ms + bytes_ / rate


def _is_prefix(prefix: Sequence[int], path: Sequence[int]) -> bool:
    return len(prefix) <= len(path) and tuple(path[: len(prefix)]) == tuple(prefix)


__all__ = [
    "FutureRadixUse",
    "ResidencyReplayMode",
    "RollingAdmission",
    "RollingMaterialization",
    "RollingPhysicalEvent",
    "RollingPhysicalReplayError",
    "RollingRadixAllocator",
    "TieredTokenRadixError",
]


def _disjoint_reclaimable_bytes(
    candidates: Sequence[TieredRadixBundle], bytes_per_token: int
) -> int:
    selected_roots: list[tuple[int, ...]] = []
    tokens = 0
    for bundle in sorted(candidates, key=lambda item: (len(item.root_path), item.root_path)):
        if any(_is_prefix(root, bundle.root_path) for root in selected_roots):
            continue
        selected_roots.append(bundle.root_path)
        tokens += bundle.gpu_tokens
    return tokens * bytes_per_token
