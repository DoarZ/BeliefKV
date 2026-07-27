from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
from struct import pack
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from beliefkv.simulator.queue_service import FrozenCounterfactualWorkload


class TokenRadixReplayError(ValueError):
    """Raised when exact token-prefix demand cannot be reconstructed."""


class TieredTokenRadixError(RuntimeError):
    """Raised when a tier or closure operation violates Radix invariants."""


@dataclass(frozen=True)
class RequestRadixDemand:
    request_id: str
    cache_hit_tokens: int
    observed_cache_hit_tokens: int
    allocator_growth_tokens: int
    unique_commit_growth_tokens: int
    duplicate_commit_tokens: int
    cache_tokens_after: int


@dataclass(frozen=True)
class TokenRadixReplayResult:
    request_demands: Mapping[str, RequestRadixDemand]
    completion_order: tuple[str, ...]
    final_unique_cache_tokens: int
    observed_hit_match_count: int
    observed_hit_mismatch_count: int
    initial_state_known: bool
    concurrent_partial_commits_modeled: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_demands",
            MappingProxyType(dict(self.request_demands)),
        )

    @property
    def valid_for_exact_prefix_demand(self) -> bool:
        return self.initial_state_known and self.concurrent_partial_commits_modeled


@dataclass
class _TrieNode:
    children: dict[int, "_TrieNode"] = field(default_factory=dict)


class TokenRadixReplay:
    """Token-granular Radix replay with SGLang's mandatory one-token miss."""

    def __init__(self, initial_paths: Iterable[Sequence[int]] = ()) -> None:
        self.root = _TrieNode()
        self.unique_tokens = 0
        for path in initial_paths:
            self.insert(tuple(int(item) for item in path))

    def match(self, prompt: Sequence[int]) -> int:
        max_prefix_tokens = max(0, len(prompt) - 1)
        node = self.root
        matched = 0
        for symbol in prompt[:max_prefix_tokens]:
            child = node.children.get(int(symbol))
            if child is None:
                break
            node = child
            matched += 1
        return matched

    def insert(self, path: Sequence[int]) -> int:
        node = self.root
        inserted = 0
        for raw_symbol in path:
            symbol = int(raw_symbol)
            if symbol < 0 or symbol >= 1 << 64:
                raise TokenRadixReplayError(
                    "token symbols must be unsigned 64-bit integers"
                )
            child = node.children.get(symbol)
            if child is None:
                child = _TrieNode()
                node.children[symbol] = child
                inserted += 1
            node = child
        self.unique_tokens += inserted
        return inserted

    def replay_completion_order(
        self,
        workload: FrozenCounterfactualWorkload,
        completion_order: Sequence[str],
        *,
        initial_state_known: bool | None = None,
        model_partial_commits: bool = False,
    ) -> TokenRadixReplayResult:
        if not workload.prefix_identity_complete:
            raise TokenRadixReplayError(
                "frozen workload lacks complete request prefix identity"
            )
        if initial_state_known is None:
            initial_state_known = workload.initial_radix_state_known
        request_by_id = {item.request_id: item for item in workload.requests}
        order = tuple(completion_order)
        if len(order) != len(set(order)) or set(order) != set(request_by_id):
            raise TokenRadixReplayError(
                "completion order must contain every request exactly once"
            )
        completed: set[str] = set()
        demands: dict[str, RequestRadixDemand] = {}
        hit_matches = 0
        for request_id in order:
            request = request_by_id[request_id]
            if not set(request.predecessor_request_ids).issubset(completed):
                raise TokenRadixReplayError(
                    f"completion order violates request dependencies: {request_id}"
                )
            hit = self.match(request.prompt_token_symbols)
            allocator_growth = (
                len(request.prompt_token_symbols)
                - hit
                + max(0, request.output_tokens - 1)
            )
            unique_growth = 0
            if model_partial_commits:
                for partial in request.partial_cache_commit_token_symbols:
                    unique_growth += self.insert(partial)
            unique_growth += self.insert(request.cache_commit_token_symbols)
            duplicate_tokens = max(
                0,
                len(request.cache_commit_token_symbols) - hit - unique_growth,
            )
            demands[request_id] = RequestRadixDemand(
                request_id=request_id,
                cache_hit_tokens=hit,
                observed_cache_hit_tokens=request.observed_cache_hit_tokens,
                allocator_growth_tokens=allocator_growth,
                unique_commit_growth_tokens=unique_growth,
                duplicate_commit_tokens=duplicate_tokens,
                cache_tokens_after=self.unique_tokens,
            )
            hit_matches += hit == request.observed_cache_hit_tokens
            completed.add(request_id)
        return TokenRadixReplayResult(
            request_demands=demands,
            completion_order=order,
            final_unique_cache_tokens=self.unique_tokens,
            observed_hit_match_count=hit_matches,
            observed_hit_mismatch_count=len(order) - hit_matches,
            initial_state_known=initial_state_known,
            concurrent_partial_commits_modeled=model_partial_commits,
        )


@dataclass(frozen=True)
class TieredRadixMatch:
    logical_hit_tokens: int
    gpu_hit_tokens: int
    cpu_only_tokens: int
    restore_tokens: int


@dataclass(frozen=True)
class TieredRadixMutation:
    operation: str
    root_path: tuple[int, ...]
    affected_tokens: int
    transfer_tokens: int
    gpu_delta_tokens: int
    cpu_delta_tokens: int
    recomputed_cpu_tokens: int = 0


@dataclass(frozen=True)
class TieredRadixBundle:
    node_id: int
    bundle_id: str
    root_path: tuple[int, ...]
    gpu_tokens: int
    cpu_tokens: int
    missing_cpu_tokens: int
    owner_context_ids: tuple[str, ...]
    locked_by_request_ids: tuple[str, ...]
    last_access_clock: int

    @property
    def actionable(self) -> bool:
        return self.gpu_tokens > 0 and not self.locked_by_request_ids


@dataclass
class _TieredTrieNode:
    node_id: int
    symbol: int | None
    parent: "_TieredTrieNode | None"
    children: dict[int, "_TieredTrieNode"] = field(default_factory=dict)
    gpu: bool = False
    cpu: bool = False
    owner_context_ids: set[str] = field(default_factory=set)
    locked_by_request_ids: set[str] = field(default_factory=set)
    last_access_clock: int = 0


@dataclass(frozen=True)
class _SubtreeStats:
    gpu_tokens: int
    cpu_tokens: int
    missing_cpu_tokens: int
    owners: frozenset[str]
    locks: frozenset[str]
    last_access_clock: int


class TieredTokenRadix:
    """Token-exact two-tier Radix state used by counterfactual replay.

    GPU residency is prefix-closed: a GPU node may never have a CPU-only
    ancestor. D2H/drop therefore acts on a complete descendant subtree, while
    H2D restores every missing ancestor on the requested prefix.
    """

    def __init__(
        self,
        *,
        bytes_per_token: int,
        validate_each_mutation: bool = True,
    ) -> None:
        if bytes_per_token <= 0:
            raise ValueError("bytes_per_token must be positive")
        self.bytes_per_token = int(bytes_per_token)
        self._validate_each_mutation = bool(validate_each_mutation)
        self._next_node_id = 1
        self.root = _TieredTrieNode(0, None, None, gpu=True)
        self._nodes_by_id: dict[int, _TieredTrieNode] = {0: self.root}
        self._context_paths: dict[str, tuple[int, ...]] = {}
        self._request_locks: dict[str, list[int]] = {}
        self._clock = 0
        self.unique_tokens = 0
        self.gpu_tokens = 0
        self.cpu_tokens = 0

    @property
    def gpu_bytes(self) -> int:
        return self.gpu_tokens * self.bytes_per_token

    @property
    def cpu_bytes(self) -> int:
        return self.cpu_tokens * self.bytes_per_token

    def match(self, prompt: Sequence[int], *, touch: bool = False) -> TieredRadixMatch:
        path = self._validate_path(prompt)
        limit = max(0, len(path) - 1)
        node = self.root
        logical_hit = 0
        gpu_hit = 0
        gpu_prefix = True
        visited: list[_TieredTrieNode] = []
        for symbol in path[:limit]:
            child = node.children.get(symbol)
            if child is None or not (child.gpu or child.cpu):
                break
            visited.append(child)
            logical_hit += 1
            gpu_prefix = gpu_prefix and child.gpu
            if gpu_prefix:
                gpu_hit += 1
            node = child
        if touch and visited:
            self._clock += 1
            for item in visited:
                item.last_access_clock = self._clock
        return TieredRadixMatch(
            logical_hit_tokens=logical_hit,
            gpu_hit_tokens=gpu_hit,
            cpu_only_tokens=logical_hit - gpu_hit,
            restore_tokens=logical_hit - gpu_hit,
        )

    def materialize_gpu(
        self,
        path: Sequence[int],
        *,
        context_id: str | None = None,
        touch: bool = True,
    ) -> TieredRadixMutation:
        normalized = self._validate_path(path)
        node = self.root
        gpu_delta = 0
        recomputed_cpu = 0
        visited: list[_TieredTrieNode] = []
        for symbol in normalized:
            child = node.children.get(symbol)
            if child is None:
                child = self._new_node(symbol, node)
                node.children[symbol] = child
            if not child.gpu:
                if not node.gpu:
                    raise TieredTokenRadixError(
                        "GPU materialization requires a GPU-resident ancestor"
                    )
                if child.cpu:
                    recomputed_cpu += 1
                child.gpu = True
                self.gpu_tokens += 1
                gpu_delta += 1
            visited.append(child)
            node = child
        if context_id is not None:
            self.bind_context(context_id, normalized)
        if touch and visited:
            self._clock += 1
            for item in visited:
                item.last_access_clock = self._clock
        self._check_after_mutation()
        return TieredRadixMutation(
            operation="materialize_gpu",
            root_path=normalized,
            affected_tokens=len(visited),
            transfer_tokens=0,
            gpu_delta_tokens=gpu_delta,
            cpu_delta_tokens=0,
            recomputed_cpu_tokens=recomputed_cpu,
        )

    def gpu_materialization_tokens(self, path: Sequence[int]) -> int:
        """Return GPU tokens needed to materialize a path without mutating it."""
        normalized = self._validate_path(path)
        node = self.root
        missing = 0
        for index, symbol in enumerate(normalized):
            child = node.children.get(symbol)
            if child is None:
                # The remaining absent suffix will be newly allocated and is
                # therefore ancestor-closed by the materialization itself.
                missing += len(normalized) - index
                break
            if not child.gpu:
                missing += 1
            node = child
        return missing

    def gpu_materialization_union_tokens(
        self, paths: Iterable[Sequence[int]]
    ) -> int:
        missing: set[tuple[int, ...]] = set()
        for raw_path in paths:
            path = self._validate_path(raw_path)
            node = self.root
            prefix: list[int] = []
            absent = False
            for symbol in path:
                prefix.append(symbol)
                child = None if absent else node.children.get(symbol)
                if child is None:
                    absent = True
                    missing.add(tuple(prefix))
                    continue
                if not child.gpu:
                    missing.add(tuple(prefix))
                node = child
        return len(missing)

    def bind_context(self, context_id: str, path: Sequence[int]) -> None:
        if not context_id:
            raise ValueError("context_id must be non-empty")
        normalized = self._validate_path(path)
        old_path = self._context_paths.get(context_id, ())
        for node in self._existing_path(old_path):
            node.owner_context_ids.discard(context_id)
        self._context_paths[context_id] = normalized
        for node in self._existing_path(normalized):
            if node.gpu or node.cpu:
                node.owner_context_ids.add(context_id)

    def lock_prefix(
        self,
        request_id: str,
        path: Sequence[int],
        token_count: int,
    ) -> None:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if request_id in self._request_locks:
            raise TieredTokenRadixError(f"request already owns a lock: {request_id}")
        normalized = self._validate_path(path)
        if token_count < 0 or token_count > len(normalized):
            raise ValueError("lock token_count is outside the path")
        nodes = self._existing_path(normalized[:token_count])
        if len(nodes) != token_count or any(not node.gpu for node in nodes):
            raise TieredTokenRadixError("active request can only lock GPU-resident prefix")
        ids = [node.node_id for node in nodes]
        for node in nodes:
            node.locked_by_request_ids.add(request_id)
        self._request_locks[request_id] = ids
        self._check_after_mutation()

    def extend_lock_prefix(
        self,
        request_id: str,
        path: Sequence[int],
        token_count: int,
    ) -> None:
        old_ids = self._request_locks.get(request_id)
        if old_ids is None:
            raise TieredTokenRadixError(f"request has no active Radix lock: {request_id}")
        normalized = self._validate_path(path)
        if token_count < len(old_ids) or token_count > len(normalized):
            raise ValueError("extended lock must contain the existing lock prefix")
        nodes = self._existing_path(normalized[:token_count])
        if len(nodes) != token_count or any(not node.gpu for node in nodes):
            raise TieredTokenRadixError("extended lock path is not GPU-resident")
        if [node.node_id for node in nodes[: len(old_ids)]] != old_ids:
            raise TieredTokenRadixError("extended lock changes the active request prefix")
        for node in nodes[len(old_ids) :]:
            node.locked_by_request_ids.add(request_id)
        self._request_locks[request_id] = [node.node_id for node in nodes]
        self._check_after_mutation()

    def gpu_extension_union_tokens(
        self,
        request_paths: Sequence[tuple[str, Sequence[int]]],
        *,
        validated: bool = False,
    ) -> int:
        """Count the union of GPU nodes needed beyond active request locks."""
        missing_existing: set[int] = set()
        virtual_edges: dict[tuple[int, int], int] = {}
        next_virtual_id = -1
        for request_id, raw_path in request_paths:
            path = tuple(raw_path) if validated else self._validate_path(raw_path)
            node, start = self._locked_prefix_endpoint(request_id, path)
            parent_id = node.node_id
            parent_is_virtual = False
            for symbol in path[start:]:
                if parent_is_virtual:
                    child = None
                else:
                    child = self._nodes_by_id[parent_id].children.get(symbol)
                if child is not None:
                    if not child.gpu:
                        missing_existing.add(child.node_id)
                    parent_id = child.node_id
                    continue
                edge = (parent_id, symbol)
                virtual_id = virtual_edges.get(edge)
                if virtual_id is None:
                    virtual_id = next_virtual_id
                    next_virtual_id -= 1
                    virtual_edges[edge] = virtual_id
                parent_id = virtual_id
                parent_is_virtual = True
        return len(missing_existing) + len(virtual_edges)

    def extend_request_gpu(
        self,
        request_id: str,
        path: Sequence[int],
        *,
        validated: bool = False,
    ) -> TieredRadixMutation:
        """Materialize and lock only the suffix beyond a request's current lock."""
        normalized = tuple(path) if validated else self._validate_path(path)
        node, start = self._locked_prefix_endpoint(request_id, normalized)
        lock_ids = self._request_locks[request_id]
        gpu_delta = 0
        recomputed_cpu = 0
        visited: list[_TieredTrieNode] = []
        for symbol in normalized[start:]:
            child = node.children.get(symbol)
            if child is None:
                child = self._new_node(symbol, node)
                node.children[symbol] = child
            if not child.gpu:
                if not node.gpu:
                    raise TieredTokenRadixError(
                        "GPU extension requires a GPU-resident ancestor"
                    )
                if child.cpu:
                    recomputed_cpu += 1
                child.gpu = True
                self.gpu_tokens += 1
                gpu_delta += 1
            child.locked_by_request_ids.add(request_id)
            lock_ids.append(child.node_id)
            visited.append(child)
            node = child
        if visited:
            self._clock += 1
            for item in visited:
                item.last_access_clock = self._clock
        self._check_after_mutation()
        return TieredRadixMutation(
            operation="extend_request_gpu",
            root_path=normalized,
            affected_tokens=len(visited),
            transfer_tokens=0,
            gpu_delta_tokens=gpu_delta,
            cpu_delta_tokens=0,
            recomputed_cpu_tokens=recomputed_cpu,
        )

    def unlock_request(self, request_id: str) -> None:
        node_ids = self._request_locks.pop(request_id, None)
        if node_ids is None:
            raise TieredTokenRadixError(f"request has no active Radix lock: {request_id}")
        for node_id in node_ids:
            node = self._nodes_by_id.get(node_id)
            if node is not None:
                node.locked_by_request_ids.discard(request_id)

    def _locked_prefix_endpoint(
        self,
        request_id: str,
        path: tuple[int, ...],
    ) -> tuple[_TieredTrieNode, int]:
        node_ids = self._request_locks.get(request_id)
        if node_ids is None:
            raise TieredTokenRadixError(
                f"request has no active Radix lock: {request_id}"
            )
        start = len(node_ids)
        if len(path) < start:
            raise TieredTokenRadixError("request path shrinks below its active lock")
        node = self.root if not node_ids else self._nodes_by_id[node_ids[-1]]
        if node_ids and node.symbol != path[start - 1]:
            raise TieredTokenRadixError("request path changes its active lock endpoint")
        if self._validate_each_mutation:
            prefix_nodes = self._existing_path(path[:start])
            if [item.node_id for item in prefix_nodes] != node_ids:
                raise TieredTokenRadixError("request path changes its active lock prefix")
        return node, start

    def restore_prefix(
        self,
        prompt: Sequence[int],
        token_count: int,
    ) -> TieredRadixMutation:
        normalized = self._validate_path(prompt)
        if token_count < 0 or token_count > max(0, len(normalized) - 1):
            raise ValueError("restore token_count violates one-token-miss boundary")
        nodes = self._existing_path(normalized[:token_count])
        if len(nodes) != token_count:
            raise TieredTokenRadixError("restore prefix is absent from all cache tiers")
        restored = 0
        for node in nodes:
            if not node.gpu:
                if not node.cpu or (node.parent is not None and not node.parent.gpu):
                    raise TieredTokenRadixError(
                        "H2D restore requires CPU source and GPU ancestor closure"
                    )
                node.gpu = True
                self.gpu_tokens += 1
                restored += 1
        if nodes:
            self._clock += 1
            for node in nodes:
                node.last_access_clock = self._clock
        self._check_after_mutation()
        return TieredRadixMutation(
            operation="restore_prefix",
            root_path=normalized[:token_count],
            affected_tokens=token_count,
            transfer_tokens=restored,
            gpu_delta_tokens=restored,
            cpu_delta_tokens=0,
        )

    def evictable_bundles(self) -> tuple[TieredRadixBundle, ...]:
        stats: dict[int, _SubtreeStats] = {}

        preorder: list[_TieredTrieNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            preorder.append(node)
            stack.extend(node.children.values())
        for node in reversed(preorder):
            gpu = int(node.gpu and node is not self.root)
            cpu = int(node.cpu and node is not self.root)
            missing_cpu = int(node.gpu and not node.cpu and node is not self.root)
            owners = set(node.owner_context_ids)
            locks = set(node.locked_by_request_ids)
            last_access = node.last_access_clock
            for child in node.children.values():
                child_stats = stats[child.node_id]
                gpu += child_stats.gpu_tokens
                cpu += child_stats.cpu_tokens
                missing_cpu += child_stats.missing_cpu_tokens
                owners.update(child_stats.owners)
                locks.update(child_stats.locks)
                last_access = max(last_access, child_stats.last_access_clock)
            result = _SubtreeStats(
                gpu,
                cpu,
                missing_cpu,
                frozenset(owners),
                frozenset(locks),
                last_access,
            )
            stats[node.node_id] = result
        candidates: list[TieredRadixBundle] = []
        for node_id, item in stats.items():
            if node_id == 0 or item.gpu_tokens == 0 or item.locks:
                continue
            node = self._nodes_by_id[node_id]
            parent = node.parent
            gpu_children = sum(
                1 for child in node.children.values() if stats[child.node_id].gpu_tokens
            )
            owner_boundary = (
                parent is None
                or parent is self.root
                or parent.owner_context_ids != node.owner_context_ids
            )
            parent_blocked = bool(
                parent is not None and stats[parent.node_id].locks
            )
            if not (owner_boundary or parent_blocked or gpu_children != 1):
                continue
            root_path = self._node_path(node)
            candidates.append(
                TieredRadixBundle(
                    node_id=node.node_id,
                    bundle_id=self._bundle_id(root_path),
                    root_path=root_path,
                    gpu_tokens=item.gpu_tokens,
                    cpu_tokens=item.cpu_tokens,
                    missing_cpu_tokens=item.missing_cpu_tokens,
                    owner_context_ids=tuple(sorted(item.owners)),
                    locked_by_request_ids=tuple(sorted(item.locks)),
                    last_access_clock=item.last_access_clock,
                )
            )
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    item.last_access_clock,
                    -item.gpu_tokens,
                    item.bundle_id,
                ),
            )
        )

    def future_use_indices(
        self, prompt_paths: Iterable[Sequence[int]]
    ) -> dict[int, int]:
        """Return the earliest future-use rank for every existing prefix node."""
        result: dict[int, int] = {}
        for index, raw_path in enumerate(prompt_paths):
            path = self._validate_path(raw_path)
            node = self.root
            for symbol in path[: max(0, len(path) - 1)]:
                child = node.children.get(symbol)
                if child is None or not (child.gpu or child.cpu):
                    break
                result.setdefault(child.node_id, index)
                node = child
        return result

    def _node_path(self, node: _TieredTrieNode) -> tuple[int, ...]:
        symbols: list[int] = []
        current = node
        while current is not self.root:
            if current.symbol is None or current.parent is None:
                raise TieredTokenRadixError("Radix node is disconnected from the root")
            symbols.append(current.symbol)
            current = current.parent
        symbols.reverse()
        return tuple(symbols)

    def prepare_host(self, root_path: Sequence[int]) -> TieredRadixMutation:
        node, normalized = self._require_subtree_root(root_path)
        nodes = self._subtree_nodes(node)
        transfer = 0
        for item in nodes:
            if item.gpu and not item.cpu:
                item.cpu = True
                self.cpu_tokens += 1
                transfer += 1
        self._check_after_mutation()
        return TieredRadixMutation(
            operation="prepare_host",
            root_path=normalized,
            affected_tokens=len(nodes),
            transfer_tokens=transfer,
            gpu_delta_tokens=0,
            cpu_delta_tokens=transfer,
        )

    def missing_cpu_tokens(self, root_path: Sequence[int]) -> int:
        node, _ = self._require_subtree_root(root_path)
        return sum(int(item.gpu and not item.cpu) for item in self._subtree_nodes(node))

    def commit_cpu(self, root_path: Sequence[int]) -> TieredRadixMutation:
        node, normalized = self._require_subtree_root(root_path)
        nodes = self._subtree_nodes(node)
        locks = {
            request_id
            for item in nodes
            for request_id in item.locked_by_request_ids
        }
        if locks:
            raise TieredTokenRadixError(
                "D2H descendant closure contains active locks: "
                + ",".join(sorted(locks))
            )
        transfer = 0
        reclaimed = 0
        for item in nodes:
            if item.gpu and not item.cpu:
                item.cpu = True
                self.cpu_tokens += 1
                transfer += 1
            if item.gpu:
                item.gpu = False
                self.gpu_tokens -= 1
                reclaimed += 1
        self._check_after_mutation()
        return TieredRadixMutation(
            operation="commit_cpu",
            root_path=normalized,
            affected_tokens=len(nodes),
            transfer_tokens=transfer,
            gpu_delta_tokens=-reclaimed,
            cpu_delta_tokens=transfer,
        )

    def drop_subtree(self, root_path: Sequence[int]) -> TieredRadixMutation:
        node, normalized = self._require_subtree_root(root_path)
        nodes = self._subtree_nodes(node)
        locks = {
            request_id
            for item in nodes
            for request_id in item.locked_by_request_ids
        }
        if locks:
            raise TieredTokenRadixError(
                "drop descendant closure contains active locks: "
                + ",".join(sorted(locks))
            )
        gpu = sum(int(item.gpu) for item in nodes)
        cpu = sum(int(item.cpu) for item in nodes)
        parent = node.parent
        if parent is None or node.symbol is None:
            raise TieredTokenRadixError("cannot drop the Radix root")
        del parent.children[node.symbol]
        for item in nodes:
            self._nodes_by_id.pop(item.node_id, None)
        self.unique_tokens -= len(nodes)
        self.gpu_tokens -= gpu
        self.cpu_tokens -= cpu
        self._check_after_mutation()
        return TieredRadixMutation(
            operation="drop_subtree",
            root_path=normalized,
            affected_tokens=len(nodes),
            transfer_tokens=0,
            gpu_delta_tokens=-gpu,
            cpu_delta_tokens=-cpu,
        )

    def _new_node(
        self, symbol: int, parent: _TieredTrieNode
    ) -> _TieredTrieNode:
        node = _TieredTrieNode(self._next_node_id, symbol, parent)
        self._next_node_id += 1
        self._nodes_by_id[node.node_id] = node
        self.unique_tokens += 1
        return node

    @staticmethod
    def _validate_path(path: Sequence[int]) -> tuple[int, ...]:
        result = tuple(int(item) for item in path)
        if any(item < 0 or item >= 1 << 64 for item in result):
            raise ValueError("token symbols must be unsigned 64-bit integers")
        return result

    def _existing_path(self, path: Sequence[int]) -> list[_TieredTrieNode]:
        node = self.root
        result: list[_TieredTrieNode] = []
        for symbol in path:
            child = node.children.get(int(symbol))
            if child is None:
                break
            result.append(child)
            node = child
        return result

    def _require_subtree_root(
        self, root_path: Sequence[int]
    ) -> tuple[_TieredTrieNode, tuple[int, ...]]:
        normalized = self._validate_path(root_path)
        if not normalized:
            raise TieredTokenRadixError("subtree operation cannot target the root")
        nodes = self._existing_path(normalized)
        if len(nodes) != len(normalized):
            raise TieredTokenRadixError("subtree root does not exist")
        return nodes[-1], normalized

    @staticmethod
    def _subtree_nodes(root: _TieredTrieNode) -> list[_TieredTrieNode]:
        result: list[_TieredTrieNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            result.append(node)
            stack.extend(node.children.values())
        return result

    @staticmethod
    def _bundle_id(path: tuple[int, ...]) -> str:
        digest = blake2b(digest_size=12, person=b"beliefkv-radix")
        for symbol in path:
            digest.update(pack("<Q", symbol))
        return f"radix:{digest.hexdigest()}"

    def assert_invariants(self) -> None:
        unique = 0
        gpu = 0
        cpu = 0
        stack = [self.root]
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            if node.node_id in seen:
                raise TieredTokenRadixError("Radix topology contains a cycle")
            seen.add(node.node_id)
            if node is not self.root:
                unique += 1
                gpu += int(node.gpu)
                cpu += int(node.cpu)
                if not (node.gpu or node.cpu):
                    raise TieredTokenRadixError("tierless Radix node remains reachable")
                if node.gpu and node.parent is not None and not node.parent.gpu:
                    raise TieredTokenRadixError(
                        "GPU residency is not ancestor-closed"
                    )
                if node.locked_by_request_ids and not node.gpu:
                    raise TieredTokenRadixError("CPU-only node retains an active lock")
            for symbol, child in node.children.items():
                if child.parent is not node or child.symbol != symbol:
                    raise TieredTokenRadixError("Radix parent/child link is inconsistent")
                stack.append(child)
        if seen != set(self._nodes_by_id):
            raise TieredTokenRadixError("Radix node index contains stale nodes")
        if (unique, gpu, cpu) != (
            self.unique_tokens,
            self.gpu_tokens,
            self.cpu_tokens,
        ):
            raise TieredTokenRadixError("tier counters disagree with Radix topology")

    def _check_after_mutation(self) -> None:
        if self._validate_each_mutation:
            self.assert_invariants()
