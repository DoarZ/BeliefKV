import json
import subprocess
import sys
import tempfile
import unittest
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from beliefkv.control.controller import BeliefKVController
from beliefkv.core.config import BeliefKVConfig
from beliefkv.core.events import (
    ExecutionMode,
    RelationType,
    RuntimeEvent,
    RuntimeEventKind,
)
from beliefkv.policy.admission import AdmissionDecision, AdmissionRequest
from beliefkv.runtime.protocol import (
    CommandAck,
    CommandKind,
    CommandStatus,
    ControlCommand,
    PageHandle,
    PhysicalBundleIntent,
    PhysicalPageAction,
    ResolvedCommand,
    ResolvedPageAction,
    TransferBlockerCode,
    TransferDirection,
    TransferTelemetry,
)
from beliefkv.runtime.sglang_adapter import (
    BASE_SGLANG_VERSION,
    BeliefKVRequestMetadata,
    SGLangSourceContract,
    assert_supported_sglang_version,
)
from beliefkv.runtime.sglang_v052rc1 import (
    EmbeddedSGLangRuntime,
    HiCacheNodeCommandBackend,
    SGLangBackendError,
    SGLangNodeRegistry,
)


class _Node:
    def __init__(self, node_id=1):
        self.id = node_id
        self.value = [1, 2, 3, 4]
        self.host_value = None
        self.lock_ref = 0
        self.loading = False
        self.host_ref_counter = 0
        self.children = {}
        self.parent = None

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None


class _CacheController:
    def __init__(self, allocator):
        self.mem_pool_device_allocator = allocator

    def evict_host(self, _indices):
        return 4


class _TreeCache:
    def __init__(self):
        self.root_node = _Node(0)
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.token_to_kv_pool_allocator = _Allocator(1_000_000)
        self.cache_controller = _CacheController(
            self.token_to_kv_pool_allocator
        )
        self.evictable_tokens = 0
        self.load_ready_calls = 0
        self.load_back_threshold = 10
        self.load_back_calls = []
        self.callback_errors = []

    def evictable_size(self):
        return self.evictable_tokens

    def write_backup(self, node):
        node.host_value = [10, 11, 12, 13]
        self.ongoing_write_through[node.id] = node
        return 4

    def check_hicache_events(self):
        for node in self.ongoing_write_through.values():
            node.lock_ref = 0
        self.ongoing_write_through.clear()
        if self.load_ready_calls:
            for ancestor, node in self.ongoing_load_back.values():
                current = node
                while current is not ancestor:
                    current.loading = False
                    current = current.parent
            self.ongoing_load_back.clear()

    def _evict_backuped(self, node):
        self.token_to_kv_pool_allocator.available_tokens += len(node.value)
        node.value = None
        return 4

    def _evict_regular(self, node):
        self.token_to_kv_pool_allocator.available_tokens += len(node.value)
        node.value = None
        return 4

    def load_back(self, node, *, force=False, allow_eviction=True):
        self.load_back_calls.append(
            {
                "node_id": node.id,
                "force": force,
                "allow_eviction": allow_eviction,
            }
        )
        leaf = node
        chain = []
        while node is not self.root_node and node.evicted:
            chain.insert(0, node)
            node = node.parent
        token_count = sum(len(item.host_value) for item in chain)
        if not force and token_count < self.load_back_threshold:
            return None
        if token_count > self.token_to_kv_pool_allocator.available_tokens:
            return None
        self.token_to_kv_pool_allocator.available_tokens -= token_count
        loaded = []
        for item in chain:
            item.value = [20] * len(item.host_value)
            item.loading = True
            loaded.extend(item.value)
        self.ongoing_load_back[leaf.id] = (node, leaf)
        return loaded

    def ready_to_load_host_cache(self):
        self.load_ready_calls += 1
        return self.load_ready_calls

    def take_beliefkv_callback_errors(self):
        errors = self.callback_errors
        self.callback_errors = []
        return errors


class _LockPropagatingTreeCache(_TreeCache):
    def __init__(self):
        super().__init__()
        self.write_order = []
        self.evict_order = []

    def write_backup(self, node):
        self.write_order.append(node.id)
        current = node
        while current is not None and current is not self.root_node:
            current.lock_ref += 1
            current = current.parent
        return super().write_backup(node)

    def check_hicache_events(self):
        for node in tuple(self.ongoing_write_through.values()):
            current = node
            while current is not None and current is not self.root_node:
                current.lock_ref -= 1
                current = current.parent
        self.ongoing_write_through.clear()
        if self.load_ready_calls:
            for _, node in self.ongoing_load_back.values():
                node.loading = False
            self.ongoing_load_back.clear()

    def _evict_backuped(self, node):
        self.evict_order.append(node.id)
        return super()._evict_backuped(node)


class _AdmissionRecorder:
    def __init__(self):
        self.cancelled = []
        self.enqueued = []

    def enqueue(self, request):
        self.enqueued.append(request)

    def cancel(self, request_id):
        self.cancelled.append(request_id)


class _Allocator:
    def __init__(self, available_tokens):
        self.available_tokens = available_tokens

    def available_size(self):
        return self.available_tokens


class _HostAllocator(_Allocator):
    def __init__(self, size, available_tokens, size_per_token):
        super().__init__(available_tokens)
        self.size = size
        self.size_per_token = size_per_token


class _Sender:
    def __init__(self):
        self.messages = []

    def send_pyobj(self, value):
        self.messages.append(value)


class _AbortScheduler:
    def __init__(self):
        self.runtime = None
        self.requests = []
        self.send_to_tokenizer = _Sender()

    def abort_request(self, request):
        self.requests.append(request)
        self.runtime.on_abort_request(request)


class _CloseRecorder:
    def __init__(self):
        self.close_count = 0
        self.events = []

    def close(self):
        self.close_count += 1

    def emit(self, event, ts_ms):
        self.events.append((event, ts_ms))


class _AuditRecorder:
    def __init__(self):
        self.events = []

    def emit(self, event, ts_ms, **fields):
        self.events.append((event, ts_ms, fields))


class _EventBatchRecorder:
    def __init__(self):
        self.events = []

    def emit_batch(self, events):
        self.events.extend(events)


class _TransferTickBridge:
    def __init__(self, transfer, *, admission=None, acks=(), telemetry=()):
        self.transfer = transfer
        self.admission = admission
        self.acks = list(acks)
        self.telemetry = list(telemetry)

    def drain_acks(self):
        result = self.acks
        self.acks = []
        return result

    def drain_transfer_telemetry(self):
        result = self.telemetry
        self.telemetry = []
        return result

    def scheduler_step(self, now_ms, *, drain_acks):
        self.drain_acks_argument = drain_acks
        return SimpleNamespace(
            now_ms=now_ms,
            admission=self.admission,
            transfer=self.transfer,
            local_acks=(),
        )


@dataclass
class _AbortRequest:
    rid: str = ""
    abort_all: bool = False


def resolved(kind, handle, action):
    command = ControlCommand(
        command_id=f"cmd-{kind.value}",
        kind=kind,
        created_ts_ms=1,
        context_id="ctx",
        context_epoch=0,
        target_bytes=400,
    )
    page_action = ResolvedPageAction(handle, action, 400)
    return ResolvedCommand(command, (page_action,), 400, "resolved")


def resolved_bundle(kind, actions, *, closure_handles=None):
    page_actions = tuple(
        ResolvedPageAction(handle, action, size_bytes)
        for handle, action, size_bytes in actions
    )
    handles = tuple(
        sorted(
            closure_handles
            if closure_handles is not None
            else (item.handle for item in page_actions)
        )
    )
    intent = PhysicalBundleIntent(
        bundle_id="bundle-test",
        closure_handles=handles,
        page_actions=page_actions,
        generation_fingerprint="bundle-generation-test",
        closure_bytes=sum(item.size_bytes for item in page_actions),
        expected_reclaimable_bytes=(
            sum(item.size_bytes for item in page_actions)
            if kind == CommandKind.OFFLOAD_CONTEXT
            else 0
        ),
    )
    command = ControlCommand(
        command_id=f"cmd-bundle-{kind.value}",
        kind=kind,
        created_ts_ms=1,
        context_id="ctx",
        context_epoch=0,
        target_bytes=intent.closure_bytes,
        physical_bundle=intent,
    )
    return ResolvedCommand(
        command,
        page_actions,
        intent.closure_bytes,
        "physical_bundle_resolved",
        closure_fingerprint=intent.generation_fingerprint,
    )


class SGLangBackendTest(unittest.TestCase):
    def test_tree_sync_defers_generation_changes_until_transfer_ack(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime._tree_dirty = True
        runtime.controller = SimpleNamespace(inflight_command_ids=("transfer-1",))

        runtime.sync_tree()

        self.assertTrue(runtime._tree_dirty)

    def test_resource_snapshot_uses_real_allocator_state_and_marks_missing_utilization(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(
            hbm_capacity_bytes=1600,
            host_capacity_bytes=3200,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=16,
        )
        runtime._now_ms = lambda: 50.0
        runtime._last_resource_telemetry_ms = None
        runtime.audit = _AuditRecorder()
        runtime.scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=_Allocator(25),
            max_total_num_tokens=100,
        )
        runtime.tree_cache = SimpleNamespace(
            token_to_kv_pool_host=_HostAllocator(200, 150, 16)
        )
        runtime.controller = SimpleNamespace(
            page_index=SimpleNamespace(gpu_bytes=700, cpu_bytes=600),
            inflight_command_ids=(),
            signals=SimpleNamespace(
                pcie_utilization=0.0,
                gpu_compute_utilization=0.0,
            ),
            _engine_request_count=2,
            _running_request_count=1,
        )

        runtime._emit_resource_snapshot(force=True)

        event, ts_ms, fields = runtime.audit.events[0]
        self.assertEqual((event, ts_ms), ("resource_snapshot", 50.0))
        self.assertEqual(fields["hbm_capacity_bytes"], 1600)
        self.assertEqual(fields["configured_hbm_capacity_bytes"], 1600)
        self.assertEqual(fields["hbm_used_bytes"], 1200)
        self.assertEqual(fields["host_used_bytes"], 800)
        self.assertIsNone(fields["pcie_utilization"])
        self.assertIsNone(fields["copy_engine_utilization"])
        self.assertIsNone(fields["gpu_compute_utilization"])

    def test_resource_snapshot_exposes_configured_allocator_capacity_mismatch(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            host_capacity_bytes=3200,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=16,
        )
        runtime._now_ms = lambda: 50.0
        runtime._last_resource_telemetry_ms = None
        runtime.audit = _AuditRecorder()
        runtime.scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=_Allocator(25),
            max_total_num_tokens=100,
        )
        runtime.tree_cache = SimpleNamespace(
            token_to_kv_pool_host=_HostAllocator(200, 150, 16)
        )
        runtime.controller = SimpleNamespace(
            page_index=SimpleNamespace(gpu_bytes=700, cpu_bytes=600),
            inflight_command_ids=(),
            signals=SimpleNamespace(
                pcie_utilization=0.0,
                gpu_compute_utilization=0.0,
            ),
            _engine_request_count=2,
            _running_request_count=1,
        )

        runtime._emit_resource_snapshot(force=True)

        fields = runtime.audit.events[0][2]
        self.assertEqual(fields["hbm_capacity_bytes"], 1600)
        self.assertEqual(fields["configured_hbm_capacity_bytes"], 2000)
        self.assertEqual(fields["hbm_used_bytes"], 1200)

    def test_native_admission_capacity_excludes_prefix_that_will_be_locked(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.tree_cache = _TreeCache()
        runtime.tree_cache.evictable_tokens = 20
        parent = _Node(1)
        child = _Node(2)
        parent.parent = runtime.tree_cache.root_node
        child.parent = parent
        parent.children[child.id] = child
        req = SimpleNamespace(last_node=child)

        result = runtime._native_admission_capacity_tokens(req, _Allocator(5))

        # 5 free + 20 evictable - 8 tokens on the request's unlocked path.
        self.assertEqual(result, (17, 5, 20, 8))

        parent.lock_ref = 1
        child.lock_ref = 1
        locked_result = runtime._native_admission_capacity_tokens(req, _Allocator(5))
        self.assertEqual(locked_result, (25, 5, 20, 0))

    def test_late_runtime_event_is_committed_at_workflow_watermark(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(runtime_event_max_lateness_ms=100.0)
        runtime.controller = BeliefKVController(runtime.config)
        runtime.audit = _AuditRecorder()
        runtime.event_log = _EventBatchRecorder()
        runtime._now_ms = lambda: 250.0
        start = RuntimeEvent("start", 100.0, RuntimeEventKind.WORKFLOW_START, "wf")
        create = RuntimeEvent(
            "create",
            200.0,
            RuntimeEventKind.INVOCATION_CREATE,
            "wf",
            invocation_id="inv",
            context_id="ctx",
            context_epoch=0,
            agent_definition_id="agent",
            agent_instance_id="agent-1",
        )
        late = RuntimeEvent(
            "late",
            150.0,
            RuntimeEventKind.CONTEXT_ADVANCE,
            "wf",
            invocation_id="inv",
            context_id="ctx",
            context_epoch=0,
        )

        runtime._process_events((start, create))
        runtime._process_events((late,))

        committed = runtime.event_log.events[-1]
        self.assertEqual(committed.ts_ms, 200.0)
        self.assertEqual(committed.attributes["beliefkv_source_ts_ms"], 150.0)
        self.assertEqual(committed.attributes["beliefkv_late_by_ms"], 50.0)
        self.assertEqual(runtime.controller.graph.timestamp_watermark("wf"), 200.0)
        event, _, fields = runtime.audit.events[-1]
        self.assertEqual(event, "runtime_event_time_adjusted")
        self.assertEqual(fields["late_by_ms"], 50.0)

    def test_runtime_event_beyond_lateness_limit_is_rejected(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(runtime_event_max_lateness_ms=10.0)
        runtime.controller = BeliefKVController(runtime.config)
        runtime.audit = _AuditRecorder()
        runtime.event_log = _EventBatchRecorder()
        runtime._now_ms = lambda: 250.0
        runtime._process_events(
            (RuntimeEvent("start", 100.0, RuntimeEventKind.WORKFLOW_START, "wf"),)
        )

        with self.assertRaisesRegex(SGLangBackendError, "50.000 ms late"):
            runtime._process_events(
                (RuntimeEvent("late", 50.0, RuntimeEventKind.WORKFLOW_END, "wf"),)
            )

    def test_deferred_request_reserves_only_uncached_prompt_tokens(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        admission = _AdmissionRecorder()
        runtime.controller = SimpleNamespace(
            admission=admission,
            submit_request=admission.enqueue,
        )
        runtime.tree_cache = object()
        runtime.config = BeliefKVConfig(kv_bytes_per_token=16)
        runtime._deferred_requests = {}
        runtime._admitted_request_ids = set()
        runtime._request_metadata_by_id = {}
        runtime._ensure_causal_identity = lambda metadata: None
        runtime._now_ms = lambda: 42.0
        runtime.audit = _AuditRecorder()
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)

        def initialize_prefix(req_tree_cache):
            self.assertIs(req_tree_cache, runtime.tree_cache)
            request.fill_ids = request.origin_input_ids
            request.prefix_indices = list(range(90))

        request = SimpleNamespace(
            rid="request-1",
            beliefkv_metadata=metadata,
            sampling_params=SimpleNamespace(max_new_tokens=10),
            origin_input_ids=list(range(100)),
            init_next_round_input=initialize_prefix,
        )

        self.assertTrue(runtime.defer_request(request))

        reserved = admission.enqueued[0]
        self.assertEqual(reserved.uncached_prompt_tokens, 10)
        self.assertEqual(reserved.estimated_incremental_bytes, 320)
        _, _, fields = runtime.audit.events[0]
        self.assertEqual(fields["estimated_cache_hit_tokens"], 90)
        self.assertEqual(fields["uncached_prompt_tokens"], 10)

    def test_scheduler_step_audits_resolved_transfer_bytes(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        transfer = resolved(
            CommandKind.OFFLOAD_CONTEXT,
            PageHandle(1, 0),
            PhysicalPageAction.START_D2H,
        )
        runtime.event_server = None
        runtime.bridge = _TransferTickBridge(transfer)
        runtime.sync_tree = lambda: None
        runtime._report_allocator_usage = lambda: None
        runtime._now_ms = lambda: 42.0
        runtime._last_admission_audit = None
        runtime.audit = _AuditRecorder()

        runtime.scheduler_step()

        self.assertFalse(runtime.bridge.drain_acks_argument)
        event, ts_ms, fields = runtime.audit.events[0]
        self.assertEqual((event, ts_ms), ("transfer_dispatched", 42.0))
        self.assertEqual(fields["selected_bytes"], transfer.resolved_bytes)
        self.assertEqual(fields["action_counts"], {"start_d2h": 1})
        self.assertEqual(len(runtime._scheduler_timing_samples), 1)

    def test_scheduler_step_audits_ack_before_performance_telemetry(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        ack = CommandAck(
            command_id="offload-1",
            status=CommandStatus.COMPLETED,
            completed_ts_ms=40.0,
            actual_bytes=400,
        )
        telemetry = TransferTelemetry(
            command_id="offload-1",
            submit_ts_ms=10.0,
            start_ts_ms=20.0,
            first_layer_ready_ts_ms=None,
            complete_ts_ms=40.0,
            compute_wait_ms=None,
            actual_bytes=400,
            closure_bytes=400,
            merged_operation_count=0,
            direction=TransferDirection.D2H,
            source_tier="gpu",
            target_tier="host",
            status=CommandStatus.COMPLETED,
        )
        runtime.event_server = None
        runtime.bridge = _TransferTickBridge(
            None, acks=(ack,), telemetry=(telemetry,)
        )
        runtime.sync_tree = lambda: None
        runtime._report_allocator_usage = lambda: None
        runtime._now_ms = lambda: 42.0
        runtime._last_admission_audit = None
        runtime.audit = _AuditRecorder()

        runtime.scheduler_step()

        self.assertEqual(
            [event for event, _, _ in runtime.audit.events],
            ["transfer_acknowledged", "transfer_telemetry"],
        )
        self.assertEqual(runtime.audit.events[1][1], 42.0)
        self.assertEqual(runtime.audit.events[1][2]["complete_ts_ms"], 40.0)
        timing = runtime._scheduler_timing_samples[-1]
        self.assertEqual(timing[2], 1)
        self.assertLessEqual(timing[1], timing[0])

    def test_admission_waits_for_same_context_h2d_ack_and_rematches_prefix(self):
        transfer = resolved_bundle(
            CommandKind.PREFETCH_CONTEXT,
            ((PageHandle(1, 0), PhysicalPageAction.START_H2D, 400),),
        )
        decision = AdmissionDecision(
            request_id="request-1",
            admitted=True,
            reason="workflow_fair_causal_frontier",
            reserved_bytes=800,
            required_bytes=800,
        )
        bridge = _TransferTickBridge(transfer, admission=decision)
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.event_server = None
        runtime.bridge = bridge
        runtime.backend = SimpleNamespace()
        runtime.controller = SimpleNamespace(actual_hbm_used_bytes=0)
        runtime.tree_cache = object()
        runtime.sync_tree = lambda: None
        runtime._report_allocator_usage = lambda: None
        runtime._now_ms = lambda: 42.0
        runtime._last_admission_audit = None
        runtime._stalled_command_audited = set()
        runtime.audit = _AuditRecorder()
        runtime._deferred_requests = {}
        runtime._admitted_request_ids = set()
        runtime._request_metadata_by_id = {}
        runtime._held_h2d_admissions = {}
        runtime._h2d_context_by_command = {}
        runtime._pending_h2d_contexts = set()
        added = []
        runtime.scheduler = SimpleNamespace(
            _add_admitted_beliefkv_request=added.append
        )
        refresh_count = 0

        def refresh(_tree_cache):
            nonlocal refresh_count
            refresh_count += 1
            request.prefix_indices = [1, 2, 3, 4]

        request = SimpleNamespace(
            rid="request-1",
            beliefkv_metadata=BeliefKVRequestMetadata("wf", "inv", "ctx", 0),
            prefix_indices=[],
            init_next_round_input=refresh,
        )
        runtime._deferred_requests[request.rid] = request
        runtime._request_metadata_by_id[request.rid] = request.beliefkv_metadata

        runtime.scheduler_step()

        self.assertEqual(added, [])
        self.assertEqual(refresh_count, 0)
        self.assertIn(request.rid, runtime._held_h2d_admissions)

        bridge.transfer = None
        bridge.admission = None
        bridge.acks = [
            CommandAck(
                command_id=transfer.command.command_id,
                status=CommandStatus.COMPLETED,
                completed_ts_ms=43.0,
                actual_bytes=400,
            )
        ]
        runtime.scheduler_step()

        self.assertEqual(refresh_count, 1)
        self.assertEqual(added, [request])
        self.assertNotIn(request.rid, runtime._held_h2d_admissions)
        self.assertNotIn("ctx", runtime._pending_h2d_contexts)

    def test_close_emits_controller_timing_summary(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime._closed = False
        runtime._now_ms = lambda: 42.0
        runtime.event_server = None
        runtime.event_log = None
        runtime.transfer_telemetry_log = _CloseRecorder()
        runtime.audit = _AuditRecorder()
        runtime.audit.close = lambda: None
        runtime._scheduler_timing_samples = deque(
            ((10.0, 0.2, 1), (20.0, 0.4, 2)), maxlen=65_536
        )

        runtime.close()

        event, _, fields = runtime.audit.events[0]
        self.assertEqual(event, "controller_timing_summary")
        self.assertEqual(fields["telemetry_event_count"], 3)
        self.assertAlmostEqual(fields["telemetry_event_overhead_ratio_p99"], 0.02)

    def test_embedded_runtime_close_is_idempotent(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime._closed = False
        runtime._now_ms = lambda: 42.0
        runtime.event_server = _CloseRecorder()
        runtime.event_log = _CloseRecorder()
        runtime.audit = _CloseRecorder()

        runtime.close()
        runtime.close()

        self.assertTrue(runtime._closed)
        self.assertIsNone(runtime.event_server)
        self.assertIsNone(runtime.event_log)
        self.assertEqual(runtime.audit.events, [("runtime_shutdown", 42.0)])
        self.assertEqual(runtime.audit.close_count, 1)

    def test_shadow_cancel_does_not_assume_submitted_dma_is_preemptible(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        node = _Node()
        handle = registry.register(node)
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)
        command = resolved(CommandKind.SHADOW_CONTEXT, handle, PhysicalPageAction.START_D2H)
        submission = backend.submit(command)
        backend.cancel(command.command.command_id)
        acks = backend.poll_acks()
        self.assertEqual(submission.started_handles, (handle,))
        self.assertEqual(acks[0].status.value, "completed")
        self.assertIn("nonpreemptible", acks[0].reason)
        self.assertFalse(node.evicted)
        self.assertTrue(node.backuped)

    def test_extent_split_after_d2h_records_dma_but_rejects_residency_commit(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        node = _Node()
        node.key = [1, 2, 3, 4]
        handle = registry.register(node)
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)

        backend.submit(
            resolved(
                CommandKind.OFFLOAD_CONTEXT,
                handle,
                PhysicalPageAction.START_D2H,
            )
        )
        node.key = node.key[2:]
        node.value = node.value[2:]

        acks = backend.poll_acks()
        telemetry = backend.poll_transfer_telemetry()

        self.assertEqual(acks[0].status, CommandStatus.REJECTED)
        self.assertEqual(acks[0].actual_bytes, 0)
        self.assertIn("Radix extent mutated", acks[0].reason)
        self.assertEqual(
            {item.code for item in acks[0].blockers},
            {TransferBlockerCode.EXTENT_MUTATED},
        )
        self.assertIsNotNone(node.value)
        self.assertEqual(telemetry[0].actual_bytes, 400)
        self.assertEqual(telemetry[0].status, CommandStatus.REJECTED)

    def test_urgent_d2h_evicts_only_after_backup_ack(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        node = _Node()
        handle = registry.register(node)
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)
        command = resolved(CommandKind.OFFLOAD_CONTEXT, handle, PhysicalPageAction.START_D2H)
        backend.submit(command)
        self.assertFalse(node.evicted)
        ack = backend.poll_acks()[0]
        telemetry = backend.poll_transfer_telemetry()[0]
        self.assertTrue(node.evicted)
        self.assertEqual(ack.actual_bytes, 400)
        self.assertEqual(telemetry.command_id, ack.command_id)
        self.assertEqual(telemetry.direction, TransferDirection.D2H)
        self.assertEqual(telemetry.actual_bytes, 400)
        self.assertEqual(telemetry.closure_bytes, 400)
        self.assertIsNone(telemetry.first_layer_ready_ts_ms)

    def test_atomic_bundle_preflight_has_no_side_effect_when_child_is_locked(self):
        tree = _LockPropagatingTreeCache()
        registry = SGLangNodeRegistry()
        parent = _Node(1)
        child = _Node(2)
        parent.parent = tree.root_node
        child.parent = parent
        parent.children["child"] = child
        child.lock_ref = 1
        parent_handle = registry.register(parent)
        child_handle = registry.register(child)
        command = resolved_bundle(
            CommandKind.OFFLOAD_CONTEXT,
            (
                (parent_handle, PhysicalPageAction.START_D2H, 400),
                (child_handle, PhysicalPageAction.START_D2H, 400),
            ),
        )
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)

        submission = backend.submit(command)
        ack = backend.poll_acks()[0]

        self.assertEqual(submission.started_handles, ())
        self.assertEqual(ack.status, CommandStatus.REJECTED)
        self.assertEqual(ack.actual_bytes, 0)
        self.assertEqual(
            {item.code for item in ack.blockers},
            {TransferBlockerCode.NODE_LOCKED},
        )
        self.assertEqual(tree.write_order, [])
        self.assertFalse(parent.backuped)
        self.assertFalse(child.backuped)
        self.assertFalse(parent.evicted)
        self.assertFalse(child.evicted)

    def test_atomic_d2h_submits_shallow_first_and_evicts_deep_first(self):
        tree = _LockPropagatingTreeCache()
        registry = SGLangNodeRegistry()
        parent = _Node(1)
        child = _Node(2)
        parent.parent = tree.root_node
        child.parent = parent
        parent.children["child"] = child
        parent_handle = registry.register(parent)
        child_handle = registry.register(child)
        command = resolved_bundle(
            CommandKind.OFFLOAD_CONTEXT,
            (
                (child_handle, PhysicalPageAction.START_D2H, 400),
                (parent_handle, PhysicalPageAction.START_D2H, 400),
            ),
        )
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)

        submission = backend.submit(command)
        self.assertEqual(tree.write_order, [parent.id, child.id])
        self.assertFalse(parent.evicted)
        self.assertFalse(child.evicted)

        ack = backend.poll_acks()[0]
        telemetry = backend.poll_transfer_telemetry()[0]

        self.assertEqual(
            submission.started_handles, tuple(sorted((parent_handle, child_handle)))
        )
        self.assertEqual(ack.status, CommandStatus.COMPLETED)
        self.assertEqual(ack.actual_bytes, 800)
        self.assertEqual(tree.evict_order, [child.id, parent.id])
        self.assertTrue(parent.evicted)
        self.assertTrue(child.evicted)
        self.assertEqual(telemetry.actual_bytes, 800)
        self.assertEqual(telemetry.status, CommandStatus.COMPLETED)

    def test_atomic_h2d_native_closure_failure_has_no_partial_ancestor(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        parent = _Node(1)
        child = _Node(2)
        parent.parent = tree.root_node
        child.parent = parent
        parent.children["child"] = child
        for node in (parent, child):
            node.value = None
            node.host_value = [10, 11, 12, 13]
        parent_handle = registry.register(parent)
        child_handle = registry.register(child)
        tree.load_back = lambda _node, **_kwargs: None
        command = resolved_bundle(
            CommandKind.PREFETCH_CONTEXT,
            (
                (child_handle, PhysicalPageAction.START_H2D, 400),
                (parent_handle, PhysicalPageAction.START_H2D, 400),
            ),
        )
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)

        submission = backend.submit(command)
        ack = backend.poll_acks()[0]
        telemetry = backend.poll_transfer_telemetry()[0]

        self.assertEqual(submission.started_handles, ())
        self.assertEqual(ack.status, CommandStatus.REJECTED)
        self.assertEqual(ack.actual_bytes, 0)
        self.assertTrue(parent.evicted)
        self.assertTrue(child.evicted)
        self.assertEqual(telemetry.actual_bytes, 0)
        self.assertEqual(telemetry.status, CommandStatus.REJECTED)

    def test_atomic_h2d_uses_one_native_operation_for_ancestor_chain(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        parent = _Node(1)
        child = _Node(2)
        parent.parent = tree.root_node
        child.parent = parent
        parent.children["child"] = child
        for node in (parent, child):
            node.value = None
            node.host_value = [10, 11, 12, 13]
        parent_handle = registry.register(parent)
        child_handle = registry.register(child)
        load_back = tree.load_back
        loaded_node_ids = []

        def traced_load(node, **kwargs):
            loaded_node_ids.append(node.id)
            return load_back(node, **kwargs)

        tree.load_back = traced_load
        command = resolved_bundle(
            CommandKind.PREFETCH_CONTEXT,
            (
                (parent_handle, PhysicalPageAction.START_H2D, 400),
                (child_handle, PhysicalPageAction.START_H2D, 400),
            ),
        )
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)

        submission = backend.submit(command)
        ack = backend.poll_acks()[0]

        self.assertEqual(loaded_node_ids, [child.id])
        self.assertEqual(
            submission.started_handles, tuple(sorted((parent_handle, child_handle)))
        )
        self.assertEqual(ack.status, CommandStatus.COMPLETED)
        self.assertEqual(ack.actual_bytes, 800)
        self.assertFalse(parent.evicted)
        self.assertFalse(child.evicted)

    def test_atomic_h2d_forces_tiny_closure_without_native_eviction(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        node = _Node(1)
        node.parent = tree.root_node
        node.value = None
        node.host_value = [10, 11, 12, 13]
        handle = registry.register(node)
        command = resolved_bundle(
            CommandKind.PREFETCH_CONTEXT,
            ((handle, PhysicalPageAction.START_H2D, 400),),
        )
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)

        submission = backend.submit(command)
        ack = backend.poll_acks()[0]

        self.assertEqual(submission.started_handles, (handle,))
        self.assertEqual(ack.status, CommandStatus.COMPLETED)
        self.assertEqual(
            tree.load_back_calls,
            [{"node_id": 1, "force": True, "allow_eviction": False}],
        )

    def test_atomic_h2d_rejects_before_load_when_allocator_cannot_fit_closure(self):
        tree = _TreeCache()
        tree.token_to_kv_pool_allocator.available_tokens = 3
        registry = SGLangNodeRegistry()
        node = _Node(1)
        node.parent = tree.root_node
        node.value = None
        node.host_value = [10, 11, 12, 13]
        handle = registry.register(node)
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)

        submission = backend.submit(
            resolved_bundle(
                CommandKind.PREFETCH_CONTEXT,
                ((handle, PhysicalPageAction.START_H2D, 400),),
            )
        )
        ack = backend.poll_acks()[0]

        self.assertEqual(submission.started_handles, ())
        self.assertEqual(tree.load_back_calls, [])
        self.assertEqual(ack.status, CommandStatus.REJECTED)
        self.assertEqual(
            {item.code for item in ack.blockers},
            {TransferBlockerCode.DEVICE_CAPACITY},
        )

    def test_h2d_rejects_divergent_tree_and_controller_allocators(self):
        tree = _TreeCache()
        tree.cache_controller.mem_pool_device_allocator = _Allocator(1_000_000)
        registry = SGLangNodeRegistry()
        node = _Node(1)
        node.parent = tree.root_node
        node.value = None
        node.host_value = [10, 11, 12, 13]
        handle = registry.register(node)
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)

        submission = backend.submit(
            resolved_bundle(
                CommandKind.PREFETCH_CONTEXT,
                ((handle, PhysicalPageAction.START_H2D, 400),),
            )
        )
        ack = backend.poll_acks()[0]

        self.assertEqual(submission.started_handles, ())
        self.assertEqual(ack.status, CommandStatus.REJECTED)
        self.assertEqual(
            {item.code for item in ack.blockers},
            {TransferBlockerCode.UNKNOWN_BACKEND},
        )

    def test_h2d_rejects_context_that_is_already_engine_visible(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        node = _Node(1)
        node.parent = tree.root_node
        node.value = None
        node.host_value = [10, 11, 12, 13]
        handle = registry.register(node)
        backend = HiCacheNodeCommandBackend(
            tree,
            registry,
            now_ms=lambda: 2,
            h2d_context_is_busy=lambda context_id: context_id == "ctx",
        )

        submission = backend.submit(
            resolved_bundle(
                CommandKind.PREFETCH_CONTEXT,
                ((handle, PhysicalPageAction.START_H2D, 400),),
            )
        )
        ack = backend.poll_acks()[0]

        self.assertEqual(submission.started_handles, ())
        self.assertEqual(ack.status, CommandStatus.REJECTED)
        self.assertEqual(
            {item.code for item in ack.blockers},
            {TransferBlockerCode.ENGINE_BUSY},
        )

    def test_h2d_callback_failure_rolls_back_atomic_gpu_residency(self):
        tree = _TreeCache()
        initial_available = tree.token_to_kv_pool_allocator.available_size()
        registry = SGLangNodeRegistry()
        node = _Node(1)
        node.parent = tree.root_node
        node.value = None
        node.host_value = [10, 11, 12, 13]
        handle = registry.register(node)
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)
        command = resolved_bundle(
            CommandKind.PREFETCH_CONTEXT,
            ((handle, PhysicalPageAction.START_H2D, 400),),
        )

        backend.submit(command)
        tree.callback_errors.append(
            {
                "direction": "h2d",
                "operation_id": node.id,
                "error": "RuntimeError: malformed completion chain",
            }
        )
        ack = backend.poll_acks()[0]

        self.assertEqual(ack.status, CommandStatus.REJECTED)
        self.assertEqual(
            {item.code for item in ack.blockers},
            {TransferBlockerCode.UNKNOWN_BACKEND},
        )
        self.assertTrue(node.evicted)
        self.assertEqual(
            tree.token_to_kv_pool_allocator.available_size(), initial_available
        )
        self.assertEqual(len(backend.poll_callback_errors()), 1)

    def test_allocator_reconciliation_claims_live_radix_indices(self):
        import torch

        class AllocatorWithPages:
            page_size = 1

            def __init__(self):
                self.free_pages = torch.tensor([1, 2, 4, 5])
                self.release_pages = torch.tensor([3])

            def available_size(self):
                return len(self.free_pages) + len(self.release_pages)

        tree = _TreeCache()
        node = _Node(1)
        node.parent = tree.root_node
        node.value = torch.tensor([2, 3])
        tree.root_node.children["node"] = node
        tree.evictable_tokens = 2
        allocator = AllocatorWithPages()
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.tree_cache = tree
        runtime.scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            max_total_num_tokens=5,
        )
        runtime.audit = _AuditRecorder()
        runtime._now_ms = lambda: 10.0

        runtime._ensure_allocator_radix_consistency(reason="test")

        self.assertEqual(allocator.free_pages.tolist(), [1, 4, 5])
        self.assertEqual(allocator.release_pages.tolist(), [])
        self.assertEqual(allocator.available_size() + tree.evictable_size(), 5)
        event, _, fields = runtime.audit.events[0]
        self.assertEqual(event, "allocator_radix_resynchronized")
        self.assertEqual(fields["overlap_tokens"], 2)

    def test_atomic_h2d_rejects_topology_change_in_non_action_gpu_anchor(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        anchor = _Node(1)
        child = _Node(2)
        anchor.parent = tree.root_node
        child.parent = anchor
        anchor.children["child"] = child
        child.value = None
        child.host_value = [10, 11, 12, 13]
        anchor_handle = registry.register(anchor)
        child_handle = registry.register(child)
        command = resolved_bundle(
            CommandKind.PREFETCH_CONTEXT,
            ((child_handle, PhysicalPageAction.START_H2D, 400),),
            closure_handles=(anchor_handle, child_handle),
        )
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)

        submission = backend.submit(command)
        sibling = _Node(3)
        sibling.parent = anchor
        anchor.children["sibling"] = sibling
        ack = backend.poll_acks()[0]

        self.assertEqual(submission.started_handles, (child_handle,))
        self.assertEqual(ack.status, CommandStatus.REJECTED)
        self.assertEqual(ack.actual_bytes, 0)
        self.assertTrue(child.evicted)
        self.assertEqual(
            {item.code for item in ack.blockers},
            {TransferBlockerCode.EXTENT_MUTATED},
        )

    def test_pinned_hicache_capabilities_do_not_claim_unobservable_features(self):
        tree = _TreeCache()
        tree.token_to_kv_pool_host = SimpleNamespace(layout="page_first")
        backend = HiCacheNodeCommandBackend(tree, SGLangNodeRegistry())

        capabilities = backend.capabilities

        self.assertFalse(capabilities.operation_merge)
        self.assertFalse(capabilities.layer_completion_events)
        self.assertTrue(capabilities.page_first_host_layout)
        self.assertTrue(capabilities.proactive_load_trigger)
        self.assertEqual(capabilities.max_inflight_operations, 1)
        self.assertEqual(capabilities.physical_unit, "node_extent")

    def test_registry_rejects_handle_after_cache_reset(self):
        registry = SGLangNodeRegistry()
        handle = registry.register(_Node())
        registry.reset()
        with self.assertRaisesRegex(RuntimeError, "stale"):
            registry.resolve(handle)

    def test_h2d_rejects_an_unselected_evicted_ancestor(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        parent = _Node(1)
        parent.parent = tree.root_node
        parent.value = None
        parent.host_value = [1]
        child = _Node(2)
        child.parent = parent
        child.value = None
        child.host_value = [2]
        handle = registry.register(child)
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)
        command = resolved(
            CommandKind.PREFETCH_CONTEXT,
            handle,
            PhysicalPageAction.START_H2D,
        )
        submission = backend.submit(command)
        self.assertEqual(submission.started_handles, ())
        self.assertEqual(tree.load_ready_calls, 0)
        ack = backend.poll_acks()[0]
        telemetry = backend.poll_transfer_telemetry()[0]
        self.assertEqual(ack.status.value, "rejected")
        self.assertEqual(
            {item.code for item in ack.blockers},
            {TransferBlockerCode.ANCESTOR_CLOSURE},
        )
        self.assertEqual(telemetry.status.value, "rejected")
        self.assertEqual(telemetry.direction, TransferDirection.H2D)
        self.assertIsNone(telemetry.start_ts_ms)
        self.assertEqual(telemetry.actual_bytes, 0)
        self.assertEqual(telemetry.closure_bytes, 400)

    def test_proactive_h2d_explicitly_starts_hicache_load_queue(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        node = _Node(1)
        node.parent = tree.root_node
        node.value = None
        node.host_value = [1, 2, 3, 4]
        handle = registry.register(node)
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)
        command = resolved(
            CommandKind.PREFETCH_CONTEXT,
            handle,
            PhysicalPageAction.START_H2D,
        )

        submission = backend.submit(command)

        self.assertEqual(submission.started_handles, (handle,))
        self.assertEqual(tree.load_ready_calls, 1)
        ack = backend.poll_acks()[0]
        self.assertEqual(ack.status.value, "completed")
        self.assertEqual(ack.actual_bytes, 400)
        self.assertFalse(node.loading)

    def test_h2d_device_allocation_failure_is_structured(self):
        tree = _TreeCache()
        tree.load_back = lambda _node, **_kwargs: None
        registry = SGLangNodeRegistry()
        node = _Node(1)
        node.parent = tree.root_node
        node.value = None
        node.host_value = [1, 2, 3, 4]
        handle = registry.register(node)
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)

        submission = backend.submit(
            resolved(
                CommandKind.PREFETCH_CONTEXT,
                handle,
                PhysicalPageAction.START_H2D,
            )
        )
        ack = backend.poll_acks()[0]

        self.assertEqual(submission.started_handles, ())
        self.assertEqual(ack.status, CommandStatus.REJECTED)
        self.assertEqual(
            {item.code for item in ack.blockers},
            {TransferBlockerCode.DEVICE_CAPACITY},
        )

    def test_d2h_rejects_a_gpu_node_below_an_evicted_ancestor(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        parent = _Node(1)
        parent.parent = tree.root_node
        parent.value = None
        parent.host_value = [1]
        child = _Node(2)
        child.parent = parent
        parent.children["child"] = child
        handle = registry.register(child)
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)
        command = resolved(
            CommandKind.OFFLOAD_CONTEXT,
            handle,
            PhysicalPageAction.START_D2H,
        )

        submission = backend.submit(command)
        ack = backend.poll_acks()[0]

        self.assertEqual(submission.started_handles, ())
        self.assertEqual(ack.status.value, "rejected")
        self.assertIn("evicted Radix ancestor", ack.reason)
        self.assertEqual(
            {item.code for item in ack.blockers},
            {TransferBlockerCode.ANCESTOR_CLOSURE},
        )
        self.assertFalse(child.backuped)

    def test_d2h_completion_refuses_to_evict_below_gpu_descendant(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        parent = _Node(1)
        parent.parent = tree.root_node
        child = _Node(2)
        child.parent = parent
        parent.children["child"] = child
        handle = registry.register(parent)
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)
        command = resolved(
            CommandKind.OFFLOAD_CONTEXT,
            handle,
            PhysicalPageAction.START_D2H,
        )

        submission = backend.submit(command)
        ack = backend.poll_acks()[0]
        telemetry = backend.poll_transfer_telemetry()[0]

        self.assertEqual(submission.started_handles, (handle,))
        self.assertEqual(ack.status.value, "rejected")
        self.assertEqual(ack.actual_bytes, 0)
        self.assertEqual(telemetry.actual_bytes, 400)
        self.assertEqual(telemetry.status.value, "rejected")
        self.assertIn("descendants off device", ack.reason)
        self.assertEqual(
            {item.code for item in ack.blockers},
            {TransferBlockerCode.DESCENDANT_CLOSURE},
        )
        self.assertFalse(parent.evicted)
        self.assertTrue(parent.backuped)

    def test_drop_dual_clean_node_keeps_the_cpu_radix_extent(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        node = _Node(1)
        node.parent = tree.root_node
        node.host_value = [10, 11, 12, 13]
        tree.root_node.children["node"] = node
        handle = registry.register(node)
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)
        command = resolved(CommandKind.DROP_UNOWNED, handle, PhysicalPageAction.DROP)

        submission = backend.submit(command)
        ack = backend.poll_acks()[0]

        self.assertEqual(submission.started_handles, (handle,))
        self.assertEqual(ack.status.value, "completed")
        self.assertTrue(node.evicted)
        self.assertTrue(node.backuped)
        self.assertIs(tree.root_node.children["node"], node)

    def test_cache_reset_aborts_pending_backend_commands(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        node = _Node()
        handle = registry.register(node)
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)
        command = resolved(
            CommandKind.SHADOW_CONTEXT,
            handle,
            PhysicalPageAction.START_D2H,
        )
        backend.submit(command)
        acks = backend.abort_all(reason="cache_reset")
        self.assertEqual(acks[0].status.value, "cancelled")
        self.assertEqual(acks[0].actual_bytes, 0)
        self.assertEqual(backend.poll_acks(), [])

    def test_abort_bridge_removes_hidden_admission_requests(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime._deferred_requests = {
            "req-deferred": SimpleNamespace(rid="req-deferred")
        }
        runtime._admitted_request_ids = {"req-admitted"}
        runtime._request_metadata_by_id = {}
        admission = _AdmissionRecorder()
        runtime.controller = SimpleNamespace(admission=admission)
        sender = _Sender()
        runtime.scheduler = SimpleNamespace(send_to_tokenizer=sender)

        removed = runtime.on_abort_request(_AbortRequest(rid="req-"))
        self.assertEqual(removed, 1)
        self.assertEqual(runtime._deferred_requests, {})
        self.assertEqual(runtime._admitted_request_ids, set())
        self.assertEqual(
            admission.cancelled, ["req-deferred", "req-admitted"]
        )
        self.assertEqual(sender.messages[0].rid, "req-deferred")

    def test_return_cancels_a_deferred_request_before_next_admission_tick(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig()
        runtime.controller = BeliefKVController(runtime.config)
        runtime.audit = _AuditRecorder()
        runtime.event_log = _EventBatchRecorder()
        runtime._now_ms = lambda: 30.0
        runtime._deferred_requests = {}
        runtime._admitted_request_ids = set()
        runtime._active_request_ids = set()
        runtime._terminal_cancelled_request_ids = set()
        runtime._request_metadata_by_id = {}
        scheduler = _AbortScheduler()
        scheduler.runtime = runtime
        runtime.scheduler = scheduler
        runtime._process_events(
            (
                RuntimeEvent(
                    "start", 10.0, RuntimeEventKind.WORKFLOW_START, "wf"
                ),
                RuntimeEvent(
                    "create",
                    11.0,
                    RuntimeEventKind.INVOCATION_CREATE,
                    "wf",
                    invocation_id="inv",
                    context_id="ctx",
                    context_epoch=0,
                ),
            )
        )
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        request = SimpleNamespace(rid="request", beliefkv_metadata=metadata)
        runtime.controller.submit_request(
            AdmissionRequest(
                request_id="request",
                workflow_id="wf",
                invocation_id="inv",
                context_id="ctx",
                context_epoch=0,
                submitted_ts_ms=12.0,
                uncached_prompt_tokens=1,
                expected_output_tokens=1,
                kv_bytes_per_token=16,
            )
        )
        runtime._deferred_requests[request.rid] = request
        runtime._request_metadata_by_id[request.rid] = metadata

        runtime._process_events(
            (
                RuntimeEvent(
                    "return",
                    20.0,
                    RuntimeEventKind.RETURN,
                    "wf",
                    invocation_id="inv",
                    context_id="ctx",
                ),
            )
        )

        self.assertEqual(runtime._deferred_requests, {})
        self.assertEqual(runtime.controller.admission.pending_count, 0)
        self.assertEqual([item.rid for item in scheduler.requests], ["request"])
        self.assertEqual(
            [item.rid for item in scheduler.send_to_tokenizer.messages],
            ["request"],
        )
        cancelled = [
            fields
            for event, _, fields in runtime.audit.events
            if event == "terminal_request_cancelled"
        ]
        self.assertEqual(cancelled[0]["phase"], "deferred")

    def test_request_arriving_after_return_is_rejected_without_admission(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.controller = BeliefKVController()
        runtime.audit = _AuditRecorder()
        runtime._now_ms = lambda: 30.0
        runtime._deferred_requests = {}
        runtime._admitted_request_ids = set()
        runtime._request_metadata_by_id = {}
        runtime.scheduler = SimpleNamespace(send_to_tokenizer=_Sender())
        runtime.controller.process_runtime_events(
            (
                RuntimeEvent(
                    "start", 10.0, RuntimeEventKind.WORKFLOW_START, "wf"
                ),
                RuntimeEvent(
                    "create",
                    11.0,
                    RuntimeEventKind.INVOCATION_CREATE,
                    "wf",
                    invocation_id="inv",
                    context_id="ctx",
                    context_epoch=0,
                ),
                RuntimeEvent(
                    "return",
                    12.0,
                    RuntimeEventKind.RETURN,
                    "wf",
                    invocation_id="inv",
                ),
            )
        )
        request = SimpleNamespace(
            rid="late-request",
            beliefkv_metadata=BeliefKVRequestMetadata("wf", "inv", "ctx", 0),
        )

        self.assertTrue(runtime.defer_request(request))
        self.assertEqual(runtime.controller.admission.pending_count, 0)
        self.assertEqual(
            runtime.scheduler.send_to_tokenizer.messages[0].rid,
            "late-request",
        )
        self.assertEqual(runtime.audit.events[0][0], "terminal_request_rejected")

    def test_waiting_queue_is_root_workflow_fair_and_preserves_untagged_slots(self):
        controller = BeliefKVController(
            BeliefKVConfig(hbm_capacity_bytes=1000, reserve_hbm_bytes=100)
        )
        for workflow_id in ("wf-a", "wf-b"):
            controller.process_runtime_event(
                RuntimeEvent(
                    event_id=f"{workflow_id}-start",
                    ts_ms=0,
                    kind=RuntimeEventKind.WORKFLOW_START,
                    workflow_id=workflow_id,
                )
            )
            for suffix in ("1", "2"):
                controller.process_runtime_event(
                    RuntimeEvent(
                        event_id=f"{workflow_id}-{suffix}",
                        ts_ms=float(suffix),
                        kind=RuntimeEventKind.INVOCATION_CREATE,
                        workflow_id=workflow_id,
                        invocation_id=f"{workflow_id}-inv-{suffix}",
                        context_id=f"{workflow_id}-ctx-{suffix}",
                        context_epoch=0,
                    )
                )
        controller.fairness.charge_service("wf-a", 100)
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.controller = controller
        runtime.config = controller.config
        untagged = SimpleNamespace(rid="untagged", beliefkv_metadata=None)

        def req(workflow_id, suffix):
            metadata = BeliefKVRequestMetadata(
                workflow_id,
                f"{workflow_id}-inv-{suffix}",
                f"{workflow_id}-ctx-{suffix}",
                0,
            )
            return SimpleNamespace(
                rid=f"{workflow_id}-{suffix}", beliefkv_metadata=metadata
            )

        queue = [untagged, req("wf-a", "1"), req("wf-a", "2"), req("wf-b", "1"), req("wf-b", "2")]
        runtime.reorder_waiting_queue(queue)
        self.assertIs(queue[0], untagged)
        self.assertEqual(
            [item.rid for item in queue[1:]],
            ["wf-b-1", "wf-a-1", "wf-b-2", "wf-a-2"],
        )

    def test_batch_time_is_charged_proportionally_to_root_workflows(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.controller = BeliefKVController()
        runtime._last_batch_selected_ms = 10.0
        runtime._last_batch_workflow_counts = {"wf-a": 1, "wf-b": 3}
        runtime._charge_previous_batch(30.0)
        self.assertEqual(
            runtime.controller.fairness.accounts["wf-a"].attained_service_ms,
            5.0,
        )
        self.assertEqual(
            runtime.controller.fairness.accounts["wf-b"].attained_service_ms,
            15.0,
        )

    def test_existing_context_epoch_advances_before_admission(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.controller = BeliefKVController()
        runtime._identity_metadata = {}
        runtime._linked_invocations = set()
        runtime._event_sequence = 0
        runtime._now_ms = lambda: 10.0
        runtime.event_log = None
        runtime.audit = _AuditRecorder()
        first = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        runtime._ensure_causal_identity(first)

        resumed = BeliefKVRequestMetadata(
            "wf", "inv", "ctx", 3, context_mode="resume"
        )
        runtime._ensure_causal_identity(resumed)

        self.assertEqual(runtime.controller.graph.contexts["ctx"].epoch, 3)
        self.assertEqual(runtime.controller.page_index.context_epoch("ctx"), 3)
        self.assertIn(
            "context_epoch_advanced",
            [event for event, _, _ in runtime.audit.events],
        )
        stale = BeliefKVRequestMetadata("wf", "inv", "ctx", 2)
        with self.assertRaisesRegex(RuntimeError, "stale context epoch"):
            runtime._ensure_causal_identity(stale)

    def test_metadata_does_not_duplicate_runtime_declared_spawn(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.controller = BeliefKVController()
        runtime._identity_metadata = {}
        runtime._linked_invocations = set()
        runtime._event_sequence = 0
        runtime._now_ms = lambda: 10.0
        runtime.event_log = None
        runtime.audit = _AuditRecorder()
        root = BeliefKVRequestMetadata("wf", "root", "ctx-root", 0)
        runtime._ensure_causal_identity(root)
        runtime._process_events(
            (
                RuntimeEvent(
                    event_id="child-create",
                    ts_ms=10.0,
                    kind=RuntimeEventKind.INVOCATION_CREATE,
                    workflow_id="wf",
                    invocation_id="child",
                    context_id="ctx-child",
                    context_epoch=0,
                    parent_invocation_id="root",
                    parent_context_id="ctx-root",
                    relation_type=RelationType.SPAWN,
                ),
                RuntimeEvent(
                    event_id="child-spawn",
                    ts_ms=10.0,
                    kind=RuntimeEventKind.SPAWN,
                    workflow_id="wf",
                    invocation_id="root",
                    target_invocation_id="child",
                    execution_mode=ExecutionMode.BACKGROUND,
                ),
            )
        )

        child = BeliefKVRequestMetadata(
            "wf",
            "child",
            "ctx-child",
            0,
            parent_invocation_id="root",
            parent_context_id="ctx-root",
            relation_type="spawn",
            execution_mode="background",
        )
        runtime._ensure_causal_identity(child)

        self.assertIn("child", runtime._linked_invocations)
        self.assertEqual(
            runtime.controller.graph.invocations["root"].child_invocation_ids,
            {"child"},
        )


class SGLangContractTest(unittest.TestCase):
    def test_metadata_wire_roundtrip(self):
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 3, "coder", "coder-1")
        self.assertEqual(BeliefKVRequestMetadata.from_wire(metadata.to_wire()), metadata)

    def test_exact_version_guard(self):
        assert_supported_sglang_version(BASE_SGLANG_VERSION)
        with self.assertRaises(RuntimeError):
            assert_supported_sglang_version("0.5.2")

    def test_source_contract_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = SGLangSourceContract().check(Path(temporary))
        self.assertFalse(report.compatible)
        self.assertTrue(report.failures)

    def test_vendored_sglang_checkout_satisfies_runtime_contract(self):
        source_root = (
            Path(__file__).resolve().parents[1] / "third_party" / "sglang"
        )
        if not source_root.is_dir():
            self.skipTest("vendored SGLang checkout is unavailable")

        report = SGLangSourceContract().check(source_root)

        self.assertTrue(
            report.compatible,
            msg="; ".join(
                f"{item.file}:{item.symbol}: {item.reason}"
                for item in report.failures
            ),
        )

    def test_checkout_contract_script_imports_local_package(self):
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    str(repository_root / "scripts" / "check_sglang_contract.py"),
                    temporary,
                ],
                cwd=repository_root,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 1)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["compatible"])
        self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
