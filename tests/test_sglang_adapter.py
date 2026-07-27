import json
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import Counter, deque
from dataclasses import dataclass, replace
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
from beliefkv.policy.admission import AdmissionRequest, AdmissionSideState
from beliefkv.experiments.policy_replay import load_replay_trace
from beliefkv.policy.joint_scheduler import JointPlannerConfig, ObservedJointPlanner
from beliefkv.policy.resource_snapshot import RuntimeResourceObservation
from beliefkv.runtime.audit import PolicySnapshotLog
from beliefkv.runtime.joint_shadow import (
    IncrementalPolicyInputAssembler,
    LatestWinsJointPlanWorker,
)
from beliefkv.runtime.page_index import PageOwnershipIndex
from beliefkv.runtime.protocol import (
    CommandAck,
    CommandKind,
    CommandStatus,
    ControlCommand,
    PageHandle,
    PhysicalBundleIntent,
    PhysicalPageAction,
    PhysicalResidency,
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

    def write_backup(self, node, *, beliefkv_source=None):
        _ = beliefkv_source
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

    def load_back(
        self,
        node,
        *,
        force=False,
        allow_eviction=True,
        beliefkv_source=None,
    ):
        _ = beliefkv_source
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

    def write_backup(self, node, *, beliefkv_source=None):
        self.write_order.append(node.id)
        current = node
        while current is not None and current is not self.root_node:
            current.lock_ref += 1
            current = current.parent
        return super().write_backup(node, beliefkv_source=beliefkv_source)

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


def test_sglang_abort_result_is_openai_schema_complete_and_idempotent():
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    manager = object.__new__(TokenizerManager)
    manager.server_args = SimpleNamespace(
        tokenizer_worker_num=1,
        weight_version="beliefkv-test-version",
    )
    state = SimpleNamespace(
        finished=False,
        finished_time=None,
        out_list=[],
        event=threading.Event(),
    )
    manager.rid_to_state = {"request-1": state}

    abort = AbortReq(rid="request-1")
    manager._handle_abort_req(abort)

    assert state.finished is True
    assert state.finished_time is not None
    assert state.event.is_set()
    assert "request-1" not in manager.rid_to_state
    assert state.out_list == [
        {
            "text": "",
            "output_ids": [],
            "meta_info": {
                "id": "request-1",
                "finish_reason": {
                    "type": "abort",
                    "message": "Abort before prefill",
                },
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "weight_version": "beliefkv-test-version",
            },
        }
    ]

    manager._handle_abort_req(abort)
    assert len(state.out_list) == 1


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


class _NoBooleanSequence:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __bool__(self):
        raise RuntimeError("sequence truth value is undefined")


class _ForwardMode:
    def __init__(self, name):
        self.name = name

    def is_decode(self):
        return self.name == "DECODE"

    def is_extend(self):
        return self.name in {"EXTEND", "MIXED"}

    def is_mixed(self):
        return self.name == "MIXED"

    def is_idle(self):
        return self.name == "IDLE"

    def is_dummy_first(self):
        return self.name == "DUMMY_FIRST"


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
    def test_config_uses_authoritative_host_allocator_capacity(self):
        scheduler = SimpleNamespace(
            max_total_num_tokens=100,
            tree_cache=SimpleNamespace(
                token_to_kv_pool_host=SimpleNamespace(size=320)
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "beliefkv.json"
            path.write_text(
                json.dumps(
                    {
                        "hbm_capacity_bytes": 1000,
                        "host_capacity_bytes": 9999,
                        "reserve_hbm_bytes": 100,
                        "kv_bytes_per_token": 10,
                    }
                ),
                encoding="utf-8",
            )
            config = EmbeddedSGLangRuntime._load_config(scheduler, str(path))

        self.assertEqual(config.host_capacity_bytes, 3200)

    def test_native_hicache_telemetry_is_attributed_without_explicit_duplication(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(kv_bytes_per_token=10)
        runtime.controller = BeliefKVController(runtime.config)
        runtime.controller.process_runtime_events(
            (
                RuntimeEvent("start", 1.0, RuntimeEventKind.WORKFLOW_START, "wf"),
                RuntimeEvent(
                    "create",
                    2.0,
                    RuntimeEventKind.INVOCATION_CREATE,
                    "wf",
                    invocation_id="child",
                    context_id="ctx-child",
                    context_epoch=0,
                ),
            )
        )
        runtime.registry = SGLangNodeRegistry()
        node = _Node(7)
        handle = runtime.registry.register(node)
        runtime.controller.page_index.register_page(handle, size_bytes=40)
        runtime.controller.page_index.bind_pages("ctx-child", 0, (handle,))
        runtime.audit = _AuditRecorder()
        runtime.transfer_telemetry_log = None
        runtime._now_ms = lambda: 30.0
        record = {
            "operation_id": "h2d-7-1",
            "backend_operation_id": 7,
            "direction": "h2d",
            "source": "native_demand_load",
            "submit_ts_ms": 10.0,
            "complete_ts_ms": 20.0,
            "token_count": 4,
            "node_ids": (7,),
            "reason": "",
        }

        runtime.on_hicache_transfer_completed(record)
        runtime.on_hicache_transfer_completed({**record, "source": "explicit"})

        self.assertEqual(len(runtime.audit.events), 1)
        event, _, fields = runtime.audit.events[0]
        self.assertEqual(event, "transfer_telemetry")
        self.assertEqual(fields["actual_bytes"], 40)
        self.assertEqual(fields["direction"], "h2d")
        self.assertEqual(fields["command_kind"], "native_demand_load")
        self.assertEqual(fields["telemetry_origin"], "native_hicache_callback")
        self.assertEqual(fields["context_id"], "ctx-child")
        self.assertEqual(fields["owner_context_ids"], ("ctx-child",))
        self.assertFalse(fields["start_timestamp_observed"])

    def test_p4_runtime_rejects_online_joint_plan_application(self):
        scheduler = SimpleNamespace(
            enable_hierarchical_cache=True,
            tree_cache=_TreeCache(),
        )

        with self.assertRaisesRegex(
            SGLangBackendError,
            "online JointPlan application is not implemented",
        ):
            EmbeddedSGLangRuntime(
                scheduler,
                config=BeliefKVConfig(joint_policy_enabled=True),
            )

    def test_gpu_service_observer_pairs_overlap_launch_and_completion(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(
            queue_service_observer_enabled=True,
            queue_service_observer_max_samples=10,
        )
        runtime.audit = _AuditRecorder()
        runtime.scheduler = SimpleNamespace(
            server_args=SimpleNamespace(num_continuous_decode_steps=1)
        )
        runtime._gpu_service_launches = deque()
        runtime._gpu_service_sequence = 0
        runtime._gpu_service_sample_count = 0
        runtime._gpu_service_previous_completion_ms = None
        runtime._now_ms = lambda: 12.5
        requests = [
            SimpleNamespace(
                rid=f"request-{index}",
                beliefkv_metadata=BeliefKVRequestMetadata(
                    f"service-calibration:train:decode-b1-r0-i{index}",
                    f"inv-{index}",
                    f"ctx-{index}",
                    0,
                ),
            )
            for index in range(2)
        ]
        batch = SimpleNamespace(
            reqs=requests,
            forward_mode=_ForwardMode("DECODE"),
        )

        runtime._observe_gpu_batch_launch(batch, 10.0)
        runtime.on_batch_completed(batch)

        event, _, fields = runtime.audit.events[-1]
        self.assertEqual(event, "gpu_service_sample")
        self.assertEqual(fields["phase"], "decode")
        self.assertEqual(fields["tokens"], 2)
        self.assertEqual(fields["batch_size"], 2)
        self.assertEqual(fields["split"], "train")
        self.assertEqual(fields["calibration_kind"], "decode")
        self.assertEqual(fields["episode_id"], "train:decode-b1-r0")
        self.assertEqual(fields["elapsed_ms"], 2.5)
        self.assertEqual(fields["launch_to_completion_ms"], 2.5)
        self.assertEqual(len(runtime._gpu_service_launches), 0)

    def test_gpu_service_observer_removes_overlap_queue_time(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(queue_service_observer_enabled=True)
        runtime.audit = _AuditRecorder()
        runtime.scheduler = SimpleNamespace(
            server_args=SimpleNamespace(num_continuous_decode_steps=1)
        )
        runtime._gpu_service_launches = deque()
        runtime._gpu_service_sequence = 0
        runtime._gpu_service_sample_count = 0
        runtime._gpu_service_previous_completion_ms = 10.0
        runtime._now_ms = lambda: 15.0
        request = SimpleNamespace(
            rid="request-0",
            beliefkv_metadata=BeliefKVRequestMetadata(
                "service-calibration:holdout:decode-b1-r0-i0",
                "inv-0",
                "ctx-0",
                0,
            ),
        )
        batch = SimpleNamespace(
            reqs=[request],
            forward_mode=_ForwardMode("DECODE"),
        )

        runtime._observe_gpu_batch_launch(batch, 2.0)
        runtime.on_batch_completed(batch)

        _, _, fields = runtime.audit.events[-1]
        self.assertEqual(fields["service_start_ts_ms"], 10.0)
        self.assertEqual(fields["service_elapsed_ms"], 5.0)
        self.assertEqual(fields["launch_to_completion_ms"], 13.0)
        self.assertEqual(fields["elapsed_ms"], 5.0)

    def test_gpu_service_observer_excludes_wrong_calibration_phase(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(queue_service_observer_enabled=True)
        runtime.audit = _AuditRecorder()
        runtime.scheduler = SimpleNamespace(
            server_args=SimpleNamespace(num_continuous_decode_steps=1)
        )
        runtime._gpu_service_launches = deque()
        runtime._gpu_service_sequence = 0
        runtime._gpu_service_sample_count = 0
        runtime._gpu_service_previous_completion_ms = None
        runtime._now_ms = lambda: 2.0
        request = SimpleNamespace(
            rid="prefill-case-output-token",
            beliefkv_metadata=BeliefKVRequestMetadata(
                "service-calibration:train:prefill-512-0",
                "inv-0",
                "ctx-0",
                0,
            ),
        )
        batch = SimpleNamespace(
            reqs=[request],
            forward_mode=_ForwardMode("DECODE"),
        )

        runtime._observe_gpu_batch_launch(batch, 1.0)
        runtime.on_batch_completed(batch)

        self.assertEqual(runtime.audit.events, [])
        self.assertEqual(runtime._gpu_service_sample_count, 0)

    def test_gpu_service_observer_excludes_non_calibration_batches(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(queue_service_observer_enabled=True)
        runtime.audit = _AuditRecorder()
        runtime.scheduler = SimpleNamespace(server_args=SimpleNamespace())
        runtime._gpu_service_launches = deque()
        runtime._gpu_service_sequence = 0
        runtime._gpu_service_sample_count = 0
        runtime._gpu_service_previous_completion_ms = None
        runtime._now_ms = lambda: 2.0
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid="native", beliefkv_metadata=None)],
            forward_mode=_ForwardMode("EXTEND"),
            extend_num_tokens=100,
        )

        runtime._observe_gpu_batch_launch(batch, 1.0)
        runtime.on_batch_completed(batch)

        self.assertEqual(runtime.audit.events, [])
        self.assertEqual(runtime._gpu_service_sample_count, 0)

    def test_gpu_service_observer_records_tagged_runtime_batch_context(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(
            queue_service_observer_enabled=True,
            queue_service_observer_include_runtime_batches=True,
        )
        runtime.audit = _AuditRecorder()
        runtime.scheduler = SimpleNamespace(
            server_args=SimpleNamespace(num_continuous_decode_steps=1)
        )
        runtime._gpu_service_launches = deque()
        runtime._gpu_service_sequence = 0
        runtime._gpu_service_sample_count = 0
        runtime._gpu_service_previous_completion_ms = None
        runtime._now_ms = lambda: 4.0
        request = SimpleNamespace(
            rid="runtime-request",
            fill_ids=list(range(4096)),
            beliefkv_metadata=BeliefKVRequestMetadata(
                "workflow",
                "invocation",
                "context",
                0,
            ),
        )
        batch = SimpleNamespace(
            reqs=[request],
            forward_mode=_ForwardMode("DECODE"),
        )

        runtime._observe_gpu_batch_launch(batch, 2.0)
        runtime.on_batch_completed(batch)

        event, _, fields = runtime.audit.events[-1]
        self.assertEqual(event, "gpu_service_sample")
        self.assertEqual(fields["observation_scope"], "runtime")
        self.assertEqual(fields["phase"], "decode")
        self.assertEqual(fields["sequence_tokens_before"], [4096])
        self.assertEqual(fields["max_sequence_tokens_before"], 4096)
        self.assertEqual(fields["workflow_ids"], ["workflow"])

    def test_request_physical_checkpoint_separates_allocator_and_radix_growth(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(
            hbm_capacity_bytes=10_000,
            host_capacity_bytes=10_000,
            reserve_hbm_bytes=0,
            kv_bytes_per_token=10,
        )
        runtime.controller = BeliefKVController(runtime.config)
        runtime.controller.page_index.register_context("ctx", "wf", 0)
        first = PageHandle(1, 0)
        runtime.controller.page_index.register_page(first, size_bytes=100)
        runtime.controller.page_index.bind_pages("ctx", 0, (first,))
        runtime.scheduler = SimpleNamespace(
            page_size=1,
            token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
        )
        runtime.audit = _AuditRecorder()
        runtime._now_ms = lambda: 20.0
        runtime._tree_dirty = False
        runtime._request_physical_start_by_id = {}
        runtime._pending_request_physical_finish_by_id = {}
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        request = SimpleNamespace(
            rid="request",
            origin_input_ids=_NoBooleanSequence(10),
            prefix_indices=_NoBooleanSequence(5),
        )

        runtime._capture_request_physical_start(request, metadata, 10.0)
        second = PageHandle(2, 0)
        runtime.controller.page_index.register_page(second, size_bytes=40)
        runtime.controller.page_index.bind_pages("ctx", 0, (second,))
        runtime._queue_request_physical_finish(
            request,
            metadata,
            output_tokens=2,
            cache_commit_tokens=11,
        )
        runtime._flush_request_physical_finishes()

        event, _, fields = runtime.audit.events[-1]
        self.assertEqual(event, "request_physical_delta")
        self.assertEqual(fields["context_path_bytes_before"], 100)
        self.assertEqual(fields["context_path_bytes_after"], 140)
        self.assertEqual(fields["context_path_growth_bytes"], 40)
        self.assertEqual(fields["cache_commit_tokens"], 11)
        self.assertEqual(fields["allocator_growth_bytes_upper_bound"], 60)
        self.assertTrue(fields["allocator_growth_exact"])
        self.assertEqual(fields["new_extent_bytes"], 40)

    def test_sparse_policy_snapshot_capture_is_replay_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
            runtime.config = BeliefKVConfig(
                hbm_capacity_bytes=1_000,
                host_capacity_bytes=1_000,
                reserve_hbm_bytes=0,
                kv_bytes_per_token=10,
                predictor_enabled=False,
                shadow_enabled=False,
                reference_policy_snapshot_min_interval_ms=1_000,
            )
            runtime.controller = BeliefKVController(runtime.config)
            runtime.scheduler = SimpleNamespace(waiting_queue=[])
            runtime.backend = SimpleNamespace(
                capabilities=SimpleNamespace(operation_merge=False)
            )
            runtime.audit = _AuditRecorder()
            path = Path(temporary) / "policy.jsonl.gz"
            runtime.policy_snapshot_log = PolicySnapshotLog(
                path,
                trace_id="trace-runtime",
                trace_sensitivity="timing_sensitive",
            )
            runtime._last_policy_snapshot_structural_signature = None
            runtime._last_policy_snapshot_physical_signature = None
            runtime._last_policy_snapshot_hbm_bucket = None
            runtime._last_policy_snapshot_ms = None
            first = RuntimeResourceObservation(
                ts_ms=10,
                hbm_capacity_bytes=1_000,
                hbm_used_bytes=0,
                host_capacity_bytes=1_000,
                host_used_bytes=0,
                host_free_bytes=1_000,
            )

            runtime._maybe_record_policy_snapshot(first)
            runtime._maybe_record_policy_snapshot(first)
            runtime.controller.process_runtime_event(
                RuntimeEvent(
                    "workflow-start",
                    11,
                    RuntimeEventKind.WORKFLOW_START,
                    "workflow",
                )
            )
            runtime._maybe_record_policy_snapshot(replace(first, ts_ms=11))
            runtime.policy_snapshot_log.close()

            snapshots = load_replay_trace(path)
            self.assertEqual(len(snapshots), 2)
            self.assertEqual(snapshots[1].policy_input.runtime_graph.graph_version, 1)
            recorded = [
                item for item in runtime.audit.events
                if item[0] == "policy_snapshot_recorded"
            ]
            self.assertEqual(len(recorded), 2)

    def test_joint_shadow_safe_point_validates_without_applying_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
            runtime.config = BeliefKVConfig(
                hbm_capacity_bytes=1_000,
                host_capacity_bytes=1_000,
                reserve_hbm_bytes=0,
                kv_bytes_per_token=10,
                predictor_enabled=False,
                shadow_enabled=False,
                reference_policy_snapshot_min_interval_ms=1_000,
            )
            runtime.controller = BeliefKVController(runtime.config)
            runtime.controller.process_runtime_events(
                (
                    RuntimeEvent(
                        "workflow-start",
                        1,
                        RuntimeEventKind.WORKFLOW_START,
                        "workflow",
                    ),
                    RuntimeEvent(
                        "root-create",
                        2,
                        RuntimeEventKind.INVOCATION_CREATE,
                        "workflow",
                        invocation_id="root",
                        context_id="ctx-root",
                        context_epoch=0,
                    ),
                )
            )
            runtime.controller.submit_request(
                AdmissionRequest(
                    request_id="request-root",
                    workflow_id="workflow",
                    invocation_id="root",
                    context_id="ctx-root",
                    context_epoch=0,
                    submitted_ts_ms=3,
                    uncached_prompt_tokens=2,
                    expected_output_tokens=2,
                    kv_bytes_per_token=10,
                )
            )
            runtime.scheduler = SimpleNamespace(waiting_queue=[])
            runtime.backend = SimpleNamespace(
                capabilities=SimpleNamespace(operation_merge=False)
            )
            runtime.audit = _AuditRecorder()
            runtime.policy_snapshot_log = PolicySnapshotLog(
                Path(temporary) / "joint-policy.jsonl.gz",
                trace_id="trace-joint-runtime",
                trace_sensitivity="timing_sensitive",
            )
            runtime._last_policy_snapshot_structural_signature = None
            runtime._last_policy_snapshot_physical_signature = None
            runtime._last_policy_snapshot_hbm_bucket = None
            runtime._last_policy_snapshot_ms = None
            runtime._last_joint_shadow_result_sequence = 0
            runtime._joint_shadow_counts = Counter()
            runtime._joint_shadow_strict_stale_reasons = Counter()
            runtime._joint_shadow_readset_stale_reasons = Counter()
            runtime._joint_shadow_timing_samples = {
                name: deque(maxlen=128)
                for name in (
                    "snapshot_build_ms",
                    "snapshot_trace_enqueue_ms",
                    "snapshot_enqueue_ms",
                    "plan_queue_wait_ms",
                    "plan_compute_ms",
                    "plan_publish_to_safe_point_ms",
                    "validation_ms",
                    "plan_age_ms",
                )
            }
            runtime.joint_shadow_worker = LatestWinsJointPlanWorker(
                ObservedJointPlanner(
                    JointPlannerConfig(max_planning_budget_ms=100)
                )
            )
            observation = RuntimeResourceObservation(
                ts_ms=10,
                hbm_capacity_bytes=1_000,
                hbm_used_bytes=0,
                host_capacity_bytes=1_000,
                host_used_bytes=0,
                host_free_bytes=1_000,
            )
            pending_before = runtime.controller.admission.pending_requests()
            history_before = tuple(runtime.controller.command_history)

            runtime._maybe_record_policy_snapshot(observation)
            for _ in range(100):
                if runtime.joint_shadow_worker.latest() is not None:
                    break
                threading.Event().wait(0.01)
            runtime.controller.fairness.charge_service("workflow", 1.0)
            runtime._maybe_record_policy_snapshot(observation)

            would_apply = [
                item
                for item in runtime.audit.events
                if item[0] == "joint_plan_would_apply"
            ]
            self.assertEqual(len(would_apply), 1)
            self.assertTrue(would_apply[0][2]["readset_fresh"])
            self.assertFalse(would_apply[0][2]["strict_global_fresh"])
            self.assertIn(
                "snapshot_id", would_apply[0][2]["strict_global_reasons"]
            )
            self.assertEqual(
                runtime.controller.admission.pending_requests(), pending_before
            )
            self.assertEqual(tuple(runtime.controller.command_history), history_before)
            self.assertEqual(runtime.controller.page_index.gpu_bytes, 0)
            self.assertEqual(runtime.policy_snapshot_log.count, 1)
            self.assertEqual(
                runtime.joint_shadow_worker.stats().submitted_count, 1
            )

            self.assertTrue(runtime.joint_shadow_worker.close())
            runtime.joint_shadow_worker = None
            runtime.policy_snapshot_log.close()

    def test_incremental_joint_shadow_builds_policy_input_off_safe_point(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
            runtime.config = BeliefKVConfig(
                hbm_capacity_bytes=1_000,
                host_capacity_bytes=1_000,
                reserve_hbm_bytes=0,
                kv_bytes_per_token=10,
                predictor_enabled=False,
                shadow_enabled=False,
                reference_policy_snapshot_min_interval_ms=1_000,
            )
            runtime.controller = BeliefKVController(runtime.config)
            runtime.controller.process_runtime_events(
                (
                    RuntimeEvent(
                        "workflow-start",
                        1,
                        RuntimeEventKind.WORKFLOW_START,
                        "workflow",
                    ),
                    RuntimeEvent(
                        "root-create",
                        2,
                        RuntimeEventKind.INVOCATION_CREATE,
                        "workflow",
                        invocation_id="root",
                        context_id="ctx-root",
                        context_epoch=0,
                    ),
                )
            )
            runtime.scheduler = SimpleNamespace(waiting_queue=[])
            runtime.backend = SimpleNamespace(
                capabilities=SimpleNamespace(operation_merge=False)
            )
            runtime.audit = _AuditRecorder()
            runtime.policy_snapshot_log = PolicySnapshotLog(
                Path(temporary) / "incremental-joint-policy.jsonl.gz",
                trace_id="trace-incremental-joint-runtime",
                trace_sensitivity="timing_sensitive",
            )
            runtime._last_policy_snapshot_structural_signature = None
            runtime._last_policy_snapshot_physical_signature = None
            runtime._last_policy_snapshot_hbm_bucket = None
            runtime._last_policy_snapshot_ms = None
            runtime._shadow_event_sequence = 0
            runtime._shadow_page_revision = 0
            runtime._shadow_telemetry_sequence = 0
            runtime._last_joint_shadow_result_sequence = 0
            runtime._joint_shadow_counts = Counter()
            runtime._joint_shadow_strict_stale_reasons = Counter()
            runtime._joint_shadow_readset_stale_reasons = Counter()
            runtime._joint_shadow_timing_samples = {
                name: deque(maxlen=128)
                for name in (
                    "snapshot_build_ms",
                    "safe_point_delta_capture_ms",
                    "snapshot_trace_enqueue_ms",
                    "snapshot_enqueue_ms",
                    "plan_queue_wait_ms",
                    "plan_compute_ms",
                    "plan_publish_to_safe_point_ms",
                    "validation_ms",
                    "plan_age_ms",
                )
            }
            runtime.joint_shadow_worker = LatestWinsJointPlanWorker(
                ObservedJointPlanner(
                    JointPlannerConfig(max_planning_budget_ms=100)
                ),
                assembler=IncrementalPolicyInputAssembler(runtime.config),
            )
            observation = RuntimeResourceObservation(
                ts_ms=10,
                hbm_capacity_bytes=1_000,
                hbm_used_bytes=0,
                host_capacity_bytes=1_000,
                host_used_bytes=0,
                host_free_bytes=1_000,
            )

            def forbidden_build(*_args, **_kwargs):
                raise AssertionError("live controller builder reached safe point")

            runtime.controller.build_policy_input = forbidden_build
            runtime._maybe_record_policy_snapshot(observation)
            for _ in range(100):
                if runtime.joint_shadow_worker.latest() is not None:
                    break
                threading.Event().wait(0.01)
            runtime.controller.fairness.charge_service("workflow", 1.0)
            runtime._maybe_record_policy_snapshot(observation)

            self.assertEqual(runtime.policy_snapshot_log.count, 1)
            self.assertEqual(
                runtime.joint_shadow_worker.stats().submitted_count, 1
            )
            delta_events = [
                item for item in runtime.audit.events
                if item[0] == "joint_plan_shadow_delta_enqueued"
            ]
            self.assertEqual(len(delta_events), 1)
            self.assertEqual(delta_events[0][2]["event_count"], 2)
            recorded = [
                item for item in runtime.audit.events
                if item[0] == "policy_snapshot_recorded"
            ]
            self.assertEqual(recorded[0][2]["safe_point_build_ms"], 0.0)

            self.assertTrue(runtime.joint_shadow_worker.close())
            runtime.joint_shadow_worker = None
            runtime.policy_snapshot_log.close()

    def test_request_restore_dependency_uses_only_matched_radix_path(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.controller = BeliefKVController()
        runtime.registry = SGLangNodeRegistry()
        runtime.tree_cache = _TreeCache()
        runtime.controller.page_index.register_context("ctx", "wf", 0)

        matched = _Node(1)
        matched.parent = runtime.tree_cache.root_node
        unrelated = _Node(2)
        unrelated.parent = runtime.tree_cache.root_node
        matched_handle = runtime.registry.register(matched)
        unrelated_handle = runtime.registry.register(unrelated)
        for handle in (matched_handle, unrelated_handle):
            runtime.controller.page_index.register_page(
                handle,
                size_bytes=100,
                residency=PhysicalResidency.CPU_ONLY,
            )
        runtime.controller.page_index.bind_pages(
            "ctx", 0, (matched_handle, unrelated_handle)
        )
        request = SimpleNamespace(last_node=matched)

        dependencies = runtime._request_restore_bundle_ids(request, "ctx")

        self.assertEqual(
            dependencies,
            (
                f"page:{matched_handle.page_id}:"
                f"{matched_handle.allocation_generation}",
            ),
        )

    def test_policy_runtime_runnable_does_not_test_tensor_truth_value(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(kv_bytes_per_token=10)
        runtime.controller = BeliefKVController(runtime.config)
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        runtime.scheduler = SimpleNamespace(
            waiting_queue=[
                SimpleNamespace(
                    rid="request",
                    beliefkv_metadata=metadata,
                    sampling_params=SimpleNamespace(max_new_tokens=20),
                    output_ids=_NoBooleanSequence(3),
                    origin_input_ids=_NoBooleanSequence(10),
                    prefix_indices=_NoBooleanSequence(5),
                )
            ]
        )

        runnable = runtime._policy_runtime_runnable(100.0)

        self.assertEqual(len(runnable), 1)
        self.assertEqual(runnable[0].request_id, "request")
        self.assertEqual(runnable[0].startup_bytes, 220)

    def test_tree_sync_defers_generation_changes_until_transfer_ack(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime._tree_dirty = True
        runtime.controller = SimpleNamespace(inflight_command_ids=("transfer-1",))

        runtime.sync_tree()

        self.assertTrue(runtime._tree_dirty)

    def test_tree_sync_applies_local_residency_insert_and_remove_deltas(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(kv_bytes_per_token=10)
        runtime.controller = BeliefKVController(runtime.config)
        runtime.registry = SGLangNodeRegistry()
        runtime.tree_cache = _TreeCache()
        runtime._terminal_node_by_context = {}
        runtime._tree_dirty = True
        runtime._tree_full_rebuild_required = True
        runtime._dirty_radix_nodes = {}
        runtime._removed_radix_nodes = {}
        runtime._dirty_context_ids = set()
        runtime._tree_sync_timing_samples = deque(maxlen=100)
        node = _Node(1)
        node.key = [1, 2]
        node.last_access_time = 0.0
        node.parent = runtime.tree_cache.root_node
        runtime.tree_cache.root_node.children[1] = node

        runtime.sync_tree(force=True)
        handle = runtime.registry.current_handle(1)
        self.assertEqual(
            runtime.controller.page_index.pages[handle].residency.value,
            "gpu_only",
        )

        node.host_value = [10, 11]
        runtime.on_radix_mutation((node,), False, False)
        runtime.sync_tree()
        self.assertEqual(
            runtime.controller.page_index.pages[handle].residency.value,
            "dual_clean",
        )
        self.assertEqual(runtime._tree_sync_timing_samples[-1][0], "incremental")

        child = _Node(2)
        child.key = [3]
        child.last_access_time = 0.0
        child.parent = node
        node.children[3] = child
        runtime.on_radix_mutation((child,), True, False)
        runtime.sync_tree()
        child_handle = runtime.registry.current_handle(2)
        self.assertEqual(
            runtime.controller.page_index.pages[child_handle].parent,
            handle,
        )

        del node.children[3]
        runtime.on_radix_mutation((child,), True, True)
        runtime.sync_tree()
        self.assertEqual(
            runtime.controller.page_index.pages[child_handle].residency.value,
            "dead",
        )

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
        self.assertEqual(fields["untracked_allocator_delta_bytes"], 500)
        self.assertIsNone(fields["engine_locked_gpu_bytes"])
        self.assertEqual(fields["kv_state_breakdown_scope"], "unavailable")
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

    def test_resource_snapshot_exposes_closure_aware_physical_kv_states(self):
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
        page_index = PageOwnershipIndex()
        parent = PageHandle(1, 0)
        child = PageHandle(2, 0)
        page_index.register_page(parent, size_bytes=100, radix_depth=1)
        page_index.register_page(
            child, size_bytes=200, radix_depth=2, parent=parent
        )
        page_index.set_engine_lock(child, 1)
        runtime.controller = SimpleNamespace(
            page_index=page_index,
            inflight_command_ids=(),
            signals=SimpleNamespace(
                pcie_utilization=0.0,
                gpu_compute_utilization=0.0,
            ),
            _engine_request_count=1,
            _running_request_count=1,
        )

        runtime._emit_resource_snapshot(force=True)

        fields = runtime.audit.events[0][2]
        self.assertEqual(fields["page_index_gpu_bytes"], 300)
        self.assertEqual(fields["untracked_allocator_delta_bytes"], 900)
        self.assertEqual(fields["engine_locked_gpu_bytes"], 200)
        self.assertEqual(fields["closure_blocked_gpu_bytes"], 100)
        self.assertEqual(fields["migratable_gpu_bytes"], 0)
        self.assertEqual(fields["dual_resident_gpu_bytes"], 0)
        self.assertEqual(
            fields["kv_state_breakdown_scope"], "physical_radix_closure"
        )

    def test_resource_snapshot_attributes_stale_locks_to_running_request_path(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(
            hbm_capacity_bytes=1600,
            host_capacity_bytes=3200,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=16,
            queue_service_observer_enabled=False,
        )
        runtime.audit = _AuditRecorder()
        runtime._last_resource_telemetry_ms = None
        tree = _TreeCache()
        tree.token_to_kv_pool_host = _HostAllocator(200, 150, 16)
        parent_node = _Node(1)
        child_node = _Node(2)
        parent_node.parent = tree.root_node
        child_node.parent = parent_node
        tree.root_node.children = {1: parent_node}
        parent_node.children = {2: child_node}
        parent_node.lock_ref = 1
        child_node.lock_ref = 1
        runtime.tree_cache = tree
        runtime.registry = SGLangNodeRegistry()
        parent = runtime.registry.register(parent_node)
        child = runtime.registry.register(child_node)
        page_index = PageOwnershipIndex()
        page_index.register_page(parent, size_bytes=100, radix_depth=1)
        page_index.register_page(
            child, size_bytes=200, radix_depth=2, parent=parent
        )
        page_index.set_engine_lock(parent, 1)
        page_index.set_engine_lock(child, 1)
        runtime.controller = BeliefKVController(runtime.config)
        runtime.controller.page_index = page_index
        runtime._active_request_ids = {"request"}
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        request = SimpleNamespace(
            rid="request",
            beliefkv_metadata=metadata,
            last_node=child_node,
        )
        batch = SimpleNamespace(
            reqs=[request],
            forward_mode=_ForwardMode("DECODE"),
        )
        runtime.scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=_Allocator(25),
            max_total_num_tokens=100,
            running_batch=SimpleNamespace(reqs=[request]),
            chunked_req=None,
        )
        runtime._observe_request_selected_for_lock_service(
            request,
            metadata,
            now_ms=0.0,
        )

        runtime._now_ms = lambda: 600.0
        runtime._emit_resource_snapshot(force=True)
        stale_fields = runtime.audit.events[-1][2]
        self.assertEqual(stale_fields["engine_lock_ref_gpu_bytes"], 300)
        self.assertEqual(stale_fields["engine_lock_fully_attributed_gpu_bytes"], 300)
        self.assertEqual(
            stale_fields["locked_but_not_served_gpu_bytes_100ms"], 300
        )
        self.assertEqual(
            stale_fields["locked_but_not_served_gpu_bytes_500ms"], 300
        )
        self.assertEqual(stale_fields["engine_lock_request_path_error_count"], 0)

        runtime._now_ms = lambda: 650.0
        runtime.on_batch_completed(batch)
        runtime._now_ms = lambda: 700.0
        runtime._emit_resource_snapshot(force=True)
        served_fields = runtime.audit.events[-1][2]
        self.assertEqual(served_fields["lock_recently_served_gpu_bytes_100ms"], 300)
        self.assertEqual(
            served_fields["locked_but_not_served_gpu_bytes_100ms"], 0
        )
        self.assertEqual(getattr(runtime, "_gpu_service_sample_count", 0), 0)

        runtime._active_request_ids.clear()
        runtime._now_ms = lambda: 750.0
        runtime.on_batch_completed(batch)
        self.assertFalse(runtime._lock_service_ledger.tracks("request"))

    def test_visible_waiting_request_does_not_block_h2d(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        request = SimpleNamespace(rid="request", beliefkv_metadata=metadata)
        runtime._request_metadata_by_id = {"request": metadata}
        runtime._active_request_ids = set()
        runtime.scheduler = SimpleNamespace(
            waiting_queue=[request],
            running_batch=SimpleNamespace(reqs=[]),
            chunked_req=None,
        )

        self.assertFalse(runtime._context_has_engine_request("ctx"))
        runtime._active_request_ids.add("request")
        self.assertTrue(runtime._context_has_engine_request("ctx"))

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

    def test_visible_request_tracks_uncached_prompt_without_reservation(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.controller = BeliefKVController()
        runtime.controller.process_runtime_events(
            (
                RuntimeEvent("start", 1.0, RuntimeEventKind.WORKFLOW_START, "wf"),
                RuntimeEvent(
                    "create",
                    2.0,
                    RuntimeEventKind.INVOCATION_CREATE,
                    "wf",
                    invocation_id="inv",
                    context_id="ctx",
                    context_epoch=0,
                ),
            )
        )
        runtime.tree_cache = object()
        runtime.config = BeliefKVConfig(kv_bytes_per_token=16)
        runtime._request_metadata_by_id = {}
        runtime._request_submitted_ts_by_id = {}
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

        self.assertTrue(runtime.register_visible_request(request))

        entry = runtime.controller.visible_admission.get("request-1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.request.uncached_prompt_tokens, 10)
        self.assertEqual(entry.request.estimated_incremental_bytes, 320)
        self.assertEqual(runtime.controller.visible_admission.reserved_bytes, 0)
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

    def test_h2d_waiter_holds_no_reservation_and_rematches_after_ack(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.controller = BeliefKVController()
        runtime.controller.process_runtime_events(
            (
                RuntimeEvent("start", 1.0, RuntimeEventKind.WORKFLOW_START, "wf"),
                RuntimeEvent(
                    "create",
                    2.0,
                    RuntimeEventKind.INVOCATION_CREATE,
                    "wf",
                    invocation_id="inv",
                    context_id="ctx",
                    context_epoch=0,
                ),
            )
        )
        runtime.tree_cache = object()
        runtime.config = BeliefKVConfig(kv_bytes_per_token=16)
        runtime._now_ms = lambda: 42.0
        runtime.audit = _AuditRecorder()
        runtime._request_metadata_by_id = {
            "request-1": BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        }
        runtime._request_submitted_ts_by_id = {"request-1": 10.0}
        runtime._pending_h2d_contexts = set()
        refresh_count = 0

        def refresh(_tree_cache):
            nonlocal refresh_count
            refresh_count += 1
            request.prefix_indices = _NoBooleanSequence(4)

        request = SimpleNamespace(
            rid="request-1",
            beliefkv_metadata=runtime._request_metadata_by_id["request-1"],
            origin_input_ids=_NoBooleanSequence(5),
            prefix_indices=_NoBooleanSequence(0),
            init_next_round_input=refresh,
        )
        runtime.scheduler = SimpleNamespace(waiting_queue=[request])
        runtime.controller.register_visible_request(
            AdmissionRequest(
                "request-1",
                "wf",
                "inv",
                "ctx",
                0,
                10.0,
                5,
                1,
                16,
                prompt_tokens=5,
            )
        )

        runtime._mark_context_wait_restore(
            "ctx", bundle_ids=("bundle",), reason="h2d_inflight"
        )
        waiting = runtime.controller.visible_admission.get("request-1")
        self.assertEqual(waiting.state, AdmissionSideState.WAIT_RESTORE)
        self.assertEqual(runtime.controller.visible_admission.reserved_bytes, 0)
        self.assertEqual(refresh_count, 0)

        runtime._release_h2d_waiters("ctx")
        self.assertEqual(refresh_count, 1)
        released = runtime.controller.visible_admission.get("request-1")
        self.assertEqual(released.state, AdmissionSideState.VISIBLE_PENDING)
        self.assertEqual(released.request.uncached_prompt_tokens, 1)

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

    def test_atomic_drop_bundle_commits_deep_first_and_keeps_host_copies(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        parent = _Node(1)
        child = _Node(2)
        parent.parent = tree.root_node
        child.parent = parent
        parent.children["child"] = child
        parent.host_value = [10, 11, 12, 13]
        child.host_value = [20, 21, 22, 23]
        parent_handle = registry.register(parent)
        child_handle = registry.register(child)
        command = resolved_bundle(
            CommandKind.DROP_CONTEXT,
            (
                (parent_handle, PhysicalPageAction.DROP, 400),
                (child_handle, PhysicalPageAction.DROP, 400),
            ),
        )
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)

        submission = backend.submit(command)
        ack = backend.poll_acks()[0]

        self.assertEqual(
            submission.started_handles, tuple(sorted((parent_handle, child_handle)))
        )
        self.assertEqual(ack.status, CommandStatus.COMPLETED)
        self.assertEqual(ack.actual_bytes, 800)
        self.assertTrue(parent.evicted)
        self.assertTrue(child.evicted)
        self.assertTrue(parent.backuped)
        self.assertTrue(child.backuped)

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

    def test_drop_host_copy_keeps_dual_clean_gpu_extent(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        node = _Node(1)
        node.parent = tree.root_node
        node.host_value = [10, 11, 12, 13]
        tree.root_node.children["node"] = node
        handle = registry.register(node)
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)
        command = resolved(
            CommandKind.DROP_TERMINAL_PRIVATE,
            handle,
            PhysicalPageAction.DROP_HOST,
        )

        submission = backend.submit(command)
        ack = backend.poll_acks()[0]

        self.assertEqual(submission.started_handles, (handle,))
        self.assertEqual(ack.status, CommandStatus.COMPLETED)
        self.assertFalse(node.evicted)
        self.assertFalse(node.backuped)
        self.assertIs(tree.root_node.children["node"], node)

    def test_drop_host_only_extent_removes_cpu_radix_leaf(self):
        tree = _TreeCache()
        registry = SGLangNodeRegistry()
        node = _Node(1)
        node.parent = tree.root_node
        node.value = None
        node.host_value = [10, 11, 12, 13]
        tree.root_node.children["node"] = node
        handle = registry.register(node)
        backend = HiCacheNodeCommandBackend(tree, registry, now_ms=lambda: 2)
        command = resolved(
            CommandKind.DROP_TERMINAL_PRIVATE,
            handle,
            PhysicalPageAction.DROP_HOST,
        )

        submission = backend.submit(command)
        ack = backend.poll_acks()[0]

        self.assertEqual(submission.started_handles, (handle,))
        self.assertEqual(ack.status, CommandStatus.COMPLETED)
        self.assertNotIn("node", tree.root_node.children)
        self.assertFalse(node.backuped)

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

    def test_abort_bridge_drops_visible_side_state_only(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.controller = BeliefKVController()
        runtime.controller.process_runtime_events(
            (
                RuntimeEvent("start", 1.0, RuntimeEventKind.WORKFLOW_START, "wf"),
                RuntimeEvent(
                    "create",
                    2.0,
                    RuntimeEventKind.INVOCATION_CREATE,
                    "wf",
                    invocation_id="inv",
                    context_id="ctx",
                    context_epoch=0,
                ),
            )
        )
        runtime.controller.register_visible_request(
            AdmissionRequest(
                "req-visible", "wf", "inv", "ctx", 0, 2.0, 1, 1, 16
            )
        )
        runtime._request_metadata_by_id = {
            "req-visible": BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        }
        runtime._request_submitted_ts_by_id = {"req-visible": 2.0}

        removed = runtime.on_abort_request(_AbortRequest(rid="req-"))
        self.assertEqual(removed, 1)
        self.assertIsNone(
            runtime.controller.visible_admission.get("req-visible")
        )
        self.assertEqual(runtime._request_metadata_by_id, {})

    def test_return_cancels_a_visible_request_before_next_prefill_epoch(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig()
        runtime.controller = BeliefKVController(runtime.config)
        runtime.audit = _AuditRecorder()
        runtime.event_log = _EventBatchRecorder()
        runtime._now_ms = lambda: 30.0
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
        runtime.controller.register_visible_request(
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

        self.assertIsNone(runtime.controller.visible_admission.get("request"))
        self.assertEqual([item.rid for item in scheduler.requests], ["request"])
        cancelled = [
            fields
            for event, _, fields in runtime.audit.events
            if event == "terminal_request_cancelled"
        ]
        self.assertEqual(cancelled[0]["phase"], "visible_pending")

    def test_request_arriving_after_return_is_rejected_without_admission(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.controller = BeliefKVController()
        runtime.audit = _AuditRecorder()
        runtime._now_ms = lambda: 30.0
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

        self.assertFalse(runtime.register_visible_request(request))
        self.assertEqual(len(runtime.controller.visible_admission.entries()), 0)
        self.assertEqual(
            runtime.scheduler.send_to_tokenizer.messages[0].rid,
            "late-request",
        )
        self.assertEqual(runtime.audit.events[0][0], "terminal_request_rejected")

    def test_return_marks_an_engine_owned_request_terminal(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig()
        runtime.controller = BeliefKVController(runtime.config)
        runtime.audit = _AuditRecorder()
        runtime.event_log = _EventBatchRecorder()
        runtime._now_ms = lambda: 30.0
        runtime._active_request_ids = set()
        runtime._terminal_cancelled_request_ids = set()
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        runtime._request_metadata_by_id = {"request": metadata}
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

        self.assertEqual(runtime._terminal_cancelled_request_ids, {"request"})
        cancelled = [
            fields
            for event, _, fields in runtime.audit.events
            if event == "terminal_request_cancelled"
        ]
        self.assertEqual(cancelled[0]["phase"], "engine_owned")

    def test_cache_finish_suppresses_a_late_terminal_llm_result(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        request = SimpleNamespace(
            rid="request",
            beliefkv_metadata=metadata,
            output_ids=[1, 2],
        )
        runtime._ensure_allocator_radix_consistency = lambda **_kwargs: None
        runtime._match_terminal_node = lambda _token_ids: None
        runtime._metadata_scope_is_terminal = lambda _metadata: True
        runtime._tree_dirty = False
        runtime._active_request_ids = {"request"}
        runtime._terminal_cancelled_request_ids = set()
        runtime._request_metadata_by_id = {"request": metadata}
        runtime._request_submitted_ts_by_id = {"request": 1.0}
        runtime._request_physical_start_by_id = {"request": {}}
        runtime._pending_request_physical_finish_by_id = {"request": {}}
        runtime._terminal_node_by_context = {}
        runtime.controller = BeliefKVController()
        runtime.audit = _AuditRecorder()
        runtime._now_ms = lambda: 30.0
        runtime._emit = lambda *_args, **_kwargs: self.fail(
            "a terminal cache callback must not emit LLM_RESULT"
        )

        runtime.on_cache_finished(request, [1, 2, 3])

        self.assertNotIn("request", runtime._active_request_ids)
        self.assertNotIn("request", runtime._request_metadata_by_id)
        terminal = [
            fields
            for event, _, fields in runtime.audit.events
            if event == "terminal_request_abort_finished"
        ]
        self.assertEqual(len(terminal), 1)
        self.assertFalse(terminal[0]["terminal_marker"])
        self.assertTrue(terminal[0]["logical_scope_terminal"])

    def test_ticket_epoch_does_not_mutate_native_queue_or_cap_workflow(self):
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
        runtime._now_ms = lambda: 10.0
        runtime.audit = _AuditRecorder()
        runtime._admission_epoch = 0
        runtime._current_ticket_epoch = None
        runtime._current_tickets_by_request = {}
        runtime._ticket_attempted_request_ids = set()
        runtime._ticket_selected_request_ids = set()
        runtime._ticket_skip_audit = set()
        runtime._ticket_selection_details = {}
        runtime._ticket_native_rejections = {}
        runtime._pending_h2d_contexts = set()
        runtime._request_metadata_by_id = {}
        runtime._request_submitted_ts_by_id = {}
        untagged = SimpleNamespace(rid="untagged", beliefkv_metadata=None)

        def req(workflow_id, suffix):
            metadata = BeliefKVRequestMetadata(
                workflow_id,
                f"{workflow_id}-inv-{suffix}",
                f"{workflow_id}-ctx-{suffix}",
                0,
            )
            request = SimpleNamespace(
                rid=f"{workflow_id}-{suffix}",
                beliefkv_metadata=metadata,
                origin_input_ids=_NoBooleanSequence(1),
                prefix_indices=_NoBooleanSequence(0),
            )
            runtime._request_metadata_by_id[request.rid] = metadata
            runtime._request_submitted_ts_by_id[request.rid] = float(suffix)
            controller.register_visible_request(
                AdmissionRequest(
                    request.rid,
                    workflow_id,
                    metadata.invocation_id,
                    metadata.context_id,
                    0,
                    float(suffix),
                    1,
                    1,
                    10,
                )
            )
            return request

        queue = [untagged, req("wf-a", "1"), req("wf-a", "2"), req("wf-b", "1"), req("wf-b", "2")]
        original = list(queue)
        candidate_view = runtime.begin_prefill_epoch(
            queue,
            SimpleNamespace(
                rem_input_tokens=100,
                rem_chunk_tokens=None,
                rem_total_tokens=100,
            ),
            max_requests=4,
        )
        self.assertEqual(queue, original)
        self.assertIs(candidate_view[0], untagged)
        self.assertEqual(len(runtime._current_ticket_epoch.tickets), 4)
        self.assertEqual(
            Counter(
                ticket.workflow_id
                for ticket in runtime._current_ticket_epoch.tickets
            ),
            {"wf-a": 2, "wf-b": 2},
        )
        selected = []
        for request in candidate_view:
            if request is untagged:
                continue
            self.assertTrue(runtime.admission_ticket_allows(request))
            self.assertTrue(
                runtime.validate_admission_ticket_after_prefix(request)
            )
            runtime.on_prefill_candidate_result(
                request, admitted=True, result="CONTINUE"
            )
            selected.append(request)
        runtime.end_prefill_epoch(selected)
        self.assertEqual(len(controller.visible_admission.entries()), 0)

    def test_observed_admission_holds_new_tickets_at_active_kv_watermark(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=1000,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=10,
            observed_admission_scheduling_enabled=True,
            observed_admission_active_kv_high_watermark_ratio=0.8,
            observed_admission_min_active_requests=1,
        )
        controller = BeliefKVController(config)
        controller.process_runtime_event(
            RuntimeEvent(
                event_id="wf-start",
                ts_ms=0,
                kind=RuntimeEventKind.WORKFLOW_START,
                workflow_id="wf",
            )
        )
        for suffix in ("a", "b"):
            controller.process_runtime_event(
                RuntimeEvent(
                    event_id=f"inv-{suffix}",
                    ts_ms=1,
                    kind=RuntimeEventKind.INVOCATION_CREATE,
                    workflow_id="wf",
                    invocation_id=f"inv-{suffix}",
                    context_id=f"ctx-{suffix}",
                    context_epoch=0,
                )
            )
        locked = PageHandle(99, 0)
        controller.page_index.register_page(locked, size_bytes=800)
        controller.page_index.set_engine_lock(locked, 1)

        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.controller = controller
        runtime.config = config
        runtime._now_ms = lambda: 10.0
        runtime.audit = _AuditRecorder()
        runtime._admission_epoch = 0
        runtime._current_ticket_epoch = None
        runtime._current_tickets_by_request = {}
        runtime._ticket_attempted_request_ids = set()
        runtime._ticket_selected_request_ids = set()
        runtime._ticket_skip_audit = set()
        runtime._ticket_selection_details = {}
        runtime._ticket_native_rejections = {}
        runtime._pending_h2d_contexts = set()
        runtime._request_metadata_by_id = {}
        runtime._request_submitted_ts_by_id = {}
        running = SimpleNamespace(
            rid="native-running",
            fill_ids=(),
            prefix_indices=(),
            origin_input_ids=(),
            output_ids=(),
        )
        runtime.scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[running]),
            chunked_req=None,
        )

        def request(suffix):
            metadata = BeliefKVRequestMetadata(
                "wf", f"inv-{suffix}", f"ctx-{suffix}", 0
            )
            item = SimpleNamespace(
                rid=f"request-{suffix}",
                beliefkv_metadata=metadata,
                origin_input_ids=_NoBooleanSequence(1),
                prefix_indices=_NoBooleanSequence(0),
            )
            runtime._request_metadata_by_id[item.rid] = metadata
            controller.register_visible_request(
                AdmissionRequest(
                    item.rid,
                    "wf",
                    metadata.invocation_id,
                    metadata.context_id,
                    0,
                    1.0,
                    1,
                    1,
                    10,
                )
            )
            return item

        queue = [request("a"), request("b")]
        adder = SimpleNamespace(
            rem_input_tokens=100,
            rem_chunk_tokens=None,
            rem_total_tokens=100,
        )
        runtime.begin_prefill_epoch(queue, adder, max_requests=2)

        self.assertEqual(runtime._current_ticket_epoch.source, "observed_active_set")
        self.assertEqual(runtime._current_ticket_epoch.tickets, ())
        self.assertEqual(
            runtime._current_observed_admission_window.mode,
            "active_kv_pressure_hold",
        )
        started = [
            fields
            for event, _, fields in runtime.audit.events
            if event == "admission_ticket_epoch_started"
        ][-1]
        self.assertEqual(
            started["observed_admission_window"]["active_kv_footprint_bytes"],
            800,
        )
        runtime.end_prefill_epoch(())

        controller.page_index.set_engine_lock(locked, 0)
        runtime.begin_prefill_epoch(queue, adder, max_requests=2)

        self.assertEqual(
            [
                ticket.request_id
                for ticket in runtime._current_ticket_epoch.tickets
            ],
            ["request-a", "request-b"],
        )
        self.assertEqual(
            runtime._current_observed_admission_window.mode,
            "active_kv_bounded",
        )
        runtime.end_prefill_epoch(())

        def fail_snapshot(**_):
            raise RuntimeError("synthetic observer failure")

        runtime._observed_admission_snapshot = fail_snapshot
        runtime.begin_prefill_epoch(queue, adder, max_requests=2)
        self.assertEqual(
            runtime._current_ticket_epoch.source,
            "observed_active_set_fallback",
        )
        self.assertEqual(len(runtime._current_ticket_epoch.tickets), 2)
        self.assertIn(
            "observed_admission_fallback",
            [event for event, _, _ in runtime.audit.events],
        )
        runtime.end_prefill_epoch(())

    def test_selective_retraction_requeue_stays_blocked_until_cooldown(self):
        config = BeliefKVConfig(
            observed_admission_scheduling_enabled=True,
            running_batch_retraction_enabled=True,
        )
        controller = BeliefKVController(config)
        controller.process_runtime_event(
            RuntimeEvent(
                event_id="wf-start-retraction",
                ts_ms=0,
                kind=RuntimeEventKind.WORKFLOW_START,
                workflow_id="wf-retraction",
            )
        )
        controller.process_runtime_event(
            RuntimeEvent(
                event_id="inv-start-retraction",
                ts_ms=1,
                kind=RuntimeEventKind.INVOCATION_CREATE,
                workflow_id="wf-retraction",
                invocation_id="inv-retraction",
                context_id="ctx-retraction",
                context_epoch=0,
            )
        )
        metadata = BeliefKVRequestMetadata(
            "wf-retraction", "inv-retraction", "ctx-retraction", 0
        )
        now_ms = [1000.0]
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.controller = controller
        runtime.config = config
        runtime.tree_cache = _TreeCache()
        runtime.audit = _AuditRecorder()
        runtime._now_ms = lambda: now_ms[0]
        runtime._pending_h2d_contexts = set()
        runtime._request_metadata_by_id = {"victim": metadata}
        runtime._request_submitted_ts_by_id = {"victim": 10.0}
        runtime._pending_selective_retraction_ids = {"victim"}
        runtime._retraction_cooldown_until_by_request = {"victim": 2000.0}
        request = SimpleNamespace(
            rid="victim",
            beliefkv_metadata=metadata,
            origin_input_ids=(1, 2),
            output_ids=(3,),
            prefix_indices=(1,),
            sampling_params=SimpleNamespace(max_new_tokens=8),
            last_node=None,
            init_next_round_input=lambda _cache: None,
        )

        runtime.on_requests_requeued((request,), is_retracted=True)
        entry = controller.visible_admission.get("victim")
        self.assertEqual(entry.state, AdmissionSideState.POLICY_BLOCKED)
        self.assertEqual(entry.blocker_reason, "retraction_cooldown")

        now_ms[0] = 2001.0
        runtime._sync_visible_gate_state("victim", metadata, req=request)
        entry = controller.visible_admission.get("victim")
        self.assertEqual(entry.state, AdmissionSideState.VISIBLE_PENDING)

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
        runtime._charge_previous_batch(40.0)
        self.assertEqual(
            runtime.controller.fairness.accounts["wf-b"].attained_service_ms,
            15.0,
        )
        self.assertIsNone(runtime._last_batch_selected_ms)
        self.assertEqual(runtime._last_batch_workflow_counts, {})

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
