import json
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import Counter, deque
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
from beliefkv.policy.online_joint import (
    OnlineJointPlanDecision,
    OnlineJointPlanView,
)
from beliefkv.policy.reference import ResidencyAction, RunnableInvocation
from beliefkv.policy.resource_snapshot import RuntimeResourceObservation
from beliefkv.runtime.audit import PolicySnapshotLog
from beliefkv.runtime.joint_shadow import (
    IncrementalPolicyInputAssembler,
    LatestWinsJointPlanWorker,
)
from beliefkv.runtime.lock_service import RequestServiceLedger
from beliefkv.runtime.page_index import PageOwnershipIndex
from beliefkv.runtime.protocol import (
    CommandAck,
    CommandKind,
    CommandStatus,
    ControlCommand,
    EnqueueOutcome,
    EnqueueStatus,
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
    close_runtime_with_signal_shield,
    install_scheduler_shutdown_handler,
)
from beliefkv.runtime.restore_obligation import (
    RestoreAuthorityMode,
    RestoreLeaseState,
    RestoreObligationCause,
    RestoreObligationIndex,
    RestoreObligationState,
    RestoreTransactionStage,
    SafePointPhysicalPhase,
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
        self.beliefkv_transfer_metadata = {"d2h": {}, "h2d": {}}
        self.token_to_kv_pool_host = SimpleNamespace(
            layout="layer_first", pin_memory=True
        )
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

    def inc_lock_ref(self, node):
        current = node
        while current is not None and current is not self.root_node:
            current.lock_ref += 1
            current = current.parent
        return 0

    def dec_lock_ref(self, node):
        current = node
        while current is not None and current is not self.root_node:
            current.lock_ref -= 1
            current = current.parent
        return 0

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
        self._next_token = 1

    def available_size(self):
        return self.available_tokens

    def alloc(self, need_size):
        if need_size > self.available_tokens:
            return None
        result = list(range(self._next_token, self._next_token + need_size))
        self._next_token += need_size
        self.available_tokens -= need_size
        return result

    def free(self, indices):
        self.available_tokens += len(indices)


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

    def emit(self, event, ts_ms, **fields):
        self.events.append((event, ts_ms, fields))


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

    def scheduler_step(
        self,
        now_ms,
        *,
        drain_acks,
        allow_reactive_transfer=True,
    ):
        self.drain_acks_argument = drain_acks
        self.allow_reactive_transfer = allow_reactive_transfer
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
            "host_copy_state": "present",
            "pinned_host": True,
            "allocator_submit_ms": 0.25,
            "native_inflight_operation_count": 2,
            "native_inflight_token_count": 8,
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
        self.assertEqual(fields["host_copy_state"], "present")
        self.assertIs(fields["pinned_host"], True)
        self.assertEqual(fields["native_concurrent_bytes"], 80)
        self.assertEqual(fields["allocator_submit_ms"], 0.25)
        self.assertEqual(fields["callback_overhead_ms"], 10.0)
        self.assertEqual(
            fields["native_inflight_operation_count_at_submit"], 2
        )
        self.assertFalse(fields["start_timestamp_observed"])

    def test_online_joint_plan_requires_shadow_observed_worker(self):
        scheduler = SimpleNamespace(
            enable_hierarchical_cache=True,
            tree_cache=_TreeCache(),
        )

        with self.assertRaisesRegex(
            SGLangBackendError,
            "online observed JointPlan requires its validated shadow worker",
        ):
            EmbeddedSGLangRuntime(
                scheduler,
                config=BeliefKVConfig(
                    joint_policy_enabled=True,
                    joint_policy_shadow_mode=False,
                ),
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
                output_ids=[],
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
        for request in requests:
            request.output_ids.append(1)
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
        self.assertEqual(
            [item["request_id"] for item in fields["request_samples"]],
            ["request-0", "request-1"],
        )
        self.assertEqual(
            [item["token_delta"] for item in fields["request_samples"]],
            [1, 1],
        )
        self.assertTrue(
            all(
                item["token_delta_semantics"] == "observed_output_ids_delta"
                for item in fields["request_samples"]
            )
        )
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

    def test_abort_race_preserves_physical_start_until_late_finish(self):
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
        runtime._aborted_request_physical_start_by_id = {}
        runtime._pending_request_physical_finish_by_id = {}
        runtime._request_submitted_ts_by_id = {"request": 1.0}
        runtime._queue_timeout_request_ids = set()
        runtime._execution_timeout_request_ids = set()
        runtime._retracted_engine_request_ids = set()
        runtime._pending_selective_retraction_ids = set()
        runtime._retraction_cooldown_until_by_request = {}
        runtime._ordinary_restore_capacity_waiters = set()
        runtime._finish_restore_obligation = lambda *args, **kwargs: None
        runtime._cancel_restore_service_grace = lambda *args, **kwargs: None
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        runtime._request_metadata_by_id = {"request": metadata}
        request = SimpleNamespace(
            rid="request",
            origin_input_ids=_NoBooleanSequence(10),
            prefix_indices=_NoBooleanSequence(5),
        )

        runtime._capture_request_physical_start(request, metadata, 10.0)
        runtime.on_abort_request(SimpleNamespace(abort_all=False, rid="request"))
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
        self.assertTrue(fields["request_aborted_during_finish"])
        self.assertEqual(fields["request_abort_reason"], "request_aborted")
        self.assertEqual(fields["context_path_growth_bytes"], 40)
        self.assertNotIn(
            "request", runtime._aborted_request_physical_start_by_id
        )

    def test_execution_timeout_falls_back_to_physical_start_without_ledger(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        metadata = BeliefKVRequestMetadata(
            "wf",
            "inv",
            "ctx",
            0,
            execution_timeout_s=2.0,
        )
        aborted = []
        runtime.scheduler = SimpleNamespace(
            abort_request=lambda request: aborted.append(request)
        )
        runtime.audit = _AuditRecorder()
        runtime._request_metadata_by_id = {"request": metadata}
        runtime._request_submitted_ts_by_id = {"request": -100_000.0}
        runtime._request_physical_start_by_id = {
            "request": {"checkpoint_ts_ms": 1_000.0}
        }
        runtime._execution_timeout_request_ids = set()
        runtime._terminal_cancelled_request_ids = set()

        runtime._enforce_execution_timeouts(now_ms=2_999.0)
        self.assertEqual(aborted, [])

        runtime._enforce_execution_timeouts(now_ms=3_001.0)

        self.assertEqual(len(aborted), 1)
        self.assertEqual(aborted[0].rid, "request")
        self.assertIn("request", runtime._terminal_cancelled_request_ids)
        timeout_events = [
            fields
            for event, _, fields in runtime.audit.events
            if event == "request_execution_timeout"
        ]
        self.assertEqual(timeout_events[0]["execution_started_ts_ms"], 1_000.0)
        self.assertEqual(
            timeout_events[0]["timeout_scope"],
            "physical_start_to_scheduler_abort_fallback",
        )

    def test_execution_timeout_uses_last_completed_gpu_service(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        metadata = BeliefKVRequestMetadata(
            "wf",
            "inv",
            "ctx",
            0,
            execution_timeout_s=2.0,
        )
        aborted = []
        runtime.scheduler = SimpleNamespace(
            abort_request=lambda request: aborted.append(request)
        )
        runtime.audit = _AuditRecorder()
        runtime._request_metadata_by_id = {"request": metadata}
        runtime._request_physical_start_by_id = {
            "request": {"checkpoint_ts_ms": 1_000.0}
        }
        runtime._execution_timeout_request_ids = set()
        runtime._terminal_cancelled_request_ids = set()
        runtime._lock_service_ledger = RequestServiceLedger()
        runtime._lock_service_ledger.observe_selected(
            request_id="request",
            workflow_id="wf",
            invocation_id="inv",
            context_id="ctx",
            ts_ms=1_000.0,
        )
        runtime._lock_service_ledger.observe_completed(
            "request",
            ts_ms=2_500.0,
            phase="decode",
        )

        runtime._enforce_execution_timeouts(now_ms=3_001.0)
        self.assertEqual(aborted, [])

        runtime._enforce_execution_timeouts(now_ms=4_501.0)

        self.assertEqual(len(aborted), 1)
        timeout_events = [
            fields
            for event, _, fields in runtime.audit.events
            if event == "request_execution_timeout"
        ]
        self.assertEqual(timeout_events[0]["execution_elapsed_ms"], 3_501.0)
        self.assertEqual(timeout_events[0]["gpu_service_no_progress_ms"], 2_001.0)
        self.assertEqual(timeout_events[0]["last_gpu_service_ts_ms"], 2_500.0)
        self.assertEqual(timeout_events[0]["completed_gpu_service_count"], 1)
        self.assertEqual(
            timeout_events[0]["timeout_scope"],
            "last_completed_gpu_service_to_scheduler_abort",
        )

    def test_queue_timeout_excludes_requests_that_started_execution(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(request_queue_timeout_ms=2_000.0)
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        aborted = []
        runtime.scheduler = SimpleNamespace(
            abort_request=lambda request: aborted.append(request)
        )
        runtime.audit = _AuditRecorder()
        runtime._request_metadata_by_id = {
            "queued": metadata,
            "started": metadata,
        }
        runtime._request_submitted_ts_by_id = {
            "queued": 1_000.0,
            "started": 1_000.0,
        }
        runtime._request_physical_start_by_id = {
            "started": {"checkpoint_ts_ms": 1_500.0}
        }
        runtime._queue_timeout_request_ids = set()
        runtime._terminal_cancelled_request_ids = set()

        runtime._enforce_queue_timeouts(now_ms=3_001.0)

        self.assertEqual([item.rid for item in aborted], ["queued"])
        self.assertIn("queued", runtime._terminal_cancelled_request_ids)
        self.assertNotIn("started", runtime._terminal_cancelled_request_ids)

    def test_final_runtime_summary_exposes_joint_correctness_gates(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.scheduler = SimpleNamespace(
            waiting_queue=(),
            running_batch=None,
            chunked_req=None,
        )
        runtime.controller = BeliefKVController()
        runtime.audit = _AuditRecorder()
        runtime._shutdown_state = "acknowledged"
        runtime._online_joint_counts = Counter()
        runtime._running_retraction_counts = Counter()
        runtime._pending_online_joint_residency = None
        runtime._pending_running_retraction_transaction = None
        runtime._current_online_joint_view = None

        payload = runtime._runtime_summary_payload(now_ms=10.0, final=True)

        self.assertTrue(
            payload["correctness_gates"][
                "all_online_actions_have_source_joint_plan_id"
            ]
        )
        self.assertTrue(
            payload["correctness_gates"]["no_pending_transactions"]
        )
        self.assertTrue(
            payload["correctness_gates"]["shutdown_summary_complete"]
        )
        self.assertTrue(
            payload["correctness_gates"]
            ["all_non_user_cancelled_obligations_satisfied"]
        )
        self.assertTrue(
            payload["correctness_gates"]
            ["shutdown_cleanup_did_not_mask_unresolved_transactions"]
        )

        runtime._shutdown_prepare_transaction_snapshot = {
            "active_obligation_ids": ["restore-1"]
        }
        payload = runtime._runtime_summary_payload(now_ms=11.0, final=True)
        self.assertFalse(
            payload["correctness_gates"]
            ["shutdown_cleanup_did_not_mask_unresolved_transactions"]
        )

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
            ],
            running_batch=SimpleNamespace(
                reqs=[
                    SimpleNamespace(
                        rid="running-request",
                        beliefkv_metadata=BeliefKVRequestMetadata(
                            "wf", "running-inv", "running-ctx", 0
                        ),
                        sampling_params=SimpleNamespace(max_new_tokens=20),
                        output_ids=_NoBooleanSequence(4),
                        origin_input_ids=_NoBooleanSequence(8),
                        prefix_indices=_NoBooleanSequence(8),
                    )
                ]
            ),
        )

        runnable = runtime._policy_runtime_runnable(100.0)

        self.assertEqual(len(runnable), 2)
        by_request = {item.request_id: item for item in runnable}
        self.assertEqual(by_request["request"].startup_bytes, 220)
        self.assertTrue(
            by_request["request"].causal_class.startswith("engine_waiting:")
        )
        self.assertEqual(by_request["running-request"].startup_bytes, 160)
        self.assertTrue(
            by_request["running-request"].causal_class.startswith(
                "engine_running:"
            )
        )

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
        self.assertFalse(runtime._context_has_engine_request("ctx"))
        request.req_pool_idx = 7
        # Native ownership is immutable inside an epoch; the next safe point sees it.
        runtime._finish_physical_safe_point()
        self.assertTrue(runtime._context_has_engine_request("ctx"))

    def test_native_and_explicit_load_ownership_are_both_visible(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        request = SimpleNamespace(
            rid="request",
            beliefkv_metadata=metadata,
            load_operation_id="native-load-1",
        )
        runtime._request_metadata_by_id = {"request": metadata}
        runtime._request_submitted_ts_by_id = {"request": 1.0}
        runtime._h2d_context_by_command = {
            "explicit-h2d-1": ("ctx", ("page:1:0",))
        }
        runtime.scheduler = SimpleNamespace(
            waiting_queue=[request],
            running_batch=SimpleNamespace(reqs=[]),
            chunked_req=None,
        )

        snapshot = runtime._context_physical_snapshots("ctx")[0]

        self.assertEqual(snapshot.native_load_operation_id, "native-load-1")
        self.assertEqual(snapshot.explicit_transfer_ids, ("explicit-h2d-1",))
        self.assertTrue(runtime._context_has_engine_request("ctx"))

    def test_native_snapshot_is_built_lazily_once_per_capture_epoch(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        first_metadata = BeliefKVRequestMetadata("wf", "inv-1", "ctx-1", 0)
        second_metadata = BeliefKVRequestMetadata("wf", "inv-2", "ctx-2", 0)
        first = SimpleNamespace(rid="request-1", req_pool_idx=1)
        second = SimpleNamespace(rid="request-2")
        runtime._request_metadata_by_id = {
            "request-1": first_metadata,
            "request-2": second_metadata,
        }
        runtime._request_submitted_ts_by_id = {
            "request-1": 1.0,
            "request-2": 2.0,
        }
        runtime.scheduler = SimpleNamespace(
            waiting_queue=[first, second],
            running_batch=SimpleNamespace(reqs=[]),
            chunked_req=None,
        )
        runtime._begin_physical_safe_point_apply_events()
        runtime._begin_physical_safe_point_capture_and_plan()

        self.assertEqual(len(runtime._context_physical_snapshots("ctx-1")), 1)
        self.assertEqual(len(runtime._context_physical_snapshots("ctx-2")), 1)
        self.assertEqual(runtime._native_physical_snapshot_counts["call_count"], 1)
        self.assertEqual(
            runtime._native_physical_snapshot_counts["cache_hit_count"], 1
        )

    def test_native_snapshot_commit_rejects_changed_request_ownership(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        request = SimpleNamespace(rid="request")
        runtime._request_metadata_by_id = {"request": metadata}
        runtime._request_submitted_ts_by_id = {"request": 1.0}
        runtime.scheduler = SimpleNamespace(
            waiting_queue=[request],
            running_batch=SimpleNamespace(reqs=[]),
            chunked_req=None,
        )
        runtime._begin_physical_safe_point_apply_events()
        runtime._begin_physical_safe_point_capture_and_plan()
        runtime._context_physical_snapshots("ctx")
        captured_epoch = runtime._safe_point_physical_epoch_sequence

        request.req_pool_idx = 9
        self.assertFalse(runtime._begin_physical_transactional_commit("ctx"))
        self.assertGreater(runtime._safe_point_physical_epoch_sequence, captured_epoch)
        self.assertEqual(
            runtime._safe_point_physical_phase,
            SafePointPhysicalPhase.CAPTURE_AND_PLAN,
        )
        self.assertEqual(
            runtime._native_physical_snapshot_counts["commit_readset_stale"], 1
        )
        self.assertEqual(
            runtime._context_physical_snapshots("ctx")[0].req_pool_slot, 9
        )

    def test_native_snapshot_commit_rejects_request_id_reuse(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        request = SimpleNamespace(rid="request")
        runtime._request_metadata_by_id = {
            "request": BeliefKVRequestMetadata("wf", "old", "old-ctx", 0)
        }
        runtime._request_submitted_ts_by_id = {"request": 1.0}
        runtime.scheduler = SimpleNamespace(
            waiting_queue=[request],
            running_batch=SimpleNamespace(reqs=[]),
            chunked_req=None,
        )
        runtime._context_physical_snapshots("old-ctx")

        runtime._request_metadata_by_id["request"] = BeliefKVRequestMetadata(
            "wf", "new", "new-ctx", 1
        )
        runtime._request_submitted_ts_by_id["request"] = 2.0

        self.assertFalse(
            runtime._begin_physical_transactional_commit("old-ctx")
        )

    def test_native_snapshot_commit_rejects_explicit_operation_change(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        runtime._request_metadata_by_id = {"request": metadata}
        runtime._request_submitted_ts_by_id = {"request": 1.0}
        runtime._h2d_context_by_command = {}
        runtime.scheduler = SimpleNamespace(
            waiting_queue=[SimpleNamespace(rid="request")],
            running_batch=SimpleNamespace(reqs=[]),
            chunked_req=None,
        )
        runtime._context_physical_snapshots("ctx")
        runtime._h2d_context_by_command["h2d"] = ("ctx", ("page:1:0",))

        self.assertFalse(runtime._begin_physical_transactional_commit("ctx"))

    def test_native_snapshot_high_cardinality_index_is_complete(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        requests = [SimpleNamespace(rid=f"request-{index:03d}") for index in range(128)]
        runtime._request_metadata_by_id = {
            request.rid: BeliefKVRequestMetadata(
                f"wf-{index:03d}",
                f"inv-{index:03d}",
                f"ctx-{index:03d}",
                0,
            )
            for index, request in enumerate(requests)
        }
        runtime._request_submitted_ts_by_id = {
            request.rid: float(index) for index, request in enumerate(requests)
        }
        runtime.scheduler = SimpleNamespace(
            waiting_queue=requests[:64],
            running_batch=SimpleNamespace(reqs=requests[64:]),
            chunked_req=None,
        )

        for index in range(128):
            snapshots = runtime._context_physical_snapshots(f"ctx-{index:03d}")
            self.assertEqual(len(snapshots), 1)
        self.assertEqual(runtime._native_physical_snapshot_counts["call_count"], 1)
        self.assertEqual(
            runtime._native_physical_snapshot_counts["cache_hit_count"], 127
        )

    def test_planning_snapshot_is_forbidden_after_commit_starts(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime._request_metadata_by_id = {}
        runtime.scheduler = SimpleNamespace(
            waiting_queue=[],
            running_batch=SimpleNamespace(reqs=[]),
            chunked_req=None,
        )
        runtime._begin_physical_safe_point_apply_events()
        runtime._begin_physical_safe_point_capture_and_plan()
        self.assertTrue(runtime._begin_physical_transactional_commit())

        with self.assertRaisesRegex(RuntimeError, "after transactional commit"):
            runtime._context_physical_snapshots("ctx")

    def test_guard_blocked_restore_does_not_allocate_a_lease(self):
        config = BeliefKVConfig(kv_bytes_per_token=10)
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = BeliefKVController(config)
        runtime.audit = _AuditRecorder()
        runtime._restore_obligation_counts = Counter()
        runtime._restore_command_sequence = 0
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        request = SimpleNamespace(rid="request", last_node=None)
        runtime._request_metadata_by_id = {"request": metadata}
        runtime._request_submitted_ts_by_id = {"request": 1.0}
        runtime.scheduler = SimpleNamespace(
            waiting_queue=[request],
            running_batch=SimpleNamespace(reqs=[]),
            chunked_req=None,
        )
        obligation = runtime._restore_obligation_index().create(
            request_id="request",
            workflow_id="wf",
            invocation_id="inv",
            context_id="ctx",
            context_epoch=0,
            source_retraction_transaction_id="retraction-1",
            source_joint_plan_id="joint-1",
            created_ts_ms=1.0,
            path_extent_ids=("page:1:0",),
        )
        obligation.source_transaction_terminal = True
        obligation.requeued = True
        bundle = SimpleNamespace(
            generation_fingerprint="closure-1",
            scope=SimpleNamespace(value="exclusive_suffix"),
            bundle_id="bundle-1",
            closure_bytes=100,
            marginal_reclaimable_bytes=0,
            exclusive_action_bytes=100,
            cross_context_action_bytes=0,
            foreign_owner_context_ids=(),
        )
        preview = SimpleNamespace(
            eligible=True,
            copy_bytes=100,
            bundle=bundle,
            context_id="ctx",
            context_epoch=0,
            command_kind=CommandKind.PREFETCH_CONTEXT,
            blockers=(),
            intent=lambda: None,
        )
        runtime._refresh_restore_obligation = lambda *_args, **_kwargs: (
            request,
            ("page:1:0",),
        )
        runtime._restore_attempt_stamp = lambda: (1,)
        runtime._allocator_available_bytes = lambda: 1000
        runtime._restore_h2d_previews = lambda *_args, **_kwargs: (preview,)
        runtime.controller.preflight_control_command = lambda command: EnqueueOutcome(
            status=EnqueueStatus.RETRY_GUARD_BLOCKED,
            canonical_command_id=None,
            attempt_key=("ctx", 0, command.kind.value),
            blocker_codes=("engine_busy",),
            wake_conditions=("engine_owner_changed",),
        )
        runtime.controller.transfer_guard.generation_for = lambda _command: 1
        runtime._grant_restore_lease = mock.Mock()

        runtime._drive_restore_obligations(now_ms=2.0)

        runtime._grant_restore_lease.assert_not_called()
        self.assertEqual(obligation.blocker_codes, ("engine_busy",))

    def test_restore_enqueue_failure_rolls_back_prepared_lease_and_pin(self):
        config = BeliefKVConfig(kv_bytes_per_token=10)
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = BeliefKVController(config)
        runtime.audit = _AuditRecorder()
        runtime._restore_obligation_counts = Counter()
        runtime._restore_command_sequence = 0
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        request = SimpleNamespace(rid="request", last_node=None)
        runtime._request_metadata_by_id = {"request": metadata}
        runtime._request_submitted_ts_by_id = {"request": 1.0}
        runtime.scheduler = SimpleNamespace(
            waiting_queue=[request],
            running_batch=SimpleNamespace(reqs=[]),
            chunked_req=None,
        )
        obligation = runtime._restore_obligation_index().create(
            request_id="request",
            workflow_id="wf",
            invocation_id="inv",
            context_id="ctx",
            context_epoch=0,
            source_retraction_transaction_id="retraction-1",
            source_joint_plan_id="joint-1",
            created_ts_ms=1.0,
            path_extent_ids=("page:1:0",),
        )
        obligation.source_transaction_terminal = True
        obligation.requeued = True
        bundle = SimpleNamespace(
            generation_fingerprint="closure-1",
            scope=SimpleNamespace(value="exclusive_suffix"),
            bundle_id="bundle-1",
            closure_bytes=100,
            marginal_reclaimable_bytes=0,
            exclusive_action_bytes=100,
            cross_context_action_bytes=0,
            foreign_owner_context_ids=(),
        )
        preview = SimpleNamespace(
            eligible=True,
            copy_bytes=100,
            bundle=bundle,
            context_id="ctx",
            context_epoch=0,
            command_kind=CommandKind.PREFETCH_CONTEXT,
            blockers=(),
            intent=lambda: None,
        )
        runtime._refresh_restore_obligation = lambda *_args, **_kwargs: (
            request,
            ("page:1:0",),
        )
        runtime._restore_attempt_stamp = lambda: (1,)
        runtime._allocator_available_bytes = lambda: 1000
        runtime._restore_h2d_previews = lambda *_args, **_kwargs: (preview,)
        runtime.controller.preflight_control_command = lambda command: EnqueueOutcome(
            status=EnqueueStatus.ENQUEUED,
            canonical_command_id=command.command_id,
            attempt_key=("ctx", 0, command.kind.value),
        )
        runtime.controller.transfer_guard.generation_for = lambda _command: 0
        events = []
        lease = SimpleNamespace(lease_id="lease-1")
        runtime._grant_restore_lease = lambda *_args, **_kwargs: (
            events.append("reserve") or lease
        )
        runtime._pin_restore_lease_prefix = lambda *_args, **_kwargs: (
            events.append("pin") or True
        )
        runtime._queue_restore_obligation_command = lambda *_args, **_kwargs: (
            events.append("enqueue")
            or EnqueueOutcome(
                status=EnqueueStatus.CONTEXT_CONFLICT,
                canonical_command_id="other-command",
                attempt_key=("ctx", 0, "prefetch_context"),
                blocker_codes=("context_command_owned",),
                wake_conditions=("command_terminal:other-command",),
            )
        )
        runtime._release_restore_lease = lambda *_args, **_kwargs: events.append(
            "rollback"
        )

        runtime._drive_restore_obligations(now_ms=2.0)

        self.assertEqual(events, ["reserve", "pin", "enqueue", "rollback"])
        transaction = runtime._restore_transactions["request"]
        self.assertIsNone(transaction.capacity_reservation_id)
        self.assertIsNone(transaction.prefix_pin_token)
        self.assertIsNone(obligation.pending_command_id)

    def test_canonical_restore_ack_notifies_all_subscribers(self):
        config = BeliefKVConfig(restore_lease_enabled=False)
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.audit = _AuditRecorder()
        runtime._restore_obligation_counts = Counter()
        runtime._restore_command_to_request = {"canonical": {"first", "second"}}
        runtime._restore_funding_target_by_command = {}
        index = runtime._restore_obligation_index()
        for request_id in ("first", "second"):
            obligation = index.create(
                request_id=request_id,
                workflow_id=f"wf-{request_id}",
                invocation_id=f"inv-{request_id}",
                context_id="ctx",
                context_epoch=0,
                source_retraction_transaction_id=f"retraction-{request_id}",
                source_joint_plan_id=f"joint-{request_id}",
                created_ts_ms=1.0,
                path_extent_ids=("page:1:0",),
            )
            obligation.start_command(
                "canonical",
                CommandKind.PREFETCH_CONTEXT,
                now_ms=1.0,
                attempt_stamp=(1,),
            )

        runtime._advance_restore_obligations(
            (
                CommandAck(
                    "canonical",
                    CommandStatus.COMPLETED,
                    2.0,
                    actual_bytes=100,
                ),
            ),
            now_ms=2.0,
        )

        self.assertEqual(
            index.get("first").state, RestoreObligationState.RESTORE_ACKED
        )
        self.assertEqual(
            index.get("second").state, RestoreObligationState.RESTORE_ACKED
        )
        self.assertNotIn("canonical", runtime._restore_command_to_request)

    def test_noncomplete_restore_acks_park_transaction_for_external_event(self):
        for status in (
            CommandStatus.PARTIAL,
            CommandStatus.REJECTED,
            CommandStatus.STALE,
        ):
            with self.subTest(status=status.value):
                config = BeliefKVConfig(restore_lease_enabled=False)
                runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
                runtime.config = config
                runtime.audit = _AuditRecorder()
                runtime._restore_obligation_counts = Counter()
                runtime._restore_command_to_request = {
                    status.value: {"request"}
                }
                runtime._restore_funding_target_by_command = {}
                obligation = runtime._restore_obligation_index().create(
                    request_id="request",
                    workflow_id="wf",
                    invocation_id="inv",
                    context_id="ctx",
                    context_epoch=0,
                    source_retraction_transaction_id="retraction-1",
                    source_joint_plan_id="joint-1",
                    created_ts_ms=1.0,
                    path_extent_ids=("page:1:0",),
                )
                obligation.start_command(
                    status.value,
                    CommandKind.PREFETCH_CONTEXT,
                    now_ms=1.0,
                    attempt_stamp=(1,),
                )

                runtime._advance_restore_obligations(
                    (
                        CommandAck(
                            status.value,
                            status,
                            2.0,
                            actual_bytes=0,
                        ),
                    ),
                    now_ms=2.0,
                )

                transaction = runtime._restore_transactions["request"]
                self.assertEqual(
                    obligation.state, RestoreObligationState.PARKED_WAIT
                )
                self.assertEqual(
                    transaction.stage, RestoreTransactionStage.WAIT_EVENT
                )

    def test_restore_drain_acquires_and_releases_exclusive_authority(self):
        config = BeliefKVConfig()
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = BeliefKVController(config)
        runtime.audit = _AuditRecorder()
        obligation = runtime._restore_obligation_index().create(
            request_id="request",
            workflow_id="wf",
            invocation_id="inv",
            context_id="ctx",
            context_epoch=0,
            source_retraction_transaction_id="retraction-1",
            source_joint_plan_id="joint-1",
            created_ts_ms=1.0,
            path_extent_ids=(),
        )
        runtime._restore_authority_mode = (
            RestoreAuthorityMode.RESTORE_DRAIN_REQUESTED
        )
        runtime._restore_authority_request_id = "request"

        runtime._advance_restore_authority(now_ms=2.0)
        self.assertEqual(
            runtime._restore_authority_mode,
            RestoreAuthorityMode.RESTORE_DRAIN_ACTIVE,
        )

        obligation.finish(
            RestoreObligationState.CANCELLED,
            now_ms=3.0,
            reason="test_cancel",
        )
        runtime._advance_restore_authority(now_ms=3.0)
        self.assertEqual(
            runtime._restore_authority_mode, RestoreAuthorityMode.NORMAL_JOINT
        )

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

    def test_h2d_dependency_blocks_only_requests_whose_radix_path_intersects(self):
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
        runtime.registry = SGLangNodeRegistry()
        runtime.tree_cache = _TreeCache()
        runtime.config = BeliefKVConfig(kv_bytes_per_token=16)
        matched = _Node(11)
        matched.parent = runtime.tree_cache.root_node
        unrelated = _Node(12)
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
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        requests = [
            SimpleNamespace(rid="matched", last_node=matched),
            SimpleNamespace(rid="unrelated", last_node=unrelated),
        ]
        runtime.scheduler = SimpleNamespace(waiting_queue=requests)
        for request in requests:
            runtime.controller.register_visible_request(
                AdmissionRequest(
                    request.rid,
                    "wf",
                    "inv",
                    "ctx",
                    0,
                    10.0,
                    1,
                    1,
                    16,
                )
            )

        runtime._mark_h2d_waiters(
            "ctx",
            restored_extent_ids=(
                f"page:{matched_handle.page_id}:"
                f"{matched_handle.allocation_generation}",
            ),
            reason="h2d_inflight",
        )

        matched_entry = runtime.controller.visible_admission.get("matched")
        unrelated_entry = runtime.controller.visible_admission.get("unrelated")
        self.assertEqual(matched_entry.state, AdmissionSideState.WAIT_RESTORE)
        self.assertEqual(
            unrelated_entry.state, AdmissionSideState.VISIBLE_PENDING
        )

    def test_online_residency_hysteresis_blocks_reclaim_after_recent_restore(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(
            hbm_capacity_bytes=1000,
            reserve_hbm_bytes=100,
            residency_hysteresis_ms=100.0,
            joint_emergency_hbm_ratio=0.98,
        )
        runtime.controller = BeliefKVController(runtime.config)
        runtime.controller.report_hbm_usage(900)
        runtime.audit = _AuditRecorder()
        runtime._online_joint_counts = Counter()
        runtime._online_joint_hysteresis_audit = set()
        runtime._online_joint_last_residency_action = {
            "bundle": (ResidencyAction.PREFETCH_GPU, 10.0)
        }

        blocked = runtime._online_residency_hysteresis_blocks(
            plan_id="plan",
            bundle_id="bundle",
            action=ResidencyAction.COMMIT_CPU,
            now_ms=50.0,
        )
        runtime.controller.report_hbm_usage(990)
        emergency = runtime._online_residency_hysteresis_blocks(
            plan_id="plan-2",
            bundle_id="bundle",
            action=ResidencyAction.COMMIT_CPU,
            now_ms=60.0,
        )

        self.assertTrue(blocked)
        self.assertFalse(emergency)

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

        event, _, fields = next(
            item
            for item in runtime.audit.events
            if item[0] == "controller_timing_summary"
        )
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
        self.assertEqual(
            [item[0] for item in runtime.audit.events],
            ["shutdown_prepare", "shutdown_ack", "runtime_shutdown"],
        )
        self.assertEqual(runtime.audit.close_count, 1)

    def test_scheduler_shutdown_shields_runtime_close_from_repeated_signals(self):
        runtime = SimpleNamespace(close=mock.Mock())
        previous = {
            signal_number: object()
            for signal_number in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT)
        }

        with mock.patch(
            "beliefkv.runtime.sglang_v052rc1.signal.signal",
            side_effect=lambda signum, handler: previous[signum],
        ) as install:
            close_runtime_with_signal_shield(runtime)

        runtime.close.assert_called_once_with()
        self.assertEqual(install.call_count, 6)

    def test_scheduler_sigterm_handler_unwinds_through_finally(self):
        previous = object()
        with mock.patch(
            "beliefkv.runtime.sglang_v052rc1.signal.signal",
            return_value=previous,
        ) as install:
            self.assertIs(install_scheduler_shutdown_handler(), previous)

        signum, handler = install.call_args.args
        self.assertEqual(signum, signal.SIGTERM)
        with self.assertRaises(SystemExit) as caught:
            handler(signal.SIGTERM, None)
        self.assertEqual(caught.exception.code, 0)

    def test_shutdown_ack_is_written_after_audit_close(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime._shutdown_state = "acknowledged"
        runtime.audit = _CloseRecorder()
        runtime.audit.run_id = "run"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime._runtime_scheduler_pid_path = root / "scheduler.pid.json"
            runtime._runtime_shutdown_ack_path = root / "shutdown_ack.json"
            runtime._runtime_summary_path = root / "latest_runtime_summary.json"

            runtime._write_scheduler_identity()
            runtime.audit.close()
            runtime._write_shutdown_ack()

            identity = json.loads(
                runtime._runtime_scheduler_pid_path.read_text(encoding="utf-8")
            )
            ack = json.loads(
                runtime._runtime_shutdown_ack_path.read_text(encoding="utf-8")
            )
        self.assertEqual(identity["pid"], ack["pid"])
        self.assertEqual(
            identity["linux_start_time_ticks"],
            ack["linux_start_time_ticks"],
        )
        self.assertEqual(ack["shutdown_state"], "acknowledged")
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
        self.assertEqual(telemetry.host_copy_state, "missing")
        self.assertIs(telemetry.pinned_host, True)
        self.assertIsNotNone(telemetry.allocator_submit_ms)
        self.assertEqual(
            telemetry.start_timestamp_semantics, "hicache_api_submit_begin"
        )

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

    def test_joint_active_window_defers_but_keeps_inactive_workflow_visible(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=1_000,
            reserve_hbm_bytes=0,
            joint_policy_enabled=True,
            joint_workflow_active_window=1,
        )
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = BeliefKVController(config)
        runtime.controller.fairness.register("wf-a")
        runtime.controller.fairness.register("wf-b")
        runtime.controller.fairness.charge_service("wf-a", 100.0)
        requests = (
            RunnableInvocation(
                request_id="request-a",
                workflow_id="wf-a",
                invocation_id="inv-a",
                context_id="ctx-a",
                context_epoch=0,
                submitted_ts_ms=1.0,
                startup_bytes=10,
            ),
            RunnableInvocation(
                request_id="request-b",
                workflow_id="wf-b",
                invocation_id="inv-b",
                context_id="ctx-b",
                context_epoch=0,
                submitted_ts_ms=1.0,
                startup_bytes=10,
            ),
        )
        runtime._policy_runtime_runnable = lambda _now_ms: requests
        runtime._current_online_joint_decision = None
        runtime._current_joint_plan_epoch = None
        runtime._online_joint_epoch_sequence = 0
        runtime._online_joint_counts = Counter()

        decision = runtime._safe_point_seed_decision(now_ms=10.0)

        self.assertEqual(decision.view.ordered_request_ids, ("request-b",))
        self.assertEqual(decision.view.deferred_request_ids, ("request-a",))
        self.assertEqual(
            set(runtime.controller.fairness.accounts), {"wf-a", "wf-b"}
        )

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

    def test_restore_obligation_supersedes_retraction_cooldown_on_requeue(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=10,
            observed_admission_scheduling_enabled=True,
            running_batch_retraction_enabled=True,
        )
        controller = BeliefKVController(config)
        controller.process_runtime_events(
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
        tree_cache = _TreeCache()
        node = _Node(17)
        node.parent = tree_cache.root_node
        registry = SGLangNodeRegistry()
        handle = registry.register(node)
        controller.page_index.register_page(
            handle,
            size_bytes=100,
            residency=PhysicalResidency.CPU_ONLY,
            radix_depth=1,
        )
        controller.page_index.bind_pages("ctx", 0, (handle,))
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = controller
        runtime.tree_cache = tree_cache
        runtime.registry = registry
        runtime.audit = _AuditRecorder()
        runtime._now_ms = lambda: 1000.0
        runtime._request_metadata_by_id = {"victim": metadata}
        runtime._request_submitted_ts_by_id = {"victim": 10.0}
        runtime._pending_selective_retraction_ids = {"victim"}
        runtime._retraction_cooldown_until_by_request = {"victim": 5000.0}
        obligation = runtime._restore_obligation_index().create(
            request_id="victim",
            workflow_id="wf",
            invocation_id="inv",
            context_id="ctx",
            context_epoch=0,
            source_retraction_transaction_id="retraction-1",
            source_joint_plan_id="joint-1",
            created_ts_ms=900.0,
            path_extent_ids=(f"page:{handle.page_id}:0",),
        )
        obligation.source_transaction_terminal = True
        request = SimpleNamespace(
            rid="victim",
            beliefkv_metadata=metadata,
            origin_input_ids=(1, 2),
            output_ids=(3,),
            prefix_indices=(),
            sampling_params=SimpleNamespace(max_new_tokens=8),
            last_node=node,
            init_next_round_input=lambda _cache: None,
        )

        runtime.on_requests_requeued((request,), is_retracted=True)

        entry = controller.visible_admission.get("victim")
        self.assertEqual(entry.state, AdmissionSideState.WAIT_RESTORE)
        self.assertEqual(entry.blocker_reason, "restore_obligation_pending")
        self.assertEqual(
            entry.restore_bundle_ids,
            (f"page:{handle.page_id}:0",),
        )
        self.assertTrue(obligation.requeued)

    def test_restore_debt_bypass_is_idle_bounded_and_smallest_first(self):
        config = BeliefKVConfig(
            kv_bytes_per_token=10,
            restore_obligation_escalation_ms=1000.0,
            restore_obligation_max_blocked_ms=2000.0,
            restore_lease_max_bypass_admissions=1,
        )
        controller = BeliefKVController(config)
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = controller
        runtime.audit = _AuditRecorder()
        runtime.scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[]),
            chunked_req=None,
        )
        runtime._metadata_scope_is_terminal = lambda _metadata: False
        runtime._request_restore_bundle_ids = lambda _req, _context_id: ()

        obligation = runtime._restore_obligation_index().create(
            request_id="restore-target",
            workflow_id="wf-restore",
            invocation_id="inv-restore",
            context_id="ctx-restore",
            context_epoch=0,
            source_retraction_transaction_id="retraction-restore",
            source_joint_plan_id="joint-restore",
            created_ts_ms=0.0,
            path_extent_ids=(),
        )
        requests = []
        for request_id, estimated_tokens in (("large", 10), ("small", 2)):
            metadata = BeliefKVRequestMetadata(
                f"wf-{request_id}",
                f"inv-{request_id}",
                f"ctx-{request_id}",
                0,
            )
            request = SimpleNamespace(
                rid=request_id,
                beliefkv_metadata=metadata,
            )
            requests.append(request)
            controller.process_runtime_events(
                (
                    RuntimeEvent(
                        f"start-{request_id}",
                        1.0,
                        RuntimeEventKind.WORKFLOW_START,
                        metadata.root_workflow_id,
                    ),
                    RuntimeEvent(
                        f"create-{request_id}",
                        2.0,
                        RuntimeEventKind.INVOCATION_CREATE,
                        metadata.root_workflow_id,
                        invocation_id=metadata.invocation_id,
                        context_id=metadata.context_id,
                        context_epoch=0,
                    ),
                )
            )
            controller.register_visible_request(
                AdmissionRequest(
                    request_id,
                    metadata.root_workflow_id,
                    metadata.invocation_id,
                    metadata.context_id,
                    0,
                    10.0,
                    estimated_tokens,
                    0,
                    10,
                )
            )

        self.assertEqual(
            runtime._select_restore_bypass_request(requests, now_ms=1001.0),
            "small",
        )
        runtime.scheduler.running_batch.reqs.append(requests[0])
        self.assertIsNone(
            runtime._select_restore_bypass_request(requests, now_ms=1002.0)
        )
        runtime.scheduler.running_batch.reqs.clear()
        obligation.bypass_count = 1
        self.assertIsNone(
            runtime._select_restore_bypass_request(requests, now_ms=1003.0)
        )

    def test_active_restore_lease_does_not_disable_debt_barrier(self):
        config = BeliefKVConfig(
            kv_bytes_per_token=10,
            restore_obligation_escalation_ms=1000.0,
            restore_obligation_max_blocked_ms=2000.0,
        )
        controller = BeliefKVController(config)
        metadata = BeliefKVRequestMetadata(
            "wf-ordinary", "inv-ordinary", "ctx-ordinary", 0
        )
        controller.process_runtime_events(
            (
                RuntimeEvent(
                    "start-ordinary",
                    1.0,
                    RuntimeEventKind.WORKFLOW_START,
                    metadata.root_workflow_id,
                ),
                RuntimeEvent(
                    "create-ordinary",
                    2.0,
                    RuntimeEventKind.INVOCATION_CREATE,
                    metadata.root_workflow_id,
                    invocation_id=metadata.invocation_id,
                    context_id=metadata.context_id,
                    context_epoch=0,
                ),
            )
        )
        controller.register_visible_request(
            AdmissionRequest(
                "ordinary",
                metadata.root_workflow_id,
                metadata.invocation_id,
                metadata.context_id,
                0,
                10.0,
                1,
                1,
                10,
            )
        )
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = controller
        runtime.audit = _AuditRecorder()
        runtime._now_ms = lambda: 2000.0
        runtime._retraction_cooldown_until_by_request = {}
        runtime._pending_h2d_contexts = set()
        runtime._restore_bypass_request_id = None
        runtime._context_bundle_generations = lambda _context_id: ()
        runtime._request_restore_bundle_ids = lambda _req, _context_id: ()
        runtime._metadata_scope_is_terminal = lambda _metadata: False
        runtime._workflow_transition_state = lambda _workflow_id: (0, False)
        obligation = runtime._restore_obligation_index().create(
            request_id="restore-target",
            workflow_id="wf-restore",
            invocation_id="inv-restore",
            context_id="ctx-restore",
            context_epoch=0,
            source_retraction_transaction_id="retraction-restore",
            source_joint_plan_id="joint-restore",
            created_ts_ms=0.0,
            path_extent_ids=(),
        )
        runtime._restore_lease_index().grant(
            obligation=obligation,
            granted_ts_ms=1000.0,
            reserved_tokens=1,
            reserved_bytes=10,
            h2d_bytes=0,
        )

        runtime._sync_visible_gate_state(
            "ordinary", metadata, req=SimpleNamespace(rid="ordinary")
        )

        entry = controller.visible_admission.get("ordinary")
        self.assertEqual(entry.state, AdmissionSideState.POLICY_BLOCKED)
        self.assertEqual(
            entry.blocker_reason,
            f"restore_debt_barrier:{obligation.obligation_id}",
        )

    def test_restore_service_grace_counts_decode_completion_not_wall_time(self):
        config = BeliefKVConfig(restore_service_grace_decode_tokens=4)
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.audit = _AuditRecorder()
        runtime.scheduler = SimpleNamespace(
            server_args=SimpleNamespace(num_continuous_decode_steps=1)
        )
        runtime._restore_obligation_counts = Counter()
        runtime._restore_service_grace_by_request = {}
        obligation = runtime._restore_obligation_index().create(
            request_id="restored",
            workflow_id="wf",
            invocation_id="inv",
            context_id="ctx",
            context_epoch=0,
            source_retraction_transaction_id="retraction-1",
            source_joint_plan_id="joint-1",
            created_ts_ms=0.0,
            path_extent_ids=(),
        )
        runtime._start_restore_service_grace(obligation, now_ms=1000.0)
        request = SimpleNamespace(rid="restored", output_ids=[])
        batch = SimpleNamespace(reqs=[request])

        request.output_ids.append(1)
        runtime._observe_restore_service_grace(
            batch, now_ms=5000.0, phase="prefill"
        )
        self.assertIn("restored", runtime._restore_service_grace_by_request)
        for offset in range(3):
            request.output_ids.append(offset + 2)
            runtime._observe_restore_service_grace(
                batch, now_ms=6000.0 + offset, phase="decode"
            )
        self.assertIn("restored", runtime._restore_service_grace_by_request)
        request.output_ids.append(5)
        runtime._observe_restore_service_grace(
            batch, now_ms=7000.0, phase="decode"
        )
        self.assertNotIn("restored", runtime._restore_service_grace_by_request)

    def test_restore_obligation_funds_h2d_then_releases_ticket(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            host_capacity_bytes=4000,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=10,
            observed_admission_scheduling_enabled=True,
            running_batch_retraction_enabled=True,
        )
        controller = BeliefKVController(config)
        events = [
            RuntimeEvent("start", 1.0, RuntimeEventKind.WORKFLOW_START, "wf")
        ]
        for suffix in ("target", "victim"):
            events.append(
                RuntimeEvent(
                    f"create-{suffix}",
                    2.0,
                    RuntimeEventKind.INVOCATION_CREATE,
                    "wf",
                    invocation_id=f"inv-{suffix}",
                    context_id=f"ctx-{suffix}",
                    context_epoch=0,
                )
            )
        controller.process_runtime_events(tuple(events))
        tree_cache = _TreeCache()
        target_node = _Node(21)
        target_node.parent = tree_cache.root_node
        registry = SGLangNodeRegistry()
        target_handle = registry.register(target_node)
        victim_handle = PageHandle(22, 0)
        controller.page_index.register_page(
            target_handle,
            size_bytes=200,
            residency=PhysicalResidency.CPU_ONLY,
            radix_depth=1,
        )
        controller.page_index.register_page(
            victim_handle,
            size_bytes=200,
            residency=PhysicalResidency.DUAL_CLEAN,
            radix_depth=1,
        )
        controller.page_index.bind_pages("ctx-target", 0, (target_handle,))
        controller.page_index.bind_pages("ctx-victim", 0, (victim_handle,))
        metadata = BeliefKVRequestMetadata("wf", "inv-target", "ctx-target", 0)
        request = SimpleNamespace(
            rid="target",
            beliefkv_metadata=metadata,
            origin_input_ids=(1, 2),
            output_ids=(),
            prefix_indices=(),
            sampling_params=SimpleNamespace(max_new_tokens=8),
            last_node=target_node,
            init_next_round_input=lambda _cache: None,
        )
        allocator = _Allocator(10)
        scheduler = SimpleNamespace(
            waiting_queue=[request],
            token_to_kv_pool_allocator=allocator,
        )
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = controller
        runtime.scheduler = scheduler
        runtime.tree_cache = tree_cache
        runtime.registry = registry
        runtime.audit = _AuditRecorder()
        runtime._now_ms = lambda: 1000.0
        runtime._request_metadata_by_id = {"target": metadata}
        runtime._request_submitted_ts_by_id = {"target": 10.0}
        runtime._retraction_cooldown_until_by_request = {}
        runtime._pending_h2d_contexts = set()
        runtime._runtime_resource_observation = lambda **_kwargs: SimpleNamespace(
            host_free_bytes=4000
        )
        controller.register_visible_request(
            AdmissionRequest(
                "target", "wf", "inv-target", "ctx-target", 0, 10.0, 1, 1, 10
            )
        )
        obligation = runtime._restore_obligation_index().create(
            request_id="target",
            workflow_id="wf",
            invocation_id="inv-target",
            context_id="ctx-target",
            context_epoch=0,
            source_retraction_transaction_id="retraction-1",
            source_joint_plan_id="joint-1",
            created_ts_ms=900.0,
            path_extent_ids=(f"page:{target_handle.page_id}:0",),
        )
        obligation.source_transaction_terminal = True
        obligation.requeued = True

        runtime._drive_restore_obligations(now_ms=1000.0)
        funding = controller.command_queue.pop()
        self.assertIsNotNone(funding, runtime.audit.events)
        self.assertEqual(funding.kind, CommandKind.OFFLOAD_CONTEXT)
        self.assertEqual(funding.context_id, "ctx-victim")
        controller._queued_by_context.pop("ctx-victim", None)
        controller.page_index.commit_cpu(victim_handle)
        allocator.available_tokens = 30
        runtime._advance_restore_obligations(
            (
                CommandAck(
                    funding.command_id,
                    CommandStatus.COMPLETED,
                    1001.0,
                    actual_bytes=200,
                    page_handles=(victim_handle,),
                ),
            ),
            now_ms=1001.0,
        )
        self.assertEqual(allocator.available_tokens, 18)
        self.assertEqual(obligation.funding_reserved_tokens, 12)
        self.assertEqual(obligation.funding_reserved_bytes, 120)

        runtime._drive_restore_obligations(now_ms=1002.0)
        restore = controller.command_queue.pop()
        self.assertIsNotNone(restore)
        self.assertEqual(restore.kind, CommandKind.PREFETCH_CONTEXT)
        self.assertEqual(restore.context_id, "ctx-target")
        self.assertEqual(allocator.available_tokens, 28)
        self.assertEqual(obligation.funding_reserved_tokens, 0)
        self.assertEqual(obligation.funding_reserved_bytes, 0)
        lease = runtime._restore_lease_index().get("target")
        self.assertIsNotNone(lease)
        self.assertEqual(lease.reserved_tokens, 2)
        # Model the H2D allocator claim followed by concurrent decode growth.
        # The competing requests can consume every unreserved token, but not
        # the two-token restore admission lease.
        self.assertIsNotNone(allocator.alloc(20))
        self.assertIsNotNone(allocator.alloc(8))
        self.assertIsNone(allocator.alloc(1))
        controller._queued_by_context.pop("ctx-target", None)
        controller.page_index.begin_transfer(target_handle, TransferDirection.H2D)
        controller.page_index.complete_transfer(
            target_handle, TransferDirection.H2D
        )
        runtime._advance_restore_obligations(
            (
                CommandAck(
                    restore.command_id,
                    CommandStatus.COMPLETED,
                    1003.0,
                    actual_bytes=200,
                    page_handles=(target_handle,),
                ),
            ),
            now_ms=1003.0,
        )
        runtime._drive_restore_obligations(now_ms=1003.1)
        runtime._sync_visible_gate_state("target", metadata, req=request)

        entry = controller.visible_admission.get("target")
        self.assertEqual(entry.state, AdmissionSideState.VISIBLE_PENDING)
        self.assertEqual(obligation.state, RestoreObligationState.TICKET_READY)
        self.assertEqual(lease.state, RestoreLeaseState.RESTORED_RESERVED)
        self.assertEqual(target_node.lock_ref, 1)

        self.assertTrue(
            runtime._begin_restore_lease_admission(obligation, now_ms=1003.5)
        )
        self.assertEqual(allocator.available_tokens, 2)
        runtime._reject_restore_lease_admission(
            obligation, now_ms=1003.6, native_result="NO_TOKEN"
        )
        self.assertEqual(allocator.available_tokens, 0)
        self.assertEqual(lease.state, RestoreLeaseState.RESTORED_RESERVED)

        self.assertTrue(
            runtime._begin_restore_lease_admission(obligation, now_ms=1003.7)
        )
        runtime._commit_restore_lease_admission(obligation, now_ms=1003.8)
        self.assertEqual(allocator.available_tokens, 2)
        self.assertEqual(lease.state, RestoreLeaseState.ADMITTED)
        self.assertEqual(target_node.lock_ref, 0)
        runtime._finish_restore_obligation(
            "target",
            RestoreObligationState.SATISFIED,
            now_ms=1004.0,
            reason="gpu_service_resumed",
        )
        self.assertEqual(obligation.state, RestoreObligationState.SATISFIED)
        self.assertEqual(lease.state, RestoreLeaseState.RELEASED)

    def test_admission_only_restore_reclaims_funding_before_granting_lease(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            host_capacity_bytes=4000,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=10,
            observed_admission_scheduling_enabled=True,
        )
        controller = BeliefKVController(config)
        events = [RuntimeEvent("start", 1.0, RuntimeEventKind.WORKFLOW_START, "wf")]
        for suffix in ("target", "victim"):
            events.append(
                RuntimeEvent(
                    f"create-{suffix}",
                    2.0,
                    RuntimeEventKind.INVOCATION_CREATE,
                    "wf",
                    invocation_id=f"inv-{suffix}",
                    context_id=f"ctx-{suffix}",
                    context_epoch=0,
                )
            )
        controller.process_runtime_events(tuple(events))
        tree_cache = _TreeCache()
        target_node = _Node(41)
        target_node.parent = tree_cache.root_node
        registry = SGLangNodeRegistry()
        target_handle = registry.register(target_node)
        victim_handle = PageHandle(42, 0)
        controller.page_index.register_page(
            target_handle,
            size_bytes=200,
            residency=PhysicalResidency.GPU_ONLY,
            radix_depth=1,
        )
        controller.page_index.register_page(
            victim_handle,
            size_bytes=200,
            residency=PhysicalResidency.DUAL_CLEAN,
            radix_depth=1,
        )
        controller.page_index.bind_pages("ctx-target", 0, (target_handle,))
        controller.page_index.bind_pages("ctx-victim", 0, (victim_handle,))
        metadata = BeliefKVRequestMetadata("wf", "inv-target", "ctx-target", 0)
        request = SimpleNamespace(
            rid="target",
            beliefkv_metadata=metadata,
            origin_input_ids=(1, 2),
            output_ids=(),
            prefix_indices=(),
            sampling_params=SimpleNamespace(max_new_tokens=8),
            last_node=target_node,
            init_next_round_input=lambda _cache: None,
        )
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = controller
        runtime.scheduler = SimpleNamespace(
            waiting_queue=[request],
            token_to_kv_pool_allocator=_Allocator(1),
        )
        runtime.tree_cache = tree_cache
        runtime.registry = registry
        runtime.audit = _AuditRecorder()
        runtime._request_metadata_by_id = {"target": metadata}
        runtime._request_submitted_ts_by_id = {"target": 10.0}
        runtime._retraction_cooldown_until_by_request = {}
        runtime._pending_h2d_contexts = set()
        runtime._runtime_resource_observation = lambda **_kwargs: SimpleNamespace(
            host_free_bytes=4000
        )
        controller.register_visible_request(
            AdmissionRequest(
                "target", "wf", "inv-target", "ctx-target", 0, 10.0, 1, 1, 10
            )
        )
        obligation = runtime._restore_obligation_index().create(
            request_id="target",
            workflow_id="wf",
            invocation_id="inv-target",
            context_id="ctx-target",
            context_epoch=0,
            source_retraction_transaction_id="retraction-target",
            source_joint_plan_id="joint-target",
            created_ts_ms=900.0,
            path_extent_ids=(f"page:{target_handle.page_id}:0",),
        )
        obligation.source_transaction_terminal = True
        obligation.requeued = True

        runtime._drive_restore_obligations(now_ms=1000.0)

        funding = controller.command_queue.pop()
        self.assertIsNotNone(funding, runtime.audit.events)
        self.assertEqual(funding.kind, CommandKind.OFFLOAD_CONTEXT)
        self.assertEqual(funding.context_id, "ctx-victim")
        self.assertEqual(
            funding.metadata["reason"], "restore_obligation_funding"
        )
        self.assertEqual(
            obligation.state, RestoreObligationState.EVICT_FOR_RESTORE
        )
        self.assertIsNone(runtime._restore_lease_index().get("target"))

    def test_allocator_backed_reservations_include_funding_and_lease_slots(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime._restore_funding_allocations = {
            "request-a": [[1, 2, 3], [4, 5]],
        }
        runtime._restore_lease_allocations = {
            "request-b": [[6, 7, 8, 9]],
        }

        self.assertEqual(runtime.allocator_backed_reservation_tokens(), 9)

    def test_full_restore_lease_table_does_not_churn_funding_capacity(self):
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = BeliefKVConfig(
            kv_bytes_per_token=10,
            restore_lease_enabled=True,
            restore_lease_max_active=1,
        )
        runtime.audit = _AuditRecorder()
        allocator = _Allocator(100)
        runtime.scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
        )
        obligations = runtime._restore_obligation_index()
        first = obligations.create(
            request_id="first",
            workflow_id="wf-first",
            invocation_id="inv-first",
            context_id="ctx-first",
            context_epoch=0,
            source_retraction_transaction_id="retraction-first",
            source_joint_plan_id="joint-first",
            created_ts_ms=1.0,
            path_extent_ids=(),
        )
        second = obligations.create(
            request_id="second",
            workflow_id="wf-second",
            invocation_id="inv-second",
            context_id="ctx-second",
            context_epoch=0,
            source_retraction_transaction_id="retraction-second",
            source_joint_plan_id="joint-second",
            created_ts_ms=2.0,
            path_extent_ids=(),
        )
        runtime._restore_lease_index().grant(
            obligation=first,
            granted_ts_ms=3.0,
            reserved_tokens=1,
            reserved_bytes=10,
            h2d_bytes=0,
        )
        funding = allocator.alloc(12)
        self.assertIsNotNone(funding)
        runtime._set_restore_funding_reservation(second, [funding])
        available_before = allocator.available_tokens

        lease = runtime._grant_restore_lease(
            second,
            h2d_bytes=0,
            now_ms=4.0,
        )

        self.assertIsNone(lease)
        self.assertEqual(allocator.available_tokens, available_before)
        self.assertEqual(runtime._restore_funding_reserved_tokens("second"), 12)
        self.assertFalse(
            any(
                event == "restore_funding_capacity_released"
                for event, _, _ in runtime.audit.events
            )
        )

    def test_restore_prefix_pin_waits_for_host_only_h2d(self):
        config = BeliefKVConfig(kv_bytes_per_token=10)
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.tree_cache = _TreeCache()
        runtime.audit = _AuditRecorder()
        runtime._restore_obligation_counts = Counter()
        runtime._restore_lease_pins = {}
        obligation = runtime._restore_obligation_index().create(
            request_id="request",
            workflow_id="wf",
            invocation_id="inv",
            context_id="ctx",
            context_epoch=0,
            source_retraction_transaction_id="retraction-1",
            source_joint_plan_id="joint-1",
            created_ts_ms=1.0,
            path_extent_ids=("page:1:0",),
        )
        runtime._restore_lease_index().grant(
            obligation=obligation,
            granted_ts_ms=2.0,
            reserved_tokens=4,
            reserved_bytes=40,
            h2d_bytes=40,
        )
        node = _Node(1)
        node.parent = runtime.tree_cache.root_node
        node.host_value = [10, 11, 12, 13]
        node.value = None
        request = SimpleNamespace(rid="request", last_node=node)

        self.assertTrue(
            runtime._pin_restore_lease_prefix(
                obligation,
                request,
                now_ms=3.0,
                allow_unmaterialized=True,
            )
        )
        self.assertNotIn("request", runtime._restore_lease_pins)
        self.assertFalse(
            runtime._pin_restore_lease_prefix(
                obligation,
                request,
                now_ms=4.0,
            )
        )

        node.value = [1, 2, 3, 4]
        self.assertTrue(
            runtime._pin_restore_lease_prefix(
                obligation,
                request,
                now_ms=5.0,
            )
        )
        self.assertIn("request", runtime._restore_lease_pins)
        self.assertEqual(node.lock_ref, 1)

    def test_blocked_restore_head_does_not_starve_later_obligation(self):
        config = BeliefKVConfig(
            kv_bytes_per_token=10,
            restore_obligation_max_active=2,
            restore_lease_enabled=True,
        )
        controller = BeliefKVController(config)
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = controller
        runtime.audit = _AuditRecorder()
        runtime._request_metadata_by_id = {}
        obligations = runtime._restore_obligation_index()
        head = obligations.create(
            request_id="head",
            workflow_id="wf-head",
            invocation_id="inv-head",
            context_id="ctx-head",
            context_epoch=0,
            source_retraction_transaction_id="retraction-head",
            source_joint_plan_id="joint-head",
            created_ts_ms=1.0,
            path_extent_ids=(),
        )
        tail = obligations.create(
            request_id="tail",
            workflow_id="wf-tail",
            invocation_id="inv-tail",
            context_id="ctx-tail",
            context_epoch=0,
            source_retraction_transaction_id="retraction-tail",
            source_joint_plan_id="joint-tail",
            created_ts_ms=2.0,
            path_extent_ids=(),
        )
        for obligation in (head, tail):
            obligation.source_transaction_terminal = True
            obligation.requeued = True
        runtime._refresh_restore_obligation = (
            lambda obligation, **_kwargs: (
                SimpleNamespace(rid=obligation.request_id),
                (),
            )
        )
        runtime._restore_attempt_stamp = lambda: (1, 1, 0, 0)
        tail_lease = SimpleNamespace(mark_restored=lambda: None)
        runtime._grant_restore_lease = (
            lambda obligation, **_kwargs: (
                None if obligation.request_id == "head" else tail_lease
            )
        )
        funding_attempts = []
        runtime._try_queue_restore_lease_funding = (
            lambda obligation, **_kwargs: (
                funding_attempts.append(obligation.request_id) or False,
                ("restore_lease_capacity", "no_funding_bundle"),
            )
        )
        runtime._pin_restore_lease_prefix = lambda *_args, **_kwargs: True
        runtime._sync_visible_gate_state = lambda *_args, **_kwargs: None

        runtime._drive_restore_obligations(now_ms=1000.0)

        self.assertEqual(head.state, RestoreObligationState.PARKED_WAIT)
        self.assertEqual(
            head.blocker_codes,
            ("no_funding_bundle", "restore_lease_capacity"),
        )
        self.assertEqual(tail.state, RestoreObligationState.TICKET_READY)
        self.assertEqual(funding_attempts, ["head"])
        self.assertTrue(
            any(
                event == "restore_obligation_ticket_ready"
                and fields["request_id"] == "tail"
                for event, _, fields in runtime.audit.events
            )
        )

        runtime._drive_restore_obligations(now_ms=1001.0)

        self.assertEqual(funding_attempts, ["head"])

    def test_ordinary_waiting_cpu_prefix_creates_and_drives_restore(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            host_capacity_bytes=4000,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=10,
            observed_admission_scheduling_enabled=True,
        )
        controller = BeliefKVController(config)
        controller.process_runtime_events(
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
        tree_cache = _TreeCache()
        node = _Node(31)
        node.parent = tree_cache.root_node
        registry = SGLangNodeRegistry()
        handle = registry.register(node)
        controller.page_index.register_page(
            handle,
            size_bytes=200,
            residency=PhysicalResidency.CPU_ONLY,
            radix_depth=1,
        )
        controller.page_index.bind_pages("ctx", 0, (handle,))
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        request = SimpleNamespace(
            rid="waiting",
            beliefkv_metadata=metadata,
            origin_input_ids=(1, 2),
            output_ids=(),
            prefix_indices=(),
            sampling_params=SimpleNamespace(max_new_tokens=8),
            last_node=node,
            init_next_round_input=lambda _cache: None,
        )
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = controller
        runtime.scheduler = SimpleNamespace(
            waiting_queue=[request],
            token_to_kv_pool_allocator=_Allocator(100),
        )
        runtime.tree_cache = tree_cache
        runtime.registry = registry
        runtime.audit = _AuditRecorder()
        runtime._now_ms = lambda: 1000.0
        runtime._request_metadata_by_id = {"waiting": metadata}
        runtime._request_submitted_ts_by_id = {"waiting": 10.0}
        runtime._retraction_cooldown_until_by_request = {}
        runtime._pending_h2d_contexts = set()
        controller.register_visible_request(
            AdmissionRequest(
                "waiting", "wf", "inv", "ctx", 0, 10.0, 1, 1, 10
            )
        )

        runtime._sync_visible_gate_state("waiting", metadata, req=request)

        obligation = runtime._restore_obligation_index().get("waiting")
        self.assertIsNotNone(obligation)
        self.assertEqual(
            obligation.cause,
            RestoreObligationCause.ORDINARY_WAITING_PREFIX,
        )
        self.assertTrue(obligation.source_transaction_terminal)
        self.assertTrue(obligation.requeued)
        self.assertEqual(
            controller.visible_admission.get("waiting").state,
            AdmissionSideState.WAIT_RESTORE,
        )

        runtime._drive_restore_obligations(now_ms=1001.0)
        restore = controller.command_queue.pop()
        self.assertIsNotNone(restore, runtime.audit.events)
        self.assertEqual(restore.kind, CommandKind.PREFETCH_CONTEXT)
        self.assertEqual(restore.context_id, "ctx")
        self.assertEqual(
            restore.metadata["joint_plan_id"],
            "joint-restore-liveness:waiting:0",
        )

        controller._queued_by_context.pop("ctx", None)
        controller.page_index.begin_transfer(handle, TransferDirection.H2D)
        controller.page_index.complete_transfer(handle, TransferDirection.H2D)
        runtime._advance_restore_obligations(
            (
                CommandAck(
                    restore.command_id,
                    CommandStatus.COMPLETED,
                    1002.0,
                    actual_bytes=200,
                    page_handles=(handle,),
                ),
            ),
            now_ms=1002.0,
        )
        runtime._drive_restore_obligations(now_ms=1002.1)
        runtime._sync_visible_gate_state("waiting", metadata, req=request)

        self.assertEqual(obligation.state, RestoreObligationState.TICKET_READY)
        entry = controller.visible_admission.get("waiting")
        self.assertEqual(entry.state, AdmissionSideState.VISIBLE_PENDING)
        self.assertEqual(
            runtime._restore_ready_ticket_priority({"waiting": entry}),
            ("waiting",),
        )

    def test_ordinary_restore_rebinds_current_request_path(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            host_capacity_bytes=4000,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=10,
            observed_admission_scheduling_enabled=True,
        )
        controller = BeliefKVController(config)
        controller.process_runtime_events(
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
        tree_cache = _TreeCache()
        stale_node = _Node(31)
        stale_node.parent = tree_cache.root_node
        current_node = _Node(32)
        current_node.parent = tree_cache.root_node
        registry = SGLangNodeRegistry()
        stale_handle = registry.register(stale_node)
        current_handle = registry.register(current_node)
        for handle in (stale_handle, current_handle):
            controller.page_index.register_page(
                handle,
                size_bytes=200,
                residency=PhysicalResidency.CPU_ONLY,
                radix_depth=1,
            )
        controller.page_index.bind_pages("ctx", 0, (stale_handle,))
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        request = SimpleNamespace(
            rid="waiting",
            beliefkv_metadata=metadata,
            origin_input_ids=(1, 2),
            output_ids=(),
            prefix_indices=(),
            sampling_params=SimpleNamespace(max_new_tokens=8),
            last_node=current_node,
            init_next_round_input=lambda _cache: None,
        )
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = controller
        runtime.scheduler = SimpleNamespace(
            waiting_queue=[request],
            token_to_kv_pool_allocator=_Allocator(100),
        )
        runtime.tree_cache = tree_cache
        runtime.registry = registry
        runtime.audit = _AuditRecorder()
        runtime._now_ms = lambda: 1000.0
        runtime._request_metadata_by_id = {"waiting": metadata}
        runtime._request_submitted_ts_by_id = {"waiting": 10.0}
        runtime._retraction_cooldown_until_by_request = {}
        runtime._pending_h2d_contexts = set()
        controller.register_visible_request(
            AdmissionRequest(
                "waiting", "wf", "inv", "ctx", 0, 10.0, 1, 1, 10
            )
        )

        runtime._sync_visible_gate_state("waiting", metadata, req=request)
        runtime._drive_restore_obligations(now_ms=1001.0)

        owned_handles = {
            page.handle for page in controller.page_index.context_pages("ctx")
        }
        self.assertEqual(owned_handles, {current_handle})
        self.assertNotIn(
            "ctx", controller.page_index.pages[stale_handle].owner_contexts
        )
        restore = controller.command_queue.pop()
        self.assertIsNotNone(restore, runtime.audit.events)
        self.assertEqual(
            restore.physical_bundle.page_actions[0].handle,
            current_handle,
        )
        self.assertTrue(
            any(
                event == "restore_obligation_path_rebound"
                for event, _, _ in runtime.audit.events
            )
        )

    def test_ordinary_restore_uses_native_fallback_when_preview_is_missing(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=2000,
            host_capacity_bytes=4000,
            reserve_hbm_bytes=100,
            kv_bytes_per_token=10,
            observed_admission_scheduling_enabled=True,
        )
        controller = BeliefKVController(config)
        controller.process_runtime_events(
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
        tree_cache = _TreeCache()
        node = _Node(31)
        node.parent = tree_cache.root_node
        registry = SGLangNodeRegistry()
        handle = registry.register(node)
        controller.page_index.register_page(
            handle,
            size_bytes=200,
            residency=PhysicalResidency.CPU_ONLY,
            radix_depth=1,
        )
        controller.page_index.bind_pages("ctx", 0, (handle,))
        controller.arbiter.bundle_builder = SimpleNamespace(
            previews_for_context=lambda *_args, **_kwargs: ()
        )
        metadata = BeliefKVRequestMetadata("wf", "inv", "ctx", 0)
        request = SimpleNamespace(
            rid="waiting",
            beliefkv_metadata=metadata,
            origin_input_ids=(1, 2),
            output_ids=(),
            prefix_indices=(),
            sampling_params=SimpleNamespace(max_new_tokens=8),
            last_node=node,
            init_next_round_input=lambda _cache: None,
        )
        allocator = _Allocator(100)
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = controller
        runtime.scheduler = SimpleNamespace(
            waiting_queue=[request],
            token_to_kv_pool_allocator=allocator,
        )
        runtime.tree_cache = tree_cache
        runtime.registry = registry
        runtime.audit = _AuditRecorder()
        runtime._now_ms = lambda: 1000.0
        runtime._request_metadata_by_id = {"waiting": metadata}
        runtime._request_submitted_ts_by_id = {"waiting": 10.0}
        runtime._retraction_cooldown_until_by_request = {}
        runtime._pending_h2d_contexts = set()
        controller.register_visible_request(
            AdmissionRequest(
                "waiting", "wf", "inv", "ctx", 0, 10.0, 1, 1, 10
            )
        )

        runtime._sync_visible_gate_state("waiting", metadata, req=request)
        runtime._drive_restore_obligations(now_ms=1001.0)

        obligation = runtime._restore_obligation_index().get("waiting")
        lease = runtime._restore_lease_index().get("waiting")
        self.assertTrue(obligation.native_admission_fallback)
        self.assertEqual(obligation.state, RestoreObligationState.TICKET_READY)
        self.assertEqual(lease.state, RestoreLeaseState.RESTORED_RESERVED)
        self.assertIsNone(controller.command_queue.pop())
        self.assertEqual(
            controller.visible_admission.get("waiting").state,
            AdmissionSideState.VISIBLE_PENDING,
        )

        runtime._sync_visible_gate_state("waiting", metadata, req=request)
        self.assertEqual(
            controller.visible_admission.get("waiting").state,
            AdmissionSideState.VISIBLE_PENDING,
        )
        self.assertTrue(
            runtime._begin_restore_lease_admission(obligation, now_ms=1002.0)
        )
        runtime._reject_restore_lease_admission(
            obligation,
            now_ms=1002.1,
            native_result="NO_TOKEN",
        )
        self.assertEqual(lease.state, RestoreLeaseState.RESTORED_RESERVED)
        self.assertTrue(
            any(
                event == "restore_obligation_native_fallback_ready"
                for event, _, _ in runtime.audit.events
            )
        )
        fallback_event = next(
            fields
            for event, _, fields in runtime.audit.events
            if event == "restore_obligation_native_fallback_ready"
        )
        self.assertEqual(fallback_event["required_extent_count"], 1)
        self.assertEqual(fallback_event["restore_bytes"], 200)

    def test_ticket_ready_restore_debt_preempts_stale_joint_order(self):
        config = BeliefKVConfig(
            hbm_capacity_bytes=1000,
            reserve_hbm_bytes=0,
            kv_bytes_per_token=10,
            joint_policy_enabled=True,
            restore_lease_enabled=False,
        )
        controller = BeliefKVController(config)
        for suffix in ("old", "new"):
            controller.process_runtime_events(
                (
                    RuntimeEvent(
                        f"start-{suffix}",
                        1.0,
                        RuntimeEventKind.WORKFLOW_START,
                        f"wf-{suffix}",
                    ),
                    RuntimeEvent(
                        f"create-{suffix}",
                        2.0,
                        RuntimeEventKind.INVOCATION_CREATE,
                        f"wf-{suffix}",
                        invocation_id=f"inv-{suffix}",
                        context_id=f"ctx-{suffix}",
                        context_epoch=0,
                    ),
                )
            )
        runtime = EmbeddedSGLangRuntime.__new__(EmbeddedSGLangRuntime)
        runtime.config = config
        runtime.controller = controller
        runtime.audit = _AuditRecorder()
        runtime._now_ms = lambda: 100.0
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
        runtime._current_online_joint_decision = None
        runtime._online_joint_result = None
        runtime._current_joint_plan_epoch = None
        runtime._online_joint_counts = Counter()
        runtime._online_joint_epoch_sequence = 0
        runtime._restore_obligations = RestoreObligationIndex(max_active=2)
        obligation = runtime._restore_obligations.create(
            request_id="oldest",
            workflow_id="wf-old",
            invocation_id="inv-old",
            context_id="ctx-old",
            context_epoch=0,
            source_retraction_transaction_id="ordinary-waiting:oldest",
            source_joint_plan_id="joint-old",
            created_ts_ms=1.0,
            path_extent_ids=("page:1:0",),
            cause=RestoreObligationCause.ORDINARY_WAITING_PREFIX,
        )
        obligation.source_transaction_terminal = True
        obligation.requeued = True
        obligation.mark_ticket_ready(now_ms=2.0)
        requests = []
        for request_id, suffix, submitted_ts_ms in (
            ("oldest", "old", 1.0),
            ("newer", "new", 2.0),
        ):
            metadata = BeliefKVRequestMetadata(
                f"wf-{suffix}", f"inv-{suffix}", f"ctx-{suffix}", 0
            )
            request = SimpleNamespace(
                rid=request_id,
                beliefkv_metadata=metadata,
                origin_input_ids=(1,),
                prefix_indices=(),
                last_node=None,
            )
            requests.append(request)
            runtime._request_metadata_by_id[request_id] = metadata
            runtime._request_submitted_ts_by_id[request_id] = submitted_ts_ms
            controller.register_visible_request(
                AdmissionRequest(
                    request_id,
                    f"wf-{suffix}",
                    f"inv-{suffix}",
                    f"ctx-{suffix}",
                    0,
                    submitted_ts_ms,
                    1,
                    1,
                    10,
                )
            )
        stale_view = OnlineJointPlanView(
            plan_id="stale-plan",
            ordered_request_ids=("newer",),
            immediate_request_ids=("newer",),
            restore_requirements=(),
            deferred_request_ids=("oldest",),
            residency_intent_indices=(),
        )
        runtime._online_joint_admission_decision = lambda **_kwargs: (
            OnlineJointPlanDecision(stale_view, "applicable")
        )

        runtime.begin_prefill_epoch(
            requests,
            SimpleNamespace(
                rem_input_tokens=10,
                rem_chunk_tokens=None,
                rem_total_tokens=100,
            ),
            max_requests=2,
        )

        self.assertEqual(
            [
                ticket.request_id
                for ticket in runtime._current_ticket_epoch.tickets
            ],
            ["oldest", "newer"],
        )
        self.assertIn(
            "restore_liveness", runtime._current_ticket_epoch.source
        )
        self.assertNotEqual(
            runtime._current_online_joint_view.plan_id, "stale-plan"
        )
        self.assertEqual(
            runtime._current_online_joint_view.immediate_request_ids,
            ("oldest", "newer"),
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

    def test_idle_memory_check_accounts_for_beliefkv_allocator_reservations(self):
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = object.__new__(Scheduler)
        scheduler.is_hybrid = False
        scheduler.max_total_num_tokens = 100
        scheduler.enable_hierarchical_cache = True
        scheduler.tree_cache = SimpleNamespace(protected_size=lambda: 0)
        scheduler._get_token_info = lambda: (20, 0.2, 10, 70)
        scheduler.beliefkv_runtime = SimpleNamespace(
            allocator_backed_reservation_tokens=lambda: 20
        )
        scheduler.disaggregation_mode = None
        scheduler.req_to_token_pool = SimpleNamespace(size=1, free_slots=[0])
        scheduler.enable_metrics = False
        scheduler._publish_kv_events = lambda: None

        Scheduler.check_memory(scheduler)

        scheduler.beliefkv_runtime = SimpleNamespace(
            allocator_backed_reservation_tokens=lambda: 19
        )
        with self.assertRaisesRegex(ValueError, "memory leak detected"):
            Scheduler.check_memory(scheduler)


if __name__ == "__main__":
    unittest.main()
