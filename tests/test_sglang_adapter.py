import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from beliefkv.control.controller import BeliefKVController
from beliefkv.core.config import BeliefKVConfig
from beliefkv.core.events import RuntimeEvent, RuntimeEventKind
from beliefkv.policy.admission import AdmissionRequest
from beliefkv.runtime.protocol import (
    CommandKind,
    ControlCommand,
    PageHandle,
    PhysicalPageAction,
    ResolvedCommand,
    ResolvedPageAction,
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
    def evict_host(self, _indices):
        return 4


class _TreeCache:
    def __init__(self):
        self.root_node = _Node(0)
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.cache_controller = _CacheController()

    def write_backup(self, node):
        node.host_value = [10, 11, 12, 13]
        self.ongoing_write_through[node.id] = node
        return 4

    def check_hicache_events(self):
        for node in self.ongoing_write_through.values():
            node.lock_ref = 0
        self.ongoing_write_through.clear()
        for node in self.ongoing_load_back.values():
            node.loading = False
        self.ongoing_load_back.clear()

    def _evict_backuped(self, node):
        node.value = None
        return 4

    def _evict_regular(self, node):
        node.value = None
        return 4

    def load_back(self, node):
        node.value = [20, 21, 22, 23]
        node.loading = True
        self.ongoing_load_back[node.id] = (None, node)
        return node.value


class _AdmissionRecorder:
    def __init__(self):
        self.cancelled = []
        self.enqueued = []

    def enqueue(self, request):
        self.enqueued.append(request)

    def cancel(self, request_id):
        self.cancelled.append(request_id)


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
    def __init__(self, transfer):
        self.transfer = transfer

    def drain_acks(self):
        return []

    def scheduler_step(self, now_ms, *, drain_acks):
        self.drain_acks_argument = drain_acks
        return SimpleNamespace(
            now_ms=now_ms,
            admission=None,
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


class SGLangBackendTest(unittest.TestCase):
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
        self.assertTrue(node.evicted)
        self.assertEqual(ack.actual_bytes, 400)

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
        ack = backend.poll_acks()[0]
        self.assertEqual(ack.status.value, "rejected")

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

        self.assertEqual(submission.started_handles, (handle,))
        self.assertEqual(ack.status.value, "rejected")
        self.assertEqual(ack.actual_bytes, 0)
        self.assertIn("descendants off device", ack.reason)
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
