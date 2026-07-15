# BeliefKV Architecture

The authoritative algorithm discussion is in
[`beliefkv_design_2026-07-14_zh.md`](beliefkv_design_2026-07-14_zh.md). This file
maps that design to the implementation.

For rendered diagrams that distinguish implemented, partial, and missing
components, see
[`architecture_status_zh.md`](architecture_status_zh.md).

## Control And Data Planes

```text
Agent runtime / tool dispatcher / message bus
                  |
                  | RuntimeEvent batches
                  v
       RuntimeCausalContextGraph
                  |
          +-------+--------+
          |                |
  Remaining-time      Causal frontier
    predictor          and fairness
          |                |
          +-------+--------+
                  |
      Admission + transfer planner
                  |
          ControlCommand queue
                  |
                  v
        SGLangSchedulerBridge
                  |
       RadixArbiter + PageIndex
                  |
                  v
       SGLang RadixCache / HiCache
                  |
             GPU KV / CPU KV
```

BeliefKV owns logical causality and policy. SGLang remains the only allocator
and tensor-location authority. A residency transition is committed only after
the scheduler-owned backend returns a generation-checked ACK.

## Runtime Transaction Order

At each patched SGLang scheduler safe point:

1. drain completed HiCache ACKs into the control state machine;
2. synchronize only if a Radix/HiCache observer marked the tree dirty;
3. report authoritative allocator usage and per-workflow charges;
4. run admission, pressure handling, prefetch, or one shadow chunk;
5. submit at most one transfer command through the scheduler thread;
6. let SGLang calculate its native queue policy;
7. reorder only metadata-tagged queue slots by root workflow and causal frontier;
8. preserve untagged requests and all SGLang allocator/lock invariants.

The ACK-before-sync order is required for immediate `COMMIT_CPU` and `DROP`
actions. H2D selection enforces HiCache's ancestor closure so implicit physical
loads cannot escape BeliefKV byte accounting.

## Safety Invariants

- `(page_id, allocation_generation)` rejects stale node reuse.
- Active readers, engine locks, semantic pins, and active shared owners prevent
  migration.
- Shared physical pages are counted once and split across root workflows.
- Admission waits for actual released-byte ACKs, not planned bytes.
- Prediction can rank PREPARE/prefetch actions but cannot bypass reactive safety.
- Cache reset cancels all in-flight bookkeeping before invalidating page handles.
- Requests without `beliefkv_metadata` bypass admission and retain their native
  queue slots.

## Validation Layers

1. Pure Python unit tests cover RCCG, ownership, policies, predictor, simulator,
   artifacts, and failure paths.
2. Fake HiCache tests cover submitted DMA, partial/rejected actions, reset,
   ancestor closure, and abort handling.
3. `beliefkv check-sglang` checks version, git commit when available, AST
   symbols, and BeliefKV patch markers.
4. A Qwen2.5-0.5B CUDA metadata/lifecycle smoke has passed; real HBM-pressure
   migration, performance, and long-running fault tests remain experimental
   requirements rather than completed validation layers.
