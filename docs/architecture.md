# BeliefKV Architecture

The authoritative algorithm discussion is in
[`beliefkv_design_2026-07-14_zh.md`](beliefkv_design_2026-07-14_zh.md). This file
maps that design to the implementation.

For rendered diagrams that distinguish implemented, partial, and missing
components, see
[`architecture_status_zh.md`](architecture_status_zh.md).

## Current P6 Control Line

The current P6 design joins an agent-side opportunity window with the physical
cost of the live Radix closure:

```text
RCCG + FrontierBeliefModel        PhysicalSnapshot + Radix closure
  -> causal slack                  -> transfer shape
                                  -> morphology debt
                 \                    /
                  ScenarioRiskPlanner
                  -> A0 / PREPARE_HOST / bounded PREFETCH_GPU
                  -> semantic intent
                  -> safe-point live-shape rematerialization
                  -> existing P5 transaction and ACK path
```

The decision margin is conceptually
`min(pressure_deadline, reentry_deadline) - Q90(shape-conditioned transfer
time) - guard`. A bytes-only estimate may not be substituted across extent-count
buckets. Unsupported shapes and stale live closures fail closed to the P5
observed plan.

As of 2026-08-10, the first implementation conditions transfer cost on bytes
and extent count; extent-size distribution and closure depth are observed but
not model inputs. M1--M5 connect that cost to predictive intent generation and
scheduler-safe live-shape rematerialization. Replay changes timing estimates
and feasibility reasons, but changes neither candidate eligibility nor the
selected action. M6 implements a run-level one-action canary and strict
five-stage attribution, but its decision-relevance and natural-action gates are
closed, so no GPU canary was forced. A controlled GPU0 result at equal 2.659 GB measured
185.69 ms for 7 extents and 765.17 ms for 106 extents; this remains development
evidence, not a cross-GPU or end-to-end benefit result.

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
  Frontier belief     Causal frontier
  and causal slack     and fairness
          |                |
          +-------+--------+
                  |
      Physical closure shape
                  |
  Morphology-aware admission + transfer planner
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
