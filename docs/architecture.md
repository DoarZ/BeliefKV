# BeliefKV Architecture

## System Shape

```text
Trace replay / Agent runtime
        |
        | workflow_id, agent_id, branch_id, agent_type
        v
BeliefKV control plane
        |
        +-- WorkflowStateStore
        +-- Belief frontier builder
        +-- KV action planner
        |
        v
SGLang 0.5.2rc1 adapter
        |
        +-- scheduler snapshot hook
        +-- KV object index hook
        +-- hierarchical residency action hook
        |
        v
GPU KV / CPU KV / raw text
```

## Core Invariant

The runtime owns tensors and memory movement. BeliefKV owns policy state and
decisions. The adapter converts runtime state into `KVObjectMeta`,
`ContinuationBelief`, and `RuntimeSnapshot`, then applies the returned
`KVDecision` objects.

## Main Loop

1. Agent requests carry workflow metadata.
2. The control plane updates the workflow frontier.
3. The runtime exports a compact snapshot of KV objects and HBM pressure.
4. The planner classifies each KV object as decode-protected, GPU-resident,
   CPU-resident, recompute-only, or raw text.
5. The adapter applies offload, prefetch, materialization, or eviction actions.
6. Metrics record TTFT, TPOT, HBM pressure, migration volume, and recompute cost.

## Why This Layout

This layout separates research logic from serving-runtime mechanics. It allows
policy experiments to run as pure Python unit tests, while the runtime adapter
can stay narrow and version-specific.
