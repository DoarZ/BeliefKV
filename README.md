# BeliefKV

BeliefKV is a single-GPU KV cache management system for multi-agent workflows.
The first implementation target is **SGLang 0.5.2rc1**.

The project is intentionally organized as an overlay instead of a full runtime
fork. BeliefKV keeps its own control plane, policy engine, trace tooling, and
experiment scripts in this repository. Runtime-specific changes are maintained
as patches or adapters under `patches/` and `beliefkv/runtime/`.

## Goal

BeliefKV manages KV objects by workflow belief instead of only by request age or
agent step order. For each KV object, the policy considers:

- probability that the workflow branch survives;
- predicted next-use time distribution;
- recompute cost;
- GPU-to-CPU and CPU-to-GPU migration cost;
- current HBM pressure;
- whether the workflow is in a decode-sensitive stage.

This makes the system suitable for semi-dynamic agent workflows where the next
agent is not fully known before runtime, but the system can maintain a belief
over likely continuations.

## Repository Layout

```text
beliefkv/
  core/        shared dataclasses and state types
  control/     workflow state store and event handling
  policy/      cost model, frontier builder, and KV action planner
  runtime/     SGLang 0.5.2rc1 adapter and hook descriptions
  traces/      trace schema and replay utilities
  metrics/     metric aggregation utilities
configs/       reusable policy/runtime configs
docs/          design notes and implementation plan
examples/      small runnable planner examples
patches/       runtime patch staging area
scripts/       local helper scripts
tests/         unit tests for policy and trace logic
experiments/   raw traces, normalized traces, and result tables
third_party/   optional local runtime checkouts, ignored by git
```

## Setup

```bash
cd /home/longhao/experiment/BeliefKV
conda create -n beliefkv python=3.10 -y
conda activate beliefkv
python -m pip install -e .
python -m unittest discover -s tests
```

Run the standalone planner example:

```bash
python -m beliefkv.cli plan examples/simple_snapshot.json
```

## Runtime Baseline

The runtime baseline is SGLang `0.5.2rc1`.

BeliefKV should initially be integrated through narrow hook points:

- request metadata: `workflow_id`, `agent_id`, `branch_id`, `agent_type`;
- scheduler state: active decode requests, waiting prefill requests, HBM pressure;
- prefix cache metadata: KV node size, owner workflow, owner branch, residency;
- hierarchical cache actions: keep on GPU, offload to CPU, prefetch, recompute.

The core policy code in this repository is independent of the runtime. This
keeps algorithm experiments testable without starting a model server.

## Development Strategy

1. Implement and test the policy engine with synthetic and trace-derived
   snapshots.
2. Add SGLang 0.5.2rc1 hooks behind a small adapter layer.
3. Build a replay runner that feeds real multi-agent traces to both the baseline
   runtime and BeliefKV.
4. Add ablation switches for belief prediction, migration-cost awareness,
   workflow fairness, and decode protection.
5. Keep a separate branch for later portability testing on newer SGLang.

## Git Plan

This directory is ready to become a standalone git repository:

```bash
cd /home/longhao/experiment/BeliefKV
git init
git add .
git commit -m "Initialize BeliefKV system scaffold"
```

Do this after the initial scaffold is reviewed.
