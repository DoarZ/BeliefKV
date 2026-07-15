# Implementation And Evaluation Plan

Status date: 2026-07-14.

## Completed In Repository

### Phase 0: Event And Simulation Foundation

- framework-neutral, strictly validated `RuntimeEvent` schema;
- atomic RCCG reducer with nested call/spawn/join/message transitions;
- ClawTrace normalizer with race repair and deterministic replay;
- page-level HBM/host/PCIe simulator and immutable artifacts.

### Phase 1: Reactive Baseline

- four-class semantic residency and terminal owner release;
- continuation ancestor and message-driven frontier ordering;
- root-workflow memory admission and attained-service fairness;
- pressure-driven drop/offload and event-driven prefetch.

### Phase 2: Ownership And SGLang Bridge

- context/page many-to-many ownership and generation checks;
- shared physical charge, engine lock, semantic pin, and active-reader rules;
- incremental Radix/HiCache dirty observer;
- exact SGLang 0.5.2rc1 source contract and distributable patch.

### Phase 3: Controlled Migration

- scheduler-owned urgent/shadow command queues;
- asynchronous start/partial/reject/cancel/complete ACK protocol;
- prepare/commit shadow states and non-preemptible chunk cancellation semantics;
- H2D ancestor and D2H leaf closure;
- abort and cache-reset cleanup.

### Phase 4: Prediction And Experiments

- censored hierarchical Kaplan-Meier tool model;
- variable-order semi-Markov action context tree;
- online LLM queue/prefill/decode cost buckets;
- model artifact training/loading, online structured features, OOD fallback, and
  interval calibration;
- configurable ablation matrix, run manifests, CSV summaries, and bootstrap CI.

## Required Before Paper Evaluation

### Phase 5: Real Runtime Validation

1. Build the separate SGLang/CUDA conda environment.
2. Run a small model smoke test with metadata disabled and enabled.
3. Verify page bytes and allocator state against GPU/host measurements.
4. Inject request abort, cache reset, rejected host allocation, stale epoch, and
   simultaneous pressure/wakeup failures.
5. Measure patched-disabled overhead against exact upstream.

Exit criteria: no OOM, stale handle, leaked reservation, location divergence,
or deadlock in long mixed-workload runs.

### Phase 6: Baselines And Real Workloads

- coding, browser/research, peer collaboration, recursive subagent, and mixed
  root-workflow traces;
- upstream SGLang LRU/Radix, HiCache write policies, reactive BeliefKV,
  heuristic shadow, direct predicted offload, full BeliefKV, and offline oracle;
- KVFlow/TokenCake/AgentServe-style policies only where code and assumptions can
  be reproduced fairly;
- identical model, backend commit, quantization, request trace, and GPU settings.

Every run must record repository commit, dirty state, model, GPU, SGLang commit,
configuration hash, workload, seed, events, and final RCCG.

### Phase 7: Predictor Generalization

- group split by project/session;
- temporal holdout;
- leave-one-workload-family-out and leave-one-tool-family-out;
- unseen agent-role evaluation;
- survival NLL/Brier/coverage plus end-to-end migration regret.

Do not tune on the test workload or randomly split events from one workflow.

### Phase 8: Ablation And Portability

- disable prediction, shadow, causal ordering, fairness, and interference
  feedback independently;
- report useful and wasted shadow bytes, not only PCIe utilization;
- compare reactive and full policy with a full-future oracle;
- port only after the pinned-version experiment is stable.

The implementation is ready for Phase 5. It is not yet evidence that the final
algorithm improves real GPU workflow completion time.
