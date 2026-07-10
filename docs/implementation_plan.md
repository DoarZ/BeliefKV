# Implementation Plan

## Phase 0: Standalone Policy Engine

- Keep the policy independent of model serving.
- Normalize traces into `ContinuationBelief`, `KVObjectMeta`, and
  `RuntimeSnapshot`.
- Validate decisions with unit tests and replay simulation.

Exit criteria:

- `python -m unittest discover -s tests` passes.
- `python -m beliefkv.cli plan examples/simple_snapshot.json` emits stable
  actions.

## Phase 1: Runtime Metadata Plumbing

- Add optional request fields for workflow and branch metadata.
- Store metadata in request objects after tokenization.
- Attach metadata to prefix-cache nodes when KV is created or reused.
- Export a scheduler snapshot every planning interval.

Exit criteria:

- Requests without BeliefKV fields behave exactly like the baseline.
- Requests with fields are visible in scheduler and cache metadata.

## Phase 2: Residency Actions

- Implement GPU keep protection for active decode workflows.
- Implement CPU offload for far-future KV objects under HBM pressure.
- Implement CPU-to-GPU prefetch when predicted next use enters the transfer
  window.
- Implement recompute-only action for low-survival branches.

Exit criteria:

- Action counters match planner output.
- No request observes invalid KV handles after migration or recompute.

## Phase 3: Trace Replay and Baselines

- Build a trace runner for coding, search, and research workloads.
- Compare against default prefix cache, LRU-like eviction, and agent-step
  policies under identical SGLang 0.5.2rc1 backend conditions.
- Store every run as JSONL events plus a CSV summary.

Exit criteria:

- Every result records git commit, model, GPU, SGLang version, config, workload,
  and random seed.

## Phase 4: Ablation and Portability

- Disable belief prediction.
- Disable migration-cost awareness.
- Disable decode protection.
- Disable workflow fairness.
- Port the narrow adapter to a newer SGLang branch for portability testing.

Exit criteria:

- The paper can separate algorithm gain from backend-version gain.
