# BeliefKV Technical Archive

> Historical note: this archive records the design as of 2026-07-10 and is no
> longer the current specification. See
> [`beliefkv_design_2026-07-14_zh.md`](beliefkv_design_2026-07-14_zh.md) for the
> current Chinese design covering the runtime causal graph, Radix ownership
> bridge, reactive policy, remaining-time predictor, and Prepare-Commit KV
> migration.

Date: 2026-07-10

This document archives the design decisions, module boundaries, and technical
routes discussed for BeliefKV. It is intended as a review checkpoint before
continuing implementation.

## 1. Research Positioning

BeliefKV targets single-GPU, HBM-constrained serving for multi-agent workflows.
The central problem is not only how to evict an existing KV cache, but how to
manage the lifecycle of KV objects under semi-dynamic workflow uncertainty:

```text
raw text / tool result
    -> prefill materialization
    -> GPU KV residency
    -> CPU offload / GPU prefetch / recompute
    -> final death
```

The intended workload family is:

- coding agents;
- browser/search agents;
- RAG-heavy research agents;
- workflows with tool calls, long observations, and repeated LLM turns;
- workflows with handoff, branching, or subagent execution;
- concurrent workflows on a single GPU where HBM is the bottleneck.

The design should not rely on perfect workflow knowledge. Some workflows may
come from templates, while others are dynamic agent programs. BeliefKV should
support both, but its main novelty should be in the semi-dynamic case where the
runtime maintains a belief over possible continuations.

## 2. Honest Scope Boundary

Several ideas were discussed and narrowed:

- Pure lazy observation materialization is not enough as a core contribution.
  It is close to delaying long prefill work through scheduling.
- Segmenting KV by semantic text regions is risky because normal transformer KV
  cache does not directly expose clean semantic segments.
- Prefix sharing alone is not novel enough, since serving systems already merge
  identical or shared prefixes.
- BeliefKV should therefore focus on belief-aware KV lifecycle management:
  predicting likely next KV owners, estimating next-use time and branch survival,
  and using these predictions inside a cost-aware residency planner.

The core claim should be:

```text
Existing request-level, agent-level, or page-level KV policies do not model the
uncertain future of semi-dynamic multi-agent workflows. BeliefKV exposes this
uncertainty as a belief frontier and uses it to decide KV birth, residency,
prefetch, offload, and recompute.
```

## 3. Runtime Baseline Strategy

The first implementation target is SGLang `0.5.2rc1`, because the existing
agent-aware baseline is built on that version and its relevant scheduler/cache
interfaces are known.

The experiment strategy should use two layers:

1. Apples-to-apples algorithm comparison on the SGLang `0.5.2rc1` baseline.
2. Later portability/SOTA validation on a newer SGLang branch.

This avoids mixing algorithmic gains with backend-version gains.

## 4. High-Level Architecture

BeliefKV should remain an overlay system instead of a full serving-runtime fork.

```text
Agent runtime / trace replay
        |
        | workflow_id, agent_id, branch_id, subagent metadata
        v
BeliefKV control plane
        |
        +-- WorkflowStateStore
        +-- Event normalizer
        +-- Belief predictor
        +-- KV action planner
        |
        v
SGLang 0.5.2rc1 adapter
        |
        +-- request metadata hook
        +-- scheduler snapshot hook
        +-- KV object index hook
        +-- residency action hook
        |
        v
GPU KV / CPU KV / recompute-only / raw text
```

The core invariant:

```text
The runtime owns tensors and actual memory movement.
BeliefKV owns workflow belief state and policy decisions.
```

The adapter converts runtime state into:

- `WorkflowState`
- `ContinuationBelief`
- `KVObjectMeta`
- `RuntimeSnapshot`

The policy returns:

- `KEEP_GPU`
- `OFFLOAD_CPU`
- `PREFETCH_GPU`
- `MATERIALIZE`
- `RECOMPUTE_LATER`
- `DROP_GPU`
- `NOOP`

## 5. Core Metadata

BeliefKV should not only track `agent_id`. It needs workflow and subagent
ownership metadata:

```text
root_workflow_id
workflow_id
parent_agent_id
agent_id
subagent_id
agent_type
branch_id
handoff_id
context_scope
tool_type
tool_status
observation_size_bucket
latency_bucket
```

This is required because modern agent frameworks may spawn subagents, perform
handoffs, isolate contexts, or run background tasks.

## 6. Next-Action Prediction Module

### 6.1 Prediction Target

The initial idea was to predict a flat `next_action` such as:

```text
agent:planner
agent:coder
tool:web_search
tool:file_read
finish
```

This is directionally reasonable, but for KV management it is better to use a
hierarchical action schema.

Recommended action schema:

```text
kind:
  llm_turn
  tool_call
  tool_result
  handoff
  spawn_subagent
  return_to_parent
  background_start
  background_join
  finish

coarse_type:
  planner
  coder
  reviewer
  summarizer
  shell
  file
  search
  web
  edit
  patch
  ask_user
  monitor

attributes:
  status = success / error / timeout / empty
  obs_size = small / medium / large
  latency = short / medium / long
  background = true / false
  isolation = shared_context / separate_context / worktree
```

BeliefKV ultimately cares about these quantities:

```text
P(next_llm_owner = agent/subagent)
P(next_use_time <= t)
P(observation_size_bucket = bucket)
P(branch_survives)
confidence
```

Therefore the predictor should output a belief frontier instead of a single
next action:

```python
ContinuationBelief(
    workflow_id="wf",
    agent_id="coder",
    branch_id="main",
    probability=0.72,
    ready_time_p50_ms=40.0,
    ready_time_p95_ms=160.0,
    expected_prompt_delta_tokens=1024,
    expected_output_tokens=256,
    branch_survival_prob=0.9,
    confidence=0.8,
)
```

### 6.2 Recommended Algorithm

Use a context tree as the implementation form, and variable-order Markov as the
modeling idea.

Module name:

```text
ContextTreeBeliefPredictor
```

The context tree should model:

```text
P(next_symbol | recent workflow symbols)
```

where symbols are normalized workflow events, not raw text tokens.

Example sequence:

```text
agent:planner
tool:file:success:small_obs
agent:coder
tool:shell:error:large_obs
agent:coder
tool:shell:success:small_obs
agent:reviewer
finish
```

### 6.3 Training Construction

Given a workflow sequence:

```text
x1, x2, ..., xn
```

For every position `t`, collect suffix contexts up to max order `K`:

```text
[]                         -> x_t
[x_{t-1}]                  -> x_t
[x_{t-2}, x_{t-1}]         -> x_t
...
[x_{t-K}, ..., x_{t-1}]    -> x_t
```

Each context node stores:

```python
context: tuple[str, ...]
next_counts: dict[str, int]
total_count: int
entropy: float
parent: ContextNode
children: dict[str, ContextNode]
```

### 6.4 Online Prediction

At runtime, use the longest suffix first, then back off:

```text
[a, b, c]
[b, c]
[c]
[]
```

Do not use pure longest-match MLE. Use hierarchical smoothing:

```text
P(next | context)
  = lambda_context * P_mle(next | context)
  + (1 - lambda_context) * P(next | parent_context)

lambda_context = count(context) / (count(context) + tau)
```

When a context is rare, it automatically falls back to shorter contexts. When a
context is frequent and informative, it dominates the distribution.

### 6.5 Overfitting Control

Overfitting is a real risk if the model stores exact agent names or exact
workflow sequences. The following controls are required:

1. Use abstract symbols as the primary state:

```text
agent_type
tool_type
tool_status
obs_size_bucket
latency_bucket
handoff/spawn/return event type
```

2. Keep concrete IDs only as optional refinement:

```text
concrete_agent_id
concrete_tool_name
```

3. Limit max order:

```text
K = 3 or 4 for the first version
```

4. Prune low-support nodes:

```text
support(context) >= n_min
```

5. Prune low-information nodes:

```text
KL(P_child || P_parent) >= epsilon
```

6. Emit confidence and let the planner behave conservatively when confidence is
low.

The target is not maximum average next-action accuracy. The target is:

```text
High-confidence predictions should be useful.
Low-confidence predictions should not harm KV scheduling.
```

### 6.6 Predictor Evaluation

Predictor evaluation should include:

- top-1 and top-k next owner accuracy;
- negative log likelihood;
- calibration curve / expected calibration error;
- high-confidence precision;
- cross-session and cross-workload generalization;
- downstream utility: TTFT, TPOT, HBM pressure, migration volume, recompute
  tokens.

The downstream utility is more important than raw action accuracy.

## 7. TraceLab Data and Correctness

The raw TraceLab data on this server is:

```text
/home/longhao/datasets/tracelab/syfi_coding_trace.jsonl.gz
```

Observed local statistics:

```text
rows: 357161
sessions: 4265
providers:
  codex: 216823
  claude: 140338

top tools:
  exec_command
  Bash
  write_stdin
  Read
  apply_patch
  Edit
  shell_command
  shell
  TaskUpdate
  Write
  Grep
  TaskCreate
  WebFetch
  update_plan
  Agent
```

There is also a previously generated serving replay workload in a legacy
evaluation tree outside this repository. That file is a compressed,
prompt-synthesized workload and should be migrated into BeliefKV only after the
TraceLab normalizer is rewritten with BeliefKV naming and schema.

Important distinction:

```text
The existing small TraceLab replay file is a compressed and prompt-synthesized serving
workload. It is useful for KV/prefill pressure tests, but it should not be used
as the direct training truth for next-action prediction.
```

For predictor training, use the raw gzip file and normalize these fields:

```text
session_id
round_index
provider
model
input_tokens_total
prefix_tokens
newly_append_tokens
output_tokens
timing_events
tools
current_tool_result_count
current_tool_result_chars
```

The normalizer should generate per-session action sequences and preserve tool
latency, result size, and status.

## 8. Subagent Architecture Impact

Modern agent systems increasingly support subagents, handoffs, independent
context windows, background tasks, and isolated workspaces. This changes the KV
management problem.

Reference examples:

- Anthropic Claude Code subagents:
  https://docs.anthropic.com/en/docs/claude-code/sub-agents
- OpenAI Agents SDK handoffs:
  https://openai.github.io/openai-agents-python/handoffs/

Implications for BeliefKV:

1. KV ownership must be hierarchical.

```text
root workflow
  -> parent agent
      -> subagent
          -> branch
              -> KV object
```

2. Fairness should be root-workflow-aware.

If a scheduler is fair only across subagents, a workflow that spawns many
subagents can occupy too much HBM and scheduling bandwidth.

3. Subagents create short-lived KV bursts.

Many subagents perform exploratory or review work and return only a summary to
the parent. Their internal KV may have low future value after return.

4. Subagent definitions are useful predictor features.

Useful features include:

```text
subagent name
description
tool permissions
model
background flag
handoff source/target
context isolation mode
max turns
```

5. The predictor should support events beyond agent/tool:

```text
spawn_subagent
handoff
return_to_parent
background_start
background_join
```

Conclusion:

```text
Subagent architectures make BeliefKV more necessary. They increase workflow
concurrency, KV fragmentation, and future-use uncertainty.
```

## 9. KV Planner Technical Route

The planner receives:

```text
KVObjectMeta[]
ContinuationBelief[]
RuntimeSnapshot
PlannerConfig
```

It outputs:

```text
KVDecision[]
```

The planner should include four decision layers:

1. Decode protection

Keep or prefetch KV needed by active decode workflows. Decode TPOT should not be
hurt by aggressive migration.

2. Urgency-aware prefetch

If CPU-resident KV is likely to be used soon and H2D transfer fits within the
remaining time window, prefetch it.

3. Pressure-aware GPU residency selection

Under high HBM pressure, keep GPU KV objects with high expected utility density:

```text
expected_saved_ms / size_bytes
```

where expected saved time accounts for reuse probability, recompute cost, and
transfer cost.

4. Low-survival recompute policy

For low-confidence or low-survival branches, avoid expensive offload/reload.
Prefer recompute later or drop GPU residency.

The planner should not be a single score formula. It should be a staged
algorithm with safety rules first, then utility optimization.

## 10. Cost Model Technical Route

The cost model should estimate:

```text
KV bytes per token
D2H transfer time
H2D transfer time
prefill recompute time
decode TPOT impact
HBM pressure
```

For a transformer-like model:

```text
kv_bytes_per_token =
  2 * num_layers * num_kv_heads * head_dim * dtype_bytes
```

Offload should only be chosen when:

```text
expected benefit of freeing HBM
  >
D2H cost + expected H2D reload cost + miss penalty
```

The first version can use measured PCIe bandwidth and measured prefill
throughput. Later versions can fit per-model/per-GPU profiles.

## 11. Runtime Adapter Route

Do not directly copy the full serving runtime into this repository. Use narrow
patches or adapters.

Required hook groups:

```text
metadata.patch
  Add workflow, agent, branch, subagent, and policy metadata.

snapshot.patch
  Export scheduler and KV cache snapshots.

actions.patch
  Apply keep, offload, prefetch, materialize, recompute decisions.

metrics.patch
  Record TTFT, TPOT, HBM, migration, recompute, and policy counters.
```

Likely SGLang 0.5.2rc1 hook points:

```text
python/sglang/srt/managers/io_struct.py
python/sglang/srt/managers/scheduler.py
python/sglang/srt/managers/schedule_batch.py
python/sglang/srt/mem_cache/radix_cache.py
python/sglang/srt/mem_cache/hiradix_cache.py
python/sglang/srt/server_args.py
```

Before runtime integration, implement a dry-run adapter that logs decisions but
does not move KV tensors.

## 12. Metrics and Experiment Route

Every experiment should record:

```text
git commit
SGLang version
model path
GPU model
CUDA version if available
policy name
config file
workload path
random seed
```

Primary metrics:

```text
workflow latency p50/p95/p99
TTFT p50/p95/p99
TPOT p50/p95/p99
throughput
HBM peak
GPU KV resident bytes
CPU KV resident bytes
D2H bytes
H2D bytes
prefetch hit rate
offload miss penalty
recompute tokens
policy overhead
```

Recommended baselines:

```text
vanilla prefix cache
LRU-like offload
agent-step policy
template-aware policy
BeliefKV without prediction
BeliefKV without transfer-cost awareness
BeliefKV without decode protection
BeliefKV full
```

Recommended validation splits:

```text
within-session split
cross-session split
cross-provider split: claude -> codex, codex -> claude
cross-tool-heavy split: shell-heavy vs search/file-heavy
```

## 13. Implementation Order

Recommended next implementation sequence:

1. TraceLab raw loader and normalizer.
2. Hierarchical action schema.
3. Context tree predictor.
4. Predictor evaluation scripts.
5. Integration from predictor output to `ContinuationBelief`.
6. Planner ablation switches.
7. Dry-run runtime adapter.
8. Runtime metadata patch.
9. Runtime snapshot patch.
10. Runtime action patch.
11. Experiment runner and result summarizer.

The next concrete files to add are:

```text
beliefkv/traces/loader.py
beliefkv/traces/normalizer.py
beliefkv/policy/predictors/base.py
beliefkv/policy/predictors/context_tree.py
scripts/build_context_tree.py
scripts/eval_predictor.py
tests/test_trace_normalizer.py
tests/test_context_tree_predictor.py
```

## 14. Key Risks

1. Prediction overfitting

Mitigation: abstract symbols, context pruning, smoothing, confidence-aware
fallback, cross-workload validation.

2. Low-confidence predictions causing harmful offload

Mitigation: planner should be conservative under low confidence.

3. Backend-version confounding

Mitigation: first compare on the same SGLang `0.5.2rc1` backend, then test
newer SGLang separately.

4. Subagent explosion

Mitigation: root-workflow budget plus subagent-local accounting.

5. Runtime patch complexity

Mitigation: keep policy pure Python first; use narrow adapter hooks; validate
with dry-run before moving tensors.

6. TraceLab replay fidelity

Mitigation: train predictor on raw trace data, not on prompt-synthesized replay
workloads.

## 15. Review Checklist

Before coding the next module, check:

- Is the action schema expressive enough for agent, tool, handoff, and subagent?
- Does the predictor output belief and confidence, not just a hard next action?
- Does the planner have a safe low-confidence behavior?
- Are TraceLab raw data and replay workload clearly separated?
- Are experiments able to separate prediction accuracy from KV scheduling
  benefit?
- Are root workflow fairness and subagent lifecycle both represented?
- Are SGLang runtime changes narrow enough to port later?
