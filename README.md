# BeliefKV

BeliefKV is a research implementation of workflow-aware KV-cache lifecycle and
agent scheduling control for memory-constrained, single-GPU multi-agent
serving. It discovers dynamic workflow structure from runtime events; it does
not require the application to submit a complete agent DAG before execution.

The current runtime target is **SGLang 0.5.2rc1** at commit
`18f91eb639084825717c0e3c3c7273492812ab71`. The current GPU workload uses
**Qwen3-Coder-30B-A3B-Instruct-FP8**, tensor parallelism 1, LangGraph/Deep
Agents orchestration, isolated SWE-bench tool containers, and a single GPU.

> Current status (2026-08-11): P5 observed JointPlan, transactional restore,
> and selective retraction have passed their focused correctness gates. P6
> R0--R4 integrate frontier belief, action-specific risk, predictive
> `PREPARE_HOST`, and retraction annotations under the same JointPlan authority.
> R5 v7 is invalid for performance comparison because its B arm exposed a
> duplicate allocator/Radix ownership defect. The defect is fixed and a 2.659
> GB deterministic D2H/H2D restore gate passed 14/14 checks; the v9 A/B inputs
> are frozen, but the six formal paired runs remain pending. No P6 online
> performance claim is made yet.

## Implemented System

- atomic Runtime Causal Context Graph (RCCG) for call, spawn, join, message,
  handoff, tool, LLM, reactivate, and terminal events;
- context-to-Radix ownership with allocation generations, shared-page charging,
  engine locks, semantic pins, stale-command rejection, and physical bundles;
- root-workflow admission and attained-service fairness with causal-frontier
  ordering;
- reactive D2H/H2D, non-destructive shadow copies, page-safe Radix arbitration,
  typed blockers, explicit asynchronous ACKs, and retry-storm suppression;
- Host KV lifecycle management, including terminal private-KV cleanup,
  capacity-aware Host eviction, and measured HBM/Host occupancy;
- latest-wins asynchronous semantic JointPlan with bounded planning work;
- safe-point local validation and physical materialization of at most one
  current Radix transfer bundle per epoch;
- bounded-seed, optimized, emergency, and no-action modes under one JointPlan
  contract; reactive transfer is disabled as an online policy source in P5;
- root-workflow active windows that retain all waiting workflows in RCCG and
  fairness accounting while limiting ticket-eligible working sets;
- optional observed active-set admission that limits lock and private-KV growth;
- explicit semantic retraction intents and the P5C transaction
  `RETRACT -> physical OFFLOAD/DROP -> ACK -> replacement ticket`;
- separate queue/admission wait, GPU-service inactivity watchdog, and
  activation deadline semantics;
- bounded asynchronous audit logging, periodic atomic runtime summaries, and
  two-phase transaction drain/abort on shutdown;
- barrier request/drain/outcome attribution and read-only
  `TentativeUnlockPreview` for estimating reclaimable physical closure;
- unified telemetry for BeliefKV commands and native HiCache demand-load or
  write-back operations;
- runtime-owned 32K dynamic-history compaction for parent and child agents,
  with an 8K recent tail, a durable task-state summary, and an explicit
  `CONTEXT_COMPACT` KV ownership transition;
- deterministic simulation, trace replay, timeline rendering, validation, and
  immutable experiment artifacts;
- a narrow, versioned patch over SGLang instead of a permanent serving fork.

P6 now includes the load-independent demand schema, structured local model,
closure-complete scenario composer, candidate-specific service timeline, formal
dataset gates, and asynchronous risk-planning path. Predictive online authority
is disabled by default. The explicit semantic overlay can authorize only
non-destructive `PREPARE_HOST`; `PREFETCH_GPU` additionally requires a bounded
canary. Predictive retraction, drop, and reclaim-and-prefetch remain disabled.
All fixed P5 w4 traces are development-only correctness evidence. Formal fitting
rejects them by provenance and also requires at least 5 train projects, 40
distinct tasks, and 40 workflow rollouts. As of 2026-08-04, the local-label corpus
contains 11,491 fit-eligible decision points but only 5 projects and 14 rollout
sources. One requests recovery rollout is usable only before runtime intervention
and is not a clean JCT or terminal episode. The training command therefore still
fails the diversity gate. Decision-point count is not treated as independent.
Summary generation remains an agent-runtime responsibility.
BeliefKV consumes only the framework-neutral compaction event and releases old
KV ownership; it does not choose summary contents or decide task progress.

## Repository Environments

The maintained local workflow uses three Conda environments:

| Environment | Python | Purpose |
| --- | --- | --- |
| `beliefkv` | 3.10 | BeliefKV control plane, patched SGLang server, tests, replay, and visualization |
| `beliefkv-agents` | 3.11 | LangGraph/Deep Agents workload client and Docker tool orchestration |
| `beliefkv-swe` | harness-defined | Official SWE-bench patch correctness evaluation only |

Create or update the first two environments from the repository root:

```bash
git clone git@github.com:SJTU-DDST/BeliefKV.git
cd BeliefKV
export BELIEFKV_ROOT="$PWD"

# First installation.
conda env create -f environment.yml
conda env create -f environment-agents.yml

# For an existing installation, use update instead.
conda env update -n beliefkv -f environment.yml
conda env update -n beliefkv-agents -f environment-agents.yml

# Install the test-only extra in the server/control environment.
conda run -n beliefkv python -m pip install -e ".[dev]"
```

`environment.yml` installs BeliefKV itself but does not build SGLang from
source. The current server environment resolves SGLang from
`third_party/sglang/python`. A clean server deployment must prepare the pinned
SGLang tree as described in [SGLang Integration](#sglang-integration).

Run the CPU regression suite before changing the runtime patch or policy:

```bash
conda run --no-capture-output -n beliefkv pytest -q
```

At the 2026-08-11 checkpoint, the split CPU regression is `668 passed, 9
skipped` (plus 6 unittest subtests) in `beliefkv` and `157 passed` across the
Deep Agents, LangGraph, collection, dataset, and characterization paths in
`beliefkv-agents`. Keep the environments split: the control environment
intentionally does not install Deep Agents.

For moving the project, model weights, datasets, and ignored experiment
artifacts to another host, follow [the migration guide](docs/migration_guide_zh.md).

## End-to-End GPU Experiment

The maintained P5 correctness path uses the native Deep Agents autonomous root
and runtime-selected FRESH subagents. The synthetic peer workflow remains an
optional stress workload; it is not the primary correctness or JCT gate. Do not
start an experiment while another process owns the GPU.

### 1. Preflight

```bash
cd "$BELIEFKV_ROOT"

nvidia-smi
test -f /opt/downloaded_models/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8/config.json
test ! -e /tmp/beliefkv-experiments.paused
test -f configs/workloads/deepagents_swebench_sympy_24.json
docker image inspect \
  swebench/sweb.eval.x86_64.sympy_1776_sympy-20590:latest >/dev/null
```

Also verify that TCP port `18000` is free. The model server uses GPU 0 only,
TP=1, a 262,144-token context limit, a 163,840-token KV pool, and a 96 GB Host
HiCache. `MEM_FRACTION_STATIC=0.952` is the calibrated capacity boundary for
the current card; changing the model, SGLang version, or GPU requires a new
calibration.

### 2. Create a fresh run directory and server config

Run this in the server terminal:

```bash
cd "$BELIEFKV_ROOT"

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$PWD/experiments/raw/p5_observed_24/$RUN_ID"

conda run --no-capture-output -n beliefkv \
  python scripts/prepare_deepagents_server_config.py \
  --server-dir "$RUN_DIR/server" \
  --queue-service-observer \
  --enable-online-joint \
  --enable-observed-admission \
  --enable-running-retraction \
  --joint-workflow-active-window 12 \
  --request-queue-timeout-seconds 1800

CONTROL_SOCKET=$(jq -r '.runtime_event_socket_path' \
  "$RUN_DIR/server/beliefkv_config.json")
printf 'RUN_DIR=%s\nCONTROL_SOCKET=%s\n' "$RUN_DIR" "$CONTROL_SOCKET"
```

The generator refuses to overwrite an existing run config or event socket.
This protects raw evidence from accidental reuse. Keep the printed `RUN_DIR`;
the workload terminal must use the same absolute path.

By default, prediction and recompute-drop are disabled. Running-batch
retraction may only evict KV through a physically acknowledged transaction.
Do not add `--allow-running-retraction-recompute-drop` to the main experiment
unless recomputation is the explicit ablation under test.

### 3. Start SGLang

Continue in the server terminal:

```bash
CUDA_VISIBLE_DEVICES=0 \
MEM_FRACTION_STATIC=0.952 \
MAX_TOTAL_TOKENS=163840 \
CONTEXT_LENGTH=262144 \
MAX_RUNNING_REQUESTS=32 \
HICACHE_SIZE_GB=96 \
scripts/launch_deepagents_swebench_server.sh "$RUN_DIR/server"
```

The launcher runs in the foreground and redirects SGLang output to
`$RUN_DIR/server/server.log`. It does not install a daemon or restart policy.
Use another terminal to monitor startup:

```bash
tail -F "$RUN_DIR/server/server.log"
```

Host HiCache allocation can make startup take about one minute. Submit no
workload until both checks succeed:

```bash
curl -fsS http://127.0.0.1:18000/health
curl -fsS http://127.0.0.1:18000/get_server_info | jq '{
  max_total_num_tokens,
  context_length,
  mem_fraction_static
}'
```

The capacity check must report at least `163840` total KV tokens. The server
hard context limit intentionally remains larger than the runtime's 32K dynamic
history budget: the static system prompt/tool schema and the summary request
need independent headroom.

### 4. Run the fixed w4 autonomous correctness gate

In a separate terminal, use the exact `RUN_DIR` from step 2:

```bash
cd "$BELIEFKV_ROOT"

RUN_DIR="$BELIEFKV_ROOT/experiments/raw/p5_observed_24/REPLACE_WITH_RUN_ID"
CONTROL_SOCKET=$(jq -r '.runtime_event_socket_path' \
  "$RUN_DIR/server/beliefkv_config.json")

conda run --no-capture-output -n beliefkv-agents \
  python scripts/run_deepagents_swebench.py \
  --mode autonomous \
  --workload-manifest configs/workloads/codex_swebench_sympy_subagents.json \
  --max-workflows 4 \
  --concurrency 4 \
  --sampling-seed 17 \
  --recursion-limit 512 \
  --request-timeout 900 \
  --control-socket "$CONTROL_SOCKET" \
  --server-audit "$RUN_DIR/server/runtime_audit.jsonl" \
  --server-events "$RUN_DIR/server/runtime_events.sglang.jsonl" \
  --server-log "$RUN_DIR/server/server.log" \
  --gate system \
  --output "$RUN_DIR/workloads"
```

`system_jct_eligible` is the P5 gate. It requires complete request/tool/event
pairing, child RETURN/JOIN closure, usable runtime-control delivery, and no
infrastructure failure. `native_agent_jct_eligible` additionally excludes
protocol repair or guard-assisted completion. SWE-bench patch correctness is a
third, independent result and must be measured with the official harness.

### 4a. Optional 24-workflow peer stress workload

In a separate workload terminal, set `RUN_DIR` to the exact value printed by
step 2:

```bash
cd "$BELIEFKV_ROOT"

RUN_DIR="$BELIEFKV_ROOT/experiments/raw/p5_observed_24/REPLACE_WITH_RUN_ID"
CONTROL_SOCKET=$(jq -r '.runtime_event_socket_path' \
  "$RUN_DIR/server/beliefkv_config.json")

conda run --no-capture-output -n beliefkv-agents \
  python scripts/run_langgraph_peer_workloads.py \
  --output-dir "$RUN_DIR/workloads" \
  --workload-manifest configs/workloads/deepagents_swebench_sympy_24.json \
  --run-namespace beliefkv-swebench-sympy24-v1 \
  --backend agentic \
  --spawn-policy required-range \
  --min-initial-subagents 2 \
  --max-initial-subagents 4 \
  --modes mixed,mixed,cyclic \
  --max-workflows 24 \
  --concurrency 24 \
  --arrival-mode batched \
  --arrival-batch-size 8 \
  --arrival-batch-interval-seconds 600 \
  --max-turns 18 \
  --max-completion-tokens 4096 \
  --context-window-tokens 32768 \
  --context-keep-tokens 8192 \
  --intermediate-completion-tokens 1024 \
  --summary-completion-tokens 2048 \
  --timeout 900 \
  --sandbox-command-timeout-seconds 600 \
  --workflow-wall-clock-seconds 7200 \
  --recursion-limit 512 \
  --stuck-repeated-call-limit 6 \
  --stuck-consecutive-error-limit 6 \
  --stuck-no-progress-limit 8 \
  --stuck-max-model-calls 48 \
  --stuck-max-tool-calls 128 \
  --min-workflow-llm-requests 16 \
  --min-workflow-tool-calls 16 \
  --min-subagent-llm-requests 4 \
  --min-subagent-tool-calls 3 \
  --min-dynamic-subagent-fraction 0.30 \
  --min-peer-reactivation-fraction 0.20 \
  --control-socket "$CONTROL_SOCKET" \
  --min-kv-pool-tokens 163840
```

This optional workload uses real SWE-bench inputs, persistent multi-turn peers, runtime
subagent selection within a required range, tool waits, join/reactivation, and
per-workflow Docker workspaces with networking disabled. It does not require
the SWE-bench harness because this run measures load, scheduling, KV lifecycle,
and migration. Run the official harness separately when measuring patch
correctness.

Important argument semantics:

- `--workflow-wall-clock-seconds 7200` is an external safety watchdog shared
  by the workflow and all descendants. It is not a request timeout or latency
  SLO. Expiry aborts active SGLang requests, cancels child/JOIN state, stops the
  isolated sandbox, and marks the trace as censored rather than naturally
  failed. `--activation-wall-clock-seconds` remains a deprecated command-line
  alias for reproducibility of older runs.
- Model/tool call thresholds are telemetry-only by default. Use
  `--enforce-stuck-call-budgets` only for an explicit bounded-loop ablation;
  normal agent workloads terminate through model completion, repeated-failure
  or no-progress detection, user cancellation, or the external watchdog.
- `--timeout 900` is carried to SGLang as a GPU-service inactivity watchdog.
  A long request may run beyond 900 seconds while it continues completing GPU
  batches; only 900 seconds without completed GPU service triggers scheduler
  cancellation. Client transport waits are bounded by the remaining workflow
  deadline, not by time already spent in queue.
- `--sandbox-command-timeout-seconds 600` bounds one offline tool command. It
  is independent of model-request and workflow deadlines, and prevents normal
  repository test commands from being censored by the older implicit 180-second
  limit.
- server queue/admission wait is measured separately and bounded by
  `--request-queue-timeout-seconds` in the generated server config.
- `--context-window-tokens 32768` limits dynamic parent/child message history,
  excluding the static system prompt and tool schema. Crossing it summarizes
  older history and retains about 8K recent tokens. The summary call has a
  separate context and is excluded from business subagent/LLM intensity
  metrics. `CONTEXT_COMPACT` advances the parent epoch and makes its old KV
  ownership droppable; the server `CONTEXT_LENGTH` is not reduced to 32K.
  Persistent peers carry only the explicit summarization cursor across
  activations and append new messages; a general LangGraph checkpointer is not
  used, so loop-guard and structured-response state cannot leak into the next
  activation.
- ordinary tool-decision turns use a 1,024-token output cap; internal summaries
  use 2,048 tokens, while terminal and guard-forced completion retain the
  4,096-token cap. These are upper bounds, not target output lengths. Reserving
  4,096 tokens for every short tool turn would enlarge SGLang admission/KV
  demand without improving ordinary tool calls.
- the 24-workflow pressure run uses three waves of eight workflows separated by
  ten minutes. Report performance over the steady-state interval from the last
  wave's admission until fewer than eight root workflows remain active. Keep the
  full drain for correctness, but do not treat the low-concurrency drain tail as
  steady-state throughput. A fixed w4 gate is a correctness test and is not
  expected to sustain high KV pressure until its final second.
- `--joint-workflow-active-window 12` limits ticket-eligible workflows per
  scheduler epoch. Inactive workflows remain visible and continue to accrue
  root-workflow fairness priority.
- the `--min-*-requests` and `--min-*-fraction` arguments are post-run validity
  gates. They do not cap LLM/tool calls or force an agent to stop at those
  values.
- the required initial subagent range applies only to the first model decision
  batch containing accepted `task` calls. Later runtime-driven spawn batches
  remain dynamic and are reported separately; they do not invalidate the
  initial fanout gate.
- the fixed `--run-namespace` keeps runtime identities stable across matched
  policy variants. Use a new namespace when changing the workload definition.
- do not use `--flush-cache` in only one member of a comparison. Cold/warm cache
  state must be held constant across variants.

The runner writes to `workloads.incomplete` during execution and atomically
renames it to `workloads` only after producing the run manifest. Exit code 2
means the workload completed but failed one or more validity gates; it does not
mean that artifacts were discarded.

### 5. Stop the server

After the workload exits, request scheduler shutdown through the explicit
two-phase handshake. The helper validates PID start time, waits for transaction
drain and `shutdown_ack.json`, validates the final summary, and only then stops
the frontend:

```bash
scripts/stop_deepagents_swebench_server.sh "$RUN_DIR/server"
nvidia-smi
jq '{shutdown_state, correctness_gates}' \
  "$RUN_DIR/server/latest_runtime_summary.json"
```

Do not use `nohup`, an auto-restart loop, or a service manager for controlled
experiments. Those mechanisms previously made stopped runs difficult to
distinguish from residual SGLang processes.

### 6. Inspect and validate artifacts

The main files are:

```text
RUN_DIR/
  server/
    beliefkv_config.json
    server.log
    latest_runtime_summary.json
    runtime_audit.jsonl
    runtime_events.sglang.jsonl
    transfer_telemetry.jsonl
    policy_snapshots.jsonl.gz
  workloads/
    manifest.json
    WORKFLOW_ID/
      runtime_events.agentic.jsonl
      workspace.json
      result.json
      model.patch
```

Check the workload and server logs before interpreting performance:

```bash
jq '{
  experiment_valid,
  workload_coverage,
  server_capacity,
  workflow_wall_clock_seconds
}' "$RUN_DIR/workloads/manifest.json"

jq '{shutdown_state, transactions, correctness_gates}' \
  "$RUN_DIR/server/latest_runtime_summary.json"

rg -n 'Traceback|ERROR|KeyError|APITimeoutError|out of memory' \
  "$RUN_DIR/server/server.log" "$RUN_DIR/workloads"
```

Validate correctness and transfer ownership first, then render the HBM/Host KV
and physical D2H/H2D timeline:

```bash
conda run --no-capture-output -n beliefkv \
  bash scripts/finalize_agentic_gpu_run.sh "$RUN_DIR"
```

The finalizer exits before rendering unless every direct workflow is
system-JCT eligible (or every legacy peer workflow is clean-JCT eligible), all
online actions have a JointPlan ID,
there are no pending transactions, shutdown completed, and transfer/allocator
consistency passes. Failed runs retain raw audit and validation data but do not
produce a new timeline.

Only non-zero, physically observed DMA contributes to the D2H/H2D lanes.
Rejected or zero-byte attempts remain visible in audit data but are not counted
as transfer bandwidth. Native HiCache and explicit BeliefKV transfers are
distinguished by `telemetry_origin`.

### Deterministic P5 restore micro-gate

Run this once before freezing the P5 physical interface. Use a separate server
directory and add `--enable-restore-micro-gate` to the config command from step
2. The option is rejected unless online JointPlan, observed admission, running
retraction, and runtime GPU service observation are all enabled.

For this correctness probe only, start SGLang with
`MAX_RUNNING_REQUESTS=2`. The victim and anchor must occupy both running slots
before the replacement is submitted; otherwise all three requests can be
admitted into the 163,840-token pool and no restore transaction is exercised.
The runner checks this server setting before sending any request and fails
closed on a mismatch.

After the server is healthy:

```bash
conda run --no-capture-output -n beliefkv \
  python scripts/run_restore_micro_gate.py \
  --runtime-audit "$RUN_DIR/server/runtime_audit.jsonl" \
  --output-dir "$RUN_DIR/workloads"

scripts/stop_deepagents_swebench_server.sh "$RUN_DIR/server"

conda run --no-capture-output -n beliefkv \
  python scripts/verify_restore_micro_gate.py \
  --runtime-audit "$RUN_DIR/server/runtime_audit.jsonl" \
  --runtime-summary "$RUN_DIR/server/latest_runtime_summary.json" \
  --output "$RUN_DIR/restore_micro_gate_validation.json"
```

This probe is a correctness test, not a benchmark. It marks one serviced
victim and one waiting replacement, forces exactly one retraction action inside
an atomic JointPlan group, and then returns to the production transaction path.
The verifier requires non-zero D2H and H2D, a durable obligation satisfied
before shutdown, a post-restore decode quantum, physical snapshot coverage,
transaction conservation, and an acknowledged shutdown. The hook is disabled
for all normal and performance runs.

## Agent Runtime Safety Boundary

Agent termination and tool recovery are runtime responsibilities, not KV policy.
The framework-neutral primitives live in `beliefkv/runtime/agent_safety.py`; the
LangChain status adapter lives in `beliefkv/runtime/langchain_tool_safety.py`.
BeliefKV consumes the resulting runtime events but does not decide whether an edit,
test, or workflow succeeded.

- Tool output content is returned unchanged to the model. Semantic failures from
  `edit_file`, `apply_patch`, and non-zero `execute` results carry
  `ToolMessage.status="error"`.
- Every tool boundary records a parameter signature and error class. Mutating tools
  additionally record before/after workspace digests; read-only tools avoid that cost.
- Physical failures and circuit-suppressed repeated intents are counted separately.
  An identical deterministic failure is suppressed in the same workspace epoch. A
  successful call with no output and no workspace change may execute twice for
  confirmation; the third identical call is suppressed until the workspace epoch
  changes. Derived suppression events never count as physical executions.
- Repeated-call, error-rate, and no-progress patterns are telemetry only. They do not
  remove tools, alter prompts, or force a model terminal result in formal collection.
- The 384-step threshold is telemetry only. LangGraph keeps a 512-step absolute fuse
  with a 32-step terminal reserve as process-level fault isolation, not normal
  scheduling.
- The absolute workflow deadline aborts active SGLang requests, cancels outstanding
  child/JOIN state, stops the per-workflow sandbox, and emits a structured
  `WORKFLOW_END(outcome="workflow_timeout")` event.

This contract keeps Deep Agents/LangGraph liveness behavior outside residency,
retraction, and JointPlan algorithms. Runtime adapters may be replaced as long as
they preserve the same event and terminal-state contract.

## Runtime Modes

Create a separate run directory for every mode. Never overwrite a config or
mix server logs from different policies.

| Mode | Config-generator flags | Purpose |
| --- | --- | --- |
| P2 reactive | `--disable-policy-shadow` | Online reactive residency/admission path without JointPlan shadow |
| P4 shadow | no P5 flags | Read-only observed JointPlan; decisions do not alter admission or residency |
| P5 observed | `--queue-service-observer --enable-online-joint --enable-observed-admission --enable-running-retraction` | Semantic JointPlan controls execution/admission/residency/retraction; reactive transfer cannot become a second online policy source |
| P6 risk shadow | P5 flags plus `--enable-predictive-risk-shadow --predictor-model experiments/models/frontier_belief_mvp_v6_calibrated_dev.json --gpu-service-model experiments/models/gpu_service_curve_cluster_cal_qwen3coder30b_rtx6000ada_20260804T060824Z.json --transfer-service-model experiments/models/transfer_service_morphology_gpu0_dev_v2.json` | Read-only A0/PREPARE_HOST/PREFETCH_GPU risk evaluation; the current transfer artifact conditions on bytes and extent count and is single-GPU development evidence |
| P6 semantic overlay | P6 risk-shadow flags plus `--enable-predictive-joint-overlay` | A selected `PREPARE_HOST` becomes a semantic intent in the current JointPlan; the safe point rebuilds and validates the live Radix bundle before dispatch |
| P6 prefetch canary | P6 semantic-overlay flags plus `--enable-predictive-prefetch-canary` | Also allows one in-flight `PREFETCH_GPU`, capped at 5% of the configured KV pool and guarded by reentry, KV-growth, HBM, and timing checks |
| P6 single PREPARE canary | P6 semantic-overlay flags plus `--predictive-prepare-canary-limit 1` | Allows at most one naturally selected PREPARE_HOST in the server run; it does not force or relax selection |

`--enable-joint-predictive` is retained only for historical config compatibility and
does not change ordering, victim selection, or migration. P6 risk planning runs
on the latest-wins worker. Only `--enable-predictive-joint-overlay` gives a
selected semantic intent online authority; the asynchronous physical certificate
is diagnostic and is never executed. Safe-point rematerialization failure falls
back to the unchanged observed JointPlan.

P6 KV actions use action-projected scenario reduction. Continuous particles are
clustered on the variables required by the candidate action, so unrelated action
boundary OOD state cannot create an opaque OTHER rejection. Transfer timing must
be warm-started from a hardware/model-specific artifact; online telemetry updates
that prior instead of resetting every server process to nominal PCIe bandwidth.

Before enabling any single-action validation, replay both transfer models on
the same frozen snapshots and classify decision relevance by direction:

- `shape_action_gate`: byte-only rejects while extent-count-aware accepts the
  same paired candidate. This opens one shape-aware PREPARE canary.
- `shape_veto_gate`: byte-only accepts while extent-count-aware rejects the
  same paired candidate. This opens a byte-only treatment with shape-aware/P5
  no-action control, not a shape-aware canary.
- `selected_action_gate`: the snapshot-level selected actions differ. This is
  reported separately and does not by itself authorize a PREPARE canary.

```bash
SNAPSHOTS=experiments/shadow/p6_predictive_overlay_fixed/20260809T105733Z/server/policy_snapshots.jsonl.gz
GPU_MODEL=experiments/models/gpu_service_curve_cluster_cal_qwen3coder30b_rtx6000ada_20260804T060824Z.json
TRANSFER_MODEL=experiments/models/transfer_service_morphology_gpu0_dev_v2.json

conda run -n beliefkv python scripts/replay_predictive_risk.py \
  --snapshots "$SNAPSHOTS" --gpu-service-model "$GPU_MODEL" \
  --transfer-service-model "$TRANSFER_MODEL" --transfer-model byte-only \
  --output experiments/analysis/p6_m5_byte_only_replay.jsonl

conda run -n beliefkv python scripts/replay_predictive_risk.py \
  --snapshots "$SNAPSHOTS" --gpu-service-model "$GPU_MODEL" \
  --transfer-service-model "$TRANSFER_MODEL" --transfer-model morphology-aware \
  --output experiments/analysis/p6_m5_morphology_aware_replay.jsonl

conda run -n beliefkv python scripts/compare_predictive_replays.py \
  --byte-only experiments/analysis/p6_m5_byte_only_replay.jsonl \
  --morphology-aware experiments/analysis/p6_m5_morphology_aware_replay.jsonl \
  --output experiments/analysis/p6_m5_replay_comparison.json

conda run -n beliefkv python scripts/analyze_predictive_prepare_canary.py \
  --comparison-summary experiments/analysis/p6_m5_replay_comparison.json \
  --output experiments/analysis/p6_m6_canary_gate.json
```

Do not force an intent or relax risk constraints. Add
`--predictive-prepare-canary-limit 1` to that trace's semantic-overlay server
configuration only when `shape_action_gate=true`; the compatibility field
`online_canary_gate` now has exactly that promotion-only meaning. When
`shape_veto_gate=true`, run the byte-only action arm and use shape-aware/P5
no-action as the control. Pass the resulting `runtime_audit.jsonl` to the
canary analyzer.

Decision-relevance characterization must use a predeclared task batch. Freeze
the predictor, transfer-service artifact, risk thresholds, workload manifest,
and arrival policy before collecting the batch; report promotion, veto,
selected-action changes, and shape support for every paired candidate. Do not
keep selecting new traces until a positive case appears.

For a matched comparison, hold constant the workload manifest, stable
`--run-namespace`, model, SGLang commit, KV/Host capacity, arrival schedule,
tool image, cache-state policy, and randomizable workload inputs. P5 is not yet
stable enough to justify adding approximate implementations of external
baselines to the online critical path.

## Non-GPU Checks

Run a deterministic migration scenario without writing artifacts:

```bash
conda run -n beliefkv beliefkv simulate \
  examples/subagent_shadow_scenario.json --no-write
```

Run the built-in reactive/shadow/full simulator ablation:

```bash
conda run -n beliefkv beliefkv matrix \
  examples/subagent_ablation_matrix.json \
  --run-id local-smoke \
  --output-root experiments/results
```

Replay the maintained B0 observed-state baseline over saved snapshots:

```bash
conda run -n beliefkv python -m beliefkv.experiments.policy_replay \
  SNAPSHOTS.jsonl OUTPUT_DIR --run-id reactive-replay
```

Earlier B1-B4 same-data-plane sketches were removed because they were not
faithful native reproductions. P8 will add only baselines that can be
implemented against a frozen workload and a defensible common interface.

### P6 training-evidence characterization

After a fixed run has completed and the server has produced its final shutdown
summary, generate P6.0 coverage and versioned training-evidence tables offline:

```bash
conda run --no-capture-output -n beliefkv \
  python scripts/characterize_p6_coverage.py RUN_DIR \
  --output experiments/processed/P6_RUN/coverage.json \
  --dataset-dir experiments/processed/P6_RUN/dataset \
  --split-manifest configs/p6/swebench_verified_split_v1.json
```

The analyzer accepts both legacy
`workloads/*/runtime_events.agentic.jsonl` traces and current Deep Agents
`workloads/workflows/*/runtime_events.deepagents.jsonl` traces. LLM calls are
joined only by an exact native request ID with matching workflow and invocation
identity; ordinal matching is intentionally disabled.

The dataset contains request calls, request-level GPU identity intervals,
unique `sample_id` batch-service rows, external waits, reentry labels, and
PCIe/HiCache operations. Its manifest
records source/output hashes, project-level split groups, label semantics,
training eligibility, and foreign-key/uniqueness checks. A zero exact-action
boundary count permits remaining-decode-demand training only; it must not be treated
as an action-hazard label. Missing direct DMA timestamps likewise restrict the
transfer target to submit-to-complete operation latency.

Freeze project-level splits and prepare immutable collection batches with the
`beliefkv-swe` environment. The checked-in `collection_v3` plan contains 91
workflows over 11 repositories; SymPy remains development-only:

```bash
conda run -n beliefkv-swe python scripts/freeze_p6_splits.py \
  --dataset-dir workloads/raw/swebench_verified-91aa3ed/test \
  --dataset princeton-nlp/SWE-bench_Verified \
  --dataset-revision 91aa3ed51b709be6457e12d00300a6a596d4c6a3 \
  --development-project sympy/sympy \
  --output configs/p6/swebench_verified_split_v1.json

conda run -n beliefkv-swe python scripts/prepare_p6_collection_plan.py \
  --dataset-dir workloads/raw/swebench_verified-91aa3ed/test \
  --split-manifest configs/p6/swebench_verified_split_v1.json \
  --source-root workloads/sources/p6_swebench_verified \
  --output-dir configs/p6/collection_v3
```

Run one train batch against a separately launched P5 observed server. The batch
runner validates the frozen plan and manifest digest, disables predictor and
predictive actions, applies the versioned repository/image harness profile, and writes its policy
contract before issuing requests. Calibration and test batches are sealed by
default and require explicit `--allow-calibration` or `--allow-test`:

```bash
# Required once on a host that will run psf__requests-5414.
scripts/build_p6_harness_images.sh

conda run --no-capture-output -n beliefkv-agents \
  python scripts/run_p6_collection_batch.py \
  --collection-plan configs/p6/collection_v3/collection_plan.json \
  --batch-id p6-012-train-mixed-r0 \
  --control-socket "$CONTROL_SOCKET" \
  --server-audit "$RUN_DIR/server/runtime_audit.jsonl" \
  --server-events "$RUN_DIR/server/runtime_events.sglang.jsonl" \
  --server-log "$RUN_DIR/server/server.log" \
  --output "$RUN_DIR/workloads"
```

`--pool-tokens` is the minimum server-reported `max_total_num_tokens` required by
the run (default 163,840). The runner queries `/get_server_info` before creating
workload output, rejects silently clipped pools, and uses the actual value for all
resident-pressure metrics. Add `--predictive-risk-shadow-enabled` only when the
server was started in P6 risk-shadow mode; this marks the collection as
development-only without claiming predictive actions were applied.

For a targeted harness recovery run, repeat `--instance-id` while still naming the
original frozen batch. The runner validates the original plan and manifest digest,
then writes an immutable `runtime_workload_manifest.json` with the selected tasks,
preflight policy, and any source-to-derived image mapping beside the run artifacts:

```bash
conda run --no-capture-output -n beliefkv-agents \
  python scripts/run_p6_collection_batch.py \
  --collection-plan configs/p6/collection_v3/collection_plan.json \
  --batch-id p6-009-train-mixed-r0 \
  --instance-id django__django-11138 \
  --instance-id django__django-14011 \
  --control-socket "$CONTROL_SOCKET" \
  --server-audit "$RUN_DIR/server/runtime_audit.jsonl" \
  --server-events "$RUN_DIR/server/runtime_events.sglang.jsonl" \
  --server-log "$RUN_DIR/server/server.log" \
  --gpu 1 \
  --output "$RUN_DIR/workloads"
```

The sandbox checkout root is always exactly `/workspace`. The common prompt contains
no repository-specific path or test example; a workload-specific profile is injected
into the supervisor, every dynamic subagent, and planned agents. Sandbox preflight
also requires both `pwd -P` and `git rev-parse --show-toplevel` to equal `/workspace`.
Do not silently rewrite a model path because that would change tool semantics.

Known harness-contaminated workflows are listed in a run-level
`TRAINING_EXCLUSIONS.json`. Dataset export retains their raw evidence for audit, clears
their split, and forces every `training_eligible*` label to false. A corrected rerun is
stored as a separate run; raw histories are never edited or spliced together.

Formal collection uses model-terminal semantics: a valid model completion ends
the activation, while repository correctness is evaluated offline. It disables
the harness's LLM-based correctness repair (`completion_repair_attempts=0`),
because those repair calls are a benchmark mechanism rather than part of the
agent workflow being modeled. The autonomous supervisor and every subagent use
one branch-private `ContextLifecycleMiddleware`: dynamic history is compacted at
32K tokens while retaining the newest 8K, intermediate/final calls use 4096
output tokens, and summaries use 2048. This explicit builder replaces Deep
Agents' default 170K summarizer and strips `_summarization_event` from the child
return path, so concurrent children cannot merge private compaction state into
the parent. The runner also fingerprints runtime/control sources before and
after each batch and writes the final eligibility result to
`p6_collection_contract.json`; changed source, incomplete system JCT, censored
collection, or a missing frozen split fails closed.

The first `p6-009` attempt on 2026-08-03 is retained as development diagnostics
only. It exposed correctness-repair workload pollution and a too-short runtime
event ACK window, and is marked by `PILOT_INVALID.json`. Dataset export forces
all of its rows to `development`, and the formal model loader rejects it.
Another interrupted attempt exposed that the autonomous builder had not yet
installed the 32K lifecycle and is marked `COLLECTION_INVALID.json`. Startup
attempts lost to external GPU occupancy are marked `STARTUP_FAILED.json` before
any workload is submitted. All three marker kinds are rejected by the dataset
exporter unless an explicit censored diagnostic export is requested.

The old `p6-009-train-mixed-r0/20260803T100537Z` run is also invalidated. A later
image-identity audit found that its collection path could reuse a repository image
with the wrong task version. Both processed copies now carry
`formal_training_eligible=false`; the historical report is retained only for audit.
See `docs/experiments/beliefkv_p6_collection_p6_009_formal_2026-08-03_zh.md`.

The currently accepted local-label evidence is the four-workflow GPU0 shard of
`p6-010`, the eight-workflow GPU1 `p6-012` batch, and two targeted recovery
rollouts for the failed `p6-010`/`p6-011` tasks. Together they cover five projects
and fourteen rollout sources. The requests recovery contributes only finite-horizon
labels before intervention and is excluded from clean JCT, terminal outcome, and
complete-trajectory analysis. The original `p6-011` shard remains
diagnostic-only because one workflow submitted an 812K-token context and failed
the system gate. The repaired rollout is a separate, provenance-stable source run.
The formal training CLI rejects the accepted corpus because 14 tasks/workflows are
below the 40/40 minimum. The minimum is an engineering guard, not the final paper
target; final fitting should use 80--120 workflow rollouts across all seven train
projects, followed by sealed calibration and unseen-project test splits.

The recovery collector enforces a versioned per-turn tool-observation budget. The
historical targeted runs used progress-based recovery prompts; decision points at or
crossing those runtime interventions remain audit-only and are excluded from Frontier
fitting. This yields 65 natural rows for
`psf__requests-5414` and 402 for `pylint-dev__pylint-4970`; intervention tails are
not learned as autonomous model behavior.

Fit only `train` (or W4 `development` for a non-reportable sanity model), then
calibrate without refitting counts and evaluate a sealed split:

```bash
conda run -n beliefkv-agents python scripts/select_frontier_hyperparameters.py \
  --dataset-dir TRAIN_DATASET_A --dataset-dir TRAIN_DATASET_B \
  --output FRONTIER_LOPO.json

conda run -n beliefkv-agents python scripts/train_frontier_belief.py \
  --dataset-dir TRAIN_DATASET_A --dataset-dir TRAIN_DATASET_B --split train \
  --hyperparameter-selection FRONTIER_LOPO.json \
  --model-version frontier-train-v1 --output MODEL_FIT.json

conda run -n beliefkv-agents python scripts/calibrate_frontier_belief.py \
  --model MODEL_FIT.json --dataset-dir CALIBRATION_DATASET \
  --target-coverage 0.9 --output MODEL_CALIBRATED.json

conda run -n beliefkv-agents python scripts/evaluate_frontier_belief.py \
  --model MODEL_CALIBRATED.json --dataset-dir TEST_DATASET \
  --split test_id --output TEST_METRICS.json
```

The model learns local boundary, remaining decode tokens, tool-risk, next-output,
and prompt-growth distributions. It does not learn GPU milliseconds. RCCG JOIN
structure, physical KV state, and scheduler actions are never learned labels. The
composer constructs top-K demand scenarios plus OTHER and retains complete JOIN/
producer dependencies; JOIN completion is resolved only after candidate JointPlan
physicalization and batch-service simulation.

Before formal fitting, audit the same tasks and semantic prompts under paired
`w1/w4/w8` loads. Each cohort must be exported with the current decision schema.
Only pairs with the same `prompt_semantic_sha256` and explicit sampling seed are
controlled evidence; unseeded pairs remain diagnostic:

```bash
conda run -n beliefkv-agents python scripts/audit_p6_load_invariance.py \
  --cohort w1=experiments/processed/P6_W1/dataset \
  --cohort w4=experiments/processed/P6_W4/dataset \
  --cohort w8=experiments/processed/P6_W8/dataset \
  --output experiments/processed/P6_INVARIANCE.json
```

The expected result is stable action/token demand with load-sensitive request
wall-clock time. Message races, `JOIN_ANY` winners, timeouts, and unseen peer
states remain `OTHER/OOD`; they are not forced into invariant labels.

GPU service and PCIe behavior use independent hardware calibration rather than
agent wall-clock labels. Run the tagged service matrix against an otherwise
idle observed server, then join requests to native `gpu_service_sample` events
and fit the service curve. Train and holdout conditions are separate. Export
restores one complete batch per unique `sample_id`; request-expanded runtime rows
are validation-only and cannot be used for fit:

```bash
conda run -n beliefkv-agents python scripts/run_queue_service_calibration.py \
  --output-dir "$RUN_DIR/gpu_service_benchmark"

conda run -n beliefkv-agents python scripts/export_gpu_service_calibration.py \
  --benchmark-dir "$RUN_DIR/gpu_service_benchmark" \
  --runtime-audit "$RUN_DIR/server/runtime_audit.jsonl" \
  --runtime-events "$RUN_DIR/server/runtime_events.sglang.jsonl" \
  --output-dir "$RUN_DIR/gpu_service_dataset"

conda run -n beliefkv-agents python scripts/train_gpu_service_curve.py \
  --dataset-dir "$RUN_DIR/gpu_service_dataset" \
  --output "$RUN_DIR/gpu_service_curve.json"
```

## SGLang Integration

The exact patched runtime is expected at `third_party/sglang`:

```bash
cd "$BELIEFKV_ROOT"
git clone --branch v0.5.2rc1 https://github.com/sgl-project/sglang.git \
  third_party/sglang
git -C third_party/sglang rev-parse HEAD
git -C third_party/sglang apply \
  "$PWD/patches/sglang-0.5.2rc1-beliefkv.patch"
conda activate beliefkv
cd third_party/sglang/python
python -m pip install -e ".[all]"
cd ../../..
conda run -n beliefkv beliefkv check-sglang "$PWD/third_party/sglang"
```

The reported upstream HEAD must be
`18f91eb639084825717c0e3c3c7273492812ab71`. A modified status in the pinned
tree is expected after applying the patch. Do not upgrade SGLang in place: the
patch touches scheduler, request lifecycle, Radix/HiCache, protocol, abort, and
callback contracts and must be ported and gated as one versioned change.

## Repository Layout

```text
beliefkv/
  control/       RCCG and unified control loop
  core/          event, identity, and configuration contracts
  experiments/   workload, replay, and experiment contracts
  metrics/       immutable artifacts, statistics, and timelines
  policy/        admission, fairness, bundles, JointPlan, transfer control
  predictor/     survival/context-tree/service models
  runtime/       SGLang bridge, page index, Radix arbiter, telemetry
  simulator/     deterministic page/HBM/PCIe simulator
  traces/        normalization, validation, and replay checks
configs/         runtime and frozen workload configurations
docs/            architecture, design, setup, and experiment records
examples/        deterministic smoke scenarios
patches/         exact SGLang integration patch
scripts/         server launchers and reproducible workload runners
tests/           correctness and regression tests
```

The implementation contract and phase status are documented in
[docs/architecture_status_zh.md](docs/architecture_status_zh.md). The authoritative
current design is
[docs/beliefkv_design_2026-07-14_zh.md](docs/beliefkv_design_2026-07-14_zh.md).
The transfer-service sample-gate fix, corrected morphology conclusion, and
native HiCache ownership boundary are recorded in
[docs/experiments/beliefkv_p6_service_contract_and_native_ownership_2026-08-11_zh.md](docs/experiments/beliefkv_p6_service_contract_and_native_ownership_2026-08-11_zh.md).
The earlier P0-P8 implementation plan is retained as historical context in
[docs/beliefkv_hicache_joint_control_improvement_plan_2026-07-18_zh.md](docs/beliefkv_hicache_joint_control_improvement_plan_2026-07-18_zh.md).
The current dynamic agent workload and its validity gates are described in
[docs/experiments/beliefkv_p3_agentic_workload_2026-07-22_zh.md](docs/experiments/beliefkv_p3_agentic_workload_2026-07-22_zh.md).

## Research Boundary

Mechanism tests, timeout-terminated workflows, guard-forced completions, and
invalid workload manifests are not paper performance evidence. A publishable
result requires clean workflow completion, stable control-plane transaction
lifecycle, allocator/Radix consistency, measured GPU and PCIe behavior,
matched workload traces, and faithful external baselines.
