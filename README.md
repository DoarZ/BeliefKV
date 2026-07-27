# BeliefKV

BeliefKV is a research implementation of workflow-aware KV-cache lifecycle and
agent scheduling control for memory-constrained, single-GPU multi-agent
serving. It discovers dynamic workflow structure from runtime events; it does
not require the application to submit a complete agent DAG before execution.

The current runtime target is **SGLang 0.5.2rc1** at commit
`18f91eb639084825717c0e3c3c7273492812ab71`. The current GPU workload uses
**Qwen3-Coder-30B-A3B-Instruct-FP8**, tensor parallelism 1, LangGraph/Deep
Agents orchestration, isolated SWE-bench tool containers, and a single GPU.

> Current status (2026-07-27): P2 reactive control and P4 observed JointPlan
> shadow are integrated. P5A observed admission, P5B selective running-batch
> retraction, and P5C transactional retraction/residency control are implemented
> behind explicit flags. Their CPU and interface tests pass, but the complete
> real-GPU correctness and performance gate is still open. No P5 performance
> claim should be made from the current repository.

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
- asynchronous observed JointPlan shadow with bounded planning work and
  current-state validation;
- optional observed active-set admission that limits lock and private-KV growth;
- optional selective running-batch retraction and the P5C transaction
  `RETRACT -> physical OFFLOAD/DROP -> ACK -> replacement ticket`;
- barrier request/drain/outcome attribution and read-only
  `TentativeUnlockPreview` for estimating reclaimable physical closure;
- unified telemetry for BeliefKV commands and native HiCache demand-load or
  write-back operations;
- deterministic simulation, trace replay, timeline rendering, validation, and
  immutable experiment artifacts;
- a narrow, versioned patch over SGLang instead of a permanent serving fork.

P6 frontier prediction and an online predictive JointPlan are **not**
implemented. Context summarization or compression is intentionally outside the
KV control-plane boundary and remains an agent-runtime responsibility.

## Repository Environments

The maintained local workflow uses three Conda environments:

| Environment | Python | Purpose |
| --- | --- | --- |
| `beliefkv` | 3.10 | BeliefKV control plane, patched SGLang server, tests, replay, and visualization |
| `beliefkv-agents` | 3.11 | LangGraph/Deep Agents workload client and Docker tool orchestration |
| `beliefkv-swe` | harness-defined | Official SWE-bench patch correctness evaluation only |

Create or update the first two environments from the repository root:

```bash
cd /home/longhao/experiment/BeliefKV

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

The latest complete local result is `404 passed, 7 skipped`.

## End-to-End GPU Experiment

The commands below are the maintained P5 observed-control experiment path. They
start one model server and then submit 24 dynamic agent workflows in three
batches of eight. Do not start this experiment while another process owns the
GPU.

### 1. Preflight

```bash
cd /home/longhao/experiment/BeliefKV

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
cd /home/longhao/experiment/BeliefKV

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$PWD/experiments/raw/p5_observed_24/$RUN_ID"

conda run --no-capture-output -n beliefkv \
  python scripts/prepare_deepagents_server_config.py \
  --server-dir "$RUN_DIR/server" \
  --queue-service-observer \
  --enable-observed-admission \
  --enable-running-retraction

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

The capacity check must report at least `163840` total KV tokens.

### 4. Launch the 24-workflow agent workload

In a separate workload terminal, set `RUN_DIR` to the exact value printed by
step 2:

```bash
cd /home/longhao/experiment/BeliefKV

RUN_DIR=/home/longhao/experiment/BeliefKV/experiments/raw/p5_observed_24/REPLACE_WITH_RUN_ID
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
  --arrival-batch-interval-seconds 20 \
  --max-turns 18 \
  --max-completion-tokens 4096 \
  --timeout 900 \
  --activation-wall-clock-seconds 1800 \
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

The workload uses real SWE-bench inputs, persistent multi-turn peers, runtime
subagent selection within a required range, tool waits, join/reactivation, and
per-workflow Docker workspaces with networking disabled. It does not require
the SWE-bench harness because this run measures load, scheduling, KV lifecycle,
and migration. Run the official harness separately when measuring patch
correctness.

Important argument semantics:

- `--activation-wall-clock-seconds 1800` is a hard liveness deadline shared by
  one root activation and its children. It is not a request timeout or latency
  SLO. Deadline expiry cancels outstanding child requests.
- `--timeout 900` is the per-API-request cap, further bounded by the remaining
  activation deadline.
- the `--min-*-requests` and `--min-*-fraction` arguments are post-run validity
  gates. They do not cap LLM/tool calls or force an agent to stop at those
  values.
- the fixed `--run-namespace` keeps runtime identities stable across matched
  policy variants. Use a new namespace when changing the workload definition.
- do not use `--flush-cache` in only one member of a comparison. Cold/warm cache
  state must be held constant across variants.

The runner writes to `workloads.incomplete` during execution and atomically
renames it to `workloads` only after producing the run manifest. Exit code 2
means the workload completed but failed one or more validity gates; it does not
mean that artifacts were discarded.

### 5. Stop the server

After the workload process exits, press `Ctrl-C` in the foreground server
terminal. Confirm that no process still owns the GPU:

```bash
nvidia-smi
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
  activation_wall_clock_seconds
}' "$RUN_DIR/workloads/manifest.json"

rg -n 'Traceback|ERROR|KeyError|APITimeoutError|out of memory' \
  "$RUN_DIR/server/server.log" "$RUN_DIR/workloads"
```

Render the HBM/Host KV and physical D2H/H2D timeline, then validate transfer
ordering and allocator ownership consistency:

```bash
conda run --no-capture-output -n beliefkv \
  beliefkv render-transfer-timeline \
  "$RUN_DIR/server/runtime_audit.jsonl" \
  "$RUN_DIR/kv_transfer_timeline.html"

conda run --no-capture-output -n beliefkv \
  beliefkv validate-transfer-telemetry \
  "$RUN_DIR/server/runtime_audit.jsonl" \
  --config "$RUN_DIR/server/beliefkv_config.json" \
  --output "$RUN_DIR/transfer_validation.json"
```

Only non-zero, physically observed DMA contributes to the D2H/H2D lanes.
Rejected or zero-byte attempts remain visible in audit data but are not counted
as transfer bandwidth. Native HiCache and explicit BeliefKV transfers are
distinguished by `telemetry_origin`.

## Runtime Modes

Create a separate run directory for every mode. Never overwrite a config or
mix server logs from different policies.

| Mode | Config-generator flags | Purpose |
| --- | --- | --- |
| P2 reactive | `--disable-policy-shadow` | Online reactive residency/admission path without JointPlan shadow |
| P4 shadow | no P5 flags | Read-only observed JointPlan; decisions do not alter admission or residency |
| P5 observed | `--queue-service-observer --enable-observed-admission --enable-running-retraction` | Current online admission/retraction transaction gate |

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

## SGLang Integration

The exact patched runtime is expected at `third_party/sglang`:

```bash
cd /home/longhao/experiment/BeliefKV
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
[docs/architecture_status_zh.md](docs/architecture_status_zh.md). The active
improvement plan is
[docs/beliefkv_hicache_joint_control_improvement_plan_2026-07-18_zh.md](docs/beliefkv_hicache_joint_control_improvement_plan_2026-07-18_zh.md).
The current dynamic agent workload and its validity gates are described in
[docs/experiments/beliefkv_p3_agentic_workload_2026-07-22_zh.md](docs/experiments/beliefkv_p3_agentic_workload_2026-07-22_zh.md).

## Research Boundary

Mechanism tests, timeout-terminated workflows, guard-forced completions, and
invalid workload manifests are not paper performance evidence. A publishable
result requires clean workflow completion, stable control-plane transaction
lifecycle, allocator/Radix consistency, measured GPU and PCIe behavior,
matched workload traces, and faithful external baselines.
