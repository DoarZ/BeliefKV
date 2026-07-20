# BeliefKV

BeliefKV is a research implementation of workflow-aware KV-cache lifecycle
management for memory-constrained, single-GPU multi-agent serving. The pinned
runtime target is **SGLang 0.5.2rc1** at commit
`18f91eb639084825717c0e3c3c7273492812ab71`.

BeliefKV is an overlay rather than a serving-runtime fork. The repository owns
the causal control plane, prediction and policy modules, trace tooling,
simulator, experiment artifacts, and a narrow versioned SGLang patch.

## Implemented System

- atomic Runtime Causal Context Graph (RCCG) for call, spawn, join, message,
  handoff, tool, and LLM events;
- context-to-Radix-node ownership with allocation generations, shared-page
  charging, engine locks, semantic pins, and stale-command rejection;
- root-workflow admission and attained-service fairness with causal-frontier
  ordering;
- reactive D2H/H2D, page-safe Radix arbitration, and explicit asynchronous ACKs;
- typed physical blockers and an event-gated transfer-attempt ledger that
  suppresses unchanged closure/capacity/lock retry storms;
- non-destructive `GPU_ONLY -> MIRRORING -> DUAL_CLEAN -> CPU_ONLY`
  prepare/commit migration;
- hierarchical Kaplan-Meier tool survival, variable-order semi-Markov action
  context tree, LLM service-cost model, OOD fallback, and online calibration;
- ClawTrace normalization, deterministic page-level simulation, immutable run
  artifacts, ablation matrices, CSV summaries, and bootstrap confidence
  intervals;
- an SGLang 0.5.2rc1 patch for metadata, admission, abort handling, scheduler
  safe points, workflow ordering, and incremental Radix/HiCache observation;
- an optional run-scoped JSONL audit trail for admission, causal identity,
  request lifecycle, and transfer ACK validation without logging prompt text;
- an acknowledged Unix-datagram agent event channel, mini-SWE-agent adapter,
  official SWE-bench evaluation gate, and hash-locked runtime trace validation.

The Python control plane and fake-HiCache integration are covered by the test
suite. The patch is checked against a clean copy of the exact upstream commit
and its modified files compile. A Qwen2.5-0.5B-Instruct CUDA smoke run validated
untagged bypass plus tagged root/spawn-child lifecycle propagation. A
Qwen2.5-7B-Instruct SWE-bench pilot then validated eight real LLM/tool rounds,
the official evaluator, audit consistency, trace freezing, and deterministic
control-plane replay. The model submitted an empty patch, so this remains a
mechanism check rather than an accuracy, throughput, pressure, or stability
result.

## Quick Start

```bash
cd /home/longhao/experiment/BeliefKV
conda env create -f environment.yml
conda activate beliefkv
python -m unittest discover -s tests -v
```

Run a deterministic migration scenario without writing artifacts:

```bash
beliefkv simulate examples/subagent_shadow_scenario.json --no-write
```

Run the built-in reactive/shadow/full smoke ablation:

```bash
beliefkv matrix examples/subagent_ablation_matrix.json \
  --run-id local-smoke \
  --output-root experiments/results
```

Normalize ClawTrace events and fit a predictor artifact:

```bash
beliefkv normalize-clawtrace RAW.jsonl experiments/processed/runtime_events.jsonl
beliefkv train-predictor \
  experiments/processed/runtime_events.jsonl \
  experiments/processed/predictor.json
```

Set `predictor_model_path` in the BeliefKV config to load the artifact. Relative
paths in an SGLang runtime config are resolved relative to that config file.
Set `runtime_audit_path` to a JSONL path when validating an integration. It is
disabled by default and records identifiers and resource counts, not prompts.
When runtime audit is enabled, the pinned SGLang adapter also records measured
HBM/Host KV occupancy. Set `transfer_telemetry_path` for a dedicated stream of
D2H/H2D submit, observable start, completion, physical bytes, and status events.

Render a self-contained KV migration timeline after a run:

```bash
beliefkv render-transfer-timeline \
  RUN_DIR/server/runtime_audit.jsonl \
  RUN_DIR/kv_transfer_timeline.html
```

The report writes a sibling JSON data artifact. Physical D2H/H2D lanes only
count operations with observed non-zero DMA bytes; zero-byte rejects remain in
the audit table and appear on a separate `No DMA` lane. Legacy runs can add
`--metrics-jsonl`, `--kv-bytes-per-token`, and `--hbm-capacity-bytes` to recover
the sampled HBM curve; Host occupancy remains explicitly unavailable when it
was not measured by the original run.

Validate command ordering, allocator/PageOwnershipIndex consistency, the
chronological service-curve holdout, and controller timing after a real run:

```bash
beliefkv validate-transfer-telemetry \
  RUN_DIR/server/runtime_audit.jsonl \
  --config RUN_DIR/server/beliefkv_config.json \
  --output RUN_DIR/transfer_validation.json
```

Only completed physical operations train the service curve. Partial/rejected
outcomes are retained in a separate rolling rejection window and cannot evict
valid bandwidth samples.

## SGLang Integration

```bash
git clone --branch v0.5.2rc1 https://github.com/sgl-project/sglang.git
cd sglang
git rev-parse HEAD
git apply /home/longhao/experiment/BeliefKV/patches/sglang-0.5.2rc1-beliefkv.patch
beliefkv check-sglang "$PWD"
```

The reported HEAD must be
`18f91eb639084825717c0e3c3c7273492812ab71`. See
[docs/setup.md](docs/setup.md) and
[docs/runtime_integration_zh.md](docs/runtime_integration_zh.md) before starting
a model server.

## Repository Layout

```text
beliefkv/
  control/       RCCG and unified control loop
  core/          event, identity, and configuration contracts
  experiments/   reproducible ablation matrix runner
  metrics/       immutable artifacts and statistics
  policy/        admission, fairness, residency, shadow, transfer planning
  predictor/     survival/context-tree/service models and training pipeline
  runtime/       event channel, page index, Radix arbiter, audit, SGLang bridge
  simulator/     deterministic page/HBM/PCIe simulator
  traces/        runtime trace normalization, validation, and replay checks
configs/         runtime configurations
docs/            architecture, setup, design, and experiment guidance
examples/        deterministic smoke scenarios
patches/         exact SGLang integration patch
scripts/         contract checks and reproducible workload runners
tests/           correctness and regression tests
```

The current Chinese system design is
[docs/beliefkv_design_2026-07-14_zh.md](docs/beliefkv_design_2026-07-14_zh.md).
The rendered implementation-status diagrams are in
[docs/architecture_status_zh.md](docs/architecture_status_zh.md).
The first real SWE-bench pilot, its exact scope, and reproduction commands are
documented in
[docs/swebench_pilot_2026-07-15_zh.md](docs/swebench_pilot_2026-07-15_zh.md).
The local Qwen3-Coder-30B-A3B-FP8 and Qwen Code parent/subagent protocol smoke,
including the autonomous-delegation failures, is documented in
[docs/experiments/qwen3_coder_qwencode_smoke_2026-07-16_zh.md](docs/experiments/qwen3_coder_qwencode_smoke_2026-07-16_zh.md).
The corrected KV-capacity boundary, context-window policy, and disposable
tool-sandbox validation are documented in
[docs/experiments/qwen3_coder_capacity_sandbox_calibration_2026-07-16_zh.md](docs/experiments/qwen3_coder_capacity_sandbox_calibration_2026-07-16_zh.md).
The fixed-model Qwen Code versus Codex runtime gate, including the corrected
resident-KV pressure and autonomous-spawn negative result, is documented in
[docs/experiments/qwen3_coder_runtime_pair_gate_2026-07-16_zh.md](docs/experiments/qwen3_coder_runtime_pair_gate_2026-07-16_zh.md).
The `recursion_limit=200` paired gate, structured completion protocol, and
progress-guard results are documented in
[docs/experiments/agent_loop_termination_gate_2026-07-17_zh.md](docs/experiments/agent_loop_termination_gate_2026-07-17_zh.md).
The P1.5 retry-guard implementation, frozen 268-retry replay, real-pressure
mechanism result, and remaining paired-performance gate are documented in
[docs/experiments/beliefkv_p1_5_retry_guard_2026-07-19_zh.md](docs/experiments/beliefkv_p1_5_retry_guard_2026-07-19_zh.md).
The older technical archive is historical and not the implementation contract.

## Research Boundary

The built-in scenario is a mechanism test, not evidence for a paper claim. A
publishable evaluation still requires real coding/browser/research workloads,
the same model and SGLang commit for every baseline, GPU-side interference and
allocator measurements, failure injection, predictor cross-workload splits,
and comparison with an offline oracle.
