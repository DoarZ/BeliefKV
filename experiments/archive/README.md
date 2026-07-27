# Experiment archive

Superseded, interrupted, and debugging-only raw runs are moved here rather than
deleted. The archive is local-only and excluded from Git; this README is the
tracked inventory.

## 2026-07-27 P1-P5 raw consolidation

Archive root: `experiments/archive/20260727/superseded_raw/`

- `deepagents_swebench/`: P1/P1.5/P2 development and validation runs. The
  original contents are preserved because historical reports cite their
  telemetry, but they are no longer active P5 reproducibility anchors.
- `p3_agentic_waves/` and `p3_gpu_ready_probe/`: P3 workload probes superseded
  by the fixed 24-workflow trace.
- `p4_joint_shadow_24/`: all P4 shadow attempts, including interrupted runs and
  the negative chunked-prefill correctness gate.
- `p5_joint_retraction_24/`: empty/incomplete P5 launch attempt.
- `p5_overlap_retraction_fix_probe/`: interrupted overlap/retraction probe.
- `p5_performance_24/`: invalid 24-workflow run; zero workflows completed.
- `p5_fixed_trace_stability_24/`: fixed-trace control-plane stability evidence.
  Retraction and barrier lifecycles passed, but all workflows hit the 600-second
  activation deadline, so it is excluded from performance results.

This cleanup moved about 33 GiB from active raw storage into the archive. It did
not delete data or reclaim disk space.

## 2026-07-17 cleanup

Archive root: `experiments/archive/20260717/raw/`

- `deepagents_swebench/20260716T160210Z/`: initial smoke iterations superseded
  by concurrent runs.
- `deepagents_swebench/20260716T162833Z/`: early planned run with server errors.
- `deepagents_swebench/20260716T165200Z/`: pre-liveness-fix comparison and
  pressure runs retained as historical evidence.
- `deepagents_swebench/20260716T175417Z/`: H2D trigger deadlock diagnosis.
- `deepagents_swebench/20260716T181114Z/`: prefetch-trigger run that exposed the
  non-idempotent `COMMIT_CPU` acknowledgement.
- `codex_qwen14b/`: all Codex/Qwen2.5-14B gate, timeout, join-guard, incomplete,
  and superseded reactive runs except baseline r5 and reactive r16.

No artifacts were permanently deleted in this cleanup. To restore an archived
run, move it back under `experiments/raw/` using its original basename.

## 2026-07-22 P3 workload reclassification

Archive root: `experiments/archive/20260722/p3_correctness_only/`

- `raw/p3_dynamic/`: one-shot or tool-free cyclic/mixed runs, including the
  12-workflow pressure run. These validate topology, event delivery, physical
  correctness, and capacity pressure only.
- `raw/p3_agentic/`: single-workflow development runs with deliberately small
  turn/model/tool limits. These validate the persistent peer and dynamic task
  mechanism only.
- `raw/deepagents_swebench-20260721T074957Z/`: the single-workflow P3 policy
  snapshot smoke.
- `processed/p3_20260721T074957Z/` and `processed/p3_dynamic_*`: derived B0-B4,
  source-context, timeline, and rolling-oracle artifacts based on the same weak
  workload assumptions.

The files remain available for correctness regression and historical auditing,
but must not be used in a performance table or as evidence for realistic Agent
load. Replacement experiments use productive multi-turn subagents, minimum
LLM/tool intensity gates, and recorded batched root arrivals.

## 2026-07-22 invalid high-pressure attempt

Archive root: `experiments/archive/20260722/p3_invalid_guard_limited/`

- `20260722T062148Z/`: the first eight-workflow batched-arrival run reached high
  HBM and Host-KV pressure, but the workload was interrupted after the legacy
  diagnostic-probe heuristic forced productive agents to complete. At each
  observed trigger, the recent `python -c` calls and outputs were distinct and
  `consecutive_no_progress` was zero. The run is retained only as failure
  evidence and must not be used for performance claims.

The diagnostic-only heuristic was removed. Subsequent runs are invalid unless
every workflow completes naturally without a stuck-guard trigger or forced
semantic completion.

Archive root: `experiments/archive/20260722/p3_invalid_task_loop/`

- `20260722T064010Z/`: diagnostic rerun after removing the false-positive
  probe guard. It reached 14.83/16.11 GB physical HBM KV and 7.16 GB Host KV,
  with 236 LLM requests, 223 repository-tool calls, and five runtime-spawned
  children before deliberate termination. `sympy__sympy-17630` repeated the
  same failing command three times, correctly triggering the generic progress
  guard, so the run is excluded. It established the load envelope and informed
  task selection only.
- `20260722T070454Z/`: eight-workflow rerun with the first task replacement.
  It produced 257 LLM requests, 299 repository-tool calls, four dynamic
  children, and one natural child RETURN/JOIN before termination.
  `sympy__sympy-17318` then accumulated three consecutive command failures and
  repeated the final failing command exactly, so this run is also excluded.

Archive root: `experiments/archive/20260722/p3_invalid_guard_tuning/`

- `20260722T072106Z/`: seven-workflow candidate run stopped when a productive
  child repeated the same successful file read three times. The detection was
  mechanically correct, but the three-call threshold was too aggressive for
  the intended high-intensity workload. Agentic experiment defaults now allow
  six identical calls/errors, while the eight-call no-progress guard and the
  48/128 activation budgets remain as hard liveness bounds.

Archive root: `experiments/archive/20260722/p3_invalid_guard_bug/`

- `20260722T072855Z/`: seven-workflow run that exposed a structured-content
  accounting bug. One child issued 35 distinct successful commands in a
  parallel tool batch, with unique command and output hashes, but
  `ToolMessage.text` flattened the structured results to an empty string and
  falsely reported 35 no-progress calls. Progress hashing now canonicalizes
  the raw `ToolMessage.content`; a regression test covers unmatched parallel
  call IDs and distinct structured outputs.
- `20260722T074817Z/`: follow-up run that exposed call-versus-batch accounting.
  A single AI response launched 44 parallel failing `read_file` calls. The old
  guard counted these as 44 consecutive failed decisions and forced the child;
  the corrected guard counts them as one failed decision batch while retaining
  all 44 calls toward the 128-call hard budget.

Archive root: `experiments/archive/20260722/p3_aborted_autonomous/`

- `20260722T080336Z/`: explicitly stopped at the user's request after confirming
  that unconstrained autonomous spawn remained unstable for a clean workload
  gate. It is retained as resource-characterization evidence only. The next
  bounded probe uses runtime-enforced initial fan-out of 2--4 subagents.
