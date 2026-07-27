# Active raw experiments

This directory contains only active scratch state and the raw runs currently used
as reproducibility anchors. Superseded and interrupted runs are moved to
`experiments/archive/` without changing their contents.

## Retained runs

- `p5_gpu_smoke_8/20260726T165151Z/`: latest bounded GPU correctness smoke;
  three of four workflows reached semantic completion.
- `service_calibration/`: reusable online service-rate calibration inputs.
- `protocol/`: protocol and policy-snapshot integration smoke evidence.
- `codex-qwen14b-subagent-baseline-c4-r5*`: Codex/Qwen2.5-14B observational
  baseline retained for the r5 versus r16 comparison.
- `codex-qwen14b-subagent-reactive-c4-r16*`: event-responsive Codex subagent
  validation referenced by `docs/experiments/codex_qwen14b_subagent_r16_zh.md`.
- `qwen-code/`: reusable local Qwen Code runtime scratch directory.
- `qwen-code-smoke/`: Qwen Code integration evidence referenced by the smoke
  report.
- `swebench-sympy-20590-reactive-s0/`: original SWE-bench pilot evidence.
- `swebench_verified_gold_gate/`: reusable SWE-bench harness gate output.

The raw artifacts are intentionally excluded from Git. See
`experiments/archive/README.md` for the archive policy and inventory.
