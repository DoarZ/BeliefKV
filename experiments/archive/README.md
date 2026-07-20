# Experiment archive

Superseded, interrupted, and debugging-only raw runs are moved here rather than
deleted. The archive is local-only and excluded from Git; this README is the
tracked inventory.

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
