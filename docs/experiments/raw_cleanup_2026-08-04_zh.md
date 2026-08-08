# 2026-08-04 Raw 实验清理记录

本轮只清理由后续修复轮次取代的失败 P5 raw 数据。实验结论、报告和最终
P5G/P6 证据保留。以下目录在删除前已确认没有运行中的服务引用：

- `experiments/raw/p5_observed_24`
- `experiments/raw/p5d_online_joint_24`
- `experiments/raw/p5d_restore_obligation_w4`
- `experiments/raw/p5e_clean_completion_w4`
- `experiments/raw/p5e_execution_watchdog_w4`
- `experiments/raw/p5e_model_terminated_w4`
- `experiments/raw/p5e_ordinary_waiting_restore_w4`
- `experiments/raw/p5e_restore_funding_fix_w4`
- `experiments/raw/p5e_restore_lease_w4`
- `experiments/raw/p5e_restore_rebind_w4`
- `experiments/raw/p5e_restore_service_grace_w4`
- `experiments/raw/p5f_context_compaction_w4`
- `experiments/raw/p5f_fixed_w4`
- `experiments/raw/p5f_private_state_w4`
- `experiments/raw/p5f_retry_guard_ack_w4`
- `experiments/raw/p5f_terminal_protocol_w4`
- `experiments/raw/p5_gpu_smoke_8`

以下数据明确保留：

- `p5g_restore_micro`、`p5g_ownership_snapshot_w4`、`p5g_autonomous_w4`
- `p6_0_labels_w4`，仅作为 development-only 管线证据
- `p6_agent_semantics_v1`，其中失效批次带显式 invalid marker
- `p6_gpu_service_v2/v3` 及对应 processed/model artifact

固定 P5 w4 数据不得用于正式 FrontierBeliefModel 训练。正式训练入口要求
P6 collection provenance、冻结的 project split，并执行 project/task/workflow
多样性门槛。
