# BeliefKV P6 Future-HBM Risk-Shadow 短实验

## 目的

本轮只验证 future KV growth HBM chance constraint 是否进入真实 GPU
risk-shadow 路径，不评价 workflow JCT、任务正确率或预测动作收益。预测动作保持
只读，`prediction_used=false`。

## 配置

- 模型：Qwen3-Coder-30B-A3B-Instruct-FP8，单张 RTX 6000 Ada。
- KV pool：163,840 tokens；Host HiCache：96 GiB；context length：262,144。
- 策略：冻结的 P5 observed JointPlan + P6 read-only risk shadow。
- Predictor：`frontier_belief_mvp_v6_calibrated_dev.json`。
- Workload：固定 development shards `p6-013` 与 `p6-014`，按 4 -> 8 -> 16
  workflow 分阶段加入。同一 server、配置和策略下只运行一次。
- 停止条件：出现可评估的 risk package 后立即受控停止，不等待任务结束。因此本轮
  workload 结果不具备 JCT 或 terminal-outcome 数据资格。

## 实现

候选时间线维护事件驱动的 HBM ledger：

1. 从当前 GPU KV 使用量和全部 reservation 开始。
2. H2D 完成时计入恢复 KV。
3. 每个 prefill/decode quantum 完成时按新增 token 和模型 KV bytes/token 计入
   future KV growth。
4. 只有明确归属于该 invocation 的 admission reservation 才能抵扣对应增长。
5. 任一 scenario 超过 KV pool 时记录 peak/overflow；package 的可行概率低于
   0.95 时以 `future_hbm_chance_constraint` 拒绝。

同时修复了一个在线 scope 错误：共享 system-prefix lock 不再把所有 active
invocation 合并成一个超预算语义原子组。当前动作仅包含 PREPARE_HOST/PREFETCH_GPU，
不会改变 active owner；Radix lock 和 allocator 可行性继续由 safe-point physical
committer 作确定性硬校验。未来加入 predictive retraction 时，必须为该动作单独扩展
完整 blocker closure。

## 结果

- `predictive_risk_shadow`：4,006 条；内部异常：0。
- `belief_compose_failed`：0，scope 修复生效。
- 实际进入候选评估：13 个 decision epoch，共 78 个 candidate package summary。
- 78/78 package 被 future-HBM chance constraint 拒绝。
- 观测到的最坏 future HBM peak：21,990,506,496 bytes。
- 观测到的最坏 overflow：5,884,379,136 bytes。
- 所有 evaluated epoch 选择 `observed_baseline`，没有实际执行预测迁移。
- 这些 belief 均为 `backoff`，因此 predictive prefetch 仍被安全门禁禁止。

## 负结果与结论边界

当前实现不满足在线开销和新鲜度门槛：

- evaluated planning：P50 4,896.565 ms，P95/P99 9,012.245 ms，范围
  4,262.423--9,012.245 ms。
- 旧汇总口径下，4,006/4,006 risk-shadow 结果对应的全局 source JointPlan 已经
  stale；该口径比较整个 graph/allocator/topology revision，并不是 predictive
  package 的 action-specific read set。
- 主要 stale 原因是 allocator version、plan expiration、graph version 和 topology
  version 变化。

本轮所有 78 个候选还同时触发 deterministic hard constraint，因此只能证明
future-HBM **拒绝路径已接入**，不能据此证明该 chance constraint 具有独立判别力，
也不能由全局 stale 推断 4,006 个具体 predictive action 都不可提交。

后续代码修复已拆分 observed/predictive worker，在 belief compose 前加入
action-specific eligibility，增加联合 reclaim+prefetch/partial-prefetch 候选、
deterministic preflight、cooperative cancellation、bucket/hysteresis 触发和候选级
certificate。四类受控测试已经分别覆盖当前不可行、当前可行但未来超配、当前及未来
均安全、reclaim 后可安全 prefetch。上述内容尚未经过新的 GPU 固定 trace 性能门槛，
因此本报告仍不能支持“P6 已可在线使用”或性能收益主张。

## 数据

- 运行目录：`experiments/shadow/p6_risk_shadow_growth_short/20260808T083942Z`
- 最终摘要：`risk_shadow_summary.json`
- Runtime summary：`server/latest_runtime_summary.json`
- Audit：`server/runtime_audit.jsonl`

受控 shutdown 已产生 ACK。所有 restore obligation、transaction、command、lease 和
funding 在关闭前守恒；实验结束后无残留 SGLang GPU 进程或 workload 容器。
