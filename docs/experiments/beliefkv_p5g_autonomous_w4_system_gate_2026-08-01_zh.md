# BeliefKV P5G Autonomous W4 System Gate（2026-08-01）

## 结论

本轮按固定 manifest 只运行一次 autonomous w4。P5G `system_jct_eligible` gate 通过：4/4 workflow
均自然发布 `WORKFLOW_END`，所有 subagent RETURN、JOIN 和 LLM/tool 调用成对；一次生产路径 running
retraction 完成 D2H、H2D、重新准入和真实 GPU service，restore obligation 在受控 shutdown 前进入
`SATISFIED`。command/ACK、allocator mirror、Host page index 和事务生命周期均守恒。

本轮同时验证了无效 overlap barrier 修复。确定性 micro-gate 中曾出现的 493 次 request/drain 在
autonomous workload 中降为 1 次；该次 drain 创建了有效 JointPlan 并完成 reclaim。active restore
debt 期间的 68 次候选在 drain 前被抑制。

该结论只关闭 P5G 系统正确性门槛。0/4 workflow 满足 `native_agent_jct_eligible`，因为四个 workflow
都出现过 agent guard intervention；2/4 workflow 没有观测到成功测试命令。因此本轮不能用于
agent-native clean JCT、SWE-bench 正确率或策略性能比较。

## 配置

- 原始目录：`experiments/raw/p5g_autonomous_w4/20260801T115617Z/`。
- 模式：Deep Agents autonomous root，4 个固定 SymPy SWE-bench workflow，并发 4。
- 模型：Qwen3-Coder-30B-A3B-Instruct-FP8，TP=1，物理 GPU 1。
- GPU KV pool：163,840 tokens；Host HiCache：96 GB。
- server context：262,144 tokens；`max_running_requests=32`。
- 预测性物理动作与确定性 restore test hook 均关闭。
- workload wall time：3,044.42 秒；server telemetry window：3,284.08 秒。

## Workload 结果

| 指标 | 结果 |
| --- | ---: |
| workflow start / end | 4 / 4 |
| system JCT eligible | 4 / 4 |
| native-agent JCT eligible | 0 / 4 |
| task measurement valid | 2 / 4 |
| LLM submit / result | 575 / 575 |
| tool start / end | 600 / 600 |
| dynamic spawn | 5 |
| join create / wait / satisfied | 5 / 5 / 5 |
| tool success / error | 506 / 94 |

四个 workflow 均 `outcome=completed`，无 API timeout、queue timeout、OOM、allocator/Radix consistency
错误或 admission stall。94 次工具错误均被记录为 `command_failed`；20 次重复工具调用被抑制，8 次
guard intervention 使所有 workflow 最终有界结束。两条 task measurement 无效的直接原因是
`no_successful_test_command_observed`，不是 BeliefKV 数据面故障。

## Barrier 与 Restore

| 指标 | 结果 |
| --- | ---: |
| overlap barrier request / drain | 1 / 1 |
| drain 后 `plan_created` | 1 |
| restore-debt pre-drain suppression | 68 |
| running retraction transaction | 1，`reclaim_confirmed` |
| exact lock-release preview / realized | 6,396,641,280 / 6,396,641,280 bytes |
| restore obligation | 1，最终 `SATISFIED` |
| restored bytes | 6,396,641,280 bytes |
| obligation wait to real service | 4,807.21 ms |
| lazy physical snapshot | 1 次，0.131 ms |
| commit read-set | 1 次，0 stale，0.074 ms |

修复将 restore authority 和 overdue restore debt 检查前移到 barrier request 之前，并要求确定性 test
pair 达到最小 private-KV 物理就绪阈值。生产路径仍需经过原有 safe point、physical closure 和
transactional commit，不以粗粒度预检查替代物理校验。

## 数据面守恒

- command dispatch / ACK：106 / 106，无 missing、orphan、duplicate 或时序错误。
- 显式 DMA telemetry：2/2；全部完成，无 partial、reject 或零字节操作。
- native HiCache telemetry：1,533 条；统一时间线总计 1,535 条物理迁移。
- HBM 峰值：16,106,127,360 bytes；Host 峰值：79,441,428,480 bytes。
- HBM mirror 超过 allocator：0 次；Host residency/page-index mismatch：0 次。
- retry guard：6 个 `descendant_closure` blocker，每个 fingerprint 最多提交一次；相同失败重试为 0。
- shutdown：`acknowledged/final=true`；无 active obligation、lease、funding、grace、inflight command
  或 pending transaction；cleanup 未掩盖 unresolved debt。

## 残余问题与边界

1. agent runtime 的工具恢复和终态质量仍不够干净。94 次 `command_failed` 使 0/4 workflow 满足
   native-agent clean JCT，但没有破坏 system JCT 或物理事务正确性。
2. service-curve holdout 只有 1 个 H2D 样本，静态估计 333.26 ms，实际 callback 538.59 ms，不能据此
   校准 PCIe 尾部模型。
3. 本轮只产生一次 ownership snapshot 和一次生产 retraction，证明功能闭环，不证明 w8 以上规模的
   snapshot 复杂度或策略收益。
4. `physical_commit_budget_exceeded` 和 stale/global validation fallback 计数仍高，属于 P6 前需要继续
   测量的控制面效率问题，但本轮没有造成在线动作绕过 JointPlan 或 liveness 失败。

## 产物

- `experiments/raw/p5g_autonomous_w4/20260801T115617Z/workloads/summary.json`
- `experiments/raw/p5g_autonomous_w4/20260801T115617Z/server/latest_runtime_summary.json`
- `experiments/raw/p5g_autonomous_w4/20260801T115617Z/transfer_validation.json`
- `experiments/raw/p5g_autonomous_w4/20260801T115617Z/kv_transfer_timeline.html`
- `experiments/raw/p5g_autonomous_w4/20260801T115617Z/kv_transfer_timeline.json`

P5 的架构和状态机可据此冻结。预测性物理动作开放前仍需一次短时 w8 correctness smoke；P6 离线
标签、FrontierBeliefModel 和 ScenarioRiskPlanner 开发不再被 P5G system gate 阻塞。
