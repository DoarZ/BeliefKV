# BeliefKV P6 Predictive Overlay 固定 Trace 验证

## 目的

验证 P6 semantic predictive overlay 开启后是否会绕过 observed JointPlan，以及
`PredictiveIntent -> safe-point rematerialization -> residency transaction` 是否能在
固定 development trace 中形成在线动作。本轮只评价控制路径和安全性，不评价 JCT。

## 配置

- 运行目录：
  `experiments/shadow/p6_predictive_overlay_fixed/20260809T105733Z`。
- 模型：Qwen3-Coder-30B-A3B-Instruct-FP8，单张 RTX 6000 Ada。
- KV pool：163,840 tokens；Host HiCache：96 GiB；context length：262,144。
- 策略：P5 observed JointPlan + P6 semantic overlay。
- 在线权限：`PREPARE_HOST`；单 in-flight、最多 5% KV pool 的
  `PREFETCH_GPU` canary。
- Predictor：`frontier_belief_mvp_v6_calibrated_dev.json`。
- Service model：
  `gpu_service_curve_cluster_cal_qwen3coder30b_rtx6000ada_20260804T060824Z.json`。
- Workload：与上一轮相同的 16 个 Django development workflow，按 4/4/8
  三批启动。

三批首请求的实际相对到达间隔为约 338.4 秒和 373.9 秒；历史轮次为 313.1 秒和
352.6 秒。任务、批大小和顺序固定，约 20--25 秒偏差来自 sandbox preflight，
因此本轮不是严格 wall-clock A/B。

## 结果

### 预测控制面

- eligibility：10,069 次。
- risk worker 实际启动/完成：1,170/1,170，failed、pending、dropped 均为 0。
- 8,898 次 unchanged-bucket 和 1 次 no-candidate 被抑制。
- 形成 1,169 条完整 risk 结果，全部选择 `observed_baseline`。这表示没有最终
  eligible/selected action，不表示候选均无正收益。
- planning P50/P95/P99：32.88/165.41/413.27 ms。
- eligibility P95/P99：2.86/11.59 ms，满足动作接入使用的 10/20 ms 同步门槛。
- service-estimate cache hit rate：98.62%。

后台 planner 的长尾没有阻塞 observed worker。复核后发现，1,169 个完整
`PREFETCH_GPU` 候选的 expected benefit 均为正；其中 15 个同时通过 deterministic、
future 和 future-HBM feasibility，5 个还满足 5% canary 容量限制。这 5 条记录来自
同一 context 的相邻 epoch，代表一个独立机会。它们仅因
`other_has_no_finite_risk_bound` 被拒绝，因此本轮没有发布 `PredictiveIntent`。

### 候选拒绝原因

- `PREPARE_HOST`：4 个候选，4/4 因 expected benefit 不足被拒绝，其中 1 个同时
  超过 CVaR risk budget。该动作没有受到 absolute future-HBM constraint 误拒绝。
- `PREFETCH_GPU`：1,169 个候选，0 eligible。旧实现按包含连续变量的完整 tuple
  对粒子去重，top-8 之外的大部分概率进入不透明 OTHER；boundary OOD 随后通过
  `other_has_no_finite_risk_bound` 一票否决原本可行的 prefetch。这是风险表示缺陷，
  不是 workload 没有机会。
- `PARTIAL_PREFETCH_GPU`：1,154 个只读候选，0 eligible，且该动作本来就没有在线
  authority。
- 最坏 future-HBM peak 为 20.41 GB，相对 16.11 GB KV pool 的最坏 overflow 为
  4.31 GB。

旧异步 physical certificate 的 stale rate 为 97.16%。它仍是诊断数据，不是在线
performance gate，也不会被执行；
本轮因为没有 semantic intent，无法测量新的 safe-point rematerialization 成功率。

### 在线动作与正确性

- predictive publish/commit/dispatch/ACK：0/0/0/0。
- predictive residency transaction：0；未决 predictive transaction：0。
- predictive worker 内部异常：0。
- P5 observed retraction transaction：11 个，11/11 residency ACK，11/11 completed。
- 全部 command、lease、funding 和在线 residency transaction 在最终状态无未决项；
  共观察到 232 个 transfer ACK。

原观测代码对没有 intent 的空 safe-point 调用也计时，18,978 个空样本中有 2 个因
GC/调度抖动超过 1 ms，被误记为 fallback。该问题不产生动作，也不改变 observed
计划。实验后已修复为只有存在 semantic intent 时才进入计时和预算门禁；因此本轮
不存在可解释为真实 predictive safe-point latency 的样本。

### 收尾边界

本轮按固定观察窗口主动停止，而不是等待 16 个任务自然结束。shutdown prepare 时
仍有 7 running、8 waiting request；7 个 restore obligation 最终以
`runtime_shutdown` 取消。因此本轮不通过 P5 clean-completion gate，不能用于 JCT、
workflow liveness 或任务正确性结论。

Scheduler 返回 shutdown ACK，最终无 SGLang/workload 进程和 BeliefKV sandbox
容器，GPU0/GPU1 显存均归零。

## 结论

本轮证明了 overlay 开启没有绕过旧风险门禁，也没有破坏 P5 事务正确性；但旧风险
表示把至少一个真实机会系统性封死，不能据此判断这些候选风险不可控。由于 selected
action 覆盖率为 0，尚未验证真实
`publish -> rematerialize -> dispatch -> ACK` GPU 闭环，不能据此声称预测性迁移
已经在线有效。

修复 action-projected scenario reduction 和 PREPARE 两阶段 recourse 后，应先离线
replay 本轮已有的 310.5 MB 机会。GPU 验证分成 injected-intent mechanism micro 和
naturally-selected policy micro，且不需要重复本轮 16-workflow 实验。

## 数据

- Risk summary：`risk_shadow_summary.json`。
- Runtime summary：`server/latest_runtime_summary.json`。
- Audit：`server/runtime_audit.jsonl`。
- Runtime events：`server/runtime_events.sglang.jsonl`。
- Shutdown ACK：`server/shutdown_ack.json`。

## 修复后离线 Replay

对本轮 `policy_snapshots.jsonl.gz` 的 126 个稀疏 PolicyInput 使用修复后的
action-projected reducer 和持久化 transfer curve 重放：

- 126 seen，2 个缺少 prediction，87 evaluated，37 skipped；
- 102 个正收益候选，0 eligible；
- `other_has_no_finite_risk_bound` 为 0，所有 action-projected scenario 的 OTHER
  probability mass 为 0；
- 102 个正收益候选当前均被 future-HBM chance constraint 拒绝；
- replay artifact：
  `experiments/analysis/p6_predictive_overlay_fixed_action_projected_replay_v2.jsonl`。

这不能推翻原始 audit 中的 5 个 310.5 MB 可行机会。对应 snapshot ID 没有被旧的
10 秒持久化间隔保存，当前 replay 的输入中不存在那 5 个 epoch。runtime 已增加
“正收益候选触发、按机会签名去重、异步写入”的精确快照路径，后续 short trace
可以直接重放自然机会。

## 后续短 Trace 诊断

`experiments/shadow/p6_action_projected_short/20260809T123833Z` 启用了
action-projected reduction、两阶段 PREPARE recourse 和持久化 transfer curve。
运行时仍然没有候选，随后定位为另一个独立问题：snapshot eligibility 把非破坏性的
`PREPARE_HOST` 与立即 `COMMIT_CPU` 共用了 `descendant_closure` 判定。

修复后离线重放同一短 trace 的 45 个 snapshot，得到 57 个 closure-complete
`PREPARE_HOST` 候选，证明该候选封死问题已解除。后续审计纠正了本报告早期的错误
解读：41/57 候选实际预测到 capacity pressure，最大 overflow 约 5.78 GB；只是大部分
scenario 中 shadow 无法在 pressure 前完成，或 pressure 已晚于 parent reentry。

进一步修复 transfer curve 的跨 size 外推后，456 个 scenario 中有 2 个满足完整
recourse 条件。旧 replay 的 D2H 收益来自新 transfer curve，interference 却来自旧
snapshot，两者比例相差约 2.47 倍，所以 0 个正收益不能作为一致模型下的结论。
统一证据后的敏感性 replay 表明：0% 干扰时有 2 个 eligible 动作；10.2% 时只剩 1 个
正收益动作；14.1% 时正收益归零；5% 干扰下虽有正收益，但仍超过当前 10 ms CVaR
budget。最终结论是“存在 pressure 和少量可行时间窗，是否可上线取决于 D2H overlap
实测”，而不是“没有 pressure”。最新 replay 为
`experiments/analysis/p6_action_projected_short_recourse_stall_*.jsonl`，详细记录见
`beliefkv_p6_predictive_joint_overlay_implementation_2026-08-09_zh.md`。
