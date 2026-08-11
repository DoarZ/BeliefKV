# BeliefKV P6 Worker Split 固定任务 Trace 验证

## 目的

本轮验证 P6 risk-shadow 控制面优化是否降低在线开销并改善计划新鲜度。预测动作
保持只读，`prediction_used=false`；本轮不评价 JCT、任务正确率或预测动作收益。

## 配置与范围

- 模型：Qwen3-Coder-30B-A3B-Instruct-FP8，单张 RTX 6000 Ada。
- KV pool：163,840 tokens；Host HiCache：96 GiB；context length：262,144。
- 策略：冻结的 P5 observed JointPlan + P6 read-only risk shadow。
- Predictor：`frontier_belief_mvp_v6_calibrated_dev.json`。
- Service model：
  `gpu_service_curve_cluster_cal_qwen3coder30b_rtx6000ada_20260804T060824Z.json`。
- 固定任务：与 2026-08-08 短实验相同的 16 个 Django development workflow，
  按 4、4、8 三批加入。

任务集合相同，但本轮实际首请求到达间隔约为 313.1 秒和 352.6 秒；旧实验约为
258.4 秒和 218.6 秒。差异来自手工阶段控制和各批 sandbox preflight 的完成时间。
因此本轮可用于控制面路径与时延 characterization，不能作为严格的负载等价 JCT
或吞吐 A/B。

## 验证实现

本轮覆盖以下修改：

1. observed JointPlan worker 与 predictive risk worker 独立运行。
2. belief compose 前执行 action-specific eligibility gate。
3. 重复的 action/resource bucket 通过 trigger signature 抑制。
4. 候选先执行 deterministic preflight，再进入 scenario risk simulation。
5. 使用有界 service-estimate cache 和 action-specific certificate。
6. 汇总器分别统计无候选 gate、worker queue、belief、candidate、preflight、risk 和
   certificate validation，不再用 0 值稀释昂贵路径 P99。

## 结果

### 触发抑制与 worker 稳定性

- eligibility checks：10,440。
- 实际 enqueue/start/complete：404/404/404。
- unchanged bucket 抑制：10,035；无候选抑制：1。
- 只有 3.87% 的触发进入 risk worker，96.13% 未进入昂贵路径。
- worker failed、pending、dropped 和 superseded 均为 0。
- service-estimate cache：10,188 hits、405 misses，hit rate 96.18%。

相较旧实现的 4,006 条 risk-shadow 计算，本轮只执行 404 次 worker 计算。但由于
实际到达间隔不同，这一数量差异只能证明新 gate 在本轮有效，不能直接换算为系统
吞吐收益。

### 控制面时延

| 阶段 | P50 | P95 | P99 | 结论 |
|---|---:|---:|---:|---|
| no-candidate eligibility | 0.371 ms | 0.956 ms | 11.535 ms | P99 未过 1 ms 门槛 |
| belief compose | 6.931 ms | 29.766 ms | 70.442 ms | 仍有明显长尾 |
| candidate generation | 11.366 ms | 42.507 ms | 83.489 ms | 仍有明显长尾 |
| deterministic preflight | 0.0019 ms | 0.0032 ms | 0.0053 ms | 通过 5 ms 门槛 |
| scenario risk | 7.940 ms | 93.624 ms | 300.879 ms | 未过 20 ms 门槛 |
| worker queue wait | 18.793 ms | 90.468 ms | 138.182 ms | 计划排队明显 |
| trigger-to-validation | 239.466 ms | 622.991 ms | 1159.511 ms | 未过 50 ms 门槛 |

单次 risk planning 的 P50/P99 为 22.771/347.812 ms，最大值 997.959 ms。旧实验
13 个 evaluated epoch 的 P50/P99 为 4,896.565/9,012.245 ms。数值分别下降约
215 倍和 25.9 倍，但两轮 evaluated 集合不同，本结果只能证明数量级改善，不能
解释为等价候选上的严格 speedup。

### 新鲜度与决策覆盖

- 403 个结果均为 `backoff`，最终动作均为 `observed_baseline`。
- 只有 2 个 candidate package 形成 action certificate。
- 2/2 certificate 在验证时 stale，未通过 `<10%` stale-rate 门槛。
- stale 涉及 bundle generation、invocation state/revision、transfer epoch 和
  transfer service curve。
- 2 个候选同时被 future-HBM chance constraint、CVaR 和收益门槛拒绝。
- 最坏预测 HBM peak 为 18,010,472,448 bytes，overflow 为 1,904,345,088 bytes。

这说明 worker 拆分解决了“每个 observed epoch 都执行昂贵预测”的问题，但尚未
解决“异步预测完成时物理对象已经变化”的问题。当前 predictor 全部回退也意味着
本轮不能评价 would-prefetch/would-prepare 的质量或 regret。

## 正确性与收尾

- predictive worker 内部异常为 0。
- scheduler 已返回 shutdown ACK；最终 running/waiting request 为 0。
- 无遗留 SGLang、workload 进程或 BeliefKV sandbox 容器，GPU0/1 显存归零。
- command、lease、funding 和在线 residency transaction 最终无未决项。
- 由于本轮按固定观察窗口主动停止 workload，1 个 restore obligation 以
  `request_aborted` 取消，且 shutdown prepare 时存在 1 个非终态 restore
  transaction。因此本轮不满足 P5 clean-completion gate，不能用来证明 P5 JCT
  或完整 workload liveness；这不影响只读预测控制面的时延统计。

## 结论与下一步

本次修改有效但未达到在线接入条件。已经成立的是：observed worker 不再等待
predictive worker，96.13% 的重复触发没有进入昂贵风险规划，worker 运行稳定，
deterministic preflight 成本可忽略。尚未成立的是：预测路径 P99、新鲜度和有效动作
覆盖率。

下一步不应启用预测性物理动作，优先处理：

1. 将 backoff/support 判定缓存到 semantic eligibility，避免为最终必然回退的状态
   重复执行 compose 与 candidate generation。
2. 将 candidate generation 和 scenario rollout 改为增量、可取消计算，并隔离
   Python GIL/GC 长尾。
3. 异步阶段只发布 semantic action；safe point 再物化 Radix bundle 并生成物理
   certificate。bundle generation 变化不能直接让整个语义候选失效，但重新物化后
   必须重新验证 bytes、HBM、transfer 和收益证书。
4. 将 predictive result drain 与 observed plan 发布解耦，缩短 queue wait 和
   trigger-to-validation。
5. 后续固定 trace 使用自动的 monotonic arrival controller，消除手工投放和
   sandbox preflight 导致的到达间隔漂移。

## 数据

- 运行目录：
  `experiments/shadow/p6_risk_shadow_fixed_optimized/20260809T092524Z`
- 最终摘要：`risk_shadow_summary.json`
- Runtime summary：`server/latest_runtime_summary.json`
- Audit：`server/runtime_audit.jsonl`
- Shutdown ACK：`server/shutdown_ack.json`
