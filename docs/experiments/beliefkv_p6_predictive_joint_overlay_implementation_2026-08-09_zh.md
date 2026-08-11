# BeliefKV P6 Predictive JointPlan Overlay 实现记录

## 目标

将异步 FrontierBelief/risk planner 的结果以语义意图接入 JointPlan，同时避免执行
异步阶段绑定的旧 Radix bundle。该记录只说明实现和 CPU correctness，不构成 GPU
性能结论。

## 在线边界

- 默认关闭：`predictive_joint_overlay_enabled=false`。
- `PREPARE_HOST`：显式打开 overlay 后可提交；保留 GPU copy，不使用 absolute
  future-HBM chance constraint。
- `PREFETCH_GPU`：还需打开 canary；最多一条 in-flight，单次 copy 不超过配置 KV
  pool 的 5%。
- `PARTIAL_PREFETCH_GPU`、`RECLAIM_AND_PREFETCH`、预测性 retraction/drop：仅保留
  shadow 评估，不产生在线 intent。
- 旧 physical action certificate 仍只用于 stale characterization，绝不直接执行；
  被选中的 intent 携带精简 causal certificate，并在发布前和 safe point 各校验一次。

## 数据流

```text
Observed JointPlan / bounded seed
             +
PredictiveIntent(causal read-set, risk, timing, benefit envelope)
             |
             v
scheduler safe point
  action-specific support/time gate
  causal certificate revalidation
  restore/PCIe authority check
  live Radix bundle rematerialization
  current HBM/Host/closure + benefit-envelope validation
             |
             v
ALL_OR_NOTHING predictive ActionGroup
             |
             v
existing residency transaction + ACK path
```

预测 intent 使用动作相关预测头：

- `PREPARE_HOST`：`remaining_window`；允许带 calibrated interval 的 backoff。
- `PREFETCH_GPU`：`reentry_window`、`future_kv_growth` 和 future-HBM feasibility。
- boundary 头不可用不再阻断上述 KV 动作。

时间门禁为：

```text
remaining_low - plan_age > transfer_p95 + commit_guard
```

prefetch 还必须进入 lead window：

```text
remaining_low - plan_age <= transfer_p95 + desired_lead
```

任何 intent 校验、物理化、预算或 enqueue 失败都只撤销预测 action，并回退到未修改
的 observed JointPlan。预测 action 不抢占 observed residency，也不创建 restore debt
或 running retraction。

## 配置

在完整 P5 + P6 risk-shadow 参数后增加：

```bash
--enable-predictive-joint-overlay
```

开启 prefetch canary 时再增加：

```bash
--enable-predictive-prefetch-canary
```

旧 `--enable-joint-predictive` 仍是无决策权的兼容字段。

## CPU 验证

截至 2026-08-09，相关回归覆盖：

- calibrated backoff 的动作相关支持度；
- PREPARE_HOST 与 absolute future-HBM overflow 解耦；
- semantic intent 不包含 physical bundle evidence；
- safe-point live bundle rematerialization；
- predictive ActionGroup 与现有 command transaction 入队；
- prefetch 超过 5% KV pool 时保持 observed epoch 不变；
- action-projected particle clustering 保留全部概率质量，KV 动作不受无关 boundary
  OOD 污染；
- future feasibility 与 future-HBM chance constraint 独立归因；
- PREPARE_HOST 只在 shadow 完成早于 pressure 且 pressure 早于 reentry 时获得
  recourse credit；
- transfer curve 可按硬件/model 条件持久化并在 server 启动时 warm-start；
- 正收益候选按动作、context、字节桶和拒绝原因去重后触发精确 PolicyInput 异步
  持久化；
- snapshot 侧 `PREPARE_HOST` 按 GPU descendant closure 投影：
  `descendant_closure` 通过同一 bundle 吸收，engine lock、active owner、in-flight、
  semantic pin 和 Host capacity 仍是硬约束；D2H 成本计完整 closure，收益只计目标
  context 的 exclusive bytes；
- safe point 在多个 live preview 中选择与 `target_bytes_hint` 最接近的 bundle，不再
  无条件选择最小 closure；选择前必须满足 `min_reclaimable_bytes`、
  `max_cross_context_bytes` 和 `max_copy_bytes`，避免物理可执行但收益语义已经失效；
- transfer service 在目标 bucket 稀疏时只使用同方向、同 compute phase、相邻
  size bucket 的有限邻域；该机制准确称为 bounded neighboring-bucket
  extrapolation，而不是插值。诊断同时输出最近 bucket 距离、样本数、估计来源和
  样本覆盖的 size range。

本轮最终 CPU 回归为 core/runtime 侧 598 passed、8 skipped、3 subtests passed，
Deep Agents/P6 collection 侧 90 passed。
固定 development trace canary 已完成，结果见
`beliefkv_p6_predictive_overlay_fixed_trace_2026-08-09_zh.md`。修复后对该轮 126 个
已持久化稀疏快照执行离线 replay：OTHER 拒绝为 0，得到 102 个正收益候选，但均未
通过 future-HBM gate。旧日志中审计发现的 5 个 310.5 MB canary 机会所在 epoch
没有被 10 秒固定快照采样命中，不能从现有 snapshot 精确重放。下一轮已通过候选
触发持久化修复该观测缺口。

GPU 验证只执行 injected-intent mechanism micro 和 naturally-selected short micro，
不重复原 16-workflow 长实验。前者只验证 publish/rematerialize/dispatch/ACK；后者
才评价 planner 是否自然选择动作。

### Action-projected short trace

短 GPU 运行目录为
`experiments/shadow/p6_action_projected_short/20260809T123833Z`。本轮运行约
538.5 秒后受控停止，shutdown ACK 完整、无未决 transaction。GPU 运行时尚未包含
上述 closure-aware PREPARE 修复，因此 6,723 次 eligibility 均未入队 risk worker；
trace 随后暴露出 `WAIT_JOIN` parent 被 `descendant_closure` 误判为不可 shadow。

使用修复后代码重放该轮 45 个 PolicyInput snapshot。旧结论“有限 horizon 内没有
预测 HBM 越界”是错误的；`recourse credit=0` 不能等价为“没有 pressure”。修正后的
结果为：

- 44 个具有 prediction，38 evaluated、6 skipped；
- 生成 57 个 `PREPARE_HOST` 候选、5 个 distinct context，其中一个 context 重复
  38 次；41/57 候选预测到 HBM overflow，最大约 5.78 GB；
- 456 个 scenario 中，373 个 horizon 内无 pressure。使用当前 bounded neighboring-
  bucket transfer curve 重放后，65 个 shadow 晚于 pressure，16 个 pressure 不早于
  parent reentry，2 个满足
  `shadow <= pressure < reentry`、exclusive bytes 足够且 parent 是当前快照近似选出的
  reactive victim。这里的 victim 仅是 snapshot-consistent conservative approximation：它把
  当前 LRU/waiting 排序投影到未来 pressure 点，不代表未来 observed JointPlan 一定
  选择该 parent，也没有重放多 victim reclaim；
- shadow closure 为约 43.4 MiB--2.59 GiB，中位约 1.21 GiB；
  `exclusive/full-closure` 比例中位约 92.30%，最小约 77.75%；
- 修复 bounded neighboring-bucket extrapolation 后，shadow D2H 完成时间 P50 约 940 ms、范围约
  75--2,007 ms；旧 replay 的数百秒估计来自跨数量级的 direction-wide rate 外推；
- 57 个候选均不再受 closure blocker 误拒绝；
- 原始 replay 的 recourse D2H 来自 warm-start transfer curve，但 proactive
  interference 来自 snapshot 内旧 `unhidden_stall_per_byte`。两个有效 scenario 的
  D2H 分别约 1,919/975 ms，旧干扰却为 4,743/2,412 ms，比例约 2.471/2.474。因此旧
  结果只能解释为不一致证据下的保守下界，不能用于否定 PREPARE_HOST；
- replay 现会逐 scenario 输出 `transfer_duration_source`、transfer/interference
  service epoch、`interference_to_transfer_ratio`、最近 bucket 距离、样本数和 size
  coverage。旧证据显式归档于
  `experiments/analysis/p6_action_projected_short_recourse_legacy_interference_v4.jsonl`；
- 在没有真实 overlap 标定前，使用同一 D2H duration 的
  `stall_fraction * D2H_duration` 做敏感性分析。结果如下。

| 未隐藏干扰比例 | 正收益候选 | 最终 eligible | 结论 |
|---:|---:|---:|---|
| 0% | 2 | 2 | planner 可自然选择 PREPARE_HOST |
| 5% | 2 | 0 | 期望收益为正，但超过 10 ms CVaR budget |
| 10% | 2 | 0 | 同上 |
| 10.2% | 1 | 0 | 2.66 GB 候选越过 break-even |
| 14.0% | 1 | 0 | 仅 1.35 GB 候选仍为正收益 |
| 14.1% | 0 | 0 | 两个候选均无正期望收益 |
| 15% / 25% | 0 | 0 | 无正期望收益 |

由此得到两个不同门槛：约 10.2%/14.1% 是正期望收益的必要上限；在当前 10 ms
CVaR risk budget 下，两个动作要成为最终 eligible，未隐藏干扰还需分别低于约
0.52%/1.03%。不能把前一门槛误写成在线上线条件。敏感性 artifact 为
`experiments/analysis/p6_action_projected_short_recourse_stall_*.jsonl`。

因此，本轮确认了候选生成、pressure 观测和 scenario-level recourse 归因均已修复，
当前 trace 已证明存在 capacity pressure 和两个时序有效的 recourse scenario，但尚未
证明真实 overlap 干扰足够低，不能声称 predictive dispatch/ACK 闭环已由真实策略
可靠触发。下一步只针对 1.35 GB 和 2.66 GB 两个候选做 D2H overlap
characterization；若真实未隐藏干扰长期高于 10%--14%，整段 PREPARE_HOST 不再是
主要方向，应转向分块 shadow 或更早触发。若目标是保持当前 10 ms CVaR 门槛，则还需
检查是否能达到约 0.5%--1% 的未隐藏干扰。在标定完成前，在线系统保持 observed
JointPlan 回退。

该测量随后完成了方法审计，详见
`beliefkv_p6_d2h_overlap_characterization_2026-08-09_zh.md`。两笔 restore 事务正确性
有效，但旧 analyzer 未将 output progress 纳入 sequence matching、遗漏 transfer 边界
interval，也没有独立 NO_D2H control；micro 的 4/7 extents 还与真实候选的
22/106 extents 不匹配。因此旧 interference 数字和“旧曲线高估约 10 倍”的结论均已
撤回，不写入 warm-start artifact，也不触发 recourse replay。下一步先执行
fragmentation-matched、双 GPU cross-over、独立 control 和重复采样。
