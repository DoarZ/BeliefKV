# BeliefKV P5G Native Ownership Snapshot W4 Characterization

## 实验定位

本轮只回答一个问题：P5G 在 scheduler safe point 从 SGLang native queue、request slot、Radix
lock、native load 和 explicit transfer 重建真实 ownership 的开销是否可控。它不是 workflow JCT
性能实验，也不是 P5G clean-completion gate。

- 模型：Qwen3-Coder-30B-A3B-Instruct-FP8，单卡 RTX 6000 Ada；
- GPU KV pool：163,840 tokens，约 15 GiB；Host HiCache：96 GiB；
- workload：固定 4 个 SWE-bench Verified SymPy workflow，3 mixed、1 cyclic；
- P5：online observed JointPlan、admission scheduling 和 running retraction 开启，预测关闭；
- 有效 raw：`experiments/raw/p5g_ownership_snapshot_w4/20260731T111458Z`；
- 采集到 105 次 ownership rebuild 后受控停止，最终 shutdown ACK 完整。

## 首轮故障与修复

首轮 raw 为 `experiments/raw/p5g_ownership_snapshot_w4/20260731T110351Z`。首次 running
retraction 完成 D2H 后，waiting request 的 `last_node` 已成为 Host-only Radix leaf，
`node.value=None`。旧实现仍在 H2D 前调用 SGLang `inc_lock_ref()`，后者执行
`len(node.value)` 并使 scheduler 退出。

修复后的事务边界为：

1. H2D 前保留 BeliefKV capacity reservation，但不对尚未在 GPU 物化的 leaf 建立 GPU pin；
2. HiCache `load_back()` 负责保护 materialized ancestor 和 native loading path；
3. H2D ACK 后确认整条 GPU Radix path 已物化，再建立跨 admission 的 restore lease pin；
4. 若 ACK 后路径仍未物化，则回滚 lease 并等待真实 Radix 变化，不能捕获异常后盲目重试。

修复后 CPU suite 为 `486 passed, 10 skipped, 3 subtests passed`。GPU 复验中出现 35 次
`restore_lease_prefix_pin_deferred`，随后均有 35 次 pin、35 次 unpin 和 35 次 admission commit；
未再出现 `node.value=None` 崩溃。

## Ownership 重建开销

| 指标 | 结果 |
| --- | ---: |
| rebuild 调用数 | 105 |
| P50 / P95 / P99 | 0.052 / 0.156 / 0.523 ms |
| 最大值 | 5.454 ms |
| 每次最多扫描 native queue records | 12 |
| 每次最多扫描 request metadata records | 12 |
| 平均 queue / metadata records | 11.0 / 11.2 |
| rebuild / scheduler step | 0.000578，约每 1,729 step 一次 |
| scheduler step P99 | 35.170 ms |

这些时间只包含内存中的 ownership 重建和排序，不包含逐次 JSON 序列化或同步审计 I/O。P99
约为 scheduler-step P99 的 1.49%，但两个分位数来自不同样本集合，只能作为量级参照，不能当作
逐 step 精确占比。

## 物理路径覆盖

受控停止前共观测到：

- 14 个 running retraction transaction 完成；
- 38 个 restore obligation 创建，并在 shutdown 时全部进入 terminal；
- 89 条 transfer command 收到 ACK；
- 35 次 restore admission commit 和 service grace；
- 最终 0 inflight command、0 active obligation、0 active lease；
- `shutdown_state=acknowledged`，所有在线动作均带 `source_joint_plan_id`。

workflow 在达到 characterization 样本门槛后被主动取消，`workloads.incomplete` 被保留。因此本轮
不生成 KV 时间线，不报告 completion rate/JCT，也不能替代固定 w4 clean-completion gate。

## 判定

在当前 w4、最多 12 个 native request 的规模上，按 restore 事件触发的 ownership rebuild 开销
可控，不是当前 P5 的主要控制面瓶颈。该结论不能直接外推到 w24/w32：现实现仍会为同一 restore
obligation 重复扫描全局 request metadata，复杂度随 obligation 数和可见 request 数相乘。

105 个样本不足以把 `P99 < 0.5 ms` 作为硬门槛；0.523 ms 与 0.5 ms 的差异没有统计意义，但
5.454 ms 离群点仍需由后续分段计时解释。后续实现已改为惰性
`SafePointPhysicalSnapshot(epoch)`：每个需要物理状态的 capture epoch 至多构建一次，并在 commit
前重验 context read-set；它不会按 scheduler tick 无条件构建。N=8--128 的 CPU 微基准见
`beliefkv_p5g_safe_point_snapshot_cpu_2026-07-31_zh.md`。完整 P5G clean-completion gate 仍待执行。
