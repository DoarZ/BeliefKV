# BeliefKV P1：HiCache 真实迁移、服务曲线与控制开销验证

日期：2026-07-18  
结论：P1 预定义工程门槛通过；该实验不构成 workflow JCT 或策略收益结论

## 1. 验证目标与固定环境

本阶段只验证 HiCache 数据面正确性、迁移可观测性、callback service curve 和控制面开销，
不启用 predictor、shadow、joint planner 或 Reveal-and-Commit。

```text
GPU                 NVIDIA RTX 6000 Ada，CUDA_VISIBLE_DEVICES=0，TP=1
模型                Qwen3-Coder-30B-A3B-Instruct-FP8
SGLang              0.5.2rc1 @ 18f91eb639084825717c0e3c3c7273492812ab71
KV pool             163,840 tokens，15 GiB，mem_fraction_static=0.952
HiCache             ratio=2，write_back，layer_first，单 inflight node extent
workload            冻结 SWE-bench Verified SymPy manifest，planned Deep Agents
策略                predictor=false，shadow=false，prefetch=true
```

两个有效 run 使用相同模型、KV pool 和控制配置。长跑用于迁移完整性、occupancy 和服务曲线；
短跑只补测 controller timing。两个 workload 都因研究者设定的墙钟停止，不得用于 workflow
成功率、JCT 或最终 patch correctness。

## 2. 首次运行暴露并修复的正确性问题

首次 instrumented run 位于：

```text
experiments/archive/20260727/superseded_raw/deepagents_swebench/20260718T150352Z/server-p1-telemetry
experiments/archive/20260727/superseded_raw/deepagents_swebench/20260718T150352Z/planned-8-p1-telemetry
```

运行约 178 个 request、59 个迁移后，Radix split 在 HiCache DMA in-flight 期间改变了 node
extent，旧 mirror generation 随后被 invalidation，触发：

```text
PageIndexError: cannot mutate in-flight page extent
```

修复保持 ACK 正确性边界不变：提交 DMA 时保存 extent fingerprint；物理完成后重新校验；若
extent 已变化，DMA 字节仍进入性能 telemetry，但拒绝旧 extent 的 residency commit；存在
in-flight command 时 `sync_tree()` 延迟完整拓扑重建，ACK 后再按 SGLang 真相源重建。

后续两次真实运行分别观察到 35 次和 7 次 extent mutation。它们均被安全拒绝，没有再产生
`PageIndexError`、stale residency commit 或 scheduler crash。这说明竞态没有消失，但失败
语义已经从“破坏 mirror”变成“可计量的 wasted DMA”。

## 3. 长跑：迁移完整性与服务曲线

路径：

```text
raw server    experiments/archive/20260727/superseded_raw/deepagents_swebench/20260718T151540Z/server-p1-telemetry
raw workload  experiments/archive/20260727/superseded_raw/deepagents_swebench/20260718T151540Z/planned-8-p1-telemetry
validation    experiments/processed/p1_20260718T151540Z/transfer_validation.json
timeline      experiments/processed/p1_20260718T151540Z/kv_transfer_timeline.html
```

### 3.1 命令与物理状态

```text
transfer dispatch / ACK                 6,415 / 6,415
预期有 DMA 的命令 / telemetry           5,693 / 5,693
dispatch < ACK < telemetry 违规         0
时间戳或 actual > closure 违规          0
watchdog / scheduler exception          0 / 0
resource snapshot                       27,101
Host allocator / CPU mirror 不一致      0
HBM mirror 超过 allocator               0
峰值 KV HBM / Host                      15.00 GiB / 16.66 GiB
```

HBM allocator 与 GPU page mirror 在 active decode 时不要求相等：allocator 还包含尚未封入
Radix node 的 request-private KV。实测 `allocator - mirror` P50 为 29.63 MiB、P99 为
201.56 MiB、最大 528.66 MiB，且从未为负；这符合 mirror 是物理 allocator 子集的边界。

### 3.2 必须区分“尝试”和“真实 DMA”

5,693 条 telemetry 不是 5,693 次物理迁移：

```text
真实 non-zero-byte DMA                  917
零字节拒绝                              4,776
D2H                                     744 次，32.82 GiB
H2D                                     173 次，10.64 GiB
completed / partial / rejected          169 / 713 / 4,811
非 completed 但已发生的 DMA             20.23 GiB
extent mutation wasted H2D              35 次，91.88 MiB
```

4,775 次主要 H2D 拒绝来自 ancestor closure 与 device allocation 不可满足；671 次 D2H
partial 主要来自 node locked/loading。如此高的无效尝试和 partial DMA 不是 P1 的成功指标，
而是 P2 必须用 bundle preview/blocker set 解释和减少的直接证据。

短跑进一步确认了 retry storm：292 个 zero-byte H2D 中，268 个来自
`deepagents-context:fdc67d3fd0c9a938`，在 7,176.690 ms 内从 `reactive-287` 重复到
`reactive-572`，且 blocker reason 完全相同。这个问题先在 P1.5 用 typed blocker 和
event-gated retry ledger 修复；P2 再用 bundle preview 降低首次不可行选择。两类收益必须分开
归因。

时间线因此只在 D2H/H2D 主轨绘制 `actual_bytes > 0` 的操作，零字节拒绝单独显示为
`No DMA` ticks，表格仍保留全部 5,693 条观察。

### 3.3 callback service curve

原实现发现并修复了两个问题：

1. partial/rejected 的非零字节曾被误用于带宽拟合；现在只有 `COMPLETED` 更新服务样本，
   outcome 使用独立 deque，拒绝风暴不会挤掉完成样本；
2. `setup + bytes / rate` 对小 operation 缺少 scheduler/HiCache completion observation 的固定
   callback 下限；现在取 size-aware 预测与 direction callback P90 floor 的最大值。

按 completion sequence 做固定的 80/20 时间顺序 holdout：

```text
completed operation                    169
train / holdout                        135 / 34
under-estimated                        1 / 34 = 2.94%
D2H / H2D under-estimation             5.56% / 0%
actual / estimate ratio P90            0.679
```

点估计低于预定义的 10% 门槛，但 Wilson 95% 区间为 `[0.52%, 14.92%]`，上界仍超过
10%。因此该结果足以作为 P1 工程 gate，不足以单独支持论文中的统计保证。

## 4. 短跑：controller telemetry 开销

路径：

```text
raw server    experiments/archive/20260727/superseded_raw/deepagents_swebench/20260718T160549Z/server-p1-overhead
raw workload  experiments/archive/20260727/superseded_raw/deepagents_swebench/20260718T160549Z/planned-4-p1-overhead
raw workload  experiments/archive/20260727/superseded_raw/deepagents_swebench/20260718T160549Z/planned-4b-p1-overhead
validation    experiments/processed/p1_20260718T160549Z/transfer_validation.json
timeline      experiments/processed/p1_20260718T160549Z/kv_transfer_timeline.html
```

先运行 4 个 workflow 时 KV 压力只有约 31%，没有产生迁移；加入 4 个不同 instance 后峰值
KV HBM 达到 14.97 GiB，产生 516 条 telemetry、224 个 non-zero-byte DMA，共迁移约
9.95 GiB。两个 workload 在达到计时样本目标后主动终止，`/get_load=0` 后正常关闭服务。

运行期只向固定长度 deque 追加 `(full_hook_ms, telemetry_ms, count)`，关闭时一次性计算分位
数，避免测量本身每 50 ms 排序。516 个含 telemetry 的同一 hook tick 上：

```text
完整 BeliefKV scheduler hook P99        28.742 ms
telemetry drain/model/log P99            0.365 ms
同 tick telemetry/full 比例 P99          2.012%
门槛                                      < 5%，通过
```

短跑 service-curve holdout 为 0/11 低估，但样本太少，Wilson 95% 上界为 25.88%；它只用于
检查修复在不同运行相位没有立即退化，不增加统计结论。

## 5. P1 Gate 与残余限制

| 退出条件 | 结果 | 结论 |
| --- | ---: | --- |
| telemetry 非常量、时间戳单调 | 5,693 条，0 违规 | 通过 |
| ACK 与 physical mirror 一致 | ACK 100%，Host mirror 0 偏差 | 通过 |
| holdout 低估率 < 10% | 2.94%，独立短跑 0% | 点估计通过 |
| telemetry 开销 < tick 5% | P99 2.012% | 通过 |

残余限制：

- 固定 `0.5.2rc1` 后端没有 `compute_wait_ms`、PCIe utilization、copy-engine utilization 或
  layer-ready event，因此当前模型预测 callback，不是 actual unhidden stall；
- 两个有效 run 都被主动截断，不能用于 JCT、成功率或 policy speedup；
- holdout 的置信区间仍宽，正式 characterization 需要更多独立 run 和更多 completed
  operations；
- 4,776 次 no-DMA reject、20.23 GiB 非 completed DMA 和 closure amplification 表明，
  当前 reactive per-context planner 的主要问题仍是逻辑选择晚于物理 closure/lock 约束。

按修订后的实施方案，下一阶段先进入 P1.5：消除 tick-driven retry storm 并冻结更强的
reactive baseline；随后进入 P2：physical causal lease、shared-owner bundle、closure
preview 和 blocker set。P2 首要目标不是宣称性能收益，而是让 planner 在 dispatch 前解释并
避免当前 trace 中的大量物理不可行动作。
