# P6 D2H Overlap 机制测量与方法审计（2026-08-09）

## 修正后的结论

本轮两笔 D2H/H2D restore 事务的正确性结论有效，但 interference 性能结论无效。
原报告中的 `0% / 8.06% / 3.23% / 33.96%` 不进入正式结果，当前两条样本也不写入
warm-start transfer artifact，不用于重放 recourse value。

仍然可信的证据只有：

- 1,352,466,432 bytes、4 extents 的 D2H 在该低碎片 micro 中用时 106.46 ms；
- 2,659,221,504 bytes、7 extents 的 D2H 在该低碎片 micro 中用时 185.53 ms；
- 两笔等量 H2D 均完成，restore obligation 获得 32-token service grace；
- 两个运行的 14 项 restore micro-gate 检查全部通过，无遗留 command、transaction、
  lease、funding 或 obligation。

这只能证明少量大 extent 的 closure 可以达到约 13--14 GB/s，不能代表动态 agent
多轮 context 的高碎片 Radix closure。

## 方法问题

### 1. sequence matching 未生效

旧 analyzer 只读取 `sequence_tokens_before`。该字段在本轮 SGLang decode trace 中固定为
初始 prompt 8,041 tokens；真实 decode 进度位于 `output_tokens_before`。正确匹配量为：

```text
effective_sequence_tokens
  = sequence_tokens_before + output_tokens_before
```

原始 trace 中，1.35 GB D2H 覆盖 anchor output 684--689，2.66 GB 覆盖
684--694；旧 baseline 却延伸到 2,531/2,789。修复后，在 effective-sequence ±256
范围内，每个运行只有 1 个相同 batch signature 的 baseline，低于最少 4 个样本门槛。
因此旧 trace 无法通过事后重算补救。

运行时现已额外发布 `effective_sequence_tokens_before`；analyzer 对旧 trace 使用
`sequence_tokens_before + output_tokens_before` 回退，并默认将无独立 control 的结果标记为
`performance_evidence_eligible=false`。

### 2. physical layout 不匹配

离线 recourse action certificate 对应的物理布局是：

| 候选 | action bytes | 真实 certificate extent | 本轮 micro extent |
|---|---:|---:|---:|
| small | 约 1.35 GB | 22 | 4 |
| large | 约 2.66 GB | 106 | 7 |

历史自然 workload 中，一笔 2,656,468,992 bytes、52-page D2H 的 start-to-complete
时间为 1,710.29 ms；旧 replay 的约 1,919 ms 只高约 12%。因此原报告“旧曲线高估
9.2/10.3 倍”的说法已撤回。正确表述是：相同字节量在不同 extent/page count 下可能有
数量级不同的完成时间，迁移模型必须显式建模 fragmentation。

### 3. post-transfer baseline 不是严格反事实

D2H 后还发生 H2D restore、replacement admission 和 victim resume。anchor 再次成为
batch=1 时，其 output position 和系统状态均已改变。正式性能测量必须使用独立
`NO_D2H` control，而不是从同一 treatment 的后半段寻找 baseline。

### 4. 边界 interval 与 GPU confound

旧 analyzer 丢弃 transfer 起止处的两个边界 interval，却用完整 D2H duration 作分母，
只能得到偏低结果。修复后会纳入全部重叠 interval，并按
`overlap_ms / service_interval_ms` 加权，同时报告 transfer coverage。

此外，small 只在 GPU0、large 只在 GPU1 测量，size 与 GPU 身份完全混淆。正式实验
必须在两张卡上执行完整 cross-over。

## 正式实验协议

### 因子矩阵

每个 size 都在 GPU0/GPU1 上同时执行 treatment 与独立 NO_D2H control：

| size | prompt tokens | low-fragment chunk | matched-fragment chunk | 目标 extent |
|---|---:|---:|---:|---:|
| small | 约 13,792 | 4,096 | 约 640 | 22 |
| large | 约 27,085 | 4,096 | 256 | 106 |

chunk size 只是构造手段，运行是否合格最终以 safe-point physical certificate 和实际
transfer telemetry 的 page count 为准。当前预注册的 large-only 因果矩阵对每个
`fragmentation x GPU x treatment/control` 接受 3 次；只有 large case 通过后才扩展
small 矩阵及其重复数。运行顺序交错，避免 fragmentation 与时间、温度或 GPU 身份再次
绑定。

### Treatment

- victim output 上限 1,024，anchor output 上限 4,096；
- replacement 在 5 秒后进入 waiting；
- 64 MiB private-suffix 只控制 hook 触发，使 D2H 约发生在 anchor output 684；
- live bundle 必须满足目标 bytes、extent count 和 pinned Host 条件；
- 使用真实 JointPlan retraction/D2H/H2D/restore 闭环。

### NO_D2H control

- 使用相同 prompt、chunk size、GPU、模型和启动/warm-up 流程；
- victim 在约 684 output token 自然结束，anchor 随后在相同 output position 成为
  batch=1；
- 不提交 replacement，关闭 restore hook 和 BeliefKV reactive transfer；
- control baseline 窗口内若出现任何 native/explicit D2H 或 H2D，则该轮作废。

### 接受门槛

- actual bytes 与目标相差不超过 2%；page count 与 22/106 匹配；
- treatment D2H 窗口的 decode coverage 不低于 95%；
- treatment/control 均有至少 4 个 `batch=1 + anchor` sample，effective sequence
  半径默认不超过 32 tokens；
- control 窗口无 transfer，事务检查全部通过；
- 每个 size 在两个 GPU 上都有有效重复，按 run 聚类 bootstrap，不把 decode interval
  当作独立实验样本。

只有完成上述矩阵后，才能按
`GPU + model + direction + command kind + bytes + extent/page count + pinned + decode`
更新 transfer/interference 模型并重放 recourse value。

## 产物资格

有效机制目录：`experiments/micro/p6_d2h_overlap/20260809T151436Z/`。原 analyzer 输出已
重命名为 `d2h_overlap_analysis_legacy_invalid.json`，每个 arm 增加
`d2h_overlap_analysis_status.json` 记录失效原因。两份
`restore_micro_gate_validation.json` 仍可作为事务正确性证据。

## 2026-08-10 测量闭环与阶段结果

测量链路已经补齐以下约束：

- telemetry 记录实际完成的 `extent_count`、extent bytes min/p50/max、
  `small_extent_ratio`、`actual_bytes`、`pinned_host` 和 `command_kind`；
- analyzer 将目标 D2H 与所有非目标 transfer 状态做区间 join。横跨后续 H2D 的 decode
  interval 会被剔除，control 中任意 transfer 都使 pair 失效；
- `transfer_dispatched` 的 selected bytes、page count 和 D2H action count 必须与 completion
  telemetry 一致；
- aggregate 以 paired run 为抽样单位，按 GPU、bytes class 和 fragmentation class 分组，
  treatment/control 计数分开，不能用 decode interval 扩大样本数；
- 完整矩阵显式期待 GPU0/GPU1 的 high/low 四组，缺组时整体 gate 必须失败。

布局 pilot 位于：

`experiments/micro/p6_d2h_layout_pilot/20260810T_pilot/`

结果如下：

| GPU | size | actual bytes | extents | D2H ms | correctness |
|---:|---|---:|---:|---:|---|
| 0 | small | 1,352,466,432 | 22 | 148.70 | 通过 |
| 0 | large | 2,659,221,504 | 106 | 512.36 | 通过 |
| 1 | small | 1,352,466,432 | 22 | 139.08 | 通过 |
| 1 | large | 2,659,221,504 | 106 | 846.80 | 通过 |

GPU0 的独立 control 配对也通过 pilot gate：small/large coverage 分别为 98.39%/97.70%，
effective-sequence 中心差为 15/12 tokens，control transfer 污染均为 0。

large-only 正式矩阵位于：

`experiments/micro/p6_d2h_fragmentation_matrix/20260810T012600Z/`

GPU0 的 12 个 arm 已全部完成：

| fragmentation | extents | D2H completion ms（3 runs） | mean ms | interference mean |
|---|---:|---|---:|---:|
| low | 7 | 196.40 / 188.05 / 172.63 | 185.69 | 36.67% |
| high | 106 | 781.11 / 773.42 / 740.98 | 765.17 | 71.80% |

相同字节量下，GPU0 high-fragment completion 平均为 low-fragment 的约 4.12 倍，且三次
重复方向一致。该结果满足单 GPU 信号门槛，但不满足预注册的双 GPU crossover 门槛。

GPU1 crossover 在首个 arm 加载模型前被外部进程重新占用约 32.7 GiB，SGLang 初始化
OOM；没有 workload 被提交，该启动失败不计入重复数。2026-08-10 再次观察时，该外部
任务仍占用约 32.3 GiB，因此没有重试必然 OOM 的 arm。当前 aggregate 正确标记 GPU1
high/low 为 `expected_group_missing`，`all_groups_eligible=false`。在 GPU1 稳定空闲并完成
剩余 12 个 arm 前，不更新 warm-start transfer artifact，也不重放 P6 recourse。
