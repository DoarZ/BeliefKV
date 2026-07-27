# BeliefKV P1.5：H2D Retry Guard 实现与验证

日期：2026-07-19  
状态：正确性修复、冻结回放和真实机制复验通过；配对 JCT 性能门槛尚未闭合

## 1. 问题锚点

P1 短跑中共有 292 个 zero-byte H2D reject，其中 268 个来自同一 context：

```text
deepagents-context:fdc67d3fd0c9a938
reactive-287 ... reactive-572
跨度 7,176.690 ms
```

失败后 context、CPU_ONLY page 和物理可分配状态没有产生足以解除 blocker 的变化，但旧
planner 在每个 scheduler tick 重新提交同一 prefetch。P1.5 将该问题作为 reactive baseline
缺陷修复，不将其计入后续联合策略的研究收益。

## 2. 实现

- `runtime/protocol.py`：增加 machine-readable `TransferBlockerCode` 和
  `TransferBlocker`，local resolve 与 backend ACK 使用同一协议。
- `policy/transfer_guard.py`：以
  `(context, epoch, command kind, closure fingerprint)` 维护 attempt ledger。
- `runtime/radix_arbiter.py`：输出 closure、lock、loading、active owner、pin、generation
  等本地 blocker。
- `runtime/sglang_v052rc1.py`：将 HiCache ancestor/device/host/lock/loading/extent
  失败映射为 typed blocker，不要求 planner 解析 reason 文本。
- `control/controller.py`：backend ACK 形成下一轮负反馈；被抑制的 context 不阻塞其他
  candidate；cache reset/context epoch change 清理 ledger。
- runtime audit：记录 `transfer_attempt_blocked`、`transfer_retry_suppressed`、
  `transfer_retry_rekeyed`、`transfer_retry_released` 和最终 summary。

已知 closure、lock、loading、inflight 和 generation blocker 只能通过 closure fingerprint
变化解除。device/host capacity 还要求可用空间相对失败快照真实增加并满足 required bytes。
只有 unknown backend 使用 10 ms 到 1,000 ms 的指数退避；同一快照连续失败 8 次后打开
circuit breaker，等待物理快照变化。

## 3. 冻结回放

fixture：`tests/fixtures/h2d_retry_storm_p1_short.json`

回放将原实验的 268 次相同 H2D reject 建模为第一次真实 backend reject 加 267 次后续
scheduler decision。结果：

```text
实际提交 command       1
实际 ACK               1
被 guard 抑制          267
同 fingerprint 重提交  0
```

同时验证：

- local ancestor-closure reject 与 backend reject 共用 ledger；
- closure fingerprint 改变后，下一个 controller event 恢复资格；
- allocator free 未增加时禁止 capacity retry，增加后立即恢复；
- blocked context 不造成其他 prefetch 的 head-of-line blocking；
- cache reset 不产生永久误屏蔽；
- unknown failure 正确指数退避并打开 circuit breaker。

全量测试结果：`172 passed, 4 skipped`。跳过项为环境相关测试；测试过程中未启动模型服务或
GPU 实验。

## 4. 真实 HiCache 压力复验

实验路径：

```text
raw server   experiments/archive/20260727/superseded_raw/deepagents_swebench/20260718T174946Z/server-p1_5-retry-guard
raw workload experiments/archive/20260727/superseded_raw/deepagents_swebench/20260718T174946Z/planned-4-p1_5-retry-guard
raw workload experiments/archive/20260727/superseded_raw/deepagents_swebench/20260718T174946Z/planned-4b-p1_5-retry-guard
validation   experiments/processed/p1_5_20260718T174946Z/transfer_validation.json
timeline     experiments/processed/p1_5_20260718T174946Z/kv_transfer_timeline.html
```

配置为单 GPU、Qwen3-Coder-30B-A3B-Instruct-FP8、SGLang `0.5.2rc1`、
`max_total_tokens=163840`、`mem_fraction_static=0.952`、HiCache ratio 2，关闭预测和
shadow，开启 reactive prefetch。两组各 4 个冻结 SWE-bench/SymPy workflow 使用与 P1
相同的 instance 集合。为达到 guard 样本目标，运行在 workload 仍继续生成模型轮次时被主动
截断；服务在 `/get_load=0` 后正常关闭。

### 4.1 Retry-storm 正确性

| 指标 | P1 旧运行 | P1.5 | 结论 |
| --- | ---: | ---: | --- |
| 原始同 context zero-byte H2D 序列 | 268 | 0 个未解除重试 | 通过，下降 100% |
| 所有 H2D no-DMA reject | 292 | 43 | 下降 85.3%，但运行长度不同 |
| 同 physical fingerprint 最大 submit | 不受控 | 1 | 通过 |
| blocker 未解除时 retry | 不受控 | 0 | 通过 |
| typed blocker / unknown | 不可用 | 1,183 / 0 | 通过 |
| guard suppression / release | 不可用 | 22,484 / 1,182 | 事件门控生效 |
| release latency P50 / P95 | 不可用 | 85.06 / 2,180.23 ms | characterization |

1,183 个 blocked attempt 的 blocker 计数为 `node_locked=1112`、
`descendant_closure=91`、`ancestor_closure=68`、`device_capacity=68` 和
`engine_busy=6`。一次 attempt 可以同时包含多个 blocker，因此计数之和可以超过 attempt 数。
`retry_without_release_count=0`，这是判断 tick-driven storm 是否消失的主指标。相同 closure
hash 在一次合法 release 后再次出现不属于 storm。

关闭时仍有 1 个 active attempt：`sympy-13878` root 的 275,546,112-byte H2D capacity
blocker。它发生在 workload 被主动中断的末尾，不能据此判定存在永久误屏蔽；正式长跑需要在
自然 terminal event 后再次检查 active count 为 0。

### 4.2 迁移、压力和开销

```text
峰值 KV HBM                         15,414,067,200 B（15 GiB pool 的 95.70%）
峰值 Host KV                         7,691,698,176 B
D2H / H2D 实际字节                  13.52 GB / 6.92 GB
physical D2H / H2D                   300 / 75
command dispatch / ACK               966 / 966
controller telemetry/full tick P99   1.314%（< 5%，通过）
```

command ordering、时间戳、ACK byte bound 和 PageOwnershipIndex mirror 检查均通过。新运行
在 11,204 个 resource snapshot 中有 60.65% 的样本处于 90% 以上 KV pressure，能够触发
真实 HBM/Host 迁移，而不是低压功能测试。

### 4.3 Admission 与 JCT 限制

| admission wait | P1 旧运行（n=352） | P1.5（n=599） |
| --- | ---: | ---: |
| P50 | 11.92 ms | 10.63 ms |
| P90 | 721.07 ms | 675.54 ms |
| P95 | 7,813.47 ms | 7,608.90 ms |
| P99 | 34,768.25 ms | 54,237.68 ms |
| mean | 1,475.14 ms | 2,340.37 ms |
| max | 68,823.89 ms | 153,231.63 ms |

P50/P90/P95 未回归，但 P99、均值和最大值变差。P1.5 运行持续 804.8 秒，旧运行持续
344.2 秒，模型生成的后续 action 路径也不同；两者不是确定性配对执行。旧运行没有完整的
workflow terminal 样本，新运行也只有少量 terminal event，因此不能计算公平的 workflow
JCT 或把尾部变化归因于 retry guard。

## 5. 实跑后追加的正确性修复

实跑包含 31 个 `PARTIAL` H2D ACK。原 P1.5 版本在 partial 后把整个 resolved bundle 记为
capacity debt，而其中一部分页已经成功恢复。这会高估下一次所需 HBM，并产生 false
suppression。最终代码改为只将失败页字节写入 guard；capacity 解除基线也不再使用提交前的
free HBM，而是在 ACK 后等待下一次 authoritative allocator snapshot。该 snapshot 只建立
基线，不触发 retry；后续 free bytes 必须相对基线真实增加且达到失败页需求才解除 blocker。
对应测试覆盖 partial ACK、post-ACK anchor、无变化抑制和空间增加后释放。

该修复发生在上述真实运行之后，因此上述 trace 对 tick-driven retry 的结论仍有效，但不能用来
验证最终 partial-debt 逻辑。下一次正式配对运行必须使用当前代码。

## 6. 严格结论

P1.5 已满足机制退出条件：冻结 fixture 中 268 次风暴降为 1 次 backend attempt，真实压力下
同一快照最多提交 1 次、blocker 未解除时 0 retry、unknown blocker 为 0，且 controller
开销低于 5%。因此 retry storm 不应再作为后续策略的收益来源。

“admission liveness/JCT 不退化”尚未被证明。冻结 storm-free reactive baseline 前仍需一次
自然完成、固定随机性、相同 workflow/action trace 的配对 A/B；至少报告完成率、workflow
JCT、admission P99、H2D false suppression 和 terminal 后 active blocker。P2 可以开始做
physical preview 的 characterization，但论文性能结论必须等待该配对 gate。
