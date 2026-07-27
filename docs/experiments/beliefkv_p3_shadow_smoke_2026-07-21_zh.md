# BeliefKV P3 统一快照与 Reference Replay Smoke

日期：2026-07-21

状态：协议与 replay 链路通过；workload correctness 和 P4 overhead gate 未通过

> 归档说明（2026-07-22）：这是单 workflow 的旧 B0-B4 机制 smoke，不是可执行的当前
> baseline suite。代码已精简为 B0-only，数据保留在 correctness-only archive 中。

## 1. 实验目的

本轮只验证以下链路，不评价 BeliefKV 性能收益：

```text
SGLang scheduler safe point
  -> RCCG / queue / disjoint physical extent / allocator snapshot
  -> gzip replay trace
  -> hindsight metadata enrichment
  -> B0-B4 same-data-plane reference replay
```

本轮运行时仍使用同步 JSON/gzip writer。实验结束后已改成后台单消费者 writer，并加入
`PageOwnershipIndex.revision` 与 bundle variant cache；因此本轮 overhead 是修复前负基线，
不能代表当前代码。

## 2. 配置与有效范围

- 模型：Qwen3-Coder-30B-A3B-Instruct-FP8；
- SGLang：固定 0.5.2rc1，GPU0，TP=1；
- KV pool：163,840 tokens，`mem-fraction-static=0.952`；
- HiCache：ratio 2、write-back；
- workload：planned Deep Agents，SWE-bench Verified `sympy__sympy-12489`；
- 1 个 root workflow、2 个 FRESH child、179 个 LLM request、58 个 tool call；
- 墙钟 464.59 秒，模型输出上限 256 tokens。

系统路径正常完成 179/179 个 request，runtime control delivery 无失败，workflow
`outcome=completed`。但是 agent 声称修改文件，实际 workspace 没有 patch，也没有成功测试命令，
故 `measurement_valid_workflows=0`。该 run 不能作为正常完成 workload、算法收益或 JCT 结果。

## 3. 快照正确性与存储

- 记录 898 个 `PolicyInput`，0 个 `policy_snapshot_failed`；
- 物理 bundle 数 P50=149、P95=320、最大=343；
- gzip 文件 11,927,153 bytes，平均约 13.28 KB/snapshot；
- snapshot、RCCG、consumer、request identity、physical extent 和 allocator 总账可以 round-trip；
- FRESH child 没有从因果 parent 继承 physical page；
- allocator 中未进入 Radix mirror 的 request-private KV 以不可迁移的
  `protected_untracked` bundle 封闭账本。

Hindsight enrichment 对 180 个 runnable request 的真实 finish/output 覆盖率为 100%。全体
context 的 next-request coverage 为 72.30%；缺失项主要是已 terminal、仍残留普通 prefix cache
的 context，不应被解释为 predictor 缺失。后续报告需单独统计 live/runnable coverage。

## 4. B0-B4 Replay

五个策略在完全相同的 898 个 core snapshot 上完成 replay：

| Baseline | Admission | 主要 residency action | Unsupported |
|---|---:|---:|---:|
| B0 Reactive | admit 175 / defer 5 | commit CPU 139 | 0 |
| B1 distance oracle | admit 175 / defer 5 | commit CPU 139 | 0 |
| B2 space-time oracle | admit 175 / defer 5 | keep 147,875 / commit CPU 57 | 0 |
| B3 phase | admit 175 / pause 5 | drop 6,536 | 898 个已声明的 global queue migration 缺口 |
| B4 congestion | admit 175 / pause 5 | 无 KV transfer | 0 |

B0 与 B1 动作计数完全相同。单 workflow trace 不含足够 invocation-distance 竞争，不能证明 B1
或 BeliefKV 优势。固定 physical trace 只支持 decision replay；所有结果均明确
`jct_reported=false`，没有伪造反事实 JCT。

## 5. 同步 Recorder 负结果

修复前同步 recorder 把 PolicyInput 构造、JSON、gzip 和 flush 全部放在 scheduler safe point：

| 指标 | 数值 |
|---|---:|
| snapshot P50 | 11.690 ms |
| snapshot P95 | 26.021 ms |
| snapshot P99 | 38.077 ms |
| snapshot max | 432.021 ms |
| scheduler-step P99 | 37.172 ms |
| snapshot/scheduler P99 | 1.024 |

该实现明确不满足 P4 的 `<1 ms` 或 `<5% scheduler step` gate。

实验后已完成两项修复：

1. scheduler 线程只入队不可变 `PolicyInput`，JSON/gzip/flush 由后台单消费者执行；
2. `PageOwnershipIndex` 提供单调 revision；未变化的 physical state、lease variant 和 bundle
   dataclass 直接复用。

当前 CPU 微基准：

| 场景 | Extents | P50 | P99 |
|---|---:|---:|---:|
| 无 physical/RCCG 变化 | 256 | 0.237 ms | 0.283 ms |
| READY/WAIT_TOOL lease 切换，variant 已 warm | 114 | 0.827 ms | 0.891 ms |
| 单 extent engine lock 改变 | 114 | 1.811 ms | 1.929 ms |

最后一项仍高于 1 ms，且异步版本尚未做真实 GPU 复验。因此 P4 overhead gate 仍为 pending。

## 6. 产物

- [workload summary](../../experiments/archive/20260722/p3_correctness_only/raw/deepagents_swebench-20260721T074957Z/planned-1-p3-shadow-smoke/summary.json)
- [raw policy snapshots](../../experiments/archive/20260722/p3_correctness_only/raw/deepagents_swebench-20260721T074957Z/server-p3-shadow-smoke/policy_snapshots.jsonl.gz)
- [hindsight summary](../../experiments/archive/20260722/p3_correctness_only/processed/p3_20260721T074957Z/hindsight_enrichment_summary.json)
- [reference replay summary](../../experiments/archive/20260722/p3_correctness_only/processed/p3_20260721T074957Z/reference_replay/summary.json)
- [snapshot overhead](../../experiments/archive/20260722/p3_correctness_only/processed/p3_20260721T074957Z/policy_snapshot_overhead_legacy_sync.json)

## 7. 下一步门槛

1. 用异步 writer 做短协议复验，确认 `written_snapshot_count == snapshot_count` 且无 writer backlog；
2. 对真实 physical-only 变化继续增量化，或证明其 P99 小于 scheduler step 的 5%；
3. 修复/替换无法产生真实 patch 的 workload，再运行 blocking、cyclic 和 mixed workload；
4. 只有完整 queue/service/allocator resimulator 建成后，才计算 O0-O3 JCT 与 joint synergy gap。
