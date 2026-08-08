# BeliefKV P5D 修复版 correctness ladder: w4 GPU 负结果

日期：2026-07-28

状态：`w4` 未通过，按 correctness ladder 规则停止，未运行 `w8/w12/w24`

## 1. 实验配置

- 模型：`Qwen3-Coder-30B-A3B-Instruct-FP8`
- SGLang：`0.5.2rc1`，TP=1，GPU 0
- GPU KV pool：163,840 token，15 GiB
- Host HiCache：96 GB，`write_back`
- `MEM_FRACTION_STATIC=0.952`
- online JointPlan、observed admission、running retraction 均开启
- workflow active window：12
- workload：4 个 SWE-bench Verified SymPy workflow，并发 4，单批到达
- mixed workflow 在运行时选择 2--4 个 subagent；cyclic workflow 不创建 subagent
- 单请求 execution timeout：900 秒
- root activation wall-clock deadline：1,800 秒

原始目录：

```text
experiments/raw/p5d_repaired_ladder/w4/20260728T053031Z/
```

## 2. Workload 结果

本轮共执行：

- 361 次 LLM request；
- 379 次真实工具调用；
- 11 个动态 subagent，均为 multi-turn；
- 4 个 workflow 均在约 1,803.6--1,803.7 秒后以
  `PartialAgentRunError: APITimeoutError` 结束；
- 0/4 workflow 正常完成，0 个 clean JCT 样本；
- 两个 mixed workflow 完成全部 child RETURN/JOIN；另一个 mixed workflow 有一个 child
  未 RETURN；cyclic workflow 没有语义完成。

所有 workflow 的结束时间与 1,800 秒 activation deadline 对齐，因此该轮不能用于性能比较。
客户端在 deadline 到达后向服务器发送了 4 次 `/abort_request`，服务器侧 4 个 terminal abort
均完成。

## 3. 迁移数据面

离线 `validate-transfer-telemetry` 通过迁移命令正确性检查：

- 30 次 dispatch 对应 30 次 ACK；
- 无 missing/orphan/duplicate ACK；
- 无 telemetry 顺序或字节上界违规；
- 无 Host page-index mismatch；
- HBM mirror 始终是 allocator 的子集；
- 两笔显式 retraction bundle 均完整完成，实际释放 4,699,717,632 bytes；
- retry guard 没有重复提交相同失败动作。

完整 native HiCache telemetry：

- D2H：666 次，24,631,934,976 bytes；
- H2D：4 次，763,330,560 bytes；
- 峰值 HBM：16,106,127,360 bytes；
- 峰值 Host KV：24,631,934,976 bytes；
- 物理迁移 partial/failed：0。

因此本轮失败不能归因于 DMA callback、ACK、allocator ownership 或 Host KV 生命周期错误。

## 4. Running retraction liveness 失败

两笔 running retraction 都完成了 barrier、D2H residency transaction 和 ACK，但被 requeue 的
请求没有再次获得 GPU service：

| Request | retraction | retraction 前 service | 后续结果 |
|---|---:|---:|---|
| `beliefkv:019fa737-bfbc-7830-bc9d-135319204d93` | `retraction-1` | 50 个 sample，跨度 10.00 s | requeue 后无 service，900.14 s execution timeout |
| `beliefkv:019fa737-e33c-7332-a2ff-1e5b63038e20` | `retraction-2` | 49 个 sample，跨度 8.63 s | requeue 后无 service，900.03 s execution timeout |

6,080 个 admission ticket epoch 中有 5,856 个 epoch 因 `wait_restore` 跳过请求。高压时物理
HBM 已接近 15 GiB 上限，但 trace 同时显示约 9 GiB KV 可迁移；emergency JointPlan 仍只输出
`defer`，没有联合生成“迁移 victim + H2D restore + admission”动作。系统只能等待其他 running
request 自然结束释放空间。

这证明当前 P5D 的 running retraction 事务在物理上正确，但 restore dependency 没有获得有界
liveness。它会将运行请求转化为长期持有 execution timeout debt 的 `wait_restore` 请求。

## 5. 控制面开销和新鲜度

- `joint_plan_stale`：4,337 次；
- `online_joint_plan_published`：6,578 次；
- `online_joint_physical_commit_budget_exceeded`：16,292 次；
- 物理提交超预算事件中 4,975 次超过 10 ms、2,164 次超过 100 ms、570 次超过 500 ms；
- 最大记录值为 2,379.03 ms，远高于配置的 1 ms safe-point budget；
- runtime audit 为 91 MiB，policy snapshots 压缩后为 12 MiB。

异步 semantic planner 虽然能持续发布计划，但 safe-point physical commit 仍包含无法被 1 ms
预算约束的同步工作。计划发布速度、新鲜度和物理提交成本尚未达到在线唯一策略源要求。

## 6. Shutdown gate

停止服务后 GPU 0 已完全释放，且没有残留 SGLang/workload 进程。但
`latest_runtime_summary.json` 最终停在：

```text
shutdown_state = preparing
final = false
shutdown_summary_complete = false
```

审计中没有 `runtime_shutdown` 终态事件。当前 Ctrl-C 只能触发 scheduler-local prepare，尚未完成
frontend/tokenizer/scheduler 的显式 `SHUTDOWN_PREPARE -> SHUTDOWN_ACK` 协议。

## 7. 结论与下一门槛

本轮验证了迁移数据面的 correctness，但否定了当前 P5D 在线调度的 liveness。不能继续扩大到
`w8/w12/w24`，也不能用该结果比较性能。

下一次 GPU 复验前必须完成：

1. 为每个 retracted request 建立强制 `RestoreIntent`，保留 workflow fairness credit；
2. `wait_restore` 超过有界阈值时，由同一 JointPlan 原子选择可迁移 victim、H2D restore 和
   replacement admission，不能只输出 `defer`；
3. restore blocked 时显式记录 capacity、ancestor、loading、inflight 等 blocker，并在相关状态
   变化时唤醒，禁止无事件忙轮询；
4. 将 physical commit 改为增量缓存和 touched-slice 校验，使严格 safe-point P99 小于 1 ms；
5. 补齐跨进程 shutdown ACK，使 final summary 可恢复且完整；
6. 重新只运行一次 `w4`，要求 4/4 clean completion、无 execution/activation timeout、无 orphan
   transaction，并且所有 retracted request 都在有界时间内 restore 或显式失败。

可视化与校验产物：

```text
experiments/raw/p5d_repaired_ladder/w4/20260728T053031Z/kv_transfer_timeline.html
experiments/raw/p5d_repaired_ladder/w4/20260728T053031Z/transfer_validation.json
```
