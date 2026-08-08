# BeliefKV P5E restore funding 修复后 w4 GPU 复验

日期：2026-07-30

状态：**失败。物理迁移正确性通过，51/52 restore obligation 恢复服务，但 restore grace 与 admission funding reservation 仍不闭环，不能进入性能比较。**

## 1. 实验范围

本轮只执行一次固定 w4 trace，没有重跑或在线修改：

- `Qwen3-Coder-30B-A3B-Instruct-FP8`，SGLang `0.5.2rc1`，TP=1，仅 GPU 0；
- KV pool 163,840 token，context 262,144 token，`MEM_FRACTION_STATIC=0.952`；
- Host HiCache 96 GiB，predictor 关闭；
- 4 个固定 SWE-bench SymPy workflow，3 个 mixed、1 个 cyclic，并发启动；
- mixed workflow 静态要求初始创建 2--4 个 subagent，后续执行仍由模型决定；
- 单请求 execution timeout 900 秒，workflow wall-clock 边界 7,200 秒；
- observed JointPlan、ordinary-waiting restore、running retraction、RestoreLease 和 restore funding 开启。

原始数据：

```text
experiments/raw/p5e_restore_funding_fix_w4/20260729T164632Z/
```

## 2. Gate 结论

| Gate | 结果 | 证据 |
| --- | --- | --- |
| 固定 trace 且只运行一次 | PASS | w4 单次运行，没有重跑 |
| 高 HBM 压力 | PASS | HBM 峰值 15.0 GiB，达到配置 KV pool 容量 |
| Host KV 压力 | PASS | Host KV 峰值 35.3 GiB |
| 物理 command/ACK 完整性 | PASS | 125 dispatch、125 ACK，0 missing/orphan/ordering violation |
| Residency 一致性 | PASS | HBM mirror 未超过 allocator，Host/page-index 0 mismatch |
| Restore funding 路径覆盖 | PASS | 37 次 funding，累计 reclaim 50,948,112,384 bytes |
| Ordinary-waiting restore | PASS（本轮样本） | 18/18 satisfied，最长等待 183.07 秒 |
| Running-retraction restore | **FAIL** | 33 satisfied，`restore-52` 在人工停止时 cancelled |
| Clean completion | **FAIL** | 4 个 workflow 均未自然结束，本轮主动停止 |
| Shutdown transaction closure | PASS | 0 running/waiting、0 pending transaction、0 active obligation |
| Final shutdown summary | FAIL | `shutdown_state=preparing`，`final=false` |

工作负载在停止前产生 305 次 LLM submit、300 次 LLM result、463 次工具完成、10 次 spawn 和 6 次 child RETURN。负载强度足以触发多轮 HBM pressure，但因系统 liveness 失败，本轮不得用于 JCT、吞吐或策略收益比较。

## 3. 已验证的修复效果

52 个 restore obligation 中，51 个最终重新获得 GPU service：

- 18 个 ordinary-waiting obligation 全部 satisfied；
- 33 个成功的 running-retraction obligation，最大等待 19.70 秒；
- 多个 admission-only restore 先执行 D2H funding，再执行 H2D 或直接准入；
- 同时存在多个 blocked obligation 时，系统没有再次出现 `0 running + waiting` 的全局死锁；
- 0 个 `physical_preview_unavailable` 永久阻塞，说明先前 current-Radix rebind 修复仍然有效。

ordinary-waiting 的等待仍很长：p50 38.90 秒、p95 167.84 秒、最大 183.07 秒。这证明 liveness 路径存在，但不能证明公平性或尾延迟已经合格。

## 4. 主要失败：恢复后立即再驱逐，funding 不形成债务所有权

失败请求为 mixed-000 的 child context `deepagents-context:e3ebeaf97aefe1ee`：

```text
restore-51 ordinary waiting
  -> funding 3.75 GB
  -> H2D 2.01 GB
  -> admission commit + first GPU service
  -> 35.66 秒后 satisfied

仅 1.77 秒后
  -> 同一 request 被选为 running retraction victim
  -> 创建 restore-52
  -> 首次 H2D 因 authoritative GPU copy 缺失被拒绝
  -> 第二次 H2D 成功恢复 5.07 GB
  -> lease 只保留 402,456,576 bytes / 4,094 token
  -> 没有发生 admission attempt
  -> 后续累计执行 7 次 funding，reclaim 11.89 GB
  -> 期间至少 4 个其他 active request 完成，普通新请求继续进入 batch
  -> restore-52 持续 blocked 553 次
  -> 423.96 秒后随人工停止进入 cancelled
```

这不是 H2D 数据损坏，也不是 command callback 丢失。`restore-52` 的 9 条命令都有 ACK，第二次 H2D 确实完成。问题位于策略状态机：

1. RestoreLease 在首次 token service 后立即释放，而 retraction cooldown 只有 1 秒；请求尚未获得有意义的 service quantum，就能再次成为 victim。
2. H2D 完成但 admission 尚未 commit 时，obligation 没有继续作为严格 restore-debt barrier。
3. 后续 funding 释放的容量没有归属于最老 restore debt；新 admission 和 active KV growth 可以重新消耗这些容量。
4. 当前 402 MB lease 只覆盖下一 prefill chunk，不能表达已经恢复的 5.07 GB prefix 加 admission headroom 的完整债务。

因此，本轮修复解决了“没有 victim 时不尝试 funding”和“单个 blocked obligation 阻塞扫描”，但没有解决 restore transaction 从首次恢复到稳定执行的端到端原子性。

## 5. 数据面结果

离线 `validate-transfer-telemetry` 通过 command integrity 和 residency consistency：

- 782 条物理 transfer telemetry，780 completed、2 rejected；
- 125 dispatch、125 ACK，0 missing ACK，0 orphan ACK，0 ordering violation；
- D2H 37,856,083,968 bytes，H2D 200,321,040,384 bytes；
- 峰值 HBM 16,106,127,360 bytes，峰值 Host KV 37,856,083,968 bytes；
- 峰值 engine-locked KV 15,506,571,264 bytes；
- 100/500 ms locked-but-not-served 峰值分别为 15,153,659,904 和 14,425,915,392 bytes；
- Host residency 与 page index 一致，HBM mirror 始终是 allocator 的子集。

H2D 约为 D2H 的 5.29 倍，且同一 context 存在 restore/retraction churn。本轮只能证明迁移 bookkeeping 正确，不能证明迁移策略有效。

## 6. 正确修复边界

下一次修改应保持为一个统一 restore transaction，而不是再增加独立分数：

1. **Restore grace**：obligation satisfied 后，request 至少获得配置的 service token/quantum，或到达 LLM call 终点前，不得再次成为 retraction victim；仅用固定 1 秒 cooldown 不足。
2. **Debt-owned funding reservation**：overdue restore 开始 funding 后，reclaim 的可用字节归属于该 obligation，直到 admission commit/rollback；普通 admission 不得消耗这部分 escrow。
3. **完整状态覆盖**：`H2D_ACKED`、`PREFIX_PINNED`、`TICKET_READY`、`ADMISSION_STARTED` 都属于未偿还 restore debt，必须继续参与 barrier 和 oldest-first aging。
4. **有界 bypass**：允许小请求 work-conserving bypass，但达到 `max_bypass_admissions` 或 age threshold 后必须冻结新 admission，为最老 debt 累积足额容量。
5. **事件驱动重试**：同一 blocker snapshot 不应产生 553 次审计记录；只在 allocator/page/topology、running set 或 transfer terminal 变化时重试。

完成 CPU 状态机测试后，再执行一次相同 w4 trace。通过条件必须包含 52/52 obligation satisfied、0 restore-after-restore thrash、0 starvation、4/4 workflow 自然完成；在此之前不进入性能实验。

## 7. 产物

```text
experiments/raw/p5e_restore_funding_fix_w4/20260729T164632Z/experiment_outcome.json
experiments/raw/p5e_restore_funding_fix_w4/20260729T164632Z/transfer_validation.json
experiments/raw/p5e_restore_funding_fix_w4/20260729T164632Z/kv_transfer_timeline.html
experiments/raw/p5e_restore_funding_fix_w4/20260729T164632Z/kv_transfer_timeline.json
```

SGLang 的唯一 traceback 是 SIGINT 关闭 detokenizer 时的 `KeyboardInterrupt`；没有 OOM、CUDA error、allocator inconsistency 或运行期未捕获异常。实验结束后 SGLang 与 workload 进程均已退出，四个临时 sandbox 已删除，本轮进程一度将 GPU0/1 显存释放至 0 MiB。最终复核时两张卡已被后续启动的 `well-native` 任务占用，与本轮实验无关。
