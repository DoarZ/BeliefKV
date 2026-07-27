# BeliefKV 满载 admission liveness 修复与验证

日期：2026-07-17

## 1. 验证目标

修复 `admission_liveness_native_reclaim` 在没有证明可回收空间时强制 admission
的问题，并在单卡、8 个并发 Deep Agents/SWE-bench workflow、163,840-token
KV pool 下验证请求能够在接近满载时持续前进。

## 2. 修复内容

1. 将 native reclaim admission 改为请求级容量证明。runtime 重新匹配目标请求，
   刷新其 uncached token 数，并计算：

   ```text
   native capacity = allocator free + Radix evictable
                     - admission 后新锁住的 prefix tokens
   ```

   只有在 `required < native capacity`、engine idle、无 reservation、无 queued/in-flight
   transfer 且请求完整 working set 不超过 pool 时才能委托给 SGLang native reclaim。

2. watchdog 不再授权无证明 admission。in-flight DMA 未终止时继续阻止 native admission。

3. 修复主动 H2D prefetch 的 HiCache 协议。`load_back()` 只把任务放入 SGLang
   `load_queue`；BeliefKV 现在在提交完整 H2D bundle 后调用
   `ready_to_load_host_cache()`，避免 admission 阻塞时因没有新 prefill batch 而永远不触发 copy。

4. 使 `COMMIT_CPU` ACK 对已经达到 `CPU_ONLY` 的后置状态幂等。其他非法状态仍然
   抛错，并在错误中报告实际 residency。

5. `transfer_dispatched` 审计新增 `action_counts`，区分 `START_D2H`、`START_H2D`、
   `COMMIT_CPU` 和 `DROP`。

## 3. 调试过程中的两个负例

### 3.1 H2D 未被启动

实验：`20260716T175417Z/planned-8-proven-native-exact`

- 243 个请求完成后停止前进。
- `reactive-124` prefetch 超过 300 秒没有 ACK。
- 原因是主动 `load_back()` 后没有触发 HiCache `load_cache_event`。
- 容量证明存在，但安全条件正确地拒绝在 in-flight transfer 上强制 admission。

### 3.2 COMMIT_CPU ACK 非幂等

实验：`20260716T181114Z/planned-8-prefetch-trigger-exact`

- H2D 死锁消失，213/214 条迁移在崩溃前收到 ACK，完成 280 个请求。
- scheduler 在应用 `COMMIT_CPU` ACK 时因目标已经是 `CPU_ONLY` 而抛出
  `PageIndexError`。
- 该实验确认 H2D 触发修复有效，同时暴露逻辑镜像与物理后置状态的幂等性缺口。

以上调试 run 及更早的 Deep Agents 中间产物已归档到
`experiments/archive/20260717/raw/deepagents_swebench/`。最终验证 run 仍保留在
`experiments/archive/20260727/superseded_raw/deepagents_swebench/20260716T182104Z/`。

## 4. 最终实验配置

- 模型：Qwen3-Coder-30B-A3B-Instruct-FP8
- GPU：单卡 RTX 6000 Ada，TP=1
- SGLang：0.5.2rc1
- `mem_fraction_static=0.952`
- `max_total_tokens=163840`
- `context_length=262144`
- `max_running_requests=16`
- `chunked_prefill_size=4096`
- HiCache：ratio=2，write-back
- workload：固定的 8 个 SWE-bench Verified SymPy 实例，planned mode，动态生成 30 个 subagent
- predictor=false，shadow=false，prefetch=true

最终产物：

- `experiments/archive/20260727/superseded_raw/deepagents_swebench/20260716T182104Z/planned-8-ack-idempotent-exact/summary.json`
- `experiments/archive/20260727/superseded_raw/deepagents_swebench/20260716T182104Z/server-ack-idempotent/runtime_audit.jsonl`

## 5. 最终结果

### 5.1 正确性与 liveness

- LLM request：562 submitted / 562 admitted / 562 started / 562 finished
- transfer：589 dispatched / 589 acknowledged
- transfer watchdog：0
- scheduler exception：0
- terminal request cancellation：0
- 有证明的 `admission_liveness_native_reclaim`：116 次
- 116 次均满足严格不等式 `required_bytes < native_reclaim_capacity_bytes`
- 最小证明余量：13,103,235,072 bytes

### 5.2 压力与迁移

- controller 安全点峰值：163,755 / 163,840 tokens，99.948% KV pool
- 轮询指标峰值：149,559 tokens，91.284%；轮询会漏掉短时峰值
- GPU 峰值占用：47,934 MiB；最小空闲：585 MiB
- 实际 KV D2H/H2D：14,826,995,712 bytes
- 其中 offload：12,466,225,152 bytes
- 其中 prefetch：2,360,770,560 bytes
- DROP reclamation：47,055,667,200 bytes（跨时间累计，可重复使用同一容量）

迁移 action 总数：

| Action | 数量 |
| --- | ---: |
| `START_D2H` | 1,437 |
| `START_H2D` | 198 |
| `COMMIT_CPU` | 1,567 |
| `DROP` | 1,428 |

### 5.3 workload 终止状态

8 个 workflow 均产生 `workflow_end`，但只有 1 个业务结果标记为 completed；其余 7 个
达到 LangGraph `recursion_limit=30`。因此 runner 返回码为 1。该结果不代表 SWE-bench
任务质量通过，只证明 KV/runtime 链路完成了全部 562 个 LLM 请求且未发生系统级停滞。

## 6. 尚未解决的性能问题

本轮证明了 correctness/liveness，不证明当前策略性能最优：

- admission wait p95：46.380 秒；max：58.320 秒
- `no_migratable_marginal_pages`：1,128 次
- 240 条迁移因 `node is engine-locked or loading` 仅部分完成
- 589 个 ACK 中 339 completed、246 partial、4 rejected

这些数据说明 locked-but-needed / closure 不可迁移状态已成为主要重试来源。后续应先按
blocker set、locked bytes、完成前后 evictable delta 和 closure amplification 做细分，
再决定是否实现 Yield-to-Reclaim。不能仅凭本实验直接声称 lock convoy 是 46 秒 p95
等待的全部原因。

## 7. 可比性限制

与 `20260716T165200Z/planned-8-fixed-exact` 相比，新版本从手工停止时的 298 个请求推进到
562 个请求并自然结束，watchdog 从 1 降为 0。由于模型采样和动态 agent 行为没有完全
确定化，两次运行的请求序列不同；该对照可用于验证 liveness 修复，不能作为正式吞吐或
延迟 A/B 结论。正式性能比较需要冻结完整 LLM/tool event trace 或至少固定采样随机性并
进行多次重复实验。

## 8. 回归测试

- `conda run --no-capture-output -n beliefkv pytest -q`
  - 143 passed, 4 skipped
- `conda run --no-capture-output -n beliefkv-agents pytest -q tests/test_deepagents_adapter.py tests/test_deepagents_swebench.py`
  - 12 passed

跳过项和警告与本次修改无关；测试沙盒中 NVML 不可用，真实 GPU 指标由端到端实验采集。
