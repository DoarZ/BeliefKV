# BeliefKV P5F 24K Context Compaction W4 实验报告

## 实验定位

- 日期：2026-07-30
- 原始目录：`experiments/raw/p5f_context_compaction_w4/20260730T075638Z`
- 模型：Qwen3-Coder-30B-A3B-Instruct-FP8，单卡 GPU0
- KV 配置：`mem_fraction_static=0.952`，`max_total_num_tokens=163840`，Host HiCache 96 GiB
- workload：4 个固定 SWE-bench Verified SymPy 实例，3 个 mixed、1 个 cyclic
- context policy：24,576 token 触发总结，保留约 8,192 token；普通调用 1,024 输出 token，总结调用 2,048 token，收尾调用最多 4,096 token
- workflow wall-clock：7,200 秒；单请求执行上限：900 秒

本轮用于验证 24K context lifecycle、旧 KV ownership 释放和迁移正确性。由于另一用户在实验前半段使用 GPU1，且所有 workflow 均未 clean completion，本轮不能用于 JCT 或绝对吞吐对比。

## 结果摘要

| 指标 | 结果 |
| --- | ---: |
| 实验时间线长度 | 45.85 min |
| Context summary / compact / commit | 9 / 9 / 9 |
| 涉及压缩的 context | 5 |
| 物理 transfer telemetry | 1,084 |
| D2H | 1,049 次，46.87 GiB |
| H2D | 35 次，34.73 GiB |
| 完成 / rejected transfer | 1,082 / 2 |
| 峰值 HBM KV | 15.00 GiB |
| 峰值 Host KV | 41.89 GiB |
| 峰值 engine-locked KV | 14.47 GiB |
| 峰值 locked-not-served KV（100/500 ms） | 13.72 / 12.95 GiB |
| 峰值 migratable KV | 14.54 GiB |
| clean completion | 0 / 4 |

所有 9 次 compaction 均提交 `old_kv_disposition=release_ownership`，压缩后请求继续运行。没有出现 summarization cutoff 越界、OOM、API timeout、Radix/allocator consistency 错误或未决 restore/retraction transaction。

迁移验证通过：90 次 dispatch 全部收到 ACK；31 次显式 DMA command 均有 telemetry；没有 orphan ACK、遗漏 telemetry、顺序错误或字节上界错误。30/32 个 explicit physical bundle 完成，reclaim realization ratio 为 99.32%。

## Workflow 结果

| Workflow | 负载 | LLM / Tool | Compaction | 终止原因 |
| --- | --- | ---: | ---: | --- |
| mixed-000 | 5 个动态 child | 207 / 211 | 2 | runtime validation error：初次 delegation 接受了 5 个 child，超过配置上限 4 |
| mixed-001 | 3 个动态 child | 91 / 97 | 2 | blocked：parent 4096-token 收尾未产生结构化结果，压缩后重试仍未自然完成 |
| cyclic-002 | 单 persistent peer | 71 / 70 | 3 | blocked：重复失败的工具调用触发卡死检测 |
| mixed-003 | 4 个动态 child | 86 / 81 | 2 | `InvalidUpdateError`：并发 child 在 JOIN 时同时写 `_summarization_event` |

`mixed-003` 的错误证明当前 `private_state_keys` 只避免了部分 history 泄漏，仍未阻止 LangGraph 在同一 superstep 合并多个 child 的内部 summarization 状态。应让该状态完全留在 child graph 内，或使用不会参与 parent state merge 的 side channel；不应把它改为简单 list reducer，因为 parent 不应消费 child 的 compaction cursor。

## 性能观测

高压阶段 HBM KV 达到配置容量，说明本轮确实触发了 KV pressure。后半段只剩两个 decode request 时，GPU 利用率较低；GPU service sample 呈约 0.56 秒和 0.03 秒交替，服务端阶段吞吐一度只有 4-6 token/s。两条请求持续获得 service，因此该现象不是 admission/restore starvation，但会使 4,096-token 收尾逼近 900 秒单请求上限。需要在后续独立实验中分离 SGLang overlap worker、控制面 safe-point 和模型 kernel 的成本。

## 判定

1. **通过**：24K 自动总结、`CONTEXT_COMPACT` 事件、context epoch 推进和旧 ownership 释放链路。
2. **通过**：显式与 native HiCache 传输的 ACK、telemetry 和 allocator 一致性。
3. **未通过**：agent runtime clean-completion gate；0/4 workflow 可进入 JCT 样本。
4. **待修复**：并发 child summarization state 隔离、初始 spawn 上限的原子 admission、结构化收尾失败后的 bounded recovery。
5. **待测量**：低并发长 decode 下的 service-quantum 抖动及控制面开销。

## 产物

- `workloads/manifest.json`：完整 workload 配置和汇总
- `workloads/*/result.json`：每个 workflow 的终止原因与强度统计
- `server/runtime_audit.jsonl`：控制面和物理状态审计
- `server/transfer_telemetry.jsonl`：物理迁移 telemetry
- `transfer_validation.json`：迁移正确性验证
- `kv_transfer_timeline.html`：HBM/Host/lock/transfer 可视化
- `kv_transfer_timeline.json`：时间线结构化数据
