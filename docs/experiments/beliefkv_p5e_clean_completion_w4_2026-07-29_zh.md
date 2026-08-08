# BeliefKV P5E fixed w4 clean-completion GPU gate

日期：2026-07-29

状态：**失败；物理迁移正确性通过，但 restore-to-admission liveness、workflow clean completion
和 shutdown summary 未通过。不得用于性能比较。**

> 后续状态（2026-07-29）：本文保留失败实验的原始结论。代码加入 allocator-backed
> `RestoreLease`、restored-prefix pin、owner-only admission credit、native admission
> commit/rollback 和 bounded bypass 后，已执行一次同一固定 w4 trace 的 GPU 复验。复验中
> 17/18 restore obligation 成功，但 parent reactivation 因 physical bundle 重绑定失败仍未通过
> clean-completion gate；见 `beliefkv_p5e_restore_lease_w4_2026-07-29_zh.md`。

## 1. 实验范围

本轮只执行一次固定 w4 trace，没有失败后重跑。配置为：

- `Qwen3-Coder-30B-A3B-Instruct-FP8`，SGLang `0.5.2rc1`，TP=1，仅 GPU 0；
- KV pool 163,840 token，`MEM_FRACTION_STATIC=0.952`，Host HiCache 96 GB；
- 4 个固定 SWE-bench Verified SymPy workflow，3 个 mixed、1 个 cyclic；
- mixed workflow 由模型在 runtime enforcement 下选择 2--4 个初始 subagent；
- 单请求 execution timeout 900 秒，workflow 绝对 deadline 1,800 秒；
- online JointPlan、observed admission、running retraction 和 restore obligation 全部开启。

原始数据：

```text
experiments/raw/p5e_clean_completion_w4/20260728T160800Z/
```

## 2. Gate 结果

| Gate | 结果 | 证据 |
| --- | --- | --- |
| 固定负载且只运行一次 | PASS | 4 个固定 workflow，未重跑 |
| 负载强度 | PASS | 253 LLM、331 tool；4/4 workflow 和 4/4 subagent intensity gate 通过 |
| 动态 subagent | PASS（机制） | 3 个 mixed workflow 共 runtime 创建 10 个 child |
| Clean completion | **FAIL** | 0/4 `clean_jct_eligible` |
| Restore liveness | **FAIL** | 25/27 satisfied，2/27 cancelled；`restore-23` 等待 874.90 秒 |
| Command/ACK 完整性 | PASS | 99 dispatch、99 ACK，0 missing/orphan/ordering violation |
| Retry guard | PASS | 0 identical zero-byte retry，0 retry-without-release |
| 退出前事务闭合 | PASS | workload 结束后 0 running、0 waiting、0 pending transaction |
| Shutdown summary | **FAIL** | `shutdown_state=preparing`、`final=false`，无 `runtime_shutdown` |

四个 workflow 的终态为：

- cyclic：19 LLM、17 tool，runtime guard 强制收口为 `blocked`；
- mixed-000：55 LLM、135 tool，4 child 中 3 RETURN、1 cancel，`workflow_timeout`；
- mixed-001：66 LLM、63 tool，2 child 全部 cancel，`workflow_timeout`；
- mixed-003：113 LLM、116 tool，4 child 中 1 RETURN、3 cancel，`workflow_timeout`。

deadline 到期时 runtime 向服务端发送 7 个 `/abort_request`，三个未闭合 JOIN 都产生
`join_timeout`，说明 workflow 级取消传播已经生效。此次没有客户端 `APITimeoutError`，服务端只有
一次严格从 physical start 计时的 900 秒 `request_execution_timeout`。

## 3. 主要失败：H2D 与 admission 之间没有持久 reservation

`restore-23` 暴露的事件链为：

```text
running request 被 retraction
  -> D2H 完成并 requeue
  -> 3.426 GB H2D 成功 ACK
  -> restore-liveness ticket 连续签发
  -> native admission: OTHER / prefix rematch / NO_TOKEN
  -> 其余 running request 继续增长 KV
  -> restored request 再次失去可准入空间
  -> overdue restore-debt barrier 阻塞其他 waiting request
  -> 0 running + 7 waiting，直到 900 秒 execution timeout abort
```

H2D ACK 后首个 resource snapshot 只剩 27,131,904 bytes HBM，而该请求仍需约
599,654,400 bytes incremental admission 空间。当前代码在提交 H2D 前检查
`copy_bytes + required_admission_bytes <= available_bytes`，但这只是瞬时检查；H2D 期间没有持久
reservation，其他 6 个 running request 可以继续增长并消费这部分空间。恢复完成的 KV 在首次
service 前也没有 pin/lease，因而 H2D ACK 不等于 admission 可完成。

前 22 个 running-retraction obligation 均在 0.64--4.85 秒内恢复 service，说明基础
`D2H -> H2D -> ACK -> ticket` 路径有效；第 23 个只在接近满载和并发增长下失败，属于条件性原子性
缺口，而非普遍的传输故障。

正确修复必须把以下过程作为一个 reservation-backed transaction：

1. 在 H2D dispatch 前预留 `restore closure + prefill chunk + decode reserve`；
2. reservation 跨 H2D ACK、prefix rematch、ticket 和 native `PrefillAdder` 保持有效；
3. 只有请求进入 running batch 或明确 abort/rollback 后才释放；
4. debt 请求暂时无法消费 reservation 时，barrier 不能让整个系统变成 0-running；
5. restored prefix 在 commit 前不得被普通 native eviction 再次逐出。

## 4. 数据面与资源结果

离线 `validate-transfer-telemetry` 通过：

- 376 条物理 DMA telemetry，其中 53 条 BeliefKV command、323 条 native HiCache callback；
- 374 completed、2 rejected；D2H 25,596,297,216 bytes，H2D 56,292,212,736 bytes；
- 峰值 HBM 16,106,127,360 bytes，峰值 Host KV 25,063,981,056 bytes；
- HBM mirror 始终为 allocator 子集，Host residency 与 page index 一致；
- 峰值 engine-locked KV 为 15,232,499,712 bytes；100/500 ms locked-but-not-served 峰值分别为
  14,848,425,984 和 12,662,341,632 bytes。

H2D 明显高于 D2H，包含 native demand-load 和同一长上下文的重复恢复，不能解释为有效迁移收益。
本轮只能证明数据面 bookkeeping 正确，不能证明 JointPlan 改善 JCT 或吞吐。

## 5. 其他失败项

mixed-000 出现 `repeated_failed_tool_call`，135 次工具调用中 58 次错误，其中 52 次为 runtime
规范化后的 `tool_error`。circuit breaker 能检测并尝试结构化收口，但该 child 最终仍未 RETURN；
这说明 BLOCKED child 到 parent JOIN 的终态传播仍需收紧。

控制面在死锁窗口高速空转：最终累计 246,312 个 safe-point seed epoch、98,258 个 ticket epoch、
96,553 次 restore-debt barrier 和 169,914 次 physical-commit budget exceeded，生成 382,589,809
bytes audit。应在物理状态未变化时 event-gate/no-op backoff，不能每个 idle tick 重编译同一空计划。

workload 结束时所有 request 和事务已经清空，但 Ctrl-C 后 detokenizer 收到 `KeyboardInterrupt`，
summary 只写到 `preparing`，没有 `shutdown_ack/runtime_shutdown`。GPU 0/1 最终均为 0 MiB、无遗留
模型进程；残留的 cyclic SWE-bench sandbox 已在保留 artifact 后删除，常驻 `node-exporter` 未受
影响。进程清理成功不等于 graceful-shutdown gate 通过。

## 6. 下一步

1. 实现 H2D-to-admission 持久 reservation 和 restored-prefix lease；
2. 将 native admission 成功/失败纳入 restore transaction commit/rollback；
3. 修复 debt barrier 的 work-conserving fallback，并增加上述竞态的确定性 CPU 测试；
4. 修复 BLOCKED child 的 RETURN/JOIN 传播和 SIGINT 两阶段 shutdown；
5. CPU 回归通过后再做一次相同 w4 gate，在此之前不执行 w8/w12/w24。

产物：

```text
experiments/raw/p5e_clean_completion_w4/20260728T160800Z/experiment_outcome.json
experiments/raw/p5e_clean_completion_w4/20260728T160800Z/transfer_validation.json
experiments/raw/p5e_clean_completion_w4/20260728T160800Z/kv_transfer_timeline.html
experiments/raw/p5e_clean_completion_w4/20260728T160800Z/kv_transfer_timeline.json
```
