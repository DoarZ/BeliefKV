# BeliefKV P5E service-quantum restore grace w4 GPU 实验

日期：2026-07-30

状态：**失败。restore grace 和 debt-owned funding 已获得在线覆盖，但 SGLang 原生 idle memory self-check 未计入 BeliefKV reservation，scheduler 在约 20 分钟时退出；本轮不得用于性能结论。**

## 1. 实验配置

本轮只执行一次固定 w4 trace：

- `Qwen3-Coder-30B-A3B-Instruct-FP8`，SGLang `0.5.2rc1`，TP=1，仅 GPU 0；
- KV pool 163,840 token，context 262,144 token，`MEM_FRACTION_STATIC=0.952`；
- Host HiCache 96 GiB，predictor 关闭；
- 4 个 SWE-bench SymPy workflow：3 个 mixed、1 个 cyclic，并发启动；
- mixed workflow 各创建 4 个 multi-turn child，共 12 个 child；
- RestoreLease、debt-owned funding、32-token service grace、observed JointPlan 和 running retraction 开启；
- 单请求 execution timeout 900 秒，workflow wall-clock 7,200 秒。

原始数据：

```text
experiments/raw/p5e_restore_service_grace_w4/20260730T014235Z/
```

## 2. 运行结果

运行 1,220 秒后，SGLang scheduler 在 idle self-check 中退出，4 个 workflow 随后因 runtime socket 消失而结束，均不具备 clean JCT 资格。

停止前 workload 已产生：

- 262 次 LLM request；
- 258 次工具调用，其中 240 success、18 error；
- 12 次 spawn、1 次 child RETURN；
- 峰值 HBM KV 15.0 GiB，峰值 Host KV 23.5 GiB。

该负载已覆盖高 HBM 压力和真实多轮工具调用，但因 scheduler 异常退出，不能报告 workflow JCT、吞吐收益或策略优于 baseline。

## 3. Restore grace 覆盖

崩溃前共创建 84 个 restore obligation，82 个恢复 GPU service：

- ordinary-waiting：69 satisfied；
- running-retraction：13 satisfied；
- 等待时间 p50 12.67 秒、p95 79.58 秒、最大 104.47 秒；
- 82 个请求进入 service grace；
- 76 个达到 32-token quantum，6 个在达到 quantum 前自然完成；
- 0 个 grace 因再次 retraction 而取消。

因此 service grace 的核心状态转换已被真实 GPU workload 覆盖，并阻止了“刚恢复立即再驱逐”。这仍不是完整 gate 通过，因为本轮存在 allocator self-check 崩溃和另一个普通请求饥饿。

## 4. 主失败：reservation 未进入原生内存守恒式

崩溃时 SGLang 报告：

```text
max_total_num_tokens = 163840
available_size       = 16747
evictable_size       = 126541
protected_size       = 0
```

原生公式认为缺失 20,552 token。BeliefKV 同时持有：

```text
funding reservation = 16466 token
RestoreLease         =  4086 token
合计                 = 20552 token
```

所以物理 token 没有泄漏，问题是 SGLang `check_memory()` 只统计 native allocator、Radix evictable 和 protected KV，不知道 BeliefKV 为 restore debt 主动占用的 allocator slots。

实验后已修复：

1. `EmbeddedSGLangRuntime.allocator_backed_reservation_tokens()` 汇总 funding 和 RestoreLease 的真实 allocator allocation；
2. SGLang idle self-check 将该值加入 token 守恒式；
3. 真正缺失 1 token 的测试仍会触发 memory-leak 异常，检查没有被关闭或放宽。

## 5. 次要问题

RestoreLease 只有一个全局槽。槽被占用时，后续 obligation 曾每个 scheduler tick 执行 funding release、lease grant 失败、funding reacquire，造成 461 次 release 事件但仅涉及 72 个 obligation。这不是 double-free，但会制造 allocator 和审计开销。

实验后已增加 lease-capacity precheck：全局槽满时保留 funding escrow，不再开始转换。

另有一个普通请求在 physical start 后持续获得 GPU service，但仍在固定 900 秒墙钟边界被取消：

```text
request  = beliefkv:019fb0b8-38ea-7e82-ae65-7c96c4a49a8e
context  = deepagents-context:121836a9d1924ac0 epoch 12
queue wait before physical start = 4.68 s
execution wall time at abort     = 900.30 s
prefill service samples          = 1
decode service samples           = 1,169
```

此前将其归因为 admission/running ownership 断裂是因为错误地按事件顶层 `request_id` 查询；`gpu_service_sample` 实际把成员记录在 `request_ids[]`。正确归因是服务端把 `execution_timeout_s` 当作 physical start 后的固定总寿命，误杀了仍在持续 decode 的长请求。

实验后已将该边界改为 GPU-service inactivity watchdog：总运行时间只做 telemetry；只要请求持续完成 GPU batch，就可以运行超过 900 秒。只有从最后一次完成 service（尚未完成过则从首次 selection）起连续无进展达到阈值时才取消。ledger 缺失的兼容路径才退回 physical-start 计时。

## 6. 迁移正确性

离线 validation 结果：

- 239 dispatch、239 ACK；
- 0 missing ACK、0 orphan ACK、0 ordering violation；
- 534 次实际 DMA；
- D2H 24,552,701,952 bytes；
- H2D 196,243,292,160 bytes；
- Host/page-index 0 mismatch，HBM mirror 始终是 allocator 子集；
- retry guard 无相同物理 fingerprint 的重复提交。

H2D 远大于 D2H，说明长多轮 context 下 native demand-load 和 restore churn 仍然显著；本轮只证明迁移 bookkeeping 完整，不证明迁移策略高效。

## 7. 修复验证与下一步

实验后执行：

```text
pytest
445 passed, 8 skipped
```

本轮没有自动重跑 GPU 实验。下一次固定 w4 gate 必须同时满足：

- 无 allocator、Radix、overlap 或 callback 异常；
- 100% restore obligation terminal，0 orphan transaction；
- 0 physical-start-without-service request；
- 4/4 workflow 自然结束；
- shutdown summary 完整。

在此之前，P5 仍属于 correctness/liveness 修复阶段，不能进入性能对比。
