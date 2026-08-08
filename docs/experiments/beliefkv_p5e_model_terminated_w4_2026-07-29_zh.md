# BeliefKV P5E model-terminated fixed w4 GPU gate

日期：2026-07-29

状态：**失败；负载与迁移 characterization 有效，但 clean completion 和性能 gate
无效。根因是 admission-only restore 的 funding 缺口与队首阻塞。**

> 后续代码状态（2026-07-29）：admission-only restore 已接入统一 funding reclaim；
> blocker 未产生物理命令时会继续扫描后续 obligation，并使用包含 request demand 的
> attempt stamp 抑制不变状态重算。CPU 回归已通过，本文仍保留原失败实验结论，GPU
> 复验尚未执行。

## 1. 实验范围

本轮只执行一次固定 w4 trace，没有失败后重跑：

- `Qwen3-Coder-30B-A3B-Instruct-FP8`，SGLang `0.5.2rc1`，TP=1，仅 GPU 0；
- KV pool 163,840 token，context length 262,144，Host HiCache 96 GB；
- 4 个固定 SWE-bench Verified SymPy workflow：3 个 mixed、1 个 cyclic-peer；
- workflow deadline 从 1,800 秒放宽到 7,200 秒，call-count budget 仅记录、不强制终止；
- online JointPlan、observed admission、running retraction、RestoreObligation 和
  `RestoreLease` 全部开启。

原始数据：

```text
experiments/raw/p5e_model_terminated_w4/20260729T114036Z/
```

## 2. Workload 事实

客户端在人工停止前记录了 256 次 LLM submit、247 次 LLM result、247 次工具调用、
12 次 runtime spawn 和 4 次 child RETURN。没有 call-count guard、语义强制完成、
API timeout、服务端 request execution timeout 或 OOM。

因此本轮不是由于任务过短、模型拒绝 spawn，或 7,200 秒 deadline 到期而失败。四个
workflow 都未自然结束，故 `clean_jct_eligible=0/4`，不能用来比较 JCT 或吞吐。

## 3. 主要失败

服务端在 19:50:50 完成最后一个请求后，保持约 323 秒的 `0 running / 9 waiting`；
GPU 利用率为 0。稳定停滞状态为：

| 指标 | 数值 |
| --- | ---: |
| HBM used / free | 15.42 GB / 0.69 GB |
| Host KV | 26.56 GB |
| migratable KV | 13.28 GB |
| dual-resident KV | 14.44 GB |
| engine-locked KV | 2.14 GB |
| active restore lease / inflight command | 0 / 0 |

此时有 3 个 active ordinary-waiting restore obligation：`restore-19/20/21`。在线
JointPlan 持续把 9 个 waiting request 全部标为 `defer`，且不产生 residency action。

精确控制流为：

```text
restore-19 位于 obligation 队首
  -> prefix 已在 HBM，无 H2D extent
  -> incremental admission capacity 不足
  -> _grant_restore_lease(h2d_bytes=0) 失败
  -> 记录 restore_lease_capacity 后直接 return
  -> 不进入 funding reclaim
  -> 也不扫描 restore-20/21
  -> 0 running / 9 waiting，GPU 空闲
```

代码中的 funding reclaim 只存在于“有 H2D preview 但容量不足”的分支；no-H2D
分支在 lease 失败后直接返回。与此同时，restore driver 对一个 obligation 的非进展结果使用
`return`，形成 head-of-line blocking。

客户端取消 `restore-19` 后，系统立刻为 `restore-20` 找到 1.10 GB funding bundle，随后
又为 `restore-21` 找到约 1.00 GB funding bundle。这是上述归因的直接反事实证据。

## 4. 正确性与迁移结果

物理数据面 gate 通过：89 dispatch 对应 89 ACK，无 missing、orphan 或时序违规；419 条
telemetry 中 411 completed、8 rejected；retry guard 没有发现 retry-without-release 或
重复零字节提交。

- D2H：26.56 GB；H2D：65.27 GB；共 417 次物理迁移；
- peak HBM：16.11 GB；peak Host KV：26.75 GB；
- peak engine-locked：15.35 GB；
- peak locked-but-not-served（100/500 ms）：均为 13.70 GB。

21 个 restore obligation 中 18 个正常 satisfied；`restore-19/20/21` 在人工终止后以
`request_aborted` 关闭。三者最终等待约 442.61、408.79 和 410.41 秒。

## 5. 控制面问题

停滞期间控制面仍高速空转：累计 199,245 次 global validation fallback、84,665 次
physical commit budget exceeded、202,248 个 safe-point seed epoch 和 40,076 次
restore-debt barrier。最终 audit 为 172.82 MB。物理状态不变时应 event-gate/no-op
backoff，不能在每个 idle tick 重编译同一空计划。

关停后 command 和 restore transaction 均已清空，GPU 0/1 无残留模型进程，也没有残留
实验容器。但两阶段 shutdown summary 仍停留在 `preparing`、`final=false`，因此 shutdown
gate 未通过。

## 6. 修复边界

1. no-H2D restore 的 admission lease 不足时，也必须进入 JointPlan funding reclaim；
2. 某个 obligation 未发出物理命令时不能阻止扫描其他 obligation；
3. safe point 每次选择一个公平且物理可行的 `funding + lease + restore/admission` 原子事务；
4. lease 继续保持到请求首次获得 GPU service，失败时显式 rollback；
5. 为不变 blocker 状态增加 backoff，并修复最终 shutdown ACK。

本轮产物：

```text
experiments/raw/p5e_model_terminated_w4/20260729T114036Z/experiment_outcome.json
experiments/raw/p5e_model_terminated_w4/20260729T114036Z/transfer_validation.json
experiments/raw/p5e_model_terminated_w4/20260729T114036Z/kv_transfer_timeline.html
experiments/raw/p5e_model_terminated_w4/20260729T114036Z/kv_transfer_timeline.json
```
