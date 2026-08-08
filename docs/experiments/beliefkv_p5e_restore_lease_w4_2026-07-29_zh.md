# BeliefKV P5E restore-lease w4 GPU gate

日期：2026-07-29

状态：**失败。RestoreLease 修复通过 17 个 restore obligation，但 parent reactivation 的
physical bundle 重绑定失败，4/4 workflow 最终超时。不得用于性能比较。**

## 1. 实验范围

本轮只执行一次固定 w4 trace，没有重跑或在线修改：

- `Qwen3-Coder-30B-A3B-Instruct-FP8`，SGLang `0.5.2rc1`，TP=1，仅 GPU 0；
- KV pool 163,840 token，context 262,144 token，`MEM_FRACTION_STATIC=0.952`；
- Host HiCache 96 GB；
- 4 个固定 SWE-bench Verified SymPy workflow，3 个 mixed、1 个 cyclic；
- mixed workflow 在 runtime enforcement 下创建 2--4 个 subagent；
- 单请求 execution timeout 900 秒，workflow 绝对 deadline 1,800 秒；
- observed JointPlan、running retraction、ordinary-waiting restore 和 RestoreLease 开启。

原始数据：

```text
experiments/raw/p5e_restore_lease_w4/20260729T080937Z/
```

## 2. Gate 结果

| Gate | 结果 | 证据 |
| --- | --- | --- |
| 固定 trace 且只运行一次 | PASS | 4 个固定 workflow，未重跑 |
| 负载强度 | PASS | 232 LLM、399 tool；4/4 workflow intensity gate 通过 |
| 动态 subagent | PASS（机制） | 3 个 mixed workflow 共创建 12 个 child |
| RestoreLease commit/rollback | PASS（已覆盖路径） | 17 grant、17 admission commit；一次 `NO_TOKEN` 后成功重试 |
| Running-retraction restore | PASS | 3/3 obligation 恢复 GPU service |
| Ordinary-waiting restore | **FAIL** | 14/15 satisfied，`restore-18` 被 deadline 取消 |
| Clean completion | **FAIL** | 0/4 `clean_jct_eligible`，4/4 `workflow_timeout` |
| Command/ACK 完整性 | PASS | 113 dispatch、113 ACK，0 missing/orphan/ordering violation |
| 迁移 retry guard | PASS | 0 identical zero-byte retry，0 retry-without-release |
| 退出前事务闭合 | PASS | 0 running、0 waiting、0 pending transaction |
| Shutdown summary | **FAIL** | `shutdown_state=preparing`，无完整 final summary |

workload 共创建 12 个 child，其中 7 RETURN、5 在 deadline 时 cancel；发生一次正常
`join_satisfied` 和两次 `join_timeout`。工具错误共 156 次，其中 mixed-003 占 148 次，因此本轮
同时存在 runtime 任务质量问题；但导致全局 GPU idle stall 的直接系统原因可以独立定位。

## 3. 主要失败：restore debt 无法重绑定 physical bundle

`restore-18` 对应 mixed-000 的 parent reactivation。其 4 个 child 已全部 RETURN，JOIN 已满足，
随后发生：

```text
parent context epoch 6 进入 waiting
  -> 观察到 15 个 CPU-only extent，共 1,229,094,912 bytes
  -> 创建 ordinary-waiting restore-18
  -> refresh 仍认为这些 extent 需要恢复
  -> bundle_builder 无法为 context epoch 6 生成包含这些 handle 的 PREFETCH_CONTEXT preview
  -> physical_preview_unavailable，无法 grant RestoreLease
  -> restore debt barrier 阻塞普通 admission
  -> 0 running + 7 waiting，GPU 利用率降为 0
  -> 1,253.55 秒后 workflow deadline abort
```

这不是 RestoreLease 容量不足。Lease 只在物理 preview 已存在后才能预留 HBM；本例中逻辑层保存的
CPU-only extent 与当前 Radix handle/owner closure 无法重新对应，因而事务停在 lease 之前。
正确修复应先重新绑定当前 request path、Radix generation 和 context ownership；若 extent 已 stale
或不再属于该 context，必须原子选择 native rematch、drop-and-prefill 或明确 recompute，不能把
`physical_preview_unavailable` 作为无限等待条件。

此外，`restore-18` 产生 1,061 次相同 blocker 记录。应按 page/topology/allocator revision 做
event-gated backoff，并确保 overdue debt 在无法形成物理动作时不阻塞无关 runnable request。

## 4. 已验证的 RestoreLease 行为

18 个 obligation 中，17 个恢复 service，3 个来自 running retraction，15 个来自 ordinary waiting。
成功 obligation 的最长等待为 260.76 秒。`restore-7` 首次 native admission 返回 `NO_TOKEN`，代码
正确回滚 admission 状态、保留 obligation，并在第二次准入后 commit 和释放 lease。这说明
`grant -> prefix pin -> native admission -> commit/rollback -> first service release` 的已覆盖路径有效。

但容量 blocker 的轮询仍然过密：全轮共有 4,152 个 `restore_obligation_blocked`、118,568 个
admission ticket epoch 和 201,750 个 physical-commit budget exceeded，审计文件达到
440,665,970 bytes。控制面需要基于状态变化触发，而不是在相同物理快照上高速重试。

## 5. 数据面结果

离线 `validate-transfer-telemetry` 通过：

- 360 条 DMA telemetry：61 条 BeliefKV command、299 条 native HiCache callback；
- 353 completed、7 rejected；D2H 23,948,427,264 bytes，H2D 52,479,885,312 bytes；
- 359 次非零物理迁移；
- 峰值 HBM 16,106,029,056 bytes，峰值 Host KV 24,886,050,816 bytes；
- HBM mirror 始终为 allocator 子集，Host residency 与 page index 一致；
- 峰值 engine-locked KV 15,361,769,472 bytes；
- 100/500 ms locked-but-not-served 峰值分别为 12,385,714,176 和 11,566,350,336 bytes。

H2D 仍显著高于 D2H，本轮只能证明迁移 bookkeeping 正确，不能证明策略改善 JCT 或吞吐。

## 6. 后续修复顺序

1. 为 ordinary-waiting restore 增加 current-Radix handle/owner rebind；
2. 对 stale/missing physical extent 实现显式 native rematch、recompute 或 drop fallback；
3. 修复 debt barrier 的 work-conserving fallback，禁止 `0 running + waiting`；
4. 对不变 blocker 做 event gate 和指数 backoff；
5. 修复两阶段 shutdown；
6. CPU 确定性测试通过后，才重新执行一次相同 w4 gate。

## 7. 实验后的修复状态

2026-07-29 已完成第 1、2 项的局部修复：

- ordinary-waiting debt 会以 waiting request 的当前 Radix 路径重新解析 live handle/generation，并
  原子替换该 context 的 owner 集合；
- 若重绑后仍无法生成 physical preview，则在同一 `RestoreLease` 中预留 native admission 所需容量，
  由 SGLang HiCache 原生 load-back 或 raw-prompt prefill 完成恢复；
- fallback 只适用于 ordinary waiting，不适用于 running retraction；native `NO_TOKEN` 仍必须重新获取
  reservation，因此该路径不会绕过 allocator 容量约束；
- 审计新增 `restore_obligation_path_rebound`、`restore_obligation_native_fallback_ready` 和终态
  `native_admission_fallback` 字段。

focused restore/admission/retraction CPU 回归为 `123 passed`，全量 CPU 回归为
`437 passed, 8 skipped`。本报告中的 GPU 结果仍保持原始失败结论。修复后的固定 w4 复验已完成：
49/49 restore obligation 全部 `SATISFIED`，0 `physical_preview_unavailable`，但 4/4 workflow 仍因
业务层绝对 deadline 超时而未通过 clean-completion gate。见
`docs/experiments/beliefkv_p5e_restore_rebind_w4_2026-07-29_zh.md`。

产物：

```text
experiments/raw/p5e_restore_lease_w4/20260729T080937Z/experiment_outcome.json
experiments/raw/p5e_restore_lease_w4/20260729T080937Z/transfer_validation.json
experiments/raw/p5e_restore_lease_w4/20260729T080937Z/kv_transfer_timeline.html
experiments/raw/p5e_restore_lease_w4/20260729T080937Z/kv_transfer_timeline.json
```
