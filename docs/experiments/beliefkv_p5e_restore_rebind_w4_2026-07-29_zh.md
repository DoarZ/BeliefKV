# BeliefKV P5E ordinary-restore rebind fixed w4 GPU gate

日期：2026-07-29

状态：**restore liveness 修复通过；完整 clean-completion gate 失败，不可用于性能比较。**

## 1. 实验配置

本轮只执行一次，与上一轮 `p5e_restore_lease_w4` 保持相同固定 trace 和参数：

- Qwen3-Coder-30B-A3B-Instruct-FP8，SGLang 0.5.2rc1，TP=1，仅 GPU 0；
- KV pool 163,840 token，context 262,144，`MEM_FRACTION_STATIC=0.952`；
- Host HiCache 96 GB；
- 4 个固定 SWE-bench Verified SymPy workflow，3 mixed、1 cyclic；
- 每个 mixed workflow 首轮动态选择 2--4 个 child；
- request execution timeout 900 秒，workflow 绝对 deadline 1,800 秒；
- online JointPlan、observed admission、running retraction、RestoreLease 全部开启。

原始数据：

```text
experiments/raw/p5e_restore_rebind_w4/20260729T094030Z/
```

## 2. Gate 结果

| Gate | 结果 | 证据 |
| --- | --- | --- |
| 固定 trace、单次运行 | PASS | 4 个固定实例，未重跑 |
| 高压 KV 覆盖 | PASS | 峰值 HBM KV 16.11 GB；运行期 token usage 多次达到 95%--96% |
| Restore liveness | **PASS** | 49/49 obligation `SATISFIED` |
| Ordinary-waiting restore | **PASS** | 17/17 `SATISFIED`，0 `physical_preview_unavailable` |
| Command/ACK 完整性 | PASS | 170 dispatch、170 ACK，0 missing/orphan/order violation |
| Retry guard | PASS | 0 retry-without-release，0 identical zero-byte retry |
| Clean completion | **FAIL** | 0/4 `clean_jct_eligible`，4/4 `workflow_timeout` |
| Host/page-index 一致性 | **FAIL** | 11,715 个采样点中有 2 个瞬时 mismatch |
| Shutdown summary | **FAIL** | `shutdown_state=preparing`，detokenizer 被 SIGINT 中断 |

## 3. 修复目标验证

上一轮失败的 `restore-18` 因逻辑 context 与当前 Radix physical ownership 脱节，持续返回
`physical_preview_unavailable` 并形成 `0 running + 7 waiting`。本轮结果为：

- 32 笔 running-retraction restore 和 17 笔 ordinary-waiting restore 全部恢复 GPU service；
- ordinary restore 最大等待 59.41 秒，不再出现永久 parked debt；
- 160 次 `restore_obligation_path_rebound` 真实覆盖了新路径；
- 没有触发 `restore_obligation_native_fallback_ready`，说明本轮重绑后均能生成 physical preview；
- 50 次 RestoreLease grant、49 次 admission commit、1 次 native rejection 后正常重试；
- workload 结束前无 active obligation、lease、transfer 或 retraction transaction。

因此可以确认：**current-request-path ownership rebind 修复了上一轮的 restore liveness bug。**
但 160 次重绑说明 ownership 会被后续物理同步反复改写，仍需降低控制面 churn。

## 4. 为什么完整 Gate 仍失败

四个 workflow 均运行到约 1,803 秒并触发绝对 deadline：

- cyclic：30 LLM、29 tool，`activation_wall_clock_exhausted`；
- mixed-000：4/4 child RETURN 且 JOIN 满足，但 parent 继续运行，最终
  `completion_budget_exhausted`；
- mixed-001：前两个 child RETURN 后，约 1,301 秒又动态 spawn 第三个 child，该 child 未 RETURN，
  最终 1 cancel、1 join timeout；
- mixed-003：4/4 child RETURN 且 JOIN 满足，但 parent 未在 deadline 前结束。

整轮共 387 次 LLM、372 次工具调用、11 个动态 child、10 RETURN。deadline 末尾有 3 个
`APITimeoutError`，它们由 workflow 剩余时间耗尽产生；服务端没有 request execution timeout、OOM
或 admission stall。当前首要失败已经从 KV restore 转移为 agent runtime 的语义终止和 late-spawn
预算管理。

## 5. 数据面与残余正确性问题

- 575 条 DMA telemetry：127 条 BeliefKV command、448 条 native HiCache callback；
- D2H 22,827,368,448 bytes，H2D 166,141,231,104 bytes；
- 峰值 engine-locked KV 15,387,721,728 bytes；
- 100/500 ms locked-but-not-served 峰值分别为 14,196,670,464 和 13,295,222,784 bytes；
- Host/page-index mismatch 仅出现 2 个瞬时采样，差值分别为 92,405,760 和 56,819,712 bytes；
- mixed-000 deadline 收口时出现 1 次 `missing request_physical_start checkpoint`；
- audit 为 96,844,909 bytes，仍有 11,567 次 physical-commit budget exceeded。

## 6. 下一步

1. 修复 agent runtime：parent 在 child/JOIN 已完成后必须产生结构化终态，不能继续循环到 deadline；
2. late spawn 必须检查 remaining workflow budget，并保证 child cancel 后 JOIN 有确定终态；
3. 将 ownership rebind 改为 generation/path 变化驱动，避免 160 次反复重绑；
4. 定位两个 Host mismatch 和一个 physical-start checkpoint 缺口；
5. 修复两阶段 shutdown 后，再运行固定 w4 clean-completion gate。

产物：

```text
experiments/raw/p5e_restore_rebind_w4/20260729T094030Z/experiment_outcome.json
experiments/raw/p5e_restore_rebind_w4/20260729T094030Z/transfer_validation.json
experiments/raw/p5e_restore_rebind_w4/20260729T094030Z/kv_transfer_timeline.html
experiments/raw/p5e_restore_rebind_w4/20260729T094030Z/kv_transfer_timeline.json
```
