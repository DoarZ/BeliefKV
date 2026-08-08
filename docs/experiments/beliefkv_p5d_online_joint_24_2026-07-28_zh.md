# BeliefKV P5D Online JointPlan 24-Workflow GPU Gate

日期：2026-07-28
结论：**负结果；物理数据面通过，但 P5 correctness/liveness gate 未通过。**

## 1. 实验配置

原始目录：

```text
experiments/raw/p5d_online_joint_24/20260727T134301Z/
```

关键配置：

```text
GPU                         NVIDIA RTX 6000 Ada，device 1，TP=1
模型                        Qwen3-Coder-30B-A3B-Instruct-FP8
SGLang                      仓库固定版本，HiCache write-back
GPU KV pool                 163,840 tokens，约 15 GiB
Host HiCache                96 GB
max_running_requests        32
context_length              262,144
joint_policy_enabled        true
observed admission          true
running retraction          true
prediction                  false
workflow                    24 个 SWE-bench Verified/SymPy
arrival                     8 个一批，批间隔 20 秒
mode                        16 mixed + 8 cyclic
subagent fan-out            mixed workflow 运行时选择 2--4
request timeout             900 秒
activation wall clock       1,800 秒
```

本轮只运行一次，没有参数 sweep 或失败后重跑。

## 2. Workload 结果

| 指标 | 结果 |
|---|---:|
| workflow result | 24 / 24 |
| semantic complete | 0 / 24 |
| clean JCT eligible | 0 / 24 |
| `APITimeoutError` | 24 / 24 |
| 平均 workflow duration | 1,749.03 s |
| LLM request | 552 |
| tool call | 474 |
| 创建的 dynamic subagent | 59 |
| runtime agent event | 2,261 |
| server request started / finished | 504 / 485 |
| `/abort_request` | 67 |

该 workload 确实形成了多工具、多轮和 subagent 压力，但没有形成正常 RETURN/JOIN 的 clean
轨迹。manifest 中 `dynamic_subagent_workflows=0` 不是没有创建 child，而是 59 个 child 所在的
workflow 均未满足正常返回和 trace-valid 条件。cyclic workflow 也没有完成可计入的 peer
reactivation。因此本轮不能报告 workflow JCT、fairness 或 action-unlock 收益。

`APITimeoutError` 的直接边界是客户端从请求提交起计算的 900 秒 cap。高压阶段 queue 一度超过
50，长 context request 的 queue + prefill + decode 超过该 cap；SGLang 仍持续完成其他请求，
并非服务整体挂死。activation deadline 到期后取消成功传播到 `/abort_request`。

## 3. 压力与数据面正确性

| 指标 | 结果 |
|---|---:|
| peak HBM KV | 16,105,734,144 / 16,106,127,360 bytes |
| peak Host KV | 32,913,457,152 bytes |
| locked GPU snapshot ratio | 85.30% |
| command dispatch / ACK | 995 / 995 |
| missing/orphan ACK | 0 / 0 |
| timestamp/byte-bound violation | 0 / 0 |
| HBM mirror exceeds allocator | 0 |
| Host/PageIndex mismatch | 0 |
| identical failed/zero-byte retry | 0 / 0 |

统一 DMA telemetry 共 554 条，其中 D2H/H2D 为 361/193，实际字节为
36,008,558,592/35,624,976,384。534 条完成，20 条被物理层安全拒绝。transfer retry guard
没有出现 retry storm；物理 reclaim realization 为 95.91%。因此 P2 数据面和 ACK 正确性在
P5D 高压路径下仍然成立。

## 4. Online JointPlan 覆盖

| 指标 | 结果 |
|---|---:|
| online plan published | 1,816 |
| shadow fallback | 225 |
| stale plan | 683 |
| partial plan | 859 |
| would-apply plan | 1,189 |
| planning budget exceeded | 376 |
| admission epoch start / finish | 2,770 / 2,770 |

Online residency 事务共 52 笔，52 笔均进入 terminal：

```text
action                 PREFETCH_GPU=48，COMMIT_CPU=4
status                 completed=50，rejected=2
completed actual bytes 9,971,662,848
reject reason          extent_mutated=2
```

两个 reject 都发生在 D2H 提交前 extent 失去 required clean copy，物理层 fail closed；没有错误
residency commit，也没有重复重试。restore ACK dependency 和多 consumer restore 路径得到真实覆盖。

## 5. Running Retraction

| 指标 | 结果 |
|---|---:|
| planned | 12 |
| completed | 11 |
| failed | 1 |
| completed explicit reclaim | 12,886,867,968 bytes |
| completed explicit transfer | 12,477,530,112 bytes |
| replacement request | 15 |

失败事务已经迁移并释放约 430 MB，但 allocator absolute startup headroom 仍不足，随后没有第二个
可行 physical bundle，最终以 `no_feasible_physical_reclaim_bundle` 结束。该失败没有伪造
replacement admission。

更关键的负结果是：12 个 retraction plan 中只有 2 个带 `source_joint_plan_id`，其余 10 个在
没有 fresh online plan 时进入独立 observed liveness fallback。这保住了部分运行活性，但违反了
“JointPlan 是 Execution/Admission/Residency 唯一策略源”的 P5 设计目标。因此当前 reactive
fallback 边界仍然过宽，不能宣称已经闭合 unified JointPlan。

## 6. 控制面与关机路径

本轮 `runtime_audit.jsonl` 达到 3.5 GiB。bounded bundle preview 不能约束包含大 request/extent
集合的 retraction barrier 和 JointPlan 事件，当前 audit 仍不适合作为性能实验配置。

SIGINT 停止前端后，scheduler CUDA 子进程短暂残留，随后自然退出并释放 GPU；detokenizer 在
阻塞 ZMQ receive 中记录了 `KeyboardInterrupt`。更严重的是 scheduler 没有落盘以下 summary：

```text
controller_timing_summary
online_joint_control_summary
running_retraction_summary
runtime_shutdown
```

因此本轮无法验证 controller critical-path P99、unresolved online transaction 和 shutdown orphan
gate。迁移完整性可由逐 command 原始事件重建，但 summary 缺失本身仍是关机可靠性缺陷。

## 7. Gate 判定

| Gate | 判定 | 原因 |
|---|---|---|
| Physical transfer correctness | PASS | ACK、allocator、Host/PageIndex 和 retry integrity 均通过 |
| Online residency transaction | PARTIAL PASS | 52/52 terminal，50 completed；存在 2 个安全 reject |
| Unified JointPlan authority | FAIL | 10/12 retraction 来自无 plan fallback |
| Workflow liveness | FAIL | 24/24 API timeout，0 clean completion |
| Cyclic/multi-consumer semantic coverage | FAIL | 无可计入 peer reactivation/normal RETURN-JOIN workflow |
| Control-plane overhead | FAIL/UNKNOWN | 376 budget fallback、3.5 GiB audit、shutdown timing summary 缺失 |

当前不能进入 P5.5 action-unlock oracle，更不能进入 P6 predictor。下一步应先：

1. 将无 fresh plan 时的同步 observed fallback 也编译为带 plan id 的 bounded fallback JointPlan，
   retraction 不得绕过统一事务接口；
2. 分离 queue-wait deadline、single-request execution cap 和 activation deadline，确保 normal gate
   workload 能完成，同时继续保留取消传播；
3. 降低 stale/partial/budget fallback，并记录 actionable coverage 与 critical-path P99；
4. 对含大集合的 audit 做采样/摘要，避免 30 分钟产生 GiB 级日志；
5. 将 scheduler summary flush 接入明确的 shutdown IPC/信号路径，而不是只依赖进程 `atexit`；
6. 修复后使用能正常 RETURN/JOIN/REACTIVATE 的 blocking、cyclic-peer、mixed 固定 trace 重新
   执行 P5 gate。

## 8. 产物

- workload manifest：`experiments/raw/p5d_online_joint_24/20260727T134301Z/workloads/manifest.json`
- runtime audit：`experiments/raw/p5d_online_joint_24/20260727T134301Z/server/runtime_audit.jsonl`
- DMA telemetry：`experiments/raw/p5d_online_joint_24/20260727T134301Z/server/transfer_telemetry.jsonl`
- transfer validation：`experiments/raw/p5d_online_joint_24/20260727T134301Z/transfer_validation.json`
- KV timeline：`experiments/raw/p5d_online_joint_24/20260727T134301Z/kv_transfer_timeline.html`
