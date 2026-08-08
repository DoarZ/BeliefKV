# BeliefKV P5E ordinary-waiting restore w4 GPU 验证

日期：2026-07-28

状态：restore correctness/liveness 子门槛通过；workload termination 和完整 P5 gate 未通过；
不可用于性能比较。

## 1. 实验配置

- 模型：`Qwen3-Coder-30B-A3B-Instruct-FP8`
- SGLang：`0.5.2rc1`，TP=1，仅使用 GPU 0
- GPU KV pool：163,840 token，15 GiB
- Host HiCache：96 GB，`write_back`
- `MEM_FRACTION_STATIC=0.952`
- online JointPlan、observed admission、running retraction 和 restore obligation 开启
- workload：固定的 4 个 SWE-bench Verified SymPy workflow，与上一轮 w4 使用相同实例、模式和
  runtime identity namespace
- 单请求 execution timeout：900 秒
- root activation wall-clock deadline：1,800 秒

原始目录：

```text
experiments/raw/p5e_ordinary_waiting_restore_w4/20260728T103439Z/
```

## 2. Restore liveness 结果

本轮创建 72 个 restore obligation：66 个来自 running retraction，6 个来自
`ordinary_waiting_prefix`。结果为：

- 72/72 最终进入 `SATISFIED`，终态原因全部为 `gpu_service_resumed`；
- 停止实验时没有 active obligation、inflight command 或 pending residency/retraction transaction；
- 累计恢复 186,626,801,664 bytes，funding reclaim 为 4,419,256,320 bytes；
- 全部 obligation 的 nearest-rank p50 等待为 2,743.90 ms，最大为 47,918.99 ms；
- 80 次 obligation 子命令中 76 次 completed、4 次 rejected；被拒绝动作没有留下 orphan debt。

6 个普通 waiting debt 全部恢复 GPU service：

| Obligation | Wait (ms) | H2D bytes | Funding bytes |
| --- | ---: | ---: | ---: |
| restore-12 | 8,079.12 | 686,161,920 | 662,372,352 |
| restore-15 | 341.60 | 251,461,632 | 0 |
| restore-17 | 2,400.46 | 911,966,208 | 0 |
| restore-20 | 12,949.51 | 202,113,024 | 0 |
| restore-39 | 2,513.00 | 430,669,824 | 864,583,680 |
| restore-58 | 5,120.28 | 1,626,144,768 | 2,051,997,696 |

上一轮两个 ordinary waiting request 分别等待约 1,726 秒和 1,658 秒且 H2D 为 0。本轮真实覆盖
`CPU-only path detection -> durable obligation -> funding（需要时）-> H2D -> ACK -> ticket -> GPU
service`，没有再出现该类 stranded request。因此 ordinary-waiting restore liveness 修复通过本次
固定 w4 GPU 子门槛。

## 3. 迁移正确性

`validate-transfer-telemetry` 通过：

- 214 次 dispatch 对应 214 次 ACK；
- 0 missing ACK、0 orphan ACK、0 ordering/timestamp violation；
- 1,111 条 DMA telemetry 中 1,107 completed、4 rejected；
- D2H 968 次，共 59,318,894,592 bytes；
- H2D 143 次，共 253,963,468,800 bytes；
- HBM mirror 始终为 allocator 子集，Host residency 与 Host page index 一致；
- retry guard 没有相同失败 fingerprint 的无事件重复提交；
- 峰值 HBM 为 16,106,127,360 bytes，峰值 Host KV 为 54,465,527,808 bytes。

H2D 是整个运行期间 native demand-load 与显式 restore 的累计字节，不是同时驻留量。它显著高于
D2H，说明长上下文阶段存在反复 demand-load，后续应单独分析 useful/wasted restore 与 KV churn；
本轮不能据此得出性能收益结论。

## 4. Workload 与终止性

本轮四个 workflow 共发起 395 次 LLM request、完成 439 次工具调用，负载强度足以触发多次
retraction、ordinary restore 和 funding restore。

- `mixed-000`：112 LLM、108 tool、3 RETURN，`semantic_complete`；
- `mixed-001`：83 LLM、93 tool、1 RETURN，1,803.93 秒后 `APITimeoutError`；
- `mixed-003`：108 LLM、135 tool、3 RETURN，1,803.91 秒后 `APITimeoutError`；
- `cyclic-002`：92 LLM、103 tool、0 RETURN，运行到 context epoch 74 后仍循环提交
  `edit_file`，没有产生 `result.json`。

`cyclic-002` 在 1,800 秒 activation deadline 后仍从 epoch 70 继续提交到 epoch 74，也明显超过
48-call stuck guard。长运行 exec 会话随后以 `CondaError: KeyboardInterrupt` 退出，并连带终止
SGLang；这不是 runner 的正常终态。退出前服务器队列归零、所有 KV 事务均已 terminal，但受控
shutdown summary 未完成，audit 中没有 `runtime_shutdown` 事件。

代码审计确认这不是 SGLang 或 KV restore 的 deadline：当前 `ActivationDeadline` 属于单个
persistent peer thread，并在每次 `backend.invoke()` 进入时重新 `start()`、退出时 `clear()`。
`LangGraphPeerWorkflow.run()` 外层同步 `graph.invoke()` 没有共享的绝对 deadline/cancellation token，
runner 又通过无 timeout 的 `as_completed()` 等待 future。因此 cyclic peer 每进入下一次 activation
都会续期 1,800 秒，48-call guard 的计数和循环状态也局限在 activation 内，不能约束整个
workflow。该问题不否定本轮 restore 路径的物理正确性，但使完整 P5 correctness/liveness gate
保持未通过，也使 workflow JCT、吞吐和性能数据无效。

## 5. 结论

本轮可以支持以下窄结论：

1. ordinary waiting CPU-prefix 已能创建 durable restore obligation；
2. running retraction 与 ordinary waiting debt 都能经过 H2D ACK、liveness ticket 后重新获得 GPU
   service；
3. HBM funding 分支得到真实覆盖；
4. 所有 dispatch/ACK 和 allocator/Host ownership 校验通过，无 orphan transaction。

本轮不能支持“P5 已完成”或任何性能优越性结论。下一步应先把绝对 activation deadline 和 stuck
guard 提升到 workflow runner 的最外层，确保 cyclic/mixed future 被统一取消，并在取消后等待所有
descendant 与 SGLang request ACK；完成确定性 CPU 测试后，再做一次 w4 clean-completion gate。

产物：

```text
experiments/raw/p5e_ordinary_waiting_restore_w4/20260728T103439Z/kv_transfer_timeline.html
experiments/raw/p5e_ordinary_waiting_restore_w4/20260728T103439Z/transfer_validation.json
experiments/raw/p5e_ordinary_waiting_restore_w4/20260728T103439Z/experiment_outcome.json
```
