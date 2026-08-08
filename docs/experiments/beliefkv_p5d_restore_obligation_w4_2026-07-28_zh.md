# BeliefKV P5D restore obligation w4 GPU 验证

日期：2026-07-28

状态：restore liveness 子门槛通过；完整 P5 correctness/liveness 门槛未通过；不可用于性能比较。

## 1. 实验配置

- 模型：`Qwen3-Coder-30B-A3B-Instruct-FP8`
- SGLang：`0.5.2rc1`，TP=1，仅使用 GPU 0
- GPU KV pool：163,840 token，15 GiB
- Host HiCache：96 GB，`write_back`
- `MEM_FRACTION_STATIC=0.952`
- online JointPlan、observed admission、running retraction 和 restore obligation 均开启
- workload：与上一轮 P5D w4 相同的 4 个 SWE-bench Verified SymPy workflow
- 单请求 execution timeout：900 秒
- root activation wall-clock deadline：1,800 秒

原始目录：

```text
experiments/raw/p5d_restore_obligation_w4/20260728T081556Z/
```

GPU 1 在运行期间曾被外部作业短暂占用。因此本轮只用于 correctness/liveness 验证，不能用于
workflow JCT、吞吐或 GPU 利用率对比。

## 2. Restore obligation 结果

本轮真实触发 19 次 running retraction：

- 19 个 restore obligation 全部进入 `satisfied`；
- 终态原因全部为 `gpu_service_resumed`；
- restore 总等待时间 p50 为 814.15 ms，最大值为 1,196.72 ms；
- 17 笔 source D2H 成功，随后完成 H2D prefetch，共恢复 19,029,098,496 bytes；
- 2 笔 source D2H 因 `node became engine-locked before bundle commit` 被安全拒绝，请求保留原 GPU
  copy，并分别在 720.40 ms 和 795.38 ms 后恢复 service；
- 没有 active obligation、orphan command 或未决 transaction；
- 没有出现旧轮 requeue 后 900 秒无 service 的 stranded request。

因此，`D2H terminal -> H2D restore -> ACK -> ticket -> GPU service` 路径在本 trace 中成立。
但 `funding_reclaim_bytes=0`，说明 HBM 不足时的 `evict_for_restore` funding 分支尚未被覆盖，
不能宣称整个 restore state machine 已完成压力验证。

## 3. 迁移正确性

`validate-transfer-telemetry` 通过：

- 61 次 dispatch 对应 61 次 ACK；
- 0 missing ACK、0 orphan ACK、0 duplicate/ordering/timestamp violation；
- 986 条物理 DMA telemetry 中 984 completed、2 rejected；
- D2H 935 次，共 35,796,123,648 bytes；
- H2D 51 次，共 32,949,239,808 bytes；
- HBM mirror 始终是 allocator 的子集，Host page index 与 Host residency 一致；
- retry guard 无相同物理 fingerprint 的重复失败提交。

峰值 HBM 为 16,106,127,360 bytes，峰值 Host KV 为 33,804,386,304 bytes。

## 4. Admission starvation

retraction restore 修复后暴露出未覆盖的 ordinary-waiting restore liveness 问题。两个从未执行过的
child request：

```text
beliefkv:019fa7d1-0b98-7521-8a2f-f11609391a03
beliefkv:019fa7d2-16bb-77d1-8fea-91b9dbb4f60f
```

分别在 waiting queue 中停留约 1,726.29 秒和 1,657.76 秒，直到 activation deadline 触发
`APITimeoutError` 和 `/abort_request`。两者都从未发生 physical start。进一步审计发现，它们的
matched Radix path 在等待期间分别累计 D2H 约 1.95 GB 和 1.15 GB，均没有对应 H2D；实验后期
`online_joint_restore_blocked_count=2`。因此它们不是在等待 tool，也不是单纯被 workflow fairness
过滤，而是普通 waiting request 只有 `WAIT_RESTORE` gate、没有 durable restore obligation，形成
`WAIT_RESTORE -> no ticket -> no native demand-load -> WAIT_RESTORE` 的 liveness 环。

全轮共出现：

- 23,543 个 admission ticket epoch；
- 46,518 次 `no_ticket` skip；
- 23,068 个 `native_batch_size=0` 但仍未发 ticket 的 epoch；
- 22,093 个 `joint_emergency` epoch。

这说明 running retraction 的 restore state machine 当时尚未覆盖“等待期间 prefix 被 HiCache
write-back”的普通 waiting request。异步计划的 `join_changed`/`plan_expired` 使问题更容易暴露，
而 bounded/emergency seed 只保留 restore blocker、不生成 H2D 事务，无法自行打破该循环。

## 5. Workload 终止性

三个 workflow 产生了 `result.json`：

- `mixed-000`：90 LLM、82 tool、4 child 中 3 个 RETURN，1,803.32 秒后 API timeout；
- `mixed-001`：59 LLM、57 tool、3 child 中 2 个 RETURN，1,803.59 秒后 API timeout；
- `cyclic-002`：47 LLM、46 tool，1,838.90 秒后 activation deadline error。

`mixed-003` 的 4 个 child 全部 RETURN 且 JOIN satisfied，但 persistent peer loop 未继承统一的
activation deadline。它运行至 context epoch 66，距首个 submit 约 2,282 秒，仍继续提交请求，
也没有被配置的 48-call stuck guard 终止。为避免无限运行，本 workflow 被人工中止，因此本轮
没有完整 manifest，且 shutdown summary 停在 `preparing`。

## 6. 控制面开销

本轮还记录到：

- `online_joint_physical_commit_budget_exceeded` 29,199 次；
- 其中 15,181 次超过 10 ms，1,761 次超过 100 ms，111 次超过 500 ms；
- 最大 physical commit 时间为 1,990.44 ms；
- runtime audit 达到约 226 MiB。

这仍远高于 1 ms safe-point budget。虽然不影响本轮 restore 正确性结论，但必须在性能实验前
降低 ticket/physical commit 热路径的频率和同步开销。

## 7. 结论与下一步

本轮证明 restore obligation 修复解决了上一轮的 retraction 后 H2D/service 断链，但不能关闭
完整 P5 gate。该 trace 的事后修复已经完成前两项，尚需固定 trace GPU 复验：

1. 已实现：将普通 waiting CPU-prefix 纳入 durable restore obligation，并在 H2D ACK 后生成 liveness ticket；
2. 已实现：restore debt 独立于 JOIN/readset 计划有效性，并让 oldest `TICKET_READY` debt 优先 admission；
3. 让 activation deadline 和 stuck guard 覆盖 persistent peer loop，并取消所有 descendant request；
4. 用确定性测试覆盖 starvation、deadline 传播和 `evict_for_restore` funding；
5. 修复后只复跑一次相同 w4，不扩大到 w8/w12/w24。

产物：

```text
experiments/raw/p5d_restore_obligation_w4/20260728T081556Z/kv_transfer_timeline.html
experiments/raw/p5d_restore_obligation_w4/20260728T081556Z/transfer_validation.json
experiments/raw/p5d_restore_obligation_w4/20260728T081556Z/experiment_outcome.json
```
