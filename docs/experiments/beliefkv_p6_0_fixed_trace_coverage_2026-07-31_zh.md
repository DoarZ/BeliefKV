# BeliefKV P6.0 固定 Trace Coverage

## 目标与结论

P6.0 只验证预测标签能否被可靠观测，不评价 JCT、吞吐或策略收益。旧 P5f trace 证明了
action/reentry 标签可识别，但缺少 GPU service 和条件化 PCIe 标签。本轮在同一组 4 个
SymPy SWE-bench workflow 上只启动了一次 GPU 采集：

- raw：`experiments/raw/p6_0_labels_w4/20260731T064912Z`
- processed：`experiments/processed/p6_0_coverage_20260731/p6_0_labels_w4_20260731T064912Z.json`
- 3 个 mixed subagent workflow，1 个 cyclic peer workflow
- Qwen3-Coder-30B-A3B-Instruct-FP8，单卡，163,840-token KV pool

已完成的 LLM call 能用 native request ID 精确连接 agent event、SGLang event 和 GPU batch；
prefill/decode service 与 PCIe 条件字段达到训练所需的完整性。Exact action boundary 仍不可用，
因此 P6.0 总 gate 未通过，未生成 KV 时间线。

## GPU 标签结果

| 指标 | 结果 |
| --- | ---: |
| Policy LLM call | 171 |
| Completed / censored call | 162 / 9 |
| Completed call native-ID 对齐 | 162/162，100% |
| Completed prompt/cache/output demand | 162/162，100% |
| Runtime-only structured boundary | 162/162，100% |
| Exact incremental action boundary | 0/162，0% |
| Reentry cause | 154/161，95.65% |
| GPU service sample | 2,020 |
| Prefill / decode sample | 174 / 1,846 |
| Request-level service row | 9,401 |
| Service row 字段完整性 | 100% |
| Exact decode token delta | 9,214/9,214，100% |
| 同时具有 prefill/decode 标签的 request | 164/164，100% |

GPU service interval 从 scheduler launch/上一 batch completion 到
`process_batch_result` completion，排除了 waiting、admission 和 HTTP queue wait。batch sample
按 native request ID 展开，不使用 invocation ordinal 推断。

## PCIe 标签结果

共记录 347 次 transfer attempt，其中 340 次进入物理传输，7 次在 preflight 阶段被拒绝。
对 340 次物理传输，以下条件字段覆盖均为 100%：

- direction、bytes、extent/page count、command kind 和 source/target tier；
- Host copy 是否存在、是否 pinned；
- native HiCache 并发流量；
- allocator/API staging、callback observation delay；
- submit-to-complete duration。

其中 69 次显式 BeliefKV transfer 具有 HiCache API submit-to-complete 时间。当前 SGLang callback
没有暴露真实 DMA start，因此 direct DMA duration 仍明确标记为 unavailable，不能用 API 时长
冒充 DMA 时长。

## 观测开销

2,020 个 service sample 的 launch-side observer CPU 开销为：P50 0.067 ms、P95 0.348 ms、
P99 0.676 ms；没有 debug event 被丢弃。由于本轮被有界中止，SGLang 只进入
`shutdown_state=preparing`，未写出 final observer/controller summary，因此 audit-enqueue P99、
scheduler critical-path P99 和 sample-cap summary 不可用。后续 clean collection 必须通过显式
shutdown ACK 保存这些字段。

## Censored 原因

采集在标签已充分覆盖后因 P5 restore liveness 阻塞而停止，不能作为性能实验：

1. `restore-30` 需要恢复约 1.153 GB parent KV。
2. 它先受 device capacity、engine busy 和 node lock 阻塞，随后重复获得 lease，又因
   `restore_h2d_queue_conflict` 回滚。
3. 共发生 963 次 retry；关闭前 obligation 已等待约 414.4 秒，服务端达到
   `0 running / 11 waiting`。
4. 中止时 pending transaction 被清空，但 workflow 没有生成正式 manifest；analyzer 将其显式
   标记为 `collection_status=censored` 和 `experiment_valid_for_performance=false`。

这说明标签链路有效，但不代表 P5 liveness gate 已通过。该问题应在进入在线 P6 策略前单独修复，
不能通过重跑掩盖。

## Gate 判定

1. **通过**：completed-call native identity 与 token demand；runtime GPU interval 只作
   service characterization，不再作为 Frontier remaining-service 标签。
2. **通过**：物理 PCIe operation 的条件字段；真实 DMA start 仍不可用。
3. **未通过**：UnlockHazard，exact incremental action boundary 为 0%。
4. **未通过**：reentry label completeness，当前为 95.65%。
5. **未通过**：完整 observer-overhead gate，censored shutdown 缺少最终 summary。

因此下一步不是训练完整预测器，而是先补 native incremental parser boundary，并修复
`restore_h2d_queue_conflict` 的队首活性；在此之前 P6 online path 保持关闭。

## 旧 Trace 对照

旧 trace `experiments/raw/p5f_fixed_w4/20260730T132313Z` 具有 466 个完整 policy call，
action type 和 reentry cause 为 100%，但 GPU service sample 为 0，且 Host/pinned、native HiCache
并发和 allocator 字段均缺失。本轮补采消除了这些缺口，但没有伪造仍不可观测的 exact action
boundary。
