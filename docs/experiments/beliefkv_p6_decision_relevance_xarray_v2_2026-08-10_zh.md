# BeliefKV P6 Decision-Relevance Confirmation v2

日期：2026-08-10

## 结论

修正动作语义并冻结新 source fingerprint 后，预声明的 Xarray w8 批次首次在独立 trace 上确认了
morphology veto：byte-only 会选择 5 次 `PREPARE_HOST`，extent-count-aware 仅选择 2 次；严格配对后
有 3 次 byte-only eligible / shape-aware rejected，来自 3 个 context-physical-shape key。

因此 M5 的 decision relevance 已获得新的固定批次支持，但 M6 尚未完成。现有在线
`byte-only + canary_limit=1` 会先执行一笔两种模型共同接受的动作，不能验证 veto。下一步必须只在
同一候选满足 byte-only 接受且 shape-aware 拒绝时授权唯一一笔 treatment。

## 冻结契约

- manifest：`configs/p6/decision_relevance_v2/manifest.json`；
- manifest digest：`ea3c576535648a4247ec62de9f191aac01b4ebfeacccec57f4f5102ba14e2811`；
- source fingerprint：`c389c0f6c4a7f7e54d50bee646513e998447ebf6e797fc05b88880e23102b769`；
- workload：8 个预声明且未参与 predictor fit 的 Xarray 任务，单批 concurrency 8；
- serving：P5 observed + P6 read-only risk shadow，预测动作全部关闭；
- predictor、GPU service 和 transfer service artifact 均由 manifest 固定；
- 不根据运行结果增加任务或修改阈值。

运行目录：
`experiments/characterization/p6_decision_relevance_xarray_v2/run1`。

## 在线采集

- 7/8 workflow 完成；总耗时 3,884.77 s；
- 568 次业务 LLM call、681 次工具调用、4 个动态 subagent；
- 峰值 7 个 running request、2 个 queued request；KV token usage 峰值 62%；
- 497 个冻结 policy snapshot；
- 5/8 workflow measurement-valid，4/8 native-agent JCT eligible；3 个 completed workflow
  含至少一次 guard intervention，其中 `xarray-4695` 的系统测量仍有效，但不用于纯自然 agent JCT，
  其余 guard-censored 路径只保留干预前的局部决策点；
- `pydata__xarray-7233` 在任何 agent event、LLM 或工具调用前发生 120 s `docker run` timeout，
  标记为 harness-censored，不进入策略统计；
- 无 scheduler exception、OOM、execution timeout 或 runtime-control degradation；
- shutdown ACK 完成，0 running/0 waiting，无未决 transaction、lease 或 restore obligation。

客户端退出码为 1，原因仅是严格的 8/8 system-JCT gate 被 pre-agent harness failure 拉低，不代表
SGLang 或 JointPlan 失败。关闭脚本在收到 ACK 后等待进程组退出超时，但进程随后已退出，GPU0 无残留。

## 配对 Replay

产物：

- `replay/byte_only.jsonl`；
- `replay/morphology_aware.jsonl`；
- `replay/comparison.json`。

| 指标 | byte-only | extent-count-aware |
|---|---:|---:|
| evaluated snapshots | 440 | 440 |
| paired PREPARE candidates | 429 | 429 |
| positive-benefit PREPARE | 19 | 9 |
| eligible PREPARE | 5 | 2 |
| selected PREPARE | 5 | 2 |
| recourse scenarios | 321 | 195 |

严格配对结果：

- timing estimate changed：407 candidates；
- feasibility reason changed：54 candidates；
- candidate eligibility / selected action changed：3；
- promotion：0；
- veto：3，来自 3 个 context-physical-shape key；
- 推荐验证 arm：`byte_only_veto_treatment`。

3 个 veto 均属于 `pydata__xarray-4695` parent context。byte-only 对其 D2H 估计约 491.53 ms；
extent-count-aware 估计约 788.15--1,053.53 ms，并因 CVaR、期望收益或 recourse-after-stall
约束拒绝动作。这证明 extent count 已改变最终策略，而不只是改变时延数字。

## 证据边界

- veto 来自一个 workflow 的三个物理 shape，不能按三个独立 workload 样本解释 prevalence；
- predictor 与 transfer model 仍是 development artifact；
- 本轮只读采集没有执行预测性 D2H，尚未证明实际避免的 stall 或 JCT 收益；
- guard-censored workflow 只贡献干预前快照，不能贡献完整轨迹/JCT；
- harness-censored workflow 不贡献任何 agent 或策略证据。

## 下一步

1. 增加 paired veto authority gate：同一候选必须 byte-only eligible 且 shape-aware rejected；
2. treatment 仍只允许全局一笔 `PREPARE_HOST`，safe point 重新物化并验证完整因果/物理证书；
3. 冻结相同 workload、arrival、source 和 artifact 的 byte-only veto treatment 与 P5 observed control；
4. treatment 验证 intent、rematerialization、D2H、ACK、pressure-time reclaim 与 parent reentry 全链路；
5. control 验证同一机会不执行预测传输，并比较 admission stall、unhidden decode stall 与 JCT；
6. 若自然在线轨迹不再产生配对 veto，则报告 `no_natural_veto`，不强制 intent 或降低门槛。
