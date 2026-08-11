# BeliefKV P6 Morphology M1--M6 开发与 Gate 结果

日期：2026-08-10

## 结论

M1--M6 的机制开发已完成。后续冻结 Xarray w8 characterization 暴露了一个动作语义 bug：
`PREPARE_HOST` 被错误施加 future-HBM gate。修正后 replay 出现 3 次 promotion 和 17 次 veto，
decision-relevance gate 已通过；一次自然在线 w8 canary 没有复现 `PREPARE_HOST`。当前证据支持
“extent count 会改变 JointPlan 决策”，仍不支持“预测性 PREPARE_HOST 已产生在线收益”。

## M1：Agent Morphology Audit

输入为既有真实 agent 固定 trace，同时按 physical generation、稳定 parked episode 和 context
三个层次聚合。

- 57 个候选 epoch，5 个 context；
- 51 个 physical generation，只用于反映 context 内的 Radix 演化，不能视为独立样本；
- 按 `context + context_epoch + parked state/join + stable morphology tuple` 聚合后为 13 个 episode；
- 其中 5 个高 extent-count WAIT_JOIN/WAIT_TOOL episode 分布在 3/5 个 context；
- 额外记录 extent-size min/P50/max 和 small-extent ratio；当前 action certificate 不含父边，
  closure depth 明确标记 unavailable。

产物：`experiments/analysis/p6_agent_morphology_audit.json`。

## M2：Extent-count-aware Service Curve 首版

使用 GPU0 同字节量 fragmentation matrix 构建 schema-v2 development artifact：

- 2,659,221,504 bytes、7 extents：D2H mean 185.69 ms；
- 2,659,221,504 bytes、106 extents：D2H mean 765.17 ms；
- 相差 4.12 倍；
- exact size/count 不足时只允许二维邻桶，extent-count bucket 距离最多 1；
- 超出支持域返回 `shape_unsupported_static_fallback`，不能跨 extent-count 乐观外推；
- 模型尚未以 extent-size distribution、small-extent ratio 或 closure depth 为条件，因此不能称为
  完整 morphology-aware model。

产物：`experiments/models/transfer_service_morphology_gpu0_dev_v2.json`。该 artifact 明确标记
`formal_crossover_complete=false`，不能作为双 GPU 泛化证据。

## M3--M4：JointPlan 接入

- PREPARE projection 绑定实际 closure root、extent IDs、copy bytes、transfer extent count、
  shape fingerprint、exclusive/cross-context bytes；
- PolicyInput 携带 immutable compact service-curve snapshot；
- ScenarioRiskPlanner 计算 byte-only cost、extent-count-aware Q90 transfer/stall、morphology debt、
  pressure/reentry deadline 和 morphology slack；
- 只有 recourse 大于 stall、future HBM 可行、shape supported 且 slack 为正时才生成 intent；
- PredictiveIntent 携带 shape/count/max transfer/max stall envelope；
- safe point 重新物化 live bundle、重新查询服务曲线，shape OOD、cost 超包络或 slack 过期时
  fail closed 到 observed P5；
- 物理提交仍复用 P5 ActionGroup、transaction、ACK 和 shutdown 守恒。

## M5：固定 Trace Replay

输入：`p6_predictive_overlay_fixed/20260809T105733Z` 的 126 个稀疏 snapshot，其中 124 个配对
参与两种模型比较。

| 指标 | Byte-only | Extent-count-aware |
|---|---:|---:|
| PREPARE candidate epochs | 229 | 229 |
| context-physical-shape keys（非独立样本） | 58 | 58 |
| shape-supported | 229 | 87 |
| shape-supported opportunities | 58 | 24 |
| positive PREPARE | 0 | 0 |
| eligible PREPARE | 0 | 0 |
| complete recourse scenarios | 18 | 4 |

四级比较结果：87 个 candidate epoch 的 timing estimate 改变，147 个的 feasibility reason 改变；
同一配对候选上的 promotion 和 veto 均为 0 次，selected action 也为 0 次变化。所有 124 个
snapshot 在两种模型下都选择
`observed_baseline`。因此 `timing_sensitivity_gate=true`，但 `decision_relevance_gate=false`，M5
没有达到进入在线 canary 的原始要求。

产物：

- `experiments/analysis/p6_m5_byte_only_replay.jsonl`；
- `experiments/analysis/p6_m5_morphology_aware_replay.jsonl`；
- `experiments/analysis/p6_m5_replay_comparison.json`。

## M6：单动作 Canary Gate

已实现：

- `predictive_prepare_host_canary_limit=1` 的 run-level 硬上限；
- commit -> queue -> transfer telemetry -> ACK -> terminal 的五段归因字段与 ID/bytes/count/shape 守恒；
- `scripts/analyze_predictive_prepare_canary.py`，用于报告预测/实际 transfer、slack、stall 和事务状态。

本轮 `shape_action_gate=false`、`shape_veto_gate=false`、`selected_action_gate=false`，因此
`decision_relevance_gate=false`；兼容字段 `online_canary_gate=false`，且只代表 promotion 是否允许
shape-aware PREPARE canary。
两种模型均无 positive/eligible PREPARE。按照预设技术路线，
不强制注入动作、不降低风险阈值，也不运行无法产生自然动作的 GPU 长跑。状态记录在
`experiments/analysis/p6_m6_canary_gate.json`。

## 下一判据

下一阶段不是继续筛选 trace 或直接运行 GPU canary，而是 Decision-Relevance Characterization：

1. 在采集前冻结 predictor、service artifact、风险阈值、workload manifest 和 arrival policy；
2. 使用预先确定的新任务集合，逐配对候选报告 promotion、veto、selected-action change 和 shape support；
3. promotion 出现时只运行一笔 shape-aware PREPARE canary；
4. veto 出现时运行 byte-only treatment，并以 shape-aware/P5 不执行为 control；
5. 固定批次仍只有 timing/reason change 时，将 morphology 降级为辅助成本模型。

禁止不断增加或筛选 trace 直到碰到正例。在完成上述验证前，P6 在线收益保持“未证明”。

## 后续 Decision-Relevance 更新

完整记录见
`docs/experiments/beliefkv_p6_decision_relevance_xarray_v1_2026-08-10_zh.md`。修正后的 938 个
paired snapshot 包含 842 个 PREPARE candidate、25 个 byte-only eligible、11 个
extent-count-aware eligible；20 个候选改变 eligibility 和 selected action，其中 promotion 3、
veto 17。随后一次 8/8 正常完成的 autonomous canary 未自然发布 PREPARE，状态为
`no_natural_prepare`。因此 M5 decision relevance 已成立，M6 mechanism/liveness 已成立，但 M6
自然动作可复现性和端到端收益仍未通过。
