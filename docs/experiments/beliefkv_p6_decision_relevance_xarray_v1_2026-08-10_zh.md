# BeliefKV P6 Decision-Relevance 与自然 Canary

日期：2026-08-10

## 结论

预先冻结的 Xarray w8 development batch 已完成。修正 `PREPARE_HOST` 的动作语义后，
extent-count-aware 与 byte-only 模型在离线 replay 中产生了双向决策变化；但使用同一任务集合的
一次 autonomous 在线重跑没有自然复现任何 `PREPARE_HOST`。因此当前结论是：

- morphology cost 已达到 decision relevance，不再只是 timing sensitivity；
- predictive overlay 与 P5 fallback 在一次完整 w8 长跑中保持正确；
- 单次 autonomous rollout 未复现离线候选，在线动作和端到端收益仍未证明；
- 不允许通过强制 intent、降低门槛或继续筛选任务来制造自然正例。

## 冻结契约

manifest：`configs/p6/decision_relevance_v1/manifest.json`。

- workload：`p6-016-train-mixed-r0`，8 个未参与 predictor fit 的 Xarray 任务；
- predictor：`frontier_belief_mvp_v6_calibrated_dev.json`；
- GPU service：`gpu_service_curve_cluster_cal_qwen3coder30b_rtx6000ada_20260804T060824Z.json`；
- transfer service：`transfer_service_morphology_gpu0_dev_v2.json`；
- arrival：单批 w8，active window 8；
- selection：采集后不增加任务，不按 promotion/veto 结果筛选 trace。

该 batch 是 development-only。它没有进入 predictor fit，但不能在本轮修复后继续充当独立确认集。

## 运行结果

characterization 目录：
`experiments/characterization/p6_decision_relevance_xarray_v1/run1`。

- 8/8 workflow 正常结束；
- 617 次业务 LLM call，1,746 次工具调用，6 个动态 subagent；
- 8/8 system-JCT eligible，4/8 native-agent JCT eligible，5/8 measurement valid；
- 无 scheduler exception，shutdown 时无未决 transaction/obligation。

原始 replay 将 `future_hbm_chance_constraint` 错误施加到 `PREPARE_HOST`。该动作只创建 Host
shadow、保留 GPU copy，并不会增加 HBM；baseline 未来 HBM overflow 应作为 recourse 诊断，
不能作为该动作的确定性拒绝条件。因此原始“0 decision flip”结论无效。

## 修正后 Replay

产物：

- `byte_only_corrected_replay.jsonl`；
- `morphology_aware_corrected_replay.jsonl`；
- `corrected_replay_comparison.json`。

| 指标 | 结果 |
|---|---:|
| paired snapshots | 938 |
| paired PREPARE candidates | 842 |
| byte-only eligible | 25 |
| extent-count-aware eligible | 11 |
| timing estimate changed | 577 |
| feasibility reason changed | 156 |
| eligibility / selected-action changed | 20 |
| shape promotion | 3 |
| shape veto | 17 |
| promotion / veto context-shape keys | 3 / 4 |

`shape_action_gate`、`shape_veto_gate` 和 `selected_action_gate` 均为 true。veto 数量更多，说明
当前 morphology 模型的主要价值可能是避免 byte-only 策略执行无法隐藏的高碎片 D2H，而不是
产生更多迁移。

这组 corrected replay 是发现动作语义 bug 后的 post-hoc development evidence，不能作为独立
confirmatory result。正式结论必须冻结修正后的 source fingerprint 后在新批次上验证。

## 自然在线 Canary

目录：`experiments/canary/p6_morphology_promotion_xarray_v1/run1`。

- 8/8 workflow 完成，耗时 6,832.59 s；
- 750 次业务 LLM call，9 个动态 subagent；
- 7/8 native-agent JCT eligible，5/8 measurement valid；
- 无 OOM、execution timeout、scheduler exception 或 runtime-control degradation；
- shutdown ACK 完整，所有 transaction/obligation 守恒；
- `natural_prepare_count=0`，无孤儿 predictive command。

本轮 RCCG 达到 79 个 invocation/context，而 characterization trace 的执行轨迹不同。自主 agent
的 spawn、工具选择和上下文演化具有随机性，所以离线 promotion 状态没有在一次重跑中复现。
`canary_analysis.json` 的状态为 `no_natural_prepare`；它不是失败事务，也不能用于评估 transfer
收益。

## 下一阶段

1. 在线配置显式区分 `morphology-aware` 与 `byte-only`，planner 和 safe-point 必须使用同一模式；
2. byte-only 只允许全局一笔 `PREPARE_HOST` treatment，避免成为默认线上策略；
3. 冻结修正后的 predictor、service artifact、policy source 和新的预声明任务批次；
4. 在冻结批次上估计 promotion/veto 的 rollout-level 出现概率；
5. promotion 自然出现时运行 shape-aware 单动作 arm；veto 自然出现时运行 byte-only treatment，
   并以 morphology-aware/P5 不执行作为 control；
6. 若新批次仍无法自然复现 action flip，则 morphology 降级为 transfer cost 辅助模型。

在完成配对 treatment/control 前，不报告 P6 JCT、吞吐或 admission stall 改善。
