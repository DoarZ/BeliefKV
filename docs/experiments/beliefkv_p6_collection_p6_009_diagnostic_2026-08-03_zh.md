# P6 `p6-009` 首批采集诊断

## 结论

本轮是 development diagnostic，不是正式训练数据。原始目录为
`experiments/raw/p6_agent_semantics_v1/p6-009-train-mixed-r0/20260803T065827Z`，并已写入
`PILOT_INVALID.json`。对应导出位于
`experiments/processed/p6_agent_semantics_v1/development_diagnostic_p6-009_20260803`。

## 观测结果

- 8/8 workflow 完成，共 820 次 LLM、1,108 次工具调用和 7 次动态 subagent 创建。
- 导出 12,796 个事件采样 decision point、37,272 条 GPU service sample、1,108 条 external wait
  和 2,226 条 PCIe operation。
- runtime-only boundary coverage 为 100%；exact incremental boundary 为 0%，第一版只声明
  LLM-result boundary 和 remaining decode demand，不声明 early dispatch；runtime batch elapsed
  仅用于独立 service model 的外部验证。
- reentry label 为 759/787；147 条 censor 全部有显式 `duplicate_suppressed` 原因。

## 无效原因

默认 `completion_repair_attempts=2` 在模型已经产生有效 `WorkflowCompletion` 后，又根据 SWE
correctness gate 启动最多两轮独立 LLM repair。这会把 benchmark repair 轨迹混入 agent 语义轨迹，
显著扭曲调用数、时长和 frontier 状态分布。另有两条 runtime event 使用 1 秒 ACK 窗口时未及时
确认，故只有 6/8 workflow 满足 system JCT gate。

## 已实施修复

正式 P6 collector 现在：

1. 使用 model-terminal semantics，仓库 correctness 仅离线记录；
2. 固定 `completion_gate_enabled=false` 和 `completion_repair_attempts=0`；
3. runtime event ACK 默认 10 秒、重试 3 次；
4. batch 前后计算 runtime/control source fingerprint；
5. 将最终 `training_eligible` 写入 collection contract；
6. dataset/model loader 对 invalid、gate failure、重复 run 和重复 decision fail closed。

本轮只用于验证数据 schema、coverage、训练和序列化管线，不参与正式模型选择、校准或论文结果。
