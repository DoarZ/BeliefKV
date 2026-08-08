# BeliefKV P6 多项目训练数据采集审计（2026-08-04）

## 结论

固定 P5 w4 只用于 correctness 与采集器开发，不能进入正式 FrontierBeliefModel 训练。大量
decision point 不等于大量独立样本：同一 task/workflow 的连续状态高度相关，直接逐行训练会让模型
记住固定任务和轨迹长度。

当前局部标签语料包含 11,491 个 fit-eligible decision point，但独立单位只有 5 个项目、14 个任务和
14 个 rollout 来源，其中 requests 定向复跑不是 clean episode。训练入口已按预期拒绝拟合：
`task_count=14<40`、`workflow_count=14<40`。因此尚未生成正式模型。

## 数据边界

| 批次 | 结果 | 训练资格 | 原因 |
|---|---:|---|---|
| P5 fixed w4 | correctness/dev | 否 | 固定 SymPy 小样本，provenance 为 development-only |
| p6-009 | 历史 run | 否 | task image identity 污染，manifest 已失效 |
| p6-010 GPU0 | 4/4 system JCT | 是 | seaborn、flask、requests，共 4 个不同任务 |
| p6-010 GPU1 | 3/4 system JCT | 否 | recursion-limit，整 shard 为 diagnostic |
| p6-011 GPU0 | 7/8 system JCT | 否 | pylint-4970 输入约 812K token，超过 262K 窗口 |
| p6-012 GPU1 | 8/8 system JCT | 是 | pylint、pytest，共 8 个不同任务 |
| p6-010 requests-5414 定向复跑 | 1/1 系统终态 | 仅局部标签 | guard bug；干预前 65 条有限 horizon 标签 |
| p6-011 pylint-4970 定向复跑 | 1/1 system JCT | 是 | 无超窗口请求；干预前 402 条自然 decision point |

`p6-012` 记录 608 个 LLM request、766 个工具调用、9 个动态 subagent 和 27 次 context compact；
runtime/control source fingerprint 前后一致。其导出包含 9,544 个 eligible decision point，remaining
decode demand 标签完整，reentry cause 覆盖为 95.57%。exact incremental action boundary 仍为 0%，
因此 unlock-hazard/run-to-action 继续关闭。

两次定向复跑均到达系统终态，且没有 server error、recursion error 或 control degradation。但
`requests-5414` 的终态由旧版不可恢复 guard 促成，不能用于 JCT、终态或完整 trajectory；只有干预前
局部标签保留。`pylint-4970` 后段触发 terminal protocol repair。导出器从首次
runtime prompt 干预开始 censor 对应尾部，并同时排除跨越该时刻的监督 horizon；这些行保留供审计，
但不参与 Frontier fit。任务 patch 是否通过官方 SWE-bench harness 不属于性能数据 gate。

## 防过拟合机制

1. 正式 loader 要求 `beliefkv_p6_training_evidence`、冻结 project split、稳定 source fingerprint、
   frozen P5 observed policy、predictor/predictive action 关闭，以及 `formal_training_eligible=true`。
2. P5 w4、censored run、重复 run、重复 decision ID 和 workload manifest 不匹配均 fail closed。
3. loss 先在 local episode 内归一化，再在 workflow rollout 内归一化。一个长 workflow 即使产生数千
   decision point，也不会按行数获得更大训练权重。
4. 正式 fit 最低要求 5 个项目、40 个不同任务和 40 个 workflow；超参数只在 train project 内做
   leave-one-project-out。
5. calibration 与 test 按 repository 完全封存；同一 repository、task、base commit 和重复 rollout
   不跨 split。

最低 40/40 门槛只防止误启动训练，不是论文级充分规模。最终训练目标仍为全部 7 个 train project 上
80--120 个 workflow；随后使用未见 calibration/test project 评价概率校准、OOD fallback 和 planner
regret，而不是只报告逐 decision 分类准确率。

## 文件

- 正式 p6-010 GPU0：
  `experiments/processed/p6_agent_semantics_v1/p6-010-train-mixed-r0-20260804T065717Z/gpu0/dataset`
- 诊断 p6-011 GPU0：
  `experiments/processed/p6_agent_semantics_v1/p6-011-train-mixed-r0-20260804T075322Z/gpu0/dataset`
- 正式 p6-012 GPU1：
  `experiments/processed/p6_agent_semantics_v1/p6-012-train-mixed-r0-20260804T075322Z/gpu1/dataset`
- p6-010 requests-5414 局部干预前 evidence：
  `experiments/processed/p6_agent_semantics_v1/p6-010-train-mixed-r0-20260804T091435Z-recovery/dataset`
- 正式 p6-011 pylint-4970 定向复跑：
  `experiments/processed/p6_agent_semantics_v1/p6-011-train-mixed-r0-20260804T092035Z-recovery/dataset`
