# P6 `p6-009` 采集记录（已作废）

> **2026-08-04 最终审计结论：该 run 不是正式训练数据。** 后续 image-identity
> 复核发现采集器可能按 repository/version 复用错误的 SWE-bench task image。原始 trace
> 仅保留作历史诊断；两个 processed dataset manifest 均已设置
> `formal_training_eligible=false`，正式 fit/calibration/test loader 会拒绝它。下文保留的是
> 作废前的历史统计，不能再作为模型训练规模、泛化性或系统性能证据。

## 结论

> 2026-08-03 harness 审计更新：`django__django-11138` 和
> `django__django-14011` 受到错误 repository path contract 污染，已通过 run-level
> `TRAINING_EXCLUSIONS.json` 排除。下列 712 条等统计是原始全量采集规模；当前导出
> 仅保留其余六个 workflow 的 650 条 remaining decode demand 训练样本。详见
> `beliefkv_p6_harness_path_recovery_2026-08-03_zh.md`。

`p6-009-train-mixed-r0` 的历史采集原始目录为：

`experiments/raw/p6_agent_semantics_v1/p6-009-train-mixed-r0/20260803T100537Z`

导出的版本化数据集和 coverage 位于：

`experiments/processed/p6_agent_semantics_v1/p6-009-train-mixed-r0-20260803T100537Z`

本轮当时通过了 P6 系统 trace 契约，但后续 provenance 审计推翻了正式训练资格。它不能用于
Frontier 模型拟合、LOPO、agent 任务成功率、clean JCT 或最终泛化效果。

## 固定配置

- 模型：`Qwen3-Coder-30B-A3B-Instruct-FP8`，单 GPU、TP=1；
- KV pool：163,840 token，峰值 resident pressure 80.88%；
- Host HiCache：96 GiB；
- workload：8 个 Django SWE-bench Verified task，并发 8；
- runtime：Deep Agents autonomous mode，动态 subagent；
- context lifecycle：32K 触发压缩、保留最近 8K，普通输出 4096 token，summary 2048 token；
- policy：冻结的 P5 observed JointPlan，predictor 和 predictive physical action 均关闭；
- request timeout：7200 秒；
- collection source fingerprint 在运行前后保持一致。

## 系统与轨迹结果

| 指标 | 结果 |
|---|---:|
| workflow 完成 | 8/8 |
| system JCT eligible | 8/8 |
| native-agent JCT eligible | 0/8 |
| task correctness/measurement valid | 0/8 |
| 总运行时间 | 3562.89 s |
| 外部 agent LLM call | 712 |
| runtime-internal summary call | 18 |
| 工具调用 | 1151 |
| 动态 subagent | 11 |
| SPAWN / JOIN_SATISFIED | 11 / 11 |
| context compact | 18 |
| duplicate tool suppression | 91 |
| SGLang error | 0 |

所有 730 个 LLM submit/result 和 1151 个 tool start/end 均成对；runtime event 投递失败为 0。受控
shutdown 前无 pending transaction、restore obligation、lease、funding 或 in-flight command，最终
`shutdown_ack` 和 `runtime_shutdown` 均已落盘，端口和本项目 GPU 进程已释放。

## 数据集导出

导出完整性检查通过，没有重复 request/decision/censor/PCIe ID、dangling service foreign key 或
partial decision scope：

| 标签表 | 可训练记录 |
|---|---:|
| frontier decision point | 11,674 |
| remaining decode demand request | 712 |
| request-level GPU service interval | 110,933 |
| external/tool survival | 1,151 |
| JOIN reentry | 11 |
| PCIe/HiCache operation telemetry | 2,239 |
| explicit censor event | 91 |

GPU service 的 request identity、prefill/decode phase 和 token delta 覆盖率均为 100%。observer build
P99 为 0.258 ms，audit enqueue P99 为 0.251 ms，未丢弃 debug event。

## 当前可训练边界

1. **可以训练**：remaining decode token demand、局部 action kind、tool competing-risk/survival、output 和
   prompt-growth 分布。训练时必须按 episode/invocation 聚类加权，并排除对应 censor 后不可观测的目标。
2. **暂不训练 reentry 模型**：689 个 eligible action 中 632 个具有显式 reentry，覆盖率 91.73%；
   仍有 57 个 function call 缺少 reentry 或逐调用 censor 原因。
3. **禁止 unlock-hazard/run-to-action 主张**：exact incremental action boundary 为 0/712。当前 runtime
   只在完整 `LLM_RESULT` 后执行工具或 spawn，因此只能学习最终 action 类型和剩余 decode token demand。
4. **不能做 LOPO 或最终 fit**：本轮只有 Django 一个 train project。至少完成包含多个新 train
   project 的 `p6-010` 后，才允许运行 project-macro LOPO。

## 负结果与风险

- 8 个 workflow 均出现 guard/censor；其中 6 次自然语义完成、12 次 guard intervention，主要原因为
  consecutive tool errors、repeated failed tool call 和 no observable progress。最长 workflow 产生 262
  次 LLM、494 次工具调用和 12 次 context compact。该现象是真实的 agent/runtime 长尾，但意味着本轮
  不能作为 clean task-success 数据。
- task correctness 为 0/8 不等于所有 patch 必然错误。当前正式 collector 按 model-terminal 结束且不
  启动 harness LLM repair；缺少可观测成功测试、没有 patch 或终态为 blocked 都会使 correctness gate
  失败。后续应由离线 SWE-bench harness 单独判分，不能修改该批在线轨迹。
- frozen P5 shadow 的 strict-global stale rate 为 99.82%，plan compute P99 为 162.71 ms，physical
  commit P99 为 160.76 ms。runtime GPU interval 排除了 queue wait，但一个共享 batch 的 elapsed 会复制
  到多个 request 行，且边界不是 CUDA event；它只能按 `sample_id` 聚合后用于 service characterization/
  外部验证，不能拟合 Frontier 语义模型或作为正式 service-model 主训练集。开放 P6 在线预测前，控制面
  仍必须使用 semantic plan、局部 read-set commit 和有界开销门槛。
- 本轮没有实际 D2H/H2D KV transfer，主要 residency command 是 terminal-private drop。因此 2,239 条
  operation telemetry 只作条件字段和 submit-to-complete 管线证据，GPU/PCIe service curve 仍需独立
  微基准，不能从该 agent trace 单独拟合。

## 暂停点

下一批为 `p6-010-train-mixed-r0`，包含 seaborn、flask、requests 和 xarray 四个 train project。
其所需 6 个 SWE-bench 镜像当前未安装；第一个镜像下载已按用户要求中止，镜像没有完成注册。
`p6-010` 未创建实验 run，也未启动模型服务。

恢复后顺序固定为：准备镜像 -> 采集 `p6-010` -> 导出并检查 formal dataset -> 在至少两个 train
project 上运行 LOPO -> 训练第一个 formal FrontierBeliefModel。calibration/test-ID 继续封存。
