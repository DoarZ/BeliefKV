# P6 失败任务定向修复与复跑（2026-08-04）

## 根因与修复

`p6-010` GPU1 的 `psf__requests-5414` 在原始 shard 中耗尽 512 个 LangGraph superstep。后续审计发现，
第一次定向复跑仍有 runtime bug：一次真实 schema failure 和一次 `duplicate_suppressed` 被重复计算，
并且 guard 进入 recovery 后不可退出。因此下表中的 requests 复跑不再视为完整正式 rollout。

修复后的 runtime 将物理失败和重复意图分别计数。审阅后没有继续扩大语义 guard：
`repeated_tool_call`、错误率和 `no_observable_progress` 仅记录 telemetry，默认不会禁用工具、修改 prompt
或强制终态。该版本曾将 512-step 阈值仅用于观测，并以 2048 superstep 和绝对 workflow deadline
作为进程级保险丝。历史可恢复状态机保留为显式回归开关，不用于正式 P6 collection。

确定性 replay 证明旧 step 62 触发点为 1 次物理失败加 1 次派生 suppression，而不是两次真实失败；
原始长轨迹虽然尾部有 55 个连续无可信进展 batch，默认在线 policy 仍产生 0 次干预。仓库或 image 的
人为错误由 repository-specific preflight 在模型启动前 fail fast，不由 guard 猜测模型语义。

`p6-011` GPU0 的 `pylint-dev__pylint-4970` 曾在一次 50 路并行工具返回后构造约 3.4 MB observation，
最终形成约 812K-token 请求并超过服务端 262K 窗口。修复增加版本化的 tool-observation budget：每个
AI turn 总计最多 65,536 字符、单结果最多 16,384 字符，按实际 fanout 公平分配，使用带原始长度和
SHA256 的 head/tail 截断。工具仍被完整执行，只有返回模型的 observation 被有界物化。

runtime 干预会改变后续 action 分布。数据导出器因此将首次 graph-budget、loop-guard 或 terminal-repair
干预之后，以及监督 horizon 跨越该时刻的样本标记为 `training_eligible=false`。这些样本保留用于审计，
不会被当作自然 autonomous 行为训练。

## GPU 复跑结果

| 任务 | 系统终态 | LLM/工具调用 | 可训练 decision point | 结果 |
|---|---:|---:|---:|---|
| `psf__requests-5414` | 1/1 | 18/15 | 65 | 仅干预前局部标签可用；JCT/终态/完整轨迹不可用 |
| `pylint-dev__pylint-4970` | 1/1 | 27/21 | 402 | 无 context overflow；terminal-repair 尾部被 censor |

后续 `recovery-v2b-gpu0` 仅用于验证派生 requests harness：repository preflight 通过，服务端完成
20 次 LLM 和 18 次工具调用，无 fixture、API timeout 或 recursion error；但该轮仍运行在旧的语义
guard 配置下并被 `no_observable_progress` 干预，因此同样不进入正式训练集。默认 observe-only policy
已经通过 CPU replay，并在下述单次 GPU 定向验证中越过 512-step 阈值。

### Observe-only 定向 GPU 验证

唯一一次定向运行位于：

`experiments/raw/p6_agent_semantics_v1/p6-010-train-mixed-r0/20260804T111841Z/guard-observe-only-gpu0`

repository-specific preflight 返回 0，未再出现 fixture、工作目录或 image 配置错误。运行完成 444 个
业务 LLM 请求和 441 次工具调用；所有 LLM/tool 事件均成对，1 个动态 subagent 正常 RETURN，JOIN
正常满足，runtime event delivery 无降级，SGLang error count 为 0。运行期间 3 次 context compact 正常
完成，没有 API timeout、GraphRecursionError、OOM、restore/admission liveness 或未决事务。

512-step soft budget 只产生一次 `agent_graph_step_soft_budget_observed`，没有修改 prompt、关闭工具或强制
终态。模型随后继续执行至 2017 step，证明原始 512-step 误终止已经消除。但模型在已有 patch 后连续
207 次执行完全相同、exit code 为 0 且输出为空的命令，累计 220 个无可信进展 batch，未自然返回终态。
2048-step 硬保险丝按设计保留 32 step，在 2017 step 触发一次有界 finalization，最终状态为 `blocked`。

因此该轮得到的是分层结论：harness/runtime 修复和 hard-fuse liveness 验证通过；模型终态能力未通过。
完整 episode、native-agent JCT 和 terminal outcome 不进入训练或性能结论；hard-fuse 前且监督 horizon 不跨越
干预点的局部 action/token-demand 样本可继续由 censor 规则保留。该失败不能归因于 BeliefKV 数据面：
shutdown 前所有 command、transaction、restore obligation、lease、funding 和 pin 均已守恒，GPU 也已释放。
受控 shutdown 已收到 ACK，但 SGLang 进程退出时仍打印 `TemporaryDirectory.cleanup` 的 `SystemExit` 和
NCCL process-group destroy warning；本轮没有资源残留，后续将其作为独立的退出清理告警处理。

该负结果随后推动了两项通用修复，不针对 `requests-5414`：成功工具调用若在同一 conversation、参数签名
和 workspace epoch 下连续两次只有标准 exit-code 0 尾注、没有实际输出或 workspace 变化，第三次起返回
结构化 `duplicate_suppressed`，不再物理执行；默认 LangGraph hard limit 恢复为 512，并从 step 481
预留 32 step 做最终收尾，384-step 仅记录 telemetry。有效输出、workspace epoch 前进及并发 in-flight
fanout 不受该 circuit 影响。

两次运行的 source fingerprint 均保持不变，数据集 foreign key、decision ID、batch sample 和 scope
完整性检查全部通过。但 source 稳定不等于 episode 完整：requests rollout 已重新导出为
`evaluation_role=partial_local_pre_intervention_evidence`、`clean_episode_eligible=false`、
`workflow_jct_eligible=false`。65 条干预前且 horizon 未跨界的局部 action/token-demand 标签保留。

`pylint-4970` 本次模型选择串行工具调用，没有再次产生 50 路 fanout。因此 GPU 复跑证明正常路径不再
超窗口，50 路极端 fanout 的预算不变量由确定性单元测试覆盖，不能声称已在 GPU 上复现该极端路径。

## 数据与训练状态

修复后的正式数据位于：

- `experiments/processed/p6_agent_semantics_v1/p6-010-train-mixed-r0-20260804T091435Z-recovery/dataset`
- `experiments/processed/p6_agent_semantics_v1/p6-011-train-mixed-r0-20260804T092035Z-recovery/dataset`

加入既有 `p6-010` GPU0 和 `p6-012` GPU1 后，local-label loader 可读取 11,491 条 fit-eligible decision
point，覆盖 5 个项目、14 个任务和 14 个 rollout 来源；其中 requests recovery 只贡献有限 horizon 的局部
标签，不能作为 clean workflow episode。训练仍按预期被 40-task/40-workflow diversity gate 阻止。

同一 requests SWE-bench image 的进一步检查确认 `pytest-httpbin` 未安装，而不是单纯版本不兼容。
现已增加版本化 repository/image profile、派生 harness image 和仓库级 preflight；preflight 覆盖模块导入、
pytest 插件注册、fixture 枚举和最小 fixture smoke，失败会在 LLM 启动前终止，不再污染 agent trace。

两张 GPU 同时启动各自 96 GiB Host HiCache 时，服务在请求提交前因并发 Host cache 分配失败。失败启动
目录已写入 `STARTUP_FAILED.json`，不进入数据集。上述两个 recovery run 随后按相同服务配置顺序执行，
没有降低 Host/KV 配置来换取通过。
