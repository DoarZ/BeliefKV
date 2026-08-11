# P6 MVP 预测器 GPU Shadow 验证计划（2026-08-07）

> 改动者：deepseek
> 状态：计划阶段（2026-08-07 创建）；所有代码改动与实验结论会持续固化到本文档。

## 背景与目标

P6 已训练 development-only 的 `FrontierBeliefModel` MVP
（`experiments/models/frontier_belief_mvp_v1/v2/v3.json`），其中 v3 最新并包含
`join_wait` 组件。MVP 模型只用于验证预测管线在真实 GPU 服务上是否跑通，不产生论文级
效果结论。

本次 GPU 实验的目标（用户 2026-08-07 确认）：

1. 在 GPU 上以 **shadow 模式**（方案 a）在线加载预测器，记录预测并与观测结果对比；
   预测模块暂不参与调度决策（`prediction_used=False` 语义保持不变）。
2. 后续若预测质量达到门槛，再按 P6.3 -> P6.4 -> P6.5 顺序将预测模块接入真实调度。
3. 使用**测试集** workflow（matplotlib / scikit-learn，collection_v4 的
   `p6-006`~`p6-012` test_id 批次）运行实验。
4. MVP 阶段允许基于当前训练集做 development 校准；正式校准仍须使用封存的
   astropy calibration split（尚未采集）。

## 决策记录

| 日期 | 决策 | 理由 | 改动者 |
| --- | --- | --- | --- |
| 2026-08-07 | GPU 实验采用 shadow 模式（方案 a），预测器不直接接入调度 | 项目门槛（P6.3 离线 risk gate / P6.4 shadow / P6.5 逐级上线）未满足；MVP 模型未校准；直接在测试集上驱动调度会污染测试证据 | deepseek |
| 2026-08-07 | 测试集选择 collection_v4 test_id 批次（matplotlib + scikit-learn） | 用户指定"选用测试集的 workflow"；split manifest 中 test_id 项目为 matplotlib/scikit-learn | deepseek |
| 2026-08-07 | 允许基于训练集做 development 校准 | 用户确认 MVP 阶段可先校准再实验；校准结果标注 development-only，不用于正式结论 | deepseek |
| 2026-08-07 | 归档旧的 `experiments/shadow/p6_shadow_v1` 目录后新建运行目录 | 旧目录存在导致服务器启动失败（`policy_snapshots.jsonl.gz` 已存在）和输出目录重复等问题 | deepseek |

## 需要的代码改动（待实现，改动者 deepseek）

1. `scripts/prepare_deepagents_server_config.py`：新增 `--predictor-model PATH`
   参数，将 `predictor_enabled` 置为 true 并写入 `predictor_model_path`。
2. `beliefkv/control/controller.py`：新增有界的 `predictor_shadow` 审计事件，
   在 LLM/tool/reentry 等边界记录预测中位数、support level、OOD 原因；不改变调度行为。
3. `scripts/calibrate_frontier_belief.py`：新增 `--development-on-train` 模式，
   允许基于训练/development 行校准并显式标记 `development_only`。
4. 新增预测 vs 观测的离线对比脚本（boundary NLL/准确率、remaining-decode MAE、
   tool wait MAE、prompt-growth MAE、校准区间覆盖率、OOD/backoff 比例）。

## 代码改动（已实现，改动者 deepseek）

1. `scripts/prepare_deepagents_server_config.py`：新增 `--predictor-model PATH`；
   配置生成验证通过（`predictor_enabled=true`、
   `predictor_model_path=<abs path>`）。
2. 新增 `beliefkv/predictor/online_shadow.py`：
   `build_frontier_shadow_records()` 有界发布每个作用域 invocation 的
   frontier 预测（签名变化 + 每 invocation 至少 1s 间隔），
   `feature_source="online_approx"`，预测失败不影响调度关键路径。
3. `beliefkv/control/controller.py`：`ControllerTickResult` 新增
   `frontier_shadow_events`；tick 内每 500ms 最多构建一次 shadow 记录。
4. `beliefkv/runtime/sglang_v052rc1.py`：scheduler safe point 将
   `frontier_shadow` 审计事件写入 runtime audit；`prediction_used` 保持 false。
5. `beliefkv/predictor/composer.py`：
   - 修复 `from_frontier_model` 未保存 `_frontier` 的缺陷（新增
     `frontier_model` 属性，`RemainingTimePredictor.load` 现在能正确加载
     MVP v3 模型）；
   - 修复 `from_dict` 对 schema v2 的分发：带 `models` 的 legacy 工件仍走
     composite 路径，否则走 frontier 路径；
   - `LLM_RESULT` 事件现在跟踪 `generated_tokens`。
6. `beliefkv/predictor/structured_frontier.py`：`calibrate()` 新增
   `allow_development` 参数（正式校准语义不变）。
7. `scripts/calibrate_frontier_belief.py`：新增 `--development-on-train`；
   校准产物显式标记 `development_only` 与
   `calibration_source="train_development_mvp"`。
8. `scripts/evaluate_frontier_mvp.py`：新增 MVP-only 离线对比脚本
   （不要求 formal eligibility，输出强制标注 development-only）。
9. `tests/test_online_shadow.py`：新增 3 个单元测试。
10. `tests/test_predictor_training.py`：修正陈旧断言（schema_version 1 -> 2，
    与 `RemainingTimePredictor.ARTIFACT_SCHEMA_VERSION` 对齐）。

### 校准结果（development-only）

`experiments/models/frontier_belief_mvp_v3_calibrated_dev.json`

```text
decision_point_count : 74014
episode_count        : 4460
local_episode_count  : 6946
boundary_temperature : 1.05
tool_temperature     : 0.60
target_coverage      : 0.9
interval_slack       : remaining_decode_tokens=118.9, prompt_growth=3.0,
                       next_output=6.0, remaining_external_wait=723.6
```

### CPU 回归结果（2026-08-07，deepseek 改动后）

- beliefkv 环境：`549 passed, 10 skipped, 3 subtests passed`；
  2 个既有失败（`test_sglang_adapter` 的
  `test_scheduler_step_audits_ack_before_performance_telemetry` 与
  `test_scheduler_step_audits_resolved_transfer_bytes`）为先前未提交的
  scheduler_step 改动导致（测试未设置 `runtime.config`），与本次改动无关，
  已定位待修。
- beliefkv-agents 环境（排除 sglang/fastapi 依赖测试）：
  `556 passed, 3 skipped`；`test_p6_collection.py` 5 passed。
- `test_responses_bridge` 在 beliefkv 环境 8 passed（agents 环境缺 fastapi，
  属环境问题）。

## 前置准备（2026-08-07 进行中）

- [x] 确认 GPU 主机侧可用（沙箱内不可见，通过提权命令确认）。
- [x] 用户批准网络权限：克隆 matplotlib / scikit-learn 源码仓库。
- [x] 用户批准 docker 权限：检查/构建测试集任务镜像。
- [x] 克隆
  `workloads/sources/p6_swebench_verified/matplotlib__matplotlib` 与
  `scikit-learn__scikit-learn`（scikit-learn 已完成，matplotlib 进行中）。
- [ ] 拉取 test_id 批次所需 swebench 镜像（p6-007 的 8 个 matplotlib 镜像
  并行拉取中）。
- [x] 归档旧 shadow 运行目录。

> 更正：冻结的 collection_v4 计划中 test_id 批次为 **p6-007 ~ p6-012**
> （matplotlib 30 + scikit-learn 16 个 workflow）；目录中残留的
> `p6-006-test_id-mixed-r0.json` 清单未被计划引用（计划中的 p6-006 是
> calibration 批次），不使用。

## 实验步骤（待执行）

## 实验步骤（已执行 2026-08-07）

1. 生成带预测器配置的服务器运行目录
   `experiments/shadow/p6_shadow_v2/gpu0/server`，使用
   `--predictor-model experiments/models/frontier_belief_mvp_v3_calibrated_dev.json`。
2. 启动 SGLang 服务器（GPU0，`MEM_FRACTION_STATIC=0.952`，163840 KV tokens，
   96GB HiCache），`/health` 通过，端口 18000 监听正常。
3. 提交 `p6-007-test_id-mixed-r0` 的 4-workflow 子集
   （matplotlib-13989 / -14623 / -20488 / -22865），
   `--allow-test --predictor-shadow-enabled`；
   collection contract 确认 `predictor_enabled=true`、
   `predictive_actions_enabled=false`、`split=test_id`。
4. （待执行）服务器关停后，用 `characterize_p6_coverage.py` 导出数据集。
5. （待执行）离线对比预测与观测标签（`evaluate_frontier_mvp.py`），
   产出指标并固化到本文档。

## 运行中发现的环境问题（2026-08-07）

- 根分区 `/dev/sda2`（docker 存储所在）在并行拉取 8 个 matplotlib 镜像时写满
  （340G / 329G used）。已成功拉取 4 个镜像（13989/14623/20488/22865），
  剩余 4 个（20826/21568/22871/23299）因磁盘满失败。
- **用户决策（2026-08-07）：不腾空间**。`/opt/downloaded_models` 与
  `/tmp` 下的模型/缓存均属其他用户，无管理权限。本轮以已就绪的
  4 个任务完成冒烟验证；完整 8 任务批次（p6-007 其余 4 个实例）暂缓，
  待磁盘空间可解决后再补跑。

## 运行结果（2026-08-07，p6-007 子集 4 workflow）

### 运行概况

- 运行目录：`experiments/shadow/p6_shadow_v2/gpu0/20260807T151000Z`
- 批次：`p6-007-test_id-mixed-r0`（matplotlib-13989 / -14623 / -20488 / -22865）
- 时长：7828s（约 2.2 小时）；LLM 请求 454 次、工具调用 946 次、
  dynamic subagent 6 个、JOIN 6 个
- **system_jct_eligible: 4/4**；**native_agent_jct_eligible: 2/4**
  （13989、22865 天然完成；14623 由 activation wall-clock 守卫终止、
  20488 由 guard 干预完成）
- runtime control delivery：0 失败；context compaction 21 次、
  context epoch advance 423 次
- 服务器解码吞吐：mean 38.9 tok/s、p50 25.0、p95 136.8、max 176.8
  （14623 长上下文收尾阶段降至 ~2-5 tok/s，为尾部效应）
- 服务器正确关闭：`shutdown_state=acknowledged`，事务/obligation 守恒通过

### 在线 Frontier Shadow（关键验证点）

- `frontier_shadow` 审计事件 **1045 条**，覆盖 4 workflow、31 个 invocation；
  状态分布：running_llm 427 / ready 314 / wait_tool 289 /
  wait_child 9 / wait_join 6
- 全部记录 `support_level=backoff`（在线特征与训练键不一致，MVP 预期内）；
  `boundary_top`：tool 427、unknown 618（unknown 即
  `boundary_unavailable` OOD）
- 预测中位数（p50-of-p50）：remaining_decode=64 tok、
  external_wait=512 ms、prompt_growth=304 tok、next_output=53.8 tok
- `prediction_used=false`，`predictive_actions_enabled=false`，
  预测未影响 admission/residency/transfer

### 数据集导出（development-only）

- `experiments/processed/p6_shadow_v2_20260807/dataset`
- 9,602 个 frontier decision point（test_id 9,367 eligible + 235 排除）
- manifest：`formal_training_eligible=false`、
  `evaluation_role=development_diagnostic`、
  `formal_ineligibility_reasons=["predictor_shadow_development_only"]`
- 新增 `--allow-development-only` 导出开关（改动者 deepseek）：
  显式标记而非静默通过；正式 loader 仍会拒绝该 manifest

### 离线对比：MVP 模型 vs 测试集观测（evaluate_frontier_mvp）

模型：`frontier_belief_mvp_v3_calibrated_dev.json`（development 校准，
target coverage 0.9）；数据：matplotlib（训练时未见项目），9,367 决策点

```text
boundary 分类        : accuracy 0.901, NLL 0.341, Brier 0.177, ECE 0.032
tool terminal 分类   : accuracy 0.865, NLL 0.332, Brier 0.196, ECE 0.045
remaining_decode     : MAE 213.7 tok, 校准区间覆盖率 91.8%
next_output          : MAE 152.1 tok, 覆盖率 90.6%
prompt_growth        : MAE 1755.7 tok, 覆盖率 84.1%
remaining_external_wait: MAE 513.7 ms, 覆盖率 99.3%
support              : 全部 backoff；OOD fallback rate 0.392
```

### 结论（MVP 阶段）

1. **管线端到端跑通**：模型加载、在线预测、审计落盘、调度零影响、
   数据集导出、离线对比全部可用。
2. **模型泛化合理**：对未见过的 matplotlib 项目，boundary 分类准确率 90%，
   校准区间覆盖率 84-99%（development 校准有效）；但所有在线预测都处于
   backoff（在线特征映射与训练键不一致），说明在线特征适配是下一步重点。
3. **已知限制**：本轮数据 `development_only`，不可作论文 test evidence；
   完整 8 任务批次因磁盘空间暂缓；14623 尾部解码慢（~2-5 tok/s）与
   调度主线程空转问题待单独 profiling（属于工作区先前 P5D 改动，
   与本次预测器改动无关，tick 开销实测 ~120-140us）。

## 在线特征键对齐（2026-08-07 完成，改动者 deepseek）

### 诊断结论

- 角色键（`agent_definition_id`）在线侧已经是稳定值
  （`autonomous-supervisor` / `general-purpose` / `repository-explorer` /
  `implementation-agent` / `context-summarizer`），与训练一致；
- 训练与导出数据的 `boundary_history` 使用运行时词汇
  （`function_call` / `tool_end` / `spawn` / `handoff` / `final_answer` /
  `return`），而在线适配器此前误用了 legacy ActionKind 词汇
  （`tool_shell` / `spawn_child`）——**主要失配**；
- `backend_pressure` 训练格式为 `unknown` / `active_family:N`，在线侧此前用
  `hbm_high` / `gpu_high`——**失配**；
- `tool_family` 训练在无活动工具时为 `unknown`，在线侧此前回退到
  `backend_class`——**失配**；
- `current_sequence_tokens` 训练与 `context_tokens` 高度接近
  （p90 差 2，power-of-two 桶化后无差别），在线侧继续使用
  `context_tokens` 即可。

### 改动

1. `beliefkv/predictor/composer.py`：
   - 新增共享 helper `observed_boundary_action(event)`（LLM_RESULT 取
     `structured_action_kinds` 首项，否则取 TOOL_END/SPAWN/HANDOFF/RETURN/
     MESSAGE 事件类型）；`InvocationPredictionFeatures` 新增
     `boundary_history`，`observe_event` 按训练语义增量维护（截断到 32）；
2. `beliefkv/experiments/p6_decision_points.py`：删除本地 `_observed_action`，
   改用共享 helper，保证离线导出与在线特征永不漂移；
3. `beliefkv/predictor/online_shadow.py`：
   - `boundary_history` 改用训练词汇（[-8:] 窗口）；
   - `tool_family` 回退改为 `unknown`；
   - `backend_pressure` 改为 `active_family:N`（按活跃 WAIT_TOOL 的
     family 计数，与导出器一致）；
4. `tests/test_online_shadow.py`：新增训练词汇 history 与特征映射测试。

### 验证

- 将本次运行的 deepagents trace 重新灌入新的 `observe_event`，
  逐 invocation 的 `boundary_history` 与导出器规则逐条一致
  （词汇、顺序完全匹配；仅存的差异是 32 条截断与决策点快照时机，
  对模型使用的 [-8:] 窗口无影响）；
- 离线诊断确认：即便使用训练同款特征，模型在 test_id（matplotlib）
  上仍为 100% `backoff`、OOD 率 0.392——backoff 是模型对未见项目的
  诚实降级行为（边界分类仍有 90% 准确率），**不是适配器失配**；
- 受影响测试 69 passed（含新增对齐测试）。

结论：在线特征键已与训练特征格式完全对齐；后续若想提升 exact 命中率，
需要扩大训练覆盖（更多项目/角色/上下文桶），而不是继续改适配器。

## 调度主线程 profiling 与优化（2026-08-07 完成，改动者 deepseek）

### Profiling 结论

用 py-spy 对真实 GPU 服务器采样（空闲与负载两轮）：

- **事件循环是纯忙等**：SGLang `event_loop_overlap` 空闲时以
  6,500-13,000 步/秒空转，每步固定开销（~240us）被放大为 ~100% 单核；
- 负载时每步 240us，其中 `controller.tick` ~138us（58%）、
  `_maybe_record_incremental_policy_snapshot` 的
  `characterize_action_frontier_coverage` ~19%、joint 决策验证 ~17%、
  seed epoch ~7.5%、HiCache `writing_check` 轮询 ~5%；
- 审计量化：33,621 次 `physical_commit_budget_exceeded` 累计
  ~5,700 CPU-秒（均值 169ms/次，预算 1ms），加上 176.5 万次失败的
  全局验证——这就是此前负载时 82-94% CPU 的来源。

### 优化（全部保留原有正确性语义）

1. **joint 决策复用**（`sglang_v052rc1.py`）：plan_id + 状态签名
   （graph/page/fairness/transfer/runnable/liveness/HBM 桶）不变时复用
   上次决策，签名检查 10ms 节流；`safe_point_decision_reused` 从 216
   增至 143 万，验证失败从 176.5 万降至 ~3,000（590 倍）。
2. **policy snapshot 廉价预检 + 惰性 coverage**：coverage 只在实际捕获
   快照时计算（原每步全量计算）；廉价字段无变化时整段跳过。
3. **controller tick 5ms 门控**（`sglang_adapter.py` bridge）：无待处理
   传输/排队请求时按 5ms 间隔运行完整 tick。
4. **HiCache ACK 轮询 5ms 门控**、**policy snapshot 检查 5ms 门控**。
5. **`--sleep-on-idle`（启动参数，已设为 launcher 默认）**：空闲时阻塞在
   zmq poller 直到请求/事件到达——根治忙等（官方 IdleSleeper 机制）。

### 验证结果（GPU0，同任务 matplotlib-13989）

```text
                               优化前(4wf 满跑)   优化后(1wf 复测)
调度空闲 CPU                     ~103%              3.1%
解码吞吐 p50                    25.0 tok/s         59.8 tok/s
同任务完成时间                   914s               229s
physical commit 均值            169ms              3.9ms
system/native JCT gate          4/4, 2/4           1/1, 1/1
runtime event delivery          0 失败             0 失败
frontier_shadow 记录            1045               99（同比例）
```

- 功能正确性保持：system gate 通过、JOIN/子代理闭合、事件零失败、
  shadow 记录正常落盘；
- 启动脚本 `launch_deepagents_swebench_server.sh` 现支持透传 SGLang
  参数，且默认 `SLEEP_ON_IDLE=1`（可 `SLEEP_ON_IDLE=0` 关闭）。

### 说明

- `--sleep-on-idle` 下 BeliefKV 运行时事件在下一个请求唤醒循环时处理；
  本验证中 176 次事件送达 0 失败、ACK 窗口正常，与采集契约兼容；
- 剩余负载 CPU（~68%）主要为 SGLang 原生 batch 准备（prepare_for_decode、
  bitmask 等）与 ~47 次/秒的 joint 验证，属正常服务开销；
- 两个 `test_sglang_adapter` 既有失败（测试未设置 `runtime.config`）为
  先前未提交的 P5D 改动遗留，与本次改动无关；全套 552 passed。

## 已知取舍

- 带 `predictor_enabled=true` 的采集 run 会被导出器标记为 formal ineligible；
  本阶段结果仅作为 MVP 管线验证。正式 test evidence 需要在预测器关闭状态下重采。
- test_id 数据只用于评估，不用于模型选择/在线更新，避免污染。

## 2026-08-08 审计更正

后续代码与物理日志审计推翻了本节“预测型 Joint Plan”作为有效 P6 方案的资格：

1. 在线 heuristic 将 `predicted_external_wait_ms` 与 token demand 混入同一
   `predicted_idle_ms` 排序，量纲不一致；
2. test-ID 上全部 composite prediction 为 `backoff`，不应获得 destructive
   residency 或 retraction 权限；
3. `victim_prediction_selected` 在物理 victim 真正选中前计数，不能证明预测
   导致了实际迁移；
4. 124 次对照迁移中 42 次只是 `drop_terminal_private`，旧统计不能等同于
   predictive D2H/H2D；
5. agent 自主轨迹在两组间不同，n=1 的 JCT/工具数差异不能归因于策略。

旧实现现已撤回：P5 JointPlanner 固定使用 observed ordering/LRU residency，
`joint_predictive_enabled` 仅保留为弃用兼容字段；在线汇总必须保持
`prediction_used=false`。新的 P6 路径使用完整局部分布、closure-complete scope、
候选相关物理化和 timeline risk evaluation，当前仅发布 read-only
`predictive_risk_shadow`。以下两节保留为历史失败记录，不再作为性能证据。

## 预测型 Joint Plan GPU 实验（2026-08-07 历史无效实验）

> 改动者：deepseek；用户决策：跳过 shadow 直接验证预测器的调度作用，
> 以 MVP 模型快速做端到端验证（不改变跨 workflow 公平性语义）。

### 代码改动（已实现，改动者 deepseek）

1. `beliefkv/core/config.py`：新增 `joint_predictive_enabled`（默认 false，
   保持 observed joint plan 为默认路径）。
2. `beliefkv/policy/reference/base.py`：`RunnableInvocation` 新增可选预测字段
   （`predicted_remaining_decode_tokens` / `predicted_external_wait_ms` /
   `predicted_next_output_tokens` / `prediction_support_level` /
   `prediction_ood_reasons`），to_dict/from_dict 同步。
3. `beliefkv/predictor/online_shadow.py`：新增共享 helper
   `build_invocation_frontier_predictions()`（shadow 与 joint plan 共用同一套
   在线特征 → 预测路径，避免双份实现漂移）。
4. `beliefkv/runtime/sglang_v052rc1.py`：调度安全点在
   `_policy_runtime_runnable` 内为 runnable 请求附加预测字段；`JointShadowDelta`
   携带全量 `frontier_predictions`；新增 `_joint_predictive_counts` 聚合，
   shutdown 的 `online_joint_control_summary` 输出 `predictive_decision_counts`
   并置 `prediction_used=true`；`runtime_initialized` 标注
   `planner=belief_joint_semantic_predictive`。
5. `beliefkv/runtime/joint_shadow.py`：`JointShadowDelta.frontier_predictions`
   字段 + coalesce 透传；`IncrementalPolicyInputAssembler.build()` 将预测包成
   `MetadataValue(source=PREDICTED)` 放入 `PolicyInput.optional_metadata`。
6. `beliefkv/policy/joint_scheduler.py`：
   - `_Candidate` 新增预测字段；`order_key` 在因果/依赖层之后用
     预测剩余 decode（SRPT 式）替换纯 submitted_ts 排序，无预测时严格回退
     原语义（单元测试覆盖）；
   - `_semantic_residency_targets` 的受害者选择：有预测时按"预测下次使用
     空闲时长"最长者优先驱逐（替代 LRU），无预测回退 last_access；
   - `JointPlan` 新增 `prediction_used` 与 `prediction_influence` 计数
     （ordering_changed / prediction_available / support_* /
     victim_prediction_selected）。
7. `scripts/prepare_deepagents_server_config.py`：新增
   `--enable-joint-predictive`（要求 `--enable-online-joint` 与
   `--predictor-model`）。
8. `scripts/run_p6_collection_batch.py`：新增 `--predictive-joint-enabled`，
   契约写入 `predictive_actions_enabled=true`、`joint_predictive_enabled=true`
   （导出器按 development-only 处理）。
9. 测试：`tests/test_predictive_joint_plan.py`（5 个）+ `test_joint_shadow_worker`
   集成测试 1 个；全量回归 `557 passed`（仅 2 个 P5D 既有失败，
   与本改动无关）。

### 实验配置

- 运行目录：`experiments/shadow/p6_predictive_joint/gpu0/`
  （结果 `20260807T125336Z/`，服务器 `server/`）
- 服务器参数：`--enable-online-joint --enable-observed-admission
  --enable-running-retraction --enable-joint-predictive
  --predictor-model experiments/models/frontier_belief_mvp_v3_calibrated_dev.json`
- 批次：`p6-007-test_id-mixed-r0` 子集 4 workflow
  （matplotlib-13989 / -14623 / -20488 / -22865），
  `--allow-test --predictive-joint-enabled`；
  契约确认 `predictor_enabled=true`、`predictive_actions_enabled=true`、
  `joint_predictive_enabled=true`、`runtime_source_stable=true`。

### 运行结果

```text
总时长                    4210.4s（shadow 基线 7828.2s，-46%）
system JCT gate           4/4（与基线一致）
native JCT gate           1/4（基线 2/4；本轮 14623/20488/22865 由
                          loop guard 干预终止，13989 自然完成）
LLM 请求 / 工具调用        602 / 615（基线 454 / 946）
解码吞吐 p50 / p95        23.5 / 58.2 tok/s（基线 25.0 / 136.8）
runtime control delivery  0 失败；transfer 42 次 0 watchdog 过期
JOIN 闭合                 7/7；context epoch advance 582
frontier_shadow           1428 条，全部 support_level=backoff
```

### 预测决策计数（shutdown online_joint_control_summary）

```text
prediction_used           true
plans_using_predictions   28,834（几乎全部发布的 joint plan 消费了预测）
prediction_available      88,710（在线附加到 runnable invocation 的预测数）
support_backoff           88,710（MVP 对未见项目全部 backoff，符合预期）
victim_prediction_selected 27,925（residency 受害者选择使用预测的次数）
ordering_changed          3（同 workflow 内预测排序改变 observed 顺序的次数）
```

### 结论与限制

1. **机制端到端跑通**：预测模型输出（剩余 decode / 外部等待 / next output /
   support）在安全点进入 `PolicyInput`，joint plan 的两个受控决策点
   （workflow 内排序、residency 受害者选择）都真实消费了预测并落审计；
   `prediction_used=true`，全程事件零失败、事务守恒。
2. **调度作用以 residency 为主**：28.8k 个计划使用预测，其中 27.9k 次
   受害者选择由预测驱动；排序改变仅 3 次（workflow 内并发 runnable 很少，
   且因果/公平层优先，MVP 预期内）。
3. **JCT 对比受混淆，不能归因于预测**：基线 shadow 运行发生在
   sleep-on-idle 与调度主线程优化之前，总时长 -46% 主要来自这两项优化，
   不是预测的净收益。要归因预测本身，需在**同一优化栈**上跑
   observed（`--enable-online-joint` 不带 `--enable-joint-predictive`）
   对照；本轮尚未做。
4. native JCT 1/4 vs 基线 2/4 属工作负载方差（loop guard 对 14623/20488/
   22865 的终止时机不同），MVP 阶段不作为策略好坏结论。
5. 全部预测仍为 backoff（训练覆盖不足），本轮只验证"预测如何进入调度"，
   不验证"预测是否比 observed 更好"；后者需扩大训练数据后重测。

## 同栈 observed 对照实验（2026-08-07 完成，A/B 归因）

> 改动者：deepseek；目标：隔离"规划器是否消费预测"这一变量，归因预测式
> 调度的净效果。对照组仅去掉 `--enable-joint-predictive`，其余参数与预测组
> 完全一致（同一优化栈：sleep-on-idle、joint 复用、tick 门控；预测器同样
> 加载、shadow 审计照常）。

### 运行信息

- 运行目录：`experiments/shadow/p6_observed_control/gpu0/`
  （结果 `20260807T141721Z/`）
- 契约：`predictor_enabled=true`、`predictive_actions_enabled=false`、
  `joint_predictive_enabled=false`、`runtime_source_stable=true`；
  shutdown 审计 `planner=belief_joint_observed`。
- 批次：与预测组相同的 4 个 test_id workflow。

### A/B 对比（同栈，n=1/组）

```text
指标                  observed 对照    预测式        Δ
总时长                4141.4s        4210.4s      +1.7%
LLM 请求              475            602          +26.7%
工具调用              1080           615          -43.1%
system JCT gate       4/4            4/4          =
native JCT gate       4/4            1/4          预测差（guard 干预）
解码吞吐 p50          40.6 tok/s     23.5 tok/s   对照更高
transfer 次数/字节     124 / 16.4GB   42 / 8.1GB  对照更多
context epoch advance 438            582          预测更多
joint plan 发布       58,018         18,423       对照更多
事件失败/错误          0              0            =
```

各 workflow 时长：13989 对照 1900s vs 预测 1807s；14623 4125s vs 4065s；
20488 2211s vs 4204s（预测 +90%）；22865 696s vs 3593s（预测 +416%）。

### 粗略判断（按用户要求，基于 LLM 数、工具调用数、用时）

1. **预测式调度目前没有可判定的端到端提升**：总时长持平（+1.7%），且
   20488/22865 两个任务在预测组显著变慢并被 loop guard 终止；对照组的
   -46%（相对旧 shadow 基线）确认为调度优化而非预测的收益。
2. **行为确实不同**：预测组工具调用 -43%、LLM 请求 +27%、context 重排更多
   （582 vs 438）、transfer 更少（8.1GB vs 16.4GB）。可能的解释是预测驱动
   的驱逐/重排减少了 agent 的重复工具尝试（工具调用少），但付出更多上下文
   重排与更长的尾部任务；MVP backoff 预测下这是"有副作用、无明显收益"。
3. **方差警告（n=1 不可定论）**：agent 轨迹随机性很大——对照组 14623 跑了
   746 次工具调用、预测组仅 169 次；native JCT 三轮（shadow 基线 2/4、
   预测 1/4、对照 4/4）完全不可比。要归因需固定 trace 重放或 ≥3 次重复。
4. **结论方向**：与离线"backoff 是诚实降级"的判断一致——MVP 预测在
   test_id 上不优于 observed；继续这条路的前置条件是扩大训练数据让
   support 脱离 backoff。
5. 已知小异常：对照组 shutdown 汇总出现 `ordering_changed=7 /
   plans_using_predictions=7`（占 58,018 个计划的 0.012%）。运行时注入已
   确认按 `joint_predictive_enabled` 门控（对照 policy snapshots 0 条预测
   字段），来源待查，对指标无影响。

## 训练语料审计与 v4 重训（2026-08-08）

> 改动者：deepseek；背景：用户确认 67 个 train workflow 已有 raw 实验数据，
> 可跳过新采集直接训练。经核查，该语料**已经被 v3 完整消费**，直接重训
> 无增量；本轮产出 v4 作为对照证据，并明确后续真正可行的扩展路径。

### 语料现状（experiments/raw + experiments/processed/p6_agent_semantics_v1）

- raw 侧：13 个 train 批次（p6-009~p6-021）均有完整运行输出
  （`experiments/raw/p6_agent_semantics_v1/p6-0XX-*/<ts>/gpu*/workloads/`）。
- processed 侧：13 个批次已导出
  `frontier_decision_points.jsonl` + `dataset_manifest.json`。
- 按 `dataset_manifest.formal_training_eligible=true` 过滤后的 train 语料：
  **84,025 决策点 / 6 项目 / 34 任务 / 34 workflow / 11 run**
  （django 77,475、pytest 46,141、pylint 6,779、seaborn 645、flask 874、
  requests 65；xarray 无 train 行，其 29,881 行为 test_id/development）。
- 注意：诊断批次（p6-010 gpu1、p6-011 gpu0、development_diagnostic_p6-009）
  的 manifest `formal_training_eligible=false`，未纳入训练。
- 多样性门槛：34 task / 34 workflow 低于正式 40/40（目标 80-120 workflow），
  MVP 阶段按 5/30/30 放宽拟合（train_frontier_mvp.py 已支持参数化）。

### v4 训练与校准（已产出）

- `scripts/train_frontier_mvp.py`：新增 CLI 参数（--data-dir/--model-version/
  --output/--minimum-*），loader 改为按 manifest 资格过滤、覆盖 gpu0+gpu1 与
  recovery 目录（修复原实现只 glob gpu0、漏掉 p6-012 gpu1 的问题）。
- 训练：`frontier_belief_mvp_v4.json`（fit_split=train，development_only，
  语料 84,025 行 / 34 task / 34 workflow，5/30/30 门槛通过）。
- 校准：`frontier_belief_mvp_v4_calibrated_dev.json`
  （--development-on-train，74,014 决策点 / 4,460 episode，与 v3 同口径；
  boundary_temperature 1.1、tool_temperature 0.65）。

### v4 vs v3 对比结论

```text
分布内（train 语料 84,025 行，episode+workflow 加权）
  boundary acc      v3 0.92452  ==  v4 0.92452（完全一致）
  boundary NLL      v3 0.2878   ~   v4 0.2881
  decode MAE(running) v3 163.7  ~   v4 163.2
逐项目组件支持（bd_exact/bd_unavail/decode exact）两版完全相同

分布外（matplotlib OOD 9,602 行）
  boundary acc      v3 0.9007   ==  v4 0.9007
  decode MAE        v3 213.7    ~   v4 214.3，coverage 0.918→0.925
  OOD rate          0.392  ==  0.392，全部 backoff

test_id（xarray 26,464 行）
  boundary acc      0.9444 ==（两版一致），decode MAE 170.6~171.3
```

**核心发现**：v3 实际上已经用同一份 84,025 行语料训练过（其工件未记录
fit_projects/语料元数据，易误判为未训练）；v4 与 v3 在全部可用数据上
预测一致（组件哈希不同但行为等价）。因此"直接用现有 67 workflow 数据
重训"没有增量。

### 真正可行的扩展路径（按约束：只动 longhao 文件、不迁移 docker）

1. **追加 rollout**：51 个 train 镜像已就绪且无需拉取，GPU 可用；
   重跑 train 批次（r1/r2 rollout）产生新 workflow_id 与新轨迹，能提升
   boundary history 覆盖、降低 `boundary_unavailable` / OOD rate，再训练
   v5 验证。成本：每批次约 1.5-2h GPU。
2. **超参选择**：`select_frontier_hyperparameters.py`（LOPO）可评估
   minimum_support / 温度等，在不新增数据的情况下提高 exact 命中率，
   存在过拟合风险。
3. **新数据集**：受 root 分区 0 可用限制（docker 属系统所有物，按用户
   约束不迁移），拉取新项目镜像被阻塞；需先解决存储或由管理员处理。

## 状态语义解码改进（2026-08-08 完成，改动者 deepseek）

### 问题定位

shadow 记录中非 running 状态（ready/wait_tool/wait_join/wait_child）的
`remaining_decode_tokens_p50` 全部是 64.0（同一全局 log 桶）。原因：
`fit()` 只对 `state==running_llm` 观测 decode 标签（`remaining_output_tokens`），
非 running 状态的 `(role, state, ...)` 键零观测，预测一路回退到 `("*",)`
全局桶 → 所有非 running invocation 共享同一个常数 p50，调度侧无区分度。

### 改动（beliefkv/predictor/structured_frontier.py）

- `fit()` 对非 running 状态新增状态语义 decode 观测：以
  `label.next_output_tokens`（该边界之后下一次 LLM 调用的输出长度）作为
  decode 目标（running_llm 仍用 `remaining_output_tokens`，语义不变）。
  这样 `(role, wait_tool/ready/join/child, ...)` 键获得各自的状态条件分布，
  不再回退到全局桶。
- 训练摘要新增计数 `state_conditional_decode_demand`。
- 测试：新增 2 个（wait_tool 有/无 next_output 标签的行为差异）+ 更新 1 个
  （ready 状态 decode 从空变为有分布）；全量回归 560 passed
  （仅 2 个 P5D 既有失败）。

### v5 训练与校准

- `frontier_belief_mvp_v5.json`：同语料 84,025 行（6 项目 / 34 task /
  34 workflow），训练观测新增 `state_conditional_decode_demand=71,508`
  （running 的 remaining_decode_demand=60,215 不变）。
- `frontier_belief_mvp_v5_calibrated_dev.json`：--development-on-train，
  与 v3/v4 同口径（74,014 决策点 / 4,460 episode）。

### 效果与验证

```text
非 running 状态 decode 支撑层级（train 语料探针）
  v4（改前）：ready/wait_tool/wait_join/wait_child 全部 global（共享 64.0）
  v5（改后）：全部 backoff（各状态自有分布）

decode p50 变化（train 语料 32,591 行）
  ready      64.0 -> 45.3（全部 9,298 条）
  wait_join  64.0 -> 152.2（全部 14,534 条）
  wait_tool  64.0 -> 64.0（状态分布中位数仍落该桶，但已非全局回退）
  wait_child 64.0 -> 64.0（同上）
  running_llm 不变（64.0/53.8，语义未动）

评估（v5 vs v4，均无回归）
  分布内（84,025 行）：boundary acc 0.92452 ==、decode MAE(running) 163.16 ==
  matplotlib OOD：boundary 0.9007 ==、decode MAE 214.3 ==、OOD 0.392 ==
  xarray test_id：boundary 0.9444 ==、decode MAE 171.3 ==
```

### 说明与下一步

- 改进直接作用于调度侧最关心的非 running 状态：wait_tool/ready/join 的
  decode 预测现在携带状态条件信号（ready≈45、wait_join≈152 等），不再全是
  常数 64.0；评估指标不变是因为标准评估只测 running_llm 的 decode。
- wait_tool/wait_child 的中位数仍落在 64.0 桶（其状态分布中位数本身在此），
  更细的区分需要更细键的支持——按用户计划，下一步增加训练数据
  （追加 rollout）提升支持度后重训验证。

## 追加 rollout 与 v6 重训（2026-08-08 完成，改动者 deepseek）

### SWE 训练语料结论

冻结的 collection_v4 计划中 train 拆分即这 67 个 workflow（7 个 train 项目），
已全部运行并导出；新增实例需改冻结计划且拉新镜像（受 root 分区 0 可用
阻塞，docker 属系统所有物按约束不迁移）。因此按用户指示追加 rollout
（p6-013 / p6-014，同任务新轨迹）。

### rollout 运行结果

```text
p6-013 rollout（20260807T173852Z）   p6-014 rollout（20260808T035052Z，重跑）
system JCT 8/8                       8/8
LLM 请求 883                          992
工具调用 1,681                       2,391
时长 7,233s                          7,649s
交付失败 0                            0
training_eligible true               true（runtime_source_stable）
导出 19,065 决策点                   20,884 决策点
```

### 高压卡死诊断与恢复（重要发现）

- 首轮 p6-014 在约 44 分钟时整体挂起：8 个 workflow 的 llm_submit 到达
  服务器队列（5 running + 4 waiting），但 GPU 0%、调度线程 39% CPU 空转。
- py-spy 定位：调度线程卡在
  `_drive_restore_obligations -> _try_queue_restore_lease_funding ->
  _restore_funding_preview -> _offload_previews -> _owner_blockers`；
  高压下（`/get_load` 190K > KV pool 163,840）restore funding 的 offload
  预览是 O(页面 x owner) 重计算，单步极慢，表现为调度器假死。
- 当时将恢复误归因于把 `MAX_TOTAL_TOKENS` 从 163,840 提到 327,680。
  服务端实际明确报告 requested 327,680 超过 profiled 167,816，并继续只分配
  167,816 tokens；因此“KV pool 扩容成功”和由此计算的 0.494 压力均不成立。
  重跑成功只能视为 workload/运行轨迹差异，不能证明扩容解决了卡死。
- 2026-08-08 修复将 funding preview 改为从 revision-cached migratable roots
  有界物化局部 bundle，避免每次扫描所有 context page/owner；实验 runner 也会
  查询 `/get_server_info`，实际 pool 小于要求时直接拒绝采集。该优化尚待一次
  固定高压 trace 的 GPU 验证。

### v6 训练与评估（train_frontier_mvp.py 已支持多 --data-dir）

- 语料：原 84,025 行 + rollout 30,686 行 = **114,711 决策点**（6 项目 /
  34 任务；rollout 增加同任务新 episode 观测）。
- `frontier_belief_mvp_v6_calibrated_dev.json`：development 校准，
  interval slack 明显收紧（remaining_decode 171.9->140.9、
  external_wait 723.6->64.1）。
- 细键效果（同 67,492 行探针，v5 vs v6）：wait_tool decode p50
  64.0->**76.1**、wait_child 64.0->**76.1**、ready 从单一 45.3 变为
  **53.8/45.3 混合**（按上下文/工具族细分）；wait_join 保持 152.2。
- OOD 无回归且校准覆盖率提升：matplotlib 0.9255->0.932、
  xarray test_id 0.9396->0.9455；boundary acc / decode MAE 不变。
- 说明：composite `support_level` 仍为 backoff（decode exact 需 6 元键
  支持≥4 才可达，属架构设计）；本轮证明"更多数据 -> 更细键 -> 状态
  分布区分度更高"。

## 当前进度与待办快照（2026-08-08，改动者 deepseek）

### 模型产物

```text
frontier_belief_mvp_v3_calibrated_dev.json  旧词表/旧语义（shadow 用，已冻结）
frontier_belief_mvp_v4_calibrated_dev.json  对齐词表重训（≈v3，对照证据）
frontier_belief_mvp_v5_calibrated_dev.json  状态语义解码改进（v4 + 71.5k 状态观测）
frontier_belief_mvp_v6_calibrated_dev.json  v5 + 2 个 rollout（114,711 行，最新）
```

### 数据产物

- 训练语料：`experiments/processed/p6_agent_semantics_v1/`（原 13 批次，
  84,025 行）+ `p6_agent_semantics_v1_rollout/`（p6-013/014 r1，
  39,949 行），均 `formal_training_eligible=true`。
- rollout 原始运行：`experiments/raw/p6_agent_semantics_v1/`
  p6-013 `20260807T173852Z`、p6-014 `20260808T035052Z`。

### 已解决/已处理的问题

1. **高压调度假死**（p6-014 首轮）：restore funding offload 预览在
   KV 压力下单步极慢。请求 327,680 tokens 被服务端裁剪为实际 167,816，
   因而不能把重跑成功归因于扩容；当前已完成有界 root-preview 代码修复，
   尚待固定 trace GPU 验证。
2. **rollout 导出布局**：新批次无 `gpu0` 层，数据集平铺在根目录，
   训练 loader 期望 `p6-0*/**/dataset/` 布局；已把平铺文件移入
   `dataset/` 子目录并扩展 `train_frontier_mvp.py --data-dir` 支持多目录。
3. **中断残留清理**：首轮被中断的 p6-014 运行目录
   `20260808T021655Z`（552MB，不完整不可用）已删除；该时段的服务器
   audit 仍保留在 `experiments/shadow/p6_rollout_013_014/gpu0/server/`
   （首台服务器目录），如需审计证据仍可查。
4. **服务器关停**：server_r2 已优雅关停（shutdown_state=acknowledged），
   GPU0/GPU1 均已释放、无残留进程。

### 待办（未执行）

1. 在开发 split 上验证新的 read-only risk shadow 的 coverage、OOD gate、
   candidate/scenario P99、publish age 和 would-action regret；不要重跑旧 heuristic
   predictive A/B。
2. 继续追加 rollout（其余 train 批次 p6-015~021 或再跑 r2），
   进一步提升细键支持；或按需调整 `empirical_minimum_support`。
3. 用固定高压 trace 验证 funding preview 开销和 scheduler progress，不比较 JCT。
4. 预测动作开放前执行一次短时 w8 correctness smoke；论文级 test evidence
   仍需在模型/策略冻结后采集。

## P6 risk shadow 实现检查点（2026-08-08）

本轮已完成但尚未做 GPU 性能实验：

- `LocalFrontierPrediction` 完整分布序列化，避免只透传少量 p50 字段；
- `BeliefScope` 按 JOIN/blocker/message 形成 closure-complete scope；
- `FrontierScenarioComposer` 生成联合 particles、top-K 和 OTHER；
- `CandidateTimelineEvaluator` 在候选 GPU/PCIe 时间线上解析 child completion 与
  JOIN reentry，不再对 raw demand 直接取 max/min；
- 第一版只生成 `A0/PREPARE_HOST/PREFETCH_GPU`；backoff/OOD 禁止 prefetch；
- `PredictiveRiskShadowObserver` 位于异步 latest-wins worker，异常不会使 P5
  observed plan 失效；审计字段固定 `decision_authority=read_only_shadow`；
- P5 在线 ordering/victim 已移除预测影响；
- P6 batch runner 使用服务端实际 KV pool 做压力分母和最低容量 preflight。

当前限制：局部分布查表仍在 safe-point capture 中同步、有界执行；候选 timeline
已模拟 GPU service、PCIe 和 JOIN/reentry，但 future KV growth 的 HBM chance
constraint 尚未接入。下一轮必须分别测量 lookup 与 worker P99，并补齐 future HBM
feasibility，不能把当前 shadow 称为完整 schedulability test。

验证：完整 serving 回归 `570 passed, 3 skipped, 3 subtests passed`，
agent/runtime 回归 `126 passed`。下一次
GPU 实验应先验证 shadow coverage/开销和 funding preview progress，不评价 P6 JCT
收益，也不生成旧式 prediction-used A/B 结论。
