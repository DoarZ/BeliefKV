# BeliefKV P6 预测式 JointPlan 执行计划

日期：2026-08-11

状态：当前 P6 开发的权威执行计划。本文覆盖旧文档中以 morphology action flip、独立
useful-action oracle 和仅做 KV 预测为核心的后续安排；历史实验仍保留用于追溯，不再决定开发顺序。

## 1. 目标与核心假设

当前关键路径只有一个：尽快验证 FrontierBelief 是否能与 RCCG、KV residency、admission 和
running retraction 共同提高单 GPU 动态 agent workflow 的吞吐。

```text
SPAWN / TOOL / RETURN / JOIN 运行时事件
                  |
                  v
        RCCG causal frontier
                  +
        FrontierBelief scenarios
                  |
                  v
  JointPlan: execution + admission + KV + retraction
                  |
                  v
更多 action 被解锁、更少 pressure-path stall
                  |
                  v
      successful workflows / hour 提升
```

预测准确率、迁移次数和 cache hit 不是最终目标。主要结果必须落到 successful workflow
throughput，并由 tool/JOIN unlock、admission stall、transfer 和 retraction telemetry 解释。

## 2. 已确定的设计决策

1. 删除 morphology 作为独立策略分支，不再做 byte-only/morphology promotion、veto 或专用
   canary。保留 `bytes + extent_count + contention` 作为唯一 transfer cost/OOD 模型。
2. 取消独立 causal useful-action oracle 作为前置 gate，不再先做复杂反事实模拟再开放在线动作。
3. 保留轻量在线动作归因，将每笔预测动作标记为 useful、wasted、too-late 或 censored；该归因
   与端到端 A/B 同时完成，不阻塞实现。
4. 增加 `parallel_analysis_2to3` workload profile。每个 parent 创建两个正交分析 child，存在
   独立依赖问题时可创建第三个；任务内容、工具调用、返回顺序和运行时间仍由 agent 决定。
5. 第一轮预测只在 SPAWN 已发生、child 集合已被 RCCG 观测后决策，不预测未知 fan-out。
6. 不新增第二套抢占状态机。FrontierBelief 只为现有 selective running retraction 提供
   `EXPAND/CLOSE/HOLD` 注解，物理提交继续复用 P5 transaction、restore obligation 和 safe point。
7. 当前不搭建外部论文 baseline，不引入 peer-agent workload，不实现通用 MPC、KV 压缩或自定义
   DMA。系统完整后再扩展正式对比。

## 3. 在线联合算法

### 3.1 Frontier 分类

每个可运行或等待的 invocation 被投影为以下短期类别：

- `EXPAND`：预计在有限 GPU service 内到达 TOOL 或 SPAWN，继续执行将启动外部并行工作；
- `CLOSE`：预计在有限 GPU service 内 RETURN/FINAL，或其完成可能满足 JOIN；
- `HOLD`：短期仍为长 decode，不能解锁工具、child 或 join；
- `UNKNOWN`：预测支持不足，完全回退 observed P5 排序。

分类来自同一个 FrontierBelief scenario，不训练独立的抢占分类器。RCCG 决定 JOIN_ALL、JOIN_ANY、
依赖成员和当前 causal frontier，模型只预测各 invocation 的局部 boundary、token demand、工具等待和
reentry demand。

### 3.2 JointPlan 决策顺序

```text
关键事件触发
  -> 捕获 immutable PolicyInput + SafePointPhysicalSnapshot
  -> 生成 closure-complete FrontierBelief scenarios
  -> RCCG 传播 TOOL/SPAWN/RETURN/JOIN readiness
  -> 物理化 PREPARE_HOST 和 admission/retraction 候选
  -> 单个 JointPlan 联合选择：
       admission set
       execution priority
       PREPARE_HOST
       optional running retraction + replacement
  -> safe point 重建 live PhysicalBundle
  -> certificate 验证后事务化提交
  -> ACK / service / terminal 归因
```

只在 SPAWN、TOOL_START/RETURN、CHILD_RETURN、JOIN 状态变化、HBM pressure crossing 和 transfer
ACK 等事件触发完整预测规划。普通 scheduler tick 继续使用最近有效 seed，不同步运行 scenario
simulation。

### 3.3 预测式 PREPARE_HOST

第一阶段只开放非破坏性的 `PREPARE_HOST`：

- GPU KV 保留，只在 Host 建立 shadow；
- 仅当预计未来 pressure-path D2H 可被当前 PCIe slack 隐藏时发布 intent；
- 同时最多一笔 predictive D2H；
- PCIe 繁忙、service model OOD、bundle generation 改变或 deadline 过期时回退 P5；
- `PREFETCH_GPU` 暂不开放，避免同时引入 HBM-time 风险。

### 3.4 Frontier-Aware Retraction

现有 `ObservedRetractionPlanner` 保留 admission stall、active floor、reclaim capacity、blocker closure、
cooldown、最大抢占次数和 restore obligation。预测只改变候选排序与保护关系：

1. 高置信度 `EXPAND` 和 JOIN-critical `CLOSE` 进入保护集；
2. 从 `HOLD` 中优先选择 reclaimable private KV/lock closure 较大的 victim；
3. replacement 优先选择 `EXPAND`，其次选择 JOIN-critical `CLOSE`；
4. 只有 retraction 后至少一个 replacement 能真实进入 batch 时才提交；
5. 预测支持不足时保持当前 observed ranking；
6. workflow fairness 只作为平局规则，不限制单 workflow 的并行 child 数量。

JOIN_ALL 下优先推动能够启动外部工具或可能成为最后完成者的 child；JOIN_ANY 下优先最可能较早
成功返回的 child。JOIN 满足后，其他不再需要的 child 由 runtime 原生取消或降级，不能继续占用
预测保护额度。

## 4. Workload 与训练数据

### 4.1 Fan-out Profile

新增配置：

```yaml
subagent_fanout_profile: natural | parallel_analysis_2to3
```

`parallel_analysis_2to3` 的约束为：

- Child A：代码路径、候选符号和关键不变量分析；
- Child B：问题复现、失败条件和回归测试分析；
- Child C：仅在存在独立依赖、协议或兼容性问题时创建；
- child 必须在同一 parent turn 中并行提交；
- child 不直接修改 parent workspace，由 parent 在 JOIN 后统一实现和测试；
- 不允许复制同一 task、固定 sleep 或人为指定返回时间来制造并发。

该 profile 是受控的 naturalistic fan-out stress workload，不用于声称真实系统中所有 parent 都会
创建多个 child。正式实验后续必须同时报告 `natural` 与 `parallel_analysis_2to3`。

### 4.2 旧 Trace 的使用边界

旧的单 child trace 继续用于训练局部 boundary、output/prompt demand、tool survival 和 role-conditioned
行为，不因 fan-out 增加而废弃。以下分布需要新数据补充：

- SPAWN fan-out 频率；
- 多 child role 组合；
- fan-out 下的 HBM pressure 和调度反馈。

第一轮保持模型冻结并在 SPAWN 后预测，由 RCCG 确定性组合 JOIN。只有当新 workload 中 exact/support
覆盖明显不足、预测持续退化到 backoff 时，才采集 fan-out train shard。训练和 A/B 按 repository
隔离，禁止同一任务同时用于模型更新和收益报告。

## 5. 分阶段实施

### R0：冻结开发基线

固定 P5 commit、模型 artifact、transfer artifact、GPU/model/KV pool、workload manifest、arrival
schedule 和随机种子。输出单一 baseline manifest；后续配置变化必须产生新版本，不覆盖旧结果。

### R1：策略清理与控制面修复

实施文件：

- `beliefkv/policy/risk_shadow.py`：删除 morphology 双分支和 promotion/veto；
- `beliefkv/policy/service_curve.py`：保留单一 transfer cost，修正 artifact/online 独立门槛；
- `beliefkv/runtime/sglang_v052rc1.py`：generation-aware submit attribution 和动作结果归因；
- `third_party/sglang/.../hiradix_cache.py`：observer fail-open，不能破坏原生 DMA 状态机；
- 当前设计、架构和实验文档：失效 morphology 结论明确标记为 historical/invalidated。

验收：

- 固定 snapshot replay 中 P5 observed 决策不变；
- oracle/归因消费的 native transfer 均有完整 generation、owner、extent 和 bytes；
- callback error 不影响 ACK/loading/metadata cleanup；
- 无 orphan command、lease、restore obligation 或 transfer metadata；
- 只运行 CPU 回归和固定 replay，不运行 GPU 长实验。

### R2：Fan-out Workload Bring-up

实施文件：

- `beliefkv/experiments/deepagents_swebench.py`：新增 profile、正交 child prompt 和只读/隔离约束；
- `scripts/prepare_deepagents_server_config.py`、`scripts/run_p6_collection_batch.py`：传递并记录 profile；
- workload summary：记录每个 parent 的 fan-out、并发 child 峰值、JOIN 类型和 child 返回跨度。

先做一个 CPU/fake-backend 功能测试，再做一次短 GPU smoke。验收：

- 每个有效 fan-out parent 创建 2 或 3 个不同职责 child；
- SPAWN/RETURN/JOIN 事件成对且 RCCG 无悬空 child；
- parent workspace 无 child 并发写冲突；
- baseline 与 treatment 使用完全相同的 fan-out profile；
- 不因某次未出现期望时序而筛选或重复 workload。

### R3：预测式 PREPARE_HOST Canary

预测 worker 只在关键事件运行，最多发布一笔自然正期望收益 PREPARE。intent 必须经过 JointPlan 和
safe point，不允许注入命令、绕过 certificate 或选择负收益动作。

动作闭环：

```text
belief -> intent -> JointPlan selected -> rematerialized
       -> D2H dispatch -> ACK -> shadow consumed/wasted -> terminal cleanup
```

验收重点是控制面正确和动作真实执行，不设置独立 oracle 前置门槛。没有自然 intent 时直接记录
`no_positive_action`，不得通过恢复 morphology 分支或反复筛选 trace 制造动作。

### R4：Frontier-Aware Retraction

在 `RunningRetractionCandidate` 和 `RetractionReplacement` 中增加可选的 frontier class、预测支持、
service-to-boundary 和 join criticality；禁用预测时序列化结果必须与当前 P5 一致。

先运行 shadow：记录 observed 与 frontier-aware victim/replacement 分歧及预计 unlock。随后只允许一笔
预测改变的 retraction canary，并复用现有 transaction、D2H、restore、replacement admission 和
service-grace 路径。

验收：victim 被恢复并重新获得 service；replacement 在同一事务后真实进入 batch；无重复抢占、
restore debt 泄漏和 active-count collapse。

### R5：短期端到端 A/B

仅比较两个 arm：

```text
A: P5 observed JointPlan
B: P5 + predictive PREPARE_HOST + Frontier-Aware Retraction
```

两个 arm 使用相同 task manifest、fan-out profile、arrival schedule、模型和系统配置。开发阶段执行三组
交替顺序的配对运行；动态内部轨迹不要求逐事件一致，按 task/workload aggregate 比较。

主要指标：

- successful workflows/hour；
- task success、WORKFLOW_END 和 runtime failure；
- tool starts/hour、JOIN completions/hour；
- GPU busy fraction、running request count、prefill batch fill；
- admission wait P50/P95 和 pressure-path D2H stall；
- predictive action useful/wasted/too-late/censored；
- retraction count、replacement service、victim restore/recompute cost；
- planning CPU overhead 和 stale intent rate。

开发阶段认为“值得继续”的最低证据是：三组配对中至少两组 workflow throughput 提升、总体 task
success 不下降、没有新增 liveness/correctness 问题。5% 仅作为需要继续优化的工程参考线，不作为
论文统计结论。

## 6. 结果诊断与停止规则

| 结果 | 下一步 |
|---|---|
| 无自然预测动作 | 检查模型 support 与策略收益；不恢复 morphology，不筛选 trace |
| intent 多但 safe point 大量 stale | 缩小 read-set、减少 planning latency，不改预测目标 |
| PREPARE 成功但均未消费 | 预测价值不足，停止 PREPARE 主线 |
| retraction 改变 victim 但 replacement 未获 service | 修复 JointPlan/retraction 接口，不训练新模型 |
| stall 降低但吞吐不变 | 瓶颈不在 KV/admission，停止增加调度复杂度 |
| throughput 提升且归因闭合 | 冻结实现，进入正式数据量、natural workload 和 baseline 评价 |

在 R5 前不增加新的预测目标、模型架构或动作类型。任何新机制都必须解释它解决了当前哪一个可观测
阻塞，否则不进入关键路径。

## 7. 2026-08-11 实施进度

| 阶段 | 代码状态 | 实验状态 |
|---|---|---|
| R0 | 完成；生成不可覆盖的 `configs/p6/predictive_joint_v1/baseline_manifest.json` | 不需要 GPU |
| R1 | 完成；统一 transfer cost，native submit-time generation/owner/extent 归因，observer fail-open，动作结果 ledger | 216 项相关 CPU 回归通过；固定 observed 行为未改变 |
| R2 | 完成；`parallel_analysis_2to3`、只读正交 child、fan-out/JOIN/return-span 汇总、1500 ms 分批到达已接入 | fake-backend gate 通过；短 GPU smoke 待空闲 GPU |
| R3 | 完成；关键事件触发、单笔 PREPARE 上限、JointPlan/safe-point/ACK 路径及八段归因 analyzer | 自然在线 canary 待空闲 GPU；零动作记为 `no_positive_action`，不重跑筛选 |
| R4 | 完成；候选可选 `EXPAND/CLOSE/HOLD/UNKNOWN` 注解、shadow diff、单笔 canary 和 restore/service gate | shadow 与单笔在线 canary 待空闲 GPU |
| R5 | 完成运行基础设施；顺序为 A-B/B-A/A-B，一次性 runner 和汇总器已实现；修复 retraction 对陈旧 `prefix_indices` 的物理释放边界 | v7 A 完成 8/8，v7 B 因 duplicate allocator/Radix ownership 退出，整组失去性能资格；修复后 2.659 GB D2H/H2D deterministic gate 14/14 通过，v9 已按相同实验契约重新冻结，六次正式运行待执行 |

R3/R4 新 analyzer 分别为 `scripts/analyze_predictive_joint_canary.py` 和
`scripts/analyze_frontier_retraction_canary.py`。R5 使用
`scripts/run_p6_predictive_joint_ab.py` 逐项执行冻结 run ID，并由
`scripts/analyze_p6_predictive_joint_ab.py` 汇总。R5 的 successful workflows/hour 使用正常完成且
满足 system-JCT gate 的 workflow 数计算；SWE 本地 correctness 单独报告，不再决定性能样本是否完成。

历史 `predictive_canary.py`、promotion/veto replay 和 morphology 专用报告仅用于复现实验演进，
不再是在线权限、R3 canary 或 R5 A/B 的输入。

### R5 ownership interruption 与 v9 边界

v7 的 `pair-1-1-a` 是完整的 8/8 observed characterization；`pair-1-2-b` 在没有选择任何
predictive physical action 的情况下触发 `multiple live Radix extents reference the same device
index`，因此不能把失败归因于预测策略，也不能继续拼接 v7 配对结果。修复保持一致性检查
fail-closed，并在实际 retraction 时根据 live Radix ownership 与 allocator free state 过滤释放页。

确定性 high-fragment restore gate 已完成 2,659,221,504-byte D2H/H2D、340 个 post-restore
service sample 和完整 shutdown 守恒。提交前的可移植性修订只改变 orchestration 源码，不改变
实验契约；当前冻结输入位于 `configs/p6/predictive_joint_v9/`，source tree SHA-256 为
`b95b5e4e7b91dce4698af5a964258698f25a19251ba8df411c548b9027a48a41`。完整证据与
限制见 `docs/experiments/beliefkv_p6_r5_retraction_ownership_repair_2026-08-11_zh.md`。
