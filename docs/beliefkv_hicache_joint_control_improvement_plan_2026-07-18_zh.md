# BeliefKV 当前版本改进方案：基于 HiCache 数据面的 Agent/KV 联合控制

初始日期：2026-07-18  
最新修订：2026-07-20  
状态：实施中；P0/P1 已通过，P2 可靠性修复通过 CPU gate，真实 GPU gate 待复验  
依赖分析：

- [HiCache / HiSparse / Theta KVPool 对 BeliefKV 的启发与差异分析](hicache_theta_kvpool_implications_2026-07-18_zh.md)
- [BeliefKV 面对动态 Agent Workflow 需要注意的地方](beliefkv_dynamic_agent_workflow_considerations_2026-07-20_zh.md)
- [P2 Physical Causal Lease 与原子 Bundle 执行](experiments/beliefkv_p2_physical_bundle_2026-07-19_zh.md)
- [ScaleSim](https://arxiv.org/abs/2601.21473)、
  [AugServe](https://arxiv.org/abs/2512.04013)、
  [ThunderAgent](https://arxiv.org/abs/2602.13692) 和
  [CONCUR](https://arxiv.org/abs/2601.22705) 的竞争边界

## 1. 文档目标

本文将当前 BeliefKV 原型改进为一个可验证、可回退的动态 agent/KV 联合控制系统。目标
不是重新实现多级 KV 缓存，也不是只优化一种固定的 parent-child fork/join，而是在
SGLang HiCache 提供的数据面之上，面对运行时逐步展开的 RCCG，联合回答四个问题：

1. 下一批应优先运行哪个 root workflow 和哪个 agent invocation；
2. 哪些 request 应 admission、等待 restore，或者继续 parked；
3. 哪些物理 KV bundle 应继续占用 HBM、建立 Host 副本、提交 CPU-only 或恢复到 GPU；
4. 在未来 agent、consumer 和工作集尚未确定时，如何利用局部 belief frontier 做可撤销
   准备，同时避免预测覆盖已观测事实和物理安全约束。

BeliefKV 的在线决策必须以一个不可拆分的接口表达：

```text
JointPlan
  = ExecutionIntent
  + AdmissionIntent
  + ResidencyIntent
  + TransferDependencies
```

`AdmissionController`、SGLang waiting queue、prefetch/offload planner 不得继续各自生成互相
独立的优先级。即使关闭预测，事件驱动 baseline 也必须通过同一个 JointPlan 接口，确保
性能差异来自策略而不是执行路径不同。

完整部署 ScaleSim、AugServe、ThunderAgent 等外部系统不阻塞 P2-P6 的功能实现，但它们的
策略空间、输入假设和输出动作必须从 P2.5 开始进入统一 reference-policy 接口。BeliefKV
功能稳定前使用同数据面的 shadow/replay 和 hindsight oracle 验证算法边界；功能稳定后再做
原生系统端到端对比。不能以“动态 workflow 更真实”为理由，把所有竞争验证推迟到系统完成
之后。

本文只定义可以落到当前仓库的模块、接口、算法、测试和退出条件。以下能力不作为
BeliefKV 的创新：

- GPU/CPU/L3 多级缓存；
- page/node 级 D2H、H2D、write-through 和 write-back；
- 通用异步 prefetch、I/O 合并、Host layout 和传输/计算 overlap；
- 单独的 CPU shadow copy 或 PCIe idle-time backup；
- 用 MLP、GNN 或其他模型输出一个 offload/prefetch score。

实现范围采用递进策略：

1. 首先完整优化同步阻塞的 subagent fork/join，因为当前 Deep Agents 路径能够稳定观测
   `SPAWN/JOIN_WAIT/RETURN`；
2. RCCG、lease 和 JointPlan 从第一天就按局部边语义支持 parent continuation、peer
   handoff、message 和 cyclic reactivation；
3. P3 开始加入真实 multi-agent workload，不把 blocking subagent 当作系统的最终边界；
4. 任何未被 adapter 可靠识别的异步关系采用保守 `READY/RUNNING` lease，不错误迁移 parent。

BeliefKV 的候选研究贡献重新收敛为：

> 在只知道运行前缀 `G_t` 的动态 MAS 中，BeliefKV 将“谁推动 workflow 前进”和“哪些
> physical KV bundle 占用 HBM/PCIe”放入同一个在线 JointPlan，并用局部 frontier belief
> 描述尚未发生的 spawn、handoff、message consumer、循环和资源增长。

ICML 2026 相关工作已经覆盖“预测下一次调用/工具时间，再决定 Preserve、Swap、Discard、
prefetch 或 eviction”的宽泛表述。因此该表述不能单独成为 Major contribution。P5.5 将额外
检验一个更窄的候选 insight：根据在线 action frontier 优先完成能够产生合法 tool/spawn/
handoff 的 decode quantum，并依据校准置信度控制 KV 动作从 KEEP、SHADOW 到 COMMIT 的
可逆程度。只有其 oracle 明显优于带 hindsight 信息的 ScaleSim/AugServe reference policy，
P6 才将其升级为核心算法；否则继续使用更保守的 Dynamic Frontier JointPlan。

Reveal-and-Commit 只是该联合框架中的条件性动作：当一个当前可运行的 agent 能在强制内存
决策前消除高代价场景分歧时才启用。它不再被预设为所有 blocking subagent workload 的
核心算法。

该表述仍是待验证假设。只有第 15 节的 Go/No-Go 条件满足后，才能作为论文主要贡献。

## 2. 当前版本基线与主要缺口

### 2.1 已有安全闭环

当前代码已经具备：

```text
RuntimeEvent / request metadata
  -> RuntimeCausalContextGraph
  -> causal frontier / prediction / residency
  -> admission / reactive transfer / shadow
  -> RadixArbiter
  -> SGLang scheduler safe point
  -> HiCache node operation
  -> actual-byte ACK
  -> PageOwnershipIndex residency commit
```

必须保留以下正确性边界：

- SGLang allocator、Radix topology、node lock 和实际 KV tensor 是物理真相源；
- RCCG 是 invocation/context 因果状态的真相源；
- `PageOwnershipIndex` 只是带 generation 的 CPU mirror；
- policy 只能产生 intent，不能直接修改 tensor 或假定 DMA 已完成；
- residency 只在 ACK 后提交；
- active reader、engine lock、semantic pin、ancestor/leaf closure 继续由
  `RadixArbiter` 强制检查。

### 2.2 当前决策被拆成四套局部排序

当前 `BeliefKVController.tick()` 的顺序是：

1. 独立生成每个 context 的 remaining-time 边际预测；
2. `AdmissionController` 按当前 HBM、workflow share 和 causal frontier 选择 request；
3. `ReactiveTransferPlanner` 在 admission 失败后选择 context victim；
4. 无紧急动作时，`ShadowController` 再选择 shadow context；
5. SGLang 适配层随后独立重排 waiting queue。

这会产生三个问题：

- admission 选择了 agent A，但 transfer planner 不知道 A 的完整物理启动成本；
- planner 可能为一个低优先级 workflow 预取，随后 waiting queue 却先运行另一个 workflow；
- eviction、prefetch、shadow 和 agent 调度分别使用不同 rank，无法解释一次决策的整体
  HBM/PCIe 后果。

### 2.3 当前预测对象不足以支持联合控制

`RemainingTimePrediction` 只有单 context 的 `p50/p90/p95`、resume probability 和
next-event distribution。`WAIT_CHILD/JOIN` 通过 `min/max` 合成边际分位数，不能表达：

- 多个未来事件的联合场景；
- 哪个 ready invocation 能揭示场景；
- 不同场景将新增哪些 context/request；
- 场景对应的 physical bundle、closure 和 HBM-time；
- 不同场景下最优 KV 动作是否一致。

改进方案不要求更复杂的神经网络，而是要求预测器输出结构化、经过校准的短期场景。

### 2.4 当前物理成本估计不足

当前 transfer watchdog 仍主要使用：

```text
transfer_ms = bytes / configured_bandwidth + fixed_overhead
```

真实实验已经显示 callback、allocator、closure 和 node lock 会使该估计显著偏离实际。
HiCache 的 overlap 还会使总 DMA 时间与真正阻塞计算的时间不同。后续优化目标必须改成：

```text
unhidden_stall_ms
= compute_wait caused by load/write/allocator/closure
```

### 2.5 当前真实实验锚点

后续改进必须以
[2026-07-17 admission liveness 最终实验](experiments/beliefkv_admission_liveness_repair_2026-07-17_zh.md)
为 correctness 锚点，而不是把它解释成性能基线。该实验使用：

```text
GPU                 RTX 6000 Ada
模型                Qwen3-Coder-30B-A3B-Instruct-FP8
SGLang              0.5.2rc1
KV pool             163,840 tokens，约 15 GiB
并发                8 个 planned Deep Agents/SWE-bench workflow
HiCache             ratio=2，write-back
BeliefKV            predictor=false，shadow=false，prefetch=true
```

已证明的事实：562/562 个 request 完成，589/589 条 transfer 收到 ACK，峰值 KV pool
占用达到 99.948%，没有 watchdog、scheduler exception 或系统级停滞。

尚未解决的性能事实：

- admission wait P95 为 46.380 秒，最大 58.320 秒；
- `no_migratable_marginal_pages` 出现 1,128 次；
- 240 条迁移因 node locked/loading 只部分完成；
- 总计发生约 14.83 GB 实际 D2H/H2D；
- 8 个 workflow 中 7 个达到 agent recursion limit；
- predictor 和 shadow 均未启用，不能从该 run 推断预测策略收益；
- 动态生成路径未完全确定化，不能直接用不同 run 的 JCT 作正式 A/B。

因此，联合控制的第一个任务不是宣称降低 JCT，而是解释 locked/closure/reclaim failure、
冻结可比 workload，并量化 current reactive policy 到 physical oracle 的 gap。

### 2.6 H2D retry storm 是独立的 baseline 缺陷

P1 短跑中共有 292 个零字节 H2D reject，其中 268 个来自同一 context：

```text
deepagents-context:fdc67d3fd0c9a938
reactive-287 ... reactive-572
时间跨度 7,176.690 ms
相同失败原因 268/268
```

失败原因均为 H2D ancestor closure 或 HiCache device allocation 不满足。当前
`ReactiveTransferPlanner` 只用 context 的 CPU_ONLY marginal bytes 与逻辑
`available_bytes` 判断 prefetch；`RadixArbiter` 虽然在 resolve 时检查当前 mirror 的
ancestor closure，但 authoritative HiCache 仍可能因 allocator、并发 lock/loading 或
TOCTOU topology mutation 拒绝。更关键的是，backend ACK/telemetry 的 blocker 没有形成下一轮
planner 的负反馈，因此同一个 IMMINENT context 在物理状态未变化时每个 tick 都可被重新选择。

当前 `_blocked_context_epochs` 只覆盖 `RadixArbiter.resolve()` 返回零 page action 的本地拒绝；
backend 已接收命令后返回的零字节 reject 不会进入该集合。即使简单扩展该集合，它仍然过于
粗糙：无法表达“等待哪个 ancestor、多少 HBM、哪个 lock 或哪次 engine completion”，容易在
“每 tick 重试”和“永久屏蔽整个 context”之间二选一。

因此 retry storm 必须在 P2 前作为强 reactive baseline 的独立修复。P2 的 physical preview
减少首次错误选择；event-gated retry guard 保证 preview 与执行间仍发生竞态时不会形成风暴。
二者不能互相替代。

### 2.7 2026-07-20 的范围修正

当前证据要求对原计划做三项修正。

第一，当前 FRESH child 与 parent 的 exact prefix affinity 极低。冷启动 child 只命中约 5
tokens，parent-child 公共 KV 相对 pair 物理并集约 0.093%。显著复用主要来自
parent-parent 和 sibling/template-compatible child-child，而不是因果 parent-child 边。因此：

- RCCG causal edge 不能转换成 cache-affinity edge；
- parent-triggered offload 继续只操作 `EXCLUSIVE_SUFFIX`；
- shared-owner lease 保留为 correctness 和通用 prefix accounting，不作为主要性能 insight；
- P3 不再假定 child 可以继承或复用 parent KV。

第二，当前 P2 workload 只是半动态 blocking subagent workload：8 个 workflow 都创建两个
child，基本只有一次 fork/join，没有 nested spawn、动态 quorum、复杂取消或 cyclic
multi-agent。它不能支撑“动态 MAS”主张。

第三，spawn 后已经观测到的 child identity、FRESH 关系和 declared join 不需要预测，但这不
意味着 workflow 变成静态 DAG。以下状态仍然未知：

```text
spawn 前的 fan-out/role/nesting
child 后续 LLM/tool 路径、返回长度和 completion
join any/all/quorum 的实际满足路径和取消
multi-agent next speaker、handoff target、message consumer
review-revise/test-fix 循环是否继续
未来 context KV growth 和执行时 physical actionability
```

因此 P3 的问题不是预测完整 DAG，也不是只在 spawn 后做确定性轮换，而是维护：

```text
observed RCCG facts G_t
  + local probabilistic frontier H_t
  + exact physical state R_t
```

JointPlan 只能调度 `G_t` 中已经 READY 的 invocation；`H_t` 只能影响已存在 agent 的排序和
KEEP/PREPARE/PREFETCH 等可撤销动作，不能创建虚构 request 或授权不可行的物理迁移。

### 2.8 竞争边界与分层对比原则

现阶段不能再把以下单点作为 BeliefKV 的创新：

- 根据未来调用远近做 KV eviction/prefetch，ScaleSim 已使用 invocation distance；
- 根据输出长度和工具耗时选择 Preserve/Swap/Discard，AugServe 已使用预测和显存-时间成本；
- 将 agent phase、waiting queue 和 KV pause/restore 联合管理，ThunderAgent 已实现；
- 根据 KV pressure/hit rate 调整 active agent 数，CONCUR 已使用 AIMD feedback control。

BeliefKV 后续比较分为三层：

```text
L1: hindsight/oracle policy replay
    给竞品真实 invocation distance、output/tool duration 或 phase metadata
    用于判断 BeliefKV 的目标和动作空间是否存在上界收益

L2: same-data-plane reference policy
    所有策略通过同一个 PolicyInput/PolicyOutput 在 BeliefKV/SGLang 上执行
    用于隔离调度/KV算法收益，避免引擎版本和数据面差异

L3: native end-to-end system
    在 BeliefKV P5/P6 稳定后部署开源原系统
    用于验证 reference policy 的忠实度和最终端到端竞争力
```

L1/L2 必须先于 predictive policy 实现；L3 可以延后，但在形成论文系统结果前必须完成。对
不支持动态 peer/subagent 语义的竞品，同时报告 common-denominator workload 和 BeliefKV
完整动态 workload，不把“不支持”直接记为性能失败。

## 3. 目标架构

```text
Agent runtime events                 Historical observations
 spawn/join/tool/return/              transition/demand/tool
 handoff/message/cancel                        |
          |                                    v
          v                           Local Frontier Belief H_t
 Observed RCCG G_t                  top-K transition scenarios
 causal control facts                confidence/OOD/consumer
          |                                    |
          +----------------+-------------------+
                           |
Data-consumer index         |           SGLang physical state R_t
observed/potential readers  |           Radix owner/tier/closure/
message and return flow     |           lock/allocator/service curve
          |                 |                    |
          +-----------------+--------------------+
                            v
                  Causal Lease Projector
                            |
                            v
                 Physical Scenario Builder
                            |
                            v
             Side-effect-free What-if Packer
                            |
                            v
                  Joint Agent/KV Planner
      +---------------------+---------------------+
      |                     |                     |
      v                     v                     v
ExecutionIntent       AdmissionIntent       ResidencyIntent
agent/request order   admit/defer/restore    keep/prepare/commit/
and causal mode       dependencies           load/recompute/drop
      +---------------------+---------------------+
                            v
                       JointPlan
                            |
          +-----------------+------------------+
          |                                    |
          v                                    v
 SGLang waiting/admission       Physical Preview + Attempt Guard
          |                                    |
          +-----------------+------------------+
                            v
              RadixArbiter execute-time check
                            |
                            v
                  SGLang HiCache data plane
                            |
                            v
                   ACK + telemetry + replan
```

职责边界如下：

| 模块 | 拥有的决策/状态 | 不允许承担的职责 |
| --- | --- | --- |
| RCCG | 已发生的因果关系、liveness、ready/parked、join/handoff/message | 把未来 hypothesis 写成事实；token prefix 和物理 residency |
| Data-consumer index | 已观测消息/return 消费者和潜在 consumer identity | 用因果 parent 替代实际数据 consumer |
| Frontier predictor | 局部结构转移、需求、consumer、置信度和 OOD | 预测完整 DAG；根据当前 HBM 直接输出迁移动作 |
| Lease projector | 将 RCCG 状态映射成有限的资源承诺 | 修改物理 page |
| Ownership bridge | owner、Radix closure、实际边际字节 | 猜测 agent 状态 |
| Joint planner | 在统一约束下选择 execution/admission/residency intent | 让下游模块重新独立排序；绕过 arbiter 和 ACK |
| Transfer attempt guard | 失败快照、typed blocker、retry eligibility | 用固定 sleep 猜测物理状态已改变 |
| HiCache backend | 合并、layout、DMA、load/write、完成事件 | 决定 agent 优先级 |

系统不设置全局 `subagent_mode` 或 `multi_agent_mode`。每个局部关系按事件语义处理：

```text
CALL/FOREGROUND 或 JOIN_WAIT       -> parent parked，child/peer 推进依赖
SPAWN/BACKGROUND 且无 JOIN_WAIT    -> parent continuation 与 child 均可运行
HANDOFF                            -> source parked 或降级，target READY
MESSAGE                            -> 一个或多个 consumer 独立 READY
RETURN/JOIN_SATISFIED              -> 根据实际 waiter/consumer 更新 frontier
```

## 4. 统一数据结构

### 4.1 Causal lease

新增 `beliefkv/policy/leases.py`：

```python
class LeaseKind(str, Enum):
    DEAD = "dead"
    SPECULATIVE = "speculative"
    CONDITIONAL_RESUME = "conditional_resume"
    READY = "ready"
    RUNNING = "running"

@dataclass(frozen=True)
class LeaseCondition:
    event_kind: str
    subject_id: str
    condition_id: str

@dataclass(frozen=True)
class ContextLease:
    context_id: str
    context_epoch: int
    workflow_id: str
    kind: LeaseKind
    condition: LeaseCondition | None
    issued_ts_ms: float
    confidence: float
    reason: str

@dataclass(frozen=True)
class BundleLease:
    bundle_id: str
    owner_context_ids: tuple[str, ...]
    strongest_kind: LeaseKind
    conditions: tuple[LeaseCondition, ...]
    scenario_support: frozenset[str]
```

context lease 的初始映射规则：

| RCCG 状态 | Lease | 允许的动作 |
| --- | --- | --- |
| `RUNNING_LLM` | `RUNNING` | KEEP/PIN，禁止策略迁移 |
| `READY`、pending message | `READY` | admission、必要时 H2D，除 liveness spill 外禁止 D2H |
| `WAIT_TOOL/WAIT_CHILD/WAIT_JOIN` | `CONDITIONAL_RESUME` | KEEP、SHADOW、压力下 COMMIT_CPU |
| 只有历史预测支持的未来 context | `SPECULATIVE` | 可 prepare，不得覆盖硬安全约束 |
| workflow/context 终止且无 live owner | `DEAD` | DROP 或最低成本驱逐 |

局部关系规则：

- `SPAWN/BACKGROUND` 本身不降低 parent lease；只有 `JOIN_WAIT`、foreground CALL 或明确
  blocking dependency 才把 parent 变为 `CONDITIONAL_RESUME`；
- persistent peer 暂时无消息时不是 DEAD，根据已观测 waiter 或 frontier belief 使用
  `CONDITIONAL_RESUME/SPECULATIVE`；
- HANDOFF source 是否 parked 由 runtime 事件决定，不能仅由 target READY 推断；
- 一个 message 有多个 consumer 时分别维护 context lease，物理 bundle 仍取 strongest owner；
- predicted next speaker 只能提高 speculative/prepare value，不能覆盖当前 READY/RUNNING owner。

同一个物理 bundle 有多个 owner 时，最终 lease 取最强 owner。强度顺序是：

```text
RUNNING > READY > CONDITIONAL_RESUME > SPECULATIVE > DEAD
```

lease 不是新的 cache coherency 协议。它只是 joint planner 的统一语义输入；实际锁、位置
和 generation 仍来自 SGLang/PageOwnershipIndex。

### 4.2 局部 dynamic frontier belief 与 hazard composition

新增 `beliefkv/predictor/scenarios.py`。它是场景组合器，不是一个高维单体 predictor。P6
通过 P5.5 gate 时由 UnlockHazard/ReentryHazard 提供边际分布；未通过时只组合有独立 oracle
证据的 generic frontier transition。预测单位不是完整 workflow，也不是固定的 `next_agent`
label，而是 RCCG 当前 frontier 上一个尚未发生的局部转移及其资源需求：

```python
class FrontierTransitionKind(str, Enum):
    SPAWN = "spawn"
    TOOL_START = "tool_start"
    RETURN = "return"
    HANDOFF = "handoff"
    MESSAGE = "message"
    REACTIVATE = "reactivate"
    LOOP_CONTINUE = "loop_continue"
    TERMINATE = "terminate"

@dataclass(frozen=True)
class DemandDistribution:
    input_tokens_p50: int
    input_tokens_p90: int
    output_tokens_p50: int
    output_tokens_p90: int
    tool_duration_p50_ms: float
    tool_duration_p90_ms: float
    fanout_p50: int
    fanout_p90: int

@dataclass(frozen=True)
class HazardDistribution:
    tokens_to_action_p50: int | None
    tokens_to_action_p90: int | None
    reentry_delay_p50_ms: float | None
    reentry_delay_p90_ms: float | None
    prompt_delta_tokens_p90: int | None
    unknown: bool

@dataclass(frozen=True)
class FrontierTransition:
    kind: FrontierTransitionKind
    source_invocation_id: str
    target_agent_definition_id: str | None
    target_context_id: str | None
    context_mode: str | None
    candidate_consumer_ids: tuple[str, ...]
    demand: DemandDistribution
    hazard: HazardDistribution

@dataclass(frozen=True)
class WorkflowScenario:
    scenario_id: str
    workflow_id: str
    probability: float
    transitions: tuple[FrontierTransition, ...]
    resolved_by_invocation_ids: tuple[str, ...]
    ood: bool

@dataclass(frozen=True)
class ScenarioSet:
    generated_ts_ms: float
    graph_version: int
    scenarios: tuple[WorkflowScenario, ...]
    covered_probability: float
```

subagent 和 multi-agent 使用同一结构，但预测目标不同：

```text
blocking subagent
  spawn 前：是否 fan-out、child role、嵌套、join/cancel 路径
  spawn 后：child token/tool demand、remaining blocker、return size

nonblocking subagent
  parent continuation demand、child completion、何时/是否消费结果

peer multi-agent
  next speaker、handoff target、message consumer set、循环继续/终止
```

约束：

- 只展开未来 1 到 2 个高影响 frontier transition，不预测完整 DAG；
- 已发生的 spawn/join/message 直接来自 RCCG，不允许 predictor 重新猜测；
- 每个 workflow 保留 top-K 场景和一个 `OTHER/OOD` 场景；
- 概率必须归一化并由现有 calibrator 校准；
- 未创建 agent 的 `target_context_id` 必须为 `None`，只允许保留 role/action class；
- structured action 不可解析时 `hazard.unknown=True`，不能填入伪造的 boundary token；
- FRESH child 只产生独立 context 工作集，不计 parent KV 复用；
- 因果 parent、数据 consumer 和 physical prefix owner 分别表示；
- 预测器不读取当前 HBM occupancy、旧策略 action 或 victim rank；
- 预测 wall-clock duration 时必须分离外生 demand 与调度产生的 queue/service time；
- predictor 不可用时生成单一 `REACTIVE/OOD` 场景。

第一版继续使用 context tree、tool survival 和在线 demand/service model，不新增 MLP。对于
multi-agent，只有真实 trace 证明 next-consumer/loop transition 具有可学习性后才扩展模型。

### 4.3 物理 bundle snapshot

新增 `beliefkv/runtime/bundles.py`，从 `PageOwnershipIndex` 和 `RadixArbiter` 构造只读快照：

```python
@dataclass(frozen=True)
class PhysicalBundle:
    bundle_id: str
    handles: tuple[PageHandle, ...]
    owner_context_ids: tuple[str, ...]
    scope: BundleScope
    exclusive_action_bytes: int
    cross_context_action_bytes: int
    foreign_owner_context_ids: tuple[str, ...]
    physical_unique_bytes: int
    gpu_bytes: int
    cpu_bytes: int
    marginal_reclaimable_bytes: int
    closure_bytes: int
    locked_bytes: int
    residency: str
    generation_fingerprint: str
    lease: BundleLease
```

bundle 是一次动作必须共同考虑的最小物理集合，不等同于一个 context。构造规则必须包含：

- D2H/COMMIT 的 leaf closure；
- H2D 的 ancestor closure；
- shared owner 和 workflow 物理分摊；
- 基于实际 action extent 的 `EXCLUSIVE_SUFFIX/SHARED_SUBTREE` scope；
- engine lock、active reader、semantic pin 和 in-flight transfer；
- HiCache 当前 node extent 粒度；
- generation fingerprint，用于执行前拒绝 stale plan。

`marginal_reclaimable_bytes` 必须由实际 closure 和 active owner 推导，禁止直接使用
`sum(context_pages)` 代替。

context-triggered shadow 和 admission frontier spill 只允许 `EXCLUSIVE_SUFFIX`。普通 pressure
严格优先独占 suffix，仅在没有可行独占候选时将 `SHARED_SUBTREE` 作为显式全局 reclaim；
H2D prefetch 不做该限制。共享 bundle 命令的 `context_id` 只是代表，不表示物理 bundle
归该 context 独占。

### 4.4 精确系统快照

新增 `beliefkv/policy/resource_snapshot.py`：

```python
@dataclass(frozen=True)
class ResourceSnapshot:
    ts_ms: float
    hbm_capacity_bytes: int
    hbm_used_bytes: int
    hbm_reserved_bytes: int
    host_free_bytes: int
    urgent_d2h_bytes: int
    urgent_h2d_bytes: int
    pcie_utilization: float
    gpu_compute_utilization: float
    recent_kv_growth_bytes_per_ms: float
    h2d_service_bytes_per_ms: float
    d2h_service_bytes_per_ms: float
    transfer_setup_p50_ms: float
    unhidden_stall_per_byte: float
```

HBM/PCIe 数据只作为在线约束和 cost model 输入，不作为 workflow predictor 的训练特征。

### 4.5 数据 consumer index

新增 `beliefkv/control/data_consumers.py`，避免把 parent-child 控制边误当作数据复用和恢复边：

```python
@dataclass(frozen=True)
class ConsumerEdge:
    producer_invocation_id: str
    consumer_invocation_id: str
    relation: str  # return/message/broadcast/workspace/handoff
    observed: bool
    confidence: float
    last_observed_ts_ms: float
```

observed edge 来自 `RETURN/MESSAGE/HANDOFF` 和 runtime adapter 的明确 result consumption；
predicted consumer 只存在于 `ScenarioSet`，不能写入 RCCG。一个 producer 可以有多个
consumer，cyclic workflow 中同一 consumer 可以多次 reactivation。该 index 只影响 causal
progress 和候选 context value，physical sharing 仍完全由 Radix ownership 决定。

## 5. 场景物理化与 What-if Packer

### 5.1 物理化

新增 `beliefkv/policy/scenario_physicalizer.py`。对每个 scenario 生成短期需求：

```text
已观测 READY/REACTIVATE context
  -> CPU_ONLY unique pages + required ancestor closure

预测或已创建的 FRESH child
  -> prompt/output KV estimate + fixed overhead
  -> 不计 parent KV 复用

peer handoff / message consumer
  -> 只计目标 context 未驻留的 physical marginal bytes
  -> consumer readiness 与 prefix ownership 分别计账

cyclic peer reactivation
  -> 历史 context bundle + 下一轮增长
  -> 记录上次迁移时间，避免 handoff thrashing

multi-consumer message
  -> 每个 consumer 独立 readiness/startup demand
  -> shared physical page 只计一次，私有增长分别计费

tool wait / parked parent
  -> 无新增 admission，保留 conditional resume obligation
```

输出：

```python
@dataclass(frozen=True)
class ScenarioDemand:
    scenario_id: str
    probability: float
    candidate_invocation_ids: tuple[str, ...]
    candidate_request_ids: tuple[str, ...]
    consumer_context_ids: tuple[str, ...]
    required_gpu_bundles: tuple[str, ...]
    optional_gpu_bundles: tuple[str, ...]
    projected_new_bytes: int
    projected_hbm_peak_bytes: int
    earliest_ready_p50_ms: float
    earliest_ready_p90_ms: float
```

`target_context_id=None` 的预测节点只贡献未来 reservation/risk，不可进入 ExecutionIntent 或
SGLang waiting queue。只有 runtime event 创建并标识 context 后，才可以参与实际 admission。

### 5.2 确定性 What-if Packing

新增 `beliefkv/policy/whatif_packer.py`。它不执行命令，只计算一个 scenario 下的可行方案：

```python
class ResidencyAction(str, Enum):
    KEEP = "keep"
    PREPARE_HOST = "prepare_host"
    COMMIT_CPU = "commit_cpu"
    PREFETCH_GPU = "prefetch_gpu"
    RECOMPUTE = "recompute"
    DROP = "drop"

@dataclass(frozen=True)
class ScenarioPlan:
    scenario_id: str
    execution_order: tuple[str, ...]
    admission_actions: Mapping[str, str]
    bundle_actions: Mapping[str, ResidencyAction]
    feasible: bool
    expected_unhidden_stall_ms: float
    hbm_time_byte_ms: float
    d2h_bytes: int
    h2d_bytes: int
    recompute_tokens: int
    blocker_reasons: tuple[str, ...]
```

约束按以下顺序处理：

1. **正确性**：RUNNING/locked/active-reader bundle 不可迁移；closure 必须完整；
2. **容量**：当前驻留、reservation、新增 KV 和 prefetch 不得超过 HBM；
3. **liveness**：至少保留一个可前进 request，不能因 shadow 阻塞 urgent operation；
4. **workflow fairness**：只有处于 bounded-lag fair window 的 workflow 可竞争；fan-out 不
   增加 root workflow 的 service budget；
5. **动态稳定性**：cyclic handoff/message 不得导致同一 bundle 高频 H2D/D2H 抖动；
6. **性能**：最大化因果进度，并最小化未隐藏 stall、无效 transfer/recompute 和 HBM-time。

第一版使用确定性的有界枚举/贪心，不引入通用 MILP 依赖：

1. 由 workflow fairness 取前 `N` 个 eligible workflow；
2. 每个 workflow 从已观测 READY frontier 取前 `M` 个 invocation；预测节点不能被执行；
3. 为每个 invocation 计算 exact startup bytes、可能唤醒的 consumer 和下一次 yield 前增长；
4. 若不 fit，按 lease 强度、physical reclaimable bytes 和 restore/recompute cost 枚举 victim
   bundle；
5. 输出每个候选 execution 的完整可行 plan；
6. 对 repeated handoff context 加入最小 residency lease/hysteresis，除非 HBM emergency；
7. 按“安全、fairness lag、因果推进、未隐藏 stall、无效字节”的字典序选 plan。

这样仍保留现有两级调度原则，但 agent 选择和 victim/prefetch 选择来自同一个候选方案。

## 6. Dynamic Frontier Joint Planning 算法

### 6.1 三种在线模式

Joint planner 每次只生成以下一种模式：

```text
OBSERVED_JOINT
  只使用已观测 RCCG/consumer/physical facts
  联合选择 READY agent、admission 和 KV 动作
  是 correctness baseline 和默认在线模式

ROBUST_BELIEF
  高概率场景影响 reservation、候选排序和可撤销 KV 动作
  对 OOD/分歧采用 worst-case/CVaR feasible plan

REVEAL_AND_COMMIT
  仅当运行一个 READY agent 能在 T_force 前消除高代价场景分歧时启用
  是可选子策略，不是 blocking subagent 的默认路径
```

例如，blocking parent 的 `SPAWN + JOIN_WAIT` 已发生后，child 集合和 parent parked 都是
事实，应直接进入 `OBSERVED_JOINT`；multi-agent supervisor 尚未选择 next speaker，或某个
reviewer 的输出将决定 coder/tester 两种不同工作集时，才可能进入后两种模式。

### 6.2 场景共识与确定性核心

对同一 belief 的每个 scenario 运行 What-if Packer，并按 bundle 比较动作：

```text
所有场景 KEEP/PREFETCH      -> consensus retain/load
所有场景 COMMIT_CPU/DROP    -> consensus reclaim
场景之间动作不同            -> conditional action
```

只有 consensus action 可以立即提交。conditional action 在无硬压力时最多执行
`PREPARE_HOST`，不能立即删除 GPU copy。

ExecutionIntent 与 AdmissionIntent 还必须满足：

- 只包含 RCCG 中已创建且 READY 的 invocation/request；
- 同一 root workflow 的 parent、child 和 peer 共享 service/HBM accounting；
- workflow 内优先级可因 join straggler、handoff consumer、pending message 和循环终止机会
  不同；
- KV 尚未恢复的 READY context 进入 `RESTORE_PENDING`，不能被 waiting queue 提前执行；
- tool/child/join 等外部等待 context 进入 `PARKED`，不进入 SGLang runnable queue。

### 6.3 强制决策时间

定义系统最晚必须作出 residency 决策的时间：

```text
T_force = min(
  当前 KV 增长耗尽可用 HBM 的时间,
  已等待 request 的 liveness/admission deadline,
  已知 READY context 的 latest non-stalling restore time
)
```

第一版分别使用：

- `recent_kv_growth_bytes_per_ms` 的保守上界估计 pool exhaustion；
- 当前 `admission_force_progress_timeout_ms` 作为 liveness 上界；
- 实测 H2D service curve 和 ready queue delay 估计 latest restore time。

不能把语义 wake time 直接当成 restore deadline。READY request 仍可能因 fairness 和 admission
继续等待。

### 6.4 是否进入 REVEAL

仅当以下条件全部满足时进入 REVEAL：

1. 至少两个高概率场景产生不同的最优 physical bundle 动作；
2. RCCG 中存在 ready invocation，其完成事件能区分这些场景；
3. 该 invocation 当前已驻留或其完整启动成本可被安全 admission；
4. `P90(reveal completion) + guard < T_force`；
5. 预计可避免的 transfer/recompute/unhidden stall 大于运行顺序变化带来的成本；
6. workflow 仍在 bounded fairness lag 内；
7. predictor 未 OOD，场景覆盖概率达到阈值。

不满足时进入 ROBUST：在高概率场景中选择 worst-case/CVaR 可行的 residency plan，不等待
额外信息。

### 6.5 状态机

```text
OBSERVE G_t / H_t / R_t
          |
          v
WHAT-IF JOINT PACKING
          |
          +-- no usable belief ----------------> OBSERVED_JOINT
          |
          +-- scenarios agree -----------------> CONSENSUS JOINT COMMIT
          |
          +-- disagree + reveal infeasible ----> ROBUST_BELIEF
          |
          +-- disagree + reveal feasible
                         |
                         v
                      PREPARE
                         |
                         v
                       REVEAL
                         +-- timeout/OOD/stale --> ROBUST_BELIEF
                         |
                         v
                       COMMIT
          |
          v
EXECUTE/ACK/OBSERVE G_(t+1)
```

安全规则：

- PREPARE 完成必须收到 D2H ACK；
- REVEAL 期间出现 HBM emergency 时立即转 ROBUST；
- graph version、page generation 或 owner set 改变时旧 plan 失效并重算；
- branch/handoff/consumer reveal 后只提交实际 scenario 的 physical bundle；
- reveal 超时不阻塞 liveness；
- P0-P3 的任意异常回退到当前 reactive baseline；P4 单写者规则启用后，任意异常必须生成
  `OBSERVED_JOINT` conservative fallback，而不是回退到多套独立排序器。

### 6.6 条件性 Action-frontier 扩展

该扩展不是 P4/P5 的 correctness 必需项，只有 P5.5 证明 action-unlock oracle 相比 B1/B2
存在独立收益后才进入 P6。其关键不是增加一个 priority score，而是把短 decode quantum 与
随后发生的 KV 生命周期转换建模为同一个候选动作：

```text
ExecutionPackage(i, q)
  = RUN invocation i for at most q decode tokens
  + probability of unlocking a valid action
  + external work/frontier released by that action
  + KV growth before boundary
  + post-boundary KV state: ACTIVE -> WAITING/DEAD/HANDOFF
  + KEEP/SHADOW/COMMIT/PREFETCH alternatives
```

每个规划事件执行：

1. `ActionFrontierObserver` 从已生成 token 和 structured parser state 更新事实；
2. `UnlockHazard` 为 active invocation 生成短期 action-boundary 场景；
3. `ReentryHazard` 为 waiting context 生成 return/reactivation 场景；
4. What-if Packer 联合枚举 `RUN(q)`、ADMIT/PAUSE 和 physical bundle action；
5. 优先满足 correctness、liveness 和 workflow fairness，再比较 valid-action delay、causal
   progress、unhidden stall、HBM-time 和 transfer/recompute bytes；
6. 根据校准置信度选择 commitment level：低置信度回退 observed plan，中置信度只建 Host
   shadow，高置信且物理安全时才提交 GPU eviction 或 proactive prefetch；
7. receding horizon 只执行首个 quantum/action，收到 token、runtime event 或 transfer ACK 后
   立即重算。

若 `RUN(q)` 不会改变合法 action 时间、runnable frontier 或 post-boundary residency，则该候选
退化为 P5 普通 observed scheduling，不允许仅因模型分数而改变执行顺序。

## 7. Agent 调度与 KV 调度的统一接口

新增 `beliefkv/policy/joint_scheduler.py`：

```python
@dataclass(frozen=True)
class ExecutionIntent:
    ordered_request_ids: tuple[str, ...]
    selected_workflow_id: str | None
    selected_invocation_id: str | None
    mode: str  # observed_joint/robust_belief/reveal/liveness
    graph_version: int
    reason: str

@dataclass(frozen=True)
class AdmissionIntent:
    request_id: str
    action: str  # admit/defer/restore_then_admit/parked
    reserved_bytes: int
    required_bundle_ids: tuple[str, ...]
    reason: str

@dataclass(frozen=True)
class ResidencyIntent:
    bundle_id: str
    action: ResidencyAction
    target_bytes: int
    deadline_ms: float
    scenario_support: frozenset[str]
    reason: str

@dataclass(frozen=True)
class TransferDependency:
    before_request_id: str | None
    residency_intent_index: int
    require_ack: bool

@dataclass(frozen=True)
class JointPlan:
    plan_id: str
    generated_ts_ms: float
    execution: ExecutionIntent
    admissions: tuple[AdmissionIntent, ...]
    residency: tuple[ResidencyIntent, ...]
    dependencies: tuple[TransferDependency, ...]
    expected_hbm_peak_bytes: int
    expected_unhidden_stall_ms: float
    fallback_reason: str | None
```

`ControllerTickResult` 增加：

```python
joint_plan: JointPlan | None
execution_intent: ExecutionIntent | None
```

SGLang waiting queue 不再自行重新计算 workflow/frontier 顺序，而是优先应用
`ExecutionIntent.ordered_request_ids`。未出现在 intent 中的请求保持上游相对顺序，未携带
BeliefKV metadata 的请求不受影响。

从 P4 开始强制以下单写者规则：

```text
JointPlanner 是 tagged agent request 顺序、admission 和策略性 KV 动作的唯一决策者。
AdmissionController 只校验/兑现 AdmissionIntent，不再重新选择 workflow。
ReactiveTransferPlanner 只把 ResidencyIntent 编译成 command，不再重新选择 victim。
SGLang reorder_waiting_queue 只应用 ExecutionIntent，不再重新计算 frontier/fairness。
RadixArbiter 仍可因物理状态拒绝计划，但不能替换为另一个策略动作。
```

未携带 BeliefKV metadata 的普通请求继续采用 SGLang 原生策略，不能被 JointPlan 饿死；其
容量和服务消耗作为 external charge 反馈给下一次规划。

### 7.1 两级公平约束

联合策略继续使用两级结构：

1. **workflow 层**：按 attained GPU service 和 physical HBM charge 维护 virtual runtime；
2. **workflow 内部**：按 causal frontier、next-consumer progress、reveal value、循环终止机会和
   exact startup cost 选择 invocation。

REVEAL 不能无限突破公平。建议增加 `fairness_lag_budget_ms`：只有 virtual runtime 不超过
最欠服务 workflow 加该预算的 workflow，才允许因 reveal 被提前。

### 7.2 与 SGLang batch 的关系

第一版只控制 waiting/admission order，不抢占正在运行的 decode batch，也不修改 CUDA
kernel。具体执行顺序：

1. scheduler safe point 应用上一轮 ACK；
2. 同步 Radix tree、allocator、lock 和 request 状态；
3. controller 生成一个带 generation 的 `JointPlan`；
4. runtime 使用 execution intent 重排 tagged waiting requests；
5. admission 只放行与 plan 一致且物理可行的 request；
6. admission 只有在依赖的 restore ACK 或 recompute plan 满足后才兑现；
7. residency intent 经 RadixArbiter 再解析为 page action；
8. plan 在 batch selection 前变 stale 时丢弃，并在同一 snapshot 上生成
   `OBSERVED_JOINT` fallback。

### 7.3 JointPlan 生成算法

每次 runtime、allocator、ACK 或 waiting-queue 变化触发一次有界规划：

1. **建立 factual runnable frontier**：只收集 RCCG 中 `READY`、pending-message 或明确
   reactivation 的 invocation；`WAIT_TOOL/WAIT_CHILD/WAIT_JOIN` 只贡献 residency obligation；
2. **选择 fair workflow window**：按 root-workflow virtual runtime、已获 GPU service 和
   physical HBM charge 选出 bounded-lag workflow，fan-out 不增加份额；
3. **构造 ExecutionPackage**：对每个 READY invocation 计算：

   ```text
   exact startup/restore bundles
   next-yield 前 token/KV demand
   可解除的 waiter、join、message consumer 或 loop termination
   admission reservation 和 restore/recompute alternatives
   ```

4. **联合枚举**：对每个候选 execution order，用 What-if Packer 同时选择 admission 和
   residency actions；不能先固定 agent，再从剩余空间找 KV victim；
5. **场景聚合**：P4/P5 只计算 observed scenario；P6 对 top-K scenario 做 consensus、robust
   或 reveal 判定；
6. **字典序选择**：先满足 correctness/liveness/fairness，再最大化 causal progress，最后最小化
   unhidden stall、迁移/重算字节和 HBM-time；
7. **生成依赖**：任何需要 H2D 的 request 都产生 `TransferDependency(require_ack=True)`；允许
   recompute 时把它作为明确 alternative，而不是隐式欠债；
8. **版本化提交**：JointPlan 带 graph、allocator、topology 和 bundle generation；任一版本变化
   使未执行部分 stale，并触发重新规划。

workflow 内 causal progress 的确定性优先级只使用已观测事实：

```text
唯一 join straggler / 已满足 handoff consumer
  > 能解除 blocking waiter 的 READY agent
  > 有 pending message 的 READY peer
  > 普通 READY foreground continuation
  > 无当前 consumer 的 background work
```

该顺序不是最终 score。What-if Packer 可以因为启动成本、HBM 可行性或 root-workflow fairness
选择后续候选，但所有偏离都必须在 JointPlan reason 中可审计。

### 7.4 Reference-policy 兼容接口

P2.5 开始提供只读、可 replay 的统一策略接口。它不是另建一套执行路径，而是让竞品策略和
BeliefKV 使用同一个 snapshot，并输出同一种 JointPlan primitive：

```python
@dataclass(frozen=True)
class PolicyInput:
    runtime_graph: RuntimeGraphSnapshot
    runnable_frontier: tuple[RunnableInvocation, ...]
    physical_kv: PhysicalKVSnapshot
    resources: ResourceSnapshot
    optional_metadata: Mapping[str, object]

@dataclass(frozen=True)
class PolicyOutput:
    execution: ExecutionIntent
    admissions: tuple[AdmissionIntent, ...]
    residency: tuple[ResidencyIntent, ...]
    dependencies: tuple[TransferDependency, ...]
    policy_name: str
    metadata_assumptions: tuple[str, ...]
```

首批 reference policy：

| Policy | 输入 | 主要动作 | 前期用途 |
| --- | --- | --- | --- |
| `ReactivePolicy` | 当前物理状态 | LRU/reactive restore | B0 correctness baseline |
| `DistancePolicy` | invocation distance | distance eviction/prefetch | ScaleSim-style B1 |
| `SpaceTimePolicy` | output/tool duration | Preserve/Swap/Discard + value-density order | AugServe-style B2 |
| `PhasePolicy` | Reasoning/Acting/program phase | Pause/Restore + shortest-first | ThunderAgent-style B3 |
| `CongestionPolicy` | KV usage/hit/eviction feedback | AIMD admit/pause | CONCUR-style B4 |
| `BeliefJointPolicy` | `G_t/H_t/R_t` | 完整 JointPlan | BeliefKV |

P2.5/P3 中 reference policy 默认只记录 shadow output。hindsight metadata 只允许用于 oracle
replay，必须在 `metadata_assumptions` 中声明，不能与在线 BeliefKV 结果混称为可部署策略。
每个 policy 明确区分 `online` 与 `oracle` mode：前者只能使用论文要求且当前 runtime 实际可得
的 metadata，后者可在 run 完成后注入真实未来信息。
原生系统与 reference policy 在 common workload 上的动作和趋势偏差由 P8 单独校验。

## 8. HiCache 数据面改进

### 8.1 不重写数据搬运

`HiCacheNodeCommandBackend` 继续负责：

- `write_backup()`；
- `load_back()` 和 `ready_to_load_host_cache()`；
- backup 完成后的 `_evict_backuped()`；
- scheduler thread safe point；
- partial/reject/cancel/complete ACK。

策略层禁止直接拼 tensor copy 或修改 node `value/host_value`。

### 8.2 增加 capability contract

在 `beliefkv/runtime/sglang_adapter.py` 新增：

```python
@dataclass(frozen=True)
class HiCacheCapabilities:
    operation_merge: bool
    layer_completion_events: bool
    page_first_host_layout: bool
    proactive_load_trigger: bool
    max_inflight_operations: int
    physical_unit: str  # node_extent/page
```

固定的 `0.5.2rc1` adapter 必须如实报告能力。joint planner 不能假定新版 HiCache 的
layer overlap、page_head 或多 in-flight 已存在。

### 8.3 传输 telemetry

在 `beliefkv/runtime/protocol.py` 增加独立的 `TransferTelemetry`，避免把性能观测混入
正确性 ACK：

```python
@dataclass(frozen=True)
class TransferTelemetry:
    command_id: str
    submit_ts_ms: float
    start_ts_ms: float | None
    first_layer_ready_ts_ms: float | None
    complete_ts_ms: float
    compute_wait_ms: float
    actual_bytes: int
    closure_bytes: int
    merged_operation_count: int
    direction: TransferDirection
    source_tier: str
    target_tier: str
```

若固定版本无法观测 first-layer event，字段记录为 `None`，不得用 callback completion 冒充。

新增 `beliefkv/policy/service_curve.py`，按最近完成的 operation 分别维护：

- H2D/D2H setup time 分布；
- effective bytes/ms；
- allocator/rejection probability；
- compute wait 和 unhidden stall；
- 按 size bucket、direction、compute phase 分桶。

数据不足或 OOD 时回退到保守静态带宽模型。

### 8.4 Event-gated transfer retry guard

新增 `beliefkv/policy/transfer_guard.py`。它不改变首次候选排序，而是约束一个 transfer intent
在失败后何时重新获得提交资格。第一版数据结构为：

```python
class TransferBlockerCode(str, Enum):
    ANCESTOR_CLOSURE = "ancestor_closure"
    DEVICE_CAPACITY = "device_capacity"
    HOST_CAPACITY = "host_capacity"
    ENGINE_BUSY = "engine_busy"
    NODE_LOCKED = "node_locked"
    NODE_LOADING = "node_loading"
    INFLIGHT = "inflight"
    STALE_GENERATION = "stale_generation"
    EXTENT_MUTATED = "extent_mutated"
    UNKNOWN_BACKEND = "unknown_backend"

@dataclass(frozen=True)
class TransferAttemptKey:
    context_id: str
    context_epoch: int
    command_kind: CommandKind
    closure_fingerprint: str

@dataclass
class BlockedTransferAttempt:
    key: TransferAttemptKey
    blocker_codes: tuple[TransferBlockerCode, ...]
    required_closure_bytes: int
    topology_epoch: int
    allocator_epoch: int
    lock_epoch: int
    engine_epoch: int
    failed_ts_ms: float
    identical_failure_count: int
```

`closure_fingerprint` 必须覆盖 ordered page handles/generations、ancestor closure 和 target
tier，不能只使用 context id。blocker 从 `RadixArbiter`/HiCache backend 以结构化字段返回，
禁止在 planner 中解析自由文本 `reason`。

解除规则按 blocker 类型定义：

| blocker | 重新获得资格的事件谓词 |
| --- | --- |
| `ANCESTOR_CLOSURE` | topology/closure fingerprint 改变，或新的 preview 将完整 ancestors 纳入同一 bundle |
| `DEVICE_CAPACITY` | `allocator_free - reservations >= required_closure_bytes`，且 allocator epoch 改变 |
| `HOST_CAPACITY` | Host free crossing required bytes 或 Host eviction completion |
| `NODE_LOCKED` | 对应 node lock epoch 改变，active reader/engine lock 归零 |
| `NODE_LOADING/INFLIGHT/ENGINE_BUSY` | 相关 operation completion/cancel event，而不是下一 scheduler tick |
| `STALE_GENERATION/EXTENT_MUTATED` | authoritative topology resync 后生成新的 closure fingerprint |
| `UNKNOWN_BACKEND` | bounded exponential backoff + circuit breaker，并记录为未建模 blocker |

已知 blocker 不使用纯时间 cooldown。timer 只能作为 unknown backend 的防御性兜底；HBM、lock
或 topology 未满足解除谓词时，即使 timer 到期也不能重试。每个 runtime event 更新单调的
typed resource epoch，guard 只重新检查订阅该 epoch 的 attempt，避免所有 context 每 tick 扫描。

ACK/telemetry 闭环为：

```text
preview -> eligible? -> dispatch -> execute-time revalidate -> ACK/telemetry
              ^                                      |
              |                                      v
        matching event predicate <- attempt ledger <- typed blockers
```

prefetch 是可选优化。被 guard 抑制时不得阻塞 admission、decode 或其他 context；planner 应继续
选择下一个可行候选。context epoch 终止、abort、cache reset 后 ledger 条目必须立即失效。

## 9. 对当前文件的具体修改

| 文件 | 修改 |
| --- | --- |
| `beliefkv/control/causal_graph.py` | 增加单调 `graph_version`；暴露 READY frontier、waiter、handoff/message 和循环 reactivation 查询 |
| `beliefkv/control/data_consumers.py` | 新增 observed producer-consumer index；与 causal parent 和 physical owner 分离 |
| `beliefkv/control/controller.py` | 构造统一 snapshot；调用 joint scheduler；只返回一个 JointPlan；保留 observed-joint fallback |
| `beliefkv/policy/residency.py` | 保留旧 classifier 作为 baseline；新策略改用 lease projector |
| `beliefkv/policy/leases.py` | 新增 context/bundle lease 和 owner 聚合 |
| `beliefkv/predictor/scenarios.py` | 新增 subagent/multi-agent 局部 frontier transition、consumer、demand 与 OOD 场景 |
| `beliefkv/predictor/composer.py` | 保留边际预测 API；为 scenario builder 提供组成事件，不直接输出迁移动作 |
| `beliefkv/runtime/bundles.py` | 生成 closure-aware physical bundle snapshot |
| `beliefkv/policy/resource_snapshot.py` | 聚合 HBM、Host、PCIe、service curve 和 reservation |
| `beliefkv/policy/scenario_physicalizer.py` | 将场景映射为未来物理需求 |
| `beliefkv/policy/whatif_packer.py` | 生成每个 scenario 的无副作用可行 plan |
| `beliefkv/policy/joint_scheduler.py` | observed/robust/reveal 模式，联合 Execution/Admission/Residency 和依赖 |
| `beliefkv/policy/reference/base.py` | 定义统一 PolicyInput/PolicyOutput、metadata assumptions 和 shadow/replay API |
| `beliefkv/policy/reference/distance.py` | ScaleSim-style invocation-distance reference policy |
| `beliefkv/policy/reference/space_time.py` | AugServe-style space-time/value-density reference policy |
| `beliefkv/policy/reference/phase.py` | ThunderAgent-style phase-aware pause/restore reference policy |
| `beliefkv/policy/reference/congestion.py` | CONCUR-style AIMD admission reference policy |
| `beliefkv/policy/admission.py` | 从独立选 workflow 改为校验并兑现 AdmissionIntent |
| `beliefkv/policy/transfer_planner.py` | 从独立选 victim 改为编译 ResidencyIntent；旧逻辑仅保留为实验 baseline |
| `beliefkv/policy/transfer_guard.py` | 新增失败 attempt ledger、typed blocker 和 event-gated retry eligibility |
| `beliefkv/policy/shadow_controller.py` | 保留 baseline；full policy 的 PREPARE 由 joint scheduler 触发 |
| `beliefkv/runtime/radix_arbiter.py` | 增加只读 preview/bundle closure API；执行时继续二次校验 |
| `beliefkv/runtime/protocol.py` | 增加 telemetry 和 plan metadata，不改变 ACK 正确性语义 |
| `beliefkv/runtime/sglang_v052rc1.py` | 只应用 ExecutionIntent 顺序和 AdmissionIntent 依赖；采集 capability/telemetry；审计 stale/fallback |
| `beliefkv/runtime/sglang_adapter.py` | 增加 capability 和 telemetry backend protocol |
| `beliefkv/runtime/action_frontier.py` | 增量记录 structured action parser 状态、合法 action boundary 和 frontier delta |
| `beliefkv/runtime/audit.py` | schema 升级并记录 scenario/lease/joint-plan 事件 |
| `beliefkv/predictor/unlock_hazard.py` | P5.5 gate 通过后预测 active invocation 到合法 action 的 token hazard |
| `beliefkv/predictor/reentry_hazard.py` | 预测 waiting context 的 return/reactivation 与 prompt-delta 分布 |
| `beliefkv/experiments/policy_replay.py` | 在冻结 trace 上运行 B0-B4、O0-O3 和 action-unlock oracle |
| `beliefkv/core/config.py` | 增加 feature flags、场景预算、公平 lag 和风险阈值 |

建议配置项及第一版默认值：

```text
joint_policy_enabled=false
joint_policy_shadow_mode=true
joint_observed_mode_enabled=true
reference_policy_shadow_enabled=true
action_frontier_observer_enabled=true
action_frontier_policy_enabled=false
frontier_belief_enabled=false
scenario_max_depth=2
scenario_top_k=4
scenario_min_covered_probability=0.90
scenario_min_branch_probability=0.05
reveal_enabled=false
reveal_guard_ms=25
reveal_min_avoidable_stall_ms=5
fairness_lag_budget_ms=50
robust_risk_quantile=0.95
residency_hysteresis_ms=100
max_frontier_candidates_per_workflow=4
max_joint_plan_budget_ms=1
service_curve_window=256
transfer_retry_guard_enabled=true
transfer_retry_max_same_snapshot_attempts=1
transfer_retry_unknown_base_ms=10
transfer_retry_unknown_max_ms=1000
transfer_retry_unknown_circuit_breaker_failures=8
```

所有会改变执行的功能默认关闭。`joint_policy_shadow_mode=true` 时只记录完整 JointPlan，不改变
现行 admission、waiting queue 或 transfer command。`joint_observed_mode_enabled` 先于
`frontier_belief_enabled` 激活，用于隔离 agent/KV 联合调度本身与预测收益。
`action_frontier_observer_enabled` 只采集结构化 action 状态，不改变 decode 顺序；
`action_frontier_policy_enabled` 必须等 P5.5 gate 通过后才能打开。

## 10. 审计与实验数据格式

新增 audit event：

```text
resource_snapshot
context_lease_issued
bundle_lease_aggregated
scenario_set_built
scenario_physicalized
scenario_plan_evaluated
joint_plan_selected
joint_plan_stale
reference_policy_decision
action_frontier_updated
valid_action_unlocked
reveal_started
reveal_resolved
reveal_aborted
residency_intent_dispatched
transfer_attempt_blocked
transfer_retry_suppressed
transfer_retry_rekeyed
transfer_retry_released
transfer_telemetry
decision_outcome
```

`joint_plan_selected` 至少记录：

```text
plan_id / graph_version / mode
scenario ids and probabilities
selected workflow/invocation/request
bundle actions and physical bytes
expected/actual HBM peak
expected/actual unhidden stall
reactive baseline counterfactual
oracle action if available during replay
fallback reason
controller planning time
```

`action_frontier_updated` 和 `valid_action_unlocked` 至少记录：

```text
workflow/invocation/request/action type
generated token index / action boundary token index
parser state / structured-action confidence
valid-action timestamp / tool-start timestamp
spawn or handoff target / observed consumer count
runnable frontier size before/after
active-to-waiting KV bytes
next reactivation timestamp if later observed
```

每次实验继续使用 immutable run directory，并新增：

```text
manifest.json
summary.json
runtime_audit.jsonl
scenario_decisions.jsonl
reference_policy_decisions.jsonl
action_frontier_events.jsonl
transfer_telemetry.jsonl
workflow_metrics.csv
bundle_metrics.csv
decision_metrics.csv
action_metrics.csv
```

## 11. 分阶段实施计划

### P0：冻结当前 correctness baseline

任务：

1. 固定当前 `0.5.2rc1` commit、BeliefKV commit/dirty patch 和最终 Deep Agents workload；
2. 保留 2026-07-17 liveness 修复实验作为 correctness reference；
3. 冻结一份可重放的 RuntimeEvent、request、Radix mutation 和 transfer ACK trace；
4. 为当前 reactive/write-back 策略生成不可覆盖的 baseline artifact。

退出条件：

- 现有测试全部通过；
- replay 的 request/ACK/graph hash 一致；
- 无 watchdog、location divergence、stale handle 或无证明 admission。

### P1：补全 capability 和真实 telemetry

实施状态（2026-07-18）：已完成。真实验证与统计限制见
[P1 HiCache 真实迁移、服务曲线与控制开销验证](experiments/beliefkv_p1_real_hicache_validation_2026-07-18_zh.md)。

实现文件：

- `runtime/protocol.py`；
- `runtime/sglang_adapter.py`；
- `runtime/sglang_v052rc1.py`；
- `policy/service_curve.py`；
- `tests/test_transfer_cost.py`；
- 新增 `tests/test_service_curve.py`。

任务：

1. 记录 submit/start/complete、actual/closure bytes、partial/reject 原因；
2. 采集 copy-engine/PCIe/GPU compute/allocator 状态；
3. 能观测时记录 compute wait；不能观测时显式标为 unavailable；
4. 用完成 operation 在线更新 H2D/D2H service curve；
5. 比较 nominal transfer time、callback time 和 unhidden stall。

退出条件：

- 真实运行中 telemetry 非常量且时间戳单调；
- ACK bytes 与 PageOwnershipIndex 状态一致；
- service curve 对 holdout operation 的 P90 under-estimation 率低于 10%；
- controller telemetry 开销低于单次 scheduler tick 的 5%。

### P1.5：修复 retry storm 并冻结 storm-free reactive baseline

该阶段是 baseline 修复，不作为论文主要创新。必须先完成它，避免后续 P2/P5 的收益来自简单
debounce。

实施状态（2026-07-19）：typed blocker、closure fingerprint、event-gated attempt ledger、
unknown exponential backoff/circuit breaker 和审计事件已经接入；冻结 268-retry fixture、
全量测试与真实 HiCache 机制短跑通过。真实运行中同快照最大提交 1 次、未解除 blocker 时
retry 为 0、unknown blocker 为 0，原 268 次同 context 风暴清零。由于新旧运行持续时间和
动态 action path 不同，且 P1.5 admission P99 高于旧运行，JCT/liveness 无退化尚未通过严格
配对验证，详见
[P1.5 retry guard 实现与验证](experiments/beliefkv_p1_5_retry_guard_2026-07-19_zh.md)。
实跑后还修复了 partial H2D 将整包误记为 capacity debt 的问题：最终版本只记录失败页，
并以 ACK 后首个 allocator snapshot 建立解除基线；该版本已通过单测，但尚未包含在上述真实
trace 中，必须进入下一次配对复验。

实现文件：

- 新增 `policy/transfer_guard.py`；
- 修改 `runtime/protocol.py`，增加结构化 blocker code/metadata；
- 修改 `runtime/radix_arbiter.py` 和 `runtime/sglang_v052rc1.py`，返回 authoritative blocker；
- 修改 `control/controller.py` 和 `policy/transfer_planner.py`，接入 attempt ledger；
- 新增 `tests/test_transfer_guard.py`，扩展真实 trace replay。

任务：

1. 固定 P1 短跑中 268 次同 context reject 的最小可重放 fixture；
2. 为 local resolve reject 与 backend ACK reject 建立统一、结构化的 attempt outcome；
3. P1.5 先以 `(context, epoch, action, closure fingerprint)` 记录失败；P2 进一步加入
   `bundle_id`，避免一个 blocked extent 屏蔽同 context 的独立 bundle；
4. 为 closure/capacity/lock/loading/inflight/generation 分别实现事件解除谓词；
5. planner 跳过未解除 attempt 后继续考察其他候选，避免 head-of-line blocking；
6. unknown backend 使用有上限的指数退避和 circuit breaker，所有已知 blocker 禁止 tick retry；
7. 审计 suppressed/released/retried attempt，并输出 storm concentration 和 false suppression。

退出条件：

- 在冻结 fixture 中，同一 physical fingerprint 最多提交 1 次；268 次相同拒绝降为不超过 1 次；
- 未发生匹配 resource/topology/lock event 时重试数为 0；
- 解除谓词满足后，attempt 在下一个 controller event 内恢复资格；
- command attempt 数量随相关物理状态变化次数增长，而不是随 scheduler tick 数增长；
- 不存在永久误屏蔽：context epoch、cache reset、abort 和 generation change 均正确清理状态；
- 真实压力短跑的 zero-byte identical-retry 数降低至少 95%，且 admission liveness/JCT 不退化；
- fallback 不解析自由文本 reason，unknown blocker 比例单独报告。

### P2：实现 physical causal lease 和 bundle preview

实施状态（2026-07-20）：causal lease、shared-owner 聚合、action-level bundle scope、D2H
descendant closure、H2D ancestor closure、Host/Device capacity preview、versioned physical
intent、bundle-scoped retry、arbiter 二次校验以及 HiCache backend 的 full-bundle
preflight/atomic commit 已接入。D2H 使用 shallow-to-deep submit 和 deep-to-shallow
eviction；H2D 复用 HiCache 原生 ancestor load。确定性与故障注入测试通过；真实高压实验中
131 次 D2H offload 全部完成、reclaim realization 达到 100%，但 H2D device-capacity reject、
重复 fingerprint retry、partial rollback 后 GPU Radix/allocator 双账本不一致和 workload
`FileNotFoundError` 使 P2 gate 失败。详见
[P2 physical bundle 实现与验证](experiments/beliefkv_p2_physical_bundle_2026-07-19_zh.md)。

可靠性修复状态（2026-07-20）：已确认 281 次小 H2D reject 的直接原因是固定 HiCache 的
10-token `load_back_threshold`，而非真实 allocator capacity；满载 fatal 同时存在 1,505-token
live Radix/free-list overlap。当前实现已加入 forced/no-eviction H2D、authoritative allocator
delta 校验、H2D ACK 前 admission barrier 与 prefix rematch、allocator/Radix resync、逐 callback
故障隔离，以及 control-socket failure 与 workflow failure 解耦。全量 CPU 测试通过；真实 GPU
复跑因两张卡被其他实验占用而未启动。因此 P2 仍未通过真实可靠性 gate，也不能开始 P4 的
主动 JointPlan 控制。

实现文件：

- 新增 `policy/leases.py`；
- 新增 `runtime/bundles.py`；
- 修改 `runtime/radix_arbiter.py` 和 `runtime/page_index.py`；
- 修改 `policy/transfer_planner.py`、`control/controller.py` 和
  `runtime/sglang_v052rc1.py`，接入 exact bundle intent 与两阶段执行；
- 复用 P1.5 `transfer_guard.py` 的 typed blocker/epoch；
- 扩展 `metrics/transfer_validation.py`，输出 physical bundle characterization；
- 新增 `tests/test_leases.py`、`tests/test_bundles.py`。

任务：

1. RCCG 状态投影为 context lease；
2. shared owner 聚合为 bundle lease；
3. preview D2H leaf closure、H2D ancestor closure、bundle scope 和 marginal bytes；
4. 输出 blocker set：lock、reader、pin、descendant、host allocation、generation；
5. preview 产生 closure fingerprint 和 retry release predicate，失败反馈回 attempt ledger；
6. 将 F1、F4 characterization 所需数据写入审计日志。

退出条件：

- RUNNING owner 永远覆盖 DEAD/SPECULATIVE owner；
- preview bytes 与实际 resolved/ACK bytes 偏差可以解释；
- stale generation 无法进入执行队列；
- shared page 不按 context 重复计费；
- parent/context-triggered shadow 与 frontier spill 不得迁移 foreign-owner extent；
- pressure 存在独占 suffix 时不得选择更大的 shared subtree；
- preview 判定不变时不会重复 dispatch 相同不可行 bundle；
- H2D full closure admission、partial rollback 和 allocator/Radix resync 不产生账本不变量失败；
- 使用同 manifest 的 P1.5/P2 配对实验闭合后，才允许 P4 主动应用 JointPlan。

### P2.5：统一 Reference-policy Contract

该阶段不部署 ScaleSim、AugServe、ThunderAgent 或 CONCUR 原系统，也不阻塞 P2 真实 GPU
可靠性修复。目标是提前固定公平比较所需的输入、动作、metadata assumption 和日志格式，避免
BeliefKV 完成后再为每个竞品重写实验路径。

实现文件：

- 新增 `policy/reference/base.py`；
- 新增 `tests/test_reference_policy_contract.py`；
- B0-B4 具体策略和 `experiments/policy_replay.py` 留到 P3B。

任务：

1. 所有策略读取相同的 `PolicyInput`，输出可审计的 `PolicyOutput`；
2. 每个输出声明使用了 observed、predicted 还是 hindsight metadata；
3. contract 明确 online/oracle metadata mode 和 hindsight 隔离；
4. reference policy 默认 shadow-only，不触发真实 admission、D2H 或 H2D；
5. 为未来 native baseline 保留 request/program/context ID 映射和 capability report。

退出条件：

- synthetic reference adapter 对相同输入生成确定性、schema-valid 的 PolicyOutput；
- hindsight 字段无法泄漏到在线 BeliefKV predictor；
- 同一物理 snapshot 下所有策略使用相同 HBM/PCIe accounting；
- unsupported action/metadata 被明确记录，不静默转换成 BeliefKV 特有动作；
- contract output 可被 audit validator 重建。

### P3：动态 Workload、Reference-policy Oracle 与 Agent 调度基线

P3 的第一目标是建立动态 agent 调度实验面、action-unlock 观测面和 joint oracle，不先训练
frontier predictor，也不要求先部署竞品原系统。该阶段可以与 P2 可靠性修复并行开发，但 P2
gate 未通过时只能做 replay/shadow，不主动控制真实 HiCache。

实现文件：

- 新增 `control/data_consumers.py`；
- 新增 `policy/scenario_physicalizer.py`、`policy/whatif_packer.py`；
- 新增 `policy/reference/distance.py`、`space_time.py`、`phase.py` 和 `congestion.py`；
- 新增 `experiments/policy_replay.py`；
- 新增 `runtime/action_frontier.py`；
- 新增 `experiments/langgraph_peer_workflow.py` 或等价真实 multi-agent adapter；
- 扩展 RuntimeEvent adapter，可靠捕获 `HANDOFF/MESSAGE/REACTIVATE/CANCEL`；
- 扩展 simulator/replay 和 audit validator；
- 新增 `tests/test_data_consumers.py`、`tests/test_whatif_packer.py`、
  `tests/test_multi_agent_runtime.py`、`tests/test_action_frontier.py` 和
  `tests/test_reference_policies.py`。

#### P3A：动态 workload 与 instrumentation

任务：

1. 在现有 blocking Deep Agents workload 之外，加入使用真实任务输入的 cyclic
   Coder-Reviewer-Test 或 Supervisor-Worker multi-agent workflow；
2. 再加入一个 mixed workflow：peer agent 内嵌 FRESH subagent fork/join；
3. 记录 `context_prefix_affinity`，区分 parent-child、sibling/template 和跨 workflow prefix；
4. 记录 observed producer-consumer、next speaker、handoff、loop、fan-out、join/cancel；
5. 增量记录 structured function call、subagent spawn、handoff 和 final answer 的 parser 状态、
   合法 action boundary token 与 valid-action timestamp；
6. 记录 action 前后 runnable frontier、tool-start gap、active-to-waiting KV bytes 和后续 reentry；
7. 将 trace 标记为 schedule-invariant、timing-sensitive 或 semantic-race-sensitive；后两类必须用
   真实 A/B 闭合，不能只依赖冻结 replay。

#### P3B：reference policy、joint oracle 与竞争上界

任务：

1. 对冻结 trace 运行五个 reference baseline：

   ```text
   B0 current reactive scheduling + current bundle policy
   B1 ScaleSim-style + hindsight invocation distance
   B2 AugServe-style + true output length/tool duration
   B3 ThunderAgent-style + perfect Reasoning/Acting phase metadata
   B4 CONCUR-style + identical cache feedback
   ```

2. B1/B2 使用的真实未来信息必须标记为 oracle metadata；B3 的 program/phase metadata 标记为
   application-provided。oracle variant 用于给竞品更强上界，不声称在线可部署；
3. B0-B4 初版只实现论文核心策略，不复制各系统的数据面和 kernel；replay 固定 semantic
   events、token demand 和 tool duration，并区分 schedule-invariant 与 schedule-sensitive trace；
4. 对冻结 trace 构造四个 jointness offline oracle：

   ```text
   O0 current separate scheduler + current bundle policy
   O1 oracle agent scheduling + current KV policy
   O2 current agent scheduling + oracle KV policy
   O3 oracle JointPlan(agent scheduling + admission + KV)
   ```

5. 对 lower-is-better cost 定义
   `joint synergy gap = min(cost(O1), cost(O2)) - cost(O3)`，证明联合决策不是两个独立优化
   的简单叠加；
6. 构造 action-unlock oracle：使用真实 action boundary 和后续因果事件，比较“优先产生合法
   action”与 B1/B2 的 invocation-distance/value-density 目标；
7. What-if packer 对 blocking、nonblocking、handoff、multi-consumer 和 cyclic reactivation
   都生成物理可行 plan；
8. oracle 冻结同一语义依赖、LLM token demand、tool duration 和 runtime transition outcome，
   但根据候选调度重新计算 service time、HBM residency 和 physical action；禁止固定原策略的
   wall-clock/physical trace 后声称模拟了新调度；
9. 对 semantic-race-sensitive workflow，oracle 结果只报告为 optimistic bound；
10. 真实端到端 A/B 使用固定任务、模型参数和随机性并重复运行；仍无法冻结动态路径时同时
   报告 transition hash 和统计置信区间。

退出条件：

- 至少一个 blocking subagent、一个 cyclic peer multi-agent 和一个 mixed workflow 可重复运行；
- FRESH child 不继承 parent pages，causal/data/prefix 三种边可分别重建；
- structured action coverage、boundary-token 分布、tool-start gap 和 action-critical inversion 被报告；
- B0-B4 与 O0-O3 可以在同一 trace、同一物理 accounting 上 replay；
- What-if packer 无副作用且满足 closure/capacity/fairness/liveness；
- 预注册 workflow JCT 为 primary metric；O3 相比该指标上的 `best(B0..B4)` 上界收益至少
  10%，且 action throughput/causal-blocked time/unhidden stall 至少一项提供一致的机制证据；
  如果只优于 B0，不能证明新的竞争边界；
- `min(cost(O1), cost(O2)) - cost(O3)` 的置信区间高于 0，否则 JointPlan 降级为工程统一接口；
- action-unlock oracle 相比 B1/B2 的增量收益单独报告，但在 P5.5 前不据此启用在线策略；
- 当前 workload topology entropy、cycle、handoff 和 consumer fan-out 被正式报告。

### P4：Observed-State JointPlan Shadow Mode

实现文件：

- 新增 `policy/joint_scheduler.py`；
- 修改 `control/controller.py`、`runtime/audit.py`；
- 新增 `tests/test_joint_scheduler.py`；
- 扩展 `tests/test_controller.py`。

任务：

1. 只使用已观测 `G_t/R_t` 生成完整 `JointPlan`，prediction 关闭；
2. JointPlan 同时给出 workflow/agent order、admission、restore dependency 和 KV bundle action；
3. blocking parent、tool wait、background spawn、handoff 和 message 分别进入正确运行队列；
4. 记录与现有 admission/transfer/waiting 独立排序的差异；
5. 分别计算 scheduler-only、KV-only 和 joint counterfactual regret；
6. 将 planner 从固定 5ms 全量扫描改为 runtime/allocator/ACK 事件触发，保留 watchdog tick；
7. 限制 workflow/frontier 候选和规划预算，超时生成 observed-state conservative JointPlan；
8. 在 shadow mode 校验 plan dependency：需要 restore 的 request 不得先于 ACK admission；
9. 同一 snapshot 记录可在线执行的 reference decision，并保存 B1/B2 oracle replay 所需输入；
   hindsight decision 只能在 run 完成后生成，所有 reference policy 都不能改变真实队列或 KV；
10. `ActionFrontierObserver` 只记录 parser/boundary 状态，不改变 decode order。

退出条件：

- P99 planning overhead 小于 1ms，或小于 scheduler step 的 5%；
- 计划在执行前 stale 的比例低于 10%；
- physical plan rejection 不高于当前 reactive planner；
- shadow-mode joint regret 稳定优于当前独立排序；
- nonblocking/background parent 从不因 `SPAWN` 被错误标为 parked/offload；
- 同一 root workflow 的 fan-out 不增加 workflow service budget；
- online/oracle reference-policy decision 与其 metadata assumption 可由 audit 完整重建；
- action observer 对未支持的自由文本输出明确返回 UNKNOWN，不伪造 action boundary；
- 在 prediction 关闭时即可重建所有 JointPlan 决策和 fallback。

### P5：启用 Observed-State Joint Agent/KV Scheduling

实现文件：

- 修改 `control/controller.py`、`policy/admission.py`、`policy/transfer_planner.py`；
- 修改 `runtime/sglang_v052rc1.py`；
- 扩展 `tests/test_sglang_adapter.py` 和真实压力脚本。

启用顺序：

1. 先应用完整 JointPlan，但 ResidencyIntent 仅允许 KEEP/DROP_DEAD 和已有 reactive bundle；
2. AdmissionController 改为只兑现 AdmissionIntent，不再自行选 workflow；
3. waiting queue 改为只应用 ExecutionIntent，不再自行重算 fairness/frontier；
4. transfer planner 改为只编译 ResidencyIntent，不再自行选 victim/prefetch context；
5. 启用 blocking parent/tool-wait exclusive-suffix offload 和 READY restore dependency；
6. 启用 cyclic peer residency hysteresis 和 multi-consumer admission；
7. prediction、scenario reservation 和 reveal 保持关闭；
8. 在 blocking、peer 和 mixed 并发 workload 中逐项 A/B；
9. B0-B4 online mode 保持 shadow logging，B1/B2 oracle mode 留到 run 后 replay，验证两类输入
   都与相应时刻物理状态一致。

退出条件：

- correctness/liveness 不低于 P0；
- workflow Jain fairness 和最大 virtual-runtime lag 不退化；
- F3 resource inversion 频率和影响显著下降；
- scheduler-selected request 与 KV/admission plan mismatch 为 0；
- `RESTORE_PENDING` request 在 bundle 可用前不会进入运行 batch；
- cyclic handoff 的无效 H2D/D2H oscillation 相比无 hysteresis baseline 显著下降；
- 实际 JCT 收益不是只来自改变随机生成路径。

### P5.5：Action-unlock Insight Gate

该阶段不立即实现新预测器，而是利用 P3 action trace 和 P5 稳定数据面判断 action-frontier
是否值得成为 P6 主线。它回答的是目标函数和控制动作是否新，而不是某个模型能否拟合标签。

任务：

1. 定义 `time_to_valid_action`：READY/active invocation 到合法 tool/spawn/handoff/final
   action 的时间；
2. 定义 action-critical inversion：B1/B2/B3 选择的 agent 与最早解除 causal blocker、启动
   external work 或扩大 runnable frontier 的 agent 不同；
3. 使用真实 boundary、fan-out、consumer 和 reentry 构造 action-unlock oracle；
4. 比较三类目标：invocation distance、AugServe value density、action unlock；
5. 将每个候选 `RUN(q)` 与其 KV 状态转换联合模拟：

   ```text
   RUNNING --valid action--> TOOL_WAIT / CHILD_WAIT / HANDOFF / DONE
          --KV effect-----> active pinned KV becomes waiting/reclaimable/dead
   ```

6. 评估 uncertainty commitment oracle：错误未来信息下，直接 KEEP/SWAP/DISCARD 与
   KEEP->SHADOW->COMMIT ladder 的 regret；
7. 单独报告 PCIe 无空闲、structured-action coverage 低、action criticality 相同等负场景。

进入 P6 action-frontier 分支的条件：

- 至少两类动态 workload 中，action-critical inversion 稳定出现且贡献至少 10% 的 JCT、
  causal-blocked time 或 tool-launch idle gap；
- action-unlock oracle 相比带 hindsight 信息的 B1/B2 在 workflow JCT 上至少提升 10%，置信
  区间不跨 0，并降低 causal-blocked time 或 tool-launch idle gap；
- structured action 或可靠 runtime boundary 覆盖主要执行时间，UNKNOWN 不成为主导状态；
- reversible commitment oracle 相比 point-decision policy 有正净收益，且收益不是只来自
  理想 PCIe 带宽假设。

若 action-unlock 条件不满足，不能为了保留论文故事强行启用。P6 回到原 Dynamic Frontier
分支，但也必须通过独立 oracle gate；两者都不成立时，predictive policy 停止扩展，P5
Observed-State JointPlan 作为最终系统策略。

### P6：Uncertainty-Gated Predictive JointPlan

P6 不再首先训练一个高维的
`P(next owner, time, prompt delta, action boundary)` 单体模型。未创建 agent 的具体 owner ID
不可预测，组合标签也难以跨 workload 校准。预测被拆成两个低维、可独立回退的 competing-risk
接口：

```text
UnlockHazard
  P(action_type, tokens_to_valid_action, fanout_class
    | active token/parser state, observed RCCG)

ReentryHazard
  P(return/reactivation within t, prompt_delta_bucket
    | waiting state, observed RCCG)
```

任务：

1. 新增 `predictor/unlock_hazard.py` 和 `predictor/reentry_hazard.py`；
2. `UnlockHazard` 使用增量 structured parser/token state；自由文本无法可靠解析时输出 UNKNOWN；
3. `ReentryHazard` 使用 tool/subagent/handoff waiting state，并输出 time 与 prompt-delta bucket；
4. 现存 agent 可以预测 owner；尚未创建的分支只能输出 `NEW_CONTEXT + role/action class`；
5. `scenarios.py` 将两个 hazard、observed consumer 和 physical demand 组成 top-K 场景；
6. 先在 shadow mode 比较 coverage、calibration、OOD、真实 transition 和 offline oracle；
7. P5.5 gate 通过时，JointPlan 联合枚举 `RUN(q)`、ADMIT/PAUSE 与
   KEEP/SHADOW/COMMIT_CPU/DISCARD/PREFETCH；否则不加入 action-frontier 动作；
8. 预测置信度决定动作可逆程度，而不只是改变 priority：

   ```text
   low/OOD     -> P5 OBSERVED_JOINT，不做 speculative commit
   medium      -> PREPARE_HOST/SHADOW，保留 GPU copy
   high + safe -> COMMIT_CPU/DISCARD/PREFETCH
   ```

9. prediction 不得直接创建/调度未观测 agent，也不得绕过 physical bundle admission；
10. OOD 或覆盖不足时整份 plan 回退到 P5 `OBSERVED_JOINT`；
11. 只有 scenario action 显著分歧、存在 READY revealer 且 `P90(T_reveal) < T_force` 时启用
   Reveal-and-Commit；
12. 对错误 action boundary、next consumer、循环提前终止、动态 fan-out、早醒、HBM emergency
   和 stale plan
   做故障注入。

退出条件：

- Unlock/Reentry hazard 分别报告 coverage、calibration、OOD、跨 workload transfer 和
  decision regret；
- prediction 相比 P5 observed-state JointPlan 有独立、稳定的 JCT 或 unhidden-stall 收益；
- 启用 action-frontier 时，time-to-valid-action、action throughput 或 causal-blocked time 至少一项
  相比 strongest B1-B4 reference policy 有稳定收益；
- wasted shadow/prefetch bytes、额外 HBM-time 和 delayed-workflow cost 小于避免成本；
- reveal precision、coverage 和净收益单独报告；若无收益则关闭 reveal，不影响 frontier policy；
- OOD fallback 与 P5 的 decision trace 一致，且不劣于 strongest observed baseline。

### P7：较新 HiCache 可移植性

任务：

1. 保留 pinned `0.5.2rc1` 归因实验；
2. 新增上游版本 adapter，不复用私有 API 假设；
3. capability negotiation 决定 merge/layer event/multi-inflight 是否可用；
4. 在新版 overlap 和 operation merge 下重新运行 strongest baseline/full policy；
5. 不将 HiSparse 纳入普通 MHA/GQA 主实验，除非模型本身使用 DSA。

退出条件：

- 收益不依赖旧版 callback、allocator 或缺失 overlap；
- 新旧数据面上的策略决策语义一致；
- 任何版本特有优化单独报告。

### P8：原生竞品系统与两层端到端对比

P8 在 BeliefKV P5/P6 功能、正确性和配置冻结后执行。此前不要求为部署外部系统暂停主线，
但 P2.5-P6 的 reference-policy 日志、ID 映射和 common workload 必须为该阶段准备完毕。

任务：

1. 在同一 BeliefKV/SGLang 数据面主动运行 B0-B4 的 online-capable variant，形成算法层对比；
   B1/B2 hindsight variant 继续只作 replay oracle；
2. 对代码可用且硬件支持的 ScaleSim、ThunderAgent 等部署原生版本；
3. 对暂时无完整代码或依赖不同 serving engine 的方法，优先复现论文核心 policy，并明确
   native/unavailable/reimplemented 状态；
4. 在 common-denominator tool workflow 上比较 reference policy 与 native system 的方向、排序
   和端到端趋势，验证重实现忠实度；
5. 在 blocking、cyclic peer 和 mixed workflow 上报告 full-scope 结果；竞品缺少语义接口时
   报告 unsupported capability，不把启动失败计为加速比；
6. 统一或显式报告模型、量化、KV pool、SGLang/vLLM 版本、PCIe、batch 和 workload arrival；
7. 分开报告：

   ```text
   same-data-plane policy comparison  -> 隔离算法差异
   native end-to-end comparison       -> 比较完整系统竞争力
   ```

退出条件：

- ScaleSim、AugServe、ThunderAgent、CONCUR 至少都有 reference-policy 结果；
- 所有代码开源且硬件兼容的直接竞品都应尝试 native 部署；至少完成一个，目标完成两个，
  其余必须记录不可运行的具体原因；
- reference 与 native 在重叠 workload 上不存在无法解释的趋势反转；
- 所有竞品的额外 metadata、硬件和部署限制在表格中明确列出；
- 论文主结果同时包含 strongest same-data-plane baseline 和 strongest runnable native baseline。

## 12. 测试计划

### 12.1 单元测试

必须覆盖：

- lease 状态映射和多 owner 最强 lease；
- conditional resume condition 的创建、满足和取消；
- FRESH/FORK/RESUME 的不同物理语义；
- D2H leaf closure 和 H2D ancestor closure；
- shared prefix 的 marginal bytes；
- top-K scenario 概率、OTHER/OOD 和校准；
- PolicyInput/PolicyOutput schema、metadata assumption 和 hindsight 在线隔离；
- B0-B4 reference policy 在固定 snapshot 上的确定性与 resource accounting 一致性；
- structured action parser、合法 boundary、UNKNOWN 和 malformed output；
- UnlockHazard/ReentryHazard 的 calibration、OOD 和 fallback；
- 置信度到 KEEP/SHADOW/COMMIT 的 commitment ladder；
- observed consumer edge 与 predicted consumer hypothesis 不混入 RCCG；
- what-if plan 容量、liveness 和 fairness 约束；
- scenario consensus 和 divergence；
- OBSERVED_JOINT/PREPARE/REVEAL/COMMIT/ROBUST_BELIEF 转移；
- Execution/Admission/Residency dependency 的一致性；
- AdmissionController、waiting queue 和 transfer compiler 不重新选择 JointPlan action；
- background SPAWN 不 park parent，JOIN_WAIT/foreground CALL 才产生 blocking lease；
- cyclic reactivation 的 residency hysteresis 和超压中止；
- stale graph/page generation 回退；
- partial/rejected/cancelled ACK；
- 同一 fingerprint 的 capacity/closure/lock reject 不发生 tick-driven retry；
- blocker 解除后立即恢复、错误 event 不解除、context epoch 变化清理 ledger；
- telemetry 缺失时的保守 service curve。

### 12.2 集成测试

使用 fake HiCache 构造：

1. 两个 context 共享 ancestor，一方 RUNNING、一方 DEAD；
2. parent 等待多个 FRESH child，parent 低 hotness 但 join 后确定复用；
3. peer agent message 使 CPU_ONLY context READY；
4. 低优先级大 prefetch 与高优先级 READY agent 竞争 PCIe/HBM；
5. reveal 前后两个场景需要相反 victim set；
6. reveal 超时或预测分支错误；
7. H2D allocator failure 和 D2H partial ACK；
8. cache reset、abort 和 node generation reuse。
9. 同一 context 的 H2D ancestor/capacity reject storm 与其他候选绕行。
10. background child 与 parent continuation 同时 READY，parent bundle 不被错误迁移；
11. Coder-Reviewer-Test cyclic handoff，多轮 reactivation 不发生无界 KV oscillation；
12. 一个 producer message 唤醒多个 consumer，共享 bytes 只计一次但 admission 分别执行；
13. peer agent 内嵌 blocking FRESH subagent，局部 lease 不依赖全局 workflow mode。
14. 两个 active agent 中，一个接近 tool action boundary，验证 shadow action-unlock plan 不修改
    P4/P5 真实执行；
15. action boundary 后 active KV 转为 waiting/dead bundle，RUN(q) 与 residency transition 原子一致；
16. B1/B2 使用 hindsight metadata 时在线 planner 无法读取对应字段。

### 12.3 真实 GPU 测试

每个正式配置至少运行：

- pinned upstream SGLang Radix LRU；
- HiCache write-back；
- HiCache write-through；
- HiCache write-through-selective；
- HiCache + reactive causal policy；
- B1-B4 online-capable same-data-plane reference policy；hindsight variant 仅 replay；
- BeliefKV observed-state JointPlan，不使用预测；
- BeliefKV frontier-belief JointPlan，不使用 reveal；
- BeliefKV full policy，Reveal-and-Commit 仅在满足 gate 时开启；
- scheduling-only oracle、KV-only oracle 和 full JointPlan oracle；
- P8 中可运行的 native ScaleSim/AugServe/ThunderAgent/CONCUR baseline。

统一模型、SGLang commit、量化、KV pool、并发度、随机种子和冻结 workload。动态生成无法完全
冻结时，必须报告每次 run 的 request/event sequence hash，不允许直接比较不同轨迹的 JCT。

## 13. Workload 与指标

### 13.1 Workload

正式矩阵不能只使用“每个 root 固定创建两个 child”。至少覆盖：

| 场景 | 必须出现的动态性 | 首选任务来源 |
| --- | --- | --- |
| Blocking subagent | variable fan-out、unbalanced join、tool wait | SWE-bench/真实 coding task |
| Nested subagent | child 动态 spawn grandchild | coding/research task |
| Early return/cancel | ANY/quorum/失败后取消剩余 child | search/validation task |
| Cyclic peer multi-agent | Coder-Reviewer-Test 多轮 handoff | SWE-bench 或真实代码任务 |
| Multi-consumer message | 一个结果唤醒多个 persistent peer | research/debate task |
| Nonblocking parent | parent continuation 与 background child 并行 | async subagent runtime |
| Mixed workflow | peer multi-agent 内嵌 blocking subagent | coding/research task |
| Sequential ReAct | 只有一个 runnable agent | 预期低收益退化基线 |

P3 的最低可执行集合是 blocking、cyclic peer 和 mixed 三类；其余按 characterization 逐步
加入。角色集合可以预定义，但 next speaker、循环次数、fan-out、consumer 和终止路径必须由
运行时输出决定，不能通过固定 trace 顺序伪造动态性。

每类 workload 都要报告：

```text
结构：fan-out、depth、cycle count、join mode、handoff entropy、consumer fan-out
时间：tool/agent demand、join span、semantic-ready 到 actual-service delay
KV：context bytes、parent-child/sibling/peer affinity、physical shared bytes
调度：runnable frontier size、workflow service、selected-agent progress
预测：top-k coverage、calibration、OOD、decision regret
Action：boundary coverage、time-to-valid-action、tool-start gap、frontier delta、inversion
系统：JCT、fairness、admission tail、HBM-time、PCIe bytes、unhidden stall
```

### 13.2 主指标

- workflow mean/P50/P95 JCT；
- agent resume TTFT 和 admission wait；
- completed workflows/hour；
- workflow Jain fairness 和最大 service lag；
- causal progress rate、ready-to-service delay 和 per-workflow runnable frontier size；
- time-to-valid-action、valid actions/s 和 causal-blocked time；
- tool/subagent launch idle gap 和 action unlock 前后 runnable frontier delta；
- GPU utilization、HBM peak 和 OOM/liveness failure。

### 13.3 KV/PCIe 指标

- actual D2H/H2D/recompute/drop bytes；
- useful/wasted shadow bytes；
- physical closure amplification；
- planned/actual freed bytes；
- Host allocation failure、partial/reject rate；
- raw transfer time 与 actual unhidden stall；
- HBM-time，单位 byte-ms；
- prefetch hit、early residency 和 late restore stall。
- identical-retry concentration、suppressed retry、retry release latency 和 false suppression；
- parent-child、sibling/template、peer 和跨 workflow prefix affinity 分布；
- cyclic handoff 的 H2D/D2H oscillation bytes 和 residency tenure。

### 13.4 联合决策指标

- scenario coverage、calibration 和 OOD rate；
- next speaker/handoff/consumer/fan-out/loop transition 的分类与 demand calibration；
- scenario-dependent optimal action 占比；
- consensus/conditional action 比例；
- revealer available 和 `T_reveal < T_force` 比例；
- reveal success/timeout/mismatch；
- decision regret 相比 reactive 和 oracle；
- plan stale、fallback 和 planning overhead；
- execution-admission-residency mismatch count，要求为 0；
- O0/O1/O2/O3 oracle 和 joint synergy gap；
- B0-B4 reference policy、best-reference gap 和 metadata advantage；
- action-critical priority inversion 的频率、代价和涉及的 active-to-waiting KV bytes；
- KEEP/SHADOW/COMMIT 的 precision、commit/cancel rate 和 commitment regret；
- F1-F4 failure 的出现频率与 JCT 贡献。

## 14. 风险与降级策略

| 风险 | 判断 | 降级 |
| --- | --- | --- |
| 多数时刻只有一个 runnable agent | reveal/agent ordering 无选择空间 | 保留 observed-state JointPlan，只做 exact admission/KV plan |
| 不同场景的物理工作集接近 | 消歧无价值 | 立即执行 consensus/普通 causal frontier |
| reveal 比强制内存 deadline 慢 | 早晚都可能损失 | 直接 ROBUST，不改变 agent 顺序 |
| predictor 跨 workload OOD | 预测迁移可能恶化 | 回退 P5 observed-state JointPlan |
| parent-child prefix affinity 很低 | 因果关系无法带来共享复用 | 优化 independent working-set admission/rotation；shared lease 只做正确性 |
| multi-agent adapter 丢失 handoff/consumer | 活跃 peer 被误迁移 | 未确认关系使用 READY/RUNNING safety lease，禁止预测性 commit |
| cyclic handoff 导致 KV 抖动 | H2D/D2H 大于复用收益 | residency hysteresis、minimum tenure、HBM emergency override |
| JointPlan 只统一接口但无 synergy | 创新退化为工程重构 | 用 O1/O2/O3 oracle gate；无 gap 时不作为 Major contribution |
| 当前 workload topology entropy 低 | 无法证明动态 MAS | P3 前置 cyclic peer 和 mixed workload，不用固定两-child trace 训练/评价 predictor |
| ScaleSim/AugServe reference 已解释主要收益 | BeliefKV 目标没有独立价值 | P3B/P5.5 使用 hindsight competitor oracle；无 gap 时停止 predictive 分支 |
| dynamic trace 对调度敏感 | 冻结 replay 产生虚假 oracle 收益 | 标注 schedule sensitivity；semantic race 只作上界并用真实 A/B 闭合 |
| structured action coverage 低 | UnlockHazard 无法稳定工作 | UNKNOWN 回退 P5；action-frontier 不作为 Major contribution |
| action criticality 与短作业/value-density 等价 | action-unlock 退化为 AugServe 变体 | P5.5 比较同真实长度/工具时间的 B2 oracle，无独立 gap 即停止 |
| CPU shadow 可用 PCIe 窗口很少 | reversible commitment 只有机制价值 | 单独报告 idle service、commit/cancel 和净收益，不作为必要组件 |
| HiCache overlap 隐藏大部分 restore | KV 预测收益缩小 | 以 unhidden stall 为准，降低该贡献优先级 |
| shared/closure 使计划频繁失效 | context 估计不可信 | 强制 physical preview 和 generation 二次校验 |
| reject 结果未反馈导致 retry storm | PCIe/CPU 与 scheduler 开销被无效命令放大 | typed blocker + event-gated attempt ledger；已知 blocker 禁止 tick retry |
| Python planner 开销过高 | scheduler step 被拖慢 | top-K/深度/候选上限，事件触发，超时回退 |
| 新版 HiCache 已覆盖部分机制 | 贡献边界收缩 | 保留 agent execution/KV joint decision，删除通用机制主张 |

## 15. Go/No-Go 与论文主张门槛

### 15.1 Characterization 门槛

在至少三类真实 MAS workload 中：

- P2 H2D/allocator/Radix 可靠性 gate 必须先通过；
- 至少覆盖 blocking subagent、cyclic peer multi-agent 和 mixed workflow；
- HBM-pressure 时必须稳定出现多个已观测 runnable agent，且 agent 选择会改变 admission 或
  optimal physical bundle plan；
- B0-B4 至少完成同 snapshot reference replay，并声明各自 metadata assumption；
- O3 full JointPlan oracle 相比 `best(B0..B4)` 的 mean/P95 workflow JCT 上界收益至少 10%，
  并由 action throughput、causal-blocked time 或 unhidden stall 中至少一项解释；只优于
  O0/B0 不足以继续；
- O3 相比 `best(O1 scheduling-only, O2 KV-only)` 的 synergy gap 置信区间高于 0；
- 至少 20% 的受压决策存在 scenario-dependent execution/admission/residency action；
- subagent 与 multi-agent 的 frontier transition 都有足够 coverage，不能只由固定二 child
  workload 支撑 predictor。

P6 action-frontier 分支还必须满足 P5.5 的独立门槛：action-critical inversion 有实际代价，
action-unlock oracle 优于带 hindsight 的 B1/B2，且 reversible commitment 在实测 PCIe service
下有正净收益。否则不能把 calibration、action SLO 或 shadow copy包装成核心贡献。

任一关键门槛不满足，应停止扩大 predictive JointPlan。JointPlan 可保留为工程统一接口，
RCCG lease/ownership bridge 保留为正确性层，但不能作为 Major contribution。

Reveal-and-Commit 使用独立可选 gate：只有高代价 scenario disagreement 中存在 READY
revealer、`P90(T_reveal) < T_force` 且 offline reveal oracle 有净收益时才实现主动版本。它不再
是整篇工作的必要门槛。

### 15.2 系统结果门槛

完整策略必须同时满足：

- P5 observed-state JointPlan 相比 strongest separate/HiCache/deployable reference baseline 有
  独立收益；hindsight oracle 只用于报告剩余上界 gap；
- P6 predictive JointPlan 相比 P5 仍有独立收益；
- workflow JCT 或 unhidden stall 至少降低 10%，且置信区间不跨 0；
- 无效 D2H/H2D/recompute 或 unhidden stall 至少降低 15%；
- fairness、OOM、liveness 和 correctness 不退化；
- execution/admission/residency mismatch、错误 parent park 和未满足依赖的 request dispatch 为 0；
- 新版 HiCache 数据面上收益仍存在；
- OOD 时不劣于 P5 observed-state JointPlan；
- P8 的 strongest runnable native baseline 和 same-data-plane reference baseline 均被纳入主表。

### 15.3 可接受的最终贡献结构

如果上述证据成立，可以形成三层贡献：

1. **Characterization**：通用 HiCache policy 在动态 MAS 中的 F1-F4 failure，尤其是
   动态 runnable frontier、agent execution 和未来物理工作集之间的反馈，以及 causal/data/
   prefix affinity 不一致；
2. **Mechanism**：observed RCCG/data-consumer state 到 physical causal lease/bundle 的安全
   bridge，以及统一 Execution/Admission/Residency dependency；
3. **Algorithm**：在 HBM/PCIe/fairness 约束下的 Dynamic Frontier JointPlan；若 P5.5 gate
   通过，则进一步包含 action-frontier `RUN(q)`、Unlock/Reentry hazard 和 uncertainty-gated
   reversible commitment；否则只保留有独立 oracle 证据的局部 frontier scenario
   consensus/robust planning，以及有证据时的 Reveal-and-Commit。

其中 HiCache、shadow copy、page transfer 和预测模型结构本身不作为贡献。

## 16. 推荐的首批提交顺序

为了保持代码可审阅和结果可归因，建议拆成以下独立提交：

```text
feat(runtime): expose HiCache capabilities and transfer telemetry
feat(policy): add online transfer service curves
fix(policy): gate transfer retries on typed physical state changes
feat(policy): project RCCG state into causal leases
feat(runtime): build closure-aware physical KV bundles
fix(runtime): make H2D closure admission and rollback allocator-consistent
test(experiments): close the paired P1.5/P2 physical reliability gate
feat(policy): add common reference-policy input and output contract
feat(policy): add ScaleSim AugServe ThunderAgent and CONCUR reference policies
feat(experiments): add hindsight metadata-isolated policy replay
feat(control): track observed producer-consumer relationships
feat(runtime): capture peer handoff message and reactivation events
feat(runtime): record structured action frontier and valid action boundaries
feat(experiments): add cyclic peer and mixed dynamic workflows
feat(policy): add side-effect-free scenario what-if packer
feat(experiments): add reference scheduling-only KV-only action and joint oracles
feat(policy): add observed-state JointPlan in shadow mode
feat(runtime): apply JointPlan execution admission and transfer dependencies
feat(policy): enable observed-state joint agent and KV scheduling
docs(experiments): report action-unlock insight gate and branch decision
feat(predictor): add calibrated unlock and reentry hazard scenarios
feat(policy): add robust predictive JointPlan actions
feat(policy): add guarded reveal-and-commit when oracle-gated
test(experiments): add pinned and current-HiCache policy matrix
test(experiments): add same-data-plane and runnable native competitor matrix
docs(results): report dynamic-workflow joint-synergy and go-no-go decision
```

每个提交必须通过当前测试，并保持新策略默认关闭。P2 reliability、P2.5 reference contract、
P3 joint/competitor oracle 和 P4 shadow-mode 是继续 P5/P6 的顺序硬门槛；P5.5 是选择 P6
action-frontier 或 generic frontier 分支的硬门槛。不能先训练 predictor，再寻找能证明它有效的
workload。P8 native deployment 可以延后，但不能缺席最终系统结果。

## 17. 最终落点

当前 BeliefKV 不需要再增加一个更复杂的预测模型，也不应重建 HiCache 数据面。最可落地
的改进是先建立以下统一决策单位：

```text
observed RCCG + data-consumer state G_t
  + calibrated Unlock/Reentry or local frontier belief H_t
  + physical Radix bundle/closure
  + exact HBM/PCIe service state
  = auditable JointPlan(execution, admission, residency, dependencies)
```

blocking subagent 是第一个可控实现切片，不是系统的最终定义。P3 必须补充 cyclic peer
multi-agent 和 mixed workflow，并通过 B0-B4 reference policy 及其 hindsight variant 检查竞争
边界；P4/P5
先证明不依赖预测的 agent/KV JointPlan 有效；P5.5 决定 action-unlock 是否值得成为 P6
主线；P6 再证明预测相对 observed-state JointPlan 和 strongest reference policy 的增量价值。
Reveal-and-Commit 只有在独立 oracle gate 通过时启用，P8 再完成同数据面与原生系统两层对比。
这样既不会把动态 workflow 简化成 spawn 后的静态 DAG，也不会用一个 MLP 掩盖 agent 调度、
物理 KV 和运行时因果之间真正需要联合解决的问题。
