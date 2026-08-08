# BeliefKV 当前版本改进方案：基于 HiCache 数据面的 Agent/KV 联合控制

初始日期：2026-07-18
最新修订：2026-07-31
状态：实施中；P2 显式 bundle 可靠性 gate 与 P2.5 已通过；P4 修复后 24-workflow GPU 复验
已形成 96% KV 压力并验证 admission/迁移活性，但 JointPlan planning budget 和 workload 终止性
未过 gate；P5D 失败项的代码修复已完成并通过全量 CPU 回归。2026-07-29 的单次固定 w4
clean gate 进一步暴露 restore 后过早再次驱逐，以及 funding capacity 未归属于 restore debt：物理
command/ACK 正确性通过，但 0/4 clean completion。service-quantum grace 与 debt-owned funding
reservation 已完成 CPU 实现和回归。P5 物理迁移、restore/retraction/admission liveness 作为
P6 的 observed-state 安全底座冻结；完整 agent workload clean-completion gate 仍未关闭，不能把
P5 描述为性能完成。P6 已进入训练前契约开发，所有预测动作默认保持 offline/shadow
依赖分析：

- [BeliefKV 相关工作与竞品对比总表](related_work_comparison_2026-07-21_zh.md)
- [HiCache / HiSparse / Theta KVPool 对 BeliefKV 的启发与差异分析](hicache_theta_kvpool_implications_2026-07-18_zh.md)
- [BeliefKV 面对动态 Agent Workflow 需要注意的地方](beliefkv_dynamic_agent_workflow_considerations_2026-07-20_zh.md)
- [P2 Physical Causal Lease 与原子 Bundle 执行](experiments/beliefkv_p2_physical_bundle_2026-07-19_zh.md)
- [P3 Rolling Physical Replay 与 Trace-order Oracle 检查点](experiments/beliefkv_p3_rolling_oracle_2026-07-21_zh.md)
- [P3 动态并发 GPU Characterization](experiments/beliefkv_p3_dynamic_gpu_validation_2026-07-21_zh.md)
- [P4 24-workflow Chunked-Prefill Correctness Gate](experiments/beliefkv_p4_chunked_prefill_gate_2026-07-23_zh.md)
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

P2-P6 不搭建 ScaleSim、AugServe、ThunderAgent、CONCUR 等方法的统一对比框架，也不为其
实现 adapter、预测器或前端 metadata producer。当前已经存在的不可变
`PolicyInput/PolicyOutput`、中立 trace 和 reactive replay 只服务于 BeliefKV 自身的正确性、
O0-O3 oracle 和回归验证，不再承担“兼容所有竞品”的职责。早期竞品策略草图和专用 hindsight
注入已从维护代码删除，不恢复也不继续扩展。

BeliefKV 的功能、正确性和配置在 P5/P6 稳定后，P8 再重新评估场景重叠、metadata 假设、公开
代码和硬件兼容性，并据此选择可以忠实运行的比较方法。当前计划不预设一定采用
same-data-plane adaptation、metadata oracle 或原生部署中的哪一种，也不要求核心系统为了未来
竞品修改在线接口。

该调整不意味着最终论文可以回避相关工作。它只避免在动态 subagent workload、JointPlan
接口和物理数据面仍变化时，用不忠实的策略改写污染核心架构。现在只采集解释 BeliefKV 决策
和复现实验所必需的运行时事件、物理 extent、资源和服务 trace；不得为了潜在竞品额外合成
invocation distance、tool duration、program phase 或其他应用先验。

BeliefKV 不再表述为 metadata-free。系统要求 invocation/context identity，以及已经发生的
spawn、wait、return、message、handoff 和 tool 等最小在线因果事件；它不要求的是先验完整 DAG、
未来 agent、invocation distance、未来工具时长或 program phase。论文定位统一为：
**无需先验 DAG、依靠最小在线因果事件动态发现 workflow**。

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
可逆程度。P5.5 先相对当前 reactive policy、scheduler-only、KV-only 和 observed-state
JointPlan 验证该目标是否有独立机制收益。与 ScaleSim/AugServe 等工作的最终实验边界在系统
冻结后重新判断，不再阻塞 P6 的实现与正确性验证。

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

### 2.8 竞争边界与后置比较原则

现阶段不能再把以下单点作为 BeliefKV 的创新：

- 根据未来调用远近做 KV eviction/prefetch，ScaleSim 已使用 invocation distance；
- 根据输出长度和工具耗时选择 Preserve/Swap/Discard，AugServe 已使用预测和显存-时间成本；
- 将 agent phase、waiting queue 和 KV pause/restore 联合管理，ThunderAgent 已实现；
- 根据 KV pressure/hit rate 调整 active agent 数，CONCUR 已使用 AIMD feedback control。

这些工作仍决定 BeliefKV 可以声明什么，但不应在系统尚未稳定时决定其内部接口，也不能假设
它们可以直接迁移到动态 multi-agent/subagent workload。当前实施边界是：

```text
P3-P6 主线
    observed-state reactive baseline
    SGLang/HiCache 原生数据面基线
    BeliefKV 自身 PolicyInput/physical trace、O0-O3 联合性 oracle
    不实现竞品 adapter，不为其合成前端 metadata

系统冻结后的 P8
    先检查论文代码、输入假设、工作流语义和硬件是否真正可比
    再为可比方法选择原生复现、共同 workload 或最小忠实适配
    不要求所有方法进入同一个 BeliefKV policy framework
```

P8 不再是 P4-P6 的 correctness 或实现前置条件。最终投稿前仍需完成可比且可运行的强基线，
但比较集合由届时的系统能力和 workload intersection 决定。对依赖 invocation distance、静态
DAG、program phase 等前端先验的方法，只有真实应用能够提供相同信息时才进入在线性能对比；
否则只在 related-work/assumption 表中说明边界，不为制造数值结果而构造不忠实实现。

## 3. 目标架构

```text
Runtime events + incremental parser       SGLang/HiCache physical state
RCCG/consumer/action frontier             allocator/Radix/lock/telemetry
                 |                                      |
                 +------------------+-------------------+
                                    v
                         P5 Observed JointPlanner A0
                 execution/admission/residency/retraction
                                    |
                       persistent liveness hard state
                    obligation / lease / escrow / grace
                                    |
          +-------------------------+-------------------------+
          |                                                   |
          v                                                   v
 P5 bounded/emergency path                         P6 closure-complete scope
 no predictor dependency                          causal atoms + OTHER
          |                                                   |
          |                                          FrontierBeliefSnapshot
          |                                      global top-K joint scenarios
          |                                                   |
          |                                      finite-horizon risk planner
          |                                  A0 vs PREPARE_HOST/PREFETCH_GPU
          |                                                   |
          +-------------------------+-------------------------+
                                    v
                       PredictivePlanEnvelope / JointPlan
                             independent ActionGroups
                                    |
                         scheduler safe-point committer
              group-local read-set validation + physical rematerialization
                                    |
                  current allocator/closure/liveness certificate
                                    |
                      RadixArbiter -> HiCache -> ACK
```

职责边界如下：

| 模块 | 拥有的决策/状态 | 不允许承担的职责 |
| --- | --- | --- |
| RCCG | 已发生的因果关系、liveness、ready/parked、join/handoff/message | 把未来 hypothesis 写成事实；token prefix 和物理 residency |
| Data-consumer index | 已观测消息/return 消费者和潜在 consumer identity | 用因果 parent 替代实际数据 consumer |
| Frontier predictor | closure-complete scope 的全局联合场景、OTHER mass、coverage 和 OOD | 预测完整 DAG；独立决定迁移或把 P5E liveness 概率化 |
| Lease projector | 将 RCCG 状态映射成有限的资源承诺 | 修改物理 page |
| Ownership bridge | owner、Radix closure、实际边际字节 | 猜测 agent 状态 |
| Joint planner | 以 A0 为保底，在有限 horizon 内选择完整 ActionGroup | 让下游模块重新独立排序；拆分原子 group；绕过 arbiter 和 ACK |
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

该扩展不是 P4/P5 的 correctness 必需项，只有 P5.5 证明 action-unlock oracle 相比当前
reactive policy 和独立 scheduling/KV oracle 存在收益后才进入 P6。其关键
不是增加一个 priority score，而是把短 decode quantum 与
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

SGLang 始终拥有 waiting queue，不再由 BeliefKV 抽取和稍后放回 request。
`ExecutionIntent.ordered_request_ids` 在当前 batch-construction epoch 被编译成短期 ticket 的
eligibility 和优先级；未获得 ticket 的 tagged request 本轮跳过但继续留在原生队列，未携带
BeliefKV metadata 的请求不受影响。

P4 只生成和验证 shadow plan，现有 reactive controller 仍是实际动作的唯一写者；**从 P5
开始**强制以下 JointPlan 单写者规则：

```text
JointPlanner 是 tagged agent request 顺序、admission 和策略性 KV 动作的唯一决策者。
AdmissionController 只校验/兑现 AdmissionIntent，不再重新选择 workflow。
ReactiveTransferPlanner 只把 ResidencyIntent 编译成 command，不再重新选择 victim。
SGLang admission gate 只消费由 ExecutionIntent 编译的当前 epoch ticket，不再重新计算策略。
RadixArbiter 仍可因物理状态拒绝计划，但不能替换为另一个策略动作。
```

未携带 BeliefKV metadata 的普通请求继续采用 SGLang 原生策略，不能被 JointPlan 饿死；其
容量和服务消耗作为 external charge 反馈给下一次规划。

### 7.1 吞吐优先的软公平约束

联合策略保留两级排序信息，但 workflow 公平不作为 batch admission 的硬配额：

1. **workflow 层**：按 attained GPU service 和 physical HBM charge 维护 virtual runtime，用于
   防饿死、等待上界和同等物理收益下的 tie-break；
2. **workflow 内部**：按 causal frontier、next-consumer progress、reveal value、循环终止机会和
   exact startup cost 选择 invocation。

不设置“每个 workflow 每轮最多一个 request”、固定 `1/N` HBM 份额或固定 round-robin。只要
多个 request 均已 READY、物理上可同时 admission，且加入它们能够提高 prefill batch fill、
action throughput 或 GPU 利用率，同一 workflow 可以在一个 scheduler epoch 获得多个 ticket。
公平只通过最大等待时间、starvation guard 和软 virtual-runtime penalty 约束；除 liveness 上界
外，不得让低收益的 workflow 均衡要求覆盖明显更高吞吐的可行 batch。

### 7.2 与 SGLang batch 的关系

第一版只控制 waiting/admission order，不抢占正在运行的 decode batch，也不修改 CUDA
kernel。具体执行顺序：

1. scheduler safe point 应用上一轮 ACK；
2. 增量同步 Radix tree、allocator、lock 和 request 状态；
3. controller 生成带 scoped generation 的 `JointPlan`；P4 reactive fallback 直接使用 observed
   state，P5 使用主动 JointPlan；
4. runtime 将 execution/admission intent 编译为当前 epoch 的 ticket，不移动 waiting request；
5. SGLang prefill loop 跳过无 ticket 的 tagged request 并继续扫描，由 `PrefillAdder` 最终决定
   哪些物理可行 request 进入 batch；
6. admission 只有在依赖的 restore ACK 或 recompute plan 满足后才兑现；
7. residency intent 经 RadixArbiter 再解析为 page action；
8. plan 的局部 component 在 batch selection 前变 stale 时只丢弃对应 ticket/action，并生成
   有界 `OBSERVED_JOINT` fallback，不使无关 request 或 bundle 失效。

### 7.3 JointPlan 生成算法

每次 runtime、allocator、ACK 或 waiting-queue 变化触发一次有界规划：

1. **建立 factual runnable frontier**：只收集 RCCG 中 `READY`、pending-message 或明确
   reactivation 的 invocation；`WAIT_TOOL/WAIT_CHILD/WAIT_JOIN` 只贡献 residency obligation；
2. **建立有界候选窗口**：按 request readiness、causal progress、启动成本和当前资源可行性
   收集候选；root-workflow virtual runtime、已获 GPU service 和 HBM charge 只提供软 penalty
   与 starvation guard，不限制单个 workflow 的候选数；
3. **构造 ExecutionPackage**：对每个 READY invocation 计算：

   ```text
   exact startup/restore bundles
   next-yield 前 token/KV demand
   可解除的 waiter、join、message consumer 或 loop termination
   startup budget 和 restore/recompute alternatives
   ```

4. **联合枚举**：对每个候选 execution order，用 What-if Packer 同时选择 admission 和
   residency actions；不能先固定 agent，再从剩余空间找 KV victim；
5. **场景聚合**：P4/P5 只计算 observed scenario；P6 对 top-K scenario 做 consensus、robust
   或 reveal 判定；
6. **字典序选择**：先满足 correctness、依赖和 liveness 硬约束，再最大化可行 batch 的 causal
   progress、action throughput 和资源利用率，最后最小化 unhidden stall、迁移/重算字节、
   HBM-time 与软公平 penalty；
7. **生成依赖**：任何需要 H2D 的 request 都产生 `TransferDependency(require_ack=True)`；允许
   recompute 时把它作为明确 alternative，而不是隐式欠债；
8. **版本化提交**：JointPlan 带 graph、topology 和 touched request/bundle 的 scoped generation；
   全局 allocator generation 只用于审计，局部依赖变化仅使相关 ticket/action stale。

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

### 7.4 BeliefKV 内部策略快照与回放接口

P2.5 已提供只读、可 replay 的统一策略快照。它只服务于当前 reactive policy、BeliefKV
JointPlan、O0-O3 和离线物理分析，不是竞品扩展点，也不是另一套在线执行路径：

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

当前内部策略：

| Policy | 输入 | 主要动作 | 当前状态 |
| --- | --- | --- | --- |
| `ReactivePolicy` | 当前物理状态 | LRU/reactive restore | 默认 correctness baseline |
| `BeliefJointPolicy` | `G_t/H_t/R_t` | 完整 JointPlan | BeliefKV |

P2.5-P6 只记录 reactive counterfactual 和 BeliefKV 所需快照。不得继续向该接口添加竞品专用
metadata、状态机或动作，也不增加动态 policy registry/plugin discovery。P8 若决定复现某个
外部方法，应在核心系统冻结后使用独立 runner 或最小适配层；是否复用该快照届时决定，不能
反向扩张 JointPlanner 的在线路径。

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
| `beliefkv/predictor/frontier_belief.py` | P6 closure-complete scope、全局联合场景、OTHER、有限 horizon 和 evidence read set |
| `beliefkv/predictor/composer.py` | 保留边际预测 API；为 scenario builder 提供组成事件，不直接输出迁移动作 |
| `beliefkv/runtime/bundles.py` | 生成 closure-aware physical bundle snapshot |
| `beliefkv/policy/resource_snapshot.py` | 聚合 HBM、Host、PCIe、service curve 和 reservation |
| `beliefkv/policy/reference/snapshot_builder.py` | 在 scheduler safe point 构造统一、可版本化的 `PolicyInput` 和不重叠物理 extent |
| `beliefkv/policy/scenario_physicalizer.py` | 将场景映射为未来物理需求 |
| `beliefkv/policy/whatif_packer.py` | 生成每个 scenario 的无副作用可行 plan |
| `beliefkv/policy/joint_scheduler.py` | observed/robust/reveal 模式，联合 Execution/Admission/Residency 和依赖 |
| `beliefkv/policy/predictive_joint.py` | P6 离线 A0/PREPARE/PREFETCH risk selection、Benefit/CVaR 和硬约束结果 |
| `beliefkv/policy/online_joint.py` | ActionGroup、事务 DAG、完整 resource certificate 和 safe-point group validation |
| `beliefkv/policy/reference/base.py` | 历史命名；定义 BeliefKV 内部 PolicyInput/PolicyOutput 和 shadow/replay API，不再扩展竞品 adapter |
| `beliefkv/policy/reference/reactive.py` | observed-state reactive correctness/replay baseline |
| `beliefkv/policy/admission.py` | 从独立选 workflow 改为校验并兑现 AdmissionIntent |
| `beliefkv/policy/transfer_planner.py` | 从独立选 victim 改为编译 ResidencyIntent；旧逻辑仅保留为实验 baseline |
| `beliefkv/policy/transfer_guard.py` | 新增失败 attempt ledger、typed blocker 和 event-gated retry eligibility |
| `beliefkv/policy/shadow_controller.py` | 保留 baseline；full policy 的 PREPARE 由 joint scheduler 触发 |
| `beliefkv/runtime/radix_arbiter.py` | 增加只读 preview/bundle closure API；执行时继续二次校验 |
| `beliefkv/runtime/protocol.py` | 增加 direction/page/pinned/native-traffic/allocator/callback telemetry，不改变 ACK 正确性语义 |
| `beliefkv/runtime/sglang_v052rc1.py` | 只应用 ExecutionIntent 顺序和 AdmissionIntent 依赖；采集 capability/telemetry；审计 stale/fallback |
| `beliefkv/runtime/sglang_adapter.py` | 增加 capability 和 telemetry backend protocol |
| `beliefkv/runtime/action_frontier.py` | 增量记录 structured action parser、合法 action boundary、frontier delta 和训练前 coverage |
| `beliefkv/runtime/audit.py` | schema 升级并记录 scenario/lease/joint-plan 事件 |
| `beliefkv/runtime/restore_obligation.py` | P5E obligation/lease/grace 与供 P6 read set 使用的独立单调 revision snapshot |
| `beliefkv/experiments/policy_replay.py` | 只运行内部 reactive policy 和 O0-O3；不承担外部方法复现 |
| `beliefkv/metrics/policy_snapshot.py` | 统计快照覆盖、safe-point 开销、writer 完整性和压缩存储成本 |
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
`reference_policy_shadow_enabled` 是现有实现的历史配置名，只控制内部 reactive
counterfactual；不得据此注册外部策略。

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

`reference_policy_decision` 和 `reference_policy_decisions.jsonl` 同样是历史 schema 名称，P2-P6
只写内部 reactive counterfactual。为避免破坏现有实验解析器暂不改名，但不再把它们解释为
通用竞品对比框架。

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

实施状态（2026-07-21 修订）：BeliefKV 显式 command 的 telemetry、ACK 和 timeline 已完成，
但全系统 HiCache operation coverage 重新打开。P3 mixed run 证明 SGLang request admission 可通过
`init_load_back()` 发起无 BeliefKV command ID 的 native H2D；当前 telemetry 不包含这类 demand
load。因此 P1 只能标记为 explicit-command telemetry 完成，不能继续声称所有 HiCache DMA
均可观测。真实验证与统计限制见
[P1 HiCache 真实迁移、服务曲线与控制开销验证](experiments/beliefkv_p1_real_hicache_validation_2026-07-18_zh.md)
和 [P3 动态并发 GPU Characterization](experiments/beliefkv_p3_dynamic_gpu_validation_2026-07-21_zh.md)。

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

可靠性修复状态（2026-07-21）：已确认 281 次小 H2D reject 的直接原因是固定 HiCache 的
10-token `load_back_threshold`，而非真实 allocator capacity；满载 fatal 同时存在 1,505-token
live Radix/free-list overlap。当前实现已加入 forced/no-eviction H2D、authoritative allocator
delta 校验、H2D ACK 前 admission barrier 与 prefix rematch、allocator/Radix resync、逐 callback
故障隔离，以及 control-socket failure 与 workflow failure 解耦。修复版随后在同一模型、
`163840`-token KV pool、8 个 planned SWE-bench workflow、并发 8 下完成真实 GPU 复跑：
8/8 workflow 和 848/848 request 都获得系统终态，1502/1502 transfer 有 ACK，scheduler
exception、watchdog、unknown blocker、identical failed retry、HBM mirror 超过 allocator 和 Host
账本偏差均为 0；52,851,376,128 bytes offload 的 reclaim realization 为 100%。旧 run 的
`device_capacity` reject 降为 0，剩余 79 次 `engine_busy` 和 8 次 `extent_mutated` 均在 DMA 前
fail closed，且没有重复失败 fingerprint。P2 真实可靠性 gate 因此通过；但只有 3/8 workflow
通过任务 correctness gate，service-curve holdout 低估率为 13.48%，controller timing summary
缺失，且尚未运行同代码/manifest 的 P1.5 配对基线。因此 P2 配对性能 gate 仍未通过，P4
只能继续 shadow，不能主动应用 JointPlan。详见 P2 实验文档第 9 节。

2026-07-21 的 P3 mixed run 对该结论增加了边界：上述可靠性 gate 只覆盖 BeliefKV 显式
command。SGLang request admission 自己发起的 native `init_load_back()` 没有 command ID，也
没有进入现有 transfer telemetry。P2 的 bundle/ACK correctness 仍成立，但“全系统 callback
coverage”重新标为未闭合；补齐并去重 native operation telemetry 后才能恢复该表述。

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

### P2.5：BeliefKV 内部决策快照与追踪 Contract

该阶段不部署或适配外部系统，也不阻塞 P2 真实 GPU 可靠性修复。目标仅是固定 BeliefKV
自身 JointPlan、reactive counterfactual 和 O0-O3 离线分析需要的不可变输入、动作与日志格式。

实施状态（2026-07-22）：已新增不可变 `PolicyInput/PolicyOutput` schema、统一 execution/
admission/residency/dependency primitive、request/program/context ID 映射和 runtime capability
report。现有 `ReferencePolicyAdapter` 是历史类名，只运行内部 reactive policy；调用前按声明
裁剪 metadata，online 模式无法接触 hindsight 字段，oracle metadata 只允许 O0-O3 replay；
所有输出保持 shadow-only。unsupported metadata/action 不做隐式转换，而是进入结构化结果。
reactive adapter 的确定性、统一物理
计费、schema round-trip、审计重建和篡改检测均通过。P3B 已在该契约上增加 physical
`extent_ids`、actionability/blocker 和 parent-child extent identity，禁止重叠 closure 双重
计费，并使 reactive policy 对不可执行 restore fail closed。`policy_state_updates` 仅作为历史决策记录的
序列化兼容字段保留；状态注入和跨 snapshot 状态回放逻辑已删除。

实现文件：

- 新增 `policy/reference/base.py`；
- 新增 `policy/reference/snapshot_builder.py` 和 `policy/resource_snapshot.py`；
- 新增 `tests/test_reference_policy_contract.py`；
- reactive policy 与 `experiments/policy_replay.py` 保持主线可用；外部策略草图与专用
  hindsight enricher 已删除。

任务：

1. 所有 BeliefKV 内部策略读取相同的 `PolicyInput`，输出可审计的 `PolicyOutput`；
2. 每个输出声明使用了 observed、predicted 还是内部 oracle metadata；
3. contract 明确 online/internal-oracle metadata mode 和 hindsight 隔离；
4. reactive counterfactual 默认 shadow-only，不触发真实 admission、D2H 或 H2D；
5. 保留 BeliefKV 审计需要的 request/program/context ID 映射和 capability report；不得因潜在
   竞品需要而扩展在线调度分支或合成应用先验；
6. 冻结为静态 internal adapter，不增加通用 registry、动态加载和 per-competitor state。

退出条件：

- reactive adapter 对相同输入生成确定性、schema-valid 的 PolicyOutput；
- hindsight 字段无法泄漏到在线 BeliefKV predictor；
- 同一物理 snapshot 下所有策略使用相同 HBM/PCIe accounting；
- unsupported action/metadata 被明确记录，不静默转换成 BeliefKV 特有动作；
- contract output 可被 audit validator 重建。
- replay 只运行 reactive policy 和 O0-O3；当前代码不存在竞品 adapter 或竞品专用 metadata
  producer。

### P3：动态 Workload、Joint Oracle 与 Agent 调度基线

P3 的第一目标是建立动态 agent 调度实验面、action-unlock 观测面和 joint oracle，不先训练
frontier predictor，也不运行竞品 reference/native 系统。该阶段可以与 P2 可靠性修复并行开发，
但 P2 gate 未通过时只能做 replay/shadow，不主动控制真实 HiCache。

实施状态（2026-07-22）：P3A 的 CPU/runtime instrumentation、scheduler-safe-point 统一
`PolicyInput` snapshot、gzip trace 和 P3B 的离线决策层已实现。
queue/service resimulator、token-exact tiered Radix、rolling allocator、arm-aware evaluator 和
完整/有界 topological-order search 已接入。单-workflow 1,000-token HBM 负例的 joint synergy
gap 仍为 0。

2026-07-22 workload 审计确认：先前 12-workflow mixed run 的 48 个 child 是 one-shot，且
repository tool call 为 0，因此只能保留为 topology/pressure characterization。新版
agentic backend 已实现持久 peer RESUME、模型运行时 task fan-out、FRESH child 多轮工具循环、
最终轮禁用 task 和独立 workload validity gate；另增加 initial coder 的 `2..4`
required-range characterization mode。单次 240 秒、4-workflow 探针实际产生 16 个 child，
瞬时达到 16 个 running request，但按时间积分后 `running <= 2` 占 68.13%，`running >= 8`
仅占 24.36%，稳定 GPU-ready 并发 gate 仍未通过。详见
[P3 真实工具型 Agentic Workload](experiments/beliefkv_p3_agentic_workload_2026-07-22_zh.md)。

正式 163,840-token、12-workflow 全 mixed GPU characterization 已完成：12/12 semantic
completion，模型创建 48 个 FRESH child，并产生 40 次 HANDOFF、17 次 REACTIVATE；峰值 HBM
96.93%。BeliefKV 显式路径完成 14 次 D2H 和 8 次 H2D，无 partial/reject/retry storm。该 run
同时发现两项硬阻塞：一笔 join 后错误恢复占显式 H2D 的 37.52%；三个 parent 通过 SGLang
native demand-load 恢复，但没有 BeliefKV telemetry。1,138 个 runtime GPU batch 还证明旧
batch-1/2/4 短 context service model 无法外推，overlap completion interval 也不能直接作为
独立 batch service time。因此 P3 workload gate 有进展，但 physical/service oracle gate 与
全量 transfer accounting 仍未通过。详见
[P3 统一快照与 Reference Replay Smoke](experiments/beliefkv_p3_shadow_smoke_2026-07-21_zh.md)
、[P3 Queue/Service Resimulator 标定](experiments/beliefkv_p3_queue_service_calibration_2026-07-21_zh.md)
、[P3 Rolling Physical Replay 检查点](experiments/beliefkv_p3_rolling_oracle_2026-07-21_zh.md)
和 [P3 动态并发 GPU Characterization](experiments/beliefkv_p3_dynamic_gpu_validation_2026-07-21_zh.md)。

实现文件：

- 新增 `control/data_consumers.py`；
- 新增 `policy/scenario_physicalizer.py`、`policy/whatif_packer.py`；
- 新增 `policy/joint_oracle.py`；
- 新增 `policy/reference/snapshot_builder.py`、`policy/resource_snapshot.py`；
- 新增 `experiments/policy_replay.py`；
- 新增 `experiments/counterfactual_trace.py`、`experiments/service_calibration.py`；
- 新增 `simulator/queue_service.py`、`simulator/token_radix.py`、
  `simulator/rolling_physical.py` 和 `simulator/rolling_queue_service.py`；
- 新增 `metrics/policy_snapshot.py`；
- 新增 `runtime/action_frontier.py`；
- 新增 `experiments/langgraph_peer_workflow.py` 和 `scripts/run_rolling_joint_oracle.py`；
- 新增 `experiments/agentic_peer_backend.py`，并扩展
  `scripts/run_langgraph_peer_workloads.py` 为真实 tool/sandbox runner；
- 扩展 RuntimeEvent adapter，可靠捕获 `HANDOFF/MESSAGE/REACTIVATE/CANCEL`；
- 扩展 simulator/replay 和 audit validator；
- 新增 `tests/test_data_consumers.py`、`tests/test_whatif_packer.py`、
  `tests/test_multi_agent_runtime.py`、`tests/test_action_frontier.py` 和
  `tests/test_reference_policies.py`。

#### P3A：动态 workload 与 instrumentation

实现状态：已加入 observed producer-consumer index、单调 graph version、`REACTIVATE` 事件、
基于真实 PageHandle 的 context prefix affinity、增量 structured-action observer，以及动态
Coder/Reviewer/Tester peer loop 内嵌 FRESH subagent 的 LangGraph workload。CPU fake backend
能够重复完成 spawn/join/handoff/reactivation 并生成 topology/consumer/action characterization；
Deep Agents 只能报告 runtime 已解析的 action，当前不能伪造原生 incremental boundary token。
早期全 mixed run 正常完成 100 个 LLM call、48 个 one-shot child，并覆盖 join 后 peer
handoff 和 cyclic reactivation；审计后它只保留为 topology/pressure smoke。新版真实工具
backend 的单 workflow gate 已覆盖动态 task、child 内 11--17 次 LLM、真实工具等待、join、
handoff 和 peer RESUME，但仍缺同 manifest 并发 A/B、fanout coverage、nested/nonblocking
subagent 和 incremental boundary-token coverage。required-range probe 只证明 fan-out
机制和峰值负载，不证明稳定 ready 并发；P3A 下一 gate 是固定 fan-out 下的 phase-aware child
release 配对，而不是继续增加 root 数量。

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

#### P3B：Reactive baseline、joint oracle 与物理有效性

实现状态：reactive baseline、ScenarioPhysicalizer、What-if Packer、O0-O3 JointPlan oracle、
rolling Radix/allocator/queue-service evaluator 已实现。`policy_replay.py` 原子写出 reactive
决策、输入指纹和 unsupported 项，并禁止从冻结 physical trace 直接报告反事实 JCT。外部
方法的 same-data-plane 草图和 hindsight enricher 已删除；P3-P6 不再重建或维护这些近似实现。

Scenario physicalizer/What-if Packer 已对 blocking、nonblocking、FRESH、handoff、multi-consumer
和 cyclic reactivation 执行 closure/capacity/fairness/hysteresis 检查，缺少 extent identity 或
closure 重叠时 fail closed。rolling evaluator 能按候选顺序重算 token-prefix、锁、allocator、
tier residency 和 transfer。单 workflow 机制跑的 synergy gap 为 0；多 workflow fairness、真实
dynamic A/B、page/node extent 对齐和 PCIe holdout 尚未闭合。

任务：

1. 使用 current reactive scheduling + current bundle policy 建立统一的无预测 correctness
   与性能基线；
2. 在真实 trace 中筛选同时具有多个 READY agent、HBM/PCIe pressure 和可选 physical action 的
   决策快照，执行局部 Jointness Audit；

   ```text
   S: 只改变当前 agent execution order
   K: 只改变当前 KV/admission action
   J: 在同一短窗口联合改变 execution、admission 和 KV action

   local synergy gap = min(cost(S), cost(K)) - cost(J)
   ```

3. 局部窗口只推进到下一个 TOOL_START、WAIT、HANDOFF、RETURN 或 JOIN 等已观测因果边界；固定
   该窗口内的 token demand、tool duration 和 action outcome，候选动作会改变语义分支时标记
   `counterfactual_unidentifiable`，不强行报告反事实收益；
4. What-if packer 对 blocking、nonblocking、handoff、multi-consumer 和 cyclic reactivation
   都生成物理可行 plan；
5. 已实现的完整 O0-O3 rolling oracle 保留为系统冻结后的可选深入分析，不再作为 P5A--P5C
   端到端系统搭建和 correctness gate 的前置条件；
6. 系统完成后的真实端到端比较使用固定任务、模型参数和随机性并重复运行；无法冻结动态路径时
   同时报告 transition hash、调用量/拓扑分布和统计置信区间；
7. 保留复现 BeliefKV 决策所需的中立 runtime/physical 字段，但 P3 不生成 invocation
   distance、perfect phase 或 true-future duration 等竞品专用 metadata；
8. 保持竞品专用 adapter/metadata producer 不进入 P3 代码，replay 只运行 reactive policy 和
   O0-O3。

研究证据条件（不作为 P5A--P5C 系统实现前置条件）：

- 至少一个 blocking subagent、一个 cyclic peer multi-agent 和一个 mixed workflow 可重复运行；
- FRESH child 不继承 parent pages，causal/data/prefix 三种边可分别重建；
- structured action coverage、boundary-token 分布和 tool-start gap 被报告；
- 真实 pressure snapshot 能生成可审计的 `S/K/J` 局部决策与不可识别标签；
- What-if packer 无副作用且满足 closure/capacity/fairness/liveness；
- execution-order reversal、KV-action reversal 和 local synergy gap 能按 blocking/peer/mixed
  关系归因；该结果是论文 JointPlan 主张门槛，不阻塞 P5A--P5C 的系统实现；
- 证据采集不依赖竞品 adapter 或任何原生竞品结果；
- 当前 workload topology entropy、cycle、handoff 和 consumer fan-out 被正式报告。

### P4：Observed-State JointPlan Shadow Mode

前置状态（2026-07-21）：统一快照和 shadow trace 已具备，但 P3 workload/oracle gate 与 P2
配对性能 gate 均未通过；当时尚未实现或启用 `joint_scheduler.py`。修复前真实 GPU 同步快照
P99=38.08 ms，明确不达标；异步 writer 后 CPU 快路径在 256 extents 无变化时 P99=0.283 ms，
114 extents 的 warm lease 切换 P99=0.891 ms，但单 extent engine-lock 改变仍为 1.929 ms。
这些 CPU 数字只用于定位开销，不替代真实 scheduler safe-point 复验，也不能提前宣称 P4
overhead gate 通过。

实施状态（2026-07-23）：已新增纯函数 `ObservedJointPlanner` 和版本化 `JointPlan`，能够从
同一个 observed snapshot 联合生成 execution、admission、residency 与 restore-ACK dependency；
已覆盖 join straggler、root-workflow 公平、restore/victim 联动、transition-open settling
barrier、nonblocking parent 和 stale physical version。策略快照同时冻结 attained service、
virtual runtime 与 root-workflow HBM charge，并修复 reserved request startup 重复计费。

runtime 技术路线已经固定为：

1. scheduler safe point 在应用 event、ACK 并同步 RCCG/Radix/allocator 后，只发布 immutable
   RCCG event、PageIndex replacement record、queue/resource 最新值和 component version；完整
   `PolicyInput` 不再在 scheduler thread 构造；
2. 容量为 1 的 latest-wins worker 持有独立 RCCG、consumer 和 PageIndex mirror，按顺序无损合并
   pending delta，在 worker 内构造 `PolicyInput` 并运行 `ObservedJointPlanner`；较新的 submission
   可以替换 pending sequence，但不能丢弃被替换 sequence 携带的 event/page delta；
3. 下一 safe point 使用 `PlanReadSet` 和 source/current component stamp 做 optimistic validation；
   `snapshot_id` 只用于 lineage，graph/topology/allocator 全局版本比较只作为
   `strict_global_stale` 对照指标；
4. source snapshot 内的 validation 同时检查 request/frontier identity、context epoch、局部
   invocation/join、transition generation、touched bundle generation/actionability、当前 workflow
   fairness priority、plan expiry 和 HBM/Host feasibility certificate；source snapshot 之后在
   scheduler safe point 按 execution、每条 admission、每条 residency 和每条 transfer dependency
   分域重验。全局 graph/topology/allocator stamp 只保留为对照诊断，单个 request 或 extent 改变
   不再使无关动作 stale；decode 推进导致的 startup bytes 减少、无关 transfer epoch 前进或不改变
   实际优先级的 fairness revision 前进不会制造假 stale；
5. fresh plan 只记录 `joint_plan_would_apply`，stale、过期或 worker failure 只记录原因，不同步
   重跑完整 planner，也不改变 waiting queue、admission、D2H/H2D 或 physical residency；
6. runtime 分别记录 safe-point delta capture、worker snapshot build、trace enqueue、queue wait、plan
   compute、publish-to-safe-point、validation、plan age、pending coalescing/drop、strict/scoped stale 和
   would-apply coverage；
7. `joint_policy_enabled=true` 在 P4 fail closed，防止尚未实现的 P5 执行路径被误认为已启用。

snapshot 提交采用事件触发加 watchdog，而不是每个 decode quantum 无条件全量重建：queue、graph、
transition 和 transfer 变化立即触发；连续物理增长与 attained-service 推进按
`reference_policy_snapshot_min_interval_ms` 周期刷新。变化探测使用 page-index revision，不在探测
阶段重复枚举 Radix extent；真实 snapshot build 开销仍必须由 P4 GPU gate 测量。

此前 P4 runtime shadow 在线路径不能继续沿用：tagged request 在进入 SGLang waiting queue 前被
长期隐藏、每个 controller tick 只放回一个 request、`WAIT_H2D` admission 持有 reservation，以及
scheduler safe point 同步构造大 snapshot，会共同造成 KV lock/admission convoy。2026-07-23 已完成
该数据路径的 CPU 机制修复，但尚未通过真实 GPU overhead/stale/coverage gate：

> **Visible-but-Gated Incremental Admission**：SGLang 始终拥有完整 waiting queue；BeliefKV
> 使用增量状态在当前 batch-construction epoch 编译短期 admission ticket，并在 SGLang 原生
> `PrefillAdder` 前做 gate。ticket 只表示执行资格，不提前占用 HBM；SGLang allocator 和
> `PrefillAdder` 保留最终容量与 batch 决定权。

该改进直接替换 `_deferred_requests -> decide_next() -> 单请求放回` 路径，不实现临时
`decide_batch()`，也不维护第二套长期请求队列。它属于 reactive admission 数据路径修复，
不等于提前启用 P5 的策略性 ResidencyIntent；P5 只需把 ticket 的决策来源从 observed/reactive
状态切换为主动 JointPlan，不再更换队列和提交协议。

实现文件：

- 新增 `policy/joint_scheduler.py`；
- 新增 `runtime/joint_shadow.py`；
- 修改 `control/controller.py`、`policy/admission.py`、`runtime/audit.py`；
- 修改 `runtime/sglang_v052rc1.py` 和 `patches/sglang-0.5.2rc1-beliefkv.patch`；
- 扩展 page/context dirty journal 和增量 resource view；
- 新增 `tests/test_joint_scheduler.py`、`tests/test_joint_shadow_worker.py`；
- 扩展 `tests/test_controller.py`、`tests/test_sglang_adapter.py`。

任务：

1. 让所有 request 立即进入 SGLang 原生 waiting queue；BeliefKV 只维护按 request ID 索引的
   `VISIBLE_PENDING / WAIT_RESTORE / POLICY_BLOCKED` side state，删除长期 hidden queue 所有权；
2. 在 page、owner、lock、residency、allocator、RCCG 和 queue 的既有 mutation point 更新 dirty
   journal、增量计数器和 scoped generation；scheduler safe point 不再扫描并排序全部 page owner；
3. 每个 batch-construction epoch 用当前 waiting request、causal frontier、restore dependency、
   prefill token budget 和 authoritative HBM view 一次生成多个短期 ticket；不设置 per-workflow
   ticket 上限，同一 workflow 的多个可行 request 可以同时进入候选 batch；
4. 在 SGLang 原生 prefill loop 中，对无 ticket、`WAIT_RESTORE` 或 blocker 未解除的 request 执行
   policy `continue`，继续扫描后续 request；实际加入 `can_run_list` 时才提交服务记账，未使用
   ticket 在 epoch 结束后自动失效；
5. ticket 校验 request/context/prompt/prefix 和相关 bundle 的局部 generation，不以全局
   allocator generation 相等为条件；`PrefillAdder` 重新匹配 prefix 并完成最终 allocation，
   BeliefKV 的 startup bytes 只用于有界 packing；
6. 普通 `WAIT_RESTORE` 不持有 admission reservation。只有最老且物理可行的 restore debt 在
   H2D commit 前获得至多一份 allocator-backed `RestoreLease`：预留容量跨 H2D ACK、
   authoritative prefix rematch、ticket 和 native admission 保持，直到目标 request 首次重新
   获得 GPU service 才释放；随后 request 还必须完成配置的 decode service quantum，或正常结束
   当前 LLM call，才可再次成为 running-retraction victim。funding ACK 释放的 allocator capacity
   先转成 debt-owned escrow，再原子转换为 lease 与 H2D headroom；该容量不能被普通 admission 或
   active KV growth 消费。partial/rejected
   ACK、request cancel、cache reset 和 shutdown 必须原子 rollback，不能泄漏 allocator token；
7. snapshot 发布和 JointPlan validation 改为 dependency-scoped、component-wise 提交：request
   消失只丢对应 admission item，bundle 改变只丢对应 residency/transfer item，fairness 变化只
   重编译当前 ticket 顺序，不使整个 plan stale；
8. planner 超时、无 fresh shadow plan 或 snapshot 暂不可用时，使用同步且有界的 observed-state
   reactive ticket fallback；显式 restore/terminal/transition-open blocker 仍 fail closed；
9. 保留 prediction 关闭、JointPlan residency shadow-only 和 atomic transition barrier；记录
   ticket issued/selected/expired/native-rejected、policy skip、batch fill、running concurrency、
   admission wait、snapshot overhead/stale 和 H2D blocking 指标。

当前完成度：任务 1--9 的 CPU 机制已完成。任务 7 使用 `PlanReadSet` 定点读取当前 request、
invocation、join、transition 与 touched extent，逐 action 输出 valid/reasons；residency 校验会对
目标 extent 重算 owner lease、lock/reader/pin/in-flight 状态和 GPU descendant closure，并滚动
重验 Host/HBM headroom。无法解析的 extent、缺失记录、generation 变化和容量不足均 fail closed；
局部 fresh component 只在 shadow audit 中标记 `joint_plan_shadow_partial`，不宣称可独立提交且
尚未执行；P5 必须在提交前再闭合 execution/admission/residency dependency。2026-07-23 首次
24-workflow GPU gate 中，24 个初始 prompt 均因完整 prompt 大于 4096-token epoch chunk budget
而被错误标记为 `prefill_token_budget`，93,883 个 epoch 未发出 ticket。修复后 ticket 仅消耗
本 epoch 的首块 token，HBM feasibility 仍按完整请求计算。独立复验记录 1,418 次 LLM submit、
1,351 次工具调用、61 次 spawn，KV usage 峰值 96%，完成 13.34 GB D2H 和 3.49 GB H2D，且
未复现 retry storm；全量 CPU 回归为 `347 passed, 7 skipped`。但 12,922 个 JointPlan 因
`max_joint_plan_budget_ms=1` 触发 `planning_budget_exceeded`；planner 自身 P50/P95 为
13.9/33.0 ms，worker 端到端 compute（含 snapshot build）为 157.7/290.8 ms。运行约 95.6
分钟后仅有 3 个 workflow 以 API timeout 自然终止。P4 因规划开销和单 activation 终止性仍
未通过，详见本阶段实验文档。

退出条件：

- scheduler safe point 的增量 snapshot、ticket compile 和 validation 总 P99 小于 1ms，或小于
  scheduler step 的 5%；
- 计划在执行前 stale 的比例低于 10%；
- physical plan rejection 不高于当前 reactive planner；
- SGLang waiting queue 成为唯一请求所有者，不再存在长期 `_deferred_requests`；
- 普通 admission ticket 不产生跨 epoch HBM reservation；只有 bounded restore liveness lease
  可以跨 epoch 持有真实 allocator 容量，active lease 默认上限为 1；
- restore lease 可用时无关 request 只能消费 lease/escrow 之外的剩余容量；最老 debt 超过
  escalation threshold 后，其 barrier 在 active lease 期间仍保持。lease 暂不可建立且 GPU idle
  时最多允许一次最小 working-set bypass，禁止 debt barrier 形成无限空转或无限旁路；
- 同一 workflow 的多个可行 request 能进入同一 prefill batch，不因公平规则被强制串行化；
- 使用现有 P4 stress workload 直接复验 correctness、吞吐、batch fill、running concurrency、JCT
  和 admission tail；当前阶段不增加四路 admission 消融矩阵；
- nonblocking/background parent 从不因 `SPAWN` 被错误标为 parked/offload；
- workflow service 按实际进入 batch 的 GPU 服务记账，fan-out 不获得虚构服务，也不被固定份额限制；
- reactive counterfactual 与其 observed-state 输入可由 audit 完整重建；
- action observer 对未支持的自由文本输出明确返回 UNKNOWN，不伪造 action boundary；
- 在 prediction 关闭时即可重建所有 JointPlan 决策和 fallback；
- 将 snapshot build、planner search 和 safe-point validation 分开预算；24-workflow 下不能以
  放宽 TTL 或把 1 ms 直接改成数百毫秒替代开销优化；
- 单次 agent activation 必须有语义完成及墙钟/调用预算，且正常 RETURN/JOIN workload 能在
  gate 内形成 workflow 终态；
- runtime audit 和 policy snapshot 改为有界采样/delta；当前 95.6 分钟产生 1.8 GiB audit 和
  603 MiB 压缩快照，不能用于性能结论；
- native demand-load/write-back telemetry 已接入统一 transfer timeline；Host 生命周期已实现
  `DUAL_CLEAN -> GPU_ONLY`、terminal-private cleanup 以及容量饱和时的淘汰/drop-recompute
  路径。上述机制均需在下一次真实 GPU run 中验证覆盖与字节一致性；
- 支持 atomic transition 的 adapter 不出现 join-to-handoff transient READY prefetch；不支持
  atomic transition 的 adapter 必须通过可审计 settling barrier 回退。

层次边界（2026-07-23）：固定 SGLang 0.5.2rc1 不提供理解 agent message/tool 语义的自动上下文
总结或压缩。模型 sliding-window attention、请求截断和 KV eviction 都不能替代语义压缩。因此
BeliefKV 不实现 tool-output summarization 或 context checkpoint；这两项属于 LangGraph/Deep Agents
业务层。serving 层只保留 context-growth admission、run-to-yield/terminal scheduling 和物理 KV
迁移，避免把质量改变混入 KV 管理贡献。

当前 context-growth admission 已在 visible-ticket 路径实现：每个 waiting request 按未命中 prompt
token 与尚未生成的最大 output token 计算增量 KV，ticket 编译再用 authoritative HBM headroom
检查；后续 agent turn 以新的 prompt/prefix match 重新估算。该机制只控制物理容量，不修改消息
内容。run-to-yield/terminal scheduling 需要改变 running decode batch，属于 P5 在线联合调度，不在
P4 shadow 修复中提前启用。

P4 开销修复将确定性的 snapshot/index preparation、可中断 package search 和 plan/read-set
materialization 分开计时。`max_joint_plan_budget_ms` 约束 anytime package search；完整 planner wall
time 仍单独报告，不能用提高 TTL 或隐藏 preparation 代替优化。4096-extent CPU 定向基准中，单
lock snapshot P50 为约 2.6 ms，单 residency snapshot 为约 6.3 ms；JointPlan preparation P50
约 3.0 ms，而 physicalize+pack search P50 约 0.48 ms。该结果只用于定位开销，仍需真实 GPU
safe-point/stale gate。

2026-07-23 的第一轮 shadow 热路径修复已经完成，但尚未执行 GPU gate：PageIndex 和 RCCG
journal 改为从最新 revision 向后读取，不再每次扫描 65K/262K 历史窗口；worker mirror 用覆盖
revision 区间的单条 mutation 记录替代逐 revision 空记录，并以 changed extent 的局部一致性校验
替代每个 delta 后的全 PageIndex 扫描；纯 lock/residency/transfer 变化使用轻量 physical-state
patch，不再复制未变化的完整 Radix topology、owner 和 context handle list；worker 在后台将
pending delta 合并为最终 page/context replacement state，
但保留全部 RCCG event 和 transfer telemetry。4096 extents、64 contexts、每轮 384 个 lock extent
变化的 200 轮 CPU 定向基准中，delta capture P50/P95/P99 为 1.69/2.21/4.00 ms，mirror apply
P50/P95/P99 为 2.61/2.69/3.33 ms，复制 context 数为 0；capture 仍出现 62.2 ms 的单次离群值，
因此还未通过 P99 gate。该基准与旧 24-workflow 在线统计不完全同构，不能直接把差值宣称为
线上加速。

P4 worker 不再依赖完整 `PolicyInput` snapshot log 才能启动。实验配置生成器提供两级隔离：
`--disable-snapshot-persistence` 保留 JointPlan shadow、关闭后台完整快照 JSON 序列化；
`--disable-policy-shadow` 同时关闭 snapshot capture 和 JointPlan worker。后续必须用同一固定
manifest 做 shadow-off、shadow-on/no-persist 和 shadow-on/persist 三组开销归因；在完成该配对前，
不能把 GPU 利用率变化归因于 agent scheduling 或 JointPlan 策略收益。

2026-07-23 已完成进入 observed-state admission scheduling 前的第二项观测前置工作：新增
`beliefkv/runtime/lock_service.py`，以已完成的 GPU batch 而非 queue/running membership 作为
request 获得 service 的证据。每个资源采样点将 tagged running request 的
`last_node -> Radix root` 精确路径与 `engine_lock_ref` extent 关联，并报告 100/500 ms
`locked_but_not_served_gpu_bytes`。共享 prefix 只计算一次；只有所有 lock ref 都已归因且全部
blocker 超窗时才计为 stale，部分归因和路径错误保守归入 unknown，初始窗口内归入 warming。
因此该指标是“持锁但未获完成 service”的物理字节下界，不等价于可立即迁移量。

实现复用 PageIndex 同 revision 的 physical breakdown 缓存，避免资源采样后再次全树扫描；额外
复杂度为当前 engine-locked extent 建表及运行 request 祖先路径总长度。现有 transfer timeline
已加入 100/500 ms 曲线、归因覆盖率和 stale/engine-lock 比例。该变更完全只读，不修改 ticket、
waiting queue、running batch 或 residency。必须先在真实 GPU trace 上同时验证归因覆盖率、
locked-but-not-served HBM-time 与低 GPU 利用率的时间重叠，才可据此决定是否进入 observed-state
admission scheduling；仅凭 Engine-locked 总量不能证明 lock/admission convoy。

为避免此前 95.6 分钟产生 GiB 级审计，在线 worker 输入不采样，但完整 replay `PolicyInput`
默认每 10 秒持久化一次，writer queue 上限为 8；设置 persist interval 为 0 可执行专用全量
capture。重复 JointPlan 状态使用 compact audit，状态类别变化、错误和每秒周期样本保留 full
detail，最终 summary 记录 compact/full、sampled-out 和 writer-drop 计数。

### P5：启用 Observed-State Joint Agent/KV Scheduling

实现文件：

- 修改 `control/controller.py`、`policy/admission.py`、`policy/transfer_planner.py`；
- 修改 `runtime/sglang_v052rc1.py`；
- 扩展 `tests/test_sglang_adapter.py` 和真实压力脚本。

启用顺序：

1. 启用完整 JointPlan，但 ResidencyIntent 仅允许 KEEP/DROP_DEAD 和已有 reactive bundle；
2. 将 JointPlan 的 ExecutionIntent、AdmissionIntent 和 restore dependency 编译成 P4 已建立的
   batch-epoch ticket；AdmissionController 不再自行选择 workflow，也不建立长期 reservation；
3. SGLang 继续拥有 visible waiting queue，ExecutionIntent 只影响当前 epoch 的 ticket eligibility
   和顺序；一个 workflow 可以同时获得多个 ticket，fairness 只作为防饿死软约束；
4. 增加 transfer-to-transfer dependency DAG，只有 completed ACK 才能解锁后继 H2D、admission
   和 execution；partial/rejected 触发 prefix rematch、计划失效和保守 replan；
5. transfer planner 改为只编译 ResidencyIntent，不再自行选 victim/prefetch context；
6. 启用 blocking parent/tool-wait exclusive-suffix offload 和 READY restore dependency；
7. 启用 cyclic peer residency hysteresis 和 multi-consumer admission；
8. prediction、scenario reservation 和 reveal 保持关闭；
9. 在现有 blocking、peer 和 mixed 并发 stress workload 中直接验证完整在线路径；暂不扩展
   admission 策略消融矩阵；
10. 仅保留 reactive counterfactual logging；P5 不包含竞品 replay 或在线 adaptation。

实施状态（2026-07-25）：已完成 admission-only 的 P5A 在线切片，尚未完成完整 P5。新增
`ObservedAdmissionScheduler`，在当前 SGLang batch epoch 直接读取 visible request、observed causal
frontier、root-workflow fairness、PageIndex engine-lock/active-reader 字节和 running-private KV。
它不等待异步 shadow JointPlan，也不执行 physicalizer/packer：

```text
active KV footprint
  = unique Radix engine-locked/active-reader bytes
  + running/chunked private KV bytes

active KV budget
  = (KV pool - reserve) * configurable high-watermark ratio
```

ticket compiler 同时受 native HBM/token/request slot 和 active KV growth headroom 约束。running 数
低于可配置 floor 时只补足 floor，这是唯一高水位 bypass；starvation 只改变 observed candidate
顺序，不允许绕过 HBM。候选顺序使用 causal rank、unblock depth、workflow virtual runtime、
frontier round、等待时间和 incremental KV，不使用预测。无 metadata request 保持 SGLang 原生行为；
restore/terminal/transition blocker 继续 fail closed。observer/policy 异常时回退 reactive ticket，
并输出 `observed_admission_fallback`。

该切片通过 `observed_admission_scheduling_enabled` 显式启用，默认关闭；初始高水位 `0.8` 和 floor
`1` 仅用于后续 sweep。审计逐 epoch 输出 active budget/footprint/headroom、Radix lock、running
private、native/policy HBM budget、mode、issued/selected/expired/native-rejected 和 fallback。全量 CPU
回归为 `377 passed, 7 skipped`。P5A 尚未运行真实 GPU 配对，不得宣称性能收益。

严格边界：P5A 只能限制新 request 加入 active set，不能缩小已有 running batch，也不接管
residency writer。因此它不等于完整 JointPlan，任务 1、4--7 仍未闭合。若真实 trace 中 admission
gate 已降低 lock growth，但 active owner 仍长期占据 8--14 GiB 且未获 service，下一步才实现
running-batch retraction；若 gate 本身显著降低 GPU-ready concurrency，则先调整/否决 active-set
算法，而不是用 retraction 掩盖问题。

实施增量（2026-07-26）：P5B 加入默认关闭的 observed selective running-batch retraction；P5C
进一步闭合 `RETRACT -> physical closure rematch -> OFFLOAD/DROP -> ACK -> replacement admission`
跨 epoch 事务。P5B 使用完整 lock blocker set 选择运行中 victim，在固定 SGLang safe point 仅释放
request-private KV 和 Radix lock。P5C 不再把 `available + evictable` 当成已释放 HBM：只有 allocator
`available` 的真实增量达到 target，且绝对 free bytes 覆盖首个 replacement，才解除 admission barrier。

不足时，P5C 根据 retracted context 集合重新构建 closure-complete physical bundle，只对该事务内的
context 豁免逻辑 RUNNING owner lease；foreign active owner 和全部物理 blocker 保持 fail closed。Host
容量允许时显式 D2H/COMMIT_CPU；`DROP_CONTEXT` 可释放 dual-clean GPU copy，GPU-only recompute drop
默认关闭。每次只允许一个精确 command 在途，partial/rejected/stale/timeout 立即终止事务，不进入
retry storm。原生 OOM retraction 仍是最终 liveness fallback。全量 CPU 回归为 `389 passed, 7 skipped`；
真实 GPU gate 尚未完成，因此不能宣称性能、JCT 或 lock-convoy 收益，P5 的通用 ExecutionIntent、
cyclic peer hysteresis 和多 consumer admission 仍未闭合。

实施增量（2026-07-27）：完成只读 `TentativeUnlockPreview`。物理层以当前 PageOwnershipIndex 和
临时 lock-ref override 做无副作用 closure 投影；provenance 层只对完整 request blocker set 减少
假设 lock ref，部分/缺失归因 fail closed。runtime 在 barrier 前记录 observed-stale blocker 的乐观
上界，在安全点 plan 后记录 selected-set 的 projected newly-migratable bytes，并在 callback 记录
realized delta、误差、exactness 与 preview 开销。该机制尚不进入 barrier gate、victim selection 或
JointPlan objective，只用于验证“主动释放哪些 running owner 才会产生真实物理闭包”这一假设。
全量 CPU 回归为 `403 passed, 7 skipped`；下一次固定 GPU trace 才能判断 attribution coverage、
closure amplification 和 preview error 是否足以支持后续控制策略。

实施增量（2026-07-27，P5D）：新增 `policy/online_joint.py`，将通过 current-state component
validation 的 observed JointPlan 编译为当前 epoch ticket。running 与 waiting request 共同进入
runnable frontier；`RESTORE_THEN_ADMIT` 必须等待同一 plan 中 `PREFETCH_GPU` 的真实 ACK。有效 plan
期间由 JointPlan 成为 execution/admission/residency 唯一策略来源。该版随后已由 2026-07-28
修复替代：没有可用异步 plan 时也生成同一算法的 bounded/emergency epoch，不再开放 reactive
transfer 的在线 victim 选择权限。online residency 每次只允许一条 command
在途，ACK 后使旧 plan 失效并重新规划；running retraction 只能从同一 plan 明确 DEFER/PAUSE 的
running set 中选择完整物理 blocker closure，并与后续 residency ACK、replacement ticket 组成同一
跨 epoch transaction。已加入 H2D 后反向 eviction hysteresis、按 request 实际 Radix 路径维护的
restore dependency，以及一个 shared closure 服务多个 consumer 的 admission。bundle preview 审计
改为有界明细加聚合统计；同一 safe point 的 validation 结果可复用。

严格边界：2026-07-28 版本已输出显式语义 `RetractionIntent`，但仍不在异步阶段构造 blocker
closure；safe-point physical solver 只验证被指定 victim 的最小可行动作。完整 action-unlock 目标函数
仍属于 P5.5。无法映射 owner context 的 unowned physical extent 会 fail closed，仅允许 lifecycle
cleanup 或 SGLang 原生 OOM safety path，不回到独立 reactive victim planner。上述代码尚未通过
blocking、cyclic-peer、mixed 的真实 GPU gate，不得据此
宣称性能收益或关闭 P5。

实施增量（2026-07-28，P5D control-plane repair）：

- `AsyncSemanticJointPlanner` 取代在线 worker 中的 physical planner。异步阶段仅输出 execution、
  admission、context-level residency target 和显式 `RetractionIntent`，不调用
  `ScenarioPhysicalizer.prepare()`，不保存 page/extent generation/closure handle；
- scheduler safe point 对 request action slice 做最大有效依赖闭包提交，并将至多一个 semantic
  residency target 解析成当前 `PhysicalBundlePreview`。无关 workflow 或 bundle 变化不再整份否决；
- `BOUNDED_SEED/OPTIMIZED/EMERGENCY/NO_ACTION` 共享 `JointPlanEpoch`。P5 开启时硬关闭 reactive
  transfer 的在线策略权限；running retraction 的语义 victim 先由 JointPlan 指定，physical solver
  只验证该 victim 的 blocker closure 和真实可回收量；
- async planning 使用 latest-wins、trigger-interval 动态预算和预算前可发布 seed；safe-point
  physical commit 使用独立 1 ms 配置门槛，预算超限转当前状态 seed，而不是延长 plan TTL；
- root-workflow active-window 只限制每 epoch ticket working set，未激活 workflow 仍保留 RCCG、
  visible admission 和 fairness credit；
- queue/admission wait、GPU-service inactivity watchdog、activation deadline 已分离；runtime 周期性原子
  写 `latest_runtime_summary.json`，最终 gate 直接报告 source plan ID 缺失和未决事务；
- `physical_bundle_preview`/`bundle_lease_aggregated` 改为 digest/counter，debug 采样并有大小上限；
  correctness/metrics 使用有界异步 writer 且不能丢弃；shutdown 显式 prepare、bounded ACK drain、
  abort unresolved transaction、final summary 和 ACK；
- 当前全量 CPU 回归为 `447 passed, 8 skipped`。下一步只运行一次 4/8/12/24 correctness ladder，
  不在同一负载档位循环调参。

严格边界：上述结果只证明代码契约和 CPU 机制正确，尚未证明 1 ms safe-point commit 的真实 P99、
active-window 的最佳大小、clean workflow completion 或性能收益。P5 仍需 GPU gate 后才能关闭。
当前 shutdown prepare/drain/abort/ACK 是 scheduler runtime 内部状态机，由 SGLang scheduler
`finally` 触发并写入审计；尚未增加 tokenizer/frontend 与 scheduler 的专用跨进程 shutdown IPC。

实施增量（2026-07-30，restore stability repair）：

- `RestoreServiceGrace` 在 restore request 首次重新进入 batch 时建立，按 batch completion 后
  `output_ids` 的实际增量累计 decode service，默认要求 32 token；prefill、排队时间和固定
  wall-clock cooldown 都不能偿还该 quantum。request 正常结束、取消、cache reset 或 shutdown
  会显式终止 grace；
- running-retraction physical solver 和 JointPlan 指定 victim 的 safe-point 重验共享同一
  `policy_eligible` 条件，active grace request 不能被 stale semantic plan 绕过；
- restore funding command 记录当前 allocator deficit。ACK 后只把 `min(actual reclaim, deficit)`
  转成真实 allocator allocation，并记入 obligation escrow；grant 时在同一 scheduler safe point
  释放 escrow、建立 admission lease 并立即提交 H2D，不保留仅存在于逻辑计数器中的 credit；
- overdue restore 的 normal-admission barrier 不再因 active lease 存在而关闭；escrow、lease、
  cache reset、request cancel 和 shutdown 共享 rollback 路径，并输出 reservation/grace telemetry；
- 新增 service-quantum、funding reservation 守恒和 active-lease barrier 定向测试；全量 CPU
  回归为 `442 passed, 8 skipped`。该结果只证明状态机和 allocator 契约，GPU liveness 与性能仍
  以一次固定 w4 trace 为准。

2026-07-31 阶段边界：P5 的 online authority、visible waiting queue、ActionSlice 局部校验、
running retraction、durable restore obligation、allocator-backed lease、debt-owned funding 和
service grace 作为 P6 的确定性安全底座冻结。当前不再为开始 P6 而继续修改 agent 终态协议或用
更长实验反复调试 workload；P5 clean-completion gate 未通过这一事实继续保留，P6 实验不能用它
掩盖或替代。P6 预测器只能向同一个 JointPlanner 提供候选，不能成为第二个 victim、admission 或
retraction 策略源。

P5 到 P6 的接口固定为：

```text
P5 observed plan A0
  + exact current allocator/Radix state
  + obligation_revision / lease_revision / grace_revision
  + active grace victim exclusion
  + one-transfer-in-flight and cancellation/deadline
  -> P6 candidate generation and risk evaluation
  -> safe point revalidates a complete ActionGroup
```

任何 revision 变化只淘汰读取该状态的 transaction group；schema/model mismatch、全局过期或
transition-open 才回退整份 predictive envelope。已提交 retraction 必须原子建立 durable
obligation，且 active grace request 永远不能成为 victim。这两项在未来开放 predictive
retraction 前再次作为独立 gate 验证。

贡献边界：P5A observed active-set 是 baseline/minor mechanism；P5B/P5C 的候选贡献是
`causal replacement -> complete physical blocker-set -> RETRACT/OFFLOAD/DROP -> ACK -> ticket`
原子事务。当前优先完成其真实 GPU 端到端路径，不把复杂 counterfactual 模拟或外部 baseline
适配放入实现关键路径。`TentativeUnlockPreview` 继续作为采样诊断，不在本阶段升级为正式执行
前置门槛。

P5 继续优化完整 JointPlan，不预先降级。正式运行同时测量 scheduler critical-path overhead、
planning fallback、actionable coverage、component freshness 和 plan age；若优化后 critical path
P99 仍超过 scheduler step 的 5%，或 actionable coverage 低于 80%、planning fallback 高于 20%、
plan age 超过主要决策窗口，才收敛为 Local Frontier JointPlan，不能仅放宽 TTL/budget。

退出条件：

- correctness/liveness 不低于 P0；
- 不出现 workflow starvation；Jain fairness 和 virtual-runtime lag 继续记录，但默认不作为牺牲
  明显吞吐收益的硬 gate；
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
2. 定义 action-critical inversion：当前公平/FIFO 顺序或 scheduler-only oracle 选择的 agent，
   与最早解除 causal blocker、启动 external work 或扩大 runnable frontier 的 agent 不同；
3. 使用真实 boundary、fan-out、consumer 和 reentry 构造 action-unlock oracle；
4. 先比较三个内部目标：当前 observed order、独立 scheduling/KV oracle 和 action unlock；
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
- action-unlock oracle 相比 reactive baseline 和 `best(O1, O2)` 在 workflow JCT 上至少提升
  10%，置信区间不跨 0，并降低 causal-blocked time 或 tool-launch idle gap；
- structured action 或可靠 runtime boundary 覆盖主要执行时间，UNKNOWN 不成为主导状态；
- reversible commitment oracle 相比 point-decision policy 有正净收益，且收益不是只来自
  理想 PCIe 带宽假设。

若 action-unlock 条件不满足，不能为了保留论文故事强行启用。P6 回到原 Dynamic Frontier
分支，但也必须通过独立 oracle gate；两者都不成立时，predictive policy 停止扩展，P5
Observed-State JointPlan 作为最终系统策略。

### P6：Uncertainty-Gated Predictive JointPlan

P6 的核心不是“置信度高就迁移”，而是：在 closure-complete 的有限因果 scope 上构造全局联合
场景，并在确定性 P5E 约束下比较 observed plan `A0` 与少量完整动作包。预测结果只能进入同一个
JointPlan；训练模型之前必须先完成观测、事务、scope、horizon 和风险选择契约。

```text
incremental parser / RCCG / P5 active window / physical telemetry
                              |
                              v
                 closure-complete BeliefScope
           JOIN、blocker、producer-consumer 原子因果组
                              |
                              v
             local conditional distributions
       boundary/service/wait/growth + support/OOD
                              |
                              v
              RCCG Scenario Composer
        shared episode factor + particles + JOIN/producer
                              |
                              v
                 FrontierBeliefSnapshot
        inferred top-K scenario + OTHER + evidence read set
                              |
                              v
             finite-horizon ScenarioRiskPlanner
          A0 + current-data-plane-compatible candidates
                              |
                              v
          PredictivePlanEnvelope with ActionGroups
                              |
                              v
       safe point full-group physical/resource validation
```

#### P6.0：训练前标签可识别性与版本化观测

三类边界必须分别建模和报告，不能互相替代：

```text
Incremental action boundary：TOOL/SPAWN/HANDOFF 在第几个生成 token 首次合法完成
Runtime transition boundary：SPAWN/HANDOFF/RETURN 事件实际发布时刻
Reentry boundary：JOIN_SATISFIED/REACTIVATE/TOOL_RETURN/MESSAGE 到达时刻
```

补齐 runtime transition/reentry 事件只能改善 RCCG 和 reentry demand；exact incremental action
boundary 仍需要在线增量 parser，或保存原始 token/timestamp 后做确定性的离线 parser replay。

任务：

1. `ActionFrontierObserver` 记录每次增量 parser 状态、generated token index、首个合法 action
   boundary、对应 TOOL/SPAWN/HANDOFF/RETURN、真实 reentry 和删失原因；
2. 报告 exact-boundary call coverage、runtime-only boundary、UNKNOWN/malformed、
   reentry cause coverage 和 demand-label completeness；
3. GPU service 标签只包含条件 prefill/decode service，不包含策略产生的 queue wait；
4. PCIe telemetry 至少按 direction、bytes、extent/page count、host-copy/pinned 状态、command kind、
   native HiCache 并发流量、allocator/callback overhead 分层；
5. 将 obligation、lease、grace 形成独立单调 revision，并随 snapshot/envelope 发布；
6. exact boundary 覆盖不足时，RUNNING_LLM 只预测 remaining decode demand，不宣称预测
   action boundary。
7. 新增事件采样的 `frontier_decision_points.jsonl`。只在 LLM submit/result、每 16 或 32 个
   decode token、tool start/end、spawn/return/join/message、HBM pressure 和 transfer completion
   生成样本；禁止每个 scheduler tick 生成训练行；
8. 每个 decision point 保存 RCCG version、closure-complete scope、invocation 状态与 elapsed
   service/wait、可观测 graph/HBM/PCIe/batch 特征、horizon 内下一 runtime boundary、剩余 service、
   reentry 后 prompt/output growth 和逐 call censor reason；
9. duplicate suppression、request/tool timeout、abort/cancel、recursion-limit 和 workflow shutdown
   必须携带 request/tool-call identity 发布显式 censor event，禁止从 aggregate count 反推样本标签。

检查点：coverage 报告先于任何模型训练；低覆盖不会通过增加不可观测标签或 prompt 先验修补。
在 native per-token service timestamp 接入前，decode-time coverage 必须报告为 unavailable，不能用
call coverage 或 token 比例代替。

2026-07-31 固定 `p5f_fixed_w4/20260730T132313Z` 离线复放结果：排除 4 次 runtime internal
summarization 后，466/466 策略 call 的 agentic/native request、token demand 和 runtime structured
action 可对齐，447/447 需要 reentry 的 call 有显式 tool/JOIN cause；但 exact boundary 为
0/466，条件 GPU service sample 为 0。故 UnlockHazard 未达到训练门槛；旧版 remaining-service
表述作废，remaining decode demand 与独立硬件 service model 必须分别判断门槛。当前保持 P5
observed fallback。PCIe 1,199 条 operation 的 direction/bytes/page/command 覆盖完整，
但 pinned state、native concurrency、allocator overhead 为 0%，native 精确 start timestamp 仅
69/1,199。机器可读结果见
`experiments/processed/p6_0_coverage_20260731/p5f_fixed_w4_20260730T132313Z.json`。

2026-08-01 在通过 P5 system gate 的固定 autonomous w4
`experiments/raw/p5g_autonomous_w4/20260801T115617Z` 上完成第二轮 P6.0。analyzer 已支持 Deep Agents
嵌套 trace，且只允许 native request ID + workflow ID + invocation ID 精确关联，禁止 ordinal
fallback。575/575 request identity、token demand 和 GPU service 完整；45,201 个 batch sample 展开为
108,344 个 request interval。600 个 external wait、5 个 closure-complete JOIN 和 1,535 个
submit-to-complete transfer operation 已写入带输入/输出 SHA-256 的版本化数据集，唯一性、外键和 JOIN
member RETURN 校验均通过。

门槛结论更新为：remaining decode demand 可进入训练代码开发；exact incremental action boundary
仍为 0/575，UnlockHazard 保持禁用；reentry 539/559，缺少的 20 个 function call 与 aggregate duplicate
suppression 数量一致，但因没有逐 call censor event 不做事后伪标；direct DMA start 为 0/1,535，transfer
目标只能是 submit-to-complete operation latency。全部样本来自 SymPy，不能执行有意义的跨项目
calibration。机器可读 coverage 与 dataset 位于
`experiments/processed/p6_0_coverage_20260801/`，完整报告见
`docs/experiments/beliefkv_p6_0_autonomous_w4_training_evidence_2026-08-01_zh.md`。

#### P6.1：Closure-complete BeliefSnapshot 与 ActionGroup

`BeliefScope` 的预算单位是原子因果组和建模成本，不是固定 invocation 数量：

```text
纳入 JOIN_ALL/ANY -> 纳入全部未完成成员与 waiter
纳入 retraction   -> 纳入完整 blocker set
纳入 consumer     -> 纳入必要 producer-consumer closure
闭包超过预算       -> 整组进入 OTHER/A0，禁止部分截断
```

新增正交类型 `DemandScenario`，不修改 observed `FrontierScenario`。一个 scenario 对整个
scope 做联合 assignment；全局只保留 top-K，剩余质量进入 action-specific OTHER。未创建的 agent
只能表示为 anonymous role/action class，不能获得 request ID、ticket 或物理资源。

在线提交单位升级为：

```text
ActionGroup:
  group_id
  atomicity = ALL_OR_NOTHING | PREFIX_COMMITTABLE
  actions
  dependency_dag
  evidence_read_set
  resource_certificate
  compensation
```

P6 第一版全部使用 `ALL_OR_NOTHING`。safe point 可以提交多个互不相关的完整 group，但不能从
`RETRACT(q) -> COMMIT_CPU(q) -> ADMIT(r)` 中只提交前缀。物理 closure 重新物化后，reclaim/startup
bytes 或 generation 改变必须重验整个 group 的 resource certificate。

#### P6.2：单一 FrontierBeliefModel 的因果需求分解

保持一个对外模型和一个预测快照，不允许多个 predictor 独立出动作。模型只学习局部条件分布，
RCCG 已知结构和全局 scenario 均由推理阶段 composer 构造。内部按因果来源建模：

```text
RUNNING_LLM -> 下一次可执行 runtime boundary 类型 + remaining decode tokens
WAIT_TOOL   -> success/error/censor competing risk + conditional residual wait
WAIT_CHILD  -> 每个 child 的 token demand、external segment 和 output demand
WAIT_MESSAGE-> producer remaining demand + message-arrival probability
reentry 后  -> prompt delta 与下一轮 output demand
```

`JOIN_ALL/JOIN_ANY`、当前 target invocation、Radix closure 和资源占用均为 RCCG/系统观测，不训练。
composer 使用共享 workflow/episode 随机因子或 cluster residual bootstrap 采样局部分布，但只保留完整
JOIN/member、producer-consumer 和 blocking-chain 依赖，不能对 raw token demand 做 max/min。每个联合
demand scenario 必须先与候选 JointPlan、safe-point physical snapshot 一起经过 Radix physicalizer，得到
候选方案下的 uncached prefill、batch composition 和 transfer；随后 service timeline 才能得到各 child
完成时刻，最后计算 JOIN_ALL 的 max 或 JOIN_ANY 的 min。因此 top-K scenario 不是直接监督标签。禁止
直接从旧 P5 wall-clock 学习 WAIT_CHILD/JOIN completion，否则会把旧调度策略固化进模型。
概率契约仅保留 scenario probability mass、OTHER mass、calibration coverage、support/backoff 和
OOD reason，不再增加语义重叠的全局 confidence。

第一版 `FrontierBeliefModel` 使用可审计的 Structured Conditional Particle Model：variable-order
context tree/层次平滑分类 boundary；分层 competing-risk survival 建模 tool wait；log-binned empirical
distribution 或 quantile GBDT 建模 prompt/output demand；GPU/PCIe 使用独立条件 service curve。所有
局部组件只向一个版本化模型发布分布，不拥有 keep/offload/prefetch 决策权。

2026-08-03 demand/service 解耦修复将在线类型拆为 `FrontierDemandOutcome/DemandScenario` 与
`TimedScenario`。前者只包含 current sequence、remaining decode、prompt growth、next output、external
segment 和 dependency relation；后者只能由 candidate JointPlan + physical snapshot + conditional batch
service model 生成。旧 `remaining_gpu_service_ms/next_gpu_service_ms` 在正式 Frontier loader 中 fail-
closed，`batch_size/observed_gpu_service_ms` 不再进入语义 feature key。tool survival 额外条件化
active-tool count 和 backend-pressure proxy。

正式 schema 进一步将这些 scheduler/load 字段从 invocation 顶层移入 `diagnostics`，loader 遇到
顶层污染字段直接拒绝拟合。相同 task 的 w1/w4/w8 paired invariance audit 必须同时匹配消息、工具
schema、采样配置的 semantic digest 和显式 seed；无 seed 配对只作诊断。candidate timeline 分离
dependency release、JOIN reentry 与 invocation completion，并让 parent batch 显式依赖 H2D transfer
completion，避免把 JOIN 时刻误当成 parent 后续 LLM 完成时刻。

#### P6.2.1：Development、训练与测试隔离

当前 autonomous w4 固定为 `development-only`。在其上实现、调试、选择数据字段和做序列化 sanity
check 不等于面向测试集训练；但该 run、对应 SymPy project/task 及重复 rollout 不得再用于最终泛化
结果。训练前必须生成并冻结显式 `split_manifest.json`，不再使用隐式 SHA-256 80/10/10 分桶：

```text
train       = 60% projects
calibration = 20% projects，只做概率/区间校准
test-ID     = 20% unseen projects，不参与模型选择和在线更新
test-OOD    = 完整未参与训练的 workload family
```

同一 repository、task、base commit 和全部重复 rollout 必须属于同一 split。train project 内通过
leave-one-project-out 选择超参数。coding 数据分批采集 80--120 个 workflow、覆盖至少 8--10 个
repository；predictive action 全部关闭并使用冻结 P5 observed policy。先接一种非 coding workload，
优先选择与现有 sandbox/tool runtime 接近的终端任务。硬件 GPU/PCIe service 数据使用独立微基准，
agent trace 中的 service interval 必须按完整 request/episode 分组，不能当作独立 IID 样本。

2026-08-03 实现状态：`frontier_decision_points.jsonl`、逐调用 `CALL_CENSORED`、显式 project split、
冻结 collection plan、单一 `FrontierBeliefModel` 和 RCCG particle composer 已进入代码。当前 W4 模型
严格标记 development-only。正式拟合按 `(episode_group_id, invocation_id)` 聚类加权；calibration
只学习分类 temperature 和 episode-level split-conformal interval，不修改 train 计数；test-ID/OOD
只允许 evaluation loader 读取。首批正式 coding 数据按 train batch 分段收集，任何 predictor 或
predictive physical action 均关闭。top-K scenario 仍只由局部分布和 RCCG 在推理阶段组合，不存在
top-K 监督标签。

首个 `p6-009` 尝试仅作为采集管线诊断：默认 correctness-repair agent 在模型自然终态后继续产生
LLM/tool 轨迹，且短 ACK 窗口造成两条 runtime event 未及时确认，因此整轮显式标记为 formal
ineligible。后续 collector 固定使用 model-terminal/no-repair 语义、10 秒 event ACK、运行时源码前后
fingerprint，并将最终 eligibility 原子写入 collection contract。数据导出和模型 loader 双重 fail-
closed：无冻结 split、source invalid、collection gate 失败、重复 run 或重复 decision 均不得进入
正式 fit/calibration/test。

后续采集又识别并修复了 autonomous runtime 的生命周期接入缺口：不再直接依赖
`create_deep_agent()` 内置的 170K summarizer，而是显式构造 Deep Agents 核心工具栈，为 parent 和
每类 child 分别安装唯一的 32K/retain-8K `ContextLifecycleMiddleware`，并在 task RETURN 边界剥离
branch-private `_summarization_event`。缺少该策略、未完成或在 model startup 前遭遇外部 GPU 抢占的
批次分别由 `COLLECTION_INVALID.json`/`STARTUP_FAILED.json` 标记；它们和旧
`PILOT_INVALID.json` 一样被 exporter fail-closed 拒绝。正式 train 仍须从新的 `p6-009` 开始，至少
积累两个 train project 后才能运行 LOPO。

修复后的正式 `p6-009-train-mixed-r0` 已于 2026-08-03 完成：8/8 workflow 通过 system JCT 和
source-stability gate，导出 11,674 个 eligible decision point、712 个 remaining-decode-demand target、
1,151 个 external survival target 和 11 个 JOIN reentry。该批只有 Django 一个 train project，且
0/8 workflow 通过 native-agent clean/task-correctness gate，所以只能作为带显式 censor 的局部训练
证据，不能执行 LOPO、最终 fit 或报告任务性能。reentry cause 覆盖 91.73%，exact incremental action
boundary 仍为 0%；对应子模型和 unlock-hazard 分支保持关闭。下一批 `p6-010` 尚未开始。

train 内模型选择固定使用 project-macro LOPO，只搜索同一个 Structured Frontier 模型的有限
context-order/support/smoothing 候选；selection manifest 的 project 集合必须与最终 fit 完全一致。
GPU service 使用独立 tagged microbenchmark 按唯一 `sample_id` 还原完整 batch；输入包含 phase、batch
composition、sequence-length distribution、chunk position、prefill/decode mixing 和 PCIe/HiCache
contention，输出 service-time 分位数。agent runtime 的 native overlap interval 只作外部验证，禁止逐
request 重复计权或进入正式 fit。PCIe service curve 继续使用独立 transfer microbenchmark。两者都不
从 agent workflow wall-clock 直接回归，避免把旧 P5 排队和调度策略固化进 predictor。

#### P6.3：有限 Horizon Scenario-Risk JointPlan

每个 scenario 的事件驱动仿真在以下任一条件出现时结束：

```text
到达下一次可观测 action/reentry 边界
任一 invocation 已模拟两个 transition
累计模拟 GPU service 达到 100 ms
```

horizon terminal cost 只包含未解除的 action-unlock delay、workflow service lag 和已创建但未履行
的 transfer/restore debt；不为 horizon 之后的猜测 unlock 记收益。

主目标为 `action-unlock delay + workflow service lag`。当前 allocator、Radix lock/closure、Host、
one-transfer-in-flight、obligation/lease/grace 和 cancellation 是 100% 确定性硬约束；未来 HBM/
reentry 才允许 chance constraint；liveness 永远不能概率化。HBM-time/PCIe-time 只作为在线 shadow
price 正则项，已经造成 service delay 的 transfer 不再重复计费。

OTHER 使用 action-specific residual：不给预测动作 unlock 收益，计入已发生的 HBM/PCIe/restore
成本，并使用最早合法 reentry。无法得到有限风险界时只允许 `PREPARE_HOST`，禁止 commit 和
retraction。

第一版只离线比较：

```text
A0 = observed P5 JointPlan
A1 = A0 + PREPARE_HOST
A2 = A0 + PREFETCH_GPU
Oracle = 使用真实短期 transition
```

暂不生成 predictive retraction、`RUN(request, quantum)`、run-to-action、DROP/RECOMPUTE 或不存在的
DISCARD。`COMMIT_CPU` 只在后续独立阶段开放。

规划器必须报告 candidate generation、single-scenario evaluation、full risk-plan、publish age、
stale rate、safe-point validation 的 P50/P95/P99，以及 rollout cache hit 和 planner CPU。按
invocation/evidence revision 缓存 service-demand rollout，不在每个 scenario 重遍完整 RCCG。预算
耗尽只停止优化，A0 始终立即可用。

#### P6.4：Predict-Plan-Publish Shadow

latest-wins worker 原子执行 `snapshot -> predict -> risk-plan -> publish`，envelope 携带 graph/page/
topology/fairness/admission/transfer、obligation/lease/grace、parser 和 model revision。safe point 按
ActionGroup 的依赖闭包局部提交；无关 workflow 变化不淘汰整份 envelope。该阶段只记录 would-
apply 和 regret，不发送预测命令。

#### P6.5：逐风险上线

上线顺序固定为：`PREPARE_HOST -> PREFETCH_GPU -> execution order/ADMIT/DEFER -> COMMIT_CPU ->
predictive retraction -> run quantum/DROP/RECOMPUTE`。每一级均可单独关闭，并回到相同 JointPlanner
的 A0。开放 predictive retraction 前必须满足：candidate 生成时有 obligation slot、commit 原子创建
durable obligation、active grace 排除 victim、revision 变化淘汰整个相关 group。

退出条件：

- P6.0 coverage 足以支撑所声明的标签，否则明确使用 service-demand fallback；
- scope 截断测试中 JOIN/blocker/consumer 不出现 partial causal atom；
- 100% predictive command 来源于通过完整证书校验的 ActionGroup；
- risk planner 的结果发布年龄小于主要状态变化窗口，safe-point validation 满足 P5 门槛；
- prediction 相比 A0 有独立的 JCT/action-unlock/unhidden-stall 净收益；
- wasted shadow/prefetch、额外 HBM-time、PCIe-time 和 delayed-workflow cost 小于避免成本；
- OOD/OTHER fallback 与 P5 A0 决策一致，且不产生新的 restore starvation。

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

### P8：系统冻结后的基线选择与对比

P8 只在 BeliefKV P5/P6 功能、正确性、workload 和配置冻结后启动。P2-P6 不为该阶段预建
统一竞品框架，也不承诺把 ScaleSim、AugServe、ThunderAgent、CONCUR 等方法全部改写到
BeliefKV 数据面。

任务：

1. 先制作 compatibility matrix，逐项检查候选工作的代码可用性、模型/硬件、serving engine、
   workflow 语义和前端 metadata 假设；
2. 将实验分为 BeliefKV full dynamic workload 和 common-denominator workload。前者证明目标
   场景收益，后者只用于与确实支持相同语义的方法比较；
3. 优先运行论文原生开源实现。只有原生实现不可运行但核心策略可被忠实表达时，才决定是否
   在独立实验目录编写最小适配；不修改 `beliefkv/policy` 核心路径，也不建立长期维护的通用
   policy framework；
4. 对依赖 invocation distance、静态 DAG、program phase 或工具时长先验的方法，仅在真实
   frontend 能提供相同信息时运行在线版本；否则作为 assumption difference 讨论，不临时构造
   hindsight 数值冒充在线结果；
5. 统一或显式报告模型、量化、KV pool、SGLang/vLLM 版本、PCIe、batch、arrival process 和
   workload 差异；
6. 如果没有方法能在完整动态 workload 上忠实运行，主表使用 SGLang/HiCache、reactive
   policy、scheduling-only、KV-only 和 BeliefKV ablation；外部方法只在可比子集单独成表。

退出条件：

- 每个外部结果都能说明代码来源、输入先验、修改范围和适用 workload；
- 不把 unsupported workload、缺失 metadata 或启动失败计为 BeliefKV 性能收益；
- 至少完成一个与 BeliefKV 存在实质场景交集的强外部基线；如果客观上不存在，必须用完整
  compatibility evidence 说明，而不能只声称“场景更新颖”；
- 最终结论分别回答 full dynamic workload 的有效性和 common subset 上的竞争力，不混写两类
  结果。

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
- reactive 默认路径在固定 snapshot 上的确定性与 resource accounting 一致性；
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
16. 通用 contract 的 hindsight metadata 无法被 online planner 读取。

### 12.3 真实 GPU 测试

P3-P6 每个正式配置至少运行：

- pinned upstream SGLang Radix LRU；
- HiCache write-back；
- HiCache write-through；
- HiCache write-through-selective；
- HiCache + reactive causal policy；
- BeliefKV observed-state JointPlan，不使用预测；
- BeliefKV frontier-belief JointPlan，不使用 reveal；
- BeliefKV full policy，Reveal-and-Commit 仅在满足 gate 时开启；
- scheduling-only oracle、KV-only oracle 和 full JointPlan oracle。

P8 在系统冻结后根据 compatibility matrix 选择可运行外部基线；它们不扩张 P3-P6 的必跑矩阵。

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
- 局部 execution-order reversal、KV-action reversal、`S/K/J` local synergy gap 和
  `counterfactual_unidentifiable` 比例；
- 相同 active count 下 locked bytes 分布、multi-blocker extent/bytes、preview/realized reclaim
  误差和 retraction 后 recompute tokens/time；这些随正式运行采样，不要求单独复杂模拟；
- 完整 O0/O1/O2/O3 和外部方法指标仅在系统冻结后的可选深入分析/独立对比阶段定义；
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
| JointPlan 只统一接口但无 synergy | 创新退化为工程重构 | 先用局部 Jointness Audit，最终用端到端结果闭合；无 gap 时不作为 Major contribution |
| 当前 workload topology entropy 低 | 无法证明动态 MAS | P3 前置 cyclic peer 和 mixed workload，不用固定两-child trace 训练/评价 predictor |
| 外部工作在 common subset 已解释主要收益 | BeliefKV 论文竞争边界收窄 | 系统冻结后运行可比强基线并收缩主张，不反向修改核心接口 |
| dynamic trace 对调度敏感 | 冻结 replay 产生虚假 oracle 收益 | 标注 schedule sensitivity；semantic race 只作上界并用真实 A/B 闭合 |
| structured action coverage 低 | UnlockHazard 无法稳定工作 | UNKNOWN 回退 P5；action-frontier 不作为 Major contribution |
| action criticality 与短作业/value-density 等价 | action-unlock 可能退化为已有目标 | P5.5 先与内部独立 oracle 比较；最终再在可比 workload 上检查已有目标 |
| CPU shadow 可用 PCIe 窗口很少 | reversible commitment 只有机制价值 | 单独报告 idle service、commit/cancel 和净收益，不作为必要组件 |
| HiCache overlap 隐藏大部分 restore | KV 预测收益缩小 | 以 unhidden stall 为准，降低该贡献优先级 |
| shared/closure 使计划频繁失效 | context 估计不可信 | 强制 physical preview 和 generation 二次校验 |
| reject 结果未反馈导致 retry storm | PCIe/CPU 与 scheduler 开销被无效命令放大 | typed blocker + event-gated attempt ledger；已知 blocker 禁止 tick retry |
| Python planner 开销过高 | scheduler step 被拖慢 | top-K/深度/候选上限，事件触发，超时回退 |
| 新版 HiCache 已覆盖部分机制 | 贡献边界收缩 | 保留 agent execution/KV joint decision，删除通用机制主张 |

## 15. Go/No-Go 与论文主张门槛

### 15.1 两层 Joint Synergy 门槛

在至少三类真实 MAS workload 中：

- P2 H2D/allocator/Radix 可靠性 gate 必须先通过；
- 至少覆盖 blocking subagent、cyclic peer multi-agent 和 mixed workflow；
- HBM-pressure 时必须稳定出现多个已观测 runnable agent，且 agent 选择会改变 admission 或
  optimal physical bundle plan；
- **系统搭建期门槛**：从正式运行采样真实 pressure snapshot，完成短窗口 Local Jointness Audit；
  execution/KV 双向决策反转和正 local synergy gap 必须可重复出现。该审计不要求完整动态 O0-O3
  模拟，也不阻塞 P5A--P5C correctness 路径完成；
- **系统完成后门槛**：通过端到端重复运行证明完整 JointPlan 相比最强可运行的独立
  scheduling/KV 组合仍有净收益，并由 action throughput、causal-blocked time、unhidden stall
  或 workflow JCT 中至少一项解释；动态路径同时报告 transition hash、调用量和拓扑分布；
- subagent 与 multi-agent 的 frontier transition 都有足够 coverage，不能只由固定二 child
  workload 支撑 predictor。

P6 action-frontier 分支还必须满足 P5.5 的独立门槛：action-critical inversion 有实际代价，
action-unlock oracle 优于 reactive baseline 和独立 scheduling/KV oracle，且 reversible
commitment 在实测 PCIe service 下有正净收益。外部方法的最终对比不作为 P6 correctness 的
前置依赖。

任一关键门槛不满足，应停止扩大 predictive JointPlan。JointPlan 可保留为工程统一接口，
RCCG lease/ownership bridge 保留为正确性层，但不能作为 Major contribution。

Reveal-and-Commit 使用独立可选 gate：只有高代价 scenario disagreement 中存在 READY
revealer、`P90(T_reveal) < T_force` 且 offline reveal oracle 有净收益时才实现主动版本。它不再
是整篇工作的必要门槛。

### 15.2 系统结果门槛

完整策略必须同时满足：

- P5 observed-state JointPlan 相比 strongest separate/HiCache/reactive baseline 有独立收益；
- P6 predictive JointPlan 相比 P5 仍有独立收益；
- workflow JCT 或 unhidden stall 至少降低 10%，且置信区间不跨 0；
- 无效 D2H/H2D/recompute 或 unhidden stall 至少降低 15%；
- fairness、OOM、liveness 和 correctness 不退化；
- execution/admission/residency mismatch、错误 parent park 和未满足依赖的 request dispatch 为 0；
- 新版 HiCache 数据面上收益仍存在；
- OOD 时不劣于 P5 observed-state JointPlan；
- 系统冻结后完成 compatibility matrix，并在可比 workload 上纳入至少一个强外部基线，或用
  充分证据说明不存在忠实可运行的交集。

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
feat(policy): add immutable internal policy snapshot and replay contract
feat(control): track observed producer-consumer relationships
feat(runtime): capture peer handoff message and reactivation events
feat(runtime): record structured action frontier and valid action boundaries
feat(experiments): add cyclic peer and mixed dynamic workflows
feat(policy): add side-effect-free scenario what-if packer
feat(experiments): add reactive scheduling-only KV-only action and joint oracles
feat(policy): add observed-state JointPlan in shadow mode
feat(runtime): apply JointPlan execution admission and transfer dependencies
feat(policy): enable observed-state joint agent and KV scheduling
docs(experiments): report action-unlock insight gate and branch decision
feat(predictor): add calibrated unlock and reentry hazard scenarios
feat(policy): add robust predictive JointPlan actions
feat(policy): add guarded reveal-and-commit when oracle-gated
test(experiments): add pinned and current-HiCache policy matrix
# After the BeliefKV runtime and configuration freeze
docs(experiments): characterize external baseline compatibility
test(experiments): add selected runnable external baselines on common workloads
docs(results): report dynamic-workflow joint-synergy and go-no-go decision
```

每个提交必须通过当前测试，并保持新策略默认关闭。P2 reliability、P2.5 internal contract 和
P4/P5 数据路径 correctness 是完成 P5 系统的顺序硬门槛；Local Jointness Audit 是扩展 P6 和
形成 Major contribution 前的门槛，但完整动态 O0-O3 模拟与外部 baseline 不是 P5 系统搭建的
前置条件。P5.5 是选择 P6 action-frontier 或 generic frontier 分支的硬门槛。不能先
训练 predictor，再寻找能证明它有效的 workload。P8 在核心系统冻结后执行，
根据 compatibility matrix 选择有效且可运行的强基线，并准确披露不兼容项。

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
multi-agent 和 mixed workflow；P4/P5 先完成不依赖预测的 agent/KV JointPlan，并从真实运行
同步采样 Local Jointness Audit。系统冻结后再用端到端比较闭合 joint synergy；P5.5 决定
action-unlock 是否值得成为 P6 主线；
P6 再证明预测相对 observed-state JointPlan 的增量价值。Reveal-and-Commit 只有在独立 oracle
gate 通过时启用。外部方法只在系统冻结后根据真实可比性选择，不预建统一竞品框架，也不反向
增加在线
核心路径的复杂度。
这样既不会把动态 workflow 简化成 spawn 后的静态 DAG，也不会用一个 MLP 掩盖 agent 调度、
物理 KV 和运行时因果之间真正需要联合解决的问题。

## 18. P5G：Transactional Restore Coordinator

P6 online path 暂停在 coverage/训练数据采集阶段。恢复预测策略之前，P5 必须先关闭以下
正确性问题：被 retraction 或 CPU-only prefix 阻塞的 request，其 restore debt 必须和 SGLang
当前物理 ownership、HiCache command ownership、allocator reservation 以及后续 admission
组成一个可审计事务，不能由 timeout、周期 retry 或延长 lease 掩盖。

### 18.1 物理真相与 ENGINE_BUSY

不得维护新的综合 residency 枚举。每个 scheduler safe point 从 SGLang 原生对象重建正交快照：

```text
NativeRequestPhysicalSnapshot
  queue_location: WAITING | RUNNING | CHUNKED | NONE
  req_pool_slot: int | None
  radix_lock_owned: bool
  native_load_operation_id: str | None
  explicit_transfer_ids: tuple[str, ...]
  request_generation: int
  terminal: bool
```

`_active_request_ids` 只保留历史观测用途，不参与 ownership 判断。`ENGINE_BUSY` 只能由当前
req-pool slot、当前 Radix lock、native load 或互斥的 explicit transfer 推导。allocator、Radix
tree 和 HiCache operation 仍是物理真相源。

### 18.2 RestoreTransaction

`RestoreTransaction` 是 aggregate root，但不压平子状态：

```text
RestoreTransaction
  obligation
  feasibility_certificate
  capacity_reservation
  prefix_pin
  physical_operations[]
  admission_state
  service_grace
  wait_condition
```

合法状态顺序为：

```text
WAIT_FEASIBILITY
  -> WAIT_FUNDING | WAIT_EVENT
  -> PREPARED(reservation_id, pin_token, certificate)
  -> H2D_QUEUED | H2D_ADOPTED
  -> RESTORED_RESERVED
  -> ADMISSION_COMMITTING
  -> ADMITTED
  -> SERVICE_GRACE
  -> SATISFIED
```

这里的原子性限定为 scheduler safe point 单写者：先做只读 feasibility 和 command preflight，
再 prepare reservation/pin，随后 enqueue-or-adopt 并提交 transaction；任一步失败都按
command subscription、pin、reservation 的逆序回滚。一个 transaction 可以按顺序拥有 funding
D2H 和 restore H2D 等多个 physical operation，但同一
`(stage, attempt_key, certificate_generation)` 最多提交一次 canonical command。

### 18.3 类型化命令所有权

`enqueue_control_command()` 返回 `EnqueueOutcome`，不再用一个布尔值混合 guard suppression、
context 冲突和真实入队：

```text
status: ENQUEUED | ADOPT_EXISTING | RETRY_GUARD_BLOCKED |
        CONTEXT_CONFLICT | STALE_CERTIFICATE
canonical_command_id
attempt_key
blocker_codes
wake_conditions
```

命令等价键至少包含 context/epoch、kind、bundle generation、closure fingerprint 和 target
residency。`ADOPT_EXISTING` 必须把所有 restore transaction 注册为 canonical ACK 的订阅者；
一次 ACK 同时推进全部订阅事务。queue/inflight 是 authoritative ownership，
`_queued_by_context` 只是可重建的二级索引。native load 与 explicit H2D 必须在快照中同时可见，
并执行互斥不变量。

### 18.4 事件门控重试

allocator 的每次真实分配/释放仍正常推进 revision。restore 另行记录排除本事务自身写入的
`ExternalProgressToken`：

```text
engine_owner_epoch(context)
closure_fingerprint
effective_capacity_threshold_epoch
command_ownership_epoch
guard_generation
native_load_generation
```

失败后同时记录明确 wait predicate。只有 `available >= required`、engine owner/closure/guard 或
command ownership 等相关谓词变真时才重试；无关 page、全局 allocator revision 和事务自己的
lease grant/rollback 不得唤醒它。guard 阻塞必须发生在 allocator reservation 之前，因此
该路径的 allocator allocation 次数应为零。

### 18.5 单一权限与条件活性

紧急 restore 使用显式互斥模式：

```text
NORMAL_JOINT -> RESTORE_DRAIN_REQUESTED -> RESTORE_DRAIN_ACTIVE -> NORMAL_JOINT
```

进入 ACTIVE 时使当前 JointPlanEpoch 失效；restore coordinator 成为唯一的 admission/residency
authority，普通 JointPlan 只更新 shadow，不能提交动作。退出后基于最新 observed state 生成新
计划。drain 为 oldest debt reclaim 的容量只能由该 debt 使用；新 victim debt 必须排在 oldest
之后，obligation table 必须有空间，且 service-grace request 不得成为 victim。

活性主张是有条件的：request 未取消、raw input 可重建、context 可装入 KV pool、后端最终响应、
且可抢占资源最终释放时，最老 restore 最终获得 service quantum。其余情况进入
`FAILED_UNRECOVERABLE`，并记录 cancellation、raw-input、capacity、backend、owner 和 shutdown
等结构化证据，不能伪装成正常完成。

### 18.6 正确性门槛

GPU gate 前必须通过确定性故障测试：历史 active ID 无实际 slot/lock、guard blocked 零 allocator
写入、ADOPT ACK 多订阅、enqueue 回滚、partial/rejected/stale ACK、native/explicit load 冲突、
abort/reset/deadline/shutdown、drain 权限隔离以及 request ID/context epoch 更新。ACK 完整率只统计
成功进入 canonical queue/inflight 的命令。固定 w4 clean-completion gate 通过后，才恢复 P6
online policy 开发。

### 18.7 Native ownership snapshot 开销门槛

2026-07-31 固定 w4 characterization 在最多 12 个 native request 下采集 105 次真实 ownership
rebuild：P50/P95/P99 为 0.052/0.156/0.523 ms，最大 5.454 ms；真实覆盖 14 次 running
retraction、35 次 post-H2D prefix pin 和 89 条 transfer ACK，受控 shutdown 后无 pending
transaction。结果证明按 restore 事件触发的重建在 w4 下可控，但不能外推到 w24/w32。105 个样本
不足以把 `P99 < 0.5 ms` 作为硬门槛；0.523 ms 与 0.5 ms 的差异没有统计意义，5.454 ms 离群点则
必须通过后续分段计时解释。

当前实现使用惰性 `SafePointPhysicalSnapshot(epoch)`，而不是每个 scheduler tick 构建：

```text
APPLY_EVENTS
  drain ACK/native completion，更新 allocator/Radix/queue
        |
CAPTURE_AND_PLAN
  首个物理状态消费者惰性构建 immutable snapshot
  同 epoch 的 progress token/blocker 查询复用 by_request/by_context
        |
TRANSACTIONAL_COMMIT
  重验 action context 的 native ownership read-set 与 canonical command ownership
  提交后销毁 snapshot；继续规划必须推进 epoch
```

构建按 queue collection、metadata indexing、Radix/request ownership lookup、native/explicit operation
indexing 和 sorting/allocation 分段，并记录 cold build、GC collection、total/mean/P99、每 record 成本
和每 scheduler step 摊销。2026-07-31 CPU 微基准在 N=8/16/32/64/128、每档 500 次下得到约
`5.03 us/request` 的线性斜率；N=128 均值/P99 为 `0.671/0.689 ms`，按旧 trace 每 1,729 step 一次
的触发率折算约 `0.00039 ms/step`。当前热点是 ownership record 组装，不是排序或 GC。结果见
`docs/experiments/beliefkv_p5g_safe_point_snapshot_cpu_2026-07-31_zh.md`。

### 18.7.1 完整 fixed-w4 实测结果

2026-07-31 只运行了一次完整固定 w4：
`experiments/raw/p5g_clean_completion_w4/20260731T124929Z`。结果没有通过 gate：1/4 workflow
`semantic_complete`，0/4 clean JCT；其余分别因 self-handoff validation、LangGraph recursion limit
和 7,200 秒 workflow deadline 结束。运行中 953 个 request 均有 physical start/finish，没有 API
timeout、queue timeout、OOM 或 admission stall，因此不能把 workload 失败归因于 P5G restore liveness。

物理层 246/246 显式 command/ACK 完整且均为 terminal-private Host cleanup；native HiCache 记录
2,923 次 D2H、20 次 H2D，HBM/Host 峰值分别为 15.0 GiB 和约 95.98 GB。但本轮没有创建 restore
obligation、running retraction 或显式迁移，故惰性 `SafePointPhysicalSnapshot` 的 GPU `call_count=0`。
native H2D 不能替代 transactional restore coverage。

受控停止也未闭环：prepare snapshot 中 command/lease/funding/pin/transaction 全空，但 summary 停在
`preparing/final=false`，audit writer 仍有 1 条 pending，独立 transfer telemetry 比 audit 少 1 条
D2H。后续顺序调整为：修复 runtime 根级终止和 self-handoff；实现 frontend 驱动的显式 shutdown
ACK/writer drain；修复单条 physical-start checkpoint 缺失；用确定性 GPU restore micro-gate 覆盖
obligation/snapshot/read-set/service quantum。完整报告见
`docs/experiments/beliefkv_p5g_clean_completion_w4_2026-07-31_zh.md`。

P6 离线标签、数据集和模型开发不再被本轮 workload 失败阻塞，但 P5 接口暂不冻结。确定性 GPU
restore micro-gate 与显式 shutdown gate 通过后，再补一次短时 w8 correctness smoke，验证超过
12 个可见 request 时的 ownership transition、request ID 重用和并发 native operation；w24/w32
留到系统功能冻结后的规模实验。

### 18.8 P5G 最终 gate 与 P6 交接

`terminal` 不是充分条件。最终固定 w4 必须满足：

1. 所有未被外部明确取消的 restore obligation 在 shutdown prepare 前为 `SATISFIED`；
2. 每个 retracted request 在 H2D/admission 后获得真实 decode service quantum；
3. shutdown prepare 时 command ACK、lease、funding、prefix pin 和 RestoreTransaction 守恒，cleanup
   没有用于清除 unresolved debt；
4. `FAILED_UNRECOVERABLE` 只接受单 context 超容量、raw input 不可重建或后端永久失败等结构化证据；
5. 无 false ENGINE_BUSY、自触发 retry、orphan command、allocator/Radix inconsistency 和 starvation。

通过后允许修 correctness bug，但不再改变 P5 架构。P6 online shadow 只记录
`would_prepare/would_prefetch`，不实际发送传输；短时 w8 correctness smoke 通过后才开放任何预测性
物理动作。

### 18.8.1 2026-08-01 autonomous w4 gate 结果

固定目录 `experiments/raw/p5g_autonomous_w4/20260801T115617Z` 只运行一次。4/4 workflow 自然发布
`WORKFLOW_END`，5/5 动态 subagent JOIN 满足，575/575 LLM 和 600/600 tool 调用成对，4/4
`system_jct_eligible`。一次生产 running retraction 完成 exact 6,396,641,280-byte lock release、D2H、
H2D、重新准入和真实 service；restore obligation 在 4,807.21 ms 后进入 `SATISFIED`。受控 shutdown
为 `acknowledged/final=true`，无 pending command、transaction、lease、funding、pin 或 obligation，
allocator mirror 与 Host page index 均一致。

barrier precheck 修复后只发生 1 次 request/drain，且结果为 `plan_created`；68 次 active restore debt
场景在 drain 前被抑制。与确定性 micro-gate 的 493 次 request/drain 相比，无效 barrier churn 已关闭。
因此 P5 system correctness gate 通过，P5 接口可以冻结。

该轮仍只有 0/4 `native_agent_jct_eligible` 和 2/4 task measurement valid：94 次工具
`command_failed` 触发 runtime guard，另有两个 workflow 未观测到成功测试命令。它们不属于 P5
restore/admission liveness 故障，但使本轮不能用于 agent-native clean JCT、SWE-bench 正确率或性能
比较。完整报告为
`docs/experiments/beliefkv_p5g_autonomous_w4_system_gate_2026-08-01_zh.md`。
