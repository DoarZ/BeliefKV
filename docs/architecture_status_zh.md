# BeliefKV 最新架构与实现状态

更新日期：2026-07-27

状态基线：当前工作区，包括 Git HEAD `d76af28` 以及截至 2026-07-27 的 P5A--P5C
实现。本文描述当前磁盘上的实际代码，不只描述已提交文件。

本文回答三个问题：

1. 当前线上真正执行的是哪条代码路径；
2. 哪些模块已经实现，但仍然只用于 shadow、replay 或 oracle；
3. 哪些结论已经有真实 GPU 证据，哪些仍然只是研究计划。

“存在代码”不等于“已经接入在线系统”，而“通过机制测试”也不等于“已经证明性能提升”。
本文使用以下状态：

- **在线完成**：已进入真实 SGLang 控制路径，并通过对应正确性验证；
- **机制完成**：代码和测试完整，但性能、泛化或真实负载退出条件未闭合；
- **部分完成**：核心代码存在，仍缺少关键接入、信号或实验；
- **Shadow/Replay**：可以生成和比较决策，但不能改变真实请求队列或 KV residency；
- **未实现**：仅存在设计和接口计划。

## 1. 当前最重要的架构结论

BeliefKV 当前有在线 reactive、JointPlan shadow 和默认关闭的 P5 observed-control 三条路径；
它们已共享 visible admission/physical bundle 基础，但尚未统一为正式在线 JointPlan。

```text
在线执行主路径（P2）
RuntimeEvent / request metadata / allocator observation
  -> RCCG + PageOwnershipIndex
  -> admission / fairness / prediction
  -> bundle-aware reactive or shadow planner
  -> ControlCommand queue
  -> RadixArbiter
  -> SGLang scheduler safe point
  -> HiCache physical action
  -> actual-byte ACK
  -> residency commit + telemetry

统一策略研究路径（P2.5/P3）
RCCG + consumer index + PageOwnershipIndex + resource observation
  -> PolicyInputSnapshotBuilder
  -> immutable PolicyInput
  -> B0 reactive baseline / ScenarioPhysicalizer
  -> WhatIfPacker / JointPlanOracle / TraceOrderJointOracle
  -> rolling token-Radix + allocator + queue/service replay
  -> shadow/replay PolicyOutput, O0-O3 and HBM/Host timeline

延后比较路径（P8，默认关闭）
immutable PolicyInput + frozen trace
  -> 按届时论文接口重新实现的有效 baseline
  -> common-denominator native systems when compatible
```

第一条路径已经接入真实 SGLang。第二条路径中的 observed JointPlan 已通过增量 latest-wins
异步 worker 接入 scheduler safe point；完整 `PolicyInput` 由 worker-owned mirror 构造，不再
阻塞 scheduler thread，但仍然是只读 shadow 路径，**尚未替换
`BeliefKVController.tick()` 的在线 admission、迁移和 waiting queue 排序**。

P5A observed active-set admission、P5B selective running-batch retraction 和 P5C
`RETRACT -> physical action -> ACK -> replacement ticket` 已完成 CPU/接口实现并保持默认关闭；
它们尚未通过真实 GPU correctness/performance gate，不能描述为完整在线策略已部署。

第一条路径现在同时记录 BeliefKV 显式 command 与 SGLang native demand-load/write-back 回调。
两者统一进入 transfer timeline，并用 `telemetry_origin` 区分；native 路径仍绕过 BeliefKV
command queue，因此控制归因与数据面完成事件必须分开解释。

因此，当前版本不能描述为“在线 JointPlan 已部署”。准确说法是：

> P2 的 bundle-aware reactive 控制器已经在线；P2.5/P3 已经建立统一策略契约与动态观测；
> P4 observed JointPlan runtime shadow 已接入；P5A--P5C 的 observed-control 机制已实现但默认
> 关闭并等待真实 GPU gate。早期 B1-B4 策略草图已删除，外部 baseline 在系统冻结后的 P8
> 按稳定 workload 和真实 metadata 条件选择。

BeliefKV 不是 metadata-free：它要求 invocation/context identity 和已经发生的 spawn、wait、return、
message、handoff 等最小在线因果事件。其定位是**无需先验完整 DAG、在线发现动态 workflow**，
而不是从完全不透明请求中推断 workflow。

## 2. 当前端到端架构

```text
┌──────────────────────────────────────────────────────────────────────┐
│ A. Agent 与请求事件源                                                │
│ Deep Agents / Codex / Responses / LangGraph / ClawTrace / SGLang    │
│ RuntimeEvent、request metadata、structured action、tool/message 事件 │
└──────────────────────────────┬───────────────────────────────────────┘
                               v
┌──────────────────────────────────────────────────────────────────────┐
│ B. 已观测状态层                                                       │
│ RuntimeCausalContextGraph     ObservedDataConsumerIndex              │
│ ActionFrontierObserver        ContextPrefixAffinityIndex             │
│ PageOwnershipIndex            RuntimeResourceObservation             │
└──────────────────────────────┬───────────────────────────────────────┘
                               v
┌──────────────────────────────────────────────────────────────────────┐
│ C. BeliefKVController                                                │
│                                                                      │
│ 在线 P2：                                                            │
│ causal frontier / fairness / admission / causal lease               │
│ residency / predictor / transfer service curve                      │
│ PhysicalBundleBuilder / ReactiveTransferPlanner / TransferGuard     │
│                                                                      │
│ 研究 P2.5/P3：                                                       │
│ PolicyInputSnapshotBuilder -> B0 / what-if / joint oracle           │
└──────────────────────────────┬───────────────────────────────────────┘
                               v
┌──────────────────────────────────────────────────────────────────────┐
│ D. 物理桥接与执行                                                     │
│ TransferCommandQueue -> RadixArbiter -> SGLangSchedulerBridge       │
│ -> HiCacheNodeCommandBackend -> GPU/Host KV -> ACK/Telemetry        │
└──────────────────────────────────────────────────────────────────────┘
```

BeliefKV 与 SGLang 的职责边界没有改变：

```text
BeliefKV：已观测因果状态、策略、物理 bundle 意图、审计和实验
SGLang：allocator、Radix topology、KV tensor、engine lock 和实际 DMA
```

`PageOwnershipIndex` 只是带 generation 的 CPU mirror，不能替代 SGLang 物理真相源。

## 3. 四个必须分开的状态视图

最新版不再尝试用一棵 workflow 图同时解释所有关系，而是维护四个正交视图。

### 3.1 因果控制关系

[`RuntimeCausalContextGraph`](../beliefkv/control/causal_graph.py) 记录已经发生的
`CALL/SPAWN/RETURN/JOIN/MESSAGE/HANDOFF/TOOL/LLM/REACTIVATE` 事件。它回答：

- 谁创建或阻塞了谁；
- 哪个 return、join、message 会唤醒谁；
- 哪些 invocation 当前为 `READY/RUNNING/WAITING/TERMINAL`；
- 谁位于当前 causal frontier。

RCCG 新增单调 `graph_version`。策略快照可以引用确定版本，避免把基于旧图生成的决策应用到
新状态。重复 `event_id` 保持幂等，原子批处理失败时回滚。

### 3.2 数据消费者关系

[`ObservedDataConsumerIndex`](../beliefkv/control/data_consumers.py) 单独记录 producer-consumer
事实，包括 `RETURN/MESSAGE/BROADCAST/WORKSPACE/HANDOFF`。它回答“谁会读取谁的结果”，而不是
“谁是谁的 parent”。

该索引只保存已观测关系。预测 consumer 不得写入 observed index，也不得伪装成 RCCG 事实。

### 3.3 物理 Prefix 共享关系

[`ContextPrefixAffinityIndex`](../beliefkv/runtime/prefix_affinity.py) 只根据真实共享
`PageHandle` 计算 byte Jaccard 和共享字节。因果 parent-child 不能直接推导 prefix affinity。

真实 P2 workload 已经显示，FRESH parent-child 的公共 prefix 很小，显著复用主要来自相同
template 的 parent-parent 或 sibling child-child。因此当前策略不能把 causal edge 直接转换成
cache-affinity edge。

### 3.4 Action Frontier

[`ActionFrontierObserver`](../beliefkv/runtime/action_frontier.py) 关联：

- structured action 何时成为合法 tool/spawn/handoff/final action；
- 合法 action 出现前后 runnable frontier 如何变化；
- tool-start gap、active-to-waiting KV 和后续 reentry；
- parser 状态为 valid、invalid、incomplete 还是 unknown。

Deep Agents 当前只能上报运行时已经解析完成的 action，不能伪造原生 incremental boundary
token。Action frontier 目前只用于观测和 characterization，不改变 decode order。

## 4. 在线 P2 控制路径

### 4.1 Controller 是组合根

[`BeliefKVController`](../beliefkv/control/controller.py) 在初始化时连接：

```text
RCCG + consumer index + page index
causal frontier + residency + fairness + admission
causal lease + physical bundle builder
predictor + transfer service curve
reactive planner + shadow controller + retry guard
command queue + RadixArbiter
PolicyInput snapshot builder
```

事件入口依次更新 RCCG、observed consumer index、predictor feature、context epoch 和 retry guard。
context 被唤醒时，过时的 shadow 会被取消，并用实际唤醒时间更新 prediction calibration。

### 4.2 当前 `tick()` 的真实顺序

```text
1. 释放 terminal non-persistent context 的语义 ownership
2. 更新 HBM/Host/engine/telemetry 信号
3. 生成可选 remaining-time prediction
4. 检查 pending admission 的 liveness 和 reservation
5. 根据 authoritative HBM 与 native reclaim capacity 作 admission
6. 若无 in-flight command，执行 shortage -> prefetch -> shadow 规划
7. 将命令加入 urgent/shadow queue
8. retry guard 判断当前物理快照是否允许重新尝试
9. RadixArbiter 在派发前重建并校验 bundle
10. 有效命令成为 in-flight；无动作命令生成结构化 local ACK
```

当前 admission、transfer planner 和 SGLang waiting queue 仍各自包含局部排序。这正是 P4/P5
要用统一 JointPlan 消除的问题。

### 4.3 Causal lease 与资源保护

[`CausalLeaseProjector`](../beliefkv/policy/leases.py) 将 RCCG 状态投影为有限资源承诺：

```text
RUNNING > READY > CONDITIONAL_RESUME > SPECULATIVE > DEAD
```

- `RUNNING_LLM` owner 禁止策略迁移；
- `READY` owner 应保留或恢复；
- `WAIT_TOOL/WAIT_CHILD/WAIT_JOIN/WAIT_MESSAGE` 可 shadow，压力下可 commit；
- 未知 context 默认保守保护，不能被当作 dead；
- 共享 bundle 取所有 owner 中的最强 lease。

Lease 不是新的 cache coherency 协议。真正的 lock、位置和 generation 仍来自 SGLang 和
`PageOwnershipIndex`。

### 4.4 Admission 与公平

[`AdmissionController`](../beliefkv/policy/admission.py) 使用：

- 未缓存 prompt 与预计 output 的增量 KV；
- authoritative HBM、已保留 reservation 和 native reclaim capacity；
- root-workflow soft share、有界借用和 attained service；
- workflow 内 causal frontier；
- admission liveness timeout 与 force-progress 条件。

请求只有在真实空间或经过 scheduler 验证的 reclaim 能力满足时才进入 SGLang。依赖 H2D 的
请求需要等待 terminal ACK，并在进入 engine 前重新匹配 authoritative prefix。

## 5. Physical Bundle 与三重校验

P2 的核心变化是把“迁移一个 context”改成“验证并执行一个版本化 physical bundle”。

[`PhysicalBundleBuilder`](../beliefkv/runtime/bundles.py) 负责 preview：

- D2H/COMMIT 包含必须共同处理的 GPU descendant closure；
- H2D 包含 CPU target 到 GPU anchor 的 ancestor closure；
- 共享 extent 只计一次物理字节；
- 区分 `EXCLUSIVE_SUFFIX` 和 `SHARED_SUBTREE`；
- 计算 unique、copy、reclaim、locked 和 foreign-owner bytes；
- 生成覆盖 topology、generation、owner、lease、lock 和 residency 的 fingerprint。

一个迁移命令需要经过：

```text
Planner preview
  -> PhysicalBundleIntent 冻结 handles/actions/fingerprint
  -> RadixArbiter 按 PageOwnershipIndex 二次重建
  -> HiCache backend 按 authoritative allocator 三次 preflight
  -> DMA / COMMIT / rollback
  -> actual-byte ACK
```

任何 generation、parent/children、owner、lock、capacity、closure 或 action bytes 不一致都会
fail closed。Blocked preview 仍进入审计，但不能转换成可执行 intent。

## 6. Command、ACK 与 Residency 状态机

公共协议位于 [`runtime/protocol.py`](../beliefkv/runtime/protocol.py)。

```text
PageHandle = page_id + allocation_generation

GPU_ONLY --START_D2H--> MIRRORING
MIRRORING --shadow ACK--> DUAL_CLEAN
MIRRORING --reactive ACK--> CPU_ONLY
DUAL_CLEAN --COMMIT_CPU--> CPU_ONLY
CPU_ONLY --START_H2D--> PREFETCHING
PREFETCHING --ACK--> DUAL_CLEAN
失败或未完成 action --> 回滚 started transfer
```

`CommandAck` 是正确性边界；`TransferTelemetry` 是性能观测，二者不能混用。只有 ACK 中明确
完成的 handles 才能改变 `PageOwnershipIndex`。Telemetry 只有在对应 correctness ACK 已提交后
才能训练 [`TransferServiceCurve`](../beliefkv/policy/service_curve.py)。

Blocker 已结构化为 closure、capacity、engine busy、lock/loading、inflight、semantic pin、
unsealed、stale generation、extent mutation 和 unknown backend 等类型。

[`TransferAttemptGuard`](../beliefkv/policy/transfer_guard.py) 使用 bundle ID、fingerprint、context
epoch 和 blocker release event 抑制相同失败在每个 scheduler tick 重复提交。它不是永久屏蔽
context；匹配的 allocator、lock、generation、engine 或 runtime 事件到达后才重新放行。

## 7. SGLang/HiCache 数据面

当前只支持：

```text
SGLang tag:    v0.5.2rc1
SGLang commit: 18f91eb639084825717c0e3c3c7273492812ab71
```

[`EmbeddedSGLangRuntime`](../beliefkv/runtime/sglang_v052rc1.py) 在 scheduler safe point 中执行：

```text
drain RuntimeEvent
-> drain ACK and retire H2D dependency
-> drain telemetry/callback errors
-> sync dirty Radix mirror
-> allocator/Radix consistency check
-> report HBM/Host and workflow charges
-> release ACK-satisfied admission
-> Controller.tick()
-> cancel or submit at most one transfer command
-> emit audit, bundle preview, blocker and timing records
```

真实后端的物理单位是 sealed Radix node extent，不是任意固定大小 page。当前 capability 为：

```text
operation_merge = false
layer_completion_events = false
max_inflight_operations = 1
physical_unit = node_extent
```

H2D 使用 `load_back(force=True, allow_eviction=False)`：允许绕过旧版小 closure 阈值，但不能
绕过容量检查或隐式驱逐其他 KV。带 bundle 的 D2H 先完成全部 copy，再按 closure 顺序释放
GPU；H2D 失败时回滚已经恢复的 extent。

## 8. P2.5 统一策略与追踪 Contract

[`PolicyInput`](../beliefkv/policy/reference/base.py) 将所有策略放在同一数据平面：

```text
RuntimeGraphSnapshot
+ runnable frontier
+ disjoint PhysicalKVSnapshot
+ ResourceSnapshot
+ typed optional metadata
+ identity mappings
+ runtime capability report
```

[`PolicyOutput`](../beliefkv/policy/reference/base.py) 统一表达：

```text
ExecutionIntent
+ AdmissionIntent
+ ResidencyIntent
+ TransferDependencies
```

[`PolicyInputSnapshotBuilder`](../beliefkv/policy/reference/snapshot_builder.py) 合并 RCCG、consumer
index、admission queue、physical page mirror、allocator observation、lease 和 service curve。未被
PageOwnershipIndex 跟踪但被 allocator 占用的字节会成为不可迁移 protected bundle，不能被
策略误当成空闲空间。

当前维护代码只保留 B0 same-data-plane policy：

| ID | 实现 | 元数据模式 | 当前状态 |
|---|---|---|---|
| B0 | Reactive baseline | Online | 默认 Shadow/Replay；当前核心 baseline |

`ReferencePolicyAdapter` 会隔离 metadata：online 模式不能读取 hindsight，oracle metadata 只能在
replay 中使用，unsupported action/capability 必须显式输出。当前所有 reference output 强制
`shadow_only=true`，不会改变真实 admission、D2H、H2D 或 waiting queue。
`PolicyReplayRunner()` 和 CLI 只支持 B0；运行时只保存中立物理快照。

2026-07-22 维护清理删除了 B1-B4 具体 policy、B1/B2 hindsight enricher、stateful replay 和
运行时 `program_phase/congestion_feedback` producer。旧实验 JSONL 保留，通用 contract 仍能
校验其中的 decision fingerprint；历史报告不作为当前可执行 baseline。

## 9. P3 动态观测、What-if 与 Joint Oracle

### 9.1 动态 workload 与 instrumentation

当前工作区已实现：

- observed producer-consumer index；
- 单调 RCCG graph version 和 `REACTIVATE`；
- 基于真实 PageHandle 的 context prefix affinity；
- structured action frontier observer；
- Coder/Reviewer/Tester 循环 handoff，并可嵌套 FRESH subagent 的 LangGraph workload；
- 持久 peer context、真实 repository tool loop 和 Deep Agents 动态 task backend；
- topology、cycle、handoff、consumer fan-out 和 action coverage characterization。

CPU/fake backend 已能重复完成 spawn、join、handoff 和 reactivation。真实模型 cyclic/mixed A/B
中的 workload 机制已经跑通：一次 12-workflow 全 mixed run 完成 100 个 LLM call、48 个
FRESH child、40 次 handoff 和 17 次 reactivation。它仍不是配对 A/B，且原生 incremental
boundary-token 覆盖率为 0。

2026-07-22 审计后，上述 12-workflow run 因 leaf one-shot 且 tool call 为 0，已降级为
topology/pressure smoke。新版 `agentic_peer_backend.py` 让 peer 在 handoff 后 RESUME 同一
context，让 FRESH child 在 task 内执行多轮 LLM/tool，并以模型实际 task call 决定 fan-out。
单 workflow GPU gate 已覆盖 253 LLM、239 个真实工具、3 个有效多轮 child、闭合 join 和
cyclic reactivation；另有两条 28/41-request tool-rich run 正常 semantic complete。正式并发
A/B 尚未执行，详见
[P3 真实工具型 Agentic Workload](experiments/beliefkv_p3_agentic_workload_2026-07-22_zh.md)。

### 9.2 Scenario physicalizer 与 What-if

[`ScenarioPhysicalizer`](../beliefkv/policy/scenario_physicalizer.py) 将 blocking、nonblocking、
FRESH、handoff、multi-consumer 和 cyclic reactivation 场景转换成物理需求。尚未创建的预测
request 不能进入实际 execution frontier；consumer readiness 与 physical ownership 分开计算。

[`WhatIfPacker`](../beliefkv/policy/whatif_packer.py) 无副作用地组合 execution order、admission、
required restore 和 victim bundle，并检查 closure、capacity、fairness、liveness 和 handoff
hysteresis。缺少 extent identity 或 closure 重叠时必须 fail closed。

### 9.3 O0-O3 Joint Oracle

[`JointPlanOracle`](../beliefkv/policy/joint_oracle.py) 定义：

```text
O0 current agent scheduling + current KV policy
O1 oracle agent scheduling + current KV policy
O2 current agent scheduling + oracle KV policy
O3 oracle joint scheduling + admission + KV policy
```

Joint synergy gap 定义为：

```text
min(cost(O1), cost(O2)) - cost(O3)
```

Oracle 只有在外部 evaluator 确认重新计算了 queue、service、residency 和 physical action 后才接受
JCT。固定原策略的 wall-clock physical trace 只能比较决策，不能报告反事实 JCT。

当前 O0-O3 数据结构、冻结 request DAG、queue/service resimulator、token-exact tiered Radix 和
rolling allocator 已实现。候选顺序会逐 quantum 重算 cache hit、active lock、unique growth、
GPU/Host residency、D2H/H2D/drop 和 HBM peak。trace-order oracle 对小 DAG 穷尽合法拓扑顺序，
超过预算时显式标记 bounded search。

一个 1,000-token HBM 的真实模型 mixed 单-workflow trace 穷尽 120 个顺序后得到 O0/O1
38,859.629 ms、O2/O3 38,853.685 ms，synergy gap 为 0。该结果是负例而不是收益证据；trace 为
semantic-race-sensitive、没有 workflow fairness 竞争，PCIe 使用配置值。

正式动态并发 GPU trace 已采集并冻结，但旧 rolling O0 与真实 run 严重不对齐：真实 mean
JCT/D2H/H2D 为 589.08 s/0.715 GB/0.189 GB，rolling O0 为 115.66 s/8.814 GB/0。全 mixed
run 又发现 native demand-load 未进入 BeliefKV telemetry，因此当前不能继续报告反事实性能。

## 10. 模块与代码落点

| 层 | 主要代码 | 当前状态 |
|---|---|---|
| Identity/Event | `core/events.py`、`core/ids.py`、`core/config.py` | 在线完成 |
| RCCG | `control/causal_graph.py` | 在线完成 |
| Consumer facts | `control/data_consumers.py` | 机制完成，P3 新增 |
| Online controller | `control/controller.py` | P2 reactive 在线；提供统一 snapshot/control generation，P4 plan 不在此执行 |
| Frontier/Residency/Fairness | `policy/causal_frontier.py`、`residency.py`、`workflow_fairness.py` | 在线完成 |
| Admission | `policy/admission.py` | 在线完成 |
| Lease/Bundle | `policy/leases.py`、`runtime/bundles.py` | P2 在线完成 |
| Transfer policy | `transfer_planner.py`、`shadow_controller.py`、`transfer_guard.py` | Reactive 在线；预测收益待证 |
| Predictor | `predictor/` | 机制完成；跨 workload 泛化待证 |
| Common policy contract | `policy/reference/`、`policy/resource_snapshot.py` | P2.5 完成，Shadow/Replay |
| What-if/Oracle | `scenario_physicalizer.py`、`whatif_packer.py`、`joint_oracle.py` | P3 离线机制完成 |
| Joint runtime shadow | `policy/joint_scheduler.py`、`runtime/joint_shadow.py`、`runtime/sglang_v052rc1.py` | P4 immutable delta + worker mirror 已接入；只校验和审计，不执行 |
| Physical mirror/arbitration | `runtime/page_index.py`、`runtime/radix_arbiter.py` | 在线完成 |
| SGLang backend | `runtime/sglang_adapter.py`、`runtime/sglang_v052rc1.py`、`patches/` | 显式 command 与 native demand-load/write-back telemetry 已接入；真实 GPU 覆盖待复验 |
| Agent event adapters | `runtime/event_channel.py`、`deepagents_adapter.py`、`codex_adapter.py` | Deep Agents/跨进程路径已验证；更多 framework 待接 |
| Action/prefix observation | `runtime/action_frontier.py`、`prefix_affinity.py` | P3 观测机制完成 |
| Trace | `traces/normalizer.py`、`runtime_validation.py`、`characterization.py` | 标准化/验证完成；P3 characterization 新增 |
| Simulator | `simulator/queue_service.py`、`token_radix.py`、`rolling_physical.py`、`rolling_queue_service.py` | rolling token/allocator 机制完成；真实 extent/PCIe/fairness gate 待闭合 |
| Experiments | `experiments/deepagents_swebench.py`、`langgraph_peer_workflow.py`、`policy_replay.py` | 12-workflow mixed characterization 已跑；配对 A/B 待跑 |
| Metrics | `metrics/transfer_timeline.py`、`transfer_validation.py` | 显式/native DMA 统一时间线，物理 lock/closure/migratable/dual-resident 及 100/500 ms locked-but-not-served 下界曲线已接入；真实 GPU coverage 待复验 |

## 11. P0-P8 实施状态

| Phase | 状态 | 已完成 | 仍缺少 |
|---|---|---|---|
| P0 Correctness baseline | 在线完成 | RCCG、admission、page mirror、ACK、trace/audit 基线 | 持续回归 |
| P1 Telemetry | 部分完成 | HBM/Host snapshot、显式与 native demand-load/write-back telemetry、统一 timeline | compute wait、copy-engine/PCIe/GPU utilization 的原生观测 |
| P1.5 Retry guard | 机制完成 | typed blocker、event-gated release、retry storm 消除 | 同 manifest 性能配对 |
| P2 Physical bundle | 可靠性 gate 通过 | lease、preview、fingerprint、atomic preflight/commit/rollback | P1.5/P2 配对性能、service curve 尾部、controller timing |
| P2.5 Common policy contract | 完成 | immutable PolicyInput/Output、metadata 隔离、B0 replay | P8 按需增加有效 baseline |
| P3A Dynamic instrumentation | 部分完成 | consumer/action/prefix/topology、required-range fan-out、固定时长 GPU-ready probe | 稳定 ready 并发、配对 A/B、fanout 多样性、boundary-token coverage |
| P3B Jointness analysis | 部分完成 | B0、what-if、bounded search、rolling Radix/allocator、HTML timeline | 从真实 pressure snapshot 做局部 Jointness Audit；完整动态 O0-O3 后置 |
| P4 JointPlan shadow | CPU 修复完成，GPU 复验待执行 | visible ticket、无 reservation restore、增量 Page/RCCG journal、lossless-coalescing worker mirror、root-workflow 公平快照、per-action current-state validation、anytime best-prefix、分阶段 planner budget、增量 bundle/lease/closure rebuild、bounded snapshot/audit、native DMA telemetry、locked-but-not-served observer、activation wall-clock guard；Host pool 默认扩到 96 GB，Host 生命周期含 terminal cleanup、Host-copy eviction 与容量饱和淘汰 | 真实 GPU safe-point/stale/coverage/终止性、Host 字节一致性和 lock-service 归因覆盖率复验；run-to-yield 留到 P5 |
| P5 Online observed JointPlan | P5A--P5C CPU/接口完成，默认关闭 | observed active-set ordering、active-KV high-watermark、lock-provenance blocker-set retraction，以及 `RETRACT -> physical OFFLOAD/DROP -> ACK -> replacement ticket` 跨 epoch 事务 | 真实 GPU correctness/performance gate；更广泛的 ExecutionIntent 与 cyclic-peer hysteresis |
| P5.5 Action-unlock gate | 未执行 | observer 基础存在 | 相对 B0/O1/O2 的真实 coverage/gap |
| P6 Predictive JointPlan | 未实现 | 旧 remaining-time predictor 可复用 | Unlock/Reentry hazard、scenario calibration、在线 gate |
| P7 New HiCache portability | 未实现 | 固定旧版本 contract | 新版 adapter 与能力协商 |
| P8 Deferred competitors | 未执行 | related-work 与通用 trace contract 已建立 | 按稳定接口实现 metadata 分层、same-data-plane 与原生系统公平对比 |

## 12. 已有真实证据

### 12.1 当前代码测试

2026-07-22 对当前工作区执行：

```text
conda run -n beliefkv pytest -q
326 passed, 7 skipped

conda run -n beliefkv-agents pytest -q \
  tests/test_deepagents_adapter.py \
  tests/test_deepagents_swebench.py \
  tests/test_multi_agent_runtime.py \
  tests/test_counterfactual_trace.py
60 passed

conda run -n beliefkv-agents pytest -q tests/test_multi_agent_runtime.py
15 passed
```

源码同时通过 `py_compile` 和 SGLang source contract；固定版本为 `0.5.2rc1`、commit
`18f91eb639084825717c0e3c3c7273492812ab71`。

### 12.2 P2 真实 GPU 可靠性复验

2026-07-21 的 Qwen3-Coder-30B-A3B-Instruct-FP8、RTX 6000 Ada、SGLang 0.5.2rc1
高压复验得到：

| 检查项 | 结果 |
|---|---:|
| workflow system terminal state | 8 / 8 |
| request started / finished | 848 / 848 |
| transfer dispatch / ACK | 1502 / 1502 |
| missing/orphan/order/byte violation | 0 / 0 / 0 / 0 |
| watchdog / scheduler exception | 0 / 0 |
| identical failed/zero-byte retry | 0 / 0 |
| dispatch without matching preview | 0 |
| HBM mirror exceeds allocator | 0 / 38,594 snapshots |
| Host page-index mismatch | 0 / 38,594 snapshots |
| offload planned/actual reclaim | 52,851,376,128 / 52,851,376,128 bytes |
| reclaim realization | 100% |

该结果证明 **P2 物理可靠性 gate 已通过**，不证明性能提升。

### 12.3 尚不能从该实验得出的结论

- 只有 3/8 workflow 通过本地任务 correctness gate，其余为 blocked/no-patch-needed；
- 没有同代码、同 manifest、同语义路径的 P1.5/P2 配对 run；
- admission P95 仍为 61.46 s，不能归因或宣称改善；
- service-curve 总体低估率为 13.48%，D2H 为 23.53%，尾部仍不够保守；
- SIGINT 路径未写出 controller timing summary；
- 真实 mixed multi-agent characterization 已执行，但没有配对 A/B 和重复实验；
- 单-workflow rolling O0-O3 的 synergy gap 为 0，尚无多-workflow 正结果；
- 12-workflow rolling O0 与真实 JCT/transfer 明显不一致，不能进入性能表；
- SGLang native demand-load 不在现有 transfer telemetry 中，当前 H2D timeline 是下界；
- action-unlock synergy 尚无真实 gap 结果。

### 12.4 P3 Queue/Service 标定

同模型、同 SGLang 配置的 39-request microbenchmark 覆盖单/多 chunk prefill 和 decode batch
1/2/4。`episode_piecewise_isotonic_v1` 在独立 holdout 上得到总体 P95 relative error 20.09%，
prefill 4.51%，decode 23.49%，通过 25% 门槛。该结果只关闭 GPU service-time 子问题；旧
agent trace 的 policy-dependent cache hit 与 future physical growth 原先不满足 timing gate。
rolling replay 已能在完整 token trace 和空 Radix epoch 下重算候选 cache hit 与 growth，但
尚未替代真实 page/node extent、PCIe holdout 和多-workflow fairness 证据。

### 12.5 P3 动态并发 GPU Characterization

12 个全 mixed workflow 在 163,840-token KV pool 下全部 semantic complete，峰值 HBM 为
96.93%。trace 含 48 次 SPAWN、12 次 JOIN、40 次 HANDOFF 和 17 次 REACTIVATE。BeliefKV
显式路径的 14 次 D2H、8 次 H2D 全部完成，且没有 partial/reject/retry storm。

该 run 同时否决了两个过强假设：一笔 join 后无消费者 H2D 占显式 H2D 的 37.52%；三个已
offload parent 通过 SGLang native demand-load 恢复，却没有 BeliefKV H2D telemetry。因此该
实验支持统一 observed-state control 的研究动机，但不支持当前性能结论。完整报告见
[P3 动态并发 GPU Characterization](experiments/beliefkv_p3_dynamic_gpu_validation_2026-07-21_zh.md)。

### 12.6 P3A 固定时长 GPU-ready 探针

2026-07-22 只执行了一次 240 秒 required-range 探针：4 个 mixed workflow 分两批到达，初始
coder 必须运行时创建 2--4 个 child。实际 4 个 workflow 各创建 4 个 child，系统瞬时达到
16 个 running request；但按 snapshot 时间区间积分后，平均 running request 仅 4.03，
`running <= 2` 占 68.13%，`running >= 8` 仅占 24.36%。同一窗口 GPU 利用率平均 26.77%，
低于 20% 的时间占 64.17%。

低 ready 并不等于低显存压力：`running <= 1` 的区间占 65.09%，其中 HBM KV 平均仍为
95.62%。267 个 SGLang batch log 观察点的 queue 均为零。因此 required-range 机制 gate
已通过，但 simultaneous fan-out 只能形成 burst，**稳定 GPU-ready 并发 gate 未通过**。
稳定 GPU-ready workload gate 仍停留在 P3A；P4 只能推进无副作用 runtime shadow 机制，
不得据此进入 P5 或报告性能收益。完整边界与数据见
[P3A GPU-ready 并发探针](experiments/beliefkv_p3_gpu_ready_probe_2026-07-22_zh.md)。

### 12.7 P4 Runtime Shadow 接入

2026-07-23 已将原混合异步路线改为增量 mirror：scheduler safe point 只复制自上次发布以来的
RCCG event、PageIndex replacement record、当前 waiting request/resource/fairness/control 值；容量
1 的 worker 顺序合并所有 delta，并在独立 RCCG、consumer、PageIndex mirror 上构造 immutable
`PolicyInput` 和 observed `JointPlan`。pending sequence 可以 latest-wins，但其 delta 被合并而非
丢弃，审计分别报告 `coalesced_pending_count` 与真正的 `dropped_pending_count`。

P4 对 fully-fresh plan 只记录 `joint_plan_would_apply`；部分 action 仍有效时记录
`joint_plan_shadow_partial`，全 stale、过期或 worker failure 均不触发同步完整规划，也不修改
admission、waiting queue、transfer queue 或 physical residency。source snapshot 使用完整
`PlanReadSet`；结果返回后按 execution、每条 admission、每条 residency 和每条 dependency 定点读取
当前 request/invocation/join/transition/touched extent 并独立校验。graph/topology/allocator 全局 stamp
变化仅作为 `strict_global_stale` 对照指标；实际公平顺序未变、无关 request/extent/transfer 变化不会
淘汰整份计划。目标 extent 会重算 lease、物理 blocker 和 descendant closure，缺失或无法解析时
fail closed。审计已覆盖每类 component 的 valid/invalid 数量与原因，以及 delta capture、worker
snapshot build、queue/compute、publish、validation、plan age、coalesce/drop 和 coverage。

全量 CPU 回归为 `344 passed, 7 skipped`，其中测试显式禁止增量 runtime 调用 live controller
完整 snapshot builder，覆盖 worker busy 时连续 delta 无损合并、单 request/bundle 局部失效、
fairness revision 前进与实际 priority 翻转，以及目标 extent bundle 与完整 snapshot 一致性。该节点
只证明接入正确性。真实 GPU
safe-point P99、GIL 干扰、plan age、drop/stale rate 与 would-apply coverage 尚未测量，因此 P4
实验 gate 仍未通过。

### 12.8 Locked-but-not-served 观测

2026-07-23 已加入只读的 `RequestServiceLedger`。SGLang batch 被选中仅建立 request 身份和
首次观测时间；只有 `process_batch_result` 完成后才记为一次真实 GPU service，因而不会把
waiting/running 状态本身误当成获得 token service。资源采样时，adapter 从每个 tagged running
request 的 `last_node` 沿父链走到 Radix root，并与 `engine_lock_ref > 0` 的物理 extent 做精确
关联。共享 prefix 在物理指标中只计一次，另行保留按 request 重复的 logical bytes 供诊断。

观测分别报告 100 ms 和 500 ms 窗口。只有 extent 的全部 `engine_lock_ref` 都能由当前运行
request 路径解释，并且所有 blocker 均已超过窗口未完成 service 时，才计入
`locked_but_not_served_gpu_bytes_*`。部分归因、缺失路径、额外 engine lock 均归入 unknown；初次
进入 batch 但尚未超过窗口的 request 归入 warming。因此该数字是保守的物理字节下界，不是
“当前可立即迁移字节”。PageIndex 复用同 revision 的 physical breakdown 缓存，新增热路径只遍历
当前 running request 的 Radix 祖先链和已缓存的 locked extent，不增加第二次全树扫描。

该 observer 不生成 ticket，不重排 waiting queue，不 retract running batch，也不改变 KV
residency。时间线新增两个下界曲线、完整归因覆盖率和 stale/engine-lock sample ratio。全量 CPU
回归为 `371 passed, 7 skipped`；尚未运行 GPU characterization，因此该节点自身不能证明锁策略
过于保守。

### 12.9 Observed Active-Set Admission

2026-07-25 已完成 P5A 的 admission-only 在线切片。该路径不消费异步 JointPlan，也不运行
physicalizer/packer；它在 SGLang 每个 batch-construction epoch 使用当前 observed state 决定哪些
tagged waiting request 获得短期 ticket。当前 active KV footprint 定义为：

```text
PageIndex 中 engine-lock/active-reader 保护的 Radix 物理唯一字节
+ running/chunked request 中尚未进入 matched prefix 的私有 KV 字节
```

策略在 `(KV pool - reserve) * active_kv_high_watermark_ratio` 上建立 active-set 高水位。普通新增
request 的 `uncached prompt + remaining max output` KV 上界必须放入剩余 headroom，并继续接受
SGLang `PrefillAdder` 的原生 token、slot 和 allocator 校验。若 running request 数低于
`observed_admission_min_active_requests`，只允许补足该固定 floor，且仍不超过原生可用 HBM；这是
唯一可绕过高水位的 work-conserving 路径。等待超时只改变候选顺序，不绕过容量 gate。

候选排序只使用已观测 causal class、unblock depth、workflow virtual runtime、workflow 内
frontier round、等待时间和增量 KV；不预测 tool duration、下一 agent 或剩余 workflow。公平是
软排序，同一 workflow 可在一个 epoch 获得多个 ticket。无 metadata request 完全绕过该 gate；
`WAIT_RESTORE`、terminal 和 transition-open 继续由 side state fail closed。active-set 计算异常时
审计 `observed_admission_fallback` 并回退原 reactive compiler，显式 blocker 不被解除。

配置项为 `observed_admission_scheduling_enabled`、
`observed_admission_active_kv_high_watermark_ratio` 和
`observed_admission_min_active_requests`，默认关闭。审计记录每个 epoch 的 active budget、footprint、
headroom、Radix lock、running-private bytes、policy/native HBM budget、mode 和 ticket 结果。全量
CPU 回归为 `377 passed, 7 skipped`。

该切片只阻止产生新的 lock owner，不能释放已经运行的 request 或其 8--14 GiB 锁路径。因此其
预期结果是降低后续 lock-footprint 墾殖，不是单独解决既有 convoy。尚未执行真实 GPU gate，当前
不能宣称 JCT、GPU utilization 或 admission tail 改善；高水位 `0.8` 只是初始实验参数，必须用
固定 trace 做 sensitivity sweep，而不能作为经验常数写入论文结论。

### 12.10 Observed Selective Running-Batch Retraction

2026-07-26 已完成默认关闭的 P5B CPU/接口切片。固定版 SGLang 新增
`ScheduleBatch.retract_selected(request_ids)`；它只在 scheduler safe point 释放被选请求的
request-private KV 和 `last_node` lock，不调用额外的 Radix LRU，也不替策略选择其他 victim。原生
allocator shortage retraction 保持不变，仍作为最终 liveness fallback。

BeliefKV 只在 observed admission 持续无 ticket、active KV 超过高水位或最高优先级 replacement
超过当前 `available + evictable` 容量时考虑 retraction。planner 从 100/500 ms service ledger、RCCG
causal class、root-workflow virtual runtime 和完整 lock provenance 构建候选；一个 extent 只有
`engine_lock_ref == |blocker set|` 且全部 blocker 同时被选择时才计入 expected unlock。选择过程按
blocker package 求解，不把共享 extent 重复计费，并至少保留一个 running request。任一 blocker
路径含 semantic pin、active reader、in-flight transfer 或 unsealed extent 时，该 request fail closed，
不能成为主动 retraction victim。

执行使用版本化事务：

```text
observed admission stall
  -> blocker-set plan
  -> selective native retraction
  -> force Radix/PageIndex rematch
  -> measure allocator available delta (not available + evictable)
  -> if insufficient, rematch exact physical closure
  -> explicit OFFLOAD_CONTEXT or DROP_CONTEXT
  -> physical ACK and allocator-free confirmation
  -> confirmed: one-epoch replacement priority
  -> partial/rejected/stale/timeout: release barrier without priority
```

被 retract 的请求以 `retraction_cooldown` 留在 SGLang 原生 waiting queue，避免同一 epoch
re-admission；其逻辑 output token 保留，但未缓存 private KV 在恢复时需要重算。只有实际
allocator `available` 增量达到 reclaim target，且当前绝对 free bytes 足以覆盖首个 replacement 时，
replacement 才获得一次性优先级。`evictable_size` 只用于 retraction 机会识别，不能作为事务提交证据。

2026-07-26 的 P5C 增量把新解锁 closure 编译为版本化 physical bundle。事务只豁免该次被 retract
context 的逻辑 RUNNING lease；engine lock、active reader、semantic pin、in-flight、foreign active owner
仍 fail closed。优先选择 exclusive suffix；共享 bundle 仅在全部 owner 都属于同一 blocker set 时可选。
Host 足够时执行显式 D2H/COMMIT_CPU；`DROP_CONTEXT` 保留 dual-clean Host copy，GPU-only drop/recompute
默认关闭，只有显式配置才允许。所有 tagged waiting request 在 physical ACK 前受 transaction barrier
约束；ACK partial/rejected/stale 或 5 秒事务超时后一次失败退出，不做盲目重试。全量 CPU 回归为
`389 passed, 7 skipped`；真实 GPU correctness、recompute、thrash、JCT 和 utilization gate 尚未执行。

2026-07-27 增加只读 `TentativeUnlockPreview`。PageOwnershipIndex 接受临时 engine-lock ref override，
在不修改 page、revision、Radix topology 或物理 breakdown cache 的情况下重新计算 descendant closure，
输出 lock-ref 归零字节、首次变为 migratable 的物理唯一字节和 closure amplification。request blocker
映射只在 `engine_lock_ref == |完整 blocker set|` 时应用；路径错误、缺失/重复 extent 和部分归因均
标记 `provenance_incomplete` 并保守保留原 lock。barrier 前记录所有 observed-stale blocker 的
unconstrained upper bound，安全点选出 plan 后记录 selected-set preview，并在真实 retraction callback
记录 realized delta 和误差。两条 preview 当前均为 `shadow_only`，不跳过 barrier、不改变 victim、
admission 或迁移事务；汇总记录 reason/exactness 和计算开销 P50/P95/P99。全量 CPU 回归为
`403 passed, 7 skipped`，GPU 外部有效性尚未验证。

## 13. 必须维持的系统不变量

1. RCCG 只保存已观测因果事实，不拥有或释放物理 KV。
2. causal edge、consumer edge 和 physical prefix owner 必须分开维护。
3. SGLang 是 allocator、Radix topology、KV tensor 和 DMA 的唯一物理真相源。
4. shared physical extent 只计费一次，并由最强 owner lease 保护。
5. admission 使用实际 HBM、reservation 和 ACK，不把计划释放当作已释放。
6. context epoch、allocation generation、graph/topology/allocator version 防止 stale action。
7. bundle preview 只是版本化意图，arbiter 和 backend 必须在执行前重新验证。
8. active reader、engine lock、semantic pin、in-flight 和 closure blocker 不能被预测绕过。
9. ACK 前不提交 residency；telemetry 不能替代 correctness ACK。
10. online policy 不能读取 hindsight metadata；B0 replay 保持 shadow-only。
11. 未创建的预测 agent/request 不能被实际调度或获得物理资源。
12. predictor/OOD 失败必须退化到 observed reactive 路径，而不是破坏 liveness。
13. 无 `beliefkv_metadata` 的请求必须保持上游 SGLang 行为。
14. cache reset 和 abort 必须先清理 in-flight bookkeeping，再失效 page handle。
15. BeliefKV command telemetry 与 native HiCache operation telemetry 必须分源记录并统一计费；
    command integrity 通过不能替代全系统 DMA coverage。

## 14. 当前关键缺口与优先级

建议下一阶段按以下顺序推进：

当前主路径是尽快闭合 P5A--P5C 的端到端 GPU correctness、控制开销和基本性能采样，先得到一套
可完整运行的 BeliefKV。运行时同步采集 GPU-ready concurrency、locked-but-not-served HBM-time、
admission wait、blocker-set、preview/realized reclaim 和 recompute 成本；不把复杂 O0-O3 动态模拟
或外部 baseline 适配作为完整系统搭建的前置条件。P5A 只是 active-set baseline/minor mechanism，
P5B/P5C 的 causal replacement + physical blocker-set 事务能否成为核心贡献由后续证据决定。

1. **闭合 P4/P5 在线 gate**：继续优化 snapshot/planner/validation budget 和 workload 终止性，
   完成 P5A--P5C 的真实 GPU correctness、控制开销和基本性能运行。完整 JointPlan 若最终无法满足
   critical-path overhead、freshness 和 actionable coverage，再收敛为 Local Frontier JointPlan，
   不通过持续放宽 TTL 掩盖失败。
2. **验证 native HiCache telemetry**：request-admission demand-load 和 native write-back 的回调、
   与 BeliefKV command DMA 去重及完整 HBM/Host/PCIe timeline 已实现并通过定向 CPU 测试；下一次
   固定 GPU gate 验证真实 callback coverage。资源图已将误导性的 `Protected` 改为
   `Untracked allocator delta`，并增加 engine-locked、closure-blocked、migratable 和
   dual-resident physical KV。
3. **修复复合事件可见性**：同一 runtime step 的 join/target-create/handoff 原子交付；无法
   原子交付时标记 transition-open，禁止 transient READY 触发不可撤销 prefetch。
4. **闭合可终止的动态 workload**：保留 blocking、cyclic peer 和 mixed 结构，为单次 activation
   加入语义完成与墙钟/调用预算，使正式 GPU run 能产生可解释的 RETURN/JOIN 和 JCT。
5. **闭合 P2 性能 gate**：使用同 manifest 重跑 P1.5/P2，补 controller timing 和更保守的
   transfer service curve。
6. **采集局部 Jointness Audit**：只在真实 trace 中出现多个 READY agent、HBM/PCIe pressure 和
   可选 physical action 的决策点保存短窗口快照；统计 execution/KV 双向决策反转和 local synergy。
   语义分支不可识别的样本明确排除，不要求现在运行完整动态 O0-O3 模拟。
7. **后置 rolling replay 深化**：系统路径稳定后再按 overlap episode 重建 GPU service model，
   对齐真实 page/node extent、PCIe、request-private allocation 和 fairness；它不阻塞 P5 完整实现。
8. **只在 P5.5 成立后扩展 P6 predictor**：否则保留 observed JointPlan，不强行增加预测故事。
9. **最后做新版 HiCache 和 P8 竞品对比**：系统和 workload 冻结后，再按真实 metadata
   条件实现有效 adaptation、oracle 与可运行原生系统。

当前最重要的研究问题已经不是“是否能迁移 KV”，而是：

> 在真实动态 subagent/multi-agent workload 上，联合 execution、admission 和 KV 决策是否
> 稳定优于 B0 以及最强的独立 Agent 调度或独立 KV oracle。

长上下文的语义总结、tool-output 压缩和 checkpoint 由 agent 业务层负责。固定 SGLang 仅提供
sliding-window、长度限制和物理 KV eviction，不提供 agent-aware 自动压缩；BeliefKV serving 层
只负责 context-growth admission 与到 yield/terminal 边界的调度。

## 15. 文档权威顺序

建议按以下顺序理解当前版本：

1. 本文：当前代码与阶段状态总览；
2. [`beliefkv_hicache_joint_control_improvement_plan_2026-07-18_zh.md`](beliefkv_hicache_joint_control_improvement_plan_2026-07-18_zh.md)：
   P0-P8 实施方案和 Go/No-Go 条件；
3. [`experiments/beliefkv_p2_physical_bundle_2026-07-19_zh.md`](experiments/beliefkv_p2_physical_bundle_2026-07-19_zh.md)：
   P2 bundle 实现与 2026-07-21 真实复验；
4. [`experiments/beliefkv_p3_dynamic_gpu_validation_2026-07-21_zh.md`](experiments/beliefkv_p3_dynamic_gpu_validation_2026-07-21_zh.md)：
   P3 12-workflow mixed characterization、迁移反例与 telemetry 边界；
5. [`beliefkv_dynamic_agent_workflow_considerations_2026-07-20_zh.md`](beliefkv_dynamic_agent_workflow_considerations_2026-07-20_zh.md)：
   动态 workflow、consumer 和 prefix 关系边界；
6. [`related_work_comparison_2026-07-21_zh.md`](related_work_comparison_2026-07-21_zh.md)：
   B0、P8 候选 baseline 和竞争边界；
7. [`architecture.md`](architecture.md) 和
   [`beliefkv_design_2026-07-14_zh.md`](beliefkv_design_2026-07-14_zh.md)：
   基础控制面/数据面原则与历史设计背景。

`figures/beliefkv_architecture_status.*` 和 `figures/beliefkv_phase_status.*` 仍是
2026-07-15 的历史图，未覆盖 physical bundle、retry guard、P2.5 contract 和 P3。重新生成新版
图源之前，不应再将这些图片作为当前状态依据。
