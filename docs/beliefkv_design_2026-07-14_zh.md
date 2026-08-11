# BeliefKV 当前系统设计

日期：2026-07-14；最后更新：2026-08-11

状态：P5 observed-state JointPlan、迁移事务和 restore liveness 已通过定向正确性验证；P6 最新 R0--R5 的代码路径、fan-out workload、单动作 gate 和配对 A/B 基础设施已经实现，GPU gate 尚未全部执行。当前服务模型首版只以 bytes、extent count 和 contention 为条件，extent-size distribution 与 closure depth 尚未进入模型。2026-08-10 的单 GPU 受控实验发现：相同 2.659 GB KV 在 106 个 extents 下的 D2H 均值为 765.17 ms，在 7 个 extents 下为 185.69 ms，相差 4.12 倍。早期 promotion/veto 结论因 service-contract 错误和固定 trace 上无稳定 action flip 已失效；morphology 不再是独立策略，仅作为统一 transfer cost/OOD guard。预测式 PREPARE、Frontier-Aware Retraction 和端到端收益仍需最新版 GPU gate 证明。

本文档取代 `technical_archive_2026-07-10.md` 作为当前设计的权威说明。旧文档保留为历史讨论记录，其中的 flat next-action predictor、静态 belief frontier 和以 `agent_id` 为核心的元数据设计不再代表当前方案。

2026-08-11 之后的 P6 可执行顺序以
[`beliefkv_p6_predictive_joint_execution_plan_2026-08-11_zh.md`](beliefkv_p6_predictive_joint_execution_plan_2026-08-11_zh.md)
为准：删除 morphology 独立策略，取消独立 oracle 前置 gate，增加受控 2--3 child fan-out，并将
FrontierBelief 作为现有 admission、KV 和 selective retraction 的统一 JointPlan 注解。

## 1. 研究目标

BeliefKV 面向以下场景：

- 单 GPU、HBM 受限；
- 多个 agent workflow 并发；
- workflow 在运行时动态产生工具调用、subagent、handoff、消息和 join；
- 不要求用户预先提供完整 agent DAG；
- KV cache 可以位于 GPU、CPU，或者被丢弃后重算；
- GPU 侧使用 SGLang RadixCache/HiCache 管理物理 KV 页面。

BeliefKV 的目标不是单纯提高单个 LLM request 的吞吐，而是降低：

```text
workflow completion time
parent/subagent resume stall
message-driven agent wake-up stall
HBM admission stall
关键路径上的 D2H/H2D 时间
无效 KV 迁移和重复计算
```

Agent workflow 不面向人类逐 token 阅读，因此 human-reading-rate 类型的 TPOT SLO 不是默认目标。系统仍记录 TPOT 作为底层干扰指标，但调度目标主要是 workflow progress、尾部完成时间和公平性。

## 2. 核心结论

当前设计建立在以下结论上。

### 2.1 不依赖预定义 DAG，但必须有运行时事件

“不依赖用户传入 DAG”不等于“完全不需要 workflow 信息”。BeliefKV 必须能够观察已经发生的：

```text
CALL / SPAWN / RETURN
WAIT_CHILD / WAIT_JOIN
MESSAGE / HANDOFF
TOOL_START / TOOL_END
LLM_START / LLM_END
```

这些事件由 agent runtime、tool dispatcher、future/join runtime 和 LLM gateway 自动捕获，而不是由模型在 prompt 中自报。

完全黑盒、只向 SGLang 提交无关联请求的应用无法被在线准确恢复出父子和通信关系。此时只能使用保守的 request-level 策略。

### 2.2 不为应用设置全局 subagent/multi-agent 模式

真实应用可以同时包含对等协作和嵌套 subagent：

```text
Planner <-> Coder <-> Reviewer
              |
              +-> Debugger
                    |
                    +-> Searcher
```

BeliefKV 按运行时边类型选择策略：

```text
CALL + RETURN       continuation/subagent 策略
MESSAGE + BARRIER   对等 multi-agent working-set 策略
FORK/RESUME/PREFIX  物理 KV 共享策略
```

### 2.3 事件驱动策略是正确性和性能下界

系统在没有预测器或预测器 OOD 时必须正常运行。预测器只用于：

- 预测性 CPU shadow copy；
- 提前 prefetch；
- 多个可迁移对象之间的排序；
- future fanout 和未来资源压力的可选增强。

实际释放 GPU KV 的 COMMIT 动作仍由真实 HBM pressure、admission 或 wake-up 事件触发。

### 2.4 RCCG 和 Radix tree 是正交结构

Runtime Causal Context Graph 描述 context 的控制依赖和未来价值；SGLang Radix tree 描述 token prefix 的物理共享。二者之间是多对多关系，必须通过 page ownership/lineage bridge 连接。

### 2.5 Agent 因果窗口必须使用可靠的物理迁移成本

仅按 KV bytes 估算传输成本是不充分的。一次 context 级动作最终会物化为 Radix
closure，其成本还取决于 extent/page 数量、extent 大小分布、closure 结构、Host
copy 状态和并发传输条件。动态 agent workflow 则通过 `TOOL_WAIT`、`WAIT_CHILD`、
`WAIT_JOIN` 和 message dependency 提供动作可以被隐藏的因果窗口。

因此 P6 不再把“未来会不会再次调用该 agent”直接映射为 offload/prefetch，而是联合计算：

```text
agent/RCCG state       -> causal slack
Radix physical closure -> morphology debt

morphology_slack = min(pressure_deadline, reentry_deadline)
                  - Q90(transfer_time | physical_shape, contention)
                  - safety_guard
```

只有 `morphology_slack > 0`，且动作净收益、未来 HBM 和物理 certificate 均成立时，
JointPlan 才能选择预测动作。预测 OOD、形态超出服务模型支持域或 safe point 重新物化
后余量转负时，系统退化到 P5 observed policy。

当前 GPU0 证据仅证明“物理成本估计不能只依赖 bytes”：相同 2,659,221,504 bytes
下，7-extents 与 106-extents 的 D2H 均值分别为 185.69 ms 和 765.17 ms，三次重复
方向一致。修复 transfer service sample-gate 后，同一冻结 trace 中该成本差异没有改变
eligibility 或 selected action，因此 morphology 只保留为 transfer cost/OOD safety，不再
承担核心创新主张。它也尚未证明 extent count 与成本的通用函数关系，或该差异由 agent
workflow 独有地造成。测量协议、
原始目录与证据限制见
[`beliefkv_p6_d2h_overlap_characterization_2026-08-09_zh.md`](experiments/beliefkv_p6_d2h_overlap_characterization_2026-08-09_zh.md)。

### 2.6 Transfer service 与 native ownership 是强数据契约

P6 的收益判断依赖 transfer artifact，但在线观测与离线校准不能共用一个隐式样本门槛：

- `runtime_min_samples` 只约束当前 server 在线积累的样本；
- `artifact_min_samples` 由校准 artifact 自己声明并约束 warm-start 样本；
- 加载 artifact 后必须至少有一个代表性 query 仍由校准证据支持；
- hardware key、两个门槛、加载样本数、bucket 数和 supported query 数写入
  `runtime_initialized.transfer_service_contract`；
- 契约不成立时在 workload 启动前 fail fast，不能静默退化到静态 PCIe 带宽。

native HiCache D2H/H2D 的 ownership 必须在 transfer submit 时冻结。完成回调时查询 live
Radix tree 会受到 context unbind、node mutation 和 ID reuse 影响。新 telemetry 保存：

```text
owner_context_ids / owner_context_epochs
extent_bytes
ownership_revision
ownership_attribution_semantics = submit_snapshot
```

旧 trace 中 1,922 次 native D2H 均缺少 owner attribution，不能用于判断某个 parked parent
是否真正成为后续 reactive write-back victim，也不能据此计算 useful-shadow oracle 收益。
旧记录只保留为总带宽/HBM 压力证据；需要新 trace 才能建立 context-level 因果归因。详细记录见
[`beliefkv_p6_service_contract_and_native_ownership_2026-08-11_zh.md`](experiments/beliefkv_p6_service_contract_and_native_ownership_2026-08-11_zh.md)。

## 3. 非目标和已否定方向

以下方向不作为当前核心创新：

- 仅把 tool observation 暂存在 raw text，推迟 prefill；
- 按自然语言语义把 Transformer KV 拆成可独立删除的片段；
- 只预测下一个 agent/subagent；
- 只按 agent role 或 agent name 设置 KV 优先级；
- 只做相同 token prefix 的合并；
- 只在 child return 后立即删除 KV；
- 仅根据平均工具时长做 D2H/H2D break-even；
- 让用户手工标注完整 workflow DAG、关键路径或 subagent 身份。

SGLang 已提供 Radix prefix sharing 和 HiCache write policy。BeliefKV 的设计必须超出通用 LRU、write-through 和 prefix hit count。

## 4. 术语和身份模型

BeliefKV 必须区分四种身份：

```text
agent_definition_id  agent 的角色/工具/系统提示定义
agent_instance_id    一个持久 agent/session 实例
invocation_id        一次具体激活、调用或执行
context_id           一条逻辑 token/KV 序列
```

同一个 `searcher` 可以被调用多次，其中两次是 fresh context，另一次 resume 旧 context。只使用 `agent_id` 会把这些调用混在一起。

每个 invocation 最少携带：

```text
root_workflow_id
invocation_id
parent_invocation_id
context_id
parent_context_id
relation_type        CALL | SPAWN | MESSAGE | HANDOFF
context_mode         FRESH | FORK | RESUME
execution_mode       FOREGROUND | BACKGROUND
return_target_id
join_id
context_epoch
```

其中 `parent_invocation_id` 表示控制关系，`parent_context_id` 表示 KV 上下文谱系。fresh subagent 有控制父节点，但不一定有物理父 prefix。

## 5. 总体架构

当前实现与设计缺口的彩色架构图见
[`architecture_status_zh.md`](architecture_status_zh.md)。该图区分“代码存在”、
“端到端机制已接入”和“论文级证据已完成”，避免把模块实现误认为性能主张成立。

```text
Agent Runtime / Tool Dispatcher / Message Bus
        |
        | causal events, futures, messages, joins
        v
+--------------------------------------------------+
| BeliefKV Control Plane                           |
|                                                  |
|  Event Normalizer                                |
|       |                                          |
|       v                                          |
|  Runtime Causal Context Graph                    |
|       |                                          |
|       +--> Observed-state JointPlanner (P5)      |
|       +--> FrontierBeliefModel (P6, optional)    |
|       |        | causal deadlines                |
|       |        v                                 |
|       +--> Morphology-aware ScenarioRiskPlanner  |
+--------------------------------------------------+
        |
        | context intents, admission decisions
        v
+--------------------------------------------------+
| SGLang Adapter                                   |
|                                                  |
|  KV Ownership / Lineage Index                    |
|  Radix Closure Shape / Arbitration Layer         |
|  Scheduler Command Queue                         |
+--------------------------------------------------+
        |
        v
SGLang Scheduler -> RadixCache/HiCache -> GPU/CPU KV
```

核心职责边界：

```text
BeliefKV：逻辑状态、因果关系、联合调度策略和迁移意图
SGLang：token、Radix topology、allocator、物理页面和实际传输
```

BeliefKV 不能直接修改 GPU tensor 或假设某个迁移命令已经成功。所有物理动作由 SGLang scheduler 在安全点执行，并通过 ACK 返回实际结果。

P5 的 observed JointPlan 是在线安全下界。P6 风险规划器与 P5 共用同一
`PolicyInput` 和物理快照；显式开关打开时，它只能把通过因果、形态、HBM 和
certificate 检查的 semantic intent 合并进同一个 JointPlan，不能成为第二个调度器。
默认配置仍为 read-only shadow。

## 6. Runtime Causal Context Graph

RCCG 不是用户提供的静态 DAG，而是由运行时事件增量构建的带类型动态图。

### 6.1 三类子图

```text
Invocation Graph
CALL / SPAWN / RETURN / WAIT / JOIN
描述 subagent 和 continuation

Communication Graph
MESSAGE / HANDOFF / BARRIER
描述对等 multi-agent 协作，允许有环

Context Lineage Graph
FRESH / FORK / RESUME / EXACT_PREFIX
描述逻辑 context 继承关系
```

Context Lineage Graph 仍不是物理 Radix tree。`FORK` 只有在 token prefix、模型和 LoRA 等缓存键完全兼容时才能产生物理共享。

### 6.2 Invocation 状态

```text
CREATED
READY
RUNNING_LLM
WAIT_TOOL
WAIT_CHILD
WAIT_JOIN
WAIT_MESSAGE
RETURNING
DONE
CANCELLED
```

典型状态转换：

```text
CALL(A, B, foreground)
    A -> WAIT_CHILD
    B -> READY

SPAWN(A, B, background)
    A 保持 READY/RUNNING
    B -> READY

MESSAGE(A, B)
    B -> READY 或 MESSAGE_PENDING

RETURN(B, A)
    B -> DONE
    A -> READY

JOIN_WAIT(A, {B, C})
    A -> WAIT_JOIN
```

### 6.3 事件可信度

```text
DECLARED_RUNTIME  来自实际 call/future/message runtime
OBSERVED_EXACT    tool_call_id、request_id 和 result 可完整关联
INFERRED          仅根据时间、prompt 或 prefix 推断
```

激进迁移只能使用前两类。`INFERRED` 关系只影响保守排序，不能覆盖 SGLang active lock。

## 7. SGLang 0.5.2rc1 的能力边界

SGLang 已提供：

- request enqueue、prefill、decode、finish 等内部生命周期；
- `rid` 和 continual prompting session；
- RadixCache 精确 token prefix 共享；
- HiCache GPU/CPU 分层缓存；
- `write_through`、`write_through_selective`、`write_back`；
- GPU/CPU KV copy、load-back 和 storage prefetch；
- `BlockStored`、`BlockRemoved`、`AllBlocksCleared` 等 KV events；
- tool-call 输出解析。

SGLang 原生不提供：

- `workflow_id/invocation_id/context_id` 的完整 scheduler 传播；
- `subagent_spawn/return/join/message` 语义事件；
- 外部策略控制的“迁移指定 context”稳定接口；
- context 到 Radix page 的 ownership 映射；
- 面向 workflow 的公平和 admission；
- 可抢占的 urgent/shadow 双队列传输。

KVFlow 源码中的 `agent_id`、AgentManager 和 workflow 优先级是论文分支扩展，不是上游 SGLang 原生接口。

## 8. RCCG 与 Radix tree 的桥接

### 8.1 多对多关系

```text
context_id -> set<physical_page_id>
physical_page_id -> set<context_id>
```

一个 context 对应 Radix path 上多个 node/page；一个共享 prefix page 可能服务多个没有控制关系的 agent。

### 8.2 Ownership Index

```text
ContextRecord:
    context_id
    invocation_ids
    lifecycle_state
    context_epoch
    radix_path
    semantic_priority

PhysicalPageRecord:
    page_id
    allocation_generation
    owner_contexts
    active_reader_count
    engine_lock_ref
    semantic_pin_count
    gpu_location
    host_location
    transfer_state
```

`page_id` 必须配合 allocation generation。Radix node 会 split/delete，allocator page index 也会在 free 后复用，单独使用 `TreeNode.id` 或 page index 都不能防止 stale command。

### 8.3 迁移意图解析

当 BeliefKV 发出：

```text
OFFLOAD_CONTEXT(parent, target_bytes=4GB, epoch=12)
```

Radix Arbitration Layer 必须：

1. 验证 context epoch；
2. 找到该 context 的实际 page 集合；
3. 过滤 engine-locked、active-shared 和 transfer-in-flight 页面；
4. 保留 HiRadix leaf-first/prefix-closure 约束；
5. 计算实际 marginal bytes；
6. 执行 D2H 或释放已有 CPU shadow 的 GPU 副本；
7. 在 ACK 后更新 residency；
8. 返回实际释放字节，触发重新规划。

如果 Parent 逻辑上拥有 10GB，其中 6GB 与活跃 child 共享，那么迁移 Parent 最多只能释放 4GB。Admission 不能按逻辑大小计费。

### 8.4 共享页面优先级

页面优先级使用最保守的 owner 聚合：

```text
任意 owner RUNNING/PINNED  -> page PINNED
否则任意 owner IMMINENT    -> page IMMINENT
否则                       -> page PARKED
```

SGLang `lock_ref` 只表示 engine 正确性保护。BeliefKV 需要独立的 `semantic_pin_ref`，不能复用或篡改 `lock_ref`。

可迁移条件：

```text
engine_lock_ref == 0
and semantic_pin_ref == 0
and active_reader_count == 0
and transfer_state == IDLE
```

### 8.5 生命周期结束与物理回收

child return 后，BeliefKV 立即解除 active/semantic reference，但不强制立即删除 Radix KV。无 HBM pressure 时，SGLang 可以继续保留它作为普通 prefix cache；有压力时，它会成为高优先级 victim。

## 9. 统一 Residency 分类

BeliefKV 将物理页面划分为：

```text
PINNED
    正在 prefill/decode，或被活跃 owner 共享

IMMINENT
    READY、收到消息、join 即将解除、最近恢复 parent

PARKED
    WAIT_CHILD、WAIT_TOOL、WAIT_MESSAGE

DEAD/UNOWNED
    one-shot invocation 完成且没有语义 owner
```

分类针对物理 page，而不是整个 agent/context。一个 Parent 的 private suffix 可以是 PARKED，共享 prefix 同时仍是 PINNED。

## 10. Reactive Causal Frontier Baseline

这是 BeliefKV 的首个完整策略，也是所有预测方法必须超过的强基线。

### 10.1 Subagent/Continuation 迁移

对于：

```text
Parent -> Searcher -> Browser
```

恢复顺序接近：

```text
Browser -> Searcher -> Parent
```

HBM pressure 下：

1. 释放 DEAD/UNOWNED 页面；
2. 迁移最晚恢复的 far ancestor private suffix；
3. 优先 Parent，再根据需求考虑 Searcher；
4. 保护所有活跃 descendant 使用的 shared prefix；
5. 仅释放 admission 所需的实际 marginal bytes。

Prefetch 顺序相反：Searcher 先于 Parent。

短 child 不应在进入 `WAIT_CHILD` 时立即触发 D2H。只有出现真实 admission pressure，或者等待持续超过迁移 break-even，才执行 reactive offload。

### 10.2 对等 Multi-Agent Working Set

对于：

```text
Planner <-> Coder <-> Reviewer
```

不存在确定调用栈。策略依据消息 frontier：

- `MESSAGE(A, B)` 使 B 进入 IMMINENT；
- 在 B 的 prompt 组装期间开始 prefetch；
- 最近反复通信的 context 构成 GPU working set；
- 长时间 `WAIT_MESSAGE` 且没有 pending message 的 context 降为 PARKED；
- 一轮 LLM 结束不代表持久 agent 的 KV 死亡；
- inactive persistent context 使用 size/cost-aware ARC/GDSF 类策略，而不是 one-shot 回收。

### 10.3 混合嵌套

同一 context 可以在不同阶段使用不同策略：

- Coder 与 Reviewer 是 MESSAGE peers；
- Coder 调用 Debugger 后，Coder 进入 WAIT_CHILD；
- Debugger 调用 Searcher 后形成 continuation stack；
- Searcher 返回后先恢复 Debugger；
- Debugger 返回后恢复 Coder；
- Coder 向 Reviewer 发消息后，Reviewer 转为 IMMINENT。

不需要全局 `mode=subagent|multi-agent`。

## 11. Admission 与调度

### 11.1 BeliefKV Admission Queue

大量 child 不能直接全部进入 SGLang waiting queue。BeliefKV 在 SGLang 前维护 admission queue：

1. 根据 prompt token 和 decode reserve 估算初始 HBM；
2. 计算 workflow 当前实际物理 HBM charge；
3. 在 root workflow budget 内增量准入 child；
4. 等 page eviction/transfer ACK 后再准入依赖释放空间的请求。

### 11.2 两级调度

第一级按 root workflow 公平，而不是按 agent 数量公平：

```text
compute fairness: attained GPU service / virtual runtime
memory fairness:  physical HBM bytes
```

同一 workflow 内的共享 prefix 只物理计费一次。跨 workflow 共享页需要定义稳定的比例 charge，避免重复计费或由创建者独占承担。

第二级在选中的 workflow 内调度 causal frontier：

1. 完成后能解除当前最长阻塞链的 child；
2. join 中最后剩余的 straggler；
3. 已收到消息、能推动协作的 peer；
4. 普通 READY agent；
5. background/speculative agent。

Prefix locality 和 batch compatibility 只作为同等级候选的 tie-breaker，不能覆盖 workflow 公平性。

Agent 场景不默认保护一个 decode 直到持续达到人类阅读速度。Decode 在安全 batch boundary 参与 workflow-level 时间片和 causal progress 调度。

## 12. Prepare-Commit CPU Shadow Migration

### 12.1 两阶段状态机

```text
GPU_ONLY
   |
   | PREPARE: PCIe/HBM 有余量时 D2H，GPU 副本仍保留
   v
DUAL_CLEAN
   |
   | COMMIT: 真实 HBM pressure/admission
   v
CPU_ONLY
```

Parent 提前恢复时：

```text
MIRRORING -> ABORT remaining copy -> 继续使用 GPU
```

错误预测不会直接产生 H2D resume stall：false positive 只浪费受限的后台资源；false negative 退化为 reactive migration。

### 12.2 Page 状态

```text
GPU_ONLY
MIRRORING
DUAL_CLEAN
CPU_ONLY
PREFETCHING
DEAD
```

只复制 sealed、page-aligned、不可变的 KV 页面。Decode 新追加页面保持 GPU_ONLY。

### 12.3 双传输队列

```text
Urgent Queue
    HBM admission、实际 eviction、READY context H2D

Shadow Queue
    parked parent/persistent context 的可暂停预备 D2H
```

规则：

```text
Urgent Queue 非空 -> 停止提交新的 shadow chunk
无 urgent 且干扰低 -> 传输一个 shadow chunk
HBM/compute 干扰超预算 -> throttle/pause
```

CUDA DMA 一旦提交不能保证中途抢占，因此 shadow 必须分块。若 chunk 为 `q`，有效带宽为 `B`，urgent transfer 的额外等待上界近似为 `q/B`。

### 12.4 Shadow 候选

优先级：

1. `WAIT_CHILD` 且有活跃 descendants 的大 Parent；
2. continuation stack 中距离叶节点较远的 ancestor；
3. 等待长工具调用的 context；
4. `WAIT_MESSAGE` 且没有 pending message 的 persistent peer；
5. 普通 inactive prefix cache。

排除 active pages、即将恢复的 context 和仍被活跃 owner 使用的 shared prefix。

### 12.5 有效 Shadow Window

对 context `i`：

```text
S_i       可迁移 private KV bytes
t_p       进入 PARKED
t_e       出现真实 eviction pressure
t_r       context 恢复
b_s(t)    不超过推理干扰预算的 shadow 带宽
```

最多可提前复制：

```text
C_i = min(S_i, integral[t_p, min(t_e,t_r)] b_s(t) dt)
```

只有 `t_e < t_r` 时 shadow 才能减少关键路径 D2H。评价时必须统计 useful shadow bytes，而不是只统计 PCIe idle ratio。

最适合的 workload 是长工具调用、长 subagent subtree 和 fanout 前存在准备窗口的突发负载。低负荷、快速 peer 轮转和持续饱和负载收益有限。

### 12.6 与 HiCache 的差异

HiCache write-through/selective 主要基于通用 cache hit 和全局 write policy。BeliefKV 使用：

- continuation/message liveness；
- HBM future pressure；
- urgent/shadow 双队列；
- inference interference feedback；
- event-driven COMMIT；
- context/page ownership 和实际 marginal bytes。

“PCIe 空闲时复制一份”本身不是创新；上述动态选择与安全 COMMIT 才构成 BeliefKV 的系统机制。

## 13. FrontierBelief 与候选相关风险规划

### 13.1 单一预测接口

P6 对外只发布一个版本化 `FrontierBeliefSnapshot`。内部可以使用 context tree、
competing-risk survival 和分位数分布，但它们不分别产生调度动作，避免再次形成
多个策略源。模型只预测局部、负载无关的需求：

```text
FrontierDemandOutcome
  next_runtime_boundary
  dependency_mode
  current_sequence_tokens
  remaining_decode_tokens
  prompt_growth_tokens
  next_output_tokens
  external_segments + censor state
```

以下信息不由模型预测：RCCG 已知的 JOIN 关系、当前 Radix closure、当前 HBM
占用，以及 keep/offload/prefetch 决策。

### 13.2 从局部需求到联合 scenario

```text
local conditional distributions
        |
        v
workflow-correlated particles
        |
        v
closure-complete BeliefScope + RCCG dependencies
        |
        v
top-K DemandScenario + OTHER
```

BeliefScope 按原子因果组扩展：纳入 `JOIN_ALL` 就纳入全部未完成 child，纳入
retraction 就纳入完整 blocker set。若组的建模成本超过预算，整组进入
`OTHER/A0`，不能截断部分 invocation 后虚构可释放空间。

JOIN 不在 raw token demand 上取 `max/min`。系统先在每个候选 JointPlan 下模拟
共享 GPU、PCIe 和外部等待，得到 child completion time，最后才按照
`JOIN_ALL=max(C_i)` 或 `JOIN_ANY=min(C_i)` 解析 reentry。

### 13.3 候选相关物理化

同一个 demand scenario 在不同调度和 KV residency 下会产生不同时间线：

```text
DemandScenario + candidate ActionGroup + PhysicalSnapshot
  -> Radix physicalizer
  -> uncached prefill / transfer demand
  -> GPUServiceCurveModel + TransferServiceModel
  -> event-driven TimedScenario
  -> ScenarioRiskPlanner
```

首版候选只使用当前可编译动作：`A0`、`PREPARE_HOST`、`PREFETCH_GPU`。
`COMMIT_CPU`、预测性 retraction、run-to-action 和 recompute 在正确性与收益门槛
通过前不开放。风险规划采用有限 receding horizon，并将当前 allocator、lock、
restore obligation、lease/grace 作为确定性硬约束；未来 reentry 才使用随机约束。

### 13.4 Fail-closed 动作门槛

`PREFETCH_GPU` 必须同时满足：

- local belief 为 exact support；
- calibration coverage 达到配置门槛；
- 没有 OOD reason；
- 当前物理 snapshot 可构造完整 startup/restore dependency；
- 候选在 HBM、PCIe 和 liveness 硬约束下可行。

backoff、OOD、因果闭包不完整或无法给出有限风险界时，只允许 `A0` 或无损
`PREPARE_HOST`。P5 observed policy 始终可独立运行。

### 13.5 训练与服务模型边界

语义模型不能使用旧负载下的 batch size 或 elapsed GPU service 作为需求标签。
训练目标是 action、token demand、prompt growth 和条件外部等待。GPU 服务时间由
独立受控微基准拟合：

```text
ServiceModel(
  phase, token demand, batch composition,
  sequence-length distribution, chunk position,
  prefill/decode mixing, HiCache/PCIe contention
) -> service-time distribution
```

数据按 repository/task/episode 分组，train、calibration、test-ID 和 test-OOD
严格隔离。当前 v6 artifact 仍是 development-only，composite support 大量回退到
backoff，因此只能用于 shadow 与管线检查。

### 13.6 当前接入状态

P5 observed planning 与 P6 predictive risk 使用两个独立 latest-wins worker。
observed worker 完成 bounded plan 后立即发布；predictive worker 只能读取已发布的
不可变 `PolicyInput/PhysicalSnapshot`，其排队、取消、失败和超时均不能阻塞 P5
计划发布。predictive worker 输出候选成本、拒绝原因和 semantic
`PredictiveIntent`；异步结果不发送命令，也不携带可长期执行的 Radix handle。
默认 shadow 模式仍保持：

```text
planner = belief_joint_observed
prediction_used = false
decision_authority = read_only_shadow
```

predictive worker 提交前先执行 action-specific eligibility gate，分别维护
`PrefetchTarget` 与 `PrepareHostVictim`，没有候选时不构造 belief、不提交 risk
job。只有 target/state、候选 KV 字节桶或 HBM/Host headroom hysteresis bucket
变化时才换代，避免每个 decode token 都取消并重算同一个 job。

候选包括 `A0`、`HOST(v)`、`PREFETCH(t)`、
`HOST(victim_set)->PREFETCH(t)` 和 `PARTIAL_PREFETCH(t,budget)`。确定性物理
preflight 位于 scenario simulation 之前；当前 HBM/Host、Radix actionability 或
private-suffix reclaim 不满足的候选不会进入多场景仿真。候选时间线已覆盖 GPU
service、PCIe serialization、RCCG dependency/JOIN 和 future KV growth HBM chance
constraint。

缓存边界遵循“语义可复用、物理必须重验”：相同 closure/prediction/RCCG evidence
可复用粒子化 demand belief；相同 `GPUServiceFeatures` 可复用有界 LRU 中的 service
estimate。allocator capacity、bundle generation、lock/actionability 和 transfer
feasibility 不跨 physical revision 缓存。

每个 predictive candidate 仍可携带 action certificate 作为诊断：相关 context epoch、
invocation state/revision、JOIN 状态、Radix bundle generation/bytes、HBM/Host
capacity floor、transfer epoch 和 model version。safe point 只校验该证书，不再用
无关 workflow 的全局 graph/allocator revision 否决候选。但在线 overlay 不执行
该旧证书，而是在 scheduler safe point 根据 semantic intent 重建 live bundle，重新
检查 context epoch、invocation state、HBM/Host、restore authority、PCIe 和 closure。

动作支持度按动作拆分：`PREPARE_HOST` 只要求 calibrated remaining-window 与
transfer evidence，不受 absolute future-HBM overflow 拒绝，因为它保留 GPU copy；
`PREFETCH_GPU` 要求 reentry-window、future KV growth 和 HBM chance constraint。
boundary 分类不可用不会再阻断与其无关的 KV 动作。

在线授权必须显式打开 `predictive_joint_overlay_enabled`。overlay 只在 observed
residency 没有动作时附加一个 `ALL_OR_NOTHING` ActionGroup；physical commit
失败、超时或超过 safe-point budget 时，只丢弃预测 intent，observed seed 原样保留。
`PREFETCH_GPU` 还要求 canary 开关、单 in-flight、copy bytes 不超过 KV pool 的 5%。
预测性 retraction、partial prefetch 和 reclaim-and-prefetch 仅保留 shadow 评估，
不能进入在线 intent。

控制面必须分别报告 eligibility、queue wait、risk compute 和 candidate certificate
validation 开销。开放动作前的目标为：no-candidate gate P99 `<1 ms`、
deterministic preflight P99 `<5 ms`、risk compute P99 `<20 ms`、
trigger-to-validation P99 `<50 ms`、action-specific stale rate `<10%`。这些门槛
尚未通过真实 GPU trace，因此不能宣称在线 schedulability 或性能收益。

只有离线 regret、校准、shadow stale rate、safe-point validation 开销和 w8
correctness smoke 均通过，才讨论给预测动作逐级授权。

### 13.7 形态债务与因果松弛量

P6 的 transfer estimate 必须绑定候选动作真实的 `PhysicalTransferShape`，最少包含：

```text
direction, actual_bytes, extent_count
extent_bytes_min/p50/max, small_extent_ratio
closure_generation, pinned_host, command_kind
native_transfer_state, contention_bucket
```

FrontierBelief/RCCG 给出 reentry、pressure 和 dependency release 的 scenario；物理
snapshot 给出 closure shape。`ScenarioRiskPlanner` 在每个 scenario 中先计算
`morphology_debt = Q90(shape-conditioned transfer time)`，再判断它是否能被 causal window
隐藏。相对 byte-only estimate 的额外成本单独记为 `morphology_penalty`。`PREPARE_HOST` 的收益仅来自
未来 pressure 到达前已经完成、并且未来 observed policy 确实会使用的 shadow bytes；
`PREFETCH_GPU` 还必须扣除提前占用 HBM 的时间成本。

首版模型保持最小化：按 `GPU/model/direction/command_kind/pinned_host` 分层，在受支持的
`bytes + extent_count` 邻域内插值并输出保守分位数。禁止退化为跨 extent-count 的纯
bytes 外推；样本不足时返回 unsupported，由 P5 接管，而不是假造低成本估计。

safe point 不执行风险规划阶段保存的旧 physical handle。它重新物化 live closure，并检查
实际 shape 是否仍落在 intent 的收益包络内；字节数、extent 数或 blocker 变化导致
`morphology_slack <= 0` 时，丢弃 intent，不影响 observed seed。

## 14. 端到端在线算法

```text
safe_point(epoch):
  APPLY_EVENTS
    drain runtime events, transfer ACK/native completion and cancellations
    update RCCG, queue state, ownership, allocator and restore obligations

  CAPTURE_AND_PLAN
    lazily build one immutable SafePointPhysicalSnapshot(epoch)
    publish the bounded observed-state JointPlan seed
    async worker may improve the semantic plan
    derive action-specific closure shape for predictive candidates
    async P6 worker evaluates causal slack minus morphology debt

  TRANSACTIONAL_COMMIT
    validate each closed ActionGroup read-set
    materialize current Radix bundle and recheck live shape/capacity/lock/lease/grace
    commit the maximal valid group
    attach source_joint_plan_id to execution/admission/residency/retraction

  if any physical state changes during commit:
    advance epoch before another planning decision
```

Restore obligation、debt-owned funding reservation 和 service-quantum grace 是
liveness 硬约束。retraction 只有在同一事务中建立确定性恢复路径后才允许提交。
P6 worker 读取快照后不得修改状态。显式 overlay 只发布 semantic intent；safe
point 物理化失败、预测过时或 OOD 时，P5 observed JointPlan 不受影响。

## 15. Runtime 事件和动作接口

### 15.1 Agent Runtime -> BeliefKV

```text
WORKFLOW_START / WORKFLOW_END
INVOCATION_CREATE / INVOCATION_CANCEL
CALL / SPAWN / RETURN
MESSAGE / HANDOFF
JOIN_CREATE / JOIN_WAIT / JOIN_SATISFIED
TOOL_START / TOOL_END
LLM_SUBMIT / LLM_RESULT
```

### 15.2 SGLang -> BeliefKV

```text
REQUEST_QUEUED
PREFIX_MATCH
PREFILL_START / PREFILL_END
DECODE_START / DECODE_END
CACHE_INSERT / NODE_SPLIT / PAGE_FREE
LOCK_CHANGE
TRANSFER_START / TRANSFER_END
REQUEST_FINISH / REQUEST_ABORT
HBM_SNAPSHOT
```

事件必须批量发送，避免 per-token control-plane 开销。

### 15.3 BeliefKV -> SGLang

```text
ADMIT_REQUEST
DEFER_REQUEST
OFFLOAD_CONTEXT
SHADOW_CONTEXT
PREFETCH_CONTEXT
DROP_UNOWNED
PIN_CONTEXT
UNPIN_CONTEXT
SET_WORKFLOW_BUDGET
```

命令携带 `context_id + epoch + target_bytes + priority/deadline`。Scheduler 可以 partial fulfill 或 reject，并返回实际 page/byte 结果。

## 16. 必须维持的系统不变量

```text
1. RCCG 不直接拥有或释放物理 KV。
2. SGLang 是 allocator、Radix topology 和 page location 的唯一真相来源。
3. 任意 active owner 都能阻止共享页面 COMMIT/offload。
4. shared page 的物理空间只计算一次。
5. admission 只使用实际 marginal bytes 和 transfer ACK。
6. stale context epoch 命令必须被拒绝。
7. engine_lock_ref 和 semantic_pin_ref 分离。
8. 初版保持 HiRadix leaf-first/prefix-closure 约束。
9. prediction failure 不能破坏正确性或使系统无法运行。
10. BeliefKV disabled 时必须保持上游 SGLang 行为。
11. retraction、funding、restore 和 replacement admission 必须属于闭合 ActionGroup。
12. safe point commit 后不得继续复用旧 PhysicalSnapshot。
13. 实验压力只按 /get_server_info 返回的实际 KV pool 计算。
14. transfer estimate 必须绑定实际 physical closure shape，不能只按 context bytes 计费。
15. safe point 的 live shape 超出 intent 收益包络时必须 fail closed，不能沿用旧成本。
16. 形态模型 unsupported/OOD 时必须回退 P5，不得用跨 bucket 的乐观外推补值。
```

## 17. 实际代码结构

当前关键模块如下；SGLang adapter 只承载状态捕获、safe-point commit 和命令
ACK，策略仍位于 BeliefKV：

```text
beliefkv/
  core/
    config.py
    ids.py
    events.py
  control/
    causal_graph.py
    controller.py
  predictor/
    frontier_belief.py
    structured_frontier.py
    hardware_service.py
    composer.py
  policy/
    joint_scheduler.py
    online_joint.py
    predictive_joint.py
    predictive_timeline.py
    risk_shadow.py
    transfer_planner.py
  runtime/
    audit.py
    joint_shadow.py
    bundles.py
    page_index.py
    radix_arbiter.py
    restore_obligation.py
    protocol.py
    sglang_v052rc1.py
  simulator/
    queue_service.py
  experiments/
    p6_dataset.py
    p6_decision_points.py
    deepagents_swebench.py
  metrics/
    artifacts.py
    summary.py
```

SGLang patch 应保持窄接口，不把完整 BeliefKV policy 写入 scheduler 源码。

## 18. 实施顺序

### P0-P4：已完成的系统基座

已完成 RCCG、SGLang metadata/ownership bridge、Radix closure 仲裁、HiCache
prepare/commit、Host 生命周期和迁移时间线观测。历史正确性问题及其修复过程保留在
实验文档中，不再作为当前策略接口。

### P5：Observed JointPlan 冻结

- observed-state execution/admission/residency/retraction 使用唯一 JointPlan 来源；
- SafePointPhysicalSnapshot 按 epoch 惰性构建；
- restore obligation、funding、grace 和 transaction ACK 保持守恒；
- reactive planner 只作为 intent compiler/liveness fallback，不独立选 victim；
- 继续允许修 correctness bug，但不再改变 P5 架构。

### P6.0-P6.1：数据与局部需求模型

- 固定 project/task split，采集 event-point decision samples；
- 区分 demand labels 与旧负载下 observed service time；
- 训练并校准结构化局部模型和独立硬件 service curve；
- 当前 v6 仅 development-only，不参与正式 test 结论。

### P6.2：候选风险链路与受限在线 overlay（当前阶段）

- 构造 closure-complete BeliefScope；
- 先采样联合 demand particles，再按候选动作投影到其实际依赖的随机变量；
- 对投影粒子做 8--16 个确定性 medoid cluster，medoid 计算期望收益，cluster 内
  earliest reentry/max growth 形成保守可行性 envelope，概率质量不再丢入不透明
  OTHER；
- observed 与 predictive worker 隔离，observed seed 立即发布；
- 在 belief compose 前执行 action-specific eligibility 与 bucket/hysteresis 去重；
- 物理化 A0/HOST/PREFETCH/RECLAIM+PREFETCH/PARTIAL_PREFETCH；
- 在 scenario simulation 前执行确定性物理 preflight；
- 在候选相关时间线上解析 JOIN/reentry；
- 维护 future KV growth HBM ledger 和 chance constraint；
- 将通用 future feasibility 与 future-HBM chance constraint 分开归因，避免同一
  overflow 被重复否决；
- 将 PREPARE_HOST 建模为两阶段 recourse：空闲期建立 CPU shadow，未来 pressure
  到来时节省关键路径 D2H，同时计入当前 PCIe 干扰和 Host residency；
- 使用按 GPU/model/direction/command/page-count/size/native contention 条件化的
  持久化 transfer service curve warm-start，在线样本只滚动修正；
- 对 PREPARE_HOST 单独投影完整 GPU descendant closure：D2H 成本包含 cross-context
  descendant，收益只计目标 context 的 exclusive copy；除 `descendant_closure` 外的
  lock、owner、in-flight 和 pin blocker 仍为硬约束；
- 生成 action-specific certificate 作为 stale characterization；
- 发布不含 physical handle 的 semantic intent，并在 safe point 重新物理化；
- 默认只读；显式 overlay 仅授权 PREPARE_HOST 和受限 PREFETCH_GPU canary。

受控 CPU case 已覆盖 deterministic reject、future-HBM reject/accept、
reclaim+prefetch shadow、calibrated action-specific backoff、PREPARE 的 HBM 解耦、
两阶段 recourse、action-projected reduction、safe-point rematerialization 和 5%
prefetch canary。正收益候选会额外触发一次去重后的异步 PolicyInput 持久化，使
候选所在精确 epoch 可离线 replay，而不是依赖 10 秒固定采样恰好命中。

### P6.3-P6.5：形态感知联合控制（已完成的止损分支）

该分支按以下最短闭环执行完毕。修正 transfer service contract 后，同一冻结 trace 中
byte-only 与 extent-count-aware 策略没有产生 eligibility 或 selected-action flip，因此不再
作为当前核心主线，也不通过扩大 GPU 矩阵继续寻找正例：

1. **M1 自然形态审计**：复用现有 P6 trace，同时报告 candidate epoch、physical generation、
   `context + context_epoch + stable morphology tuple` parked episode 和 context-weighted 统计。
   physical generation 只反映 Radix 演化，不能作为独立 workload 样本。
2. **M2 extent-count-aware 首版服务模型**：将 2026-08-10 GPU0 受控矩阵作为 development
   warm-start，以 `bytes + extent_count` 输出保守分位数；extent-size distribution 和 closure depth
   暂不作为模型输入。禁止跨 extent-count 的 bytes-only 回退，并显式报告 unsupported。
3. **M3 精确 closure plumbing**：candidate physicalization 将实际
   `PhysicalTransferShape` 传给 timeline/risk planner；safe point 重新物化后验证 shape
   envelope。复用现有 snapshot、certificate 和 transaction，不引入第二套状态机。
4. **M4 JointPlan 决策**：在现有 recourse 中用
   `causal slack - morphology debt` 替代 bytes-only D2H 成本；动作集合仍只有 A0、
   PREPARE_HOST 和受限 PREFETCH_GPU，不改 predictor、RCCG、fairness 或 restore 协议。
5. **M5 固定 trace 反事实**：在同一批保存的 PolicyInput 上比较 bytes-only 与
   extent-count-aware 的候选排序、timing change、feasibility-reason change、eligibility flip、
   selected-action flip 和 unsupported rate。eligibility flip 必须按同一个配对候选分为
   promotion（byte-only 不执行、shape-aware 执行）和 veto（byte-only 执行、shape-aware 阻止）；
   纯 timing/reason sensitivity 不进入在线验证。
6. **M6 单动作 canary**：原计划只运行一个自然选中的 PREPARE_HOST 或受限 PREFETCH_GPU，验证
   intent -> live-shape rematerialization -> dispatch -> ACK -> terminal，并与 A0 比较实际
   stall/收益。promotion 只开放 shape-aware PREPARE canary；veto 只开放 byte-only treatment，
   并以 shape-aware/P5 不执行为 control。selected-action change 单独报告，不自动开放动作。
   通过后才扩大到端到端 workflow。

M1--M4 的服务成本、物理化与 fail-closed 机制继续保留；M5--M6 没有通过策略相关性门槛。
GPU1 crossover、small-size 完整矩阵、progressive slicing、KV compaction、自定义 DMA 和
预测性 retraction 均停止投入。P6 后续回到 FrontierBelief 的因果预测价值验证。

### P7：可移植性与正式实验

完成固定版本实验后再适配新版 HiCache/SGLang，并在同一模型、同一版本和固定
trace 上比较 SGLang/HiCache、P5 observed、P6 shadow/action 与 offline oracle。

## 19. 实验设计

### 19.1 Workload

- coding agent：shell、file、test、persistent search agent；
- browser/research agent：search、fetch、并发 subagent、join；
- 对等 multi-agent：planner/coder/reviewer 循环通信；
- recursive subagent：多层 foreground/background spawn；
- 混合 workload：多个 root workflow 单卡并发。

真实 trace 优先；合成 trace 只用于隔离变量和边界测试。

### 19.2 Baseline

```text
SGLang RadixCache/LRU
HiCache write_back
HiCache write_through
HiCache write_through_selective
Reactive Causal Frontier
Next-agent Markov/Context Tree
Prediction + direct offload
BeliefKV Prepare-Commit
Offline Oracle
```

### 19.3 指标

```text
workflow completion p50/p95/p99
request TTFT 和 agent resume stall
HBM admission stall
GPU/CPU KV bytes 和峰值
urgent D2H/H2D bytes
useful/wasted shadow bytes
shadow hit ratio
PCIe queue 和 copy interference
recompute tokens/time
workflow fairness
predictor calibration/coverage/OOD rate
planner 和 adapter CPU overhead
```

### 19.4 在线动作归因

独立 useful-action oracle 不再作为开放在线预测动作的前置 gate。为缩短关键路径，系统在单动作
canary 和端到端 A/B 中同步回答以下问题：

1. predictive PREPARE 是否在后续 pressure 时被实际消费并避免 reactive D2H；
2. predictive retraction 是否让 replacement 真实进入 batch 并获得 service；
3. 节省的 admission/transfer stall 是否覆盖预测规划、迁移干扰和 victim restore 成本。

每笔动作进入 `useful/wasted/too-late/censored` 之一。归因仍要求 submit-time ownership、
generation-aware physical extent、下一次真实 reentry、pressure/transfer 和 service 结果；缺少标签的
记录只能 censored，不能用于收益声明。旧 trace 不能恢复这些标签，但这不再阻塞新 canary。

如果在线动作长期不被消费，或 stall 降低但 workflow throughput 不变，预测动作停止扩展；不能通过
筛选 trace、恢复 morphology 分支或降低为负收益阈值制造正例。

## 20. 当前创新主张

当前核心假设，而不是已经完成的论文结论，是：

> 动态 agent workflow 的 SPAWN/JOIN、工具等待和 reentry 会形成不断扩张或闭合的 causal
> frontier。FrontierBelief 若能识别哪些 GPU service 将快速解锁 TOOL/SPAWN/RETURN/JOIN，
> JointPlan 就可以联合调整 admission、KV residency 和 selective retraction，将有限 GPU service
> 分配给能更快产生下一次有效 action 的请求。

围绕该假设，贡献层次收敛为：

1. **核心算法假设**：学习局部 demand/reentry 分布，由 RCCG 组合联合 scenario，并在有限
   horizon 内将 invocation 分类为 EXPAND、CLOSE、HOLD 或 UNKNOWN；
2. **统一控制**：预测只发布 semantic intent，safe point 将其与 observed JointPlan、实时
   allocator、ownership、admission、retraction 和 restore obligation 一起物理化，避免第二个策略源；
3. **安全下界**：预测不确定、因果 read-set 失效或服务模型 OOD 时回退 observed P5；
   non-destructive PREPARE 与事务化 restore 保证错误预测不破坏活性；
4. **Agent workload**：受控 2--3 child fan-out 暴露多 child 工具并行、JOIN long tail 和 HBM
   competition；旧单 child trace 继续训练局部需求，新 fan-out 数据按 repository 与 A/B 隔离；
5. **支撑模型**：`bytes + extent_count + contention` transfer curve 只负责物理成本和 OOD 安全，
   不再作为核心创新或独立动作来源。

单独的预测器、shadow copy、状态机、fragmentation 测量或 Radix ownership 都不足以支撑论文。
开发阶段直接比较 P5 observed JointPlan 与完整 predictive JointPlan。核心主张成立的硬门槛仍是：
在相同 workload profile 上，FrontierBelief 产生可归因的 PREPARE/retraction action change，降低
实际 admission/restore stall，并最终提高 successful workflows/hour；否则 P6 只是一项安全的
工程 overlay。外部 baseline 和正式自然 workload 统计在系统闭环后补充。

## 21. 主要风险

### 21.1 可观测性

不同 agent framework 的 call/message/future hook 不统一。需要以最小事件协议为抽象，并至少实现多个真实 runtime adapter。

### 21.2 Ownership 开销

精确 page owner set 可能占用 CPU 内存和调度时间。应 page-align、批量更新，并区分 correctness lock 与仅影响策略的 owner metadata。

### 21.3 HiCache 结构限制

现有 HiRadix 主要支持 leaf-first eviction 和连续 prefix load-back。初版不追求任意 page placement。

### 21.4 PCIe/HBM 干扰

PCIe idle 不代表 D2H 免费，bytes 也不能唯一决定 D2H 成本。当前 4.12 倍差异只来自
GPU0、单一字节量和两个极端布局；服务模型必须保留 shape support/OOD，并用实测
inference slowdown 而不是标称带宽。GPU1 crossover 是后续泛化证据，不是当前实现阻塞项。

### 21.5 预测泛化

TraceLab 等数据存在 workload、tool name 和平台偏差。必须使用 taxonomy、hierarchical backoff、survival calibration、在线 residual 和 OOD fallback。

### 21.6 创新性

HiCache 已有 write-through，KVFlow/TokenCake/ScaleSim/AugServe 已有 workflow-aware
eviction/prefetch 或 future-use 估计，TokenCake/PBKV/Continuum 也覆盖了等待时间或未来复用预测。
BeliefKV 必须证明 RCCG 上的联合不确定性、action-specific scenario reduction 和 observed/Predictive
统一提交能产生这些 per-context 方法无法给出的有效动作。形态信息只用于避免错误估算迁移成本，
不能再被用来弥补因果预测本身缺少决策收益。

## 22. 当前待决问题

1. observed P5 与完整未来 oracle 之间是否存在足够大的 useful-action gap？
2. FrontierBelief 的 reentry、future pressure 和 prompt/output demand 分布能否跨 repository 与
   workload family 保持校准，并在 OOD 时可靠回退？
3. 联合 RCCG scenario 是否比 median/EWMA 或独立 child quantile 产生更低的 planner regret？
4. 自然 workload 中 PREPARE/PREFETCH 的 useful-action rate 是否足以覆盖控制面和 PCIe 干扰？
5. transfer artifact 在不同 GPU/SGLang/HiCache 版本上的校准成本与 unsupported 比例是多少？
6. blocking subagent、工具等待和 cyclic peer 三类场景中，哪类因果窗口真正贡献净收益？

## 23. 当前实现状态

截至本文档日期：

- 已实现原子 RCCG reducer、nested call/spawn/join/message 状态转换和事件幂等；
- 已实现 context/page ownership bridge、共享页物理计费、lock/reader/pin 分离和 Radix closure 仲裁；
- 已实现 observed-state JointPlan、workflow fairness/admission、running retraction、restore obligation、debt-owned funding reservation 和 service grace；
- 已实现 SafePointPhysicalSnapshot、ActionGroup/read-set 局部校验、迁移 transaction/ACK 守恒及显式 shutdown；
- 已实现 Host KV 生命周期、terminal private-KV cleanup、HiCache D2H/H2D 与有界审计；
- 已实现 transfer service warm-start 契约：artifact 与 runtime 使用独立 sample gate，启动时
  验证校准证据仍能支持真实 query，并将契约摘要写入初始化审计；
- 已实现 native HiCache submit-time ownership snapshot，在 DMA 完成前冻结 owner context、epoch、
  extent size 和 page-index revision；历史 completion-time lookup 仅作为旧记录兼容路径；
- 已实现 P6 decision-point 数据集、结构化 FrontierBelief artifact、GPU service curve、scenario composer、candidate timeline 和 risk shadow；
- 已拆分 observed/predictive worker，并实现 action gate、联合候选、确定性 preflight、cooperative cancellation、事件桶去重和候选级 read-set validation；
- 已实现不携带 Radix handle 的 PredictiveIntent、action-specific support、发布/safe
  point 双重 causal certificate 校验、收益包络约束、safe-point rematerialization、
  PREPARE_HOST overlay 和 5% PREFETCH_GPU canary；
- 旧 `joint_predictive_enabled` 在线启发式已降为兼容元数据开关，不再改变排序、victim 或迁移动作；
- restore funding preview 已改为从 revision-cached migratable roots 做有界局部 bundle 物化，等待下一次高压 GPU 验证；
- P6 批量采集会查询 `/get_server_info`，实际 KV pool 低于实验要求时在发起 workflow 前失败；
- 当前完整 CPU serving 回归为 598 passed、8 skipped、3 subtests passed；P6
  Deep Agents/collection 侧回归为 90 passed；
- 已完成一次固定 16-workflow development trace，证明 worker 隔离和 observed
  fallback 稳定，但旧 scenario reduction 使 selected action 为 0；修复后对 126 个
  稀疏快照离线 replay，OTHER 拒绝已归零，现有快照中的 102 个正收益候选均被
  future-HBM gate 拒绝；
- 已完成一次 4-workflow short trace。运行时发现 PREPARE 错误复用 destructive
  closure gate；修复后重放 45 个 snapshot 生成 57 个 closure-complete PREPARE
  候选。41/57 候选预测到 capacity pressure，最大 overflow 约 5.78 GB；修复
  transfer size-neighbor 回退后，456 个 scenario 中有 2 个满足完整 recourse 条件，
  但概率质量不足以覆盖保守 D2H interference cost，候选层仍无正收益在线 intent；
- 已在 GPU0 完成相同 2,659,221,504 bytes、7/106 extents 的受控 D2H 对照，三次
  重复均显示高碎片更慢，均值分别为 185.69/765.17 ms；该结果只作为形态感知路线的
  development evidence，双 GPU formal gate 仍未完成；
- 已完成 M1 自然形态审计：57 个候选 epoch 对应 51 个物理 generation，但只形成 13 个稳定
  parked episode；其中 5 个高 extent-count WAIT_JOIN/WAIT_TOOL episode 分布在 3/5 个 context。
  该结果证明问题存在，不能将 33/51 解释为 prevalence；
- M2 单 GPU development artifact 是 extent-count-aware 首版；M3 closure-level shape plumbing 和
  M4 JointPlan 机制已完成。PredictiveIntent 现在携带 shape/count/transfer/stall
  envelope，safe point 重建实时形态并在 OOD、成本超界或 slack 过期时回退 observed P5；
- 初始 M5 trace 只改变 timing/reason，未改变动作。后续冻结 Xarray w8 characterization 中，
  `PREPARE_HOST` 的 future-HBM gate 被确认是动作语义 bug；修正后的 938 个 paired snapshot 含
  842 个 PREPARE candidate，byte-only/extent-count-aware 分别有 25/11 个 eligible，产生 3 次
  promotion、17 次 veto 和 20 次 selected-action change。该修正发生在首次观察后，因此属于
  post-hoc development evidence；
- M6 单动作 canary 基础设施和显式 transfer-model arm 已实现。一次 morphology-aware autonomous
  w8 canary 8/8 正常结束、750 次 LLM call、9 个动态 subagent，shutdown 时无未决事务，但
  `natural_prepare_count=0`。随后一次冻结的 veto-only treatment 运行 7,216.16 秒，7/8 workflow
  完成；旧在线阈值使 16 个 paired veto 全部错误回退为 `shape_unsupported`，11 个在发布前
  stale，5 个在 safe point 被拒绝，0 commit/0 predictive D2H。修复 warm-start 资格后，14 个
  可用源 snapshot 上的 16 个候选产生 14 次 timing change、6 次 reason change，但 0 action flip。
  严格门禁现已要求 counterfactual shape support。尚未证明 P6 在线净收益，也尚未完成论文
  baseline/oracle 实验。

下一阶段按 2026-08-11 P6 执行计划推进：一次性删除 morphology 策略分支并补齐 transfer attribution
正确性；加入 `parallel_analysis_2to3` workload；直接运行单笔 PREPARE_HOST canary；随后用
EXPAND/CLOSE/HOLD 注解现有 selective retraction，并进行 P5 对完整 predictive JointPlan 的短期
端到端 A/B。独立 oracle 不再是前置 gate，PREFETCH_GPU、peer-agent、外部 baseline 和更复杂模型
均不进入当前关键路径。

因此，当前可以主张 P5 物理控制面和 P6 shadow 链路已实现，不能把 2026-08-07
低支持度 heuristic A/B、请求值形式的 KV pool 或单次 agent rollout 当作预测收益证据。
