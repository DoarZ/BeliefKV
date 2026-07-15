# BeliefKV 当前系统设计

日期：2026-07-14

状态：设计与实现基线。核心控制面、模拟器和固定版本 SGLang patch 已落地；已完成单模型 CUDA 集成 smoke，论文级 GPU 实验尚未完成。

本文档取代 `technical_archive_2026-07-10.md` 作为当前设计的权威说明。旧文档保留为历史讨论记录，其中的 flat next-action predictor、静态 belief frontier 和以 `agent_id` 为核心的元数据设计不再代表当前方案。

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
|       +--> Reactive Causal Frontier Scheduler    |
|       +--> Remaining-time Predictor (optional)   |
|       +--> Prepare-Commit Migration Planner      |
+--------------------------------------------------+
        |
        | context intents, admission decisions
        v
+--------------------------------------------------+
| SGLang Adapter                                   |
|                                                  |
|  KV Ownership / Lineage Index                    |
|  Radix Arbitration Layer                         |
|  Scheduler Command Queue                         |
+--------------------------------------------------+
        |
        v
SGLang Scheduler -> RadixCache/HiCache -> GPU/CPU KV
```

核心职责边界：

```text
BeliefKV：逻辑状态、因果关系、预测、策略和迁移意图
SGLang：token、Radix topology、allocator、物理页面和实际传输
```

BeliefKV 不能直接修改 GPU tensor 或假设某个迁移命令已经成功。所有物理动作由 SGLang scheduler 在安全点执行，并通过 ACK 返回实际结果。

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

## 13. Remaining-Time Predictor

### 13.1 预测目标

预测器不输出单点 agent duration，而输出：

```text
P(context resumes within tau)
p50/p90/p95 remaining time
next event distribution
confidence
OOD score
backoff level
```

其中 `tau` 由当前 KV 的 D2H/H2D 时间决定。

### 13.2 三部分模型

```text
Tool Survival Model
Agent Semi-Markov Context Tree
LLM Service Cost Model
```

#### Tool Survival Model

工具时长具有长尾、timeout 和 right-censoring。模型预测生存函数：

```text
S(t | x) = P(T_tool > t | x)
```

工具已运行 `e` 后的剩余时间：

```text
P(T_remaining > u | T > e, x) = S(e+u | x) / S(e | x)
```

实现建议：

- hierarchical Kaplan-Meier survival curve；
- 浅层 GBDT 预测 AFT time-scale residual；
- global -> tool family -> exact backend/endpoint 分层回退；
- online EWMA/quantile residual correction。

#### Agent Semi-Markov Context Tree

将 agent 轨迹归一化为：

```text
LLM_TEXT
LLM_TOOL_CALL
TOOL_SHELL
TOOL_SEARCH
TOOL_FILE
SPAWN_CHILD
WAIT_CHILD
WAIT_JOIN
MESSAGE
RETURN
```

Variable-order context tree 预测下一状态，survival distribution 预测每个状态驻留时间，`RETURN` 为吸收状态。并发 join 的 parent resume time 由运行时已知的 min/max/join 语义组合。

#### LLM Service Cost Model

不能直接学习旧调度器下的 agent wall time。应分离 intrinsic work 和系统服务时间：

```text
T_LLM = T_queue + T_prefill + T_decode
```

行为模型预测 remaining output tokens、future rounds 和 action transition；在线 cost model 根据 prompt/cache-hit tokens、batch、context length 和 GPU profile 转换为 wall time。

### 13.3 特征原则

使用稳定结构化特征：tool family、input size、文件/URL 数、timeout、command executable、endpoint class、并发度、prompt/cache-hit tokens、模型、调用深度、round 和最近 action suffix。

避免 project/session ID、原始 agent 名称、完整 prompt embedding 和未来 result size。

### 13.4 训练和泛化

训练数据必须包含完整事件时间、timeout/cancel censoring、context relation、token/KV 大小和运行负载。

数据切分：

```text
group by project/session
temporal holdout
leave-one-workload-family-out
leave-one-tool-family-out
unseen agent-role test
```

不能随机拆分同一 workflow 的 event。

评估指标：survival NLL、Integrated Brier Score、interval coverage、calibration error、ranking accuracy、最终 offload/prefetch regret 和 workflow latency。

### 13.5 在线开销

- 不使用 LLM 或大 Transformer 作为 predictor；
- context tree 最大阶数先取 3-5；
- 使用浅层 GBDT/查找表；
- 只在事件和 elapsed-time bucket crossing 时更新；
- 缓存同一状态的预测；
- 目标为单 context 数十微秒、一次事件批量更新低于 0.5ms，最终以实测为准。

### 13.6 OOD 回退

预测区间持续失配、出现未知 tool family 或 calibration coverage 下降时：

```text
降低 confidence
扩大区间
退回上层 survival prior
停止基于预测的 aggressive action
保留 reactive causal frontier policy
```

## 14. 端到端在线算法

```text
on_runtime_event(event):
    1. normalize event and validate causal identity
    2. update RCCG invocation/communication/context state
    3. update context epoch and wake-up frontier
    4. bind new requests/contexts to Radix ownership index
    5. release semantic refs of DONE/CANCELLED invocations
    6. process completed transfer ACKs
    7. compute actual HBM free/marginal bytes
    8. run workflow-level admission and fairness
    9. if actual pressure:
           free unowned pages
           commit DUAL_CLEAN pages
           select PARKED victims using causal constraints
           issue urgent D2H until enough actual bytes are ACKed
   10. mark READY/message/join targets as IMMINENT
   11. issue urgent or deadline-aware H2D prefetch
   12. if no urgent transfer and interference budget allows:
           update remaining-time predictions
           copy one shadow chunk from eligible PARKED context
   13. select workflow by attained service
   14. select request from workflow causal frontier
   15. admit request into SGLang scheduler
```

预测器从不绕过 active lock、共享 owner、epoch 和 transfer ACK 检查。

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
```

## 17. 实际代码结构

当前仓库只保留基于 RCCG、page ownership 和 scheduler safe point 的实现。早期
snapshot planner 已删除，避免与实际在线控制面形成两套不一致的 API：

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
    taxonomy.py
    tool_survival.py
    action_context_tree.py
    service_cost.py
    composer.py
    calibration.py
    training.py
  policy/
    admission.py
    workflow_fairness.py
    causal_frontier.py
    residency.py
    shadow_controller.py
    transfer_cost.py
    transfer_planner.py
  runtime/
    agent_runtime_adapter.py
    audit.py
    event_channel.py
    sglang_adapter.py
    page_index.py
    radix_arbiter.py
    command_queue.py
    protocol.py
    sglang_v052rc1.py
  simulator/
    schema.py
    page_simulator.py
  experiments/
    matrix.py
  traces/
    normalizer.py
    runtime_validation.py
  metrics/
    artifacts.py
    summary.py
```

SGLang patch 应保持窄接口，不把完整 BeliefKV policy 写入 scheduler 源码。

## 18. 实施顺序

### Phase 0：Trace 与离线模拟器

- 扩展 trace schema 支持 invocation/context/event identity；
- 构建 RCCG event reducer；
- 构建 page-level HBM/PCIe replay simulator；
- 验证 causal state transition 和 nested workflow。

退出条件：相同 trace 重放产生确定的 RCCG 和资源状态。

### Phase 1：Reactive Baseline

- 实现四级 residency 分类；
- 实现 continuation far-ancestor migration；
- 实现 message-driven working set；
- 实现 workflow admission/fairness；
- 不使用 predictor。

退出条件：在合成和真实 trace 上无非法迁移、无重复物理计费，并能与 offline oracle 比较。

### Phase 2：SGLang Metadata 与 Ownership Bridge

- 传播 workflow/invocation/context metadata；
- 建立 page ownership index；
- 接入 lock、insert、split、free 和 transfer ACK；
- 实现 Radix arbitration 和 marginal byte reporting。

退出条件：BeliefKV disabled 行为不变；active/shared KV 不会被错误迁移。

### Phase 3：可控 HiCache 迁移

- 实现 context intent -> leaf/page action；
- 实现 urgent D2H/H2D；
- 实现 admission 等待 transfer ACK；
- 验证 CPU/GPU state consistency。

退出条件：压力测试无 stale page handle、无 OOM、无 location divergence。

### Phase 4：Prepare-Commit Shadowing

- 实现 DUAL_CLEAN 状态；
- 实现 urgent/shadow 双队列和小 chunk；
- 建立 interference feedback controller；
- 先使用无预测 heuristic shadow ordering。

退出条件：false shadow 不制造 H2D stall；urgent transfer 的额外等待受 chunk 上界控制。

### Phase 5：Remaining-Time Predictor

- 训练 hierarchical tool survival baseline；
- 训练 semi-Markov context tree；
- 接入 online cost model 和 calibration；
- 只用于 shadow/prefetch/order，保留 OOD fallback。

退出条件：跨 workload calibration 合格，并在 end-to-end 决策 regret 上超过 per-tool median/EWMA。

### Phase 6：完整实验

- 对比 SGLang LRU/HiCache 三种 write policy；
- 对比 reactive baseline、next-action predictor、完整 BeliefKV；
- 对比 KVFlow/TokenCake/Agentix 风格策略；
- 计算 offline full-future oracle；
- 在相同 SGLang 版本和模型上做 apples-to-apples 实验；
- 再在较新 SGLang 分支验证可移植性。

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

### 19.4 关键离线 Oracle

首先回答三个必要问题：

1. Reactive Causal Frontier 与完整未来 oracle 的差距是否显著？
2. parked window 内最多能提前复制多少最终被驱逐的 useful bytes？
3. 完整 subtree/remaining-time prediction 是否显著优于事件驱动和简单 median/EWMA？

如果 reactive 已接近 oracle，复杂预测器不应成为核心。如果 useful-shadow oracle 收益很低，Prepare-Commit 只保留为工程选项。

## 20. 当前创新主张

目前最可辩护的系统主张是：

1. 使用运行时因果关系而不是用户 DAG，统一处理 subagent continuation 和 cyclic peer collaboration；
2. 将 causal context value 与 token-prefix physical sharing 分离，通过 ownership bridge 做实际 marginal HBM 管理；
3. 使用 reactive causal frontier 提供无预测、跨 workload 的安全下界；
4. 将预测从 destructive offload 降级为 non-destructive PREPARE，实际 GPU release 由事件驱动 COMMIT；
5. 使用 remaining-time distribution 和 OOD calibration，而不是单点工具/agent duration；
6. 联合管理 workflow fairness、child admission、KV residency 和 PCIe transfer。

单独的状态机、shadow copy、survival predictor 或 Radix ownership 都不足以支撑高水平系统论文。贡献必须体现在完整闭环及真实 workload 中现有策略的可重复 failure。

## 21. 主要风险

### 21.1 可观测性

不同 agent framework 的 call/message/future hook 不统一。需要以最小事件协议为抽象，并至少实现多个真实 runtime adapter。

### 21.2 Ownership 开销

精确 page owner set 可能占用 CPU 内存和调度时间。应 page-align、批量更新，并区分 correctness lock 与仅影响策略的 owner metadata。

### 21.3 HiCache 结构限制

现有 HiRadix 主要支持 leaf-first eviction 和连续 prefix load-back。初版不追求任意 page placement。

### 21.4 PCIe/HBM 干扰

PCIe idle 不代表 D2H 免费。Shadow controller 必须使用实测的 inference slowdown budget，而不是标称带宽。

### 21.5 预测泛化

TraceLab 等数据存在 workload、tool name 和平台偏差。必须使用 taxonomy、hierarchical backoff、survival calibration、在线 residual 和 OOD fallback。

### 21.6 创新性

HiCache 已有 write-through，KVFlow/TokenCake 已有 workflow-aware eviction/prefetch，Agentix 已有 program-level scheduling。BeliefKV 必须通过 runtime causal graph、ownership bridge、reactive oracle gap 和 safe speculation 的组合证明不可被现有策略简单覆盖。

## 22. 当前待决问题

1. 哪些 agent runtime 作为首批 adapter 和真实 workload？
2. SGLang page ownership hook 的最低修改集合是什么？
3. HiRadix 现有 node-level write operation 如何切成可抢占小 chunk？
4. Shared page 在跨 workflow 公平中采用比例 charge 还是其他规则？
5. Reactive baseline 与 offline oracle 的真实 gap 有多大？
6. Tool/subagent parked window 中 useful shadow opportunity 是否足够大？
7. Predictor 是否应先只预测 tool residual time，再逐步加入 agent semi-Markov composition？
8. 对等 multi-agent 的 message arrival 是否具有足够稳定的在线局部性？

## 23. 当前实现状态

截至本文档日期：

- 已实现原子 RCCG reducer、nested call/spawn/join/message 状态转换和事件幂等；
- 已实现 context/page ownership bridge、allocation generation、共享页物理计费、engine/semantic lock 分离和 Radix closure 仲裁；
- 已实现 root-workflow admission/fairness、causal frontier、reactive D2H/H2D 和 ACK 后状态提交；
- 已实现 Prepare-Commit shadow、urgent/shadow 队列、干扰反馈和非抢占 DMA chunk cancel 语义；
- 已实现 hierarchical Kaplan-Meier、semi-Markov context tree、LLM service cost、artifact 训练/加载、OOD fallback 和在线 calibration；
- 已实现 ClawTrace normalizer、页级确定性模拟器、原子实验产物、ablation matrix、CSV 和 bootstrap CI；
- 已实现并保存 SGLang `0.5.2rc1`（commit `18f91eb639084825717c0e3c3c7273492812ab71`）窄补丁，包括 metadata、admission/abort、scheduler safe point、workflow queue ordering 和增量 Radix/HiCache observer；
- 已实现默认关闭的 run-scoped JSONL runtime audit，可审计 causal identity、admission、request lifecycle 和 transfer ACK，且不记录 prompt/observation 内容；
- 控制面单元测试、fake HiCache 故障路径、干净源码 `git apply --check`、AST 契约和修改文件 Python 编译均已执行；
- 已在单卡上启动 Qwen2.5-0.5B-Instruct，并验证未标注旁路、tagged root 和 spawn child 的 `deferred -> admitted -> started -> finished` 路径及 parent-child RCCG 因果边；
- 尚未完成高 HBM pressure、长时间混合 workload、GPU/PCIe 干扰测量和论文 baseline 实验；
- 尚未实现第二个真实 agent framework adapter 和 offline full-future oracle。

因此，当前代码已越过真实 runtime 集成的最小门槛，但不能把单次 smoke 或页级模拟结果当作论文端到端性能结论，也不能声称已经完成生产级 GPU 验证。
