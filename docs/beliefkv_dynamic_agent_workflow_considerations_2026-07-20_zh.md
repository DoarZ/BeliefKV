# BeliefKV 面对动态 Agent Workflow 需要注意的地方

日期：2026-07-20

状态：设计约束与实验审查文档。本文补充
[`beliefkv_design_2026-07-14_zh.md`](beliefkv_design_2026-07-14_zh.md)，不取代当前系统设计。

## 1. 文档目的

BeliefKV 不要求应用在运行前提交完整 agent DAG，而是根据运行时事件维护
Runtime Causal Context Graph（RCCG），并据此管理单 GPU 上的 KV residency、迁移、恢复和
调度。

这里的关键难点不是笼统的“workflow 是动态的”，而是明确：

1. 哪些事实在当前时刻已经可观测，可以确定性处理；
2. 哪些未来状态仍然未知，确实需要预测；
3. subagent 和 multi-agent 的动态性是否相同；
4. 因果关系、数据消费关系和物理 prefix 共享是否一致；
5. 当前实验是否真的覆盖了系统声称支持的动态性。

本文给出 BeliefKV 后续设计和实验必须遵守的边界。

## 2. 动态 Workflow 的准确定义

Agent workflow 的动态性不只是工具调用耗时随机，也不意味着整个 workflow 完全不可知。
更准确的描述是：

> 调度器在时刻 `t` 只能看到已经展开的运行前缀 `G_t`；未来节点、边、状态和资源需求由
> 后续 LLM 输出、工具结果和 agent runtime 事件逐步生成。

可以将在线状态表示为：

```text
G_t = (V_t, E_t, S_t, R_t)

V_t: 已创建的 workflow、agent、subagent、tool invocation 和 context
E_t: 已观测的 spawn、return、join、handoff、message 和数据依赖
S_t: RUNNING、READY、WAIT_TOOL、WAIT_CHILD、WAIT_JOIN、TERMINAL 等状态
R_t: 当前 HBM、Host KV、PCIe、Radix residency、lock 和 allocator 状态
```

运行时事件将 `G_t` 更新为 `G_(t+1)`。尚未发生的节点和边不应被当作事实写入 RCCG，只能
作为带置信度的 frontier hypothesis 存在。

因此动态性至少包含以下五个维度：

| 动态维度 | 典型未知量 | 对 BeliefKV 的影响 |
|---|---|---|
| 结构动态性 | 是否 spawn、fan-out、agent 类型、嵌套深度、handoff 目标 | 未来 context 集合和依赖边未知 |
| 路径动态性 | 条件分支、重试、review-revise 循环、提前终止 | 下一个 KV consumer 和复用次数未知 |
| 时间动态性 | tool/agent 时长、join 时间、排队时间 | offload break-even 和 prefetch deadline 未知 |
| 数据动态性 | observation、消息、报告和输出长度 | context growth、prefill 和 KV bytes 未知 |
| 资源动态性 | HBM pressure、PCIe service、lock、allocator fragmentation | 预测动作可能在执行时变得不可行 |

最后一项还包含调度反馈：BeliefKV 自己改变 GPU admission 和执行顺序后，也会改变 child
完成时间和 parent 恢复时间。预测器不能把这些时间视为与调度策略无关的外生常数。

## 3. Subagent 场景的动态性

Subagent workflow 通常是动态展开的 fork-join 树或 DAG。

### 3.1 结构动态性

- parent 是否调用 `task()` 由模型输出决定；
- 一次调用可能创建 0、1 或多个 child；
- child 的 role、task description、execution mode 和 context id 在 spawn 时才确定；
- parent 可能等待全部 child、任意一个 child、quorum，或者提前取消剩余 child；
- child 可能正常 return、异常失败、超时或被取消。

### 3.2 时间与资源动态性

- child 内部会执行多少轮 LLM/tool call 事先未知；
- tool duration、observation size 和 decode 长度具有长尾；
- parent 的 resume readiness 由多个 blocking child 联合决定；
- child 的执行顺序受 GPU admission 和 workflow fairness 影响；
- parent resume 时可以选择 H2D、recompute 或利用仍在 GPU 的 prefix，而不一定必须偿还
  一笔固定 upload debt。

### 3.3 已发生事件与未来预测的边界

```text
SPAWN 之前：child 是否存在、类型和数量是未来变量
SPAWN 之后：parent-child 边、FRESH/FORK/RESUME 模式是已观测事实
JOIN_CREATE 之后：已声明的 join 语义和成员是已观测事实
RETURN 之前：child 剩余执行时间和返回数据量仍是未来变量
JOIN_SATISFIED 之后：parent READY 是已观测事实
```

因此，已经发生的 spawn/join/return 不需要预测。预测器只应覆盖尚未发生的局部转移和剩余
时间，而不是重新猜测 RCCG 已经知道的关系。

## 4. Multi-agent 场景的动态性

Multi-agent 应用可能预定义角色集合，但实际通信和控制路径仍然动态。

### 4.1 常见动态行为

- supervisor 根据当前结果选择下一发言者；
- agent 动态 handoff 给 coder、reviewer、researcher 或 validator；
- debate、review-revise、test-fix 可能循环多轮；
- 消息可以点对点、广播，或通过共享 workspace 间接传播；
- 一个输出可能被多个 peer 消费；
- 终止可能依赖投票、共识、validator、预算或超时；
- 同一 agent/context 可能被多次重新激活；
- 通信图可能包含环，不存在唯一 parent 或单一 join。

### 4.2 与 Subagent 的差异

| 属性 | Subagent | Multi-agent |
|---|---|---|
| 典型拓扑 | fork-join 树/DAG | 有向通信图，可能有环 |
| 控制关系 | parent-child 较明确 | 对等、handoff 或 supervisor routing |
| 恢复目标 | child 完成后通常恢复 parent | 输出可能唤醒任意一个或多个 peer |
| context 生命周期 | child 常为短生命周期 FRESH context | agent context 可能长期存在并反复激活 |
| 关键预测目标 | fan-out、time-to-join、remaining blockers | next consumer、next speaker、再次激活概率 |
| KV 保护重点 | parked parent 与 active child | 多个 peer 的动态 working set |

BeliefKV 不能为整个应用设置一个全局的 `subagent_mode` 或 `multi_agent_mode`。真实应用会在
同一 workflow 中嵌套两种结构，例如：

```text
Supervisor
  |
  +-- Coder <----> Reviewer
  |      |
  |      +-- Test subagent
  |      +-- Search subagent
  |
  +-- Researcher
         |
         +-- Browser subagents
```

策略应由 RCCG 中的局部边类型、同步条件和 context 状态决定。

## 5. 三种关系不能混为一谈

动态 agent workflow 至少包含三个正交视图。

### 5.1 因果控制关系

RCCG 记录：

```text
谁创建了谁
谁正在等待谁
哪个 return/join 会使谁 READY
谁位于当前 workflow progress frontier
```

它主要决定语义活性、关键路径和恢复紧迫性。

### 5.2 数据消费关系

数据关系记录：

```text
谁会读取某个 child report
谁会消费某条 peer message
一个结果是否会被广播给多个 agent
共享 workspace 的修改会唤醒哪些 agent
```

因果 parent 不一定是唯一数据 consumer；对等 agent 之间也可能存在强数据依赖。

### 5.3 物理 Prefix 共享关系

SGLang Radix tree/HiCache 记录 exact-token prefix 的物理共享和 residency。它主要决定：

```text
哪些 KV bytes 只应存储一次
迁移一个 extent 会影响哪些 context
shared prefix 应采用哪个 owner lease
恢复一个 context 需要哪些 ancestor closure
```

因果亲缘不代表物理 KV 亲缘。FRESH child 可能几乎不继承 parent prefix，而不同 workflow 中
使用同一 agent template 的两个 peer 反而可能共享很长的 prefix。

## 6. 当前 P2 Workload 给出的直接证据

当前 P2 高压 workload 使用：

- Qwen3-Coder-30B-A3B-Instruct-FP8；
- 单张 RTX 6000 Ada；
- 8 个 SWE-bench Verified/SymPy workflow，并发 8；
- 16 个 subagent、559 个 LLM request、444 个 tool call；
- Deep Agents planned mode。

实验配置和有效范围见
[`beliefkv_p2_physical_bundle_2026-07-19_zh.md`](experiments/beliefkv_p2_physical_bundle_2026-07-19_zh.md#81-配置与有效范围)。

实际 child 具有独立 `child_context_id`，并被标记为 `FRESH`；Deep Agents 创建 child state 时
清空 parent 消息，只保留 task description。对应实现见
[`deepagents_adapter.py`](../beliefkv/runtime/deepagents_adapter.py) 和环境中的
`deepagents/middleware/subagents.py::_validate_and_prepare_state`。

### 6.1 Parent-child Prefix 结果

同一服务器上的第一个冷启动 fresh child 首次 prompt 为 4,043 tokens。在它之前只有 parent
请求，其 SGLang `cache_hit_tokens` 只有 5。由于实验使用 `page_size=1`，这说明当前 prompt
结构下 parent-child exact prefix 至多约 5 tokens。

冷启动证据位于
[`runtime_events.sglang.jsonl`](../experiments/raw/deepagents_swebench/20260719T114541Z/server-p2-bundle-scope/runtime_events.sglang.jsonl#L41)。
高压运行本身复用了同一服务器之前建立的 Radix cache，不能仅根据高压运行的 hit 值归因
parent-child prefix。

本模型 BF16 KV 的单 token 大小为：

```text
48 layers * 2(K/V) * 4 KV heads * 128 head_dim * 2 bytes
= 98,304 bytes/token
```

因此 parent-child 的公共 KV 约为：

```text
5 tokens * 98,304 bytes/token
= 491,520 bytes
= 0.469 MiB
```

在 16 对 parent-child 上，按 spawn 前最后一次 parent 请求的 `prompt+output` 估算：

| 指标 | Parent | Child 首次 prompt |
|---|---:|---:|
| token 范围 | 829-2,140 | 3,516-4,825 |
| 平均 KV | 1,334 tokens / 125.1 MiB | 4,019 tokens / 376.7 MiB |
| 公共部分占比范围 | 0.234%-0.603% | 0.104%-0.142% |
| 加权公共部分占比 | 0.375% | 0.124% |

相对于 parent-child pair 的物理并集，公共部分约占 0.093%。随着双方继续生成，该比例还会
下降。

### 6.2 真正显著的 Prefix Affinity

当前 trace 更接近以下 Radix 分支：

```text
公共 chat framing：5 tokens / 0.469 MiB
├── parent template branch：约 492 tokens / 46.1 MiB
└── subagent template branch：至少 3,397 tokens / 318.5 MiB
```

高压运行中 child 首请求出现的 3,397-4,774 token cache hit 来自此前 child/template 或 sibling
child，不是 parent KV。由此得到：

例如，高压 workload 中一个 fresh child 的第一次请求命中 3,397 tokens，见
[`runtime_events.sglang.jsonl`](../experiments/raw/deepagents_swebench/20260719T114541Z/planned-8-p2-bundle-scope-pressure/server/runtime_events.sglang.jsonl#L95)。

```text
parent-parent：存在一定 cache affinity
subagent-subagent：存在很强 cache affinity
parent-subagent：几乎没有 cache affinity
```

这证明 RCCG causal edge 不能直接转换成 cache affinity edge。

### 6.3 当前 Workload 的动态性边界

当前 workload 是半动态 subagent workload，而不是高度动态 agent workflow：

- 系统没有接收预定义 DAG，child 在运行时 spawn 后才被 RCCG 观测；
- spawn 时刻、task description、tool path、执行时间和 KV growth 是动态的；
- 但 8 个 workflow 都实际创建了 2 个 child；
- 每个 workflow 基本只有一次 fork/join；
- 没有 nested subagent、动态 quorum、复杂取消或 multi-agent cycle。

因此当前实验能够验证动态时序、动态 context growth 和 HBM pressure，不能证明 BeliefKV 已经
处理高度动态 topology，也不能用于训练或评价完整 workflow graph predictor。

## 7. 对 BeliefKV 调度设计的要求

### 7.1 Workflow 公平性和 Agent 优先级必须分开

parent 和其 child 应共享同一个 workflow service budget，防止 fan-out 较大的 workflow 获得
成倍 GPU 份额。但它们不应因此具有相同的执行优先级：

```text
parent = WAIT_CHILD/WAIT_JOIN：不可运行，不消耗 compute service
child = RUNNING/READY：推进 join，应获得 workflow 内部执行机会
parent 接近或达到 READY：提升 restore/admission 优先级
```

同属一个 workflow accounting group，不等于同属一个 runtime priority class。

### 7.2 Cache Affinity 和 Causal Urgency 必须分开

RCCG 用于计算谁对 workflow progress 紧迫；Radix ownership 用于计算保留哪些物理 KV 更有
价值。两者应联合决策，但不能互相替代。

当前 workload 下建议：

```text
parent-private suffix       -> CONDITIONAL_RESUME，可 shadow/offload
active subagent private KV  -> RUNNING/READY
subagent common prefix      -> 只要仍有 active owner 就保留高 lease
parent-child 5-token root   -> 采用最强 owner lease，但只保护该物理 extent
```

不能因为 0.469 MiB 的 shared root 被 child 使用，就把 parent 的整个 KV context 提升为
`RUNNING`。CausalLease 必须停留在 physical extent/bundle 范围，不能粗粒度传播到整个
context。

### 7.3 Join 前后的优先级必须发生状态转移

parent 从 spawn 到 join 不应始终维持同一优先级：

```text
WAIT_JOIN 且所有 child 尚早
  -> parent private KV 是主要 offload/shadow 候选

部分 child 已完成，但 join 尚未满足
  -> 根据 remaining blocker 和真实资源压力逐步提高恢复价值

JOIN_SATISFIED
  -> parent 进入 READY，触发确定性的 restore/admission 路径
```

是否提前 prefetch 仍需结合 HBM opportunity cost、PCIe service 和调度后的 parent earliest
service time，而不能只预测语义 wake time。

### 7.4 Multi-agent 不能套用固定 Parent Resume 策略

对于 cyclic multi-agent，系统需要维护动态 working set：

- 当前发言者和明确 next consumer 使用强 lease；
- 近期可能被 handoff 的 peer 使用概率性 lease；
- 长期 inactive 但可能再次激活的 agent 不能永久 pinned；
- 一个 message 的多个 consumer 应共享物理计费，但分别维护 readiness；
- repeated handoff 应避免 KV 在 HBM/CPU 之间来回抖动。

## 8. 预测器应该预测什么

BeliefKV 不应尝试一次性预测完整未来 DAG。更可行的目标是对 RCCG 当前 frontier 做局部、
条件化预测。

### 8.1 Subagent Frontier

```text
未来是否继续 fan-out
fan-out count/type distribution
每个 blocking child 的 remaining-time distribution
join mode 和 remaining blocker count
parent latest non-stalling restore time
child/context KV growth distribution
```

### 8.2 Multi-agent Frontier

```text
next speaker / next consumer distribution
handoff destination
同一 agent 再次激活概率
review-revise 循环是否继续
消息 fan-out 和下一轮 context growth
```

### 8.3 共同要求

- 输出分布或多个 scenario，而不是单点时间；
- 显式给出置信度和 OOD 状态；
- 只影响 shadow、prefetch 和候选排序等可撤销动作；
- GPU eviction/commit 仍需真实 HBM pressure 和 physical bundle admission；
- 服务时间估计必须根据当前调度、batch、PCIe 和 allocator 状态在线校准；
- 预测失败时必须退化到事件驱动策略，不能破坏 correctness 或 liveness。

## 9. 运行时观测必须补充的字段

当前 P2 trace 的 `physical_bundle_preview` 记录 closure 聚合 owner，但没有记录逐 extent 的
owner-byte 交集，因此无法从现有审计文件精确重建任意时刻的 parent-child physical shared
bytes。

后续应在 child 第一次 LLM request admission 后增加 `context_prefix_affinity` 事件：

```text
workflow_id
left_context_id / right_context_id
causal_relation
context_mode
shared_prefix_tokens
shared_physical_handles
shared_gpu_bytes / shared_cpu_bytes
left_total_physical_bytes / right_total_physical_bytes
left_share_ratio / right_share_ratio / union_share_ratio
prefix_source_owner_ids
radix_generation_fingerprint
```

还应记录：

- SPAWN 时的 declared join mode 和 member closure 状态；
- 每次 child return 后的 remaining blocker count；
- parent semantic-ready time、scheduler-eligible time 和 actual service time；
- next consumer/handoff 的预测分布与真实结果；
- HBM、PCIe、allocator 和 lock 的 authoritative snapshot；
- 预测动作的 benefit、wasted residency 和错误迁移成本。

高压实验与冷启动 prefix characterization 应分开。高压运行可能继承上一轮 Radix cache，单看
`cache_hit_tokens` 会把历史 child-child 命中误归因给当前 parent-child。

## 10. 正式实验必须覆盖的 Workflow Matrix

只使用“每个 root 固定创建两个 child”的 workload 无法支撑动态 workflow 论证。至少需要
覆盖：

| 场景 | 必须变化的结构 |
|---|---|
| Variable fan-out | 每次 spawn 产生 0-N 个 child |
| Conditional role | 根据 observation 动态选择 Searcher/Coder/Reviewer |
| Nested subagent | child 继续 spawn grandchild |
| Early return/cancel | 任一成功、quorum 或失败后取消剩余 child |
| Unbalanced join | child duration 和 KV growth 呈重尾 |
| Cyclic handoff | Coder-Reviewer-Test 多轮循环 |
| Multi-consumer message | 一个结果唤醒多个 peer |
| Mixed workflow | 对等 multi-agent 内嵌 subagent fork/join |

每类 workload 至少报告：

```text
结构统计：fan-out、depth、cycle、join mode、handoff entropy
时间统计：tool/agent duration、join span、resume delay
KV 统计：context bytes、pairwise affinity、shared physical bytes
预测统计：calibration、top-k coverage、OOD、decision regret
系统统计：workflow JCT、fairness、admission tail、HBM-time、PCIe bytes
```

需要分别设置：

1. 事件驱动 baseline；
2. 只使用静态模板的策略；
3. frontier predictor；
4. offline oracle。

只有 predictor 相比事件驱动 baseline 存在稳定 oracle gap，并且在线策略能够实现其中的显著
部分，预测性 KV 管理才有成为主要贡献的依据。

## 11. 容易出现的错误设计

### 11.1 把动态性等同于 next-agent 分类

只预测 `Searcher/Coder/Reviewer` 不能得到 fan-out、join、duration、KV growth 和 physical
actionability，无法独立指导 KV 迁移。

### 11.2 把 Parent-child 边等同于 Prefix 共享

当前 FRESH workload 已经给出反例：parent-child 只共享约 5 tokens，而不同 child 共享超过
3,000 tokens。按 causal family 粗粒度绑定 residency 会保留错误的 KV。

### 11.3 把 Wake Time 等同于 Restore Deadline

parent READY 后仍可能因 workflow fairness、HBM admission 和运行 batch 等待。restore deadline
应接近 latest non-stalling service time，而不是语义 join 时刻。

### 11.4 把预测结果当成物理执行授权

预测器只能生成候选意图。执行前仍必须由 PhysicalBundleBuilder 和 RadixArbiter 检查 exact
closure、owner、generation、lock、Host/Device capacity 和 allocator 状态。

### 11.5 用单一 Workload 声称支持动态 Agent

当前 P2 workload 的 topology entropy 很低。即使它产生真实 tool call 和 subagent，也不能据此
证明系统对 variable fan-out、nested subagent 或 cyclic multi-agent 有效。

## 12. 当前阶段的设计结论

BeliefKV 面对动态 agent workflow 时应坚持以下原则：

1. RCCG 只保存已发生的运行时事实，未来状态保存在独立 belief frontier；
2. 事件驱动策略是 correctness/liveness 基线，预测只能增强可撤销动作；
3. subagent 和 multi-agent 不使用全局模式开关，而按局部边与状态处理；
4. 因果控制、数据消费和物理 prefix ownership 必须分别建模；
5. workflow 公平性与 workflow 内 agent 优先级必须分层；
6. CausalLease 必须作用于真实 physical extent/bundle，不能由共享 root 粗粒度提升整个 context；
7. 预测目标应是局部 frontier 的 fan-out、next consumer、remaining time 和 KV growth，而不是
   完整未来 DAG；
8. 当前 planned SWE-bench workload 只能证明半动态执行，必须补充结构动态性更强的 workload；
9. cold-cache prefix characterization 与高压性能实验必须独立进行；
10. 所有论文结论都需要事件驱动 baseline、在线预测器和 offline oracle 三者之间的定量差距。

一句话概括：

> BeliefKV 需要同时回答“谁将推动 workflow 前进”和“哪些物理 KV 真正被共同使用”；动态
> agent workflow 的核心风险，正是把这两个不同问题误认为同一个问题。
