# HiCache / HiSparse / Theta KVPool 对 BeliefKV 的启发与差异分析

日期：2026-07-18

## 1. 文档目标与证据边界

本文分析 2026 年 1 月 17 日“蚂蚁开源 x SGLang Meetup”中“蚂蚁面向大规模
分布式推理的 KVCache 多级缓存系统”分享对 BeliefKV 的影响，回答三个问题：

1. 分享中的系统实际解决了什么问题；
2. 哪些能力已经覆盖 BeliefKV 原计划中的通用机制；
3. BeliefKV 还能在哪些 agent-specific 问题上形成可验证的差异。

证据按可靠性分为三层：

1. 用户提供的回放文字和 slides 解读，用于恢复分享结构与 Theta KVPool 的生产设计；
2. SGLang 官方博客、文档和上游源码，用于核对 HiCache/HiSparse 的公开机制；
3. 本地 BeliefKV 固定的 SGLang `0.5.2rc1` 源码，用于确认当前可复现实验能力。

需要先纠正一个术语：这里的“多级缓存”是 GPU HBM、CPU pinned memory 和外部
存储组成的层次存储，不是 CAKE 的跨 Transformer layer 非均匀 KV 压缩。CAKE
与本次分享没有直接对应关系。

公开参考：

- [SGLang HiCache 官方介绍](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)
- [SGLang HiSparse 官方介绍](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/)
- [Mooncake 作为 HiCache L3 后端](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/mooncake_store/README.md)
- [SGLang HiCache roadmap](https://github.com/sgl-project/sglang/issues/18239)

## 2. 被分析系统不是一个单独算法

分享实际串联了三套层次不同的系统：

```text
SGLang HiCache
  通用 prefix KV 的 L1/L2/L3 分层、异步搬运和存储后端抽象

SGLang HiSparse
  DSA sparse attention 下的完整 Host KV + GPU hot buffer + 按需 swap-in

Theta KVPool
  将 HiCache/Mooncake 能力平台化，提供跨实例 metadata、资源解耦和共享 KVPool
```

三者分别回答：

- HiCache：prefix KV 放在哪里、怎样搬运；
- HiSparse：稀疏 attention 实际访问哪些 token 时，怎样缩小 decode 的 HBM 工作集；
- Theta KVPool：怎样把多级 KV 存储变成生产级共享基础设施。

因此，BeliefKV 不能把“增加 CPU 层”“异步 offload”“page 级缓存”或“预取”本身
写成贡献。这些机制已经是 HiCache 的基础能力。

## 3. HiCache 的核心机制

### 3.1 HiRadixTree 作为多级 page table

HiCache 没有放弃 RadixAttention，而是扩展 Radix tree 节点，使其同时记录：

```text
device value    GPU page index
host value      CPU page index
backup state    是否存在下层副本
evicted state   GPU 副本是否已被驱逐
```

prefix 匹配仍由 token radix path 完成。区别是 GPU miss 后，系统可以继续识别 CPU
或 L3 hit，而不是直接 recompute。

这一设计确认了 BeliefKV 当前架构中的一个正确决策：RCCG 不应取代 Radix tree。
Radix tree 是 token prefix 和物理位置的真相源，RCCG 只提供运行时因果语义。

### 3.2 L1/L2/L3 数据路径

```text
L1 GPU HBM
  <-> L2 CPU pinned memory
  <-> L3 file / Mooncake / 3FS / NIXL / AIBrix
```

HiCache 支持：

- GPU 到 CPU 的 write-through、selective write-through 和 write-back；
- CPU 到 GPU 的 load-back；
- L3 到 CPU 的后台 prefetch；
- CPU 到 L3 的异步 backup；
- prefix 过短时跳过传输的收益阈值；
- 多个 node operation 的合并，减少小 DMA 和 kernel launch。

### 3.3 计算与传输 overlap

CPU 命中后，HiCache 不必等所有层恢复完成才开始计算。它可以在计算 layer N 时，
加载 layer N+1，并通过 per-layer completion event 建立依赖。

这意味着 BeliefKV 不能继续用简单的：

```text
restore_time = bytes / nominal_PCIe_bandwidth
```

作为真实 stall 估计。应区分：

```text
raw_transfer_time
overlapped_transfer_time
unhidden_critical_path_time
```

BeliefKV 实验中已经观察到 callback 和 allocator 开销使实际 H2D 明显偏离理想带宽。
HiCache 的 layer-wise overlap 进一步说明，最终要优化的是未隐藏 stall，而不是总传输时间。

### 3.4 Host layout 与 I/O kernel

GPU attention 偏好 layer-first，而外部 I/O 偏好 page-first。HiCache 为此支持不同
Host layout，并使用专门 kernel 完成布局转换：

- `layer_first`：匹配计算布局；
- `page_first` / `page_first_direct`：匹配 page 级外部 I/O；
- `page_head`：支持按 head 切片和异构 TP 共享。

这部分对 BeliefKV 的启发是：策略层不应自行拼装 tensor copy。BeliefKV 应提交
物理迁移意图，由 HiCache 数据面决定合并、布局和 kernel。

### 3.5 HiCache 调度能力的准确边界

HiCache 确实有 scheduling pipeline，但其主要调度对象是 I/O operation：

- 什么时候 write/load/prefetch；
- 是否等待、超时或 best-effort；
- 多个 operation 如何合并；
- 如何与 layer compute overlap；
- TP ranks 如何保持命中长度一致。

因此不能笼统声称“HiCache 不做调度”。更准确的差异是：

> HiCache 调度缓存 I/O，BeliefKV 试图联合决定哪个 agent invocation 先运行，以及
> 哪些物理 KV bundle 应为该运行顺序占用 HBM/PCIe。

## 4. HiSparse 的核心机制及适用边界

Sparse attention 降低 attention 计算量，但 Top-k 每层每步动态变化。若全量 KV
仍常驻 GPU，系统依然受 HBM capacity 限制。

HiSparse 使用：

```text
CPU pinned memory：完整 KV
GPU HBM：每个 request 固定大小 hot buffer
decode step：Top-k -> hit/miss diff -> LRU victim -> swap-in
```

专用 kernel 同时完成：

1. 检查 Top-k token 是否已在 GPU；
2. 选择 hot-buffer victim；
3. 更新 token-to-slot/page table；
4. 将 miss KV 从 Host 搬到 device。

### 对 BeliefKV 的直接启发

- 物理驻留粒度应由访问语义决定，不能总以完整 request/context 为单位；
- CPU 完整副本和 GPU working set 可以由不同管理器维护；
- 相邻访问集合的 overlap 可以显著降低迁移量；
- hot buffer 大小必须与并发度联合配置。

### 不能直接借用的部分

HiSparse 的 Top-k 是模型 attention kernel 给出的精确访问集合。普通 agent workflow
没有同等精确的未来 KV token access oracle。把“下一 agent 概率”类比成 Top-k token
会掩盖预测误差，也无法保证模型质量。

因此，BeliefKV 不应声称实现 workflow 版 HiSparse，除非目标模型本身使用 DSA，且
BeliefKV 只负责 request/agent 级 admission，HiSparse 负责 request 内 token working set。

## 5. Theta KVPool 的生产化贡献

Theta KVPool 将 engine 内 cache controller 扩展成平台服务：

```text
SGLang Engine / Dummy Client
          |
       KVMaster             metadata 和路由
          |
KVPool Real Client sidecar  内存注册、RDMA、SSD/DRAM 资源
```

主要价值包括：

- engine 生命周期与存储/RDMA 资源解耦；
- KV metadata 可以跨 P/D instance 共享；
- Real Client 直接访问注册 buffer，减少中间复制；
- 不同 TP 通过 `page_head` 和细粒度 head shard 共享 L3 KV；
- KVPool 可扩展到 DSA indexer、Mamba state 等非标准 KV 状态。

Theta 的优化目标主要是多机、PD 分离、远端共享和平台稳定性。BeliefKV 的目标是
单卡并发 MAS 下的 HBM/PCIe 竞争。硬件规模不同，但 Theta 证明了一点：策略必须建立
在稳定的物理对象、异步 ACK 和可观测传输之上，不能只在 simulator 中操作逻辑 KV。

## 6. HiCache 已经覆盖 BeliefKV 的哪些部分

| BeliefKV 候选表述 | HiCache/Theta 已有能力 | 结论 |
| --- | --- | --- |
| GPU/CPU 多级 KV | L1/L2/L3 分层 | 不能作为创新 |
| page 级迁移 | HiRadix page table 和 Host page pool | 不能作为创新 |
| 异步 offload/prefetch | CacheController 后台 pipeline | 不能作为创新 |
| 传输与计算 overlap | layer-wise loading | 不能作为创新 |
| PCIe 空闲时 write-through | 多种 write policy | 单独表述不成立 |
| 根据命中价值选择 backup | selective write-through | 必须证明 agent 因果信号更有效 |
| storage hit 后预取 | L3 opportunistic prefetch | 不能作为创新 |
| page 合并/批量 DMA | `CacheOperation.merge` | 应直接复用 |
| 跨实例 KVPool | Mooncake/Theta | 不属于单卡 BeliefKV 的贡献 |
| physical Radix ownership | HiCache 有物理树，但无 agent owner | BeliefKV 仍有差异空间 |
| workflow-aware agent scheduling | 未由 HiCache 提供 | 可研究，但需端到端证据 |

尤其需要删除或降级以下宽泛表述：

> “现有系统只管理已经存在的 KV，BeliefKV 首次控制其生命周期。”

HiCache 已经管理 backup、load-back、eviction、prefetch 和 storage residency。BeliefKV
真正可能不同的是 agent causal state 如何改变这些动作，而不是“生命周期”这个词。

## 7. HiCache 没有表达的 agent-specific 信息

### 7.1 Radix path 不等于 agent causal graph

HiRadixTree 能回答：

- 哪些请求共享 token prefix；
- prefix 当前在哪一层；
- 从下层恢复哪些连续 page。

它不能回答：

- context 为什么暂停；
- parent 是否正在等待 FRESH child；
- 哪个 child 是 join straggler；
- 哪个 invocation 的完成会解锁后续 agent；
- peer agent 的 message 是否已使另一个 context ready；
- 一个 context 逻辑结束后是否仍有共享物理 owner。

这正是 RCCG 与 Radix ownership bridge 的必要性，而不是重复建设另一棵 cache tree。

### 7.2 低频但因果确定的复用

HiCache 的 selective write-through 依赖通用 cache hit/hotness。Agent 场景存在一种重要
反例：

```text
parent spawn FRESH children
parent 进入 WAIT_CHILD/JOIN
parent KV 历史命中次数可能只有 0 或 1
但只要 workflow 不取消，join 后 parent 必须恢复一次
```

这是“低频、单次、但因果上高确定性”的未来复用。频率策略可能把它视为 cold；全量
write-through 又会产生过多流量。RCCG 可以为 parent 建立 conditional reuse certificate：

```text
condition = JOIN_SATISFIED(parent)
reuse_required = true
deadline = unknown until child progress narrows
```

这个 certificate 本身只能证明值得保留一个下层副本，不能自动证明应该立即从 GPU
驱逐或提前恢复。后两者仍需结合 HBM pressure、child remaining time 和实际 transfer
service curve。

### 7.3 Agent 调度会改变未来 HBM 工作集

HiCache 的输入通常是 scheduler 已选中的 request 和当前 prefix hit。动态 MAS 中，选择
哪个 frontier agent 运行，会影响：

- 哪个 context 的 KV 增长；
- 何时产生 tool wait 或 FRESH child；
- 哪些 parent 转为 parked；
- 哪些 join/message context 变成 READY；
- 下一批 admission 需要多少 HBM。

因此，BeliefKV 的研究问题不是给 HiCache 再加一个 eviction score，而是：

> 在 workflow fairness 和物理 HBM/PCIe 约束下，联合选择 agent execution 与 KV
> residency，使缓存动作服务于未来可运行集合，而不是服务于孤立 request 的 hit rate。

### 7.4 共享页的因果价值不能按 request 独立计算

一个 Radix page 可能由多个 peer agent/context 共享。HiCache 知道物理 prefix，但不知道
owner 的 agent 状态。BeliefKV 的 ownership bridge 可以对每个物理 bundle 聚合：

```text
owner contexts
RCCG liveness
ready/parked/join state
physical marginal bytes
ancestor closure
```

这样才能避免按 context 重复计费，也避免迁移仍被 active peer 使用的共享页。

## 8. 建议收敛后的 BeliefKV 分层架构

### 8.1 明确职责边界

```text
Agent runtime
  -> 产生 spawn/tool/join/message/return 事件

RCCG control plane
  -> 维护 invocation/context liveness 与 causal frontier

Causal lease and admission policy
  -> 选择 workflow、agent 和 KV residency intent

Radix ownership bridge
  -> 将 context intent 转成 physical bundle + closure

SGLang HiCache data plane
  -> merge、layout、kernel、DMA、ACK、L2/L3 backend
```

HiCache 应被视为 BeliefKV 的执行底座和强基线，而不是竞争系统中需要重新实现的模块。

### 8.2 Causal lease 作为统一元数据

可以把 RCCG 信息转换成有限类型的 lease，而不是训练端到端 MLP 输出迁移动作：

```text
RUNNING_LEASE
  context 正在执行，KV 不可迁移

READY_LEASE
  已可运行，参与 HBM admission 和 agent scheduling

CONDITIONAL_RESUME_LEASE
  parent 等待 tool/child/join，未来使用有因果条件

SPECULATIVE_LEASE
  仅由历史预测支持，允许 shadow，不覆盖安全约束

DEAD_LEASE
  无活跃 owner，可作为高优先级 victim
```

一个 physical bundle 的最终 lease 由所有 owner 中最强的 lease 决定。它把 RCCG、共享
Radix page 和 HBM admission 放入同一数据结构，同时保持 SGLang 为物理真相源。

该机制的创新风险是“lease”在缓存系统中并非新概念。论文贡献必须落到 agent runtime
产生的 conditional causal lease、共享物理 bundle 映射和端到端 failure characterization，
而不能仅靠重新命名状态。

### 8.3 预测器与系统状态分离

历史 workflow 数据只预测外生行为：

```text
P(next event | RCCG context)
P(tool/child remaining time > t | observations)
fanout / next-agent distribution
confidence and OOD
```

当前 HBM、PCIe backlog、实际 bundle bytes 和 overlap profile 是在线精确系统状态，不应
混入 MLP 作为最终 action predictor。否则模型会学习旧调度策略造成的资源分布，策略改变
后产生 policy leakage 和 covariate shift。

控制器在运行时组合两类信息：

```text
workflow belief + exact physical state -> constrained action
```

这比“MLP 输出 offload score”更可靠，也更容易解释泛化失败。

## 9. 可立即吸收的工程设计

### 9.1 保留和复用

1. 使用 HiCache ACK 后提交 residency，不预测 DMA 已完成；
2. 使用 operation merge，避免 BeliefKV 制造大量小命令；
3. 区分 raw bandwidth 与 unhidden stall，记录 per-layer overlap；
4. 使用 page-first Host layout，不在策略层进行 tensor layout transform；
5. 以 load/recompute 实测 crossover 决定最小 prefetch size；
6. 将 write-through/selective/write-back 作为正式 baseline；
7. 保持 Radix prefix closure 和 allocator generation 检查。

### 9.2 当前实现需要新增的观测

建议每个 transfer 记录：

```text
command_submit_ts
first_layer_ready_ts
last_layer_ready_ts
compute_wait_begin/end
raw_bytes
physical_closure_bytes
merged_operation_count
actual_unhidden_stall_ms
source/target tier
RCCG reason and lease type
```

当前只记录 callback 总耗时无法判断 HiCache overlap 是否有效，也无法正确训练未来的
transfer cost model。

## 10. 版本与对比实验要求

### 10.1 两条实验线必须分开

BeliefKV 当前固定：

```text
SGLang v0.5.2rc1
commit 18f91eb639084825717c0e3c3c7273492812ab71
```

该版本已有早期 HiCache、三种 write policy、CPU/GPU load/write 和 storage prefetch，
但材料中的 HiSparse、Hybrid PoolTransfer、完整 page_head 异构 TP 和后续 Mooncake 能力
来自更新的上游版本。

因此需要：

1. **Pinned comparison**：同一 `0.5.2rc1` 上比较 SGLang/HiCache 与 BeliefKV，隔离策略收益；
2. **Current-upstream comparison**：在可移植版本上比较最新 HiCache，证明收益不是旧后端缺陷；
3. 不把 HiSparse 纳入普通 MHA/GQA 单卡主实验，除非使用 DSA 模型；
4. 不把 Theta 多机扩容数字与 BeliefKV 单卡结果直接比较。

### 10.2 必须包含的 baseline

```text
SGLang Radix LRU
HiCache write_back
HiCache write_through
HiCache write_through_selective
HiCache + reactive event policy
BeliefKV without predictor
BeliefKV full policy
offline oracle
```

若预测模块不能稳定超过 reactive policy，应降级为可选优化。

## 11. 需要优先验证的 HiCache failure

### F1：Hotness 错过因果确定但低频的 parent reuse

测量 `WAIT_CHILD/JOIN` parent 中：

- selective write-through 未建立 Host 副本的比例；
- parent resume 时 recompute/H2D 的成本；
- causal certificate 相比 write-through 增加多少无效写流量。

### F2：Request-local prefetch 与 agent readiness 不一致

HiCache 通常在当前 request prefix lookup 后触发 L3 prefetch。测量 parent/tool context 在
真正进入 request queue 前是否存在可利用窗口，以及 BeliefKV 提前恢复是否减少未隐藏
stall，而不是只增加 HBM residency time。

### F3：独立 I/O policy 与 agent scheduler 造成资源反转

检查是否存在：低优先级 workflow 的大 prefetch 占用 HBM/PCIe，阻塞已经 READY 的
关键 agent。必须记录发生频率和 JCT 影响，不能只构造示例。

### F4：逻辑 context 大小与物理可释放字节不一致

测量 shared prefix、ancestor closure、active lock 导致的 estimated/actual freed bytes 偏差。
这是 BeliefKV ownership bridge 最直接、也最容易验证的系统价值。

## 12. 审稿人式评审

### 12.1 可以说服审稿人的部分

1. BeliefKV 明确建立在 HiCache 之上，不重复多级缓存数据面；
2. 用真实 trace 证明 generic hit-count/write policy 在动态 MAS 中出现稳定 failure；
3. RCCG lease 能识别低频但因果确定的 future reuse；
4. agent scheduling 和 physical KV admission 使用同一个 HBM/PCIe 约束；
5. 所有收益都以 actual freed bytes、unhidden stall 和 workflow JCT 衡量；
6. 在同版本和最新上游两条实验线上都成立。

### 12.2 当前不能说服审稿人的部分

1. “HiCache 不懂 agent，所以 BeliefKV 更新颖”不是性能证据；
2. parent 等 child 时 offload 属于直接事件响应，单独不足以成为 Major contribution；
3. PCIe idle 时建立 shadow copy 与 write-through 高度接近；
4. 用下一 agent MLP 替换 hit count 只是启发式 score；
5. 只对比 SGLang `0.5.2rc1` 会被质疑利用旧版本缺陷；
6. simulator 中有收益但真实 HiCache overlap 后收益消失，不构成系统贡献。

### 12.3 Go/No-Go 条件

只有满足以下条件，agent-causal cache policy 才值得作为核心方向继续：

- 至少三类真实 MAS workload 中均出现 HiCache baseline failure；
- failure 不是单个异常 trace，且占 HBM pressure 决策的比例足够高；
- offline causal oracle 相比最强 HiCache policy 将 workflow JCT 或 P95 JCT 降低至少 10%；
- 实际无效 D2H/H2D/recompute bytes 有显著下降；
- 加入最新 HiCache 的 merge、overlap 和 prefetch 后收益仍存在；
- predictor OOD 时，reactive causal policy 不劣于 HiCache baseline。

若上述条件不成立，RCCG/ownership 应保留为工程安全层，BeliefKV 的 Major contribution
需要转向其他问题，而不是继续增加预测模型复杂度。

## 13. 最终结论

这套系统给 BeliefKV 最重要的启发不是再增加一个存储层，而是明确研究边界：

```text
HiCache / Theta KVPool
  解决 KV 在不同介质和实例之间如何高效存储、共享和移动

BeliefKV
  需要证明动态 agent 因果状态能够改进“谁先运行、谁占 HBM、谁使用 PCIe”
```

目前最可信的差异是：HiCache 的 hotness/prefix metadata 无法表达 FRESH subagent、
tool wait、join 和 peer message 产生的 conditional future reuse，也不能将这些信息与
workflow fairness、physical shared-page accounting 和 HBM admission 联合起来。

但该差异是否足以成为论文贡献，必须由 F1-F4 characterization 和 oracle gap 决定。
在得到证据前，应把 HiCache 视为 BeliefKV 的强数据面和强基线，而不是一个容易通过
“加入 agent graph”超越的弱对手。
