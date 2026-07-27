# BeliefKV P2：Physical Causal Lease 与原子 Bundle 执行

日期：2026-07-19，2026-07-21 更新
状态：P2 CPU 与真实 GPU 可靠性 gate 已通过；同 manifest 配对性能 gate 待完成

## 1. 本阶段解决的问题

P1.5 仍然以 context 为候选单位，执行前才把 context 展开成多个 Radix node extent。
这会把“逻辑上希望迁移多少 KV”和“此刻真正可以完整释放多少 HBM”混为一谈。

冻结的 P1.5 压力 trace 中：

- 418 条 transfer telemetry 中只有 47 条 completed，310 条 partial，61 条 rejected；
- D2H 共 300 条，其中 279 条 partial，partial 比例为 93%；
- 279 条 partial D2H 全部包含 `node is engine-locked`；
- 这些 partial D2H 选择了 67,574,366,208 bytes closure，实际完成
  10,154,606,592 bytes，聚合 realization 只有 15.03%；
- guard 记录了 1,112 次 `node_locked` blocker。

上述数据证明 context 级选择与物理可行 extent 之间存在严重失配，但不能单独证明锁是
JCT 的主要瓶颈。

进一步检查固定 SGLang 0.5.2rc1 数据面后发现一个具体执行缺陷：

1. `HiRadixCache.write_backup(node)` 会调用 `inc_lock_ref(node)`；
2. `inc_lock_ref` 会沿 node 到 root 的整条路径增加 lock；
3. 旧 backend 按 planner 给出的逐页顺序执行，先备份 child 后可能由本次操作锁住 parent；
4. 同一命令随后处理 parent 时，会把自己刚制造的 lock 判断为外部 `node_locked`；
5. 完成阶段又遍历无序 set，存在先尝试 parent eviction、后处理 child 的风险。

因此 P2 不是只替换 victim score，而是同时改变候选单位、失败身份和后端提交协议。

## 2. P2 架构

```text
Runtime events                         SGLang physical facts
      |                                         |
      v                                         v
     RCCG                              PageOwnershipIndex
      |                              owner / generation /
      v                              topology / lock / tier
CausalLeaseProjector                           |
      |                                         |
      +----------------+------------------------+
                       v
             PhysicalBundleBuilder
        D2H descendant closure / H2D ancestor closure
        exclusive suffix / shared subtree action scope
        strongest owner lease / exact marginal bytes
        blocker set / generation fingerprint
                       |
                       v
            ReactiveTransferPlanner
         selects an eligible versioned bundle intent
                       |
                       v
        bundle-scoped TransferAttemptGuard
                       |
                       v
             RadixArbiter revalidation
                       |
                       v
      HiCacheNodeCommandBackend safe point
       full preflight -> DMA -> atomic commit/rollback
                       |
                       v
        exact ACK + transfer telemetry + resync
```

## 3. 核心实现

### 3.1 Causal lease

新增 `beliefkv/policy/leases.py`，将 RCCG 的已观测状态投影为资源承诺：

```text
RUNNING_LLM                         -> RUNNING
CREATED / READY / RETURNING         -> READY
WAIT_TOOL / WAIT_CHILD / WAIT_JOIN  -> CONDITIONAL_RESUME
预测但尚未创建的 context           -> SPECULATIVE
workflow ended / 全部 terminal      -> DEAD
未知 owner                          -> RUNNING safety pin
```

共享 physical extent 的 bundle lease 取所有 owner 中的最强状态：

```text
RUNNING > READY > CONDITIONAL_RESUME > SPECULATIVE > DEAD
```

因此 parked parent 不能迁移仍由 running child/peer 使用的 shared prefix。lease 只表达语义
承诺，不替代 SGLang 的 engine lock、allocator 或 cache coherency。

### 3.2 Physical bundle preview

新增 `beliefkv/runtime/bundles.py`。一个 bundle 是一次动作必须共同验证的最小物理集合，
而不是一个 context 的全部 KV：

- D2H/COMMIT：从候选 node 向下闭合所有 GPU-resident descendants；
- H2D：从 CPU-only target 向上闭合到首个 GPU anchor；
- shared extent：聚合全部 owner，只对物理字节计费一次；
- blocker：lock、active reader、semantic pin、in-flight、unsealed、owner lease、
  ancestor/descendant closure、Host/Device capacity；
- fingerprint：覆盖 handle generation、parent/children topology、residency、owner 和动作
  extent 的锁状态。

blocked preview 仍保留潜在 `page_actions` 和 required bytes，用于 characterization；
`intent()` 对 blocked preview 直接拒绝，不能进入执行队列。

#### 3.2.1 Bundle scope

一个 context 的完整 Radix 路径通常同时包含共享 prefix 和该 context 的私有 suffix。仅有
closure 并不足以区分两者：D2H closure 还可能被迫纳入其他 context 的 descendant；H2D
closure 则可能包含一个仅用于校验、并不发生迁移的 GPU anchor。因此 P2 显式记录本次
`page_actions` 的跨 context 影响范围：

```text
EXCLUSIVE_SUFFIX
  所有被改变 residency 的 extent 均无 foreign owner
  该动作不会改变任何其他 context 的物理驻留

SHARED_SUBTREE
  至少一个被改变 residency 的 extent 具有 foreign owner
  包含 shared prefix，或 closure 被迫带入其他 context 的私有分支
```

scope 只由实际 action extent 计算，不由整个 closure 计算。因而 H2D 中已经在 GPU 的共享
ancestor anchor 不会把一次纯私有 suffix 恢复误判成 `SHARED_SUBTREE`。bundle 同时记录
`exclusive_action_bytes`、`cross_context_action_bytes` 和 `foreign_owner_context_ids`，使审计
可以重建策略是否越过 context 边界。

策略边界如下：

| 控制路径 | 允许的 scope | 原因 |
|---|---|---|
| parked parent/context shadow | `EXCLUSIVE_SUFFIX` | 预测性 prepare 不应改变 peer/child 驻留 |
| admission liveness frontier spill | `EXCLUSIVE_SUFFIX` | 为保护请求让路时不能连带驱逐其他 READY context |
| 普通 HBM pressure | 独占优先；无独占候选时才允许共享 | 先释放目标私有 KV，共享动作退化为显式全局 reclaim |
| H2D prefetch | 两者均可 | 一次共享恢复可能同时服务多个 owner，不能禁止 |

因此“offload parent”现在严格表示优先迁移 parent-private suffix。若物理 closure 必须触及
child/peer，命令会被标成 `global_shared_bundle_reclaim`；此时命令中的 `context_id` 只是候选
枚举和记账代表，不再声称整个 bundle 属于该 context。

### 3.3 Planner 与 retry identity

`ReactiveTransferPlanner` 不再按 `sum(context_pages)` 估计释放量，而是：

1. 枚举 context 对应的 physical bundles；
2. 跳过 blocker 非空或 marginal reclaim 为 0 的 bundle；
3. 将 `EXCLUSIVE_SUFFIX` 和 `SHARED_SUBTREE` 放入不同候选池，严格先考察独占池；
4. 在池内使用 exact reclaim、copy bytes、causal distance 和 lease 排序；
5. 把完整 handles、actions、scope、bytes、bundle id 和 fingerprint 固化进 command；
6. 某个 bundle 被拒绝后继续考察同 context 的其他独立 bundle。

P1.5 attempt key 从 context-wide identity 收紧为：

```text
(context_id, context_epoch, command_kind, bundle_id, generation_fingerprint)
```

同一 bundle 的 fingerprint 变化触发 event-gated release；一个 locked bundle 不会封锁同一
context 的另一个可迁移 suffix。

### 3.4 Arbiter 二次校验

`RadixArbiter` 在 dispatch 前重新构建同 bundle：

- bundle 不存在：`EXTENT_MUTATED`；
- fingerprint 改变：返回新的 authoritative blocker；
- handles/actions/bytes 与 intent 不一致：拒绝 stale plan；
- 完全一致：只返回 intent 中的 exact physical actions。

因此 planner preview 不是执行授权，只是版本化意图。

### 3.5 后端两阶段执行

`HiCacheNodeCommandBackend` 对带 physical intent 的命令执行：

```text
Phase A: full preflight
  resolve 所有 closure handles，而不只是 action handles
  检查 generation、parent/children topology、lock/loading、closure
  检查可观测的 Host/Device allocator capacity
  任一失败 -> 0 side effect REJECTED

Phase B: execute and commit
  D2H submit: parent -> child
  wait all D2H callbacks
  D2H commit: child -> parent
  H2D: 对 deepest leaf 调用一次原生 load_back
       由 HiCache all-or-nothing 加载完整 evicted ancestor chain
  callback 后再次校验完整 closure topology
```

D2H 在全部 DMA 完成前不释放任何 GPU extent。若 commit 中出现不可预期错误，只 ACK 已经
实际释放的 extent，PageOwnershipIndex 得到精确 PARTIAL。H2D 若在 callback/校验阶段失败，
按 deep-to-shallow 回滚已恢复 extent；回滚失败的 residual GPU extent 才进入 PARTIAL ACK。

这里的“原子”指在单 scheduler safe point 上先全量校验、再提交 closure residency，不声称
底层 allocator 提供硬件事务。

### 3.6 2026-07-20 可靠性修复

对失败的 `planned-8-p2-bundle-scope-pressure` 做 command、allocator 和 server log 对齐后，
确认此前的 `DEVICE_CAPACITY` 不能全部解释为真实容量不足：

- 281 次零字节 H2D reject 的 required closure 只有 1 至 5 tokens；
- 固定 SGLang 0.5.2rc1 的 `HiRadixCache.load_back_threshold=10` 会对这些恢复返回 `None`；
- backend 将该返回统一解释为 device allocation failure，guard 又会在 allocator epoch 改变后
  释放 attempt，因而同一 fingerprint 最多重复 7 次；
- scheduler fatal 前 `page_index GPU tokens=151,427`，allocator used tokens 为 149,922，
  恰好相差 1,505 tokens；server 原生统计也出现 `#token: -1505`，说明部分 live Radix index
  同时出现在 allocator free/release list；
- 四个 workflow 的 `FileNotFoundError` 来自 SGLang 崩溃后 Unix control socket 消失，
  该异常被 callback 当作 tool/workflow failure 向上传播，并不是真实工具错误。

本轮修复把上述三条路径分开处理：

1. BeliefKV H2D 使用 `load_back(force=True, allow_eviction=False)`。`force` 仅绕过上游小传输
   策略阈值，不能绕过 mem quota 或 allocator 失败；`allow_eviction=False` 禁止一次 prefetch
   在 backend 内隐式选择其他 victim。
2. atomic H2D 在提交前读取 HiCache controller 的 authoritative allocator，验证 tree/controller
   引用同一 allocator、完整 closure 可容纳，并在提交后检查精确 token reservation delta；
   planner 的可用容量同时扣除尚未进入 batch 的 admission reservation。
3. 同一 context 的 proactive H2D 与 engine-visible request 互斥。若 controller 在同一 tick
   同时产生 prefetch 和 admission，request reservation 保留，但请求直到 H2D terminal ACK 后
   才进入 SGLang；进入前重新执行 `init_next_round_input()`，避免原生 prefill 再次加载或使用
   stale `prefix_indices/host_hit_length`。
4. scheduler safe point 对 `available + evictable + protected > max_total_tokens` 做 fail-closed
   检查。对当前 token-granular allocator，只移除 free/release list 与 live Radix value 的交集，
   记录 `allocator_radix_resynchronized`；无法解释的偏差继续抛错，禁止静默掩盖。
5. HiCache completion queue 逐 ACK 隔离异常，不再因一个 malformed/unknown callback 停止清空
   后续 ACK。与在途 bundle 匹配的 callback bookkeeping failure 会使命令结构化拒绝；H2D
   bundle 回滚 GPU residency，D2H 不执行 eviction commit。
6. runtime event control sink 失败不会再改变 agent/tool 轨迹。完整 trace 先写入本地 sink，
   control failure 作为 `runtime_control_delivery.degraded` 进入结果，实验因此判为控制面降级，
   但不伪装成 workload failure。

## 4. 审计与指标

新增事件：

- `context_lease_issued`；
- `bundle_lease_aggregated`；
- `physical_bundle_preview`。

preview 记录 closure handles、scope、exclusive/cross-context action bytes、foreign owners、
unique/GPU/CPU bytes、copy/reclaim/locked bytes、lease、fingerprint 和逐 extent blocker。
`validate_transfer_audit()` 新增
`physical_bundle_characterization`，输出：

- eligible/blocked/shared-owner preview 数；
- exclusive/shared scope preview 与 dispatch 数、两类 action bytes；
- locked GPU snapshot ratio；
- physical/action bytes 比例；
- preview 与 dispatch fingerprint 匹配；
- bundle partial/reject rate；
- predictable blocker residual reject；
- expected/actual reclaim bytes、realization ratio 和绝对误差。

这些字段可直接用于下一轮 HBM/Host 时间线可视化和 F1/F4 characterization。

## 5. 已验证的不变量

确定性测试和 fake HiCache 故障注入覆盖：

1. RUNNING owner 覆盖 parked/dead owner，shared page 不重复计费；
2. locked 大 extent 不会隐藏同 context 的独立可回收 extent；
3. H2D 自动包含 CPU ancestor，GPU anchor 的 lock 不被误判为 H2D blocker；
4. stale fingerprint、generation 和非 action anchor topology mutation 均无法 commit；
5. 任一 child 在 preflight 时 locked，parent 不产生 host shadow side effect；
6. D2H 严格按 shallow-to-deep submit、deep-to-shallow eviction；
7. H2D ancestor chain 只发起一个原生 `load_back(deepest_leaf)`；
8. Host capacity blocker 在 preview 阶段出现，并保留 required bytes；
9. bundle 级 retry 不发生 context-wide false suppression；
10. preview-to-ACK characterization 指标可由审计文件重建。
11. parent private suffix 与 shared-root closure 被赋予不同 scope；pressure 私有优先，
    shadow/frontier spill 不会提交跨 context bundle。
12. 1 至 5 token 的 H2D closure 能绕过 native threshold，但不能触发隐式 Radix eviction；
13. H2D 容量不足在 DMA 前零副作用拒绝，tree/controller allocator 分叉 fail closed；
14. 同 context request 必须等待 H2D ACK，并在 admission 前重新匹配 authoritative prefix；
15. callback bookkeeping failure 使 atomic H2D 回滚且 allocator available 恢复；
16. 构造的 live-Radix/free-list overlap 能被精确回收，修复后 allocator invariant 成立；
17. control socket 消失不会中断真实 workflow，且结果中保留可审计的 degraded 标记。

截至 2026-07-20，`beliefkv` 环境全量测试为 `204 passed, 5 skipped`；
`beliefkv-agents` 的 Deep Agents/SWE-bench/runtime matrix 相关测试为 `39 passed`。跳过项是测试
sandbox 不允许创建 Unix datagram socket，不是逻辑失败。

## 6. 尚未完成的证据

P2 现在只完成了机制和确定性 correctness，不能据此声称性能提升：

- 旧 P1.5 trace 没有新 bundle preview 事件，不能离线伪造 P2 的真实 realization ratio；
- 必须在同一冻结 workload 上重跑 P1.5 与 P2，比较 D2H partial 率、actual/planned reclaim、
  admission wait、workflow JCT 和 controller overhead；
- PageOwnershipIndex 与 authoritative HiCache 之间仍存在 scheduler-tick 级 TOCTOU，P2 通过
  revalidation/rollback 保证收敛，但可能产生 wasted DMA；
- Host/Device preflight 无法完全消除 allocator fragmentation；
- D2H 没有上游 multi-node transaction，只能在 scheduler thread 中按 closure 顺序提交；
- 原子 bundle 可能增加命令数量或降低可释放粒度，必须报告 planning/dispatch overhead；
- 原失败 run 的 control-socket `FileNotFoundError` 已与真实 tool failure 隔离；正式性能实验仍
  要求 workflow 正常完成、无 recursion-limit，并且 `runtime_control_delivery.degraded=false`。

## 7. 下一步真实 gate

使用同一模型、KV pool、并发 workflow、trace hash 和 runtime 事件序列，至少比较：

```text
P1.5 context-level reactive + retry guard
P2 physical-bundle reactive + atomic backend
P2 oracle bundle eligibility replay
```

P2 进入 P3 前至少需要满足：

- D2H predictable-lock partial 数相对 P1.5 降低至少 90%；
- 相同 bundle/fingerprint 的重复 reject 为 0；
- `dispatch_without_matching_preview_count == 0`；
- actual/planned reclaim realization 显著高于 P1.5 的 15.03% 锚点；
- 无 location divergence、OOM、watchdog 和未 ACK command；
- admission P95/JCT 不因 bundle 粒度与额外 planning 明显退化。

在该 gate 通过前，P2 应被描述为修复 context/physical mismatch 的必要基础设施，而不是论文
Major contribution 或已成立的性能优化。

2026-07-20 检查时，两张 RTX 6000 Ada 分别已有约 15.8 GiB 和 36.0 GiB 显存占用，且均为
其他实验进程。因此没有启动 SGLang 或执行 P2 复跑；GPU gate 状态仍为 **pending**。

## 8. 真实 GPU 高压实验

### 8.1 配置与有效范围

本轮只评估完整 P2 实现，不进行 P1.5/P2 性能配对。实验配置为：

- 模型：`Qwen3-Coder-30B-A3B-Instruct-FP8`；
- SGLang：`0.5.2rc1`，单张 RTX 6000 Ada，TP=1；
- `mem-fraction-static=0.952`，`max-total-tokens=163840`；
- HiCache ratio 2、`write_back`，BeliefKV prefetch 开启，prediction/shadow 关闭；
- workload：SWE-bench Verified 固定的 8 个 SymPy workflow，并发 8，planned mode；
- 共观测 16 个 subagent、559 个 LLM request 和 444 个 tool call；
- 运行时长 1482.05 s。

本轮达到了需要迁移 KV 的压力区间：

- GPU 板级显存峰值 47,946 MiB / 48,519 MiB，即 98.82%；
- 配置的 KV pool 物理镜像峰值 14.983 GiB / 15 GiB，即 99.89%；
- SGLang resident-token pressure 峰值 91.91%；
- Host KV 峰值 7.335 GiB；
- GPU active sample 中平均 compute utilization 为 38.58%，峰值 100%。

板级显存、KV pool 物理镜像和 SGLang resident-token pressure 的分母不同，不能互换。

### 8.2 P2 scope 结果

context offload 共 131 次：

| Scope | Completed | Partial | Rejected |
|---|---:|---:|---:|
| `EXCLUSIVE_SUFFIX` | 129 | 0 | 0 |
| `SHARED_SUBTREE` | 2 | 0 | 0 |

offload 计划释放与 ACK 实际释放均为 22,823,141,376 bytes（21.256 GiB），reclaim
realization 为 100%。112 次操作产生实际 D2H DMA，共 11.788 GiB；其余 offload 复用了已存在
的 Host shadow，不需要再次复制。这个结果说明 private-first scope 和 D2H 原子 bundle 修复在
本轮 trace 中生效：没有 context offload 因 `node_locked` 或 closure 约束变成 partial。

但 H2D prefetch 尚未修复完整：

| Scope | Completed | Partial | Rejected |
|---|---:|---:|---:|
| `EXCLUSIVE_SUFFIX` | 152 | 16 | 283 |
| `SHARED_SUBTREE` | 4 | 0 | 5 |

288 次 H2D reject 中，281 次为 `device_capacity`，6 次为 `extent_mutated`，1 次为
`node_locked`；16 次 partial 均为 `descendant_closure`。validator 仍发现 76 次相同失败
attempt 重试，同一 physical fingerprint 最多提交 7 次。因此 P1.5 retry storm 只被抑制，
没有被消除。

所有 979 个 dispatch 都收到了 ACK，无 orphan/missing ACK、byte-bound violation 或 watchdog。
累计 residency change 为 21.256 GiB offload、7.319 GiB prefetch，另有 97.311 GiB
`drop_unowned` reclaim。后端实际 DMA 为 11.788 GiB D2H 和 7.329 GiB H2D；residency
change 与 DMA bytes 不等价，因为 Host shadow 复用和 H2D partial rollback 都会造成差异。

### 8.3 性能与正确性结果

- admission wait：p50 38.80 ms，p95 70.34 s，p99 284.33 s，最大 345.52 s；
- transfer dispatch-to-ACK：p50 98.60 ms，p95 402.32 ms；
- 物理 DMA callback：p50 32.32 ms，p90 124.72 ms；
- server decode batch throughput：均值 86.29 token/s，p50 53.65，p95 235.92；
- 8 个 workflow 中 4 个正常返回，2 个通过本地 correctness gate；另外 4 个由
  `FileNotFoundError` 终止。

运行末尾 SGLang allocator self-check 崩溃：

```text
max_total_num_tokens = 163840
available_size       = 13918
evictable_size       = 151427
protected_size       = 0
```

三者合计 165,345，比容量多 1,505 token。validator 同时发现 18,011 个 resource snapshot 中
有 9,140 个不满足 HBM mirror/allocator subset 关系，最大账本差异为 542,048,256 bytes。
Host page index 与 Host residency 则保持一致。这表明故障边界位于 GPU Radix residency 与
allocator 记账，而不是普通的显存不足。

### 8.4 Gate 判定

| P2 gate | 结果 | 证据 |
|---|---|---|
| D2H predictable-lock partial 降低 | 通过本轮绝对检查 | 131 次 offload 全部 completed |
| 相同 bundle/fingerprint 重复 reject 为 0 | 失败 | 76 次 identical failed retry，单 fingerprint 最多 7 次 |
| dispatch 均有 matching preview | 通过 | mismatch 为 0 |
| planned/actual reclaim realization | 通过 | 100% |
| 无 location divergence/OOM/invariant failure | 失败 | allocator self-check 崩溃 |
| admission/JCT 不明显退化 | 失败 | admission p95 70.34 s；运行非正常结束 |

因此本轮可以证明 P2 的 **D2H bundle scope 修复有效**，但不能声称 P2 整体已经可靠，更不能
将这些 JCT 数字用于最终性能对比。下一步应先修复 H2D capacity admission、partial rollback
后的 Radix/allocator 双账本一致性，以及 workload adapter 的 `FileNotFoundError`，再用相同
manifest 重跑。

### 8.5 产物

- 高压迁移时间线：
  [`high_pressure_kv_transfer_timeline.html`](../../experiments/processed/p2_20260719T114541Z/high_pressure_kv_transfer_timeline.html)
- 高压遥测校验：
  [`high_pressure_transfer_validation.json`](../../experiments/processed/p2_20260719T114541Z/high_pressure_transfer_validation.json)
- 原始汇总：
  [`summary.json`](../../experiments/archive/20260727/superseded_raw/deepagents_swebench/20260719T114541Z/planned-8-p2-bundle-scope-pressure/summary.json)

实验退出后已确认两张 GPU 均为 0 MiB used，没有遗留 SGLang 或 workload 进程。

## 9. 2026-07-21 可靠性修复版真实复验

### 9.1 配置与 workload 边界

复验固定旧 P2 run 的前 8 个 SymPy 实例和运行参数：

- Qwen3-Coder-30B-A3B-Instruct-FP8，SGLang `0.5.2rc1`，TP=1；
- GPU 0 为 RTX 6000 Ada，GPU 1 全程 0 MiB，`mem_fraction_static=0.952`；
- KV pool 为 163,840 tokens / 15 GiB，HiCache ratio 2、write-back；
- planned mode，并发 8，每个 workflow 创建 2 个 FRESH child；
- prediction/shadow 关闭，reactive prefetch 开启；
- 运行 3027.15 s，16 个 child、848 个 LLM request、673 个 tool call。

8/8 workflow 都产生 `workflow_end`，control delivery failure 为 0，没有 recursion-limit 或
runtime exception。其中 3/8 通过本地任务 correctness gate；其余 5 个以 `blocked` 或
`no_patch_needed` 返回。实验脚本因此按任务正确性返回非零退出码，但这不是 SGLang、控制 socket
或 KV 数据面失败。性能主表只能使用 3 个 measurement-valid workflow，P2 可靠性检查则使用
全部系统终态和物理审计。

### 9.2 修复闭环结果

| 检查项 | 修复版结果 |
|---|---:|
| request started/finished | 848 / 848 |
| transfer dispatch/ACK | 1502 / 1502 |
| missing/orphan/order/byte violation | 0 / 0 / 0 / 0 |
| watchdog / scheduler exception | 0 / 0 |
| identical failed/zero-byte retry | 0 / 0 |
| unknown blocker / active blocked attempt | 0 / 0 |
| dispatch without matching preview | 0 |
| HBM mirror exceeds allocator | 0 / 38,594 snapshots |
| Host page-index mismatch | 0 / 38,594 snapshots |
| offload planned/actual reclaim | 52,851,376,128 / 52,851,376,128 bytes |
| reclaim realization | 100% |

物理 DMA 包含 184 次 D2H、16,011,264,000 bytes，以及 520 次 H2D、17,784,274,944
bytes。另有 87 次 H2D 在 DMA 前安全拒绝：79 次因为同 context 已成为 engine-visible，8 次
因为 Radix extent generation 已变化。旧 run 的 281 次小 closure `device_capacity` reject 降为
0；79 个 event-gated blocker 全部在匹配事件后释放，只产生 6 次 suppression，没有 retry
without release。

23 个 command-level partial 全部来自原生 `DROP_UNOWNED` 遇到非 leaf 的
`descendant_closure`；bundle D2H 没有 partial/reject，What-if/JointPlan 后续不能把这部分原生
drop partial 误记为 context bundle 失败。峰值 HBM mirror 为 16,096,690,176 bytes，Host KV
为 4,924,145,664 bytes；SGLang resident-token pressure 峰值为 92.10%。

### 9.3 尚未通过的性能与测量门槛

- admission wait p50 为 41.22 ms，p95 为 61.46 s，p99 为 208.26 s；没有同代码、同语义路径
  的 P1.5 配对 run，不能据此声称 JCT 或 admission 性能改善；
- chronological service-curve holdout 的总体低估率为 13.48%，D2H 为 23.53%，高于 10%
  目标；这说明当前 P90/P10 curve 对长跑 D2H 尾部仍不够保守；
- `controller_timing_summary` 没有在 SIGINT shutdown 路径写出，无法由本 run 证明 controller
  p99 开销低于 scheduler tick 的 5%；
- 79 次 engine-visible H2D reject 和 8 次 extent mutation 虽然正确 fail closed，但仍是
  preview-to-submit TOCTOU 开销，后续 JointPlan 应避免先产生注定失效的 prefetch intent。

因此本轮只将 **P2 真实可靠性 gate** 标记为通过。P1.5/P2 配对性能 gate、service-curve
保守性和 controller timing 仍是进入主动 P4 前的待办。

### 9.4 产物

- [修复版 summary](../../experiments/archive/20260727/superseded_raw/deepagents_swebench/20260721T054654Z/planned-8-p2-reliability-fixed/summary.json)
- [transfer validator](../../experiments/processed/p2_20260721T054654Z/transfer_validation.json)
- [HBM/Host KV 迁移时间线](../../experiments/processed/p2_20260721T054654Z/kv_transfer_timeline.html)
- [时间线结构化数据](../../experiments/processed/p2_20260721T054654Z/kv_transfer_timeline.json)

复验结束后已正常停止 SGLang，删除所有临时 SWE-bench 容器，并确认两张 GPU 均为 0 MiB。

## 10. 2026-07-21 P3 审计后的适用范围修订

后续 12-workflow mixed run 发现，SGLang 正常 request admission 会对 host-hit prefix 调用
`HiRadixCache.init_load_back()`。这类 native demand-load 不经过 BeliefKV command queue，当前
`transfer_telemetry.jsonl` 不记录其 start/complete/bytes。

因此第 9 节的可靠性结论应严格解释为：

- BeliefKV 显式 command 的 bundle、ACK、retry、allocator 和 telemetry 不变量通过；
- 不能由该 validator 推导所有 SGLang HiCache DMA 都被记录；
- 当前 H2D 总量、PCIe utilization 和迁移 timeline 是显式 command 下界；
- P2 bundle correctness 结论不撤销，但 P1/P2 的全系统 transfer callback coverage 重新打开。

修复要求是在 HiCache `write/load` enqueue 与 ACK drain 处记录 native operation，并与
BeliefKV command 按 node/operation identity 去重。补齐前不得用现有 timeline 计算完整 PCIe
占用或将未记录 demand-load 当成 recompute。证据见
[P3 动态并发 GPU Characterization](beliefkv_p3_dynamic_gpu_validation_2026-07-21_zh.md)。
