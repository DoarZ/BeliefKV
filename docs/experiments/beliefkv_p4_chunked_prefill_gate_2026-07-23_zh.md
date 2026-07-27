# BeliefKV P4 24-workflow 单次 GPU Correctness Gate

日期：2026-07-23

状态：P4 未通过。第一次运行定位 chunked-prefill starvation；修复后只执行了一次独立 GPU
复验，admission 与迁移链路恢复活性，但 JointPlan planning budget 和 workload 终止性未过 gate。

## 1. 实验目的

本次实验用于验证 P4 Visible-but-Gated Incremental Admission 与 JointPlan shadow 路径在真实
24-workflow 压力下的 correctness、batch fill、运行并发和 safe-point 开销。它不是策略性能
对比，也不包含预测模块。

固定配置如下：

- 模型：`Qwen3-Coder-30B-A3B-Instruct-FP8`，单张 RTX 6000 Ada；
- 24 个 SWE-bench Verified SymPy workflow，并发上限 24；
- 每批 8 个 workflow，批间隔 20 秒；
- mixed/cyclic workload，静态要求模型创建 2--4 个 subagent；
- SGLang `max_running_requests=32`、`max_total_tokens=163840`、
  `chunked_prefill_size=4096`、`mem_fraction_static=0.952`；
- `joint_policy_enabled=false`、`joint_policy_shadow_mode=true`、预测关闭；
- workload 仅提交一次，失败后没有自动重试或重新启动实验。

原始数据位于：

```text
experiments/archive/20260727/superseded_raw/p4_joint_shadow_24/20260723T045754Z/
```

## 2. 观测结果

24 个 Docker sandbox 均完成 preflight，24 个 root LLM request 均写入 `llm_submit` 并与本地
SGLang 建立连接，但 workload request 没有一个进入 prefill。`server.log` 中仅有服务启动时的
两个探测 prefill，不包含 workload prefill。

关键计数：

| 指标 | 结果 |
|---|---:|
| workflow / root request | 24 / 24 |
| completed workflow | 0 |
| admission epoch | 93,883 |
| issued ticket | 0 |
| GPU 采样 | 910 |
| GPU utilization 平均值 / 最大值 | 0% / 0% |
| GPU 显存占用 | 46,850 MiB |
| runtime audit 大小 | 约 304 MiB |

因此本次数据不能用于评价 JointPlan 的吞吐、JCT、迁移收益或 stale rate，只能作为 admission
correctness failure 的证据。高显存占用来自模型权重和预分配的 15 GB KV pool，不代表存在
有效 KV 工作集。

## 3. 根因

SGLang 的 `rem_chunk_tokens=4096` 表示当前 scheduler epoch 可处理的 prefill chunk，而不是
一个请求允许拥有的最大未缓存 prompt。原实现将完整 `uncached_prompt_tokens` 与该值直接比较：

```text
full uncached prompt > current epoch chunk budget
    -> prefill_token_budget
    -> no ticket
    -> request remains in waiting queue
    -> next epoch repeats the same rejection
```

所有初始 prompt 均超过 4096 token，因而形成确定性的 admission starvation。重复的 started/
finished audit 还将该错误放大为 304 MiB 日志，但日志写入不是零 GPU 利用率的根因。

## 4. 修复

`AdmissionTicketCompiler` 现在区分两个预算：

- prefill ticket 只消耗当前 epoch 实际可执行的 `min(uncached_prompt_tokens,
  remaining_prefill_tokens)`；
- HBM feasibility 仍按请求的完整 `estimated_incremental_bytes` 保守检查；
- SGLang `PrefillAdder` 继续拥有最终 chunk、allocator 和 batch 决定权。

新增回归测试覆盖“20-token prompt、10-token epoch budget”场景，要求长 prompt 获得 10-token
首块 ticket；另保留 HBM 不可行时跳过大请求并继续扫描后续请求的测试。修复后全量 CPU 测试为
`345 passed, 7 skipped`。

## 5. 结论

P4 GPU gate 尚未通过。本次失败排除了 workload 并发不足、Docker 初始化失败和模型未主动
spawn 等解释，直接证明 admission ticket 的 chunked-prefill 语义此前不正确。下一次 GPU
gate 必须从该修复版本开始，但不应把下一次运行描述为本次结果的成功样本或覆盖本次负结果。

## 6. 修复后独立 GPU 复验

修复后于同日执行了一次新的 24-workflow 复验，不复用第一次运行的数据，也没有自动重试：

```text
experiments/archive/20260727/superseded_raw/p4_joint_shadow_24/20260723T062631Z/
```

该运行同时包含后续修复：request-path restore dependency、WAIT_RESTORE 优先恢复、较大可行
closure 优先、P4 增量触发去除逐 token fairness revision，以及 admission/validation 审计压缩。
CPU 回归结果为 `347 passed, 7 skipped`。

### 6.1 负载和压力确实形成

本次 workload 共记录 1,418 次 LLM submit、1,392 次 LLM result、1,351 次完整工具调用、
61 次动态 spawn 和 35 次 child return。16 个 mixed workflow 均创建 subagent，共建立 16 个
join，其中 6 个在停止前自然闭合。服务端成功返回 1,381 个 HTTP 200，未观察到 CUDA OOM。

| 指标 | 结果 |
|---|---:|
| 运行时长 | 5,736.1 s |
| GPU utilization 平均 / P50 / P95 / 最大 | 23.0% / 1% / 100% / 100% |
| GPU 显存平均 / 最大 | 47,783 / 47,798 MiB |
| SGLang token usage 平均 / 最大 | 83.9% / 96.0% |
| running request 平均 / 最大 | 12.66 / 32 |
| waiting queue 平均 / 最大 | 37.48 / 60 |
| KV HBM / Host 峰值 | 15.0 / 30.0 GiB |

这排除了“未形成 GPU-ready 并发”或“KV pool 没有压满”的解释。GPU utilization 的低 P50 与
100% P95 同时出现，说明执行呈现强烈 burst：长上下文 prefill/decode 可跑满 GPU，但调度、
工具和 join 边界之间仍有大量空窗。

### 6.2 物理迁移正确性

迁移时间线和结构化校验位于：

- `experiments/processed/p4_20260723T062631Z/kv_transfer_timeline.html`；
- `experiments/processed/p4_20260723T062631Z/kv_transfer_timeline.json`；
- `experiments/processed/p4_20260723T062631Z/transfer_validation.json`。

共记录 220 次 transfer telemetry，其中 208 次发生物理 DMA。累计实际 D2H 为 13.34 GB，H2D
为 3.49 GB；202 次完成，18 次因 extent mutation 或 engine lock 被拒绝。所有预期 DMA 均有
dispatch/ACK/telemetry，未出现 orphan、缺失 ACK、时间倒置或 byte bound violation。

retry guard 没有出现相同 physical fingerprint 的重复失败或零字节 retry storm。physical bundle
partial/reject rate 为 6.87%，reclaim realization ratio 为 91.87%；`locked_gpu_snapshot_ratio`
仍高达 82.62%，说明 engine lock/closure 仍是主要物理约束。service curve holdout 的传输时延
低估率为 21.95%，高于 10% gate，当前校准模型不能用于 P5 主动决策。

时间分布并非稳定的分层周转：176 次完成的 D2H 和前 25 次完成的 H2D 全部发生在首个 20 分钟，
之后约 75 分钟只有结束前一次 96 KiB H2D。HBM 在约 3.75 分钟达到 90%，Host 在约 14.4 分钟
达到 90%，之后两层长期接近满载。显式 D2H 涉及 77 个 context，而只有 16 个 context 出现显式
H2D；其中一部分是未再次消费的 one-shot child，但 SGLang native demand-load 未进入 BeliefKV
telemetry，因此不能把剩余差额都解释为无用 offload，当前 H2D 仍只是下界。

旧时间线的 `Protected/non-evictable KV` 实际是
`allocator HBM - page-index GPU bytes`，不是策略保护的低优先级 context。该序列现已重命名为
`Untracked allocator delta`；新 trace 另外直接记录 engine-locked、closure-blocked、migratable
和 dual-resident physical KV。20 分钟后旧差值平均约 0.19--0.31 GB，不能解释约 15.87 GB 的
HBM 占用。
真实缺陷是当前在线 planner 在 Host 满时只 drop unowned page，不会对仍有 owner 的 inactive KV
选择 drop/recompute；H2D 完成后又保留 `DUAL_CLEAN` Host copy。相较 reference LRU 在 Host
不足时选择 `DROP` 的行为，这可能造成 Host 饱和后失去继续 reclaim 的能力，但在补齐 native
telemetry 和同 trace 配对实验前，不能宣称整体性能已经劣于 LRU。

### 6.3 JointPlan shadow 未通过

离线统计位于：

```text
experiments/processed/p4_20260723T062631Z/joint_shadow_summary.json
```

13,151 个已发布并验证的计划中，12,072 个（91.8%）为 fully fresh，1,079 个为 partial；
说明前述增量触发和局部 validation 修复有效。plan age P50/P95 为 273.3/571.1 ms，仍在
750 ms TTL 内。

但 12,922 个计划以 `planning_budget_exceeded` fallback。直接原因是运行配置仍使用
`max_joint_plan_budget_ms=1`，而 planner 自身报告的 P50/P95 已达 13.9/33.0 ms，每次通常只完成
1 次 package evaluation。worker 端到端 compute（包含 snapshot build 和 planner）P50/P95 为
157.7/290.8 ms，其中 snapshot build 为 141.7/277.6 ms；safe-point validation P50/P95 为
25.3/106.4 ms。因此这不是简单的 stale-plan 问题：planner、snapshot 和 validation 三段均需
分别优化和预算，不能通过放宽 TTL 掩盖，也不能仅把 budget 调大后宣称通过。

观测路径同样未过生产开销 gate：95.6 分钟产生约 1.8 GiB `runtime_audit.jsonl`、603 MiB
压缩 policy snapshots 和 203 MiB request-token trace。后续必须把高频完整快照改为采样摘要、
delta 或离线按需重建；这些文件可用于本次 characterization，但不能作为默认性能实验配置。

### 6.4 Workload 终止性未通过

运行约 95.6 分钟后，只有 3 个 workflow 自然写出终态，且均为单次 API 请求超过 900 秒；
固定观察窗口内结果数不再增长，但服务端仍维持约 7--10 个 running 和 19--23 个 waiting
request。停止时额外 3 个 workflow 因服务关闭写出 connection error，其余 18 个保持 incomplete，
不得计入成功或失败完成率。

这表明请求级 liveness 已修复，但 `agentic` backend 的一次 graph activation 可以在
`recursion_limit=512` 内产生大量内部 LLM/tool loop；外层 `max_turns=18` 不能约束单次 activation
的墙钟时间。该运行可以用于 KV 压力、迁移和 lock characterization，不能用于端到端 JCT 或
策略收益比较。

## 7. Gate 结论和下一步

本次复验证明：chunked admission starvation 和 retry storm 已修复，真实 HBM/Host 迁移在满压
下可持续执行；但 P4 仍失败于三个独立硬门槛：

1. JointPlan 必须先降低 snapshot/plan/validation 成本并建立可解释的分阶段 budget，不能直接
   把 1 ms 改成数百毫秒；
2. workload 必须给单次 agent activation 增加语义完成和墙钟/调用预算，使 24 个 workflow 能
   正常 RETURN/JOIN，而不是依赖 API timeout 收尾。
3. 补齐 native HiCache demand-load/write-back telemetry，并增加 Host-copy eviction、terminal
   cleanup 和 Host 满时的 drop/recompute；否则 D2H/H2D usefulness 与 LRU 对比均不可验证。

在这三个问题修复前，不进入 P5 在线统一调度，也不重复本次 GPU 实验。

## 8. 2026-07-23 Host 生命周期修复状态

后续实现已将 Deep Agents 启动器从固定 `hicache-ratio=2` 改为显式
`hicache-size=96` GB，并允许通过 `HICACHE_SIZE_GB=128/156` 覆盖。控制面启动时不再信任
配置文件中的旧 Host 容量，而以 HiCache allocator 的 token capacity 乘实际
`kv_bytes_per_token` 为准。

runtime 收到真实 `RETURN`、`INVOCATION_CANCEL` 或 `WORKFLOW_END` 后，会在释放 context owner
前记录仅由该 terminal context 独占的 Host-resident generation handles。新的
`DROP_TERMINAL_PRIVATE/DROP_HOST` 路径在 ACK 后执行两种状态转换：

- `CPU_ONLY -> DEAD`：仅删除无 owner 的 Radix leaf；
- `DUAL_CLEAN -> GPU_ONLY`：只释放 Host 副本，不影响 GPU copy。

共享 prefix、persistent context、engine-locked/in-flight extent 均不会被误删；失败路径进入
physical-fingerprint retry guard，不按 scheduler tick 重试。该修复当前只通过 CPU/controller 与
HiCache backend 定向测试，尚未用 96 GB Host pool 重跑长 GPU workload。native demand-load
telemetry 和 Host 满时 inactive-owned drop/recompute 仍是 P4 未关闭项。
terminal handle 收集与已有 owner-release 合并为同一次遍历，因此 RETURN 后晚到的 final
cache-finish/Radix rebind 也会在下一 safe point 被捕获，而不会增加第二次全图扫描。本次修改后的
时间线将 terminal Host drop 显示为 `reclaim` 事件和 Host occupancy 下降，但不把释放字节误计为
PCIe DMA。全量 CPU 回归为 `355 passed, 7 skipped`。
