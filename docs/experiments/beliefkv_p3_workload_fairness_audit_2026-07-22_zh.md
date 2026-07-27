# BeliefKV P3 Workload 与 Workflow Fairness 审计

日期：2026-07-22

范围：`experiments/archive/20260722/p3_correctness_only/raw/p3_dynamic/20260721T155058Z`。
本文件解释本轮负载、GPU/Host
现象和当前 workflow fairness 的实现边界，不把该轮实验当作性能 baseline。

> 2026-07-22 更新：该 run 的 48 个 leaf 均为 one-shot 且没有真实工具调用，现已明确降级为
> topology/pressure smoke。替代 workload、动态 task、多轮 child、真实工具 sandbox 和新
> validity gate 见 [P3 真实工具型 Agentic Workload](beliefkv_p3_agentic_workload_2026-07-22_zh.md)。

## 1. 为什么请求和 Host KV 显著减少

| 指标 | P3 dynamic mixed | 先前 Deep Agents planned-8 |
|---|---:|---:|
| workflow | 12 | 8 |
| LLM request | 100 | 848 |
| request/workflow | 8.33 | 106.0 |
| tool call | 0 | 673 |
| wall time | 990.34 s | 3027.15 s |
| peak HBM allocator | 15.612 GB | 16.097 GB |
| peak Host KV | 1.983 GB | 4.924 GB |
| completed D2H bytes | 1.983 GB parent suffix | 52.851 GB |

旧实验只有 3/8 workflow 通过 measurement-valid gate，因此只能用于解释负载量级，不能用于
性能优劣比较。当前实验刻意移除了 repository tool loop，目的是先获得 12/12 正常完成、无
retry/recursion-limit 的动态图 trace。代价是每个 workflow 的调用数从约 106 降到 6--10。

当前每个 workflow 恰好产生：

```text
peer turn_count (2..6) + four one-shot leaf calls = 6..10 LLM calls
```

48 个 FRESH leaf 的 prompt 总量只有 8,345 token，其中 cache hit 6,509，实际新增 prompt
只有 1,836 token。leaf 只收到短 instruction，不继承 parent KV，也不调用工具。每个 workflow
仅有一次 JOIN_WAIT，所以可形成的 parked-parent D2H 机会远少于旧实验的 673 个工具等待窗口。

Host tier 是按迁移需求增长的，不是预先填满的 swap 区。本轮只发生 14 次显式 D2H；每次只
迁移 parent 的 physical exclusive suffix，单 parent 聚合 24.5--341.6 MB。共享 prefix、active
owner 和不满足 Radix closure 的 extent 不会因 Host 尚有空间就被复制。因此 peak Host 只有
1.983 GB 是 workload 与 bundle scope 的直接结果，不表示 KV pool 没有压力。HBM 峰值实际上
达到 15.612/16.106 GB，即 96.93%。

## 2. 为什么 nvtop 可能显示低 GPU utilization

本轮没有启动独立 `nvidia-smi` monitor。`runtime_initialized` 明确记录
`gpu_compute_utilization=false`，所有 `resource_snapshot.gpu_compute_utilization` 都是 null。
因此现有 trace 不能给出可信的平均 SM utilization；肉眼观察的 nvtop 现象不能用 running
request 数代替。

可以确定的事实是请求供给并未长时间断流：

- 100 个请求从首个 start 到最后 finish 跨 989.11 s；
- GPU service observer 的相邻 batch interval 覆盖该窗口的 99.98%；
- decode service-time 加权 batch size 为 11.56；
- batch size 1--4 只占 decode interval 的 36.44/876.56 s，即 4.16%；
- 9,185 个输出 token 的端到端聚合速率只有 9.29 token/s。

所以“当前只有 100 个请求”解释了实验总工作量和 Host churn 下降，却不足以解释 active
窗口内的低 SM utilization。更可能的系统原因包括：

1. `Qwen3-Coder-30B-A3B` 每 token 只激活约 3B 参数，batch 上限又是 16；对 48 GB Ada 卡，
   decode 可能形成短 kernel burst 或 memory/dispatch-bound，而不是持续 compute-bound。
2. server log 明确报告 RTX 6000 Ada 缺少 W8A8 block-FP8 和 fused-MoE tuned config，退回默认
   kernel，并提示性能可能不是最优。
3. 当前 SGLang 0.5.2rc1、HiCache page size 1、BeliefKV physical snapshot 和 observer 都可能
   有额外 CPU/scheduler 开销。已测 controller scheduler-step P99 为 9.36 ms，但没有关闭各
   组件的配对 ablation，不能把 9.29 token/s 单独归因于 BeliefKV。

下一次性能实验必须对同一冻结 trace 以 200 ms 周期采集 GPU utilization、memory
utilization、power 和 PCIe，同时依次关闭 policy snapshot、service observer 和 BeliefKV
control。否则只能说明“吞吐异常低”，无法定位是 kernel、runtime 还是控制面。

## 3. 本轮实际 Workflow 图

![P3 actual mixed workflow](../figures/p3_actual_mixed_workflow_20260721.svg)

图中的 handoff 数字来自 12 个 workflow 的实际 orchestration trace，而不是允许边集合：

- initial Coder 在 JOIN_ALL 后有 11 次交给 Reviewer、1 次交给 Tester；
- 后续 28 次 handoff 为 `R->T=12`、`C->R=6`、`T->C=5`、`R->C=3`、
  `C->T=1`、`T->R=1`；
- 17 次回到已存在 peer context，因而产生 REACTIVATE；
- 12 个 workflow 之间没有语义依赖边，只在单卡 scheduler 和 Radix tree 上竞争资源。

## 4. 当前 Fairness 实现做了什么

当前设计是 root-workflow 级、work-conserving 的 equal-weight attained-service fairness，不是
静态平分 GPU：

1. 所有 invocation 归账到 `root_workflow_id`，fanout 不会创建独立公平账户。
2. SGLang 对 waiting queue 先执行原生 priority，再按 workflow virtual runtime 排序；每轮从
   各 workflow bucket 取一个请求，保留 workflow 内的 causal-frontier 顺序。
3. 上一个 batch 的 wall-clock interval 按该 batch 中各 workflow 的 request 数比例计入
   attained service。
4. HBM admission 先优先选择低于 soft share 的 workflow；所有候选超过 share 后允许最多
   `25% * allocatable HBM` 的有界借用；engine idle 时还能借 reserve。因此空闲 workflow
   不会永久保留一块不能被别人使用的 GPU/HBM。

这套语义比“每个 workflow 固定 1/N 资源”合理，但当前实现仍不足以作为最终公平性算法。

## 5. Fairness 审计发现

### 5.1 权重接口没有接通

`WorkflowAccount` 声明了 weight，但 runtime 的 `WORKFLOW_START` 只调用默认
`register(workflow_id)`。配置、请求 metadata 和 runtime event 都没有可用的 weight/priority/SLO
入口，所以真实运行中全部是 weight=1。更直接的实现缺陷是：即使调用方先执行
`register(id, weight=2)`，后续 `select()` 或 `charge_service()` 内部再次调用默认
`register(id)` 时也会把已有 weight 重置为 1。

waiting queue 的 `ordered()` 也只生成 workflow 的先后顺序；后面的循环仍然从每个 bucket
各取一个 request，并不会按 weight=2:1 生成 2:1 的 slot 频率。因此当前类名虽称 weighted
fair queue，实际运行既没有持久权重，也没有 weighted round-robin/WFQ 语义。异构租户、交互
优先级和 deadline 目前无法表达。

### 5.2 Service 计费不代表真实 GPU work

同一 batch 内按 request count 平分 elapsed time，会把短 leaf、长上下文 decode 和大 prefill
视为相同成本。它也包含 batch selection 之间的控制面和 result-processing wall time。fanout
确实会被多计一份 slot，不会免费占满 batch，但不同 request 的 marginal GPU cost 没有被
正确归因。

### 5.3 公平性只控制新 admission

waiting queue reorder 在 patched SGLang 的 `get_new_batch_prefill()` 中真实生效；但已进入
`running_batch` 的 decode 会一直保留到完成，BeliefKV 不会在 token quantum 上重排或抢占。
一个先进入的高 fanout workflow 仍可能长期占据多个 decode slot。当前 12 个 workflow 都是
fanout=4，无法检验不等 fanout 下的该问题。

### 5.4 不应把 HBM bytes 当作必须均分的服务

当前 `fair_memory_shares()` 对 pending workflow 按 weight 划 soft share。它不是硬分区，但仍
把“公平”近似成瞬时 HBM bytes。KV residency 是可复用状态：给高复用 workflow 更多 HBM
可能同时降低所有 workflow 的 PCIe contention 和 JCT。公平约束应落在可消费服务和可观察
损害上，例如 normalized GPU service lag、admission wait、slowdown 和 HBM-byte-time debt，
而不是要求瞬时 resident bytes 相等。

## 6. 建议的公平性定义

P3/P5 应采用 bounded-lag weighted work-conserving policy：

```text
normalized_service_i = attributed_GPU_service_i / weight_i
floor = min(normalized_service_j for runnable workflow j)
eligible_i iff normalized_service_i <= floor + lag_budget_i
select JointPlan objective only among eligible workflows
```

- `weight_i` 由 workload class、priority 或 SLO 显式传入，缺省为 1；它代表长期竞争份额，不是
  固定瞬时分区。
- 只有多个 workflow 同时 runnable 时才应用 lag gate；其余时间允许单个 workflow 使用全部
  可用计算、HBM 和 PCIe。
- batch 计费按校准后的 marginal service demand 分摊，而非 request count。至少要区分 prefill
  uncached tokens、decode sequence length、batch size 和模型 kernel class。
- HBM 使用 minimum safety floor + unrestricted idle borrowing；将超额 HBM-byte-time 和恢复
  PCIe debt 作为 JointPlan cost，而不是硬性逐 workflow 上限。
- workflow 内仍由 causal frontier、JOIN/HANDOFF urgency 和 physical reclaim value 排序；
  workflow 间只设置最大 service lag，避免公平性覆盖 agent 语义。

结论：当前实现没有静态平分资源，这一点是正确的；但它实际上是“全部 weight=1、权重不可
持久、粗粒度 service accounting、admission-only”的原型。它还不能支持异构 workflow，也没有被本轮同
fanout workload 充分验证。后续性能结论必须加入不等 fanout、不等 context、不等 weight 和
错峰 arrival 的公平性矩阵，并报告 normalized slowdown、service lag、admission tail 和
work-conservation loss。
