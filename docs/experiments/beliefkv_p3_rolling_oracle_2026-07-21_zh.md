# BeliefKV P3 Rolling Physical Replay 与 Trace-order Oracle 检查点

日期：2026-07-21

状态：CPU/rolling 机制、正式动态并发 GPU characterization 和冻结 trace 已完成；真实
physical/service 对齐与 positive joint synergy gate 未通过

> 归档说明（2026-07-22）：rolling 机制仍有效，但本报告中的 workload-derived 数值来自
> correctness-only 负载，只能用于回归，不可作为真实性能结果。

## 1. 本轮完成内容

本轮补齐了旧 `FrozenTracePlanEvaluator` 无法随候选调度重建物理状态的问题。新增路径为：

```text
frozen semantic/request DAG + exact anonymized token paths
  -> candidate topological execution order
  -> calibrated prefill/decode queue-service replay
  -> rolling token-Radix match/materialize/lock/unlock
  -> rolling GPU/CPU/BOTH residency and allocator accounting
  -> closure-safe D2H/H2D/drop action regenerated after every event
  -> O0/O1/O2/O3 cost and HBM/Host transfer timeline
```

具体实现：

- `simulator/token_radix.py`：增加 token-exact tiered Radix、共享 owner、active request lock、
  D2H descendant closure、H2D ancestor closure 和稳定 bundle identity；
- `simulator/rolling_physical.py`：增加 rolling allocator、无副作用容量预检、batch path 并集
  计费、reactive LRU 与 hindsight next-use residency policy；
- `simulator/rolling_queue_service.py`：在每个 prefill/decode quantum 前重新计算 prefix hit、
  materialization demand、HBM admission 和物理动作；
- `policy/joint_oracle.py`：增加 arm-aware evaluator、完整或显式截断的 topological-order 搜索；
- `experiments/counterfactual_trace.py`：冻结 `observed_request_order`，不再把任意排序伪装成
  observed baseline；
- `scripts/run_rolling_joint_oracle.py`：生成 manifest、O0-O3 结果、物理事件 JSON 以及 HBM/Host
  KV 时间线 HTML。

所有 timing evidence 仍要求 calibrated service model、完整 token identity 和已知 cache-reset
epoch。缺失任一条件均 fail closed。

## 2. 单 workflow 离线机制跑

输入为 2026-07-21 成功完成的一个真实模型 mixed workflow：

- Qwen3-Coder-30B-A3B-Instruct-FP8；
- 1 个 mixed workflow、8 个 LLM request；
- 模型在初始 turn 自主选择 4 个 FRESH child；
- trace 包含 12 条 request dependency、完整 token identity 和已知空 Radix epoch；
- trace 为 `semantic_race_sensitive`，因此结果只能作为 optimistic mechanism bound；
- HBM 容量人为压到 1,000 tokens，即 98,304,000 bytes；Host 容量 4,000 tokens；
- GPU service 使用独立 holdout 通过的模型
  `queue-service-c3e9318662e6a828b2f57256`；
- PCIe 暂用配置值 24 GB/s、setup 0.08 ms，不是新的 transfer holdout 标定。

搜索穷尽 120 个合法 topological order，没有截断。结果如下：

| Arm | Workflow JCT | D2H | H2D | Peak HBM |
|---|---:|---:|---:|---:|
| O0 observed-arrival order + reactive LRU | 38,859.629 ms | 191,791,104 B | 0 B | 98,009,088 B |
| O1 best order + reactive LRU | 38,859.629 ms | 191,791,104 B | 0 B | 98,009,088 B |
| O2 observed-arrival order + hindsight KV | 38,853.685 ms | 0 B | 0 B | 97,812,480 B |
| O3 best order + hindsight KV | 38,853.685 ms | 0 B | 0 B | 97,812,480 B |

关键结论：

```text
joint synergy gap = min(O1, O2) - O3 = 0 ms
```

该 trace **不支持 JointPlan major contribution**。O2/O3 相比 O0 只减少约 5.94 ms，原因是
hindsight policy 直接 drop 无未来复用的分支，避免 7 次 D2H；收益约为 JCT 的 0.015%，且没有
H2D resume。长达约 34.76 秒的外部 causal delay 主导 JCT，改变 sibling 顺序没有作用。

这是一项有价值的负结果，但不能外推到并发多 workflow。当前 trace 还没有 workflow fairness
竞争，脚本 manifest 明确标记 `fairness_constraint=not_modeled_trace_order_upper_bound`。

有效产物：

- `experiments/archive/20260722/p3_correctness_only/processed/p3_dynamic_20260721T100408Z/mixed_counterfactual_workload_v2.json`；
- `experiments/archive/20260722/p3_correctness_only/processed/p3_dynamic_20260721T100408Z/rolling_oracle_1000t_v2/`；
- 其中 `O0..O3_kv_timeline.html` 可视化 HBM/Host occupancy 和 D2H/H2D；
- 同级 `rolling_oracle_1000t/` 是 HTML summary 适配失败的首轮工件，不是最终结果。

## 3. 正式动态并发 GPU 运行

GPU 释放后完成两类 12-workflow、TP=1、163,840-token KV pool 运行。

第一轮 `20260721T144331Z` 使用 6 个 cyclic 和 6 个 mixed workflow：12/12 正常完成，75 个
model request，峰值 allocator HBM 为 15,026,061,312 B，显式 D2H/H2D 为
714,964,992/189,235,200 B。stable prompt 后 parent/peer reactivation 可命中 9.9K--12.5K
tokens。该 run 的迁移对象全部是 cyclic root context，没有直接覆盖 mixed parked parent。

第二轮 `20260721T155058Z` 使用 12 个全 mixed workflow，定向验证 parent：

- 12/12 semantic completion，100 个 model request；
- 48 SPAWN、12 JOIN、40 HANDOFF、17 REACTIVATE；
- 峰值 HBM 15,611,854,848 / 16,106,127,360 B，即 96.93%；
- 12 个 parent 都发生 D2H，6 个发生在 JOIN_WAIT 内；
- 显式 D2H/H2D 为 1,982,988,288/532,119,552 B；
- 22 个 DMA 全部 completed，无 partial/reject/zero-byte/retry storm；
- 一笔无消费者 H2D 为 199,655,424 B，占显式 H2D 的 37.52%；
- 三个 context 出现无 BeliefKV H2D command 的 SGLang native demand-load。

完整证据见 [P3 动态并发 GPU Characterization](beliefkv_p3_dynamic_gpu_validation_2026-07-21_zh.md)。

## 4. 多 Workflow Rolling Replay 结果与否决

第一轮正式 run 冻结出 75-request、12-workflow、99-dependency-edge trace。bounded-lag fair
search 使用 4 个候选，执行 10 次唯一 simulation：

| Arm | mean JCT | transfer |
|---|---:|---:|
| O0 | 115.658 s | D2H 8.814 GB |
| O1 | 114.532 s | D2H 9.012 GB |
| O2 | 115.602 s | 0 |
| O3 | 114.555 s | 0 |

`min(O1,O2)-O3 = -22.684 ms`，没有 joint synergy。更重要的是该 replay 与真实 run 明显
不一致：真实 mean JCT 为 589.08 s，D2H/H2D 为 0.715/0.189 GB；rolling O0 却为
115.66 s、8.814/0 GB。因此这些 O0--O3 数字只用于定位模型错误，不能进入论文性能表。

全 mixed run 新采集 1,138 个 runtime GPU batch，覆盖 decode batch 1--16 和最长 12.6K
sequence。它确认旧 service model 的 batch/context 覆盖不足；同时 overlap completion interval
不能直接视为独立 GPU service time。再加上 native demand-load telemetry 缺失，本轮没有继续
用旧模型重跑 O0--O3。

## 5. 验证状态

```text
conda run -n beliefkv pytest -q
310 passed, 6 skipped

conda run -n beliefkv-agents pytest -q tests/test_multi_agent_runtime.py
14 passed

python scripts/check_sglang_contract.py third_party/sglang
compatible=true, SGLang=0.5.2rc1,
commit=18f91eb639084825717c0e3c3c7273492812ab71
```

两个正式 run 结束后均停止 SGLang；最终 compute-app 列表为空。

## 6. 尚未完成与恢复点

P3 仍未通过退出 gate。后续顺序调整为：

1. 观测并区分 BeliefKV command DMA 与 SGLang native demand-load/write-back；
2. 用完整 transfer trace 修复 HBM/Host/PCIe rolling accounting；
3. 将 runtime GPU sample 聚合为 overlap episode，建立覆盖 batch/context 的独立 holdout；
4. 重跑 observed-order replay，先要求 O0 与真实 JCT、transfer、peak HBM 对齐；
5. 对 blocking、cyclic、mixed 三类 trace 运行 B0--B4 与 fair O0--O3；
6. 只有 positive synergy gap 和 strongest-baseline gain 同时成立，才进入 P4/P5；否则将
   JointPlan 降级为工程统一接口。
