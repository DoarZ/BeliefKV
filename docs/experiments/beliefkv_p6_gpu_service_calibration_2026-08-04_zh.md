# P6 独立 GPU Service Calibration（2026-08-04）

## 目的与边界

本轮只采集与 agent workflow 语义模型解耦的 GPU service 数据，不执行固定任务的
w1/w4/w8 invariance audit。目标是拟合：

```text
phase + token demand + batch composition + sequence length
+ chunk position + cache hit + HiCache contention
-> scheduler/worker service-time distribution
```

时间边界为 SGLang scheduler launch/previous completion 到 result processing 完成，
不是 CUDA event 意义上的纯 kernel 时间。prefill interval 还包含首 token sampling。

## 实验配置

- GPU：NVIDIA RTX 6000 Ada 48 GiB，使用 device 1，TP=1。
- 模型：Qwen3-Coder-30B-A3B-Instruct-FP8。
- SGLang：0.5.2rc1，源码提交 `18f91eb`。
- GPU KV pool：163,840 token，约 15 GiB。
- Host HiCache：96 GiB，write-back。
- 最大 running request：16。
- 覆盖 batch size：1/2/4/8/16。
- sequence length：约 1K/2K/4K/8K/16K/24K/30K。
- prefill cache-hit target：train 0/0.5/0.9，holdout 0.25/0.75。
- 单 batch token budget：143,360；超过物理容量的 7 个 profile 被显式跳过。
- BeliefKV predictor、JointPlan、observed admission、running retraction 和 reactive
  transfer 均关闭，仅保留 runtime identity 与 queue-service observer。

## 运行结果

- 客户端请求：213/213 成功；train 118，holdout 95。
- batch-unique service interval：2,374；train 1,326，holdout 1,048。
- phase：prefill 69，decode 2,305。
- batch size 分布：b1 519、b2 575、b4 576、b8 448、b16 256。
- 最大观测 sequence length：30,781 token。
- 服务端无 OOM、scheduler exception 或 allocator/Radix consistency error。
- BeliefKV reactive transfer：0。
- 原生 HiCache transfer：765；其中 10 个 service interval 与可观测传输窗口相交。
- observer CPU：P50 0.056 ms，P95 0.106 ms，P99 0.137 ms，max 0.252 ms。
- shutdown 前无未决 transaction；correctness summary 完整。

PCIe overlap 使用 transfer `start_ts_ms -> complete_ts_ms`；原生 HiCache 缺少 start
时间时使用 `submit_ts_ms -> complete_ts_ms` 作为保守上界，不声称是精确 DMA overlap。

## 模型结果

模型使用 1,304 个非 warmup train batch，holdout 使用 1,036 个非 warmup batch。

| 指标 | 结果 |
|---|---:|
| exact support | 372 |
| phase/batch backoff | 664 |
| P50 预测相对误差中位数 | 20.49% |
| P50 预测相对误差 P95 | 130.24% |
| 名义 P90 区间覆盖率 | 86.29% |
| 名义 P95 区间覆盖率 | 90.25% |
| decode 相对误差 P95 | 126.08% |
| prefill 相对误差 P95 | 620.33%（仅 12 个 holdout interval） |

结论：采集链路与负载覆盖通过，但当前经验分桶模型未通过校准门槛，不能接入在线
P6 predictive action。主要问题是单次采集下 prefill 条件组支持不足，稀疏条件退化到
phase/batch backoff；当前 artifact 只能用于调试模型结构。

## 失败轮次说明

`20260803T165922Z` 是失败轮次。配置虽然关闭 policy shadow，但没有关闭旧 reactive
transfer planner，产生 21 笔显式迁移，并与原生 HiCache write-back 并发，最终在高压
chunked prefill 中触发重复 device-index consistency assertion。该轮不得进入训练集。

修复后新增 `reactive_transfer_enabled`，硬件校准通过
`--disable-reactive-transfer` 显式隔离 BeliefKV 迁移策略；默认值仍为 true，不改变普通
BeliefKV 实验行为。校准客户端也改为逐请求写 checkpoint，异常时不再丢失已完成样本。

## 数据位置

- 有效 raw：`experiments/raw/p6_gpu_service_v2/20260803T171255Z/`
- 失败 raw：`experiments/raw/p6_gpu_service_v2/20260803T165922Z/`
- 处理后数据：`experiments/processed/p6_gpu_service_v2/20260803T171255Z/`
- 模型 artifact：
  `experiments/models/gpu_service_curve_qwen3coder30b_rtx6000ada_20260803T171255Z.json`
- 修复前错误 contention 标签的数据保留为 `.pre-contention-fix`，不得用于正式结果。

下一步应优先改为支持 sequence/token 邻域插值和不确定性校准的 service model，并用
已有 holdout 固定评估。不要通过降低 minimum support 或把 holdout 回灌训练来掩盖误差。
