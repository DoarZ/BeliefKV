# BeliefKV P3A GPU-ready 并发探针

日期：2026-07-22

状态：单次、固定 240 秒 characterization 已完成；稳定 GPU-ready 并发 gate **未通过**。
当前仍处于 P3A，未进入 P4。

## 1. 问题与实验边界

本次只回答一个问题：在不反复调参和复跑的前提下，给初始 coder 规定 `2..4` 个 subagent
的运行时 fan-out 范围，四个并发 SWE-bench workflow 能否形成稳定的 GPU-ready 并发。

运行到 240 秒后使用一次 `SIGINT` 停止，因此 workload 目录保留 `.incomplete`。这是预先固定
的探针边界，不是 runner 崩溃。该 trace 不用于报告 JCT、SWE-bench correctness 或策略加速比。

原始数据位于
`experiments/archive/20260727/superseded_raw/p3_gpu_ready_probe/20260722T083314Z`，机器可读摘要位于
`experiments/processed/p3_gpu_ready_probe_20260722T083314Z/summary.json`。

## 2. 配置

- 模型：Qwen3-Coder-30B-A3B-Instruct-FP8；
- 服务：SGLang 0.5.2rc1，单卡 RTX 6000 Ada，TP=1；
- KV pool：163,840 token，`mem_fraction_static=0.952`；
- 调度上限：`max_running_requests=16`；
- workload：4 个真实 SymPy SWE-bench 任务，全部为 mixed workflow；
- 到达：每批 2 个 root，两批间隔 15 秒；
- spawn policy：initial coder 必须在运行时产生 `2..4` 个 task call，但具体 child 类型和任务
  仍由模型输出；本次每个 workflow 实际都产生 4 个 child。

## 3. 结果

`runtime_audit.resource_snapshot` 按相邻单调时间戳积分，窗口从第一个 `workflow_start` 到固定
240 秒截止点。GPU 利用率使用同一墙钟窗口内 GPU 0 的 1 秒采样。

| 指标 | 结果 |
|---|---:|
| SPAWN / root workflow | 16 / 4 |
| LLM submit / cutoff 前 result | 201 / 186 |
| repository tool start / end | 181 / 181 |
| running request 平均值 | 4.03 |
| running request 样本中位数 / 峰值 | 1 / 16 |
| `running >= 8` 时间占比 | 24.36% |
| `running >= 12` 时间占比 | 15.33% |
| `running <= 2` 时间占比 | 68.13% |
| GPU 利用率平均值 / 峰值 | 26.77% / 100% |
| GPU 利用率 `< 20%` 时间占比 | 64.17% |
| HBM KV 平均值 / 峰值 | 86.31% / 99.64% |
| HBM KV `>= 80%` 时间占比 | 81.35% |
| Host KV 平均值 / 峰值 | 2.77 / 5.42 GiB |
| SGLang batch log 中 queue 非空观察点 | 0 / 267 |

四个 root 都满足 required range，共产生 16 个 child；第一次到最后一次 SPAWN 相隔
29.27 秒。系统确实短时达到 16 个 running request，但没有维持该状态。

最关键的交叉证据是：`running <= 1` 占整个窗口 65.09%，这些区间内 HBM KV 平均仍为
95.62%，且 98.63% 的时间高于 80%。因此本次负载呈现的是 **HBM 接近满载但 GPU-ready
工作不足**，不是简单的“KV pool 没压满”。

显式迁移路径完成 32 次 D2H（3.87 GiB）和 91 次 H2D（1.65 GiB）；另有 2 次 D2H 和
1 次 H2D rejected。迁移活跃并未转化为持续 compute-ready batch。

## 4. 判定

本次 gate 判为失败，理由不是没有出现高峰，而是高峰只占少数时间：`running >= 8` 仅
24.36%，中位数为 1，GPU 利用率低于 20% 的时间达到 64.17%。267 个 SGLang batch log
观察点的 queue 都为零，也说明大量 child 同步进入工具阶段后，服务端缺少可运行请求。

这支持两个同时成立、不能互相替代的判断：

1. 当前 reactive-only residency 很可能保留了大量等待态 context，使低 runnable 区间仍保持
   极高 HBM 占用；但在完成 physical owner attribution 前，不能把全部占用都称为 parent KV。
2. 单纯增加 root 数量或加入 predictor 不能凭空创造 ready work。当前静态 fan-out 形成同步
   burst，随后 child 一起进入工具阶段；继续加并发可能先增加迁移和抖动，而不是稳定吞吐。

## 5. 下一项 P3A Gate

下一步不应立刻扩大到 8/12 个 root，也不应重复本探针。应先实现可复现的 **phase-aware
release**：保留每个 workflow 的 `2..4` child 范围，但用全局 ready target 控制 child 释放，
避免同一 activation 一次性提交全部 child；同时在真实 `JOIN_WAIT` 事件后立即把 parent 变成
可迁移候选。之后用同一四任务 manifest 做一组配对验证：

- fan-out 数量、任务、模型、采样参数和 240 秒窗口保持不变；
- 对照为本次 simultaneous release trace；
- 只检查 `running >= target` 时间占比、GPU 利用率、低 runnable 时 HBM 和迁移量；
- gate 通过前不进入 P4，不训练预测器，也不扩展实验矩阵。

本次只有一个 child 在截止前 RETURN，4 个 JOIN 均未闭合，并出现一次真实 repeated-read guard。
这些是固定时长截断与 agent 行为事实，因此该 trace 只承担负载诊断作用。
