# Codex + Qwen2.5-14B Subagent 事件响应实验 r16

日期：2026-07-16（Asia/Shanghai）

## 实验目的

验证无 DAG、无预测条件下，BeliefKV 能否从 Codex runtime 的动态事件中识别
parent/subagent 关系，并在单 GPU 高 KV 压力下完成事件响应的 admission、D2H
offload 和 unowned KV 回收。本实验首先验证系统闭环；单次动态 agent 轨迹不用于
给出性能因果结论。

## 固定条件

- GPU：RTX 6000 Ada 48 GB，仅使用 GPU 0，TP=1。
- 模型：Qwen2.5-14B-Instruct，BF16。
- Runtime：Codex CLI 0.144.2，本地 Responses 协议桥。
- SGLang：0.5.2rc1，commit `18f91eb639084825717c0e3c3c7273492812ab71`。
- KV pool：32,768 tokens，K/V 各 3 GB；HiCache host pool 约 12.88 GB。
- 并发：4 个 SWE-bench root workflow；每个 root 在运行时 spawn 2 个 subagent。
- workload SHA-256：`8945288a77a2fb806337a5fb87039a333397be35202a896456b7d6c11f10b8fd`。
- BeliefKV：predictor、CPU shadow 和主动 prefetch 均关闭；只启用事件响应策略。

实例为：`sympy__sympy-13877`、`sympy__sympy-17630`、
`sympy__sympy-13878`、`sympy__sympy-17318`。

## 完整性结果

- subagent gate 通过：4/4 root、8/8 subagent；每个 workflow 均包含
  `spawn`、`join_wait` 和 `join_satisfied`。
- 12 个 invocation 全部 return，4 个 workflow 全部 end。
- 36 个请求全部经历 deferred、admitted、started 和 finished。
- 83 个 transfer dispatch 全部收到 ACK；47 completed、36 partial。
- `runtime_initialized` 与 `runtime_shutdown` 各 1 次。
- 无 scheduler exception、无 rejected runtime event、无 transfer watchdog 超时。
- 总耗时 105.686 s；SGLang 峰值 KV token usage 为 0.85。

## 机制证据

- 61 次 `offload_context`，实际 D2H 6,747,389,952 bytes。
- 22 次 `drop_unowned`，实际回收 1,621,819,392 bytes。
- 总 ACK bytes 为 8,369,209,344 bytes。
- transfer callback：均值 43.73 ms，p50 36.99 ms，p95 53.06 ms，最大 448.29 ms。
- admission queue wait：均值 888.31 ms，p50 47.78 ms，p95 4,283.74 ms，
  最大 10,737.69 ms。
- 出现 98 次 `no_migratable_marginal_pages` 本地拒绝，说明 victim 过滤和
  不可迁移页的负缓存仍需优化。

## 与 baseline r5 的观察性对比

baseline r5 makespan 为 83.704 s，r16 为 105.686 s；表面 speedup 为 0.792，
即 r16 慢 26.3%。该结果不能解释为策略的纯性能影响，因为两次动态轨迹不等价：

- request 数：40 -> 44（+10.0%）；
- upstream request 数：32 -> 36（+12.5%）；
- prompt tokens：316,070 -> 341,627（+8.09%）；
- completion tokens：6,972 -> 8,502（+21.94%）；
- tool start/end：14 -> 18。

按 completion token/makespan 粗略归一化后，r16 吞吐为 baseline 的 0.966，
仍低约 3.4%。SGLang 日志中的平均 decode throughput 从 89.56 降至
84.57 token/s。另一方面，r16 的实际 prefill compute tokens 为 49,233，
低于 baseline 的 51,571，说明本次负收益不能简单归因为 prefix 重算。

## 结论

r16 证明了 Codex 动态 subagent 事件到 SGLang 物理 KV 操作的闭环已经跑通，
并修复了此前主动 H2D 卡死导致的实验不可完成问题。但当前结果不支持性能收益：
事件响应策略存在 admission 长尾、频繁 D2H 和无效 victim 重试。下一轮必须使用
冻结的请求/事件 trace 或至少多次配对重复，并优先实现不可迁移 victim 负缓存与
基于迁移收益的 offload 抑制，再评价 workflow latency 和 throughput。

## 原始产物

- 运行目录：`experiments/raw/codex-qwen14b-subagent-reactive-c4-r16/`
- SGLang 日志：`experiments/raw/codex-qwen14b-subagent-reactive-c4-r16.server.log`
- 调度审计：`experiments/raw/codex-qwen14b-subagent-reactive-c4-r16.sglang-audit.jsonl`
- 实验配置：`experiments/raw/codex-qwen14b-subagent-reactive-c4-r16.config.json`
- A/B 汇总：`experiments/processed/codex-qwen14b-subagent-baseline-r5-vs-reactive-r16.json`
