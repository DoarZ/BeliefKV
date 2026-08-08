# P5G 确定性 Restore Micro-gate（2026-08-01）

## 结论

本轮未通过 restore micro-gate，因此按预先约定没有启动 autonomous w4 system gate。
失败来自 workload 前置条件，而不是已执行 restore 事务的数据面错误。

## 配置与结果

- GPU：RTX 6000 Ada，物理 GPU 1，TP=1。
- 模型：Qwen3-Coder-30B-A3B-Instruct-FP8。
- KV pool：163,840 tokens（约 15 GB）。
- Host HiCache：96 GB。
- `MAX_RUNNING_REQUESTS=16`。
- victim：48,042 prompt tokens + 1,024 completion tokens。
- anchor：48,041 prompt tokens + 1,024 completion tokens。
- replacement：64,042 prompt tokens + 256 completion tokens。
- 三请求总运行上下文峰值约 160.6K tokens，SGLang 日志观测 token usage 约 0.98。

三个请求均被 admission，replacement 没有稳定停留在 waiting queue。running retraction 只在
安全点对 waiting replacement 构造联合计划，因此没有产生 retraction、D2H、restore obligation
或 H2D；ownership snapshot 也未被物理路径请求。

Verifier 结果：

- `passed=false`；
- D2H/H2D 均为 0；
- restore obligation 为 0；
- physical snapshot call count 为 0；
- shutdown acknowledged；
- 无 pending transaction；
- shutdown cleanup 未掩盖未完成事务；
- GPU 与服务进程已完全释放。

原始数据位于：

`experiments/raw/p5g_restore_micro/20260731T174605Z/`

## 修复

确定性 micro-gate 将 `MAX_RUNNING_REQUESTS=2` 设为硬前置条件：victim 与 anchor 先占满两个
running slot，随后提交的 replacement 必须保持 waiting。runner 在发出任何 workload 请求之前
查询 `/get_server_info`，配置不匹配时直接失败且不计为实验。该修复只改变 micro-gate harness，
不改变 BeliefKV 在线策略。
