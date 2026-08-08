# P5G 确定性 Restore Micro-gate 复验（2026-08-01）

## 结论

本轮只提交了一次确定性 micro-gate workload。micro-gate 未通过，因此没有启动 autonomous
w4 system gate，也没有生成 KV 迁移时间线。

这次已经排除上一轮的 harness 并发配置问题：服务端实际报告
`MAX_RUNNING_REQUESTS=2`，日志在 holder decode 期间稳定出现 `2 running / 1 waiting`。
失败来自 overlap drain barrier 与测试 hook 的触发顺序，而不是 replacement 未进入 waiting。

## 配置与观测

- 原始目录：`experiments/raw/p5g_restore_micro/20260801T080048Z/`。
- GPU：RTX 6000 Ada，物理 GPU 1，TP=1。
- 模型：Qwen3-Coder-30B-A3B-Instruct-FP8。
- KV pool：163,840 tokens（约 15 GB）；Host HiCache：96 GB。
- victim 和 anchor 分别执行 48K prompt + 1,024 token decode。
- replacement 为 64K prompt + 256 token decode，在两个 holder 运行时进入 waiting。
- 三个请求最终均正常返回，workload 总时长 54.63 秒。
- shutdown acknowledged，0 pending transaction/lease/funding/obligation，GPU 已释放。

Verifier 结果为 `passed=false`：没有 retraction、physical ownership snapshot、D2H、restore
obligation、H2D 或 post-restore service quantum。

## 根因

SGLang overlap 调度期间没有天然 safe point。BeliefKV 必须先请求 drain barrier，才能进入
`plan_running_batch_retraction()` 并执行 micro hook。原实现的 barrier 仍使用普通在线策略：

```text
2 running + 1 waiting
        |
        v
replacement_deficit = 0, active_excess = 0
        |
        v
barrier_no_pressure，拒绝 drain
        |
        v
micro hook 所在的 safe-point planner 永远不可达
```

最终 summary 记录 `barrier_no_pressure=1025`、`active_floor=1401`，micro-gate 状态一直为
`armed`。这解释了为何 waiting 条件成立，却没有产生任何物理事务。

## 实验后修复

仅当测试 hook 显式启用、指定 victim 已获得真实 GPU service、且指定 replacement 正在
waiting 时，overlap barrier 可以忽略“自然 pressure 不足”并执行一次 drain。drain 后仍必须通过
生产路径的 physical snapshot、read-set、allocator、closure、D2H/obligation/H2D 和 service-grace
检查；普通 workload 行为不变。

针对性 CPU 测试共 23 项通过。该修复尚未进行新的 GPU workload 验证。
