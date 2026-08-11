# P6 R5 selective retraction ownership 修复

日期：2026-08-11

## 结论

冻结 v7 的第一笔 A arm 正常完成，但第二笔 B arm 因 allocator/Radix ownership 一致性错误退出。
因此 v7 不能形成配对性能结果；A arm 只保留为正确性 characterization，B arm 只保留为失败证据。
修复后确定性 GPU restore/retraction micro-gate 通过，新的 v8 输入已冻结，正式 A/B 必须从 v8
重新开始。

## v7 结果资格

- `pair-1-1-a`：8/8 workflow 完成，8/8 `system_jct_eligible`，649 次 LLM、846 次工具调用，
  运行 3552.57 秒；13/13 restore obligation satisfied，无未决事务，shutdown ACK 完整。
- `pair-1-2-b`：运行约 741.34 秒后 scheduler fail closed，0/8 workflow 完成；服务端报错为
  `multiple live Radix extents reference the same device index`。
- B arm 退出前没有选择 predictive PREPARE_HOST 或 predictive retraction；错误不能归因于一笔
  破坏性的预测动作。
- A arm runner 的 return code 120 来自父会话 stdout 被中断后的 CPython flush 失败。workload summary
  和 runtime shutdown 均已完整落盘，因此不改变该轮的正确性资格，但它不能与失败的 B arm配对。

原始目录：

- `experiments/ab/p6_predictive_joint_v7_formal/pair-1-1-a/`
- `experiments/ab/p6_predictive_joint_v7_formal/pair-1-2-b/`

## 根因与修复

SGLang 的 retraction 原路径使用 `len(req.prefix_indices)` 推导可直接释放的 suffix。BeliefKV 的
H2D restore、reentry 和 chunked prefill 会使该逻辑视图落后于当前物理 Radix ownership；错误释放
仍由 live Radix extent 引用的 device index 后，allocator 可以把同一 index 再分配给新 extent。

修复保持 allocator/Radix 一致性检查为 fail-closed，不在错误发生后去重或修改 Radix。实际
retraction 时只释放：

```text
request suffix candidate
  - live Radix device indices/pages
  - allocator free/release/free-group pages
  - invalid and duplicate candidates
```

释放前若已存在重复 live device index 或 Radix 拓扑环，立即失败。过滤只在实际 retraction 执行，
不进入每个 scheduler tick；审计记录 candidate、protected、already-free 和 released page 数。
同一保护同时覆盖 selective retraction 和 BeliefKV 环境下可能触发的 native retraction。

## 验证

CPU 与契约检查：

- retraction/chunked-prefill/adapter 定向回归：162 passed，6 subtests passed；
- 核心回归：667 passed，8 skipped，另外两项仅因运行在无 Deep Agents 的环境失败；
- Deep Agents/collection：99 passed；characterization：3 passed；
- `beliefkv check-sglang` compatible；patch 可从 `v0.5.2rc1` clean apply；
- `git diff --check` 通过。

确定性 GPU gate：

- 目录：`experiments/micro/p6_r5_retraction_ownership_v8/20260811T_gate/`；
- 14/14 correctness checks 通过；
- D2H/H2D 均为 2,659,221,504 bytes；
- restore 后观察到 340 个 GPU service sample；
- command/ACK、transaction、obligation 和 shutdown 守恒；
- 该笔 retraction 释放 683 个 request-private 页，0 个 live-Radix 页和 0 个已空闲页。

本 micro-gate 证明修复没有破坏完整 restore 闭环，但不能单独证明高并发下不再出现 ownership
错误。该结论必须由冻结 v8 的 R5 压力运行继续验证。

## v8 冻结边界

- baseline：`configs/p6/predictive_joint_v8/baseline_manifest.json`；
- A/B plan：`configs/p6/predictive_joint_v8/ab_run_plan.json`；
- source tree SHA-256：`cf6680569a042b01e24279ee1691814b00393c69d4aa513b6d39df92814cfcbc`；
- 顺序仍为 A-B / B-A / A-B，每个 run ID 只运行一次，不按结果筛选或重试。
