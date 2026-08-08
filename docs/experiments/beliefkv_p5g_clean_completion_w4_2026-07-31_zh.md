# BeliefKV P5G fixed w4 full GPU gate

日期：2026-07-31

状态：**失败。数据面 command/ACK 与 allocator/page-index 一致性通过，但 0/4 workflow 进入 clean JCT，P5G restore transaction 没有被覆盖，受控 shutdown 也未完成 ACK。不得用于性能比较，也不能据此冻结 P5 接口。**

## 1. 实验范围

本轮只执行一次固定 w4 trace，没有参数扫掠、在线修改或失败后重跑：

- `Qwen3-Coder-30B-A3B-Instruct-FP8`，SGLang `0.5.2rc1`，TP=1，仅 GPU 0；
- KV pool 163,840 token，context 262,144 token，`MEM_FRACTION_STATIC=0.952`；
- Host HiCache 96 GB，predictor 关闭；
- 固定 4 个 SWE-bench Verified SymPy workflow，3 个 mixed、1 个 cyclic，同时启动；
- mixed workflow 首次 fan-out 为 2--4，后续 spawn/peer reactivation 由模型和 runtime 决定；
- 动态上下文 32K，压缩后保留约 8K；单请求 GPU-service inactivity 为 900 秒；
- workflow 绝对 wall-clock 边界为 7,200 秒；
- observed JointPlan、admission tickets、running retraction、Host lifecycle cleanup 和 native HiCache telemetry 开启。

原始数据：

```text
experiments/raw/p5g_clean_completion_w4/20260731T124929Z/
```

本轮 manifest 无效，因此没有生成 KV 迁移时间线。

## 2. Gate 结论

| Gate | 结果 | 证据 |
| --- | --- | --- |
| 固定 trace 且只运行一次 | PASS | 单次 w4，没有重跑 |
| workload 强度 | PASS | 943 次业务 LLM、973 次工具调用、11 个动态 child、10 次 context compaction；4/4 workflow 通过强度门槛 |
| clean completion | **FAIL** | 1/4 `completed=true`，0/4 `clean_jct_eligible` |
| API/admission liveness | PASS（本轮样本） | 953 个 request physical start、953 个 finish；运行中无 API timeout、queue timeout、OOM 或 admission stall |
| 显式 command/ACK | PASS | 246 dispatch、246 ACK、0 missing/orphan/ordering violation；全部是 terminal-private Host cleanup |
| allocator/page-index 一致性 | PASS | HBM mirror 未超过 allocator；Host inflight/quiescent mismatch 均为 0 |
| retry storm | PASS | 41 次 descendant-closure block、40 次 event-gated suppress；0 identical failed/zero-byte retry |
| P5G restore obligation | **未覆盖** | 0 restore obligation、0 running retraction、0 explicit D2H/H2D command |
| lazy physical snapshot | **未覆盖** | `call_count=0`；没有 restore transaction 触发物理 ownership rebuild |
| shutdown | **FAIL** | prepare snapshot 为全空，但最终 summary 停在 `preparing/final=false`；audit writer 仍有 1 条 pending event |

## 3. Workflow 结果

| instance | mode | 时长 | 结果 | 主要原因 |
| --- | --- | ---: | --- | --- |
| `sympy__sympy-13878` | cyclic | 6,379.16 s | runtime error | 218 次 LLM、213 次工具后达到 LangGraph recursion limit 512 |
| `sympy__sympy-12489` | mixed | 7,205.29 s | workflow timeout | 393 次 LLM、430 次工具、4 个初始 child 和后续 peer activation 均持续推进，但根 workflow 未在 7,200 秒内收敛 |
| `sympy__sympy-13852` | mixed | 1,867.69 s | runtime error | 模型产生 self-handoff，runtime 直接抛出 `handoff must target a different peer role` |
| `sympy__sympy-14248` | mixed | 4,615.52 s | semantic complete | 4/4 child RETURN、JOIN 满足并自然完成；一次 `repeated_failed_tool_call` 使 `guard_valid=false`，因此不进入 clean JCT |

所有 mixed workflow 的 child RETURN/JOIN 都完整。失败不来自 parent restore、request starvation 或 GPU execution timeout，而集中在 agent runtime 的根级终止、peer transition 和异常决策恢复。mixed-003 还发生过一次成功的结构化协议修复，因此 `native_protocol_valid=false`；该字段是“模型是否原生输出协议”的严格诊断项，并不参与当前 `clean_jct_eligible` 判定。其直接 gate 原因只有 `guard_valid=false`。

工具错误总计 122 次，主要是模型命令返回非零和 `edit_file` 字符串不匹配；日志中没有 sandbox 权限拒绝。已有 duplicate-call guard 抑制了 34 次重复调用，但它只能阻止相同失败动作，不能让外层 peer graph 在局部 agent 已完成后可靠结束。

## 4. KV 与物理数据面

离线 transfer validator 通过 command integrity 和 residency consistency：

- HBM 峰值 `16,106,127,360 bytes`，达到配置 KV pool；
- Host 峰值 `95,977,046,016 bytes`，接近 96 GB 容量；
- native HiCache audit：2,923 次 D2H、20 次 H2D，全部 completed；
- 独立 transfer telemetry 文件完整写入 2,922 次 D2H，约 116.11 GiB，以及 20 次 H2D，约 8.78 GiB；
- audit 与独立 telemetry 相差 1 条 D2H，和 shutdown 时 `audit_writer.pending_count=1` 一致；
- 246 个 BeliefKV command 全部为 `drop_terminal_private`，不是显式迁移或 restore；
- Host/page-index mismatch、HBM capacity mismatch 和 command ordering violation 均为 0。

因此，本轮证明了长时间运行下 native HiCache callback、terminal Host cleanup 和 allocator mirror 基本守恒，但没有证明 P5G transactional restore。native demand-load 不会自动创建 BeliefKV restore obligation，不能用 20 次 native H2D 替代 P5G gate。

## 5. 控制面与 shutdown

两小时运行中记录：

- 52,607 次 JointPlan publish；
- 58,110 次 physical commit budget exceeded；
- 12,666 次 stale plan；
- 65,392 次 GPU service sample；
- `runtime_audit.jsonl` 约 430 MB，`policy_snapshots.jsonl.gz` 约 95 MB。

这些数值不等于在线动作失败，但说明控制面开销与观测量仍然偏大。更重要的是，当前前台 `Ctrl-C` 会同时打断 scheduler/detokenizer 进程组：scheduler 写出了空的 shutdown prepare snapshot，却未发布 `SHUTDOWN_ACK`，detokenizer 以 `KeyboardInterrupt` 退出，最终 summary 未 flush。后续必须由 frontend 显式发送 shutdown prepare，等待 ACK 和 writer drain 后，再终止进程组。

另有 1 次 `request_physical_checkpoint_failed`：一个 mixed-000 request 在 finish 时缺少对应 physical-start checkpoint。请求本身正常完成，但该观测缺口需要在下一次 correctness smoke 前修复。

## 6. 结论与下一步

本轮不能关闭 P5G gate。按优先级应当：

1. 将 self-handoff 规范化为留在当前 peer 或一次有界重决策，不能让模型的可恢复决策直接杀死 workflow；
2. 在根 workflow 层聚合 local semantic completion、测试状态和 peer reactivation，局部 agent 已完成时必须能够关闭外层 graph；recursion limit 只保留为最后防线；
3. 修复显式两阶段 shutdown，使 `SHUTDOWN_PREPARE -> writer/transaction drain -> SHUTDOWN_ACK -> process termination` 真正落地；
4. 修复 physical-start/finish checkpoint 的单条缺口；
5. 用确定性 GPU restore micro-gate 覆盖 obligation、snapshot、read-set 和 restored service quantum，再决定是否重跑完整 w4；不能依赖自然 w4 恰好触发 retraction。

在这些条件满足前，P6 可以继续离线标签和预测模型开发，但不能开放预测性物理动作。
