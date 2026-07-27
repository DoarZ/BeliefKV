# BeliefKV P3 真实工具型 Agentic Workload

日期：2026-07-22

状态：运行时、强度 gate、batched-arrival runner 和 required-range fan-out 已完成；单次固定
时长探针未形成稳定 GPU-ready 并发，当前仍处于 P3A workload characterization。

## 1. 为什么替换旧 workload

`experiments/archive/20260722/p3_correctness_only/raw/p3_dynamic/20260721T155058Z`
的 12-workflow mixed run 只适合作为
拓扑、HBM 压力和迁移机制 characterization：每个 workflow 只有 2--6 次 peer turn 和 4 个
one-shot leaf，合计 6--10 个 LLM 请求；48 个 leaf 均不调用 repository tool，整个 run 的
tool call 为 0。它不能代表 Codex、Claude Code 或 Deep Agents 中“一个 task child 在独立
context 内连续执行 model-tool-observation loop”的负载。

新版 workload 的目标不是人为增加 token，而是恢复真实 agent runtime 的四种行为：

1. peer 是持久线程，handoff 后保留完整历史，再次获得控制时使用同一个 context RESUME；
2. child 是 FRESH context，task 生命周期结束后退出，但内部允许多轮 LLM 和多种工具；
3. autonomous mode 下 subagent 数量由模型在运行时选择；required-range characterization
   mode 只规定初始 fan-out 范围，具体 child 类型与任务仍由模型生成；
4. 工具在每 workflow 独立的离线 Docker/Git workspace 中真实执行。

## 2. 运行时结构

```text
SWE-bench issue
      |
      v
Persistent Coder context --HANDOFF--> Persistent Reviewer context
      ^                                      |
      |                                      v
      +----------- Persistent Tester context+
                 (cyclic REACTIVATE)

Any non-final peer activation in mixed mode
      |
      +-- model chooses zero or more task calls
              |
              +-- SPAWN FRESH child A: LLM -> file/shell/test -> ... -> completion
              +-- SPAWN FRESH child B: LLM -> file/shell/test -> ... -> completion
              |
              +-- JOIN_ALL -> child reports -> parent LLM continuation
```

`cyclic` mode关闭 task middleware，只验证持久 peer handoff/re-engagement；`mixed` mode 开放
`repository-explorer`、`test-analyst`、`dependency-tracer` 和 `invariant-auditor` 四类 child。
autonomous mode 允许模型选择 0 个 child。required-range mode 用于受控负载探针：初始 coder
必须一次产生范围内的 task call，达到有效范围后关闭后续 delegation；它不预先指定精确 child
数、类型或任务内容，不能替代 autonomous mode 的真实性实验。

最终允许的 peer activation 会从模型工具集合中删除 `task`。这保证 `max_turns` 是可执行的
终止边界，而不是在最后一轮再次创建长 child。未知 `subagent_type` 只记录为 rejected action，
不再错误创建 physical invocation/join。

## 3. Context 与 KV 语义

| 对象 | 创建 | 后续调用 | 生命周期 |
|---|---|---|---|
| Coder/Reviewer/Tester | `FRESH` | 相同 context，epoch 单调增长，`REACTIVATE` | workflow 结束 |
| task child | `FRESH`，不继承 parent KV | child 内相同 context 多轮增长 | `RETURN` 后结束 |
| parent 在 task 期间 | `JOIN_WAIT` | child 返回后继续原 context | 可形成 parked-parent KV 机会 |

一次 peer activation 可能包含几十次内部 LLM 调用。backend 会返回该 activation 的最终
`context_epoch`，外层 handoff/final action 必须使用该 epoch；否则 RCCG 会把旧 epoch action
判为 stale。该问题在第一条真实 smoke 中被发现并修复。

## 4. 工具与隔离

每个 workflow 从冻结 `base_commit` 创建独立 shared Git clone，并启动一个 Docker 容器：

- `--network none`、read-only root、drop all capabilities、`no-new-privileges`；
- 限制 CPU、内存和 PID；只有 `/workspace` 可写；
- filesystem tool 和 shell 共用 `/workspace` 路径；
- 使用镜像内预构建 test environment，禁止 pip/network install；
- coder 可使用 filesystem edit/apply_patch；reviewer/tester/analysis child 的 filesystem
  write/edit 工具被移除，但 execute 仍挂载可写 workspace，因此“不编辑”目前是角色策略，
  不是可作为安全证明的强制边界；
- command、input 和 output 只记录长度与 SHA-256，不把代码/提示正文写入控制日志。

真实外部工具包括 `ls/read_file/glob/grep/edit_file/apply_patch/execute`。结构化
`AgenticPeerDecision/ChildCompletion` 不计作外部工具，`task` 计作因果 SPAWN/JOIN，而不计作
repository tool。

## 5. 终止与防失控

- 每次 activation 必须返回结构化 `AgenticPeerDecision`；普通文本不算 terminal；
- 默认单 activation 上限为 48 次 model、128 次普通 tool；这些只是防失控的宽松上限，
  prompt 不限制完成任务所需的有效工具次数；
- 连续相同调用、ABAB、连续错误或连续无新 observation 会进入 finalization；
- 已删除“连续若干 `python -c` 即为循环”的工具类型启发式。真实运行证明不同命令和不同
  输出仍会被它误杀，工具名称本身不能证明 agent 没有进展；
- 持久 peer 在上一次 structured completion 后 RESUME 时，guard 计数从新 activation 重置；
- 成功 `edit_file/write_file/apply_patch` 都被识别为已有 patch，recovery 转入 test，而不是再次
  强制 apply patch；
- 模型自报 `tests` 字段不参与在线终止硬拒绝。正确性由真实 workspace、sandbox test 和后续
  SWE-bench harness 事后验证，避免自报格式错误改变负载轨迹。

## 6. 自动真实性指标

runner 为每个 workflow 输出两个独立 gate：

```text
load_valid = natural semantic_complete
             and no stuck-guard trigger
             and LLM requests >= configured minimum
             and real repository tools >= configured minimum

subagent_trace_valid = spawn > 0
                       and every child has LLM requests > 1
                       and every child returned
                       and every JOIN was satisfied
```

`agent_runtime_trace` 还包含 per-invocation LLM/tool 数、tool name 分布、tool duration
p50/p95/max、rejected task 数和每个 child 的最终 epoch。manifest 汇总 tool-rich、dynamic
subagent、multi-turn child 和 peer reactivation 的 workflow coverage。`agent_control` 单独记录
stuck 原因和 natural/forced completion；任一 forced completion 会使整次实验无效。

## 7. GPU Smoke 结果

模型与系统：Qwen3-Coder-30B-A3B-Instruct-FP8、SGLang 0.5.2rc1、单卡 RTX 6000 Ada、
`max_total_tokens=163840`、`mem_fraction_static=0.952`。本轮忽略 kernel 优化，只验证 workload。

| Run | 结果 | LLM | real tool | valid multi-turn child | peer turn |
|---|---|---:|---:|---:|---:|
| `workload-smoke-v2` | 旧在线 gate 误拒绝 | 253 | 239 | 3 | 7 |
| `workload-smoke-v3` | semantic complete | 28 | 26 | 0 | 2 |
| `workload-smoke-v4` | semantic complete | 41 | 38 | 0 | 3 |

v2 的三个真实 child 分别执行 11/15/17 次 LLM 和 19/14/16 次工具；所有 valid join 均闭合，
并覆盖 7 次 handoff 和 5 次 peer reactivation。另有一个不存在的 `general-purpose` 请求在旧
adapter 中被误记为 child，但实际 LLM/tool 均为 0；最新版已在创建 invocation 前过滤。
v2 最终失败来自已删除的“terminal 必须填写 tests/unresolved 数组”规则，不是 child、join、
event channel 或 SGLang 失败，因此不能作为性能结果，但可以用于定位上述机制问题。

v3/v4 证明修复后的 terminal 路径可稳定完成、产生真实 patch 和工具负载。二者的模型均选择
0 个 child，这一事实必须如实保留：动态模式表示模型可以选择 fan-out，不保证每条 task 都
会 spawn。

## 7.1 Required-range GPU-ready 探针

随后只运行一次 240 秒、4-workflow required-range 探针，范围设为 `2..4`。四个初始 coder
都产生 4 个 child，共 16 次 SPAWN、201 次 LLM submit 和 181 次真实工具调用。系统峰值达到
16 个 running request，但按时间积分后平均值为 4.03，`running <= 2` 占 68.13%，
`running >= 8` 仅占 24.36%；GPU 利用率平均 26.77%。与此同时，`running <= 1` 区间的 HBM
KV 平均仍为 95.62%。

因此该模式解决了“模型不 spawn 导致负载退化”的机械问题，但 simultaneous fan-out 只制造
短时 birth burst，没有形成稳定 ready supply。完整报告见
[P3A GPU-ready 并发探针](beliefkv_p3_gpu_ready_probe_2026-07-22_zh.md)。

## 8. 严格边界

这些 smoke 仍不能支撑 BeliefKV 性能结论：

1. 只运行了一个 workflow，不能评价并发调度、KV offload 收益或 workflow fairness；
2. v2 中大量 `execute` 尚未经过任务正确性与重复进展审计，不能只因工具类型将其认定为有效
   或无效 agent work；
3. 当前 child 是 blocking、depth-1 task；尚未覆盖 resumable child、nested spawn 和
   nonblocking background child；
4. 同一 workflow 内 Docker `execute` 有互斥锁，并行 child 的 shell 阶段会串行；
5. semantic completion 不等于 SWE-bench resolved，正式正确性必须另跑 harness；
6. 动态 trajectory 是 timing/semantic-race-sensitive，不能用单次 run 做严格策略 A/B。

因此旧 one-shot workload 降级为 topology smoke；新版 agentic workload 才能作为后续并发
characterization 输入，但仍需先通过 suite coverage 和重复性 gate。

## 9. 正式实验建议

正式 characterization 使用同一冻结 manifest，先运行 4 个、再运行 8 个 workflow。root 不再
全部同时启动：默认每 20 秒启动一批 2 个，使不同 wave 的 root prefill、动态 spawn、join 和
peer resume 在时间线上重叠。`mixed,mixed,cyclic` 循环使大多数 workflow 开放 subagent，同时
保留纯 peer-agent 对照。

`max_turns=18`、每 activation 48 次 model call、128 次 tool call 和 `recursion_limit=512` 都是
失控保护上限，不是让模型提前结束的目标配额。productive workload 反向设置最低强度 gate：
每 workflow 至少 16 次 LLM/16 次工具；mixed workflow 的每个 child 至少 4 次 LLM/3 次工具。
低于门槛的 run 可以验证正确性，但不能进入性能分析。autonomous mode 不规定 child 数量；
受控 characterization 可以规定 `2..4` 的范围，但必须单独标记，且不能把它当作自治 fan-out
证据。任务仍应包含多模块调用链、复现和回归测试需求，并要求整组实验满足：

- tool-rich workflow 比例 100%；
- 至少 30% workflow 产生 valid multi-turn child；
- 至少 20% workflow 产生 peer reactivation；
- 无 stale epoch、event rejection、recursion limit 或未闭合 join；
- forced-completion 必须为 0；任何 guard 触发都保留为失败证据，但不能作为主实验；
- baseline 与 BeliefKV 使用相同任务、模型、并发、采样参数、sandbox 和容量；
- 同时运行 harness correctness，但不要让 harness 结果反向改写 runtime trace。

运行入口：

```bash
conda run --no-capture-output -n beliefkv-agents \
  python scripts/run_langgraph_peer_workloads.py \
  --output-dir experiments/raw/p3_agentic_waves/<run>/workloads \
  --backend agentic \
  --spawn-policy required-range \
  --min-initial-subagents 2 \
  --max-initial-subagents 4 \
  --modes mixed,mixed,cyclic \
  --max-workflows 8 \
  --concurrency 8 \
  --arrival-mode batched \
  --arrival-batch-size 2 \
  --arrival-batch-interval-seconds 20 \
  --max-turns 18 \
  --max-completion-tokens 4096 \
  --recursion-limit 512 \
  --stuck-max-model-calls 48 \
  --stuck-max-tool-calls 128 \
  --min-workflow-llm-requests 16 \
  --min-workflow-tool-calls 16 \
  --min-subagent-llm-requests 4 \
  --min-subagent-tool-calls 3 \
  --control-socket /tmp/<beliefkv-runtime>.sock \
  --min-kv-pool-tokens 163840
```

本轮结束后 SGLang 服务与 Docker workload 容器均已停止，GPU 无残留进程。
