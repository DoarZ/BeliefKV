# Qwen3-Coder 固定模型 Runtime 配对 Gate

日期：2026-07-16

## 1. 问题

在模型固定为 `Qwen3-Coder-30B-A3B-Instruct-FP8` 时，比较 Qwen Code 0.19.10
与 Codex CLI 0.144.5，判断“复杂可并行任务不主动创建 subagent”主要来自 runtime
还是模型。两侧使用相同 user prompt、相同任务结构和相同本地 SGLang 0.5.2rc1
服务；提示要求在运行时自行决定 subagent 数量，没有预设数值。

这只是机制 gate，不是 BeliefKV 性能实验，也不是模型质量 benchmark。

## 2. 修正后的实验条件

- SGLang：TP=1，逻辑窗口 262,144，KV pool 167,816 tokens，CUDA Graph bs 16；
- 客户端窗口 163,840，实际请求保留完整 system prompt 和工具 schema；
- Qwen Code：Docker yolo，一次性 clone，完整工具权限，本地 API 精确 allowlist；
- Codex：`multi_agent` enabled、workspace-write，一次性 clone，外部凭证和 proxy 清除；
- 每个 condition 前 flush radix cache；每 100 ms 读取
  `sglang:num_used_tokens`，不以预分配显存代替 resident KV；
- permission rejection 非零、runtime 失败或内容 marker 缺失时不计 task success。

## 3. 有效结果

| Runtime | 模型请求 | 真实工具调用 | Spawn | 权限拒绝 | Marker | Resident KV 峰值 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen Code | 15 | 14 | 0 | 0 | 2/3 | 57,726（34.40%） |
| Codex | 9 | 8 | 0 | 0 | 2/3 | 23,859（14.22%） |

Qwen 证据位于 `runtime_matrix_gate/20260716T112000Z/`，Codex 证据位于
`runtime_matrix_gate/20260716T113000Z/`。统一机器可读摘要为
`runtime_matrix_gate/summary.json`。

两侧都串行读取仓库，没有产生 child session/thread。Codex 的 Responses bridge
完成 9 个请求且零 bridge/upstream error，runtime event 中有 8 对
`tool_start/tool_end`，所以 `spawn=0` 不能归因于工具协议失效。Qwen 的首轮 tool
schema 也包含 `agent`，且所有工具调用均成功，因此也不是权限拒绝造成的。

内容 gate 失败是实质性错误：两侧都把 Audit A 的中心符号写成
`CodexAppServerClient`，而实际通知到因果事件的适配入口是
`beliefkv/runtime/codex_adapter.py::CodexRuntimeAdapter`。因此不能只看 runtime
完成状态或生成了格式正确的答案。

## 4. 无效预检及修复

- `20260716T110000Z`：Qwen batch JSON 在 Docker attach 路径被截断为 65,536
  bytes。截断文件以 `.json.truncated` 保存，runner 已改为官方 `stream-json`，
  逐事件 JSONL 落盘；
- `20260716T111000Z`：Qwen `action_stagnation` 把连续读取不同文件误判为循环。
  正式 profile 关闭该检测，并保留 wall-time、turn、tool-call 三重硬预算；
- 早期 Codex 派生统计把所有 `item/started` 当工具调用，得到 17。正确口径是
  runtime event 中的 8 个 `tool_start`；runner 已修正，原始事件不变。

## 5. 结论与下一步

当前证据不支持“把 Qwen Code 替换成 Codex 就能让 30B-A3B 主动 spawn”。在相同
强委派提示下两者均为 0，问题更接近模型的规划/指令遵循能力，而非单一 runtime。
显式要求调用 `agent` 的 protocol smoke 仍能成功，因此机制可用与自然使用率必须
分开报告。

也不应继续运行完整 16-condition 矩阵：两个有效 gate 的 resident KV 都低于 50%，
既不能形成单卡 KV pressure，也没有 parent-waits-for-children 事件，无法评价事件响应
offload。下一阶段必须先满足至少一个前置条件：获得能自然 spawn 的本地模型，或从
真实可审计 runtime 收集动态 parent/child trace。仅增加并发串行 workflow 可以测试
通用多请求调度，但不能替代 subagent workload。
