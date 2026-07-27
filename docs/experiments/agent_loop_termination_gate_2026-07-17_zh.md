# Agent loop 语义终止与卡死检测 gate

日期：2026-07-17

## 1. 目标

验证将 LangGraph `recursion_limit` 从 30 提升到 200 后，Qwen3-Coder 驱动的
`create_agent` 是否仍会形成长时间不收敛的 model-tool loop；同时验证显式语义完成
协议和卡死检测能否在不依赖低 recursion limit 的情况下结束 child 和 parent。

本轮只使用一个 SWE-bench Verified workflow（`sympy__sympy-13852`），不用于 KV
调度性能结论。

## 2. 实现

### 2.1 语义完成协议

- child 必须返回 `ChildCompletion`，包含 status、summary、evidence、tests、
  files_changed、unresolved 和 confidence。
- parent 必须返回 `WorkflowCompletion`，包含 workflow terminal status、summary、
  files_changed、tests 和 unresolved。
- 两个 schema 均通过 LangChain `ToolStrategy` 暴露为结构化完成工具。
- 普通文本不再视为成功结束。middleware 检测到没有 tool call 的 AI response 后，
  将其重定向到只保留结构化完成工具的 finalization call。

### 2.2 卡死检测

默认策略：

| 条件 | 阈值 |
| --- | ---: |
| 连续相同 tool call | 3 |
| ABAB tool cycle | 3 组 |
| 连续 tool error | 3 |
| 连续无新 action/observation | 5 |
| 未提交 completion 的 model call | 32 |

触发后不直接抛异常，而是删除普通 filesystem/execute tools，只允许模型基于已有
上下文返回结构化 completion。`recursion_limit=200` 保留为最终保险丝。

## 3. 配对实验

模型与服务配置：Qwen3-Coder-30B-A3B-Instruct-FP8，SGLang 0.5.2rc1，单卡
RTX 6000 Ada，`max_total_tokens=163840`，`mem_fraction_static=0.952`。

| Run | Guard | 结果 | LLM | Tool | 墙钟 |
| --- | --- | --- | ---: | ---: | ---: |
| `planned-1-recursion200-guard-gate` | 开 | 1/1 语义终止 | 80 | 74 | 64.2 s |
| `planned-1-recursion200-semantic-only` | 关 | `GraphRecursionError(200)` | 135 | 181 | 331.6 s |
| `planned-1-recursion200-guard-v2` | 开 | 1/1 语义终止 | 82 | 75 | 73.8 s |

semantic-only run 中：

- 3 个 child 在 19、20、21 次左右的 LLM call 后自然提交结构化结果。
- 1 个 child 在 7 次调用后输出普通文本，被 runner 判为缺失 `ChildCompletion`。
- parent 持续执行到图的 200-superstep 上限，最终抛出 `GraphRecursionError`。

因此，提高 recursion limit 不能解决不收敛，只会将原来约 15 轮的失败延后。显式
completion schema 能让部分 agent 自然结束，但不能单独保证 parent 收敛。

最终 v2 run 中：

- 5/5 agent 均产生结构化 completion。
- 4 个 agent 因连续 3 次真实工具错误进入 finalization。
- 1 个 agent 的普通文本结束被协议层重定向并修复。
- 没有 `GraphRecursionError`、scheduler exception 或 transfer watchdog。

连续错误主要来自错误工作目录、镜像缺少 `mpmath`，以及离线 Docker 中尝试
`pip install`。因此本轮 guard 触发不是重复哈希误报，但暴露出 SWE-bench 镜像依赖
准备仍不完整。

## 4. 严格边界

- 最新 workflow 的结构化状态是 `blocked`，`patch_chars=0`。本轮只证明控制流
  有界终止，不证明 SWE-bench 业务成功。
- v2 峰值 KV pool pressure 约 27.7%，不能用于评价 BeliefKV 迁移收益。
- 三次模型执行的具体 trajectory 不完全相同，LLM/tool 数量只能作为 gate 级证据，
  不能直接作为正式性能 A/B。
- completion 中模型声称修改了文件，但实际 git workspace 未修改；后续正式评测应以
  git diff 和 harness 结果为准，不信任模型自报的 files_changed。

## 5. 产物

- `experiments/archive/20260727/superseded_raw/deepagents_swebench/20260717T101611Z/planned-1-recursion200-guard-gate/`
- `experiments/archive/20260727/superseded_raw/deepagents_swebench/20260717T101611Z/planned-1-recursion200-semantic-only/`
- `experiments/archive/20260727/superseded_raw/deepagents_swebench/20260717T101611Z/planned-1-recursion200-guard-v2/`

回归测试：

```text
tests/test_deepagents_adapter.py + tests/test_deepagents_swebench.py
19 passed
```
