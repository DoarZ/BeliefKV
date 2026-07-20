# Qwen3-Coder-30B-A3B-FP8 + Qwen Code 单卡链路验证

日期：2026-07-16

> 2026-07-16 勘误：原始 smoke 使用 32K context、6 GiB KV pool 和显式只读
> 权限，只能证明协议链路，不能作为工具 workload 或性能结果。当前容量、窗口和
> sandbox 配置以
> `qwen3_coder_capacity_sandbox_calibration_2026-07-16_zh.md` 为准。

## 1. 目标与边界

本次实验验证以下真实运行链路：

```text
Qwen Code 0.19.10
  -> OpenAI-compatible chat/completions
  -> SGLang 0.5.2rc1 (TP=1)
  -> Qwen3-Coder-30B-A3B-Instruct-FP8
  -> Qwen Code tool / agent / child session
```

这是 runtime 可行性与协议 smoke，不是 BeliefKV 性能结果。原始服务端未启用
`--enable-beliefkv`，CUDA Graph 也被关闭，因此不能用原始吞吐数字与正式系统或
论文 baseline 比较。后续修正只扩大能力面，不追溯性改变原始结果。

## 2. 固定环境

| 项目 | 固定值 |
|---|---|
| 模型 | `Qwen3-Coder-30B-A3B-Instruct-FP8` |
| 本地路径 | `/opt/downloaded_models/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` |
| 模型分片 | 4 个 safetensors，头部均可解析，无 `.incomplete` 文件 |
| Qwen Code | `0.19.10` |
| Node.js | `22.20.0`，位于 `/opt/node-v22.20.0-linux-x64` |
| SGLang | `0.5.2rc1`，commit `18f91eb` |
| GPU | GPU 0，NVIDIA RTX 6000 Ada Generation 48 GiB，TP=1 |
| API | `http://127.0.0.1:18000/v1`；Docker sandbox 使用 bridge 地址 |
| 原始 smoke context | 32,768 tokens/sequence，仅保留为历史协议验证 |
| 当前服务端 context | 262,144 tokens；客户端声明 163,840 tokens |
| 原始 smoke KV pool | 65,536 tokens，共 6.00 GiB K/V |
| 当前保守 KV pool | 163,840 tokens，共 15.00 GiB K/V |

原始服务启动后，GPU 0 的实测显存占用为 37,712 MiB；这正说明原始 smoke 没有
制造目标场景的 KV pressure。重新标定的暂定边界为
`mem_fraction_static=0.952 / 167,816 tokens`，正式默认使用 163,840 tokens。

## 3. 必要兼容修复

### 3.1 输出预算与窗口

原始实验把 `samplingParams.max_tokens` 降到 2,048/4,096 来绕过 32K 服务端拒绝，
这会改变模型输出、工具调用和长 agent loop，不能用于正式对比。修正方案是把
SGLang 逻辑窗口设为模型原生 262,144，并把 Qwen Code 客户端窗口设为与保守物理
pool 对齐的 163,840；不再固定小输出预算，也不删减 system prompt 或 tool schema。

修正后的实际请求包含 18,748 个首轮 prompt tokens、15 个工具 schema，捕获到的
`max_tokens` 为 32,768，服务端正常接受。客户端不能声明 262K，因为当前单卡物理
KV pool 只有约 164K；逻辑模型窗口不能替代实际 KV 容量。

### 3.2 Qwen3-Coder tool-call parser

模型会生成完整的 `<function=agent>...</function></tool_call>`，但有时省略开头
`<tool_call>`。SGLang 0.5.2rc1 会把整段当普通文本，Qwen Code 因而看不到工具
调用。该现象与 Qwen Code 的公开本地模型 bug 报告一致：

- https://github.com/QwenLM/qwen-code/issues/176

本地回移了新版 SGLang parser 的容错语义：将开头的 `<function=...>` 识别为
隐式 tool call，并在流式 chunk 边界保留不完整 marker。新增 non-stream 与
stream 回归测试后，Qwen3-Coder detector 测试结果为 `16 passed`。

### 3.3 非交互权限与沙盒

原始实验只允许 `Agent`/`Read` 并显式拒绝 shell、edit、write 和 WebFetch，导致
执行轨迹偏离真实 coding workflow。因此该权限配置仅保留为只读诊断 profile。

当前正式 profile 使用 Docker sandbox 内的 `yolo`：模型可以调用完整工具，但每个
condition 只挂载从固定 commit 创建的一次性 clone；真实 BeliefKV 工作树不进入
容器。容器 drop capabilities、启用 `no-new-privileges` 并限制 CPU、内存和 PID。
任何 Qwen permission rejection 都会使样本无效，而不是作为普通 tool failure 计入。
通过的 smoke 有 4 次工具调用、零拒绝，临时目录写入成功且宿主越界写入失败。

## 4. 已验证结果

| 测试 | 结果 | 证据 |
|---|---|---|
| 基础 chat | 成功 | 返回 `QWEN_CODE_LOCAL_READY` |
| OpenAI tool call | 成功 | SGLang 返回结构化 `tool_calls` |
| Qwen Code 文件工具 | 成功 | `read_file=1/1`，正确读出项目名 `beliefkv` |
| 显式 subagent protocol smoke | 成功 | `agent=1/1`、`read_file=1/1`，归档脚本复测耗时 30.9 s |
| 完全自主仓库审查 | 失败 | 自主选择 `agent=0`，480 s wall budget 用尽 |
| 提示“自行决定是否委派” | 失败 | 自主选择 `agent=0`，12 turns 用尽 |

成功的父子链路中，parent 发出 `agent(subagent_type=Explore)`；child 事件带有
与父调用相同的 `parent_tool_use_id`，child 独立执行 `read_file`，结果随后作为
parent 的 `tool_result` 返回。这是 Qwen Code 运行时创建的真实子会话，不是运行
前生成的固定 subagent 数量或离线 trace。

## 5. 自主委派负结果

第一个不提 subagent 的跨模块审查会话共发出 22 次模型请求，其中 21 次成功；
工具统计为 `glob=1`、`read_file=15`、`agent=0`。原始 32K 服务端窗口在输入增长
到 29,283 token 后触发 overflow。这个失败同时混入了错误窗口配置，不能据此判断
扩大窗口后的自主委派率，后续必须在修正配置上重新测量。

第二个 prompt 明确允许模型自行决定是否委派及数量，但不指定数量。它共发出
14 次模型请求，工具统计为 `glob=6`、`read_file=6`、`agent=0`，最终触发
12-turn 限制。这说明：

1. Qwen Code runtime 支持动态 subagent，并不等于 30B-A3B 会可靠地主动使用它；
2. 当前模型倾向串行读取，复杂任务容易造成 parent context 膨胀；
3. 正式 workload 不能把显式要求 N 个 subagent 的 protocol smoke 当成自然负载；
4. BeliefKV 需要记录真实 `agent`/child 事件，并将“未委派”保留为 workload 行为，
   不能在 trace 生成器中补造 spawn。

因此，Qwen Code 比 Codex 更适合作为可观测、可修改的开源 runtime 接口，但当前
30B-A3B 只能证明机制可运行，尚不能替代高质量动态 subagent workload。正式实验
应同时报告自然 spawn rate、tool-call parse failure、turn/context exhaustion 和
任务成功率。

修正窗口、权限、沙盒和输出采集后的 Qwen Code/Codex 配对 gate 仍然得到
`spawn=0`；详见 `qwen3_coder_runtime_pair_gate_2026-07-16_zh.md`。因此原结论未被
早期 32K 窗口混杂因素推翻，但现在有了无 overflow、无权限拒绝的直接证据。

## 6. 复现

终端 1：

```bash
cd /home/longhao/experiment/BeliefKV
scripts/launch_qwen3_coder_qwencode_smoke.sh
```

终端 2：

```bash
cd /home/longhao/experiment/BeliefKV
scripts/smoke_qwen_code_subagent.sh
```

运行完整工具沙盒验证：

```bash
OPENAI_BASE_URL=http://172.20.0.1:18000/v1 \
  scripts/smoke_qwen_code_sandbox.sh
```

`run_qwen_code_local.sh` 会拒绝在真实工作树中启动 sandboxed yolo；矩阵 runner 会为
每个 condition 自动创建 disposable clone。

原始本轮日志保存在：

```text
/home/longhao/experiment/BeliefKV/experiments/raw/qwen-code-smoke/
```

该目录被 `.gitignore` 排除；正式实验应把不可变 raw log 通过 manifest/hash 纳入
BeliefKV 的实验产物流程，而不是提交到源码仓库。
