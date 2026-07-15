# BeliefKV SWE-bench Pilot（2026-07-15）

## 1. 实验定位

本轮只验证第一阶段 SWE-bench 链路，不用于比较 BeliefKV 的性能收益。需要分开看待
以下三个结论：

1. 官方 evaluator 可用：固定实例的 gold patch 通过官方 Docker harness；
2. 真实 agent 链路可用：Qwen2.5-7B、mini-SWE-agent、SGLang 和 BeliefKV
   共同产生了完整的 LLM/tool/runtime audit trace；
3. 本次任务未解决：模型提交为空 patch，官方 evaluator 将其归类为
   `empty_patch`，不是系统运行错误。

## 2. 固定版本与数据

| 项目 | 固定值 |
|---|---|
| 数据集 | SWE-bench Verified，`test` split，500 条 |
| 数据 revision | `91aa3ed51b709be6457e12d00300a6a596d4c6a3` |
| Pilot 实例 | `sympy__sympy-20590` |
| SWE-bench harness | 4.1.0，commit `f7bbbb2ccdf479001d6467c9e34af59e44a840f9` |
| Agent runtime | mini-SWE-agent 2.4.5，commit `388da74aad620a384ab47669b17c52133e30e7c3` |
| Serving runtime | SGLang 0.5.2rc1，commit `18f91eb639084825717c0e3c3c7273492812ab71` |
| Model | `/opt/downloaded_models/Qwen/Qwen2.5-7B-Instruct` |
| BeliefKV policy | reactive；predictor 和 shadow copy 均关闭 |

Agent 只接收 `instance_id` 和 `problem_statement`。`patch`、`test_patch` 等 gold
字段不会进入模型上下文。

## 3. 已执行步骤

### 3.1 Gold gate

固定输入由
`configs/workloads/swebench_verified_gold_gate.json` 管理。官方 evaluator 结果为：

```text
total=1, completed=1, resolved=1, errors=0
```

这一步只证明 Docker image、数据字段、gold patch 和 evaluator 可用。

### 3.2 真实 agent pilot

服务端使用本地 Qwen2.5-7B 和 run-specific BeliefKV 配置：

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path /opt/downloaded_models/Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 18000 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --tool-call-parser qwen25 \
  --enable-beliefkv \
  --beliefkv-config \
  /home/longhao/experiment/BeliefKV/experiments/raw/swebench-sympy-20590-reactive-s0/server_config.json
```

Agent 入口为：

```bash
/home/longhao/miniconda3/envs/beliefkv-swe/bin/python \
  scripts/run_swebench_agent_trace.py \
  --instance-id sympy__sympy-20590 \
  --run-dir experiments/raw/swebench-sympy-20590-reactive-s0 \
  --event-socket /tmp/bkv-swe-sympy-s0.sock
```

需要保证调用用户可访问 Docker；当前登录会话可使用 `sg docker -c '...'`。

### 3.3 官方评测、冻结与复核

Agent prediction 继续走官方 `swebench.harness.run_evaluation`，结果为：

```text
submitted=1, empty_patch=1, errors=0
```

固定 trace 使用以下命令生成和复核：

```bash
python scripts/freeze_swebench_trace.py \
  --run-dir experiments/raw/swebench-sympy-20590-reactive-s0 \
  --evaluation-report \
  experiments/raw/swebench-sympy-20590-reactive-s0/evaluation/openai__Qwen2.5-7B-Instruct.swebench-sympy-reactive-s0.json \
  --destination \
  workloads/frozen/swebench_verified_reactive_qwen2_5_7b/sympy__sympy-20590/reactive-s0-pilot-v2

python scripts/verify_frozen_swebench_trace.py \
  workloads/frozen/swebench_verified_reactive_qwen2_5_7b/sympy__sympy-20590/reactive-s0-pilot-v2
```

## 4. Trace 结果

| 指标 | 值 |
|---|---:|
| Runtime events | 36 |
| LLM calls | 8 |
| Tool calls | 8 |
| Workflow span | 86.033 s |
| Tool time | 3.859 s |
| Prompt tokens（逐轮总和） | 14,963 |
| Radix cache-hit tokens（逐轮总和） | 13,098 |
| Uncached prompt tokens（逐轮总和） | 1,865 |
| Output tokens | 763 |
| Audit records | 59 |
| Event deliveries | 18，全部成功 |

冻结器检查连续 sequence、唯一 event ID、workflow/invocation 生命周期、LLM/tool
成对关系、audit/request 一致性、文件哈希，并将 trace 重放到 BeliefKV 控制平面。
最终 invocation 状态为 `done`。

## 5. 失败归因与使用边界

模型定位到了 `sympy/core/symbol.py`，但生成的 `sed` 表达式没有匹配缩进后的
源码行，因此工作树未发生修改，后续 `git diff` 为空。这解释了空 patch；agent
runner、Docker 环境、事件通道和 evaluator 均没有报告错误。

该 pilot 可以用于事件接入回归和确定性重放，不能用于调度性能实验。它只有一个
workflow，没有形成 HBM pressure、workflow 竞争或 D2H/H2D 迁移。正式性能阶段至少
需要固定多个有非空轨迹的实例，构造可复现的并发到达过程，并在相同 trace 下比较
upstream SGLang、patched-disabled 和 BeliefKV reactive baseline。

此外，本次运行前没有自动保存 BeliefKV 源码快照；冻结清单记录的是运行后的文件
哈希并明确标记 `generation_exact=false`。下一轮正式 trace 必须在启动服务前生成
source/config/environment lock。
