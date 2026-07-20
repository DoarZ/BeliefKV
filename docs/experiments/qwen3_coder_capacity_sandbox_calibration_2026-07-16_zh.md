# Qwen3-Coder 单卡 KV 容量、上下文窗口与工具沙盒标定

日期：2026-07-16

## 1. 暂定结论

在 GPU 0（NVIDIA RTX 6000 Ada 48 GiB）、TP=1、SGLang 0.5.2rc1、
Qwen3-Coder-30B-A3B-Instruct-FP8、BF16 KV、CUDA Graph 最大 batch 16 的固定
配置下：

- `mem_fraction_static=0.952` 自动得到 `167,816 tokens`，即 15.364 GiB KV；
- 该点完成 5 次 10 路独立前缀压力测试，全部请求成功，resident KV 峰值均为
  `167,283 tokens`（99.68%），最低剩余显存为 1,179 MiB；
- `mem_fraction_static=0.953` 得到 `168,329 tokens`，请求虽全部成功，但最低
  剩余显存只有 1,011 MiB，低于预先固定的 1,024 MiB 安全线，因此判失败；
- 暂定容量边界为 `0.952 / 167,816 tokens`，粒度为 0.001；
- 正式实验默认使用更保守且可解释的 `163,840 tokens`，恰好 15.000 GiB KV，
  不直接运行在标定边界上。

这是当前硬件、驱动、SGLang commit、模型量化和 CUDA Graph 形状下的暂定边界，
不是跨环境常数。后续若出现 OOM、allocator 异常或 workload 特有临时显存峰值，应
重新标定，而不是修改既有实验结果。

## 2. KV 容量计算

本地模型配置为 48 层、4 个 KV heads、head dimension 128，KV dtype 为 BF16。
因此每个 token 的 KV 大小为：

```text
2 (K/V) * 48 * 4 * 128 * 2 bytes = 98,304 bytes = 96 KiB
```

所以 `163,840 tokens` 对应 15 GiB，`167,816 tokens` 对应 15.364 GiB。SGLang
`/get_server_info` 在 0.952 点报告的权重、KV 与 CUDA Graph 分别为 29.25、
15.36 和 0.18 GiB。

## 3. 标定方法

每个容量点均使用以下条件：

1. 服务端逻辑 context 为模型原生的 262,144 tokens，chunked prefill 为 4,096；
2. 启用 CUDA Graph，捕获 batch size 1、2、4、8、16；
3. 首先重放完整 Qwen Code 请求，显式验证 `max_tokens=32,768` 可被服务端接受；
4. 随后 flush radix cache，并同时发送 10 个请求；每个请求在 system prompt 最前
   插入不同 marker，使其不能共享后续长 prefix；
5. 请求保留完整 Qwen Code system prompt 和 tool schema，不做压缩；
6. 每 50 ms 查询 SGLang `/get_load` 与 Prometheus
   `sglang:num_used_tokens`，后者作为 resident KV 的服务端观测值；
7. 每 200 ms 采样 `nvidia-smi`，记录最低空闲显存；
8. 通过条件为：零请求失败、服务仍存活、resident KV 峰值至少达到 pool 的 80%，
   且最低空闲显存不少于 1,024 MiB。

严格边界结果如下：

| 配置 | Pool tokens | 重复 | Resident 峰值 | 最低空闲显存 | 结果 |
|---|---:|---:|---:|---:|---|
| 0.952 + graph | 167,816 | 5 | 167,283（99.68%） | 1,179-1,203 MiB | 通过 |
| 0.953 + graph | 168,329 | 1 | 167,283（99.38%） | 1,011 MiB | 失败 |

早期 0.94、0.95、0.96 扫描只使用客户端 token 总和估计压力，保留为探索性证据；
最终边界只采用启用 metrics 后的服务端 resident 数据。原始 JSON 位于
`experiments/calibration/qwen3_coder_30b_a3b_fp8/20260716/`。

## 4. 上下文窗口修正

模型 `config.json` 声明 `max_position_embeddings=262144`，因此 SGLang 使用
`--context-length 262144`。不能再通过删除基础 system prompt、删减工具 schema 或
把输出预算压到 2K/4K 来规避 32K 服务端拒绝。

但逻辑模型窗口和单卡物理 KV pool 是两个约束：SGLang 单请求仍不能超过当前
`max_total_tokens`。当前配置因此采用：

```text
model native / SGLang logical context: 262,144
Qwen Code and Codex advertised context: 163,840
physical KV pool default:              163,840
effective output budget in smoke:       32,768
```

Qwen Code 0.19.10 源码中的 fallback 输出常量是 32,000，但当前 OpenAI provider
路径捕获到的实际请求为 32,768；实验以线上的实际 request log 为准。通过的沙盒
smoke 首轮含 18,748 个 prompt tokens、15 个工具 schema 和 32,768 输出预算，服务端
正常接受。客户端窗口不声明为 262K，是因为当前单卡并没有 262K 的物理 KV 容量；
虚报该值只会在长会话中造成后端拒绝或 OOM。

## 5. 工具调用安全方案

成熟方案的共同模式不是在真实宿主机上逐个拒绝危险工具，而是给模型完整任务所需
权限，同时把执行环境做成每个 sample 独享、可销毁的容器：

- Qwen Code 官方建议本地或共享机器在 headless/yolo 模式下启用 sandbox，并同时
  固定 wall-time、turn 和 tool-call budgets；
- Qwen Code 在 Linux 上使用 Docker/Podman，默认会挂载 workspace 和 `~/.qwen`，
  因而 workspace 本身必须是一次性副本；
- Inspect AI 为每个 sample 创建独立 sandbox，并支持容器资源限制；
- SWE-bench 使用完全容器化的 Docker evaluator 保证实例隔离和可复现性；
- OpenHands 同样把 agent 的 shell/code execution 放在 Docker sandbox 中。

BeliefKV 当前实现为：

1. 每个矩阵 condition 从固定 Git commit 创建 `--no-local` 一次性 clone；
2. Qwen Code 使用 `approvalMode=yolo` 和固定镜像
   `ghcr.io/qwenlm/qwen-code:0.19.10`；
3. 预先创建 `qwen-code-sandbox-proxy` bridge，宿主网关固定为
   `172.20.0.1`；SGLang 只监听该私有接口，不监听物理网卡；
4. agent 容器进入无外部路由的 internal network，单独的代理容器只放行
   `172.20.0.1:18000`，其他 HTTP/CONNECT 目标返回 403；
5. 容器 drop all capabilities，只补 UID/GID 所需 capability，启用
   `no-new-privileges`，限制为 16 CPU、16 GiB RAM、1,024 PIDs；
6. runner 拒绝在没有 disposable marker 的 workspace 中启动 sandboxed yolo；
7. condition 结束后保存 workspace status 和 binary diff，再销毁副本；
8. Qwen 输出中的 permission rejection 非零时，样本标记为 invalid；
9. 正式任务不再显式拒绝 shell、edit、write、test 或 agent 工具。

通过的能力 smoke 共执行 6 次 shell/write/read 调用，全部为 `auto_accept`、零
permission rejection。临时 workspace 内写入成功，宿主越界路径不存在；本地模型
API 经代理成功访问，对 `example.com:443` 的 CONNECT 被明确拒绝。首轮请求含
18,838 prompt tokens、15 个工具 schema 和 32,768 输出预算。证据位于
`experiments/calibration/qwen3_coder_30b_a3b_fp8/sandbox_smoke/20260716T105500Z/`。

当前网络策略适用于不需要外网的 coding workload。search/web workload 不能复用此
配置，应提供另一份显式域名 allowlist，并独立验证 DNS、重定向和目标端口策略。

## 6. 负结果与防回归

两次失败 smoke 被保留而没有覆盖：

- `20260716T100000Z.incomplete`：6-turn budget 不足，触发
  `FatalTurnLimitedError`；
- `20260716T100100Z.incomplete`：测试脚本创建了临时目录但未切换 cwd，导致 Qwen
  实际挂载真实 BeliefKV 仓库。模型生成的两个 probe 文件已确认并删除，该样本无效。
- `20260716T102500Z.incomplete`、`20260716T103500Z.incomplete` 和
  `20260716T104500Z.incomplete`：启用 internal network 后，SGLang 仍只监听
  `127.0.0.1`，代理容器无法连接宿主 loopback。修正为仅监听私有 bridge 网关后通过。

第二个负结果说明“使用 Docker”本身不足以证明安全。必须同时验证模型看到的 cwd、
实际 mount、宿主路径不存在，以及运行后容器被销毁。

## 7. 复现入口

启动保守默认服务：

```bash
cd /home/longhao/experiment/BeliefKV
gateway="$(scripts/prepare_qwen_sandbox_network.sh)"
HOST="${gateway}" MEM_FRACTION_STATIC=0.952 MAX_TOTAL_TOKENS=167816 \
  scripts/launch_qwen3_coder_qwencode_smoke.sh
```

运行完整工具沙盒 smoke：

```bash
OPENAI_BASE_URL=http://172.20.0.1:18000/v1 \
  scripts/smoke_qwen_code_sandbox.sh
```

容量压力脚本为 `scripts/stress_qwen_code_kv_pool.py`。每次变更 GPU、驱动、模型、
KV dtype、SGLang commit、CUDA Graph shape 或并发上限后，都应重新执行容量标定。

## 8. 参考

- Qwen Code headless safety: https://qwenlm.github.io/qwen-code-docs/en/users/features/headless/
- Qwen Code sandbox: https://qwenlm.github.io/qwen-code-docs/en/users/features/sandbox/
- Qwen Code allowlist proxy example: https://github.com/QwenLM/qwen-code/blob/main/docs/developers/examples/proxy-script.md
- Inspect AI sandboxing: https://inspect.aisi.org.uk/sandboxing.html
- SWE-bench: https://github.com/SWE-bench/SWE-bench
- OpenHands Docker sandbox: https://docs.openhands.dev/openhands/usage/sandboxes/docker
