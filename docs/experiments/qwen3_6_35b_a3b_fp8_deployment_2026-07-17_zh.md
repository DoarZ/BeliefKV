# Qwen3.6-35B-A3B-FP8 单卡部署记录

## 结论

Qwen3.6-35B-A3B-FP8 已部署到：

```text
/opt/downloaded_models/Qwen/Qwen3.6-35B-A3B-FP8
```

模型可在一张 NVIDIA RTX 6000 Ada 48 GB 上以 TP=1、128K context、8 个最大
并发请求运行。当前部署使用 vanilla SGLang 0.5.10.post1；BeliefKV 的
0.5.2rc1 runtime patch 尚未移植到该版本，因此本次结果只证明新模型和新 serving
runtime 可用，不证明 BeliefKV cache policy 已经接入。

## 环境选择

- conda 环境：`beliefkv`
- Python：3.10.20
- SGLang：0.5.10.post1
- PyTorch：2.9.1+cu128
- Transformers：5.3.0
- FlashInfer：0.6.7.post3
- SGLang kernel：0.4.1
- NVIDIA driver：570.195.03

本次直接原地升级既有 `beliefkv` 环境，没有创建新的 conda 环境或 SGLang
worktree。升级前后的环境快照均已保存，便于后续复现或回滚。

Qwen 官方建议 SGLang 0.5.10 或更高。0.5.11 及之后的当前 wheel 已切换到
PyTorch 2.11 和 CUDA 13 依赖，而该机器仍是 570 驱动和 CUDA 12.8。因此选择
0.5.10.post1 是当前硬件上支持 Qwen3.6 的最高保守版本，不是随意停留在旧版本。

升级前的完整 conda/pip 快照和升级 dry-run 位于本目录。旧的
`third_party/sglang` patched 0.5.2rc1 源码没有删除，但不再是当前环境中的
editable 安装来源。

## 模型完整性

模型通过 Qwen 官方 ModelScope 仓库下载。Hugging Face 首次请求出现 TLS EOF，
因此切换下载源；最终目录结构仍是标准 Transformers/Safetensors 格式。

- ModelScope 文件数：56
- Safetensors 分片数：42
- Safetensors 总大小：37,463,662,160 bytes（34.89 GiB）
- 索引 tensor 映射项：64,196
- 架构：`Qwen3_5MoeForConditionalGeneration`
- 文本层数：40
- 原生 context：262,144

所有 safetensors 文件均已通过 header 打开检查。

## 最终启动配置

```bash
cd /home/longhao/experiment/BeliefKV
./scripts/launch_qwen3_6_35b_a3b_fp8.sh
```

脚本以前台方式运行，使用 `Ctrl+C` 即可完整停止，不创建 systemd unit、Docker
容器或后台重试进程。默认参数为：

```text
CUDA_VISIBLE_DEVICES=0
TP=1
context_length=131072
mem_fraction_static=0.952
max_running_requests=8
chunked_prefill_size=4096
CUDA graph batch sizes=1,2,4,8
piecewise CUDA graph=disabled
MTP=disabled
reasoning parser=qwen3
tool parser=qwen3_coder
```

可以用环境变量覆盖模型路径、端口、context、显存比例和并发数。脚本会拒绝
SGLang 版本不等于 0.5.10.post1 的环境，避免静默运行在未校准版本上。

## 显存与缓存

最终 `mem_fraction_static=0.952` 配置的实际分配为：

| 组成 | 大小 |
| --- | ---: |
| 模型权重 | 34.19 GB |
| Mamba/Gated DeltaNet conv state | 0.11 GB |
| Mamba/Gated DeltaNet SSM state | 4.86 GB |
| BF16 K cache | 2.74 GB |
| BF16 V cache | 2.74 GB |
| CUDA Graph 额外占用 | 0.13 GB |
| 自动 KV token capacity | 286,932 token |

初始的 `0.90` 配置只能提供 221,974 token，最终配置增加了 64,958 token，
容量提高约 29.3%。由于 Qwen3.6 使用混合 GDN/Mamba 架构，扩容同时将 Mamba
cache 槽位从 62 增加到 82，并非所有新增显存都进入标准 attention KV pool。
CUDA Graph 捕获后 SGLang 报告仍有 1.58 GB 可用显存；`nvidia-smi` 显示 GPU 0
使用 47,516 MiB、空闲 1,003 MiB。

CUDA Graph 开启后，batch size 1 的稳定 decode 日志约为 121 token/s；完全关闭
CUDA Graph 时约为 12.8 token/s。因此最终配置保留受控 CUDA Graph，只捕获
1/2/4/8 四种 batch size。

`--language-only` 在 SGLang 0.5.10 中属于 encoder-disaggregation 功能，并且其
架构白名单尚未包含 `Qwen3_5MoeForConditionalGeneration`，不能用于本模型的
独立文本服务。本次部署没有使用 dummy encoder 或修改 site-packages 绕过校验。

## API 验证

服务地址：`http://127.0.0.1:18000/v1`

已验证：

- `/health` 返回成功；
- `/v1/models` 报告 `max_model_len=131072`；
- OpenAI chat completions 非 thinking 请求正确返回；
- `qwen3` parser 能将 thinking 内容分离到 `reasoning_content`；
- `qwen3_coder` parser 能返回标准 OpenAI `tool_calls`；
- 完整 thinking 样例生成 669 token，总耗时 5.69 秒；
- 短非 thinking 样例总耗时 1.42 秒。

最终验收请求精确返回 `BELIEFKV_QWEN36_READY`；强制工具调用请求返回
`get_weather({"city": "Beijing"})`，结束原因为 `tool_calls`。启动后的
`0.952` 扩容验收请求精确返回 `KV_POOL_0952_OK`。

## GPU 进程说明

启动脚本显式导出 `CUDA_VISIBLE_DEVICES=0`，因此该部署只使用物理 GPU 0。
SGLang 会创建 HTTP/管理进程和持有模型的 scheduler 子进程；`nvtop` 可能将它们
显示为两个独立进程，但不代表使用了两张显卡。验收时两个进程报告相同的 GPU
UUID，GPU 1 的显存占用为 0 MiB。

项目回归测试通过：`143 passed, 4 skipped`。跳过项是受运行环境限制或可选组件
控制的测试，不存在失败用例；`pip check` 同样通过。

响应 JSON、server 日志、metrics 和环境快照位于：

```text
experiments/calibration/qwen3_6_35b_a3b_fp8/deployment_20260717/
```

上述延迟是部署 smoke 数据，不是正式性能结论。RTX 6000 Ada 当前使用通用
FP8/MoE kernel config，正式实验前还需要固定 workload、预热状态并做重复测量。
