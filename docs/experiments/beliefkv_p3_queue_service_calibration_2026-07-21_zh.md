# BeliefKV P3 Queue/Service Resimulator 标定

日期：2026-07-21

状态：GPU service curve holdout gate 通过；完整 physical counterfactual gate 未通过

## 1. 目的与边界

本实验只标定 frozen-trace resimulator 的 GPU prefill/decode 服务时间，不评价 BeliefKV
策略收益。标定数据不包含 HTTP 和 admission queue 时间；模型只有在 allocator、future KV
growth 和 cache-hit outcome 也能针对候选策略重算后，才允许用于 O0-O3 JCT。

固定环境：

- Qwen3-Coder-30B-A3B-Instruct-FP8；
- RTX 6000 Ada GPU0，TP=1；
- SGLang 0.5.2rc1，overlap scheduler 和 CUDA graph 开启；
- `mem-fraction-static=0.952`，KV pool 163,840 tokens（15 GiB）；
- `chunked-prefill-size=4096`，decode batch size 为 1/2/4。

## 2. Observer 语义修复

SGLang overlap scheduler 会先 launch 当前 batch，再处理前一个 batch 的结果。直接记录
`launch -> process_batch_result` 会把前序 GPU batch 的排队时间计入当前 service time。本轮将
一次 batch 的 service interval 定义为：

```text
service_start = max(current_launch, previous_batch_completion)
service_ms    = current_completion - service_start
```

audit 同时保留 `launch_to_completion_ms`，便于检查 overlap queue。observer 只接受
`service-calibration:{train|holdout}:{prefill|decode}-*`，并丢弃 prefill case 的输出 token 和
decode case 的短 prefill，避免 phase 污染。shutdown 时 785 个 sample 全部配对，pending launch
为 0，没有 `gpu_service_sample_failed`。

## 3. 为什么一维吞吐模型无效

首版 `launch + tokens / rate` 在单 chunk 样本上可通过，但跨 4096-token 边界后 prefill
holdout P95 达 55.65%。原因不是简单的 context 长度线性项，而是首 chunk 与 continuation
chunk 的执行曲线明显不同。例如同一 trace 中：

- 首个 4096-token chunk 约 276--281 ms；
- continuation 1036-token chunk 为 16.83 ms；
- continuation 4096-token chunk 约 271--275 ms。

最终模型使用两条按 chunk token 数建立的 piecewise-linear service curve：首 chunk 和
continuation chunk。train 点先按 64-token 邻域合并，再用 weighted isotonic regression 强制
服务时间非递减。该模型不读取 workflow、agent role 或 task 标签；affine 参数仅作为曲线覆盖
外的保守回退。

## 4. 独立 Holdout 结果

正式运行包含 39 个请求：6 个 train prefill episode、5 个 holdout prefill episode，以及
train/holdout 各 6 个 decode episode。多 chunk 覆盖如下：

| 覆盖项 | 数值 |
|---|---:|
| train multi-chunk prefill episode | 2 |
| holdout multi-chunk prefill episode | 2 |
| 首 chunk curve points | 5 |
| continuation curve points | 2 |
| decode batch sizes | 1, 2, 4 |

预注册门槛为各 phase 及总体 holdout P95 relative error 小于等于 25%。最终结果：

| 指标 | 数值 |
|---|---:|
| overall relative error P50 | 4.60% |
| overall relative error P95 | 20.09% |
| prefill relative error P95 | 4.51% |
| decode relative error P95 | 23.49% |
| absolute error P95 | 129.20 ms |

模型 `queue-service-c3e9318662e6a828b2f57256` 标记为 `calibrated=true`。loader 会同时检查
artifact schema、`episode_piecewise_isotonic_v1` 算法、`gpu_service_interval_v1` 时间语义、
coverage、rejection list 和 model 状态，失败诊断文件不能被误加载。

## 5. 产物

- [正式 raw audit](../../experiments/raw/service_calibration/20260721T091448Z/server/runtime_audit.jsonl)
- [客户端 manifest](../../experiments/raw/service_calibration/20260721T091448Z/client/benchmark_manifest.json)
- [正式 service model](../../experiments/processed/service_calibration_20260721T091448Z/queue_service_model.json)
- [单 chunk 调试运行](../../experiments/raw/service_calibration/20260721T090835Z/server/runtime_audit.jsonl)

同目录的 `attempt*.json` 是 fail-closed 过程证据，不是可部署模型。

## 6. 剩余风险

- decode P95 23.49% 接近 25% 门槛，正式论文实验需要更多 seed/repeat 和置信区间；
- 当前只覆盖 batch size 1/2/4、单模型和单 GPU，不能移植到其他模型或 SGLang 配置；
- service curve 没有建模 preemption、mixed prefill/decode、CUDA graph miss 或高 pressure 下的
  retraction；
- 旧 P3 agent trace 的 `request_physical_delta` 只对应原调度的 cache-hit outcome，不能作为
  候选调度的 exact future growth；
- 因此 `FrozenTracePlanEvaluator` 对该旧 trace 仍令
  `physical_actions_recomputed=false`，`JointPlanOracle` 必须拒绝 JCT。

下一步不是放宽 physical gate，而是记录可重建的 request prefix identity，并在 resimulator 中
针对每个候选调度重算 Radix hit、allocator growth 和 bundle lifecycle。
