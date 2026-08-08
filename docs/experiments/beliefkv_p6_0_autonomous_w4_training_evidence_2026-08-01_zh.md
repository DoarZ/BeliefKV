# BeliefKV P6.0 Autonomous W4 训练前证据（2026-08-01）

## 结论

P6.0 第 1 至第 3 步已经完成：analyzer 可同时读取 legacy agentic 与当前 Deep Agents trace，LLM
调用只允许按 native request ID、workflow ID 和 invocation ID 精确关联；版本化训练数据导出器已经
生成 request、GPU service、external wait、reentry 和 PCIe 表，并通过唯一性、外键和 JOIN 闭包
校验。整个过程只离线读取固定 raw，没有重新启动 GPU 或修改在线策略。

当前可以开发并训练 remaining decode demand、status-conditioned tool survival 和
submit-to-complete transfer service；不能训练 exact UnlockHazard。reentry 还缺逐 call 的显式 censor
cause，且全部样本来自 SymPy，不能声称跨项目泛化。

## 输入与产物

- 固定输入：`experiments/raw/p5g_autonomous_w4/20260801T115617Z/`。
- Coverage：
  `experiments/processed/p6_0_coverage_20260801/p5g_autonomous_w4_20260801T115617Z.json`。
- Dataset manifest：
  `experiments/processed/p6_0_coverage_20260801/p5g_autonomous_w4_20260801T115617Z_dataset/dataset_manifest.json`。
- 数据表总大小约 142 MB；输入和输出均记录 SHA-256。

复现命令：

```bash
conda run --no-capture-output -n beliefkv \
  python scripts/characterize_p6_coverage.py \
  experiments/raw/p5g_autonomous_w4/20260801T115617Z \
  --output experiments/processed/p6_0_coverage_20260801/p5g_autonomous_w4_20260801T115617Z.json \
  --dataset-dir experiments/processed/p6_0_coverage_20260801/p5g_autonomous_w4_20260801T115617Z_dataset
```

## 覆盖率

| 标签 | 结果 | 判定 |
| --- | ---: | --- |
| exact request identity | 575/575 | 可训练；ordinal fallback 已禁用 |
| token-demand completeness | 575/575 | 可训练 |
| GPU service request coverage | 575/575 | 可训练 |
| conditioned GPU batch samples | 45,201 | queue/HTTP wait 已排除 |
| request-level service intervals | 108,344 | 字段完整率 100% |
| exact decode token delta | 107,626/107,626 | 可训练 remaining service |
| exact incremental action boundary | 0/575 | 不可训练 UnlockHazard |
| runtime-only action boundary | 575/575 | 只能描述完整响应后的 action |
| reentry cause | 539/559，96.42% | 未过 gate |
| external waits | 600 | 可做 status-conditioned survival |
| complete JOIN reentry | 5/5 | 可作结构组合样本，数量不足 |
| PCIe/HiCache operations | 1,535 | 条件字段完整率 100% |
| direct DMA timestamp | 0/1,535 | 只能训练 submit-to-complete，不能冒充纯 DMA |

45,201 个 GPU batch sample 展开为 108,344 个 request interval。每行保留共享 batch elapsed time 和
batch size，不把整段时延错误地当成单请求独占 service。

## 数据契约

| 表 | 行数 | 含义 |
| --- | ---: | --- |
| `request_calls.jsonl` | 575 | 一次 agent-visible LLM call、token demand、action 和 eligibility |
| `gpu_service_intervals.jsonl` | 108,344 | batch 内每个 request 的 prefill/decode service interval |
| `external_waits.jsonl` | 600 | tool start 到 terminal return，保留 success/error/censor 状态 |
| `reentries.jsonl` | 605 | 600 个 tool return 和 5 个 closure-complete JOIN |
| `pcie_operations.jsonl` | 1,535 | direction/bytes/pages/tier/pinned/concurrency/allocator/callback |

完整性检查为 PASS：无 request/tool/reentry/command ID 重复，无 service 外键悬空，无缺失 request 或
PCIe command ID，也没有缺少 member RETURN 却被标为可训练的 JOIN。

split 单位固定为 `dataset + project`，按 SHA-256 分桶。当前只有 SymPy 一个 project，因此 575 个
request 全落在同一个 split，并由 manifest 明确告警。这批数据只用于 schema、特征和训练代码开发，
不能单独报告 validation/test 泛化结果。

## 未闭合问题

1. 620 个 function-call occurrence 只有 600 个 tool start，缺少 20 个 reentry。该差值与 workload
   summary 中 20 次 duplicate-tool suppression 数量一致，但 raw trace 没有逐 call suppression event，
   因此不能事后把具体 call 强行标成 guard-censored。下一轮 runtime event 必须携带 request ID、
   tool-call ID 和 censor reason。
2. exact action boundary 仍为 0。第一版 `RUNNING_LLM` 必须使用 remaining decode demand，不得宣称
   预测 TOOL/SPAWN 在第几个 token 可执行。
3. PCIe operation 有完整 submit-to-complete 和条件特征，但没有真实 DMA start。模型目标必须命名为
   callback/operation latency，而不是纯 PCIe copy time。
4. 该 workload 4/4 system JCT eligible，但 0/4 native-agent JCT eligible、2/4 task measurement valid。
   GPU/PCIe 物理标签仍可用于机制模型；agent JCT 和任务正确率不能作为训练或性能结论。

## 下一步

进入 P6.2 前先增加逐 call suppression/censor 事件协议，并从多个 SWE-bench project 与至少一种非
coding agent workload 收集相同 schema。模型开发可以立即使用当前数据实现 remaining decode demand、
tool survival 和 transfer-latency baseline，但模型选择与 calibration gate 必须等跨项目 split 后执行。
