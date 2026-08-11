# BeliefKV P6 Morphology Veto Canary v1

日期：2026-08-10

## 结论

本轮完成了预声明 Xarray w8 上的 byte-only veto treatment，但没有执行预测性
`PREPARE_HOST`。根因不是 workload 缺少形态覆盖，而是在线 `TransferServiceCurve` 使用默认
`min_samples=8` 重新判定了以 3 个受控重复校准的 artifact，导致 byte-only 与 morphology-aware
两个 arm 都静默回退到静态 24 GB/s。旧 authority gate 又将 `shape_unsupported` 视为有效 veto，
于是形成 16 个伪 paired veto；11 个在发布前 stale，另 5 个在 safe point 被拒绝，最终
0 commit、0 predictive D2H。

修复 warm-start 资格语义后，对 16 个源 snapshot 中可持久化重放的 14 个进行一致模型 replay：
16 个 PREPARE candidate 中 byte-only 与 morphology-aware 都各有 2 个 eligible/selected，
promotion=0、veto=0、selected-action change=0。形态只改变 14 次 timing estimate 和 6 次
feasibility reason。按照预先声明的 go/no-go gate，M6 decision relevance 未通过；morphology
降级为 transfer cost/OOD safety 辅助模型，不再作为 P6 核心 insight。由于 treatment 没有执行
动作，未运行约两小时的 P5 control；此时 control 无法提供 action-level 反事实收益。

## 冻结契约

- manifest：`configs/p6/veto_canary_v1/manifest.json`；
- manifest digest：`d8e052214e5d12031ebec4029571a54fcdb75962e04e76dae0ffed84760d22bf`；
- workload：与 Decision-Relevance v2 相同的 8 个 Xarray workflow，单批 concurrency 8；
- treatment：byte-only 主模型 + morphology-aware counterfactual；
- authority：同一 package 必须 byte-only eligible 且 counterfactual rejected；
- 全局最多一笔 `PREPARE_HOST`，预测 prefetch/retraction 关闭；
- 无强制 intent、无运行后阈值修改、无自适应任务筛选。

运行目录：`experiments/canary/p6_morphology_veto_v1/treatment`。运行期间的 74-file source
fingerprint start/end 均为
`da3d6fc7f65c3342fb5f4137177b4da190e9a7443e81a3f8a4443f093e9e4216`，未发生在线改码。
本文使用修复后的 analyzer 对历史日志重新归因；后述严格 runtime gate 是实验后修复，不能追溯
声称当时已启用。

## Workload 结果

- 总时长：7,216.16 s；
- 7/8 workflow 完成，626 次业务 LLM call、753 次工具调用、6 个动态 subagent；
- 5/8 measurement-valid，3/8 native-agent JCT eligible，7/8 system-JCT eligible；
- 3 个 workflow 经 guard 收尾，不进入 native-agent JCT；
- 14 次重复工具调用被抑制，runtime-control delivery 无降级；
- `pydata__xarray-7233` 连续两次 120 s `docker run` 超时，且未创建容器。该失败发生在任何
  agent event、LLM 或工具调用之前，属于 harness censor，不进入 P6 策略证据。

P5 物理正确性通过：shutdown ACK 完成，最终 0 running/0 waiting，无未决 transaction、lease、
restore obligation 或无来源 JointPlan 的在线动作。81 笔、约 11.39 GB 的已完成传输均为 observed
terminal-private cleanup，不是预测性迁移。

## 在线 Veto 漏斗

| 阶段 | 数量 | 结果 |
|---|---:|---|
| byte-only 主模型选择 PREPARE，counterfactual 拒绝 | 16 | 旧 gate 记为 raw paired veto |
| morphology counterfactual shape-supported | 0 | 受错误 `min_samples=8` 门槛影响 |
| 发布前 causal read-set 校验失败 | 11 | invocation revision/state 或 context epoch 已变化 |
| semantic intent 发布 | 5 | 未产生物理副作用 |
| safe-point 拒绝 | 5 | 均含 observed residency 优先和剩余窗口不足 |
| commit / predictive D2H | 0 | 无 transaction、ACK 或收益样本 |

16 次 paired evaluation 的控制面开销：

| 指标 | P50 | P95 | 最大值 |
|---|---:|---:|---:|
| paired worker compute | 1,009.86 ms | 1,940.13 ms | 2,675.22 ms |
| primary planning | 340.95 ms | 870.42 ms | 1,111.41 ms |
| worker queue wait | 6.28 ms | 71.09 ms | 129.63 ms |
| trigger-to-validation | 1,082.51 ms | 2,206.55 ms | 2,735.47 ms |

这些时延不会阻塞 observed scheduler，但会使秒级因果 read-set 频繁过期。当前串行执行完整
byte-only 与 morphology-aware risk evaluation 的实现只适合长工具/WAIT_JOIN 窗口；后续应复用
相同 particles、candidate timeline 和 recourse 计算，只替换 transfer-cost evaluation。

## 根因与修正 Replay

artifact 的校准契约为 `min_samples=3`，在线 curve 则以 `min_samples=8` 初始化。旧实现载入 6 条
warm-start 样本后仍按 8 条门槛判断，因而所有 bucket 都变成 unsupported。以 1.96 GB/119 extents
为例：错误在线路径给出约 102.09 ms 的静态 morphology fallback；恢复 artifact 门槛后，正确
shape-aware P90 为 788.15 ms，byte-only 为 491.53 ms。

修正 replay 产物位于 `treatment/replay_corrected/`：

| 指标 | byte-only | extent-count-aware |
|---|---:|---:|
| replayed snapshots | 14 | 14 |
| paired PREPARE candidates | 16 | 16 |
| positive-benefit PREPARE | 12 | 9 |
| eligible / selected PREPARE | 2 / 2 | 2 / 2 |
| shape-supported PREPARE | 16 | 14 |

严格配对结果为 timing changed 14、reason changed 6，但 eligibility flip、promotion、veto 和
selected-action change 均为 0。另 2 个 source snapshot 未出现在去重持久化文件中，不能重放；
这不改变已重放候选中“没有 action flip”的结论。

## 修复

实验后已完成以下 correctness 修改：

1. warm-start bucket 保留 artifact 的校准门槛；在线新桶仍使用运行时 8-sample 门槛，两者不再
   被一个全局阈值混淆；
2. immutable service snapshot 显式携带 `warm_start_min_samples` 和每个 bucket 的
   `warm_start_usable_count`；
3. `PairedPrepareVetoEvidence` 显式携带 counterfactual shape support、extent count、shape digest
   和 transfer estimate；
4. `evaluate_paired_prepare_veto()` 要求 counterfactual shape-supported，`shape_unsupported` 不再
   能授权 treatment；
5. `PredictiveIntent(authority_gate="byte-only-veto")` 必须携带 supported morphology evidence；
6. 新 freeze manifest 必须满足 `supported_shape_veto_gate`，raw veto 不再足够；
7. replay 支持按 snapshot ID 定向验证，analyzer 分开报告 raw veto、supported veto 和发布漏斗。

## 下一阶段

1. 停止为 morphology action flip 追加 GPU matrix 或筛选新 trace；
2. 保留 extent-count-aware curve 作为统一 transfer service model 和 OOD fail-closed 机制；
3. 将 P6 核心重新收敛到 FrontierBelief 对 agent causal slack/reentry demand 的预测，以及该 belief
   是否能让 `PREPARE_HOST/PREFETCH_GPU` 相比 observed P5 产生稳定净收益；
4. 后续预测实验必须先通过一致 service-contract 校验，再开放任何在线 intent；
5. paired evaluator 的单次共享场景优化只在新的核心策略出现可执行 action 后再做，避免继续优化
   已未通过 decision-relevance gate 的 morphology treatment。
