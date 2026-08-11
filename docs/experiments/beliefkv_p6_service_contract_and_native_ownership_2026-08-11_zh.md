# P6 Transfer Service Contract 与 Native Ownership 修复

日期：2026-08-11

## 1. 结论

本轮没有证明 morphology-aware policy 的在线收益。相反，它发现并修复了两个会污染 P6
结论的数据契约问题：

1. runtime 的在线样本门槛错误覆盖了 transfer artifact 的校准门槛；
2. native HiCache transfer 在完成时才查询 ownership，导致真实 D2H 无法归因到 context。

修复后的同源 replay 中，byte-only 与 extent-count-aware 都选择 2 个 PREPARE，promotion、
veto 和 selected-action change 均为 0。因此 morphology 降级为物理成本/OOD safety 模型。
P6 的下一项科学门槛是 causal useful-action oracle，而不是继续寻找 morphology action flip。

## 2. Sample-Gate 根因

校准 artifact 使用 `min_samples=3`，online server 配置使用 `service_curve_min_samples=8`。
旧实现加载了 3-run bucket，却在查询时统一要求 8 个样本，造成两个策略 arm 都回退到
`shape_unsupported_static_fallback`。在线 treatment 观测到的 16 个 veto 因而是伪结果。

新契约区分：

```text
runtime_min_samples  -> 当前 server 在线观测门槛
artifact_min_samples -> 持久化校准证据门槛
```

校准样本保留 provenance。一个 query 只要满足 runtime 门槛或 artifact 自身门槛即可获得
support；在线新增的少量样本不能借用 artifact 的较低门槛。

启动 preflight 输出：

```text
artifact_loaded
hardware_key
runtime_min_samples
artifact_min_samples
loaded_sample_count
artifact_bucket_count
supported_representative_count
contract_satisfied
```

加载样本为零或没有 supported representative 时 fail fast。该摘要同时写入
`runtime_initialized.transfer_service_contract`，供实验 harness 审计。

## 3. 修复后 Replay

对 veto treatment 中可恢复的 14 个源 snapshot 重新执行 paired replay：

| 指标 | 结果 |
|---|---:|
| paired PREPARE candidates | 16 |
| timing estimate changed | 14 |
| feasibility reason changed | 6 |
| eligibility changed | 0 |
| promotion | 0 |
| veto | 0 |
| selected-action changed | 0 |
| byte-only selected PREPARE | 2 |
| extent-count-aware selected PREPARE | 2 |

结论是物理形态会改变成本估计，但现有冻结证据没有证明它改变最终控制动作。旧在线 veto
不能作为论文证据，也不再触发新的 GPU morphology matrix。

## 4. Native Ownership 根因与修复

历史 treatment 记录了 1,922 次 native D2H，总量约 93.9 GB，但所有
`owner_context_ids` 都为空。原实现流程为：

```text
native transfer submit
  -> Radix/context ownership 继续变化
  -> DMA complete callback
  -> 查询当前 PageOwnershipIndex
```

context 可能已经解绑，node 也可能发生 mutation，因此完成时查询不是可靠标签。

新流程为：

```text
native transfer submit
  -> 冻结 owner_context_ids / epochs / extent_bytes / ownership_revision
  -> 保存到 HiCache operation metadata
  -> DMA complete callback 原样发布 submit snapshot
```

新记录使用 `ownership_attribution_semantics=submit_snapshot`。没有新字段的旧记录才使用
`completion_lookup`，二者不能在正式 oracle 中混合。

## 5. 验证

- service-contract/controller/SGLang 定向回归：161 passed；
- native HiCache ownership/telemetry 定向回归：144 passed；
- 新测试覆盖 context 在 submit 后解绑、complete 前 ownership 消失的情况；
- Python 编译检查通过；
- 本轮未启动 GPU 实验。

## 6. 证据边界

旧 trace 无法追溯恢复 submit-time ownership，因此只能继续用于：

- HBM pressure 时间线；
- native transfer 总量和完成时延；
- GPU service 与控制面开销。

它不能用于：

- 判断哪个 parked parent 成为 reactive victim；
- 计算 shadow copy 的 useful/wasted bytes；
- 证明 PREPARE 替代了关键路径 D2H；
- 构造 context-level useful-action oracle。

此外，当前 submit snapshot 尚未记录 generation-aware `PageHandle`。只记录 Radix node ID
仍无法完全排除 ID reuse，因此 oracle analyzer 尚未实施。

## 7. 下一阶段

1. 在 native telemetry 中加入 generation-aware physical handle；
2. 实现 causal useful-action oracle analyzer；
3. coverage gate 要求 source snapshot、target context/epoch、physical handle、真实 reentry 和
   pressure/transfer outcome 全部可关联；
4. 只运行一次短冻结 trace；
5. oracle gap 不显著则停止预测动作主线，显著时再评价 FrontierBelief regret 与在线 canary。
