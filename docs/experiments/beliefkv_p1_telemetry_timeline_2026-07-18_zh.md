# BeliefKV P1：HiCache 迁移遥测与 KV 时间线

日期：2026-07-18
状态：P1 工程退出条件已通过；正式统计置信与 unhidden stall 仍待扩展后端验证

## 1. 阶段边界

本阶段落实
[`beliefkv_hicache_joint_control_improvement_plan_2026-07-18_zh.md`](../beliefkv_hicache_joint_control_improvement_plan_2026-07-18_zh.md)
中的 P1，不启用 joint planner、预测迁移或 Reveal-and-Commit。当前正确性边界保持不变：

- SGLang allocator、Radix topology、node lock 和 KV tensor 是物理真相源；
- RCCG 是 invocation/context 因果真相源；
- policy 只产生 intent；
- residency 只在 `CommandAck` 后提交；
- `TransferTelemetry` 只用于性能建模，不能推动状态提交。

2026-07-17 的最终压力运行仍只作为 correctness anchor。该运行没有完整的 Radix mutation
trace，也没有本阶段新增的真实 DMA telemetry，因此不能被追溯解释为完整的 P0 physical
replay 或 P1 性能基线。

## 2. 已完成实现

### 2.1 HiCache capability contract

`beliefkv/runtime/sglang_adapter.py` 新增 `HiCacheCapabilities`。固定的 SGLang
`0.5.2rc1` adapter 如实报告：

- 不支持 operation merge；
- 不支持 layer completion event；
- 按实际 Host pool layout 报告 `page_first_host_layout`；当前启动脚本默认
  `layer_first`，因此该 run 中为 false；
- 支持主动触发 load；
- 最大并发 operation 为 1；
- 物理动作粒度为 Radix node extent。

后续 planner 必须基于 capability 分支，不能假定新版 HiCache 的 overlap 或 multi-inflight
能力存在。

### 2.2 ACK 与性能遥测分离

`beliefkv/runtime/protocol.py` 新增独立 `TransferTelemetry`。时间戳定义为：

- `submit_ts_ms`：命令提交给 scheduler backend；
- `start_ts_ms`：HiCache operation 实际进入执行队列；
- `first_layer_ready_ts_ms`：固定版本不可观测，记录为 `null`；
- `complete_ts_ms`：scheduler 观察到 HiCache operation 完成。

遥测同时记录 direction、source/target tier、actual bytes、closure bytes、operation count、
status 和 reject/partial reason。没有发生 DMA 的 COMMIT/DROP 不计入 D2H/H2D 字节。

若 D2H 已物理完成并形成 Host copy，但后续 GPU eviction 因 descendant closure 被拒绝，
ACK 仍按实际 residency 结果报告失败或零释放；telemetry 单独报告已发生的 DMA 字节。这样可
显式量化 wasted transfer，而不会污染正确性状态。

### 2.3 真实可用性语义

`beliefkv/runtime/sglang_v052rc1.py` 定期写入 `resource_snapshot`：

- HBM 使用量和容量来自 SGLang KV allocator；配置容量另存为
  `configured_hbm_capacity_bytes`，用于暴露配置与实际 token pool 的偏差；
- Host 使用量和容量来自 HiCache Host KV pool；
- 固定 adapter 当前无法可靠获得 PCIe utilization、copy-engine utilization 和 GPU compute
  utilization，这些字段写为 `null`，不得用 0 代替 unavailable。

主审计日志 schema 升级到 v2；迁移事件同时写入独立的
`transfer_telemetry.jsonl`，便于流式分析和长期归档。

### 2.4 在线传输服务曲线

`beliefkv/policy/service_curve.py` 按 direction、size bucket 和 compute phase 维护滚动窗口，
使用保守的 setup-time P90 与 effective-bandwidth P10 预测完成时间，并记录 partial/reject
概率。样本不足或字段 unavailable 时回退到带安全系数的静态 PCIe 模型。

controller 只有在对应 command ACK 已被处理后才接收 telemetry，审计日志也先记录 ACK，
再记录 telemetry，避免离线分析把性能观测误认为状态提交依据。telemetry 顶层 `ts_ms` 是
scheduler 观察时刻，物理完成时刻保存在 `complete_ts_ms`。

### 2.5 HBM/Host KV 时间线

新增 CLI：

```bash
conda run -n beliefkv beliefkv render-transfer-timeline \
  RUN_DIR/runtime_audit.jsonl \
  RUN_DIR/kv_transfer_timeline.html
```

输出包括：

1. HBM 与 Host KV occupancy 随时间的曲线；
2. D2H、H2D、mixed/reclaim operation 的时间区间；
3. command、context、physical bytes、closure bytes、状态与原因的 hover 明细；
4. 可机器读取的同名 `.json` 文件。

HTML 是自包含文件，不依赖外部前端库。新 trace 使用实际 submit/start/complete 和
resource snapshot。旧 trace 只能用 dispatch-to-ACK 近似 operation 区间，报告会明确标记
`legacy_dispatch_ack_aggregate`；旧 trace 没有 Host occupancy 时保持 unavailable。

当前已为 correctness anchor 生成兼容性报告：

```text
experiments/processed/p0_20260717_correctness/kv_transfer_timeline.html
experiments/processed/p0_20260717_correctness/kv_transfer_timeline.json
```

它适合检查旧运行的整体 HBM 压力和命令时序，不适合推断精确 PCIe DMA 带宽或 Host 占用。
该图使用旧 `sglang_metrics.jsonl` 的轮询 HBM 样本，峰值为 149,559/163,840 tokens
（91.284%）；原 correctness 报告中的 163,755/163,840（99.948%）来自 scheduler
安全点瞬时值。二者口径不同，轮询曲线会漏掉短时峰值，HTML 已直接标注这一限制。

## 3. 当前验证结果

- 完整测试：161 passed，4 skipped；
- pinned SGLang contract：`0.5.2rc1`，commit
  `18f91eb639084825717c0e3c3c7273492812ab71`；
- timeline renderer 已覆盖新 telemetry 和 legacy fallback；
- 测试覆盖 D2H/H2D、无 DMA reject、DMA 完成后 closure reject、ACK-before-telemetry、真实
  allocator snapshot 和 unavailable 指标。

真实 GPU 验证详见
[P1 HiCache 真实迁移、服务曲线与控制开销验证](beliefkv_p1_real_hicache_validation_2026-07-18_zh.md)。

## 4. P1 退出条件结论

1. 6,415/6,415 个长跑命令收到 ACK，5,693 条 DMA telemetry 全部满足
   `dispatch < ACK < telemetry` 和 `submit <= start <= complete`；
2. 27,101 个资源快照中 Host allocator 与 `PageOwnershipIndex.cpu_bytes` 零偏差；HBM
   mirror 从不超过 allocator，运行中差额是尚未进入 Radix mirror 的 active request KV；
3. 修正后的 callback service curve 只用 completed operation，并增加 direction-level
   callback P90 floor。长跑时间顺序 80/20 holdout 低估率为 1/34（2.94%），短跑为
   0/11；
4. 516 个含 telemetry 的真实 scheduler hook tick 上，telemetry P99 开销为 0.365 ms，
   同 tick P99 比例为 2.01%，低于 5%；
5. 时间线将 917 个真实非零字节 DMA 与 4,776 个零字节拒绝明确分轨；HBM/Host 指标来自
   allocator，而不是从命令推算；
6. `compute_wait`、PCIe/copy-engine/GPU compute utilization 在固定后端仍为 unavailable，
   验证器不会将其当作 0。

按预先定义的点估计门槛，P1 可以进入 P1.5 retry guard，然后再进入 P2。严格统计限制是：长跑低估率的 Wilson 95% 上界
仍为 14.92%，短跑为 25.88%，因此论文级“低于 10%”结论仍需更多独立 completed
operation 扩样。固定 SGLang 版本无法观测 `compute_wait_ms`，所以当前只验证 callback
service curve，不能声称已经优化 actual unhidden stall。
