# P5G 确定性 Restore Micro-gate 通过记录（2026-08-01）

## 结论

修复后的确定性 GPU restore micro-gate 通过。一次原子 JointPlan retraction 事务完整经历：

```text
PAIR_ELIGIBLE
  -> TRANSACTION_CREATED
  -> durable obligation CREATED
  -> selective RETRACTION_COMMITTED
  -> D2H COMPLETED
  -> H2D COMPLETED
  -> victim GPU service resumed
  -> obligation SATISFIED
  -> gate COMPLETED
```

独立 verifier 的 14 项检查全部为 true。该实验只验证 P5G 单事务物理正确性，不用于吞吐、
JCT 或控制面性能结论，也不能代替 autonomous w4 system gate。

## 实验配置

- 原始目录：`experiments/raw/p5g_restore_micro/20260801T082631Z/`。
- GPU：RTX 6000 Ada，物理 GPU 1，TP=1。
- 模型：Qwen3-Coder-30B-A3B-Instruct-FP8。
- GPU KV pool：163,840 tokens（约 15 GB）。
- Host HiCache：96 GB。
- `MAX_RUNNING_REQUESTS=2`，稳定形成 `2 running / 1 waiting`。
- workload 总时长 68.40 秒；三个请求均以预期 token limit 正常返回。
- test-only restore hook 开启；预测模块与 recompute-drop 关闭。

## 物理结果

| 指标 | 结果 |
| --- | ---: |
| retraction transaction | `retraction-1` |
| restore obligation | `restore-1`，最终 `SATISFIED` |
| D2H | 4,719,378,432 bytes（约 4.40 GiB），1 次 |
| H2D | 4,719,378,432 bytes（约 4.40 GiB），1 次 |
| physical snapshot | 1 次，0.119 ms |
| commit read-set validation | 1 次，0.066 ms，0 stale |
| post-restore service samples | 340 |
| command dispatch / ACK | 2 / 2，0 missing、0 orphan |
| retry/partial/reject | 0 / 0 / 0 |

HBM mirror 未超过 allocator，Host residency 与 page index 一致。shutdown 已 ACK，queue、transfer、
transaction、lease、funding 和 obligation 均无残留；server log 无 traceback、OOM 或 API timeout。

## 残余问题

本轮发生 493 次 overlap barrier request/drain：

- 250 次 drain 后 pressure 已消失或物理 victim 尚不可用，没有生成计划；
- 242 次发生在 `restore-1` 尚未完成期间，随后被 restore-debt barrier 拒绝；
- 只有 1 次真正创建 retraction transaction。

这不影响本轮物理事务的正确性，但会显著污染性能测量。进入 autonomous w4 前，应在请求 drain
之前检查 micro pair 的粗粒度物理就绪信号，并在存在 active restore debt 时直接抑制 barrier，避免
无效 drain 周期。

## 可视化

- `experiments/raw/p5g_restore_micro/20260801T082631Z/kv_transfer_timeline.html`
- `experiments/raw/p5g_restore_micro/20260801T082631Z/kv_transfer_timeline.json`
- `experiments/raw/p5g_restore_micro/20260801T082631Z/restore_micro_gate_validation.json`
- `experiments/raw/p5g_restore_micro/20260801T082631Z/transfer_validation.json`
