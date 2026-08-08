# BeliefKV P5G SafePointPhysicalSnapshot CPU Microbenchmark

## 目标

验证惰性 `SafePointPhysicalSnapshot(epoch)` 在可见 request 数增加时的构建成本，并确认同一 epoch
只构建一次。该结果不使用 GPU，不评价 workflow JCT，也不能替代固定 w4 clean-completion gate。

执行命令：

```bash
conda run --no-capture-output -n beliefkv \
  python scripts/benchmark_physical_snapshot.py \
  --cardinalities 8,16,32,64,128 \
  --iterations 500 \
  --scheduler-steps-per-build 1729
```

## 结果

| Request 数 | 均值 | P99 | 最大值 | 平均每 record |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.066 ms | 0.072 ms | 0.080 ms | 8.24 us |
| 16 | 0.109 ms | 0.115 ms | 0.134 ms | 6.83 us |
| 32 | 0.187 ms | 0.200 ms | 0.212 ms | 5.85 us |
| 64 | 0.348 ms | 0.358 ms | 0.372 ms | 5.43 us |
| 128 | 0.671 ms | 0.689 ms | 0.705 ms | 5.24 us |

均值对 request 数的线性斜率为约 `5.03 us/request`。按此前真实运行约每 1,729 个 scheduler step
触发一次 ownership rebuild 估算，N=128 的摊销为约 `0.00039 ms/step`。该值只用于量级判断，实际
触发率仍须由下一次固定 w4 gate 观测。

N=128 的平均分段成本为：queue collection 0.042 ms、metadata indexing 0.006 ms、ownership
lookup/record construction 0.524 ms、native/explicit operation indexing 0.030 ms、
sorting/allocation 0.066 ms。
500 次 warm build 关闭 GC 后没有 GC collection，说明主要成本来自线性 ownership record 组装。

## 判定

CPU 结果支持惰性快照在 128 request 内开销可控，但不能证明真实 SGLang 并发 callback、request ID
重用和 native operation transition 的规模正确性。下一步只运行一次完整固定 w4 P5G gate；通过后
进入 P6 离线开发。开放预测性物理动作前再做一次短时 w8 correctness smoke，w24/w32 延后。
