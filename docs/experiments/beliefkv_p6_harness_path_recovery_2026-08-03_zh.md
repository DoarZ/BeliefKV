# P6 Django harness 路径恢复记录

## 原始污染与排除

`p6-009-train-mixed-r0/20260803T100537Z` 中的
`django__django-11138` 和 `django__django-14011` 受到错误路径契约污染。公共
prompt 错误地包含 SymPy 路径和测试命令，Django 任务没有明确说明 checkout
根目录就是 `/workspace`。原始 run 现包含 `TRAINING_EXCLUSIONS.json`；重新导出
后两者的 split 为 `null`，全部 `training_eligible*` 标签为 `false`，其余六个
workflow 不受影响。

## Harness 修复

- 公共契约只描述稳定的 `/workspace` namespace，不再包含 SymPy 示例；
- 按 workload 生成 repository contract，Django 明确映射
  `django/db/... -> /workspace/django/db/...`，禁止重复为
  `/workspace/django/django/...`；
- contract 同时进入 supervisor、动态 subagent、planned child 和 implementer；
- sandbox preflight 验证 `pwd -P` 和 Git top-level 都是 `/workspace`；
- collector 支持从已冻结 batch 通过重复 `--instance-id` 定向重采集；
- P6 exporter 支持逐 workflow fail-closed 排除，不删除或修改 raw trace。

## R1 诊断运行

目录：

`experiments/raw/p6_agent_semantics_v1/p6-harness-recovery-django-r1/20260803T140338Z`

`django__django-11138` 自然完成；`django__django-14011` 暴露两个新的 runtime
harness 问题：并发相同工具调用可在首个调用登记结果前穿透 duplicate circuit，
且 context compaction 后 guard 从可截断消息历史反推 recovery attempt，导致
attempt 反复为 1。该 run 已标记 `COLLECTION_INVALID.json`，不得训练。

后续修复在工具执行前原子登记 in-flight reservation，并将
`guard_recovery_attempt` 改为独立单调私有状态。相关 harness/P6 测试 66 个全部
通过，SGLang 专属回归 137 个全部通过。

## R2 启动状态

目录：

`experiments/raw/p6_agent_semantics_v1/p6-harness-recovery-django-r2/20260803T144548Z`

尚未提交 workload。启动时两张 RTX 6000 Ada 均被外部 vLLM 进程占用，模型加载
前 CUDA OOM，因此已写入 `STARTUP_FAILED.json`。恢复条件是任一 GPU 能稳定提供
完整 49 GiB 显存；不能通过降低 163,840-token KV pool 来绕过正式配置。
